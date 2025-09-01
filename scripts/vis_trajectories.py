import os, sys, pathlib
import argparse
from io import BytesIO
import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
import pickle
import h5py
import ruamel.yaml as yaml

# Project paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import models, tools


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def to_uint8(x):
    if x.dtype != np.uint8:
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return x


def expand3(x):
    """Expand grayscale → RGB for visualization"""
    if x is None:
        return None
    if x.ndim == 4 and x.shape[-1] == 1:
        return np.repeat(x, 3, axis=-1)
    elif x.ndim == 3:
        return np.stack([x]*3, axis=-1)
    return x


def save_composite_video(cam0, cam2, hotinner, failure, eff_state, actions, filename, fps=20):
    """
    cam0, cam2, hotinner: (T,H,W,3) np.uint8
    failure: (T,)
    eff_state: (T,D)   # D-dim end effector state
    actions: (T,A)     # A-dim actions
    """
    cam0, cam2, hotinner = map(to_np, [cam0, cam2, hotinner])
    failure, eff_state, actions = map(to_np, [failure, eff_state, actions])

    cam0, cam2, hotinner = map(to_uint8, [cam0, cam2, hotinner])
    cam0, cam2, hotinner = map(expand3, [cam0, cam2, hotinner])

    T = len(failure)
    frames_out = []

    for t in range(T):
        # Layout: 4 rows (top row has 3 subplots side by side)
        fig = plt.figure(figsize=(12, 12))  # wider for side-by-side images
        gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 1, 1])  # 4 rows, 3 cols

        # --- Cameras (row 0, 3 columns) ---
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])

        ax0.imshow(cam0[t]); ax0.axis("off"); ax0.set_title("Camera 0")
        ax1.imshow(cam2[t]); ax1.axis("off"); ax1.set_title("Camera 2")
        ax2.imshow(hotinner[t]); ax2.axis("off"); ax2.set_title("Hot-Inner")

        # --- Failure plot (row 1, span all 3 cols) ---
        axf = fig.add_subplot(gs[1, :])
        axf.plot(failure[:t+1], color="red", drawstyle="steps-post")
        axf.set_xlim(0, T); axf.set_ylim(-0.1, 1.1)
        axf.set_title("Failure Label")
        axf.set_xlabel("t"); axf.set_ylabel("failure")

        # --- End effector state (row 2, span all 3 cols) ---
        axe = fig.add_subplot(gs[2, :])
        for d in range(eff_state.shape[1]):
            axe.plot(eff_state[:t+1, d], label=f"state{d}")
        axe.set_xlim(0, T)
        axe.set_title("End Effector State")
        axe.legend(fontsize=6, loc="upper right")

        # --- Actions (row 3, span all 3 cols) ---
        axa = fig.add_subplot(gs[3, :])
        for a in range(actions.shape[1]):
            axa.plot(actions[:t+1, a], linestyle="--", label=f"act{a}")
        axa.set_xlim(0, T)
        axa.set_title("Actions")
        axa.legend(fontsize=6, loc="upper right")

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = imageio.v3.imread(buf)
        frames_out.append(img)
        plt.close(fig)

    imageio.mimsave(filename, frames_out, fps=fps)
    print(f"Saved composite video: {filename}")



def main(cfg, ckpt_path=None):
    # Load dataset (example: from pickle/h5)
    path = "/data/mattkiim/heat_actuated2_consolidated.h5"
    i = 0
    with h5py.File(path, "r") as f:  # use "r+" only if you need to write
        for run in f:  # each top-level group name
            
            video_dir = pathlib.Path("traj_videos")
            video_dir.mkdir(parents=True, exist_ok=True)
            
            i += 1
            # if i > 10:
            #     break

            cam0 = f[run]["camera_0"]   # (T,H,W,3)
            cam2 = f[run]["camera_2"]   # (T,H,W,3)
            heat_inner = f[run]["hot_inner"][:]
            heat_inner = np.squeeze(heat_inner, -1)      # (T,H,W)

            T = heat_inner.shape[0]
            failure = []
            for t in range(T):
                frame = heat_inner[t]
                pixel_count = np.sum(frame > 0.0)
                frac_too_hot = np.sum(frame > 0.6) / pixel_count if pixel_count > 0 else 0.0
                failure.append(frac_too_hot > 0.5)
            failure = np.array(failure, dtype=np.int32)  # shape (T,)
            
            # print(failure.shape)
            eff_state = f[run]["ee_states"] # (T,D)
            actions = f[run]["actions"]         # (T,A)

            out_path = video_dir / f"traj_{i}.mp4"
            save_composite_video(cam0, cam2, heat_inner, failure, eff_state, actions, out_path, fps=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs.yaml", type=str)
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load((pathlib.Path(sys.argv[0]).parent / f"../{args.config_path}").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    cfg = parser.parse_args(remaining)

    main(cfg)
