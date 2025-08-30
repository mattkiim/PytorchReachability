# eval_only_lz.py
import argparse
import os
import pathlib
import sys
import pickle
import collections
import warnings
from io import BytesIO
import imageio

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import gym 
import ruamel.yaml as yaml

warnings.simplefilter("ignore", category=FutureWarning)
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["WANDB_MODE"] = "disabled"

# Project paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, logger.step // config.action_repeat, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)

        if (config.compile and os.name != "nt"):
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    @torch.no_grad()
    def margin_over_time(self, batch, gamma=None, warmup=5):
        wm = self._wm
        cfg = self._config
        data = wm.preprocess(batch)

        embed = wm.encoder(data)
        actions = data["action"]
        firsts  = data["is_first"]

        B, T = actions.shape[:2]

        # -------- Warmup latent state --------
        latent = None
        for t in range(warmup):
            latent, _ = wm.dynamics.obs_step(latent, actions[:, t], embed[:, t], firsts[:, t])

        # -------- Rollout after warmup --------
        posts = []
        for t in range(warmup, T):
            latent, _ = wm.dynamics.obs_step(latent, actions[:, t], embed[:, t], firsts[:, t])
            posts.append(latent)

        # Stack into [B, T-warmup, ...]
        if posts:
            post = {k: torch.stack([p[k] for p in posts], dim=1) for k in posts[0]}
            feat = wm.dynamics.get_feat(post)
        else:
            feat = torch.empty((B, 0), device=embed.device)

        # margin predictions
        margin_vals = wm.heads["margin"](feat).squeeze(-1)
        gamma = cfg.gamma_lx if gamma is None else gamma

        # l(z) per timestep
        failure = data["failure"].float()[:, warmup:]
        lz = torch.zeros_like(margin_vals)
        safe_mask   = (failure == 0)
        unsafe_mask = ~safe_mask
        lz[safe_mask]   = gamma - margin_vals[safe_mask]
        lz[unsafe_mask] = gamma + margin_vals[unsafe_mask]

        return {
            "margin_vals": margin_vals.cpu().numpy(),   # shape [B, T-warmup]
            "lz": lz.cpu().numpy(),                     # shape [B, T-warmup]
            "failure": failure.cpu().numpy(),           # aligned per-frame labels
            "warmup": warmup,
        }




def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)

    def filtered_generator():
        for traj in generator:
            failure = traj["failure"]
            if failure[0] == 0 and failure[-1] == 1:
                yield traj

    dataset = tools.from_generator(filtered_generator(), 16)
    return dataset

def has_spike(traj, threshold=0.8):
    diffs = np.abs(np.diff(traj))
    # return np.all(diffs <= threshold)
    return np.any(diffs > threshold)

def save_rgb_heat_video(rgb_frames, heat_frames, filename, fps=10):
    """
    rgb_frames: (T, H, W, 3)
    heat_frames: (T, H, W, 1) or (T, H, W)
    """
    if torch.is_tensor(rgb_frames):
        rgb_frames = rgb_frames.detach().cpu().numpy()
    if torch.is_tensor(heat_frames):
        heat_frames = heat_frames.detach().cpu().numpy()

    # Ensure uint8 range
    if rgb_frames.dtype != np.uint8:
        rgb_frames = (np.clip(rgb_frames, 0, 1) * 255).astype(np.uint8)
    if heat_frames.dtype != np.uint8:
        heat_frames = (np.clip(heat_frames, 0, 1) * 255).astype(np.uint8)

    # If heat is grayscale, expand to 3 channels
    if heat_frames.ndim == 4 and heat_frames.shape[-1] == 1:
        heat_frames = np.repeat(heat_frames, 3, axis=-1)
    elif heat_frames.ndim == 3:  # (T, H, W)
        heat_frames = np.stack([heat_frames]*3, axis=-1)

    # Concatenate side by side
    frames = np.concatenate([rgb_frames, heat_frames], axis=2)  # along width
    imageio.mimsave(filename, frames, fps=fps)

def save_rgb_heat_hotinner_lz_failure_video(rgb_frames, heat_frames, hotinner_frames, lz, failure, filename, fps=10):
    """
    Composite video with:
      - RGB
      - Heat
      - Hot-Inner
      - l(z) curve
      - True safety label (failure) over time
    """
    import matplotlib.pyplot as plt
    from io import BytesIO
    import imageio
    import numpy as np
    import torch

    # Convert tensors to numpy
    to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else x
    rgb_frames     = to_np(rgb_frames)
    heat_frames    = to_np(heat_frames)
    hotinner_frames= to_np(hotinner_frames) if hotinner_frames is not None else None
    lz             = to_np(lz)
    failure        = to_np(failure)

    # Normalize images
    def to_uint8(x):
        if x.dtype != np.uint8:
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)
        return x

    rgb_frames = to_uint8(rgb_frames)
    heat_frames = to_uint8(heat_frames)
    if hotinner_frames is not None:
        hotinner_frames = to_uint8(hotinner_frames)

    # Expand grayscale to 3 channels
    def expand3(x):
        if x is None:
            return None
        if x.ndim == 4 and x.shape[-1] == 1:
            return np.repeat(x, 3, axis=-1)
        elif x.ndim == 3:
            return np.stack([x]*3, axis=-1)
        return x

    heat_frames = expand3(heat_frames)
    hotinner_frames = expand3(hotinner_frames)

    frames_out = []
    T = len(lz)

    for t in range(T):
        n_rows = 5 if hotinner_frames is not None else 4
        fig, axes = plt.subplots(n_rows, 1, figsize=(5, 12))

        # RGB
        axes[0].imshow(rgb_frames[t])
        axes[0].axis("off"); axes[0].set_title("RGB")

        # Heat
        axes[1].imshow(heat_frames[t])
        axes[1].axis("off"); axes[1].set_title("Heat")

        row_idx = 2
        if hotinner_frames is not None:
            axes[2].imshow(hotinner_frames[t])
            axes[2].axis("off"); axes[2].set_title("Hot-Inner")
            row_idx = 3

        # l(z) curve
        axes[row_idx].plot(lz[:t+1], color="blue", label="tanh(lz)")
        axes[row_idx].set_xlim(0, T)
        axes[row_idx].set_ylim(-1.05, 1.05)
        axes[row_idx].set_title("tanh(l(z)) over time")
        axes[row_idx].set_xlabel("t"); axes[row_idx].set_ylabel("l(z)")

        # True failure/safety label
        axes[row_idx+1].plot(failure[:t+1], color="red", drawstyle="steps-post", label="failure")
        axes[row_idx+1].set_xlim(0, T)
        axes[row_idx+1].set_ylim(-0.1, 1.1)
        axes[row_idx+1].set_title("True Safety Label")
        axes[row_idx+1].set_xlabel("t"); axes[row_idx+1].set_ylabel("failure (0=safe,1=fail)")

        plt.tight_layout()

        # Save frame to numpy
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = imageio.v3.imread(buf)
        frames_out.append(img)
        plt.close(fig)

    # Save video
    imageio.mimsave(filename, frames_out, fps=fps)


def main(config, ckpt_path=None):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    # ----- Spaces -----
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    bounds = np.array([
        [config.x_min, config.x_max],
        [config.y_min, config.y_max],
        [0, 2 * np.pi],
        [0, 1],
    ])
    low, high = bounds[:, 0], bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval / 2),
        np.float32(midpoint + interval / 2),
    )

    image_size = config.size[0]
    if config.multimodal:
        if config.aug_rssm:
            image_observation_space = gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        else:
            image_observation_space = gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 4), dtype=np.uint8
            )
    else:
        image_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )

    if config.obs_priv_heat:
        obs_observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    else:
        obs_observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    if config.aug_rssm:
        heat_observation_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 1), dtype=np.uint8
        )
        observation_space = gym.spaces.Dict({
            'state': gt_observation_space,
            'obs_state': obs_observation_space,
            'image': image_observation_space,
            'heat': heat_observation_space,
        })
    else:
        observation_space = gym.spaces.Dict({
            'state': gt_observation_space,
            'obs_state': obs_observation_space,
            'image': image_observation_space,
        })

    config.num_actions = action_space.shape[0]

    # dataset
    expert_val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)
    eval_dataset = make_dataset(expert_val_eps, config)
    print("Length of validation data:", len(expert_val_eps))

    # agent
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        dataset=None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    # checkpoint
    ckpt_path = ckpt_path or config.rssm_ckpt_path
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)

    # eval
    batch = next(eval_dataset)
    res = agent.margin_over_time(batch)

    # plotting
    lz = res["margin_vals"]
    lz = np.tanh(lz)
    B, T = lz.shape
    timesteps = np.arange(T)
    
    video_dir = pathlib.Path("spiky_videos", config.name)
    video_dir.mkdir(exist_ok=True)

    for i in range(B):
        if has_spike(lz[i], threshold=0.8):
            rgb_frames = batch["image"][i]
            heat_frames = batch["heat"][i]
            hotinner_frames = batch["heat_inner"][i]
            lz_traj = lz[i]
            failure_traj = batch["failure"][i]

            out_path = video_dir / f"traj_{i}_spike.mp4"
            save_rgb_heat_hotinner_lz_failure_video(
                rgb_frames, heat_frames, hotinner_frames, lz_traj, failure_traj, out_path, fps=20
            )
            print(f"Saved RGB+Heat+l(z) video for trajectory {i} → {out_path}")

    plt.figure(figsize=(8,5))
    for i in range(min(50, B)):
        plt.plot(timesteps, lz[i], color="blue", alpha=0.5)

    plt.title("l(z) over time")
    plt.xlabel("Timestep")
    plt.ylabel("l(z)")
    plt.savefig(f"{config.name}_lz_over_time_samples.png")
    plt.close()

    print("Saved l(z) plot for trajectories at:", "lz_over_time_samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs.yaml", type=str)
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--ckpt_path", type=str, default=None)
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

    cfg.rssm_train_steps = 0
    main(cfg, ckpt_path=args.ckpt_path)
