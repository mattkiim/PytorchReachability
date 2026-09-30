"""Export held-out WM reconstructions and open-loop predictions without W&B."""

import argparse
import base64
import html
import io
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
import ruamel.yaml as yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dreamerv3-torch"))
import models


def panel(images, labels, title):
    size = 192
    canvas = Image.new("RGB", (len(images) * size, size + 60), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    for i, (array, label) in enumerate(zip(images, labels)):
        pixels = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        if pixels.shape[-1] == 1:
            pixels = np.repeat(pixels, 3, axis=-1)
        canvas.paste(Image.fromarray(pixels).resize((size, size)), (i * size, 52))
        draw.text((i * size + 8, 30), label, fill="black")
    return canvas


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/configs_po.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, help="Actual .pkl filename, including heat-rate suffix")
    parser.add_argument("--modality", choices=("rgb", "mm"), required=True)
    parser.add_argument("--num_train_trajs", type=int, default=3800)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0", help="CUDA device (the current WM implementation requires CUDA)")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if min(args.history, args.horizon, args.examples) < 1 or args.num_train_trajs < 0:
        parser.error("history, horizon and examples must be positive; num_train_trajs must be nonnegative")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("The current WM implementation requires an available CUDA device")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    config = SimpleNamespace(**yaml.YAML(typ="safe").load(Path(args.config_path).read_text())["defaults"])
    config.device = args.device
    config.no_heat = args.modality == "rgb"
    config.num_actions = 2
    shape = tuple(config.size)
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, shape=(*shape, 3), dtype=np.uint8),
        "heat": gym.spaces.Box(0, 255, shape=(*shape, 1), dtype=np.uint8),
        "obs_state": gym.spaces.Box(-1, 1, shape=(4 if config.obs_priv_heat else 3,), dtype=np.float32),
    })
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {}
    for key, value in checkpoint["agent_state_dict"].items():
        if key.startswith("_wm."):
            state[key.removeprefix("_wm.").removeprefix("_orig_mod.")] = value
    wm = models.WorldModel(obs_space, gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32), 0, config)
    wm.load_state_dict(state, strict=True)
    wm.to(args.device).eval()
    step = checkpoint.get("step", "unknown")
    del checkpoint, state

    print(f"Loading {args.dataset}; checkpoint step {step}", flush=True)
    with open(args.dataset, "rb") as stream:
        trajectories = pickle.load(stream)
    length = args.history + args.horizon
    selected = [(i, demo) for i, demo in enumerate(trajectories)
                if i >= args.num_train_trajs and len(demo["actions"]) >= length][:args.examples]
    del trajectories
    if not selected:
        raise ValueError("No held-out trajectories long enough for the requested history and horizon")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, views = [], []
    for index, demo in selected:
        obs = demo["obs"]
        states = np.asarray(obs["state"][:length])
        obs_state = np.stack([np.cos(states[:, 0]), np.sin(states[:, 0]), states[:, 1]], -1)
        if config.obs_priv_heat:
            obs_state = np.concatenate([obs_state, np.asarray(obs["priv_heat"][:length])[:, None]], -1)
        raw = dict(image=np.asarray(obs["image"][:length])[None],
                   heat=np.asarray(obs["heat"][:length])[None], obs_state=obs_state[None],
                   action=np.asarray(demo["actions"][:length])[None],
                   is_first=np.array([[True] + [False] * (length - 1)]),
                   is_terminal=np.asarray(demo["dones"][:length])[None])
        data = wm.preprocess(raw)
        embed = wm.encoder({key: value[:, :args.history] for key, value in data.items()})
        post, _ = wm.dynamics.observe(embed, data["action"][:, :args.history], data["is_first"][:, :args.history])
        prior = wm.dynamics.imagine_with_action(data["action"][:, args.history:],
                                               {key: value[:, -1] for key, value in post.items()})
        # Match the action alignment used by WorldModel.video_pred.
        features = torch.cat([wm.dynamics.get_feat(post), wm.dynamics.get_feat(prior)], 1)
        decoded = wm.heads["decoder"](features)
        predictions = {key: dist.mode()[0].cpu().numpy() for key, dist in decoded.items() if key in ("image", "heat")}
        truth = raw["image"][0].astype(np.float32) / 255
        prediction = predictions["image"]
        errors = np.abs(prediction - truth)
        frames, encoded = [], []
        for t in range(length):
            phase = "reconstruction" if t < args.history else "open-loop prediction"
            arrays = [truth[t], prediction[t], errors[t]]
            labels = ["Actual RGB", "Model RGB", "Absolute RGB error"]
            if args.modality == "mm":
                arrays += [raw["heat"][0, t].astype(np.float32) / 255, predictions["heat"][t]]
                labels += ["Actual IR", "Model IR"]
            frame = panel(arrays, labels, f"Trajectory {index} | frame {t} | {phase}")
            frames.append(frame)
            stream = io.BytesIO()
            frame.save(stream, format="PNG")
            encoded.append(base64.b64encode(stream.getvalue()).decode("ascii"))
        frames[0].save(output / f"trajectory_{index}.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
        times = sorted(set([0, args.history - 1, args.history, length // 2, length - 1]))
        sheet = Image.new("RGB", (frames[0].width, frames[0].height * len(times)), "white")
        for row, t in enumerate(times):
            sheet.paste(frames[t], (0, row * frames[t].height))
        sheet.save(output / f"trajectory_{index}.png")
        records.append(dict(trajectory=index, rgb_prediction_mse=float(np.mean((prediction[args.history:] - truth[args.history:]) ** 2))))
        views.append(dict(trajectory=index, frames=encoded))
        print(f"Exported trajectory {index}", flush=True)
    metadata = dict(checkpoint=args.checkpoint, checkpoint_step=step, dataset=args.dataset,
                    modality=args.modality, history=args.history, horizon=args.horizon,
                    seed=args.seed, num_train_trajs=args.num_train_trajs, examples=records)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    warning = "This checkpoint has barely trained; use these views to diagnose training, not assess final performance." if isinstance(step, int) and step < 100 else "Qualitative examples, not aggregate safety validation."
    document = """<!doctype html><html><meta charset="utf-8"><title>World-model evaluation</title>
<style>body{font:16px system-ui;max-width:1050px;margin:35px auto;padding:0 20px;background:#f4f6f8;color:#17202a}section{background:white;padding:20px;margin:22px 0;border-radius:12px}img{display:block;width:100%;max-width:960px}input{width:65%;margin:15px}button{padding:8px 18px}small{display:block;overflow-wrap:anywhere}</style>
<h1>World-model evaluation: MODALITY</h1><p>Checkpoint step: STEP. WARNING</p>
<p>First HISTORY frames: reconstruction with observations. Next HORIZON frames: open-loop prediction using recorded actions, with no future image inputs. Error images use absolute pixel differences at the original scale.</p>
<small>CHECKPOINT</small><div id="runs"></div><script>
const runs=RUN_DATA;
for(const run of runs){const s=document.createElement('section');s.innerHTML=`<h2>Held-out trajectory ${run.trajectory}</h2><button>Play</button><input type="range" min="0" max="${run.frames.length-1}" value="0"><span></span><img>`;document.querySelector('#runs').append(s);const slider=s.querySelector('input'),img=s.querySelector('img'),label=s.querySelector('span'),button=s.querySelector('button');const draw=()=>{img.src='data:image/png;base64,'+run.frames[Number(slider.value)];label.textContent='Frame '+slider.value;};slider.oninput=draw;let timer=null;button.onclick=()=>{if(timer){clearInterval(timer);timer=null;button.textContent='Play';}else{button.textContent='Pause';timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%run.frames.length;draw();},200);}};draw();}
</script></html>"""
    for key, value in {"MODALITY": args.modality.upper(), "STEP": str(step), "WARNING": warning,
                       "HISTORY": str(args.history), "HORIZON": str(args.horizon),
                       "CHECKPOINT": html.escape(args.checkpoint), "RUN_DATA": json.dumps(views)}.items():
        document = document.replace(key, value)
    (output / "index.html").write_text(document)
    print(f"Open {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
