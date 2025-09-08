import os, sys, pathlib
import argparse
from io import BytesIO
import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
import pickle
import gym
import h5py
import ruamel.yaml as yaml
from PyHJ.data import Batch
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.exploration import GaussianNoise
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

# Project paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import models, tools


IMG_SIZE = 224  # match training
SAFE_SCALE = 0.15  # normalized-space scale; adjust as desired

def normalize_acs(acs, device='cuda:1'):
    max_ac = torch.tensor([0.10543401, 0.00693101, 1., 0.00348351 ,0.04336035, 0.00416869, 1.]).to(device)
    min_ac = torch.tensor([-0.07480399, -0.00952092, -0.80855778, -0.00182587, -0.03781896, -0.00121621,
 -1.]).to(device)
    
    acs = (acs - min_ac) / (max_ac - min_ac)
    acs = (acs*2) - 1
    return acs

def unnormalize_acs(acs, device='cuda:0'):
    max_ac = torch.tensor([0.10543401, 0.00693101, 1., 0.00348351, 0.04336035, 0.00416869, 1.]).to(device)
    min_ac = torch.tensor([-0.07480399, -0.00952092, -0.80855778, -0.00182587, -0.03781896, -0.00121621, -1.]).to(device)
    acs = (acs + 1) / 2.0
    acs = acs * (max_ac - min_ac) + min_ac
    return acs


def eef_pose_to_state(T, gripper):

    # Extract translation
    x, y, z = T[:3, 3]

    # Extract rotation matrix and convert to quaternion
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # Returns [qx, qy, qz, qw]

    # Concatenate into final state
    eef_state = np.concatenate(([x, y, z], quat, [gripper]))
    return eef_state


def build_world_model(cfg):
    image_size = 224
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    cam_obs_space = gym.spaces.Box(low=0, high=1, shape=(image_size, image_size, 3), dtype=np.float32)
    bool_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
    obs_observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
    heat_observation_space = gym.spaces.Box(low=0, high=1, shape=(image_size, image_size, 1), dtype=np.float32)

    observation_space = gym.spaces.Dict({
        'obs_state': obs_observation_space,
        'image': cam_obs_space,
        'heat': heat_observation_space,
        'is_first': bool_space,
        'is_last': bool_space,
        'is_terminal': bool_space,
    })

    device = cfg.device
    wm = models.WorldModel(observation_space, action_space, 0, cfg).to(device)

    ckpt_path = cfg.eval_rssm_ckpt_path

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = {k[14:]:v for k,v in ckpt['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    return wm

def build_policy_for_value(cfg):
    device = cfg.device
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    state_shape = 544
    action_shape = action_space.shape
    max_action = float(action_space.high[0])

    # Activations
    if cfg.actor_activation == 'ReLU':
        actor_activation = torch.nn.ReLU
    elif cfg.actor_activation == 'Tanh':
        actor_activation = torch.nn.Tanh
    elif cfg.actor_activation == 'Sigmoid':
        actor_activation = torch.nn.Sigmoid
    elif cfg.actor_activation == 'SiLU':
        actor_activation = torch.nn.SiLU

    if cfg.critic_activation == 'ReLU':
        critic_activation = torch.nn.ReLU
    elif cfg.critic_activation == 'Tanh':
        critic_activation = torch.nn.Tanh
    elif cfg.critic_activation == 'Sigmoid':
        critic_activation = torch.nn.Sigmoid
    elif cfg.critic_activation == 'SiLU':
        critic_activation = torch.nn.SiLU


    # Nets
    critic_layers = cfg.critic_net
    actor_layers  = cfg.control_net

    critic_net = Net(
        state_shape,
        action_shape,
        hidden_sizes=critic_layers,
        activation=critic_activation,
        concat=True,
        device=device
    )
    critic = Critic(critic_net, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=critic_optim, gamma=0.995)

    actor_net = Net(state_shape, hidden_sizes=actor_layers, activation=actor_activation, device=device)
    actor = Actor(actor_net, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr)
    actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=actor_optim, gamma=0.995)
    
    policy = DDPGPolicy(
        critic,
        critic_optim,
        critic_scheduler=critic_scheduler,
        tau=cfg.tau,
        gamma=cfg.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=cfg.exploration_noise),
        reward_normalization=cfg.rew_norm,
        estimation_step=cfg.n_step,
        action_space=action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_scheduler=actor_scheduler,
        actor_gradient_steps=cfg.actor_gradient_steps,
    )

    policy_ckpt = cfg.eval_policy_ckpt_path
    sd = torch.load(policy_ckpt, map_location=device)
    policy.load_state_dict(sd)

    return policy

def evaluate_V(policy, state):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs=tmp_obs, info=Batch())
    tmp = policy.critic_old(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()

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


def save_composite_video(
    cam0, cam2, hotinner, failure, eff_state, actions, filename, fps=20,
    lz=None, Vz=None, safe_actions=None,
):
    """
    cam0, cam2, hotinner: (T,H,W,3) or (T,H,W,1) np.uint8/float in [0,1]
    failure: (T,)
    eff_state: (T,D)
    actions: (T,A)
    lz: (T,) or None       # NEW
    Vz: (T,) or None       # NEW (a.k.a. BRT / value)
    """
    cam0, cam2, hotinner = map(to_np, [cam0, cam2, hotinner])
    failure, eff_state, actions = map(to_np, [failure, eff_state, actions])
    lz = None if lz is None else to_np(lz)
    Vz = None if Vz is None else to_np(Vz)

    cam0, cam2, hotinner = map(to_uint8, [cam0, cam2, hotinner])
    cam0, cam2, hotinner = map(expand3, [cam0, cam2, hotinner])

    T = len(failure)
    if lz is not None:   T = min(T, len(lz))
    if Vz is not None:   T = min(T, len(Vz))
    T = min(T, len(actions), len(eff_state), len(cam0), len(cam2), len(hotinner))

    frames_out = []

    # --- Dynamic grid: add rows if lz/Vz provided ---
    have_lz = lz is not None
    have_V  = Vz is not None
    have_safe = safe_actions is not None
    # base rows: cameras (row0), failure (row1), actions (last row)
    extra_rows = (1 if have_lz else 0)
    nrows = 3 + extra_rows  # 3 base rows + extras
    # Make actions the final row
    # Height ratios: cameras big, others small
    height = [3] + [1] * (nrows - 1)

    for t in range(T):
        fig = plt.figure(figsize=(12, 2.75 * nrows))
        gs = fig.add_gridspec(nrows, 3, height_ratios=height)

        # --- Cameras (row 0, 3 columns) ---
        ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[0, 2])
        ax0.imshow(cam0[t]); ax0.axis("off"); ax0.set_title("Camera 0")
        ax1.imshow(cam2[t]); ax1.axis("off"); ax1.set_title("Camera 2")
        ax2.imshow(hotinner[t]); ax2.axis("off"); ax2.set_title("Hot-Inner")

        # --- Failure (row 1, span) ---
        r = 1
        axf = fig.add_subplot(gs[r, :])
        axf.plot(failure[:t+1], color="red", drawstyle="steps-post")
        axf.set_xlim(0, T); axf.set_ylim(-0.1, 1.1)
        axf.set_title("Failure Label")
        axf.set_xlabel("t"); axf.set_ylabel("failure")
        r += 1

        # --- l(z) and V(z) (same plot) ---
        if have_lz or have_V:
            axlv = fig.add_subplot(gs[r, :])
            if have_lz:
                axlv.plot(np.arange(t+1), lz[:t+1], color="green", label="l(z)")
            if have_V:
                axlv.plot(np.arange(t+1), Vz[:t+1], color="blue", label="V(z)")
            axlv.set_xlim(0, T)
            axlv.axhline(0.0, linestyle="--", linewidth=1, color="black")
            axlv.set_title("Latent metrics")
            axlv.set_xlabel("t")
            axlv.legend(fontsize=6, loc="upper right")
            r += 1

        # --- Actions (final row, span) ---
        axa = fig.add_subplot(gs[r, :])
        # Example: plotting only z-action (index 2) like your original
        axa.plot(actions[:t+1, 2], linestyle="--", label="z-action")
        if have_safe:
            axa.plot(safe_actions[:t+1, 2], linestyle="-", label="z-safe")
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

def main(cfg, ckpt_path=None, wm=None, policy=None):
    cfg.num_actions = 7
    wm = build_world_model(cfg)
    policy = build_policy_for_value(cfg)

    path = "/data/mattkiim/merged_5hz.h5"
    video_dir = pathlib.Path("traj_videos"); video_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    j=0
    SEQ = 3
    device = next(wm.parameters()).device
    use_amp = True if getattr(cfg, "precision", 32) == 16 else False

    with h5py.File(path, "r") as f:
        for run in f:
            i += 1

            # if run != "trajectory_1":
            #     continue

            cam0 = f[run]["camera_0"][:]
            cam2 = f[run]["camera_2"][:]

            # if cam2.shape[3] == 1: 
            #     j += 1
            # print(cam2.shape, j)
            # continue
            heat_inner = np.squeeze(f[run]["hot_inner"][:], -1)
            arm_states = torch.tensor(f[run]['ee_states'][:])
            grip_states = torch.tensor(f[run]['gripper_states'][:])

            actions = f[run]["actions"][:]

            T = heat_inner.shape[0]

            # Failure labels (unchanged)
            failure = []
            if not cfg.avg:
                for t in range(T):
                    frame = heat_inner[t]
                    pixel_count = np.sum(frame > 0.0)
                    frac_too_hot = np.sum(frame > 0.6) / pixel_count if pixel_count > 0 else 0.0
                    failure.append(frac_too_hot > 0.5)
            else:
                for t in range(T):
                    frame = heat_inner[t]
                    heat_avg = np.mean(frame) if frame.size > 0 else 0.0
                    failure.append(heat_avg > 0.1)
            failure = np.asarray(failure, dtype=np.float32)

            # Compute l(z) and V(z)
            lz_list, V_list = [], []

            resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()  # -> CHW in [0,1]
            ])

            SEQ = 3
            rs_hist, flir_hist = [], []
            state_hist, action_hist = [], []
            safe_actions = []

            for t in range(T):
                # take current frames
                img_rs = cam0[t] * 255
                img_flir = cam2[t] * 255
                arm_state = arm_states[t]
                grip_state = grip_states[t]
                action = actions[t]
                norm_ac = normalize_acs(torch.tensor([action]).to(cfg.device), device=cfg.device)

                ee_state = eef_pose_to_state(
                    np.array(arm_state).reshape(4, 4).T,
                    np.array(grip_state)
                )

                rs_t   = resize_transform(img_rs.astype(np.uint8)).to(device)

                img_flir = img_flir[..., :1] # some IR images are stored as (H, W, 3)
                flir_t = resize_transform(img_flir.astype(np.uint8)).squeeze().to(device)
                flir_t = torch.stack([flir_t]*3, dim=0)

                # manage rolling window
                if len(rs_hist) == SEQ: rs_hist.pop(0)
                if len(flir_hist) == SEQ: flir_hist.pop(0)
                if len(state_hist) == SEQ: state_hist.pop(0)
                if len(action_hist) == SEQ: action_hist.pop(0)

                rs_hist.append(rs_t)
                flir_hist.append(flir_t)
                state_hist.append(torch.tensor(ee_state, device=device, dtype=torch.float32))
                action_hist.append(torch.tensor(norm_ac, device=device, dtype=torch.float32).squeeze())

                rs_seq_cf   = torch.stack(rs_hist, dim=0).unsqueeze(0).float()
                flir_seq_cf = torch.stack(flir_hist, dim=0).unsqueeze(0).float()
                states_seq  = torch.stack(state_hist, dim=0).unsqueeze(0).float()
                acs_seq     = torch.stack(action_hist, dim=0).unsqueeze(0).float()

                rs_seq_cl   = rs_seq_cf.permute(0, 1, 3, 4, 2)
                flir_seq_cl = flir_seq_cf.permute(0, 1, 3, 4, 2)
                flir_1ch    = flir_seq_cl[..., :1]

                B, L = acs_seq.shape[:2]
                is_first = torch.zeros((B, L, 1), dtype=torch.float32, device=device)
                is_term  = torch.zeros((B, L, 1), dtype=torch.float32, device=device)
                is_first[:, 0] = 1.0

                obs_batch = {
                    "obs_state":  states_seq,
                    "image":      rs_seq_cl,
                    "heat":       flir_1ch,
                    "action":     acs_seq,
                    "is_first":   is_first,
                    "is_terminal":is_term,
                }

                data  = wm.preprocess(obs_batch)

                with torch.cuda.amp.autocast(use_amp):
                    embed = wm.encoder(data)
                    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
                    feat  = wm.dynamics.get_feat(post)

                last_feat = feat[:, -1]
                margin = wm.heads["margin"](last_feat).item()

                lz_list.append(np.tanh(0.1 * margin))

                z_np   = last_feat.squeeze(0).detach().cpu().numpy()
                V_val  = evaluate_V(policy, z_np)[0]
                V_list.append(V_val)

                # Policy expects a Batch with obs in numpy shape (1, 544)
                tmp_batch = Batch(obs=z_np.reshape(1, -1), info=Batch())

                with torch.no_grad():
                    # Get normalized action from the (old) actor, then map_action to env space [-1,1]
                    a_norm = policy(tmp_batch, model="actor_old").act              # torch, normalized
                    a_norm = policy.map_action(a_norm).detach().cpu().numpy()[0]   # np, in [-1,1]

                # Optional: scale down for safety margin in normalized space
                a_norm = a_norm * SAFE_SCALE

                # Map back to original action scale (your dataset’s action space)
                a_safe = unnormalize_acs(torch.tensor(a_norm, device=device), device=device)
                a_safe = a_safe.detach().cpu().numpy()  # shape (7,)

                safe_actions.append(a_safe)

            safe_actions = np.asarray(safe_actions, dtype=np.float32)  # (T, 7)
            lz = np.asarray(lz_list, dtype=np.float32)
            Vz = np.asarray(V_list, dtype=np.float32)

            out_path = video_dir / f"traj_{i}.mp4"
            save_composite_video(
                cam0, cam2, heat_inner, failure, arm_states, actions, out_path, fps=20,
                lz=lz, Vz=Vz, safe_actions=safe_actions
            )
            print(f"[{run}] wrote {out_path}")


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
