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
    
    # TODO: only for eval trajectories
    
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
    cam0, cam2, hotinner, failure, eff_state, filename, fps=20,
    # Multi-policy inputs:
    policies_metrics=None,     # dict[name] -> {"lz": (T,), "Vz": (T,)}
    actions_dict=None,         # dict[name] -> (T, A)
    action_index=2,            # which action dim to overlay (z by default)
    # Backward-compat (single policy):
    actions=None,              # (T, A)
    lz=None, Vz=None,          # 1D arrays
    safe_actions=None,         # (T, A) optional
):
    """
    Composite video layout:
      Row 0: Cameras (cam0, cam2, hotinner)
      Row 1: Failure (steps)
      Row 2: V(z) per policy (SOLID; consistent color per policy)
      Row 3: l(z) per policy (DASHED; same color per policy)
      Row 4: Actions a[action_index] per policy (SOLID; same color per policy)

    Backward compatible with legacy lz/Vz/actions args (merged as policy 'policy').
    """
    # ---------- helpers ----------
    def _to_np(x):
        import numpy as _np, torch as _torch
        if isinstance(x, _np.ndarray): return x
        if isinstance(x, _torch.Tensor): return x.detach().cpu().numpy()
        return _np.asarray(x)

    def _to_uint8(x):
        import numpy as _np
        x = _to_np(x)
        if x.dtype == _np.uint8: return x
        return (_np.clip(x, 0, 1) * 255).astype(_np.uint8)

    def _expand3(x):
        import numpy as _np
        if x.ndim == 4 and x.shape[-1] == 1:  # (T,H,W,1)
            return _np.repeat(x, 3, axis=-1)
        if x.ndim == 3:                        # (H,W,?) -> (H,W,3)
            return _np.stack([x]*3, axis=-1)
        return x

    import numpy as np
    import imageio
    import matplotlib.pyplot as plt
    from io import BytesIO

    # ---------- coerce inputs / legacy merge ----------
    if policies_metrics is None: policies_metrics = {}
    if actions_dict is None: actions_dict = {}

    if lz is not None or Vz is not None:
        policies_metrics.setdefault("policy", {})
        if lz is not None: policies_metrics["policy"]["lz"] = _to_np(lz)
        if Vz is not None: policies_metrics["policy"]["Vz"] = _to_np(Vz)
    if actions is not None:
        actions_dict.setdefault("policy", _to_np(actions))
    if safe_actions is not None:
        actions_dict.setdefault("safe", _to_np(safe_actions))  # include only if you want it overlaid

    # ---------- normalize & align lengths ----------
    cam0, cam2, hotinner = map(_to_np, [cam0, cam2, hotinner])
    failure, eff_state   = map(_to_np, [failure, eff_state])
    cam0, cam2, hotinner = map(_to_uint8, [cam0, cam2, hotinner])
    cam0, cam2, hotinner = map(_expand3,   [cam0, cam2, hotinner])

    T = min(len(failure), len(cam0), len(cam2), len(hotinner))
    for d in policies_metrics.values():
        if "lz" in d: T = min(T, len(_to_np(d["lz"])))
        if "Vz" in d: T = min(T, len(_to_np(d["Vz"])))
    for arr in actions_dict.values():
        T = min(T, _to_np(arr).shape[0])

    failure = failure[:T]
    cam0, cam2, hotinner = cam0[:T], cam2[:T], hotinner[:T]
    for name, d in list(policies_metrics.items()):
        if "lz" in d: d["lz"] = _to_np(d["lz"])[:T]
        if "Vz" in d: d["Vz"] = _to_np(d["Vz"])[:T]
        if ("lz" not in d) and ("Vz" not in d):
            policies_metrics.pop(name)
    for name, arr in actions_dict.items():
        actions_dict[name] = _to_np(arr)[:T]

    have_metrics = len(policies_metrics) > 0
    have_actions = len(actions_dict) > 0

    # ---------- stable colors per policy across all plots ----------
    policy_names = sorted(set(list(policies_metrics.keys()) + list(actions_dict.keys())))
    base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', []) or [f"C{i}" for i in range(10)]
    policy_color = {name: base_colors[i % len(base_colors)] for i, name in enumerate(policy_names)}

    # ---------- layout (cams, failure, V(z), l(z), actions) ----------
    nrows = 2 + (1 if have_metrics else 0) + (1 if have_metrics else 0) + (1 if have_actions else 0)
    height = [3] + [1] * (nrows - 1)

    frames_out = []
    x_full = np.arange(T)

    for t in range(T):
        fig = plt.figure(figsize=(12, 2.75 * nrows))
        gs  = fig.add_gridspec(nrows, 3, height_ratios=height)

        # row 0: cameras
        ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[0, 2])
        ax0.imshow(cam0[t]); ax0.axis("off"); ax0.set_title("Camera 0")
        ax1.imshow(cam2[t]); ax1.axis("off"); ax1.set_title("Camera 2")
        ax2.imshow(hotinner[t]); ax2.axis("off"); ax2.set_title("Hot-Inner")

        r = 1
        # row 1: failure
        axf = fig.add_subplot(gs[r, :])
        axf.plot(x_full[:t+1], failure[:t+1], drawstyle="steps-post")
        axf.set_xlim(0, T); axf.set_ylim(-0.1, 1.1)
        axf.set_title("Failure Label"); axf.set_xlabel("t"); axf.set_ylabel("failure")
        r += 1

        # row 2: V(z) only (SOLID lines), one per policy
        if have_metrics:
            axV = fig.add_subplot(gs[r, :])
            for name in policy_names:
                d = policies_metrics.get(name, {})
                if "Vz" not in d: continue
                col = policy_color[name]
                axV.plot(x_full[:t+1], d["Vz"][:t+1], linestyle="-", color=col, label=f"{name} V(z)")
            axV.set_xlim(0, T)
            axV.axhline(0.3, linestyle="--", linewidth=1, color="k", alpha=0.25)
            axV.set_title("BRT V(z)")
            axV.set_xlabel("t")
            axV.legend(fontsize=6, loc="upper right")
            r += 1

        # row 3: l(z) only (DASHED lines), one per policy
        if have_metrics:
            axL = fig.add_subplot(gs[r, :])
            for name in policy_names:
                d = policies_metrics.get(name, {})
                if "lz" not in d: continue
                col = policy_color[name]
                axL.plot(x_full[:t+1], d["lz"][:t+1], linestyle="-", color=col, label=f"{name} l(z)")
            axL.set_xlim(0, T)
            axL.axhline(0.0, linestyle="--", linewidth=1, color="k", alpha=0.25)
            axL.set_title("Safety Margin l(z)")
            axL.set_xlabel("t")
            axL.legend(fontsize=6, loc="upper right")
            r += 1

        # row 4: Actions (same dimension, SOLID; consistent colors)
        if have_actions:
            axA = fig.add_subplot(gs[r, :])
            for name in policy_names:
                arr = actions_dict.get(name, None)
                if arr is None or arr.ndim != 2 or action_index >= arr.shape[1]:
                    continue
                col = policy_color[name]
                axA.plot(x_full[:t+1], arr[:t+1, action_index], linestyle="-", color=col, label=f"{name} a[{action_index}]")
            axA.set_xlim(0, T)
            axA.set_title(f"Actions (z-space)")
            axA.set_xlabel("t")
            axA.legend(fontsize=6, loc="upper right")

        fig.tight_layout()
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        frames_out.append(imageio.v3.imread(buf))
        plt.close(fig)

    imageio.mimsave(filename, frames_out, fps=fps)
    print(f"Saved composite video: {filename}")

def collect_policy_timeseries_open_loop(
    wm, policy,
    cam0, cam2, heat_inner, arm_states, grip_states, actions,
    device, use_heat: bool,
    warm: int = 5,
    precision16: bool = False,
):
    import numpy as np
    import torch
    from torchvision import transforms

    T = heat_inner.shape[0]
    use_amp = precision16

    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    Lw = min(warm, T)
    rs_list, heat1_list, state_list, act_list = [], [], [], []
    for t in range(Lw):
        rs_img = (cam0[t] * 255).astype(np.uint8)
        rs_list.append(resize_transform(rs_img).to(device))

        heat_img = (cam2[t] * 255).astype(np.uint8)
        h = resize_transform(heat_img).to(device)
        heat1_list.append(h)

        ee = eef_pose_to_state(np.array(arm_states[t]).reshape(4,4).T, np.array(grip_states[t]))
        state_list.append(torch.tensor(ee, device=device, dtype=torch.float32))

        norm_ac = normalize_acs(torch.tensor([actions[t]], device=device), device=device)
        act_list.append(norm_ac.squeeze(0))

    assert Lw > 0, "Need at least one warm step"

    rs_seq   = torch.stack(rs_list,   dim=0).unsqueeze(0).float().permute(0,1,3,4,2)
    heat_seq = torch.stack(heat1_list,dim=0).unsqueeze(0).float().permute(0,1,3,4,2)
    states_seq = torch.stack(state_list, dim=0).unsqueeze(0).float()
    acs_seq    = torch.stack(act_list,  dim=0).unsqueeze(0).float()

    is_first = torch.zeros((1, Lw, 1), dtype=torch.float32, device=device)
    is_term  = torch.zeros((1, Lw, 1), dtype=torch.float32, device=device)
    is_first[:, 0] = 1.0

    obs_batch = {
        "obs_state":   states_seq,
        "image":       rs_seq,
        "heat":        heat_seq if use_heat else torch.zeros_like(heat_seq),
        "action":      acs_seq,
        "is_first":    is_first,
        "is_terminal": is_term,
    }
    A = 7
    lz = np.full((T,), np.nan, dtype=np.float32)
    Vz = np.full((T,), np.nan, dtype=np.float32)
    acts_env = np.full((T, A), np.nan, dtype=np.float32)

    # closed loop
    data = wm.preprocess(obs_batch)
    
    embed = wm.encoder(data)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat_warm = wm.dynamics.get_feat(post)

    # closed l(z)
    margin_warm = wm.heads["margin"](feat_warm).squeeze(0).squeeze(-1)
    lz[:Lw] = np.tanh(0.1 * margin_warm.detach().cpu().numpy()).astype(np.float32)

    # closed V(z)
    feats_np_warm = feat_warm.squeeze(0).detach().cpu().numpy()
    V_list_warm = [float(evaluate_V(policy, feats_np_warm[k])[0]) for k in range(feats_np_warm.shape[0])]
    Vz[:Lw] = np.asarray(V_list_warm, dtype=np.float32)

    # # open loop
    if T > Lw:
        ds_actions_norm = normalize_acs(torch.tensor(actions[Lw:T], device=device), device=device)
        a_seq = ds_actions_norm.unsqueeze(0).float()
        curr_boundary = {k: v[:, Lw-1] for k, v in post.items()}

        prior_tail = wm.dynamics.imagine_with_action(a_seq, curr_boundary)
        imag_feat  = wm.dynamics.get_feat(prior_tail) 

        # l(z)
        margins_tail = wm.heads["margin"](imag_feat).squeeze(0).squeeze(-1)
        lz[Lw:T] = np.tanh(0.1 * margins_tail.detach().cpu().numpy()).astype(np.float32)

        # V(z)
        feats_np_tail = imag_feat.squeeze(0).detach().cpu().numpy()
        V_list_tail = [float(evaluate_V(policy, feats_np_tail[k])[0]) for k in range(feats_np_tail.shape[0])]
        Vz[Lw:T] = np.asarray(V_list_tail, dtype=np.float32)

        acts_env[Lw:T] = np.asarray(actions[Lw:T], dtype=np.float32)
    else:
        print("T cannot be < Lw"); quit()

    return {"lz": lz, "Vz": Vz, "actions": acts_env}


def collect_policy_timeseries(
    wm, policy, cam0, cam2, heat_inner, arm_states, grip_states, actions,
    device, use_heat: bool, seq_len: int = 3, safe_scale: float = SAFE_SCALE,
    precision16: bool = False,
):
    T = heat_inner.shape[0]
    use_amp = precision16

    resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    rs_hist, flir_hist, state_hist, action_hist = [], [], [], []
    lz_list, V_list, act_list = [], [], []

    for t in range(T):
        img_rs  = (cam0[t] * 255).astype(np.uint8)
        img_flr = (cam2[t] * 255).astype(np.uint8)
        arm     = torch.tensor(arm_states[t])
        grip    = torch.tensor(grip_states[t])
        act_env = actions[t]  # original env scale in dataset

        norm_ac = normalize_acs(torch.tensor([act_env]).to(device), device=device)
        ee_state = eef_pose_to_state(np.array(arm).reshape(4, 4).T, np.array(grip))

        rs_t = resize_transform(img_rs).to(device)

        img_flr_1 = torch.stack([torch.tensor(img_flr).squeeze()]*3, dim=0) if img_flr.shape[-1] == 1 else img_flr
        flir_t = resize_transform(img_flr_1.squeeze()).to(device)

        # rolling window
        if len(rs_hist) == seq_len:   rs_hist.pop(0)
        if len(flir_hist) == seq_len: flir_hist.pop(0)
        if len(state_hist) == seq_len:state_hist.pop(0)
        if len(action_hist) == seq_len:action_hist.pop(0)

        rs_hist.append(rs_t)
        flir_hist.append(flir_t)
        state_hist.append(torch.tensor(ee_state, device=device, dtype=torch.float32))
        action_hist.append(torch.tensor(norm_ac, device=device, dtype=torch.float32).squeeze())

        rs_seq   = torch.stack(rs_hist,   dim=0).unsqueeze(0).float().permute(0,1,3,4,2)
        flir_seq = torch.stack(flir_hist, dim=0).unsqueeze(0).float().permute(0,1,3,4,2)
        
        flir_1ch = flir_seq[..., :1]
        
        states_seq = torch.stack(state_hist, dim=0).unsqueeze(0).float()
        acs_seq    = torch.stack(action_hist, dim=0).unsqueeze(0).float()

        B, L = acs_seq.shape[:2]
        is_first = torch.zeros((B, L, 1), dtype=torch.float32, device=device)
        is_term  = torch.zeros((B, L, 1), dtype=torch.float32, device=device)
        is_first[:, 0] = 1.0

        obs_batch = {
            "obs_state":   states_seq,
            "image":       rs_seq,
            "heat":        flir_1ch if use_heat else torch.zeros_like(flir_1ch),
            "action":      acs_seq,
            "is_first":    is_first,
            "is_terminal": is_term,
        }

        data = wm.preprocess(obs_batch)

        embed = wm.encoder(data)
        post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat  = wm.dynamics.get_feat(post)

        last_feat = feat[:, -1]
        # print(last_feat.shape); quit()
        margin = wm.heads["margin"](last_feat).item()
        lz_list.append(np.tanh(0.1 * margin))

        z_np  = last_feat.squeeze(0).detach().cpu().numpy()
        V_val = evaluate_V(policy, z_np)[0]
        V_list.append(V_val)

        tmp_batch = Batch(obs=z_np.reshape(1, -1), info=Batch())
        with torch.no_grad():
            a_norm = policy(tmp_batch, model="actor_old").act
            a_norm = policy.map_action(a_norm).detach().cpu().numpy()[0]
        a_norm = a_norm * safe_scale
        a_env  = unnormalize_acs(torch.tensor(a_norm, device=device), device=device).detach().cpu().numpy()
        act_list.append(a_env)

    return {
        "lz": np.asarray(lz_list, dtype=np.float32),
        "Vz": np.asarray(V_list, dtype=np.float32),
        "actions": np.asarray(act_list, dtype=np.float32),
    }


def main(cfg, ckpt_path=None, wm=None, policy=None):
    cfg.num_actions = 7

    # --- Build both agents
    # RGB-only
    cfg_rgb = argparse.Namespace(**vars(cfg))
    cfg_rgb.eval_rssm_ckpt_path  = cfg.eval_rssm_rgb_only_ckpt_path
    cfg_rgb.eval_policy_ckpt_path= cfg.eval_policy_rgb_only_ckpt_path
    cfg_rgb.rgb_only = True
    wm_rgb      = build_world_model(cfg_rgb)
    policy_rgb  = build_policy_for_value(cfg_rgb)

    # Multimodal
    cfg_mm = argparse.Namespace(**vars(cfg))
    cfg_mm.eval_rssm_ckpt_path   = cfg.eval_rssm_mm_ckpt_path
    cfg_mm.eval_policy_ckpt_path = cfg.eval_policy_mm_ckpt_path
    wm_mm       = build_world_model(cfg_mm)
    policy_mm   = build_policy_for_value(cfg_mm)

    path = cfg.dataset_path
    vid_path = f"traj_videos/test"
    video_dir = pathlib.Path(vid_path); video_dir.mkdir(parents=True, exist_ok=True)

    device = next(wm_rgb.parameters()).device
    use_amp = True if getattr(cfg, "precision", 32) == 16 else False

    with h5py.File(path, "r") as f:
        s = 222
        e = s + 21
        for i, run in enumerate(f, start=1):
            cam0        = f[run]["camera_0"][s:e]
            cam2        = f[run]["camera_2"][s:e]
            heat_inner  = np.squeeze(f[run]["hot_inner"][s:e], -1)
            arm_states  = torch.tensor(f[run]['ee_states'][s:e])
            grip_states = torch.tensor(f[run]['gripper_states'][s:e])
            actions_ds  = f[run]["actions"][s:e]
            T           = heat_inner.shape[0]

            if i != 27: continue

            # Failure labels (same as your code)
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

            # --- Roll out both policies (open-loop with dataset actions)
            rgb_ts = collect_policy_timeseries(
                wm_rgb, policy_rgb, cam0, cam2, heat_inner, arm_states, grip_states, actions_ds,
                device=device, use_heat=False, seq_len=5, safe_scale=SAFE_SCALE, precision16=use_amp
            )
            mm_ts = collect_policy_timeseries(
                wm_mm, policy_mm, cam0, cam2, heat_inner, arm_states, grip_states, actions_ds,
                device=device, use_heat=True, seq_len=5, safe_scale=SAFE_SCALE, precision16=use_amp
            )
            
            # --- Package for plotting: two policies on shared axes
            policies_metrics = {
                "rgb_only":   {"lz": rgb_ts["lz"], "Vz": rgb_ts["Vz"]},
                "multimodal": {"lz": mm_ts["lz"],  "Vz": mm_ts["Vz"]},
            }
            actions_dict = {
                "rgb_only":   rgb_ts["actions"],   # (T, 7)
                "multimodal": mm_ts["actions"],    # (T, 7)
            }

            out_path = video_dir / f"traj_{i}.mp4"
            save_composite_video(
                cam0, cam2, heat_inner, failure, arm_states,
                filename=out_path, fps=20,
                policies_metrics=policies_metrics,
                actions_dict=actions_dict,
                action_index=2,  # plot z for both policies on one plot
            )
            print(f"[{run}] wrote {out_path}")

            
            rgb_ts = collect_policy_timeseries_open_loop(
                wm_rgb, policy_rgb,
                cam0, cam2, heat_inner, arm_states, grip_states, actions_ds,
                device=device, use_heat=False, warm=5, precision16=use_amp
            )
            mm_ts = collect_policy_timeseries_open_loop(
                wm_mm, policy_mm,
                cam0, cam2, heat_inner, arm_states, grip_states, actions_ds,
                device=device, use_heat=True, warm=5, precision16=use_amp
            )

            # --- Package for plotting: two policies on shared axes
            policies_metrics = {
                "rgb_only":   {"lz": rgb_ts["lz"], "Vz": rgb_ts["Vz"]},
                "multimodal": {"lz": mm_ts["lz"],  "Vz": mm_ts["Vz"]},
            }
            actions_dict = {
                "rgb_only":   rgb_ts["actions"],   # (T, 7)
                "multimodal": mm_ts["actions"],    # (T, 7)
            }

            out_path = video_dir / f"traj_{i}_ol.mp4"
            save_composite_video(
                cam0, cam2, heat_inner, failure, arm_states,
                filename=out_path, fps=20,
                policies_metrics=policies_metrics,
                actions_dict=actions_dict,
                action_index=2,  # plot z for both policies on one plot
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
