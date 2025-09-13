import os, sys, pathlib
import argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import ruamel.yaml as yaml

from PyHJ.data import Batch
from scipy.spatial.transform import Rotation as R

# project paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import models, tools


IMG_SIZE = 224
SAFE_SCALE = 0.15


def normalize_acs(acs, device):
    max_ac = torch.tensor([0.10543401, 0.00693101, 1., 0.00348351,
                           0.04336035, 0.00416869, 1.]).to(device)
    min_ac = torch.tensor([-0.07480399, -0.00952092, -0.80855778, -0.00182587,
                           -0.03781896, -0.00121621, -1.]).to(device)
    acs = (acs - min_ac) / (max_ac - min_ac)
    return (acs * 2) - 1


def eef_pose_to_state(T, gripper):
    x, y, z = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate(([x, y, z], quat, [gripper]))


def build_world_model(cfg):
    import gym
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    obs_observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
    cam_obs_space = gym.spaces.Box(low=0, high=1, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    heat_obs_space = gym.spaces.Box(low=0, high=1, shape=(IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    bool_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)

    observation_space = gym.spaces.Dict({
        'obs_state': obs_observation_space,
        'image': cam_obs_space,
        'heat': heat_obs_space,
        'is_first': bool_space,
        'is_last': bool_space,
        'is_terminal': bool_space,
    })

    wm = models.WorldModel(observation_space, action_space, 0, cfg).to(cfg.device)
    ckpt = torch.load(cfg.eval_rssm_ckpt_path, map_location=cfg.device, weights_only=True)
    state_dict = {k[14:]: v for k, v in ckpt['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    return wm


def build_policy_for_value(cfg):
    from PyHJ.utils.net.common import Net
    from PyHJ.utils.net.continuous import Actor, Critic
    from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
    from PyHJ.exploration import GaussianNoise

    device = cfg.device
    state_shape = 544
    action_shape = (7,)

    critic_net = Net(state_shape, action_shape, hidden_sizes=cfg.critic_net,
                     activation=torch.nn.ReLU, concat=True, device=device)
    critic = Critic(critic_net, device=device).to(device)

    actor_net = Net(state_shape, hidden_sizes=cfg.control_net,
                    activation=torch.nn.ReLU, device=device)
    actor = Actor(actor_net, action_shape, max_action=1.0, device=device).to(device)

    policy = DDPGPolicy(
        critic, torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr),
        actor=actor, actor_optim=torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr),
        exploration_noise=GaussianNoise(sigma=cfg.exploration_noise),
        tau=cfg.tau, gamma=cfg.gamma_pyhj,
        action_space=None, reward_normalization=cfg.rew_norm,
        estimation_step=cfg.n_step, actor_gradient_steps=cfg.actor_gradient_steps,
    )

    sd = torch.load(cfg.eval_policy_ckpt_path, map_location=device)
    policy.load_state_dict(sd)
    return policy


def evaluate_V(policy, z):
    tmp_batch = Batch(obs=z.reshape(1, -1), info=Batch())
    tmp = policy.critic_old(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()[0]


def first_crossing_intensity(wm, policy, cam0, cam2, heat, arm_states, grip_states, actions, device, use_heat):
    """
    Returns avg pixel intensity at first timestep where V(z) >= 0.3, else None.
    """
    resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    rs_hist, flir_hist, state_hist, action_hist = [], [], [], []
    seq_len = 5

    T = cam0.shape[0]
    for t in range(T):
        img_rs = cam0[t]
        rs_t = torch.tensor(img_rs).permute(2, 0, 1).to(device) # shape 224 224 3

        img_flr = cam2[t]
        flir_t = torch.tensor(img_flr[..., :1]).permute(2, 0, 1).to(device)
        # print(flir_t.shape) # shape 224 224 1
        flir_t = torch.cat([flir_t]*3, dim=0)
        
        if not use_heat: 
            flir_t *= 0.0
        
        # print(rs_t.max(), flir_t.max()); quit()

        ee_state = eef_pose_to_state(arm_states[t].reshape(4, 4).T, grip_states[t])
        norm_ac = normalize_acs(torch.tensor([actions[t]], device=device), device=device)

        if len(rs_hist) == seq_len: rs_hist.pop(0)
        if len(flir_hist) == seq_len: flir_hist.pop(0)
        if len(state_hist) == seq_len: state_hist.pop(0)
        if len(action_hist) == seq_len: action_hist.pop(0)
        
        
        # print(rs_t.shape, flir_t.shape); quit()

        rs_hist.append(rs_t)
        flir_hist.append(flir_t)
        state_hist.append(torch.tensor(ee_state, device=device, dtype=torch.float32))
        action_hist.append(norm_ac.squeeze())

        if len(rs_hist) < seq_len: 
            continue

        rs_seq = torch.stack(rs_hist, dim=0).unsqueeze(0).float().permute(0,1,3,4,2)
        flir_seq = torch.stack(flir_hist, dim=0).unsqueeze(0).float().permute(0,1,3,4,2)
        flir_1ch = flir_seq[..., :1]

        states_seq = torch.stack(state_hist, dim=0).unsqueeze(0).float()
        acs_seq = torch.stack(action_hist, dim=0).unsqueeze(0).float()

        B, L = acs_seq.shape[:2]
        is_first = torch.zeros((B, L, 1), dtype=torch.float32, device=device)
        is_term  = torch.zeros((B, L, 1), dtype=torch.float32, device=device)
        is_first[:, 0] = 1.0

        obs_batch = {
            "obs_state": states_seq,
            "image": rs_seq,
            "heat": flir_1ch if use_heat else torch.zeros_like(flir_1ch),
            "action": acs_seq,
            "is_first": is_first,
            "is_terminal": is_term,
        }

        data = wm.preprocess(obs_batch)
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)
        z = feat[:, -1].detach().cpu().numpy().squeeze()

        Vz = evaluate_V(policy, z)
        if Vz <= 0.3:
            return heat[t].mean()

    return None


import json

def run_histogram(cfg, deployments):
    results = {name: [] for name in deployments}
    hist_data = {}

    for name, (rssm_ckpt, policy_ckpt, use_heat) in deployments.items():
        cfg.eval_rssm_ckpt_path = rssm_ckpt
        cfg.eval_policy_ckpt_path = policy_ckpt
        wm = build_world_model(cfg)
        policy = build_policy_for_value(cfg)
        device = cfg.device

        with h5py.File(cfg.dataset_path, "r") as f:
            for run in list(f.keys()):
                cam0 = f[run]["camera_0"][:]
                cam2 = f[run]["camera_2"][:]
                heat = f[run]["hot_inner"][:]
                arm_states  = f[run]['ee_states'][:]
                grip_states = f[run]['gripper_states'][:]
                actions     = f[run]["actions"][:]

                val = first_crossing_intensity(
                    wm, policy, cam0, cam2, heat,
                    arm_states, grip_states, actions,
                    device=device, use_heat=use_heat
                )
                if val is not None:
                    results[name].append(val)

        vals = results[name]
        print(vals)

        counts, bin_edges = np.histogram(vals, bins=40, range=(-0.1, 0.2))
        hist_data[name] = {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist()
        }

        plt.figure(figsize=(6,4))
        plt.hist(vals, bins=40, alpha=0.7, color="C0", range=(-0.1, 0.2))
        plt.xlabel("Avg pixel intensity at first V(z) ≥ 0.3")
        plt.ylabel("Count")
        plt.title(f"Avg Pixel Intensity Histogram {name}")
        plt.tight_layout()
        plt.savefig(f"eval/histograms/histogram_vz03_{name}.png")
        plt.close()

    # with open("eval/histograms/histogram_values_masked.json", "w") as f:
    #     json.dump(hist_data, f, indent=2)

    return hist_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs.yaml", type=str)
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load((pathlib.Path(sys.argv[0]).parent / f"../{args.config_path}").read_text())

    def recursive_update(base, update):
        for k,v in update.items():
            if isinstance(v, dict) and k in base:
                recursive_update(base[k], v)
            else:
                base[k] = v

    defaults = {}
    for n in ["defaults", *args.configs] if args.configs else ["defaults"]:
        recursive_update(defaults, configs[n])

    parser = argparse.ArgumentParser()
    for k,v in sorted(defaults.items()):
        arg_type = tools.args_type(v)
        parser.add_argument(f"--{k}", type=arg_type, default=arg_type(v))
    cfg = parser.parse_args(remaining)
    cfg.num_actions = 7

    deployments = {
        # "rgb": (cfg.eval_rssm_rgb_only_ckpt_path, cfg.eval_policy_rgb_only_ckpt_path, False),
        # "mm": (cfg.eval_rssm_mm_ckpt_path, cfg.eval_policy_mm_ckpt_path, True),
        "masked": (cfg.rssm_ckpt_path, cfg.eval_policy_ckpt_path, True),
    }

    run_histogram(cfg, deployments)
