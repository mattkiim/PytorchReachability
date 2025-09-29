import os, sys, pathlib
import argparse
import numpy as np
import torch
import h5py
import ruamel.yaml as yaml
from PyHJ.data import Batch
from torchvision import transforms
import matplotlib.pyplot as plt
from io import BytesIO
import imageio

from scipy.spatial.transform import Rotation as R

# project paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import models, tools
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
from PyHJ.exploration import GaussianNoise


IMG_SIZE = 224
SAFE_SCALE = 0.15

def normalize_acs(acs, device):
    max_ac = torch.tensor([0.10543401, 0.00693101, 1., 0.00348351,
                           0.04336035, 0.00416869, 1.]).to(device)
    min_ac = torch.tensor([-0.07480399, -0.00952092, -0.80855778, -0.00182587,
                           -0.03781896, -0.00121621, -1.]).to(device)
    acs = (acs - min_ac) / (max_ac - min_ac)
    return (acs * 2) - 1

def unnormalize_acs(acs, device):
    max_ac = torch.tensor([0.10543401, 0.00693101, 1., 0.00348351,
                           0.04336035, 0.00416869, 1.]).to(device)
    min_ac = torch.tensor([-0.07480399, -0.00952092, -0.80855778, -0.00182587,
                           -0.03781896, -0.00121621, -1.]).to(device)
    acs = (acs + 1) / 2.0
    return acs * (max_ac - min_ac) + min_ac


def eef_pose_to_state(T, gripper):
    x, y, z = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate(([x, y, z], quat, [gripper]))


def build_world_model(cfg):
    import gym
    device = cfg.device
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    cam_obs_space = gym.spaces.Box(low=0, high=1, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    bool_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
    obs_observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
    heat_obs_space = gym.spaces.Box(low=0, high=1, shape=(IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    observation_space = gym.spaces.Dict({
        'obs_state': obs_observation_space,
        'image': cam_obs_space,
        'heat': heat_obs_space,
        'is_first': bool_space,
        'is_last': bool_space,
        'is_terminal': bool_space,
    })
    wm = models.WorldModel(observation_space, action_space, 0, cfg).to(device)
    ckpt = torch.load(cfg.eval_rssm_ckpt_path, map_location=device, weights_only=True)
    state_dict = {k[14:]: v for k, v in ckpt['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    return wm


def build_policy_for_value(cfg):
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
        reward_normalization=cfg.rew_norm, estimation_step=cfg.n_step,
        actor_gradient_steps=cfg.actor_gradient_steps,
    )
    sd = torch.load(cfg.eval_policy_ckpt_path, map_location=device)
    policy.load_state_dict(sd)
    return policy


def evaluate_V(policy, z):
    tmp_batch = Batch(obs=z.reshape(1, -1), info=Batch())
    tmp = policy.critic_old(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()[0]


def collect_policy_timeseries(wm, policy, cam0, cam2, heat, arm_states, grip_states, actions, device, use_heat):
    seq_len = 3
    resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    rs_hist, flir_hist, state_hist, action_hist = [], [], [], []
    lz_list, Vz_list, act_list = [], [], []
    for t in range(len(cam0)):
        rs_t = torch.tensor(cam0[t]).permute(2,0,1).to(device)
        flir_t = torch.tensor(cam2[t][..., :1]).permute(2,0,1).to(device)
        flir_t = torch.cat([flir_t]*3, dim=0)
        if not use_heat: flir_t *= 0.0

        ee_state = eef_pose_to_state(arm_states[t].reshape(4,4).T, grip_states[t])
        norm_ac = normalize_acs(torch.tensor([actions[t]], device=device), device=device)

        if len(rs_hist) == seq_len: rs_hist.pop(0)
        if len(flir_hist) == seq_len: flir_hist.pop(0)
        if len(state_hist) == seq_len: state_hist.pop(0)
        if len(action_hist) == seq_len: action_hist.pop(0)

        rs_hist.append(rs_t); flir_hist.append(flir_t)
        state_hist.append(torch.tensor(ee_state, device=device, dtype=torch.float32))
        action_hist.append(norm_ac.squeeze())

        if len(rs_hist) < seq_len: continue

        rs_seq = torch.stack(rs_hist, dim=0).unsqueeze(0).float().permute(0,1,3,4,2)
        flir_seq = torch.stack(flir_hist, dim=0).unsqueeze(0).float().permute(0,1,3,4,2)
        flir_1ch = flir_seq[..., :1]

        states_seq = torch.stack(state_hist, dim=0).unsqueeze(0).float()
        acs_seq = torch.stack(action_hist, dim=0).unsqueeze(0).float()
        is_first = torch.zeros((1, seq_len, 1), dtype=torch.float32, device=device)
        is_first[:, 0] = 1.0
        data = wm.preprocess({
            "obs_state": states_seq,
            "image": rs_seq,
            "heat": flir_1ch if use_heat else torch.zeros_like(flir_1ch),
            "action": acs_seq,
            "is_first": is_first,
            "is_terminal": torch.zeros_like(is_first),
        })
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)
        z = feat[:, -1].detach().cpu().numpy().squeeze()

        # value
        Vz_list.append(evaluate_V(policy, z))
        # margin (safety)
        margin = wm.heads["margin"](feat[:, -1]).item()
        lz_list.append(np.tanh(0.1 * margin))
        act_list.append(actions[t])
    return {"lz": np.array(lz_list), "Vz": np.array(Vz_list), "actions": np.array(act_list)}


def main(cfg):
    cfg.num_actions = 7
    use_heat = True  # set False for rgb-only
    cfg.eval_rssm_ckpt_path = cfg.eval_rssm_mm_ckpt_path if use_heat else cfg.eval_rssm_rgb_only_ckpt_path
    cfg.eval_policy_ckpt_path = cfg.eval_policy_mm_ckpt_path if use_heat else cfg.eval_policy_rgb_only_ckpt_path
    cfg.dataset_path = cfg.eval_mm_dataset_path if use_heat else cfg.eval_rgb_dataset_path
    cfg.dataset_name = "multimodal" if use_heat else "rgb_only"

    wm = build_world_model(cfg)
    policy = build_policy_for_value(cfg)
    device = cfg.device

    with h5py.File(cfg.dataset_path, "r") as f:
        for i, traj in enumerate(f.keys()):
            cam0 = f[traj]["camera_0"][:]
            cam2 = f[traj]["camera_2"][:]
            heat = f[traj]["hot_inner"][:]
            arm_states = f[traj]["ee_states"][:]
            grip_states = f[traj]["gripper_states"][:]
            actions = f[traj]["actions"][:]
            print(f"Running {traj}...")

            ts = collect_policy_timeseries(wm, policy, cam0, cam2, heat, arm_states, grip_states, actions, device, use_heat)

            # simple plot example
            plt.figure(figsize=(10,5))
            plt.plot(ts["Vz"], label="V(z)")
            plt.plot(ts["lz"], label="l(z)")
            plt.legend()
            plt.title(f"{cfg.dataset_name} — {traj}")
            plt.savefig(f"traj_{i}_{cfg.dataset_name}.png")
            plt.close()
            print(f"Saved plot traj_{i}_{cfg.dataset_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs.yaml")
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
        parser.add_argument(f"--{k}", type=tools.args_type(v), default=v)
    cfg = parser.parse_args(remaining)

    main(cfg)
