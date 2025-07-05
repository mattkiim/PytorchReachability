import argparse
import os
import sys
import pickle

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import gymnasium #as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from itertools import product

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/PyHJ'))
sys.path.append(saferl_dir)
import models
import tools
import ruamel.yaml as yaml
import wandb
from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs

from termcolor import cprint
from datetime import datetime
import pathlib
from pathlib import Path
import collections
from PIL import Image
import io
from PyHJ.data import Batch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# note: need to include the dreamerv3 repo for this
from dreamer import make_dataset
from generate_data_traj_cont import get_frame_eval, HeatFrameGenerator

# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
"""
    Note that, we can pass arguments to the script by using
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9
    
    For learning the classical reach-avoid value function (baseline):
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True

"""
def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--config_path", default="configs.yaml", type=str)
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    # environment parameters
    config, remaining = parser.parse_known_args()


    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yml = yaml.YAML(typ="safe", pure=True)
    configs = yml.load(
        (pathlib.Path(sys.argv[0]).parent / f"../{config.config_path}").read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir+'/PyHJ'}/{config.expt_name}"
    # final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir+'/PyHJ'}", "cyan", attrs=["bold"])
    print("---------------------")
    return final_config



args = get_args()
config = args


env = gymnasium.make(args.task, params=[config])
config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]

if config.multimodal:
    env.observation_space_full['image'] = gymnasium.spaces.Box(
        low=0,
        high=255,
        shape=(128, 128, 3), # TODO: softcode input
        dtype=np.uint8
    )
    env.observation_space_full['heat'] = gymnasium.spaces.Box(
        low=0,
        high=255,
        shape=(128, 128, 1), # TODO: softcode input
        dtype=np.uint8
    )

wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)
# print(env.observation_space_full['image']); quit()

ckpt_path = config.rssm_ckpt_path
checkpoint = torch.load(ckpt_path, weights_only=True)
state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
wm.load_state_dict(state_dict)
wm.eval()

offline_eps = collections.OrderedDict()
config.batch_size = 1
config.batch_length = 2

config.dataset_path = f"{config.dataset_path}_{config.alpha_in}.pkl"
tools.fill_expert_dataset_dubins(config, offline_eps)
offline_dataset = make_dataset(offline_eps, config)

env.set_wm(wm, offline_dataset, config)


# check if the environment has control and disturbance actions:
assert hasattr(env, 'action_space') #and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
args.max_action = env.action_space.high[0]



train_envs = DummyVectorEnv(
    [lambda: gymnasium.make(args.task, params = [wm, offline_dataset, config]) for _ in range(args.training_num)]
)
test_envs = DummyVectorEnv(
    [lambda: gymnasium.make(args.task, params = [wm, offline_dataset, config]) for _ in range(args.test_num)]
)
# -------
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed) # problematic line
test_envs.seed(args.seed)

# -------

# model
if args.actor_activation == 'ReLU':
    actor_activation = torch.nn.ReLU
elif args.actor_activation == 'Tanh':
    actor_activation = torch.nn.Tanh
elif args.actor_activation == 'Sigmoid':
    actor_activation = torch.nn.Sigmoid
elif args.actor_activation == 'SiLU':
    actor_activation = torch.nn.SiLU

if args.critic_activation == 'ReLU':
    critic_activation = torch.nn.ReLU
elif args.critic_activation == 'Tanh':
    critic_activation = torch.nn.Tanh
elif args.critic_activation == 'Sigmoid':
    critic_activation = torch.nn.Sigmoid
elif args.critic_activation == 'SiLU':
    critic_activation = torch.nn.SiLU

if args.critic_net is not None:
    critic_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.critic_net,
        activation=critic_activation,
        concat=True,
        device=args.device
    )
else:
    # report error:
    raise ValueError("Please provide critic_net!")


critic = Critic(critic_net, device=args.device).to(args.device)
critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj)
critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=critic_optim, gamma=0.995)

log_path = None

from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

actor_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
actor = Actor(
    actor_net, args.action_shape, max_action=args.max_action, device=args.device
).to(args.device)
actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=actor_optim, gamma=1.0)


policy = DDPGPolicy(
critic,
critic_optim,
critic_scheduler=critic_scheduler,
tau=args.tau,
gamma=args.gamma_pyhj,
exploration_noise=GaussianNoise(sigma=args.exploration_noise),
reward_normalization=args.rew_norm,
estimation_step=args.n_step,
action_space=env.action_space,
actor=actor,
actor_optim=actor_optim,
actor_scheduler=actor_scheduler,
actor_gradient_steps=args.actor_gradient_steps,
)

log_path = os.path.join(args.logdir+'/PyHJ', args.task, 'wm_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_gamma_{}'.format(
args.actor_activation, 
args.critic_activation, 
args.actor_gradient_steps,args.tau, 
args.training_num, 
args.buffer_size,
args.critic_net[0],
len(args.critic_net),
args.control_net[0],
len(args.control_net),
args.gamma_pyhj)
)


# collector
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(args.buffer_size, len(train_envs)),
    exploration_noise=True
)
test_collector = Collector(policy, test_envs)

if args.warm_start_path is not None:
    policy.load_state_dict(torch.load(args.warm_start_path, weights_only=True))
    args.kwargs = args.kwargs + "warmstarted"

epoch = 0
# writer = SummaryWriter(log_path, filename_suffix="_"+timestr+"epoch_id_{}".format(epoch))
# logger = TensorboardLogger(writer)
log_path = log_path+'/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
        args.exploration_noise, 
        args.actor_lr, 
        args.critic_lr, 
        args.batch_size_pyhj,
        args.step_per_epoch,
        args.kwargs,
        args.seed
    )


if args.continue_training_epoch is not None:
    epoch = args.continue_training_epoch
    policy.load_state_dict(torch.load(
        os.path.join(
            log_path+"/epoch_id_{}".format(epoch),
            "policy.pth"
        ),
        weights_only=True
    ))


if args.continue_training_logdir is not None:
    policy.load_state_dict(torch.load(args.continue_training_logdir, weights_only=True))
    # epoch = int(args.continue_training_logdir.split('_')[-9].split('_')[0])
    epoch = args.continue_training_epoch

def save_best_fn(policy, epoch=epoch):
    torch.save(
        policy.state_dict(), 
        os.path.join(
            log_path+"/epoch_id_{}".format(epoch),
            "policy.pth"
        )
    )

def stop_fn(mean_rewards):
    return False

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img.convert('RGB')

if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path+"/epoch_id_{}".format(epoch))

def make_cache(config, thetas, heat_values):
    nx, ny = config.nx, config.ny
    cache = {}

    xs = np.linspace(-1.5, 1.5, nx, endpoint=True)
    ys = np.linspace(-1.5, 1.5, ny, endpoint=True)

    for heat_value in heat_values:
        for theta in thetas:
            v = np.zeros((nx, ny))
            key = (theta, heat_value)
            print('creating cache for key', key)

            idxs = []
            imgs_prev = []
            heat_imgs = []
            no_heat_imgs = []
            thetas_prev = []
            heat_values_prev = []
            states = []

            # Compute previous positions assuming constant motion
            xs_prev = xs - config.dt * config.speed * np.cos(theta)
            ys_prev = ys - config.dt * config.speed * np.sin(theta)
            theta_prev = theta

            it = np.nditer(v, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                x_prev = xs_prev[idx[0]]
                y_prev = ys_prev[idx[1]]

                prev_state = torch.tensor([x_prev, y_prev, theta_prev])
                img = get_frame_eval(prev_state, config)
                gen = HeatFrameGenerator(config)
                gen._compute_geometry(img.shape)

                # Generate heat and no-heat images
                # TODO: when i make the images, i need to get them at specific heat_values. theta is observable, but we are not doing that
                if config.heat_mode == 0:
                    heat = gen.get_heat_frame_v0(img, heat=True)
                    no_heat = gen.get_heat_frame_v0(img, heat=False)
                elif config.heat_mode == 1:
                    heat = gen.get_heat_frame_v1(img, heat=True)
                    no_heat = gen.get_heat_frame_v1(img, heat=False)
                elif config.heat_mode == 2:
                    img = gen.get_rgb_v2(img, config, heat=True)
                    heat, _ = gen.get_heat_frame_v2(img, heat=True, heat_value=heat_value)
                    no_heat, _ = gen.get_heat_frame_v2(img, heat=False, heat_value=heat_value)
                elif config.heat_mode == 3:
                    img = gen.get_rgb_v3(img, config, heat=True, heat_value=heat_value)
                    heat, _ = gen.get_heat_frame_v3(img, heat=True, heat_value=heat_value)
                    no_heat, _ = gen.get_heat_frame_v3(img, heat=False, heat_value=heat_value)
                else:
                    raise ValueError(f"Unknown heat_mode: {config.heat_mode}")

                # Accumulate data
                idxs.append(idx)
                imgs_prev.append(img)
                heat_imgs.append(heat)
                no_heat_imgs.append(no_heat)
                thetas_prev.append(theta_prev)
                heat_values_prev.append(heat_value)
                states.append(prev_state)

                it.iternext()

            # Convert to arrays
            idxs = np.array(idxs)
            theta_prev_lin = np.array(thetas_prev)
            heat_values_prev = np.array(heat_values_prev)

            cache[key] = [
                idxs,
                imgs_prev,
                heat_imgs,
                no_heat_imgs,
                theta_prev_lin,
                states,
            ]
            
    cache_path = f"{config.hj_cache_path}_{config.alpha_in}.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    return cache


def load_cache(config):
    cache_path = f"{config.hj_cache_path}_{config.alpha_in}.pkl"

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found at: {cache_path}")
    else:
        print(f"Loading cache at {cache_path}")
        
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    return cache
    
def get_latent(wm, thetas, imgs, heat_imgs, no_heat_imgs, heat_bool=True): 
    # TODO 1: turn 'imgs' + 'heat' into a dictionary called 'obs'
    # TODO 2: ^ heat_on: (imgs + thermal) + heat_off: (imgs + thermal)
    thetas = np.expand_dims(np.expand_dims(thetas,1),1)
    imgs = np.expand_dims(imgs, 1)
    heat_imgs = heat_imgs if heat_bool else no_heat_imgs
    heat_imgs = np.expand_dims(heat_imgs, 1)
    
    # print(f"[ddpg-wm/get_latent] shapes: {imgs.shape}, {heat_imgs.shape}") # shapes: (1681, 1, 128, 128, 3), (1681, 1, 128, 128, 1)
    dummy_acs = np.zeros((np.shape(thetas)[0], 1))
    firsts = np.ones((np.shape(thetas)[0], 1))
    lasts = np.zeros((np.shape(thetas)[0], 1))
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    states = np.concatenate([cos, sin], axis=-1)
    chunks = 21
    if np.shape(imgs)[0] > chunks:
      bs = int(np.shape(imgs)[0]/chunks)
    else:
      bs = int(np.shape(imgs)[0]/chunks)
    for i in range(chunks):
      if i == chunks-1:
        data = {'obs_state': states[i*bs:], 'image': imgs[i*bs:], 'heat': heat_imgs[i*bs:], 'action': dummy_acs[i*bs:], 'is_first': firsts[i*bs:], 'is_terminal': lasts[i*bs:]}
      else:
        data = {'obs_state': states[i*bs:(i+1)*bs], 'image': imgs[i*bs:(i+1)*bs], 'heat': heat_imgs[i*bs:(i+1)*bs], 'action': dummy_acs[i*bs:(i+1)*bs], 'is_first': firsts[i*bs:(i+1)*bs], 'is_terminal': lasts[i*bs:(i+1)*bs]}
      
    #   print("[ddpg-wm/get_latent] input batch shapes:")
    #   for k, v in data.items():
    #     print(f"  {k}: {np.shape(v)}")
    #   quit()
      
      data = wm.preprocess(data)
      embeds = wm.encoder(data)
      if i == 0:
        embed = embeds
      else:
        embed = torch.cat([embed, embeds], dim=0)

    data = {'obs_state': states, 'image': imgs, 'heat': heat_imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}
    data = wm.preprocess(data)
    post, _ = wm.dynamics.observe(
        embed, data["action"], data["is_first"]
        )
    
    feat = wm.dynamics.get_feat(post).detach()
    lz = torch.tanh(wm.heads["margin"](feat))
    return feat.squeeze().cpu().numpy(), lz.squeeze().detach().cpu().numpy(), post

def evaluate_V(state):
    tmp_obs = np.array(state)#.reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy.critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()

@torch.no_grad()
def rollout_dubins(lz, feat, post, states, policy, T=100,
                   rollout_batch_size=100, heat=True):
    post = {k: v.clone() for k, v in post.items()}
    # Make sure the shape matches what wm.dynamics.get_feat expects
    for k in post:
        if post[k].ndim == 3 and post[k].shape[1] == 1:
            post[k] = post[k].squeeze(1)

    vf_binary = lz > 0
    results = dict(TP=0, TN=0, FP=0, FN=0)

    N = states.shape[0]
    for start in range(0, N, rollout_batch_size):
        end = min(start + rollout_batch_size, N)

        post_b = {k: v[start:end] for k, v in post.items()}
        feat_b = torch.tensor(feat[start:end], dtype=torch.float32).to(config.device).clone()          # current actor input
        states_b = states[start:end].clone()      # x, y, theta
        x, y, theta = states_b.t()
        unsafe = torch.zeros_like(x, dtype=torch.bool)

        for _ in range(T):
            # 1. closed-loop action
            act = policy.actor(feat_b)[0]         # (B, act_dim)

            # 2. imagine next latent
            post_b = wm.dynamics.img_step(post_b, act)
            feat_b = wm.dynamics.get_feat(post_b).detach()

            # 3. geometric update for safety check
            if heat:
                x += config.speed * torch.cos(theta) * config.dt
                y += config.speed * torch.sin(theta) * config.dt
                theta = theta + act[:, 0] * config.dt
                theta = (theta + np.pi) % (2 * np.pi) - np.pi
                unsafe |= ((x - config.obs_x) ** 2 +
                           (y - config.obs_y) ** 2).sqrt() < config.obs_r

        # confusion-matrix update
        for i in range(end - start):
            is_unsafe = unsafe[i].item()
            pred_safe = vf_binary[start + i].item()
            if not is_unsafe and pred_safe:
                results["TP"] += 1
            elif is_unsafe and not pred_safe:
                results["TN"] += 1
            elif is_unsafe and pred_safe:
                results["FP"] += 1
            else:
                results["FN"] += 1
    return results


def get_eval_plot(cache, thetas, heat_values, use_rollout_eval=True):
    theta_heat_pairs = list(product(thetas, heat_values))

    fig1, axes1 = plt.subplots(2, len(theta_heat_pairs), figsize=(3 * len(theta_heat_pairs), 6))
    fig2, axes2 = plt.subplots(2, len(theta_heat_pairs), figsize=(3 * len(theta_heat_pairs), 6))

    ground_truth = np.load(f"{config.ground_truth_path}_{config.nx}.npz")
    x = np.linspace(config.x_min, config.x_max, config.nx)
    y = np.linspace(config.y_min, config.y_max, config.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i, (theta, heat_value) in enumerate(theta_heat_pairs):
        idxs, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, thetas_prev, states = cache[(theta, heat_value)]
        states = torch.stack(states).float().to(config.device)

        pred_data = {}
        for heat_bool, label in [(True, 'heat'), (False, 'no_heat')]:
            feat, lz, post = get_latent(wm, thetas_prev, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, heat_bool)
            
            # feat1, lz1, post1 = get_latent(wm, thetas_prev, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, heat_bool)
            # feat2, lz2, post2 = get_latent(wm, thetas_prev, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, heat_bool)

            # # Compare:
            # def check_diff(a, b, name):
            #     if isinstance(a, torch.Tensor):
            #         if not torch.allclose(a, b, rtol=1e-5, atol=1e-7):
            #             print(f"[WARNING] get_latent is non-deterministic in {name}")
            #     elif isinstance(a, dict):
            #         for k in a:
            #             check_diff(a[k], b[k], f"{name}.{k}")

            # check_diff(feat1, feat2, "feat")
            # check_diff(lz1, lz2, "lz")
            # check_diff(post1, post2, "post")

            vals = evaluate_V(feat)
            combined = np.minimum(vals, lz)
            pred_data[label] = {'feat': feat, 'lz': lz, 'post': post, 'combined': combined}

        for row, key in enumerate(['heat', 'no_heat']):
            binary = pred_data[key]['combined'].reshape(config.nx, config.ny).T > 0
            continuous = pred_data[key]['combined'].reshape(config.nx, config.ny).T

            axes1[row, i].imshow(binary, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1)
            axes2[row, i].imshow(continuous, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1)

            # Plot GT only on 'heat' row
            if row == 0:
                gt_key = f"theta_{theta:.4f}_{heat_value:.4f}_rad"
                if gt_key in ground_truth:
                    gt_slice = ground_truth[gt_key]
                    axes1[row, i].contour(X, Y, gt_slice, levels=[0], colors='black', linewidths=1.5)
                    axes2[row, i].contour(X, Y, gt_slice, levels=[0], colors='black', linewidths=1.5)

            # TP/TN/FP/FN only on continuous plot
            if row == 0 and gt_key in ground_truth:
                gt_slice = ground_truth[gt_key]
                pred = pred_data[key]['combined'].reshape(config.nx, config.ny).T
                gt_flat = gt_slice.flatten()
                pred_flat = pred.flatten()

                if use_rollout_eval:
                    conf = rollout_dubins(
                        pred_data[key]['lz'], pred_data[key]['feat'],
                        pred_data[key]['post'], states, policy,
                        heat=(key == 'heat')
                    )
                    TP, TN, FP, FN = conf['TP'], conf['TN'], conf['FP'], conf['FN']
                else:
                    safe = np.where(gt_flat >= 0)[0]
                    unsafe = np.where(gt_flat < 0)[0]
                    TP = (pred_flat[safe] > 0).sum()
                    FN = (pred_flat[safe] <= 0).sum()
                    FP = (pred_flat[unsafe] > 0).sum()
                    TN = (pred_flat[unsafe] <= 0).sum()

                T = TP + TN + FP + FN
                text = f"TP:{TP/T:.2f}  TN:{TN/T:.2f}\nFP:{FP/T:.2f}  FN:{FN/T:.2f}"
                axes2[row, i].text(
                    0.02, 0.02, text,
                    transform=axes2[row, i].transAxes,
                    fontsize=8,
                    color='black',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
                )

        # Draw obstacle on both rows
        for ax in [axes1[:, i], axes2[:, i]]:
            for a in ax:
                a.add_patch(patches.Circle(
                    (config.obs_x, config.obs_y),
                    config.obs_r,
                    linewidth=1.5,
                    edgecolor='red',
                    facecolor='none',
                    linestyle='--'
                ))
                a.axis("off")

        # Title per subplot
        axes2[0, i].set_title(f"Theta {theta:.2f}, Heat {heat_value:.2f} (HEAT)")
        axes2[1, i].set_title(f"Theta {theta:.2f}, Heat {heat_value:.2f} (NO HEAT)")
        axes1[0, i].set_title(f"Theta {theta:.2f}, Heat {heat_value:.2f} (HEAT)")
        axes1[1, i].set_title(f"Theta {theta:.2f}, Heat {heat_value:.2f} (NO HEAT)")

    # Add row labels
    axes1[0, 0].set_ylabel("Binary (HEAT)")
    axes1[1, 0].set_ylabel("Binary (NO HEAT)")
    axes2[0, 0].set_ylabel("Continuous (HEAT)")
    axes2[1, 0].set_ylabel("Continuous (NO HEAT)")

    fig1.suptitle("Binary Reach-Avoid: min(V, g) > 0", fontsize=14)
    fig2.suptitle("Continuous min(V, g)", fontsize=14)
    fig1.tight_layout()
    fig2.tight_layout()

    return fig1, fig2

# -------------------------------------------------------------------------------------------

if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path+"/epoch_id_{}".format(epoch))

heat_values = [0.2, 0.4, 0.6, 0.8] # TODO: stick this in config
thetas = [3 * np.pi / 2, 7 * np.pi / 4, 0]

cache_path = f"{config.hj_cache_path}_{config.alpha_in}.pkl"

if not os.path.exists(cache_path):
    cache = make_cache(config, thetas, heat_values)
else:
    cache = load_cache(config)

logger = None
warmup = 1
plot1, plot2 = get_eval_plot(cache, thetas, heat_values)

eval = False
if eval:
    print(f"[EVAL] Loading trained policy from: {log_path}/epoch_id_{epoch}/policy.pth")
    path = "/home/matthew/PytorchReachability/logs/dreamer_dubins_multimodal_v3plus/PyHJ/0627/232838/PyHJ/dubins-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_128_3_a1_128_3_gamma_0.9999/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_16/policy.pth"
    policy.load_state_dict(torch.load(f"{path}", weights_only=True))

    # Evaluate on test_envs
    print("[EVAL] Running evaluation episodes...")
    test_result = test_collector.collect(n_episode=args.test_num, render=args.render_eval if hasattr(args, 'render_eval') else 0)

    avg_reward = test_result["rews"].mean()
    avg_length = test_result["lens"].mean()
    print(f"[EVAL] Average reward: {avg_reward:.2f} | Average episode length: {avg_length:.2f}")

    # Generate and show evaluation plots
    print("[EVAL] Generating evaluation plots...")
    plot1, plot2 = get_eval_plot(cache, thetas, heat_values)

    # Save plots locally
    plot_dir = os.path.join(log_path, f"epoch_id_{epoch}", "eval_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot1_path = os.path.join(plot_dir, "binary_reach_avoid_plot.png")
    plot2_path = os.path.join(plot_dir, "continuous_minVg_plot.png")
    plot1.savefig(plot1_path)
    plot2.savefig(plot2_path)
    print(f"[EVAL] Plots saved to {plot_dir}")

    # Log to wandb
    if args.use_wandb:
        wandb.log({
            "eval/binary_reach_avoid_plot": wandb.Image(plot1),
            "eval/continuous_plot": wandb.Image(plot2),
            "eval/avg_reward": avg_reward,
            "eval/avg_length": avg_length
        })

    print("[EVAL] Evaluation complete.")

else:
    for iter in range(warmup+args.total_episodes):
        if iter  < warmup:
            policy._gamma = 0 # for warming up the value fn
            policy.warmup = True
        else:
            policy._gamma = config.gamma_pyhj
            policy.warmup = False

        if args.continue_training_epoch is not None:
            print("epoch: {}, remaining epochs: {}".format(epoch//args.epoch, args.total_episodes - iter))
        else:
            print("epoch: {}, remaining epochs: {}".format(iter, args.total_episodes - iter))
        epoch = epoch + args.epoch
        print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        if args.total_episodes > 1:
            writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        else:
            if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
                print("Just created the log directory!")
                print("log_path: ", log_path+"/total_epochs_{}".format(epoch))
                os.makedirs(log_path+"/total_epochs_{}".format(epoch))
            writer = SummaryWriter(log_path+"/total_epochs_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        if logger is None:
            logger = WandbLogger()
            logger.load(writer)
        logger = TensorboardLogger(writer)
        
        # import pdb; pdb.set_trace()
        result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size_pyhj,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
        )
        
        save_best_fn(policy, epoch=epoch)
        plot1, plot2 = get_eval_plot(cache, thetas, heat_values)
        wandb.log({"binary_reach_avoid_plot": wandb.Image(plot1), "continuous_plot": wandb.Image(plot2)})

        policy.critic_scheduler.step()
        policy.actor_scheduler.step()

