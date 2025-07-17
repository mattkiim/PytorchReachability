import argparse
import os
import sys
import pickle
import cv2

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
    
    if config.obs_priv_heat:
        env.observation_space_full['obs_state'] = gymnasium.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
    else:
        env.observation_space_full['obs_state'] = gymnasium.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
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
actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=actor_optim, gamma=0.995)


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
                    if config.include_no_heat_vis:
                        no_heat = gen.get_heat_frame_v0(img, heat=False)
                elif config.heat_mode == 1:
                    heat = gen.get_heat_frame_v1(img, heat=True)
                    if config.include_no_heat_vis:
                        no_heat = gen.get_heat_frame_v1(img, heat=False)
                elif config.heat_mode == 2:
                    img = gen.get_rgb_v2(img, config, heat=True)
                    heat, _ = gen.get_heat_frame_v2(img, heat=True, heat_value=heat_value)
                    if config.include_no_heat_vis:
                        no_heat, _ = gen.get_heat_frame_v2(img, heat=False, heat_value=heat_value)
                elif config.heat_mode == 3:
                    img = gen.get_rgb_v3(img, config, heat=True, heat_value=heat_value)
                    heat, _ = gen.get_heat_frame_v3(img, heat=True, heat_value=heat_value)
                    if config.include_no_heat_vis:
                        no_heat, _ = gen.get_heat_frame_v3(img, heat=False, heat_value=heat_value) # BUG: heat=False seems to remove vehicle
                else:
                    raise ValueError(f"Unknown heat_mode: {config.heat_mode}")

                # Accumulate data
                idxs.append(idx)
                imgs_prev.append(img)
                heat_imgs.append(heat)
                if config.include_no_heat_vis:
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
    
def get_latent(wm, thetas, heat_values, imgs, heat_imgs, no_heat_imgs, heat_bool=True):
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
    heat_values = np.ones_like(cos) * heat_values # TODO: after adding to cache, fix this
    if not heat_bool: heat_values *= 0
    
    # print(heat_values.mean())
    if config.obs_priv_heat:
        states = np.concatenate([cos, sin, heat_values], axis=-1)
    else:
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
def rollout_dubins(
    lz, feat, post, states,
    heat_value_init, policy,
    T=100, rollout_batch_size=100,
    heat=True 
):
    """
    Rolls trajectories and builds a confusion matrix wrt
        combined_pred = min(l(x), V(x)) > 0.
    Returns dict(results), trajectories, failures.
    """
    # TODO: save a boundary IC trajectory (which is currently in latents), and convert to RGB + Heat images, then create video
    
    # ------------------------------------------------------------------ #
    # 1.  Prediction (static): min(l, V) > 0
    EPS           = 1e-6
    V_vals        = evaluate_V(feat)                     # (N,)
    V_vals        = V_vals.cpu().numpy() if torch.is_tensor(V_vals) else V_vals
    combined      = np.minimum(lz, V_vals)               # (N,)
    vf_binary     = combined > EPS                       # True  ⇒ predicted safe
    # ------------------------------------------------------------------ #
    # 2.  Prep latent tensors
    post = {k: v.clone() for k, v in post.items()}
    for k in post:
        if post[k].ndim == 3 and post[k].shape[1] == 1:
            post[k] = post[k].squeeze(1)

    N            = states.shape[0]
    trajectories = []          # list of (xs, ys)
    failures     = []          # list of bool
    fp_trajs = []  # list of bool
    fn_trajs = []  # list of bool
    results      = dict(TP=0, TN=0, FP=0, FN=0)

    # ------------------------------------------------------------------ #
    for start in range(0, N, rollout_batch_size):
        end     = min(start + rollout_batch_size, N)
        post_b  = {k: v[start:end] for k, v in post.items()}
        feat_b  = torch.tensor(feat[start:end], dtype=torch.float32,
                               device=config.device).clone()
        states_b = states[start:end].clone()
        x, y, theta = states_b.t()

        heat_vals = torch.full_like(x, heat_value_init)

        # -------- initial failure check BEFORE any heat decay ----------
        if heat:
            failure = heat_vals >= (config.heat_threshold - EPS)
        else:
            failure = torch.zeros_like(x, dtype=torch.bool)

        xs_all = [x.cpu().numpy()]
        ys_all = [y.cpu().numpy()]

        for _ in range(T):
            # ----------------- dynamics & control ----------------------
            act     = policy.actor(feat_b)[0]
            post_b  = wm.dynamics.img_step(post_b, act)
            feat_b  = wm.dynamics.get_feat(post_b).detach()

            x     += config.speed * torch.cos(theta) * config.dt
            y     += config.speed * torch.sin(theta) * config.dt
            theta += act[:, 0] * config.dt
            theta  = (theta + np.pi) % (2 * np.pi) - np.pi

            # ----------------- heat update (optional) ------------------
            if heat:
                dist       = ((x - config.obs_x)**2 + (y - config.obs_y)**2).sqrt()
                inside_obs = dist < config.obs_r

                heat_vals = torch.where(
                    inside_obs,
                    heat_vals + config.alpha_in  / (255 / 1.1), # TODO: add vehicle_heat to config
                    heat_vals - config.alpha_out / (255 / 1.1)
                )
                heat_vals = torch.clamp(heat_vals, min=0.0, max=1.0)

                # failure if heat ≥ threshold (equality included)
                failure |= heat_vals >= (config.heat_threshold - EPS)

            xs_all.append(x.cpu().numpy())
            ys_all.append(y.cpu().numpy())

        xs_all = np.stack(xs_all, axis=1)
        ys_all = np.stack(ys_all, axis=1)

        for i in range(xs_all.shape[0]):
            trajectories.append((xs_all[i], ys_all[i]))
            failures.append(bool(failure[i]))
            
            if bool(failure[i]) and vf_binary[start+i]:        # False-positive
                fp_trajs.append((xs_all[i], ys_all[i]))
            elif (not bool(failure[i])) and (not vf_binary[start+i]):  # False-negative
                fn_trajs.append((xs_all[i], ys_all[i]))

        # ---------------- confusion-matrix update ----------------------
        for i in range(end - start):
            is_unsafe = bool(failure[i])
            pred_safe = bool(vf_binary[start + i])

            if (not is_unsafe) and pred_safe:
                results["TP"] += 1
            elif is_unsafe and (not pred_safe):
                results["TN"] += 1
            elif is_unsafe and pred_safe:
                results["FP"] += 1
            else:
                results["FN"] += 1

    return (results,
            np.array(trajectories, dtype=object),
            np.array(failures,     dtype=bool),
            np.array(fp_trajs,     dtype=object),
            np.array(fn_trajs,     dtype=object))

@torch.no_grad()    
def single_rollout(initial_conditions, config, T=100, target=None):
    """
    1. Take the initial condition, generate RGB + heat image.
    2. Roll out trajectory using the Dreamer policy.
    3. Convert real states to RGB + Heat.
    4. Save trajectory as video (optional).
    """
    gen = HeatFrameGenerator(config)
    trajectories_rgb_obs = []
    trajectories_heat_obs = []

    for initial_condition in initial_conditions:
        state = torch.tensor(initial_condition[:3])  # (x, y, theta)
        heat_value = initial_condition[-1]
        vehicle_heat = heat_value
        traj_rgb = []
        traj_heat = []

        for t in range(T):
            # Generate RGB and heat images from current state
            img = get_frame_eval(state, config)
            gen._compute_geometry(img.shape)

            if config.heat_mode == 3:
                img = gen.get_rgb_v3(img, config, heat=True, heat_value=vehicle_heat)
                heat, _ = gen.get_heat_frame_v3(img, config, heat=True, heat_value=vehicle_heat)
            elif config.heat_mode == 2:
                img = gen.get_rgb_v2(img, config, heat=True)
                heat, _ = gen.get_heat_frame_v2(img, config, heat=True, heat_value=vehicle_heat)
            elif config.heat_mode == 1:
                heat = gen.get_heat_frame_v1(img, heat=True)
            elif config.heat_mode == 0:
                heat = gen.get_heat_frame_v0(img, heat=True)
            else:
                raise NotImplementedError("Unsupported heat_mode")

            # Store images
            img = cv2.putText(
                img,
                f"Heat: {vehicle_heat:.2f}",
                org=(5, 15),  # (x, y) position
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )
            traj_rgb.append(img)
            traj_heat.append(heat)

            # Convert to Dreamer latent
            feat, lz, post = get_latent(
                wm, [state[2]], vehicle_heat, [img], [heat], no_heat_imgs=None, heat_bool=True
            )

            # Get latent feature
            feat_tensor = torch.tensor(feat, dtype=torch.float32, device=config.device)
            dreamer_action = policy.actor(feat_tensor.unsqueeze(0))[0][0]  # shape: (1,1)

            if target is not None:
                # Current position and orientation
                x, y, theta = state[0].item(), state[1].item(), state[2].item()
                tx, ty = target

                # Compute nominal heading correction
                desired_heading = np.arctan2(ty - y, tx - x)
                heading_error = (desired_heading - theta + np.pi) % (2 * np.pi) - np.pi

                # Nominal action to correct heading
                nominal_turn = heading_error / config.dt
                nominal_turn = np.clip(nominal_turn, -config.turnRate, config.turnRate)
                nominal_action = torch.tensor([nominal_turn], dtype=torch.float32, device=config.device)

                # Evaluate safety of nominal action
                value_nominal = policy.critic(feat_tensor.unsqueeze(0), nominal_action.unsqueeze(0))[0]
                is_safe = value_nominal > 0

                action = nominal_action if is_safe else dreamer_action
            else:
                action = dreamer_action

            # Simulate Dubins dynamics
            speed = config.speed
            dt = config.dt
            turn_rate = action[0]  # assuming scalar turning
            theta = state[2]
            dx = speed * torch.cos(theta) * dt
            dy = speed * torch.sin(theta) * dt
            dtheta = turn_rate * dt

            state[0] += dx
            state[1] += dy
            state[2] = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi

            # Heat update (like rollout_dubins)
            dist = ((state[0] - config.obs_x)**2 + (state[1] - config.obs_y)**2).sqrt()
            inside_obs = dist < config.obs_r

            if inside_obs:
                vehicle_heat += config.alpha_in / (255 / 1.1)
            else:
                vehicle_heat -= config.alpha_out / (255 / 1.1)
            # vehicle_heat = torch.tensor(vehicle_heat, dtype=torch.float32)
            vehicle_heat = np.clip(vehicle_heat, 0, 1)

        trajectories_rgb_obs.append(traj_rgb)
        trajectories_heat_obs.append(traj_heat)

    return trajectories_rgb_obs, trajectories_heat_obs


def get_eval_plot(cache, thetas, heat_values, rollout_T=100, boundary_eps=1e-3):
    from itertools import product
    from matplotlib.colors import ListedColormap
    from matplotlib import colors as mcolors

    theta_heat_pairs = list(product(thetas, heat_values))
    nrows = 2 if config.include_no_heat_vis else 1
    ncols = len(theta_heat_pairs)
    figsize = (3 * ncols, 6)

    # function to make figures with subplots
    def _make_fig():
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, np.atleast_2d(ax)

    # create figures
    fig_lz, axes_lz = _make_fig()
    fig_lz_bin, axes_lz_bin = _make_fig()
    fig_v, axes_v = _make_fig()
    fig_v_bin, axes_v_bin = _make_fig()
    fig_combined, axes_combined = _make_fig()
    fig_combined_bin, axes_combined_bin = _make_fig()
    fig_rollout, axes_rollout = _make_fig()

    # colour map for binary safe/unsafe (0 unsafe = red, 1 safe = green)
    binary_cmap = ListedColormap(["#276fae", "#e6dc22"])

    # ground‑truth slice
    gt = np.load(f"{config.ground_truth_path}_{config.nx}.npz")
    x_lin = np.linspace(config.x_min, config.x_max, config.nx)
    y_lin = np.linspace(config.y_min, config.y_max, config.ny)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

    plot_list = [(True, "heat")]
    if config.include_no_heat_vis:
        plot_list.append((False, "no_heat"))

    # iterate over each (theta, heat) column
    for col, (theta, heat_value) in enumerate(theta_heat_pairs):
        idxs, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, thetas_prev, states_lst = cache[(theta, heat_value)]
        states_tensor = torch.stack(states_lst).float().to(config.device)

        # evaluate both HEAT / NO‑HEAT rows
        for row, (heat_bool, lbl) in enumerate(plot_list):
            feat, lz, post = get_latent(
                wm,
                thetas_prev,
                heat_value,
                imgs_prev,
                heat_imgs_prev,
                no_heat_imgs_prev,
                heat_bool,
            )
            vals = evaluate_V(feat)
            combined = np.minimum(vals, lz)

            # reshape for image display
            lz_img = lz.reshape(config.nx, config.ny).T
            v_img = vals.reshape(config.nx, config.ny).T
            comb_img = combined.reshape(config.nx, config.ny).T

            lz_bin = (lz_img > 0).astype(float)
            v_bin = (v_img > 0).astype(float)
            comb_bin = (comb_img > 0).astype(float)

            # continuous heatmaps
            axes_lz[row, col].imshow(lz_img, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1, cmap="seismic")
            axes_v[row, col].imshow(v_img, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1, cmap="viridis")
            axes_combined[row, col].imshow(comb_img, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1, cmap="coolwarm")

            # binary (colour) maps
            axes_lz_bin[row, col].imshow(lz_bin, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=0, vmax=1, cmap=binary_cmap)
            axes_v_bin[row, col].imshow(v_bin, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=0, vmax=1, cmap=binary_cmap)
            axes_combined_bin[row, col].imshow(comb_bin, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=0, vmax=1, cmap=binary_cmap)
            
            # trajectory rollouts
            results, trajectories, failures, fp_trajs, fn_trajs = rollout_dubins(
                lz, feat, post, states_tensor,
                heat_value_init=heat_value,
                policy=policy,
                T=rollout_T,
                heat=heat_bool
            )

            ax = axes_rollout[row, col]
            # trajectory rollout plots
            for (xs, ys), is_failure in zip(trajectories, failures):
                # build one RGBA array whose alpha increases with time
                base_rgb   = mcolors.to_rgba('red' if is_failure else 'green')
                Tpts       = xs.shape[0]
                alphas     = np.linspace(0.15, 1.0, Tpts)
                colors_rgba = np.tile(base_rgb, (Tpts, 1))
                colors_rgba[:, 3] = alphas
                ax.scatter(xs, ys, s=3, marker='o', color=colors_rgba, linewidths=0)
                
            # misclassified trajectories
            # print(fp_trajs.shape, fn_trajs.shape); quit()
            for xs, ys in fp_trajs:
                base_rgb   = mcolors.to_rgba('dodgerblue')
                Tpts       = xs.shape[0]
                alphas     = np.linspace(0.05, 0.4, Tpts)
                rgba_arr   = np.tile(base_rgb, (Tpts, 1))
                rgba_arr[:, 3] = alphas
                ax.scatter(xs, ys, s=2.5, marker='o',
                        color=rgba_arr, linewidths=0, zorder=3)

            # false-negatives (safe but predicted unsafe)
            for xs, ys in fn_trajs:
                base_rgb   = mcolors.to_rgba('magenta')
                Tpts       = xs.shape[0]
                alphas     = np.linspace(0.05, 0.4, Tpts)
                rgba_arr   = np.tile(base_rgb, (Tpts, 1))
                rgba_arr[:, 3] = alphas
                ax.scatter(xs, ys, s=2.5, marker='o',
                        color=rgba_arr, linewidths=0, zorder=3)
                
            # overlay the BRT in black
            key = f"theta_{theta:.4f}_{heat_value:.4f}_rad"
            if key in gt:
                gt_slice = gt[key]
                ax.contour(
                    X, Y, gt_slice,
                    levels=[0],
                    colors="black",
                    linewidths=1.0,
                    zorder=4
                )
    
            # legend
            ax.plot([], [], color='dodgerblue',  linewidth=1.0, label='FP')
            ax.plot([], [], color='magenta',  linewidth=1.0, label='FN')
            ax.plot([], [], color='green',  linewidth=1.0, label='Safe')
            ax.plot([], [], color='red',  linewidth=1.0, label='Unsafe')
            ax.legend(loc='lower right', fontsize=6, framealpha=0.6)

            # obstacle
            ax.add_patch(patches.Circle((config.obs_x, config.obs_y),
                                        config.obs_r, edgecolor='black',
                                        facecolor='none', linestyle='--', linewidth=1.5))
            ax.set_xlim(config.x_min, config.x_max)
            ax.set_ylim(config.y_min, config.y_max)
            ax.set_aspect('equal')
            ax.set_title(f"Rollouts Θ={theta:.2f} H={heat_value:.2f}", fontsize=8)
            ax.axis("off")
            
            # rollout-based confusion matrix text
            total = sum(results.values())
            if total > 0:
                txt = (
                    f"TP: {results['TP']/total:.2f}  TN: {results['TN']/total:.2f}\n"
                    f"FP: {results['FP']/total:.2f}  FN: {results['FN']/total:.2f}"
                )
                ax.text(
                    0.02, 0.02, txt,
                    transform=ax.transAxes,
                    fontsize=7,
                    color="black",
                    verticalalignment="bottom",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
                )

            # title
            title = f"Θ={theta:.2f}  H={heat_value:.2f}  ({lbl.upper()})"
            for ax_group in [axes_lz, axes_lz_bin, axes_v, axes_v_bin, axes_combined, axes_combined_bin]:
                ax_group[row, col].set_title(title, fontsize=8)

            # ground‑truth overlay (row 0 only)
            if row == 0:
                key = f"theta_{theta:.4f}_{heat_value:.4f}_rad"
                if key in gt:
                    gt_slice = gt[key]
                    for ax_group in [axes_lz, axes_v, axes_combined, axes_lz_bin, axes_v_bin, axes_combined_bin]:
                        ax_group[row, col].contour(X, Y, gt_slice, levels=[0], colors="black", linewidths=1.0)
                        
                    # confusion matrix on continuous min(V,l)
                    gt_flat   = gt_slice.flatten()
                    pred_flat = comb_img.flatten()
                    EPS = 1e-6
                    safe_idx   = gt_flat >= EPS
                    unsafe_idx = gt_flat < EPS
                    TP = np.logical_and(pred_flat > 0, safe_idx  ).sum()
                    FN = np.logical_and(pred_flat <=0, safe_idx  ).sum()
                    FP = np.logical_and(pred_flat > 0, unsafe_idx).sum()
                    TN = np.logical_and(pred_flat <=0, unsafe_idx).sum()
                    total = TP + TN + FP + FN
                    if total > 0:
                        txt = f"TP:{TP/total:.2f}  TN:{TN/total:.2f}\nFP:{FP/total:.2f}  FN:{FN/total:.2f}"
                        axes_combined[row, col].text(
                            0.02, 0.02, txt,
                            transform=axes_combined[row, col].transAxes,
                            fontsize=7,
                            color="black",
                            verticalalignment="bottom",
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
                        )
                        
            # obstacle
            for ax_group in [axes_lz, axes_lz_bin, axes_v, axes_v_bin, axes_combined, axes_combined_bin]:
                ax = ax_group[row, col]
                ax.add_patch(patches.Circle((config.obs_x, config.obs_y), config.obs_r, linewidth=1, edgecolor="red", facecolor="none", linestyle="--"))
                ax.axis("off")
                
            # ------------------------------------------------------------------ #
            initial_state = np.array([-0.9, 0., 0.])
            # print(initial_state); quit()
            traj_rgb, traj_heat = single_rollout([initial_state], config, T=rollout_T, target=[0.9, 0.])

            import imageio
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_vid:
                imageio.mimsave(temp_vid.name, traj_rgb[0], fps=8)
                temp_vid.flush()
                traj_vid_path = temp_vid.name

    # y‑labels on leftmost column
    labels = [
        (axes_lz, "l(x)"),
        (axes_lz_bin, "Binary l(x)"),
        (axes_v, "V(x)"),
        (axes_v_bin, "Binary V(x)"),
        (axes_combined, "min(V,l)"),
        (axes_combined_bin, "Binary min(V,l)"),
        (axes_rollout, "Trajectories")
    ]
    for ax_arr, base_lbl in labels:
        ax_arr[0, 0].set_ylabel(f"{base_lbl} (HEAT)")
        if config.include_no_heat_vis:
            ax_arr[1, 0].set_ylabel(f"{base_lbl} (NO HEAT)")

    # figure‑level titles & layout
    fig_lz.suptitle("Safety Margin l(x)", fontsize=14)
    fig_lz_bin.suptitle("Binary l(x) > 0", fontsize=14)
    fig_v.suptitle("Critic Value V(x)", fontsize=14)
    fig_v_bin.suptitle("Binary V(x) > 0", fontsize=14)
    fig_combined.suptitle("min(V(x), l(x))", fontsize=14)
    fig_combined_bin.suptitle("Binary min(V(x), l(x)) > 0", fontsize=14)
    fig_rollout.suptitle("Trajectories", fontsize=14)

    for fig in [
        fig_lz, fig_lz_bin, fig_v, fig_v_bin, fig_combined, fig_combined_bin, fig_rollout
    ]:
        fig.tight_layout()

    return (
        fig_lz,
        fig_lz_bin,
        fig_v,
        fig_v_bin,
        fig_combined,
        fig_combined_bin,
        fig_rollout,
        traj_vid_path
    )


if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path+"/epoch_id_{}".format(epoch))

heat_values = [0.2, 0.4, 0.6, 0.8] # TODO: stick this in config, and generate ground truths in this script
# thetas = [3 * np.pi / 2, 7 * np.pi / 4, 0]
thetas = [3 * np.pi / 2, 0]

cache_path = f"{config.hj_cache_path}_{config.alpha_in}.pkl"

if not os.path.exists(cache_path):
    cache = make_cache(config, thetas, heat_values)
else:
    cache = load_cache(config)

logger = None
warmup = 1
plot1, plot2, plot3, plot4, plot5, plot6, plot7, traj_vid_path = get_eval_plot(cache, thetas, heat_values)

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
    plot1, plot2, plot3, plot4, plot5, plot6, plot7, traj_vid_path = get_eval_plot(cache, thetas, heat_values)
    log_dict = {
        "eval/lz_continuous": wandb.Image(plot1),
        "eval/lz_binary": wandb.Image(plot2),
        "eval/v_continuous": wandb.Image(plot3),
        "eval/v_binary": wandb.Image(plot4),
        "eval/min_v_l_continuous": wandb.Image(plot5),
        "eval/min_v_l_binary": wandb.Image(plot6),
        "eval/rollout_trajectories": wandb.Image(plot7),
    }
    if traj_vid_path and os.path.exists(traj_vid_path):
        log_dict["eval/rollout_video"] = wandb.Video(traj_vid_path, fps=8, format="mp4")

    wandb.log(log_dict)
    
    if traj_vid_path and os.path.exists(traj_vid_path):
        os.remove(traj_vid_path)
    
    policy.critic_scheduler.step()
    policy.actor_scheduler.step()

