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


bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi], [0, 1]])
low = bounds[:, 0]
high = bounds[:, 1]
midpoint = (low + high) / 2.0
interval = high - low
gt_observation_space = gymnasium.spaces.Box(
    np.float32(midpoint - interval/2),
    np.float32(midpoint + interval/2),
)

# action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
# action_space = action_space
# config.num_actions = action_space.shape[0]

env = gymnasium.make(args.task, params=[config])
config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]

image_size = config.size[0]

env.observation_space_full['image'] = gymnasium.spaces.Box(
    low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
)

if config.obs_priv_heat:
    env.observation_space_full['obs_state'] = gymnasium.spaces.Box(
        low=-1, high=1, shape=(9,), dtype=np.float32
    )
else:
    env.observation_space_full['obs_state'] = gymnasium.spaces.Box(
        low=-1, high=1, shape=(8,), dtype=np.float32
    )

env.observation_space_full['heat'] = gymnasium.spaces.Box(
    low=0, high=255, shape=(image_size, image_size, 1), dtype=np.uint8
)

wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)
# quit()

ckpt_path = config.rssm_ckpt_path
checkpoint = torch.load(ckpt_path, weights_only=True)
state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}

wm.load_state_dict(state_dict)
wm.eval()

offline_eps = collections.OrderedDict()
config.batch_size = 1
config.batch_length = 2

config.dataset_path = f"{config.dataset_path}"
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
    os.makedirs(log_path+"/epoch_id_{}".format(epoch))

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
    
    # lz_image = lz.reshape(41, 41).T

    # plt.imshow(lz_image, origin="lower", cmap="seismic", vmin=-1, vmax=1)
    # plt.colorbar(label="lz (safety margin)")
    # plt.title("Safety Margin Output (lz)")
    # plt.savefig("lz_margin_plot.png", dpi=300); quit() # TODO: visualize this tomorrow
    
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
                    heat_vals + config.alpha_in  / (255 / 1.1),
                    heat_vals - config.alpha_out / (255 / 1.1)
                )

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


if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path+"/epoch_id_{}".format(epoch))

logger = None
warmup = 1

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
    # wandb.log({
    #     "eval/lz_continuous": wandb.Image(plot1),
    #     "eval/lz_binary": wandb.Image(plot2),
    #     "eval/v_continuous": wandb.Image(plot3),
    #     "eval/v_binary": wandb.Image(plot4),
    #     "eval/min_v_l_continuous": wandb.Image(plot5),
    #     "eval/min_v_l_binary": wandb.Image(plot6),
    #     "eval/rollout_trajectories": wandb.Image(plot7),
    # })


    policy.critic_scheduler.step()
    policy.actor_scheduler.step()

