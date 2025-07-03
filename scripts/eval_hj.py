import os
import sys
import torch
import collections
import numpy as np
import gymnasium
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from datetime import datetime
from termcolor import cprint


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/PyHJ'))
sys.path.append(saferl_dir)


from dreamer import make_dataset
import models
import tools
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.exploration import GaussianNoise
from PyHJ.data import Batch
from generate_data_traj_cont import get_frame_eval, HeatFrameGenerator
import ruamel.yaml as yaml
import argparse

import pathlib
from pathlib import Path


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


def evaluate_V(policy, state):
    tmp_obs = np.array(state)#.reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy.critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()


def make_cache(config, thetas, wm):
    nx, ny = config.nx, config.ny
    cache = {}
    for theta in thetas:
        v = np.zeros((nx, ny))
        xs = np.linspace(-1.5, 1.5, nx)
        ys = np.linspace(-1.5, 1.5, ny)
        xs_prev = xs - config.dt * config.speed * np.cos(theta)
        ys_prev = ys - config.dt * config.speed * np.sin(theta)

        idxs, imgs_prev, thetas_prev, heat_imgs, no_heat_imgs = [], [], [], [], []

        for i, x in enumerate(xs_prev):
            for j, y in enumerate(ys_prev):
                idxs.append((i, j))
                thetas_prev.append(theta)
                img = get_frame_eval(torch.tensor([x, y, theta]), config)
                gen = HeatFrameGenerator(config)

                if config.heat_mode == 0:
                    heat = gen.get_heat_frame_v0(img, heat=True)
                    no_heat = gen.get_heat_frame_v0(img, heat=False)
                else:
                    heat = gen.get_heat_frame_v2(img, heat=True)
                    no_heat = gen.get_heat_frame_v2(img, heat=False)

                imgs_prev.append(img)
                heat_imgs.append(heat)
                no_heat_imgs.append(no_heat)

        cache[theta] = [
            np.array(idxs),
            np.array(imgs_prev),
            np.array(heat_imgs),
            np.array(no_heat_imgs),
            np.array(thetas_prev)
        ]
    return cache


def get_latent(wm, thetas, imgs, heat_imgs, no_heat_imgs, heat_bool=True):
    imgs = np.expand_dims(imgs, 1)
    heat_imgs = np.expand_dims(heat_imgs if heat_bool else no_heat_imgs, 1)
    thetas = np.expand_dims(np.expand_dims(thetas, 1), 1)

    dummy_acs = np.zeros((thetas.shape[0], 1))
    firsts = np.ones((thetas.shape[0], 1))
    lasts = np.zeros((thetas.shape[0], 1))
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    states = np.concatenate([cos, sin], axis=-1)

    embed = []
    chunks = 21
    bs = max(1, imgs.shape[0] // chunks)
    for i in range(chunks):
        start, end = i * bs, None if i == chunks - 1 else (i + 1) * bs
        data = {
            'obs_state': states[start:end],
            'image': imgs[start:end],
            'heat': heat_imgs[start:end],
            'action': dummy_acs[start:end],
            'is_first': firsts[start:end],
            'is_terminal': lasts[start:end],
        }
        data = wm.preprocess(data)
        embeds = wm.encoder(data)
        embed.append(embeds)
    embed = torch.cat(embed, dim=0)

    data = {
        'obs_state': states,
        'image': imgs,
        'heat': heat_imgs,
        'action': dummy_acs,
        'is_first': firsts,
        'is_terminal': lasts,
    }
    data = wm.preprocess(data)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post).detach()
    lz = torch.tanh(wm.heads["margin"](feat))
    return feat.squeeze().cpu().numpy(), lz.squeeze().detach().cpu().numpy()


def get_eval_plot(cache, thetas, wm, config, policy):
    fig1, axes1 = plt.subplots(2, len(thetas), figsize=(3 * len(thetas), 6)) # binary map
    fig2, axes2 = plt.subplots(2, len(thetas), figsize=(3 * len(thetas), 6)) # continuous
    ground_truth = np.load("gt_dubins/value_function_theta_slices_test.npz") # load ground truth
    
    for i, theta in enumerate(thetas):
        idxs, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, thetas_prev = cache[theta]

        # get latent and value predictions
        feat_heat, lz_heat = get_latent(wm, thetas_prev, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, heat_bool=True)
        vals_heat = evaluate_V(policy, feat_heat)
        combined_heat = np.minimum(vals_heat, lz_heat)

        feat_no_heat, lz_no_heat = get_latent(wm, thetas_prev, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, heat_bool=False)
        vals_no_heat = evaluate_V(policy, feat_no_heat)
        combined_no_heat = np.minimum(vals_no_heat, lz_no_heat)

        # binary plots
        axes1[0, i].imshow(combined_heat.reshape(config.nx, config.ny).T > 0, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1)
        axes1[1, i].imshow(combined_no_heat.reshape(config.nx, config.ny).T > 0, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1)

        # continuous plots
        axes2[0, i].imshow(combined_heat.reshape(config.nx, config.ny).T, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1)
        axes2[1, i].imshow(combined_no_heat.reshape(config.nx, config.ny).T, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1)

        # overlay true unsafe region (red circle)
        for ax in [axes1[0, i], axes1[1, i], axes2[0, i], axes2[1, i]]:
            circle = patches.Circle(
                (config.obs_x, config.obs_y),
                config.obs_r,
                linewidth=1.5,
                edgecolor='red',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(circle)
            ax.axis("off")
            
            gt_key = f"theta_{theta:.4f}_rad"
            if gt_key in ground_truth:
                gt_slice = ground_truth[gt_key]
                # print(np.min(gt_slice)); quit()
                
                # Physical coordinate grid for contours
                x = np.linspace(config.x_min, config.x_max, config.nx)
                y = np.linspace(config.y_min, config.y_max, config.ny)
                X, Y = np.meshgrid(x, y, indexing="ij")

                # Plot contour on all subplots
                for ax in [axes1[0, i], axes1[1, i], axes2[0, i], axes2[1, i]]:
                    ax.contour(
                        X, Y, gt_slice,
                        levels=[0],
                        colors='black',
                        linewidths=1.5,
                        linestyles='-'
                    )

                # heat confusion matrix
                pred_heat = combined_heat.reshape(config.nx, config.ny).T
                gt_slice = gt_slice

                # Flatten arrays for indexed comparison
                pred_flat = pred_heat.flatten()
                gt_flat = gt_slice.flatten()

                # Safe and unsafe indices from ground truth
                safe_idxs = np.where(gt_flat >= 0)[0]
                unsafe_idxs = np.where(gt_flat < 0)[0]

                # Classification metric indices
                tp = np.where(pred_flat[safe_idxs] > 0)[0]
                fn = np.where(pred_flat[safe_idxs] <= 0)[0]
                fp = np.where(pred_flat[unsafe_idxs] > 0)[0]
                tn = np.where(pred_flat[unsafe_idxs] <= 0)[0]

                # Counts (optional)
                TP = len(tp)
                FN = len(fn)
                FP = len(fp)
                TN = len(tn)
                T = TP + FN + FP + TN
                
                # print(TP, TN, FP, FN)
                text = f"TP:{TP/T:.2f}  TN:{TN/T:.2f}\nFP:{FP/T:.2f}  FN:{FN/T:.2f}"
                axes2[0, i].text(
                    0.02, 0.02, text,
                    transform=axes2[0, i].transAxes,
                    fontsize=8,
                    color='black',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
                )

                # no-heat confusion matrix
                pred_no_heat = combined_no_heat.reshape(config.nx, config.ny)
                pred_reach_no_heat = pred_no_heat > 0
                gt_reach = gt_slice == gt_slice

                TP = np.logical_and(pred_reach_no_heat, gt_reach).sum()
                TN = np.logical_and(~pred_reach_no_heat, ~gt_reach).sum()
                FP = np.logical_and(pred_reach_no_heat, ~gt_reach).sum()
                FN = np.logical_and(~pred_reach_no_heat, gt_reach).sum()

                text = f"TP:{TP/T:.2f}  TN:{TN/T:.2f}\nFP:{FP/T:.2f}  FN:{FN/T:.2f}"
                axes2[1, i].text(
                    0.02, 0.02, text,
                    transform=axes2[1, i].transAxes,
                    fontsize=8,
                    color='black',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
                )

            # Add contour at level 0 to show boundary of the set
            x = np.linspace(config.x_min, config.x_max, config.nx)
            y = np.linspace(config.y_min, config.y_max, config.ny)
            X, Y = np.meshgrid(x, y, indexing="ij")

            for ax in [axes1[0, i], axes1[1, i], axes2[0, i], axes2[1, i]]:
                ax.contour(
                    X, Y, gt_slice,  # use physical coordinates
                    levels=[0],
                    colors='black',
                    linewidths=1.5,
                    linestyles='-'
                )

        axes1[0, i].set_title(f"Theta {theta:.2f} (HEAT)")
        axes1[1, i].set_title(f"Theta {theta:.2f} (NO HEAT)")

    fig1.suptitle("Binary Reach-Avoid: min(V, g) > 0", fontsize=14)
    fig2.suptitle("Continuous min(V, g)", fontsize=14)
    fig1.tight_layout()
    fig2.tight_layout()

    return fig1, fig2


def main():
    args = get_args()

    env = gymnasium.make(args.task, params=[args])
    args.num_actions = env.action_space.shape[0]
    
    if args.multimodal:
        env.observation_space_full['image'] = gymnasium.spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )
        env.observation_space_full['heat'] = gymnasium.spaces.Box(
            low=0, high=255, shape=(128, 128, 1), dtype=np.uint8
        )

    wm = models.WorldModel(env.observation_space_full, env.action_space, 0, args)
    
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
    
    log_path = log_path+'/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
            args.exploration_noise, 
            args.actor_lr, 
            args.critic_lr, 
            args.batch_size_pyhj,
            args.step_per_epoch,
            args.kwargs,
            args.seed
        )
    
    epoch=16
    
    model_path = "/home/matthew/PytorchReachability/logs/dreamer_dubins_multimodal_v2plus/PyHJ/0627/184105/PyHJ/dubins-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_128_3_a1_128_3_gamma_0.9999/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_16/policy.pth"
    
    # model_path = os.path.join(
    #         log_path+"/epoch_id_{}".format(epoch),
    #         "policy.pth"
    #     )
    
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

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    
    ckpt_path = args.rssm_ckpt_path
    checkpoint = torch.load(ckpt_path, weights_only=True)
    state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    wm.eval()

    offline_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(args, offline_eps)
    dataset = make_dataset(offline_eps, args)
    env.set_wm(wm, dataset, args)
    
    # DDPG
    
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

    policy_dict = torch.load(model_path, weights_only=True)
    
    load_info = policy.load_state_dict(torch.load(model_path), strict=True)
    print("MISSING :", len(load_info.missing_keys), load_info.missing_keys[:5])
    print("UNEXP.  :", len(load_info.unexpected_keys), load_info.unexpected_keys[:5])
    quit()

    
    policy.load_state_dict(policy_dict)
    policy.eval()

    thetas = [3*np.pi/2, 7*np.pi/4, 0]
    cache = make_cache(args, thetas, wm)
    fig1, fig2 = get_eval_plot(cache, thetas, wm, args, policy)
    fig1.savefig("eval/continuous.png")
    fig2.savefig("eval/binary.png")


if __name__ == "__main__":
    main()


# /home/matthew/PytorchReachability/logs/dreamer_dubins_multimodal_v3plus/PyHJ/0625/022609/PyHJ/dubins-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_128_3_a1_128_3_gamma_0.9999/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_16/policy.pth