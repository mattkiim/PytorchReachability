import argparse
import functools
import os
import pathlib
import sys
import pickle

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools

import torch
from torch import nn
import collections

from tqdm import trange
from termcolor import cprint
import matplotlib.pyplot as plt
import gym
from io import BytesIO
from PIL import Image

to_np = lambda x: x.detach().cpu().numpy()
from generate_data_traj_cont import get_frame_eval, get_frame_eval_pil, HeatFrameGenerator

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        # print(f"[Dreamer]: {obs_space}, {act_space}"); quit()
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        self._make_pretrain_opt()
        
        if self._config.fill_cache:
            cache_path = f"{self._config.wm_cache_path}_{self._config.alpha_in}.pkl"
            if os.path.exists(cache_path):
                self.load_cache() 
            else:
                print(f"No cache file found at {cache_path}; filling cache...")
                self.fill_cache()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _make_pretrain_opt(self):
        config = self._config
        use_amp = True if config.precision == 16 else False
        if (
            config.rssm_train_steps > 0
            or config.from_ckpt is not None
        ):
            # have separate lrs/eps/clips for actor and model
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            standard_kwargs = {
                "lr": config.model_lr,
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
                "opt": config.opt,
                "use_amp": use_amp,
            }
            model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
            }
            model_params["params"] += list(self._wm.heads["decoder"].parameters())
            actor_params = {
                "params": list(self._task_behavior.actor.parameters()),
                "lr": config.actor["lr"],
                "eps": config.actor["eps"],
                "clip": config.actor["grad_clip"],
            }
            self.pretrain_params = list(model_params["params"]) + list(
                actor_params["params"]
            )
            self.pretrain_opt = tools.Optimizer(
                "pretrain_opt", [model_params, actor_params], **standard_kwargs
            )
            self.actor_params = list(self._task_behavior.actor.parameters())
            
            print(
                f"Optimizer pretrain has {sum(param.numel() for param in self.pretrain_params)} variables."
            )

    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _maybe_log_metrics(self, video_pred_log=False):
        if self._logger is not None:
            logged = False
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    if not np.isnan(np.mean(values)):
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                logged = True

            if video_pred_log and self._should_log_video(self._step):
                # print(f"[Dreamer/_maybe_log_metrics]: step: {self._step}")
                video_pred, video_pred2 = self._wm.video_pred(next(self._dataset))
                self._logger.video("train_openl_agent", to_np(video_pred))
                self._logger.video("train_openl_hand", to_np(video_pred2))
                logged = True

            if logged:
                self._logger.write(fps=True)

    def pretrain_model_only(self, data, step=None):
        metrics = {}
        wm = self._wm
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        
        with tools.RequiresGrad(wm), tools.RequiresGrad(actor):
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                embed = wm.encoder(data)
                # post: z_t, prior: \hat{z}_t
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                if (step <= self._config.rssm_train_steps):
                    preds = {}
                    for name, head in wm.heads.items():
                        if name != "margin":
                            grad_head = name in self._config.grad_heads
                            feat = wm.dynamics.get_feat(post)
                            feat = feat if grad_head else feat.detach()
                            pred = head(feat)
                            if type(pred) is dict:
                                preds.update(pred)
                            else:
                                preds[name] = pred
                    # preds is dictionary of all MLP+CNN keys
                    for name, pred in preds.items():
                        if name == "cont":
                            cont_loss = -pred.log_prob(data[name])
                        # elif name == "vehicle_presence":
                        #     # Ground truth vehicle mask: (B, T, H, W, 1)
                        #     heat = data["heat"]  # assumed normalized [0,1] shape (B, T, H, W, 1)
                        #     vehicle_hi = (255 / 1.1) / 255
                        #     obstacle_px = (255 / 2 + 0.001) / 255.0  # px where vehicle is not present
                        #     vehicle_mask = ((heat <= vehicle_hi) & (heat != obstacle_px)).float()

                        #     pred_mask = pred.mode().values.unsqueeze(-1)
                        #     bce_loss = nn.functional.binary_cross_entropy(pred_mask, vehicle_mask, reduction="none")
                        #     bce_loss = bce_loss.mean(dim=(2, 3, 4))  # avg over H, W, C → shape (B, T)
                        #     losses["vehicle_presence"] = bce_loss
                            
                        elif name != "margin":
                            loss = -pred.log_prob(data[name])
                            assert loss.shape == embed.shape[:2], (name, loss.shape)
                            losses[name] = loss

                    recon_loss = sum(losses.values())
                    # failure margin
                    # vis_failure_data = data["vis_failure"]
                    failure_data = data["failure"] # 1 for unsafe, 0 for safe
                    
                    # print(vis_failure_data.shape, heat_failure_data.shape); quit()
                    safe_mask = failure_data == 0
                    unsafe_mask = ~safe_mask

                    safe_data = torch.where(safe_mask)
                    unsafe_data = torch.where(unsafe_mask)
                    # print(unsafe_data); quit()
                    
                    safe_dataset = feat[safe_data]
                    unsafe_dataset = feat[unsafe_data]
                    pos = wm.heads["margin"](safe_dataset)
                    neg = wm.heads["margin"](unsafe_dataset)
                    
                    gamma = self._config.gamma_lx
                    lx_loss = 0.0
                    if pos.numel() > 0:
                        lx_loss += torch.relu(gamma - pos).mean()
                    if neg.numel() > 0:
                        lx_loss += torch.relu(gamma + neg).mean() # weight this to scale lx_loss 

                    lx_loss *=  self._config.margin_head["loss_scale"]
                    if step < 3000:
                        lx_loss *= 0
                        cont_loss *= 0
            

                model_loss = kl_loss + recon_loss + lx_loss + cont_loss
                metrics = self.pretrain_opt(
                    torch.mean(model_loss), self.pretrain_params
                )
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))
        metrics["lx_loss"] = to_np(lx_loss)
        metrics["cont_loss"] = to_np(cont_loss)

        with torch.amp.autocast("cuda", enabled=wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
        metrics = {
            f"model_only_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics()
        self._step += 1
        # print(f"[Dreamer/pretrain_model_only]: step: {self._step}")
        self._logger.step = self._step
        
    def pretrain_regress_obs(self, data, obs_mlp, obs_opt, eval=False):
        wm = self._wm
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        if eval:
            obs_mlp.eval()
        with tools.RequiresGrad(obs_mlp):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = self._wm.encoder(data)
                post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])

                feat = self._wm.dynamics.get_feat(prior).detach() # want the imagined prior to be strong
                target = torch.Tensor(data["privileged_state"]).to(self._config.device)
                pred_state = obs_mlp(feat)
                obs_loss = torch.mean((pred_state - target) ** 2)
            if not eval:
                obs_opt(torch.mean(obs_loss), obs_mlp.parameters())
            else:
                obs_mlp.train()
        return obs_loss.item()
    
    def load_cache(self):
        cache_path = self._config.wm_cache_path
        cache_path = f"{cache_path}_{self._config.alpha_in}.pkl"

        if not os.path.exists(cache_path):
            print(f"No cache file found at {cache_path}")
            return False

        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        self.idxs = data["idxs"]
        self.safe_idxs = data["safe_idxs"]
        self.unsafe_idxs = data["unsafe_idxs"]
        self.theta_lin = data["theta_lin"]
        self.imgs = data["imgs"]
        self.heat_imgs = data["heat"]
        self.no_heat_imgs = data["no_heat"]
        self.v = np.zeros((self._config.nx, self._config.ny, 3))
        self.nz = 3  # assuming fixed

        print(f"Cache loaded from {cache_path}")
        
    def fill_cache(self, heat_values=[0.2, 0.4, 0.6, 0.8]):
        print("filling cache")
        cache_path = self._config.wm_cache_path
        if cache_path is None:
            raise NameError("No cache_path")

        nx, ny, nz = self._config.nx, self._config.ny, 3
        self.nz = nz
        self.v = np.zeros((nx, ny, nz))
        xs = np.linspace(self._config.x_min, self._config.x_max, nx)
        ys = np.linspace(self._config.y_min, self._config.y_max, ny)
        thetas = np.linspace(0, 2 * np.pi, nz, endpoint=True)

        all_rgb_imgs = {}
        all_heat_imgs = {}
        all_no_heat_imgs = {}
        imgs = []
        labels = []
        idxs = []

        it = np.nditer(self.v, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]
            theta = thetas[idx[2]]
            x -= np.cos(theta) * 0.05
            y -= np.sin(theta) * 0.05

            is_unsafe = (x**2 + y**2) < (self._config.obs_r**2)
            labels.append(int(is_unsafe))
            idxs.append(idx)

            if self._config.use_pil:
                img = get_frame_eval_pil(torch.tensor([x, y, theta]), self._config)
            else:
                img = get_frame_eval(torch.tensor([x, y, theta]), self._config)
            imgs.append(img)

            it.iternext()

        # generate all heat/no_heat images *after* image eval
        for heat_val in heat_values:
            print(f"processing heat value {heat_val}")
            rgb_imgs = []
            heat_imgs = []
            no_heat_imgs = []

            for img in imgs:
                gen = HeatFrameGenerator(self._config)
                gen._compute_geometry(img.shape)

                if self._config.heat_mode == 0:
                    heat = gen.get_heat_frame_v0(img, heat=True)
                    no_heat = gen.get_heat_frame_v0(img, heat=False)
                elif self._config.heat_mode == 1:
                    heat = gen.get_heat_frame_v1(img, heat=True)
                    no_heat = gen.get_heat_frame_v1(img, heat=False)
                elif self._config.heat_mode == 2:
                    rgb = gen.get_rgb_v2(img, self._config, heat=True)
                    heat, _ = gen.get_heat_frame_v2(rgb, heat=True, heat_value=heat_val)
                    no_heat = None
                    if self._config.include_no_heat_vis:
                        no_heat, _ = gen.get_heat_frame_v2(rgb, heat=False, heat_value=heat_val)
                elif self._config.heat_mode == 3:
                    rgb = gen.get_rgb_v3(img, self._config, heat=True, heat_value=heat_val)
                    heat, _ = gen.get_heat_frame_v3(rgb, heat=True, heat_value=heat_val)
                    no_heat = None
                    if self._config.include_no_heat_vis:
                        no_heat, _ = gen.get_heat_frame_v3(rgb, heat=False, heat_value=heat_val)
                else:
                    raise ValueError("Invalid heat_mode")

                heat_imgs.append(heat)
                rgb_imgs.append(rgb)
                if self._config.include_no_heat_vis:
                    no_heat_imgs.append(no_heat)

            all_heat_imgs[heat_val] = heat_imgs
            all_rgb_imgs[heat_val] = rgb_imgs
            if self._config.include_no_heat_vis:
                all_no_heat_imgs[heat_val] = no_heat_imgs

        idxs = np.array(idxs)
        self.idxs = idxs
        self.safe_idxs = np.where(np.array(labels) == 0)
        self.unsafe_idxs = np.where(np.array(labels) == 1)
        self.theta_lin = thetas[idxs[:, 2]]
        self.imgs = all_rgb_imgs
        self.heat_imgs = all_heat_imgs
        self.no_heat_imgs = all_no_heat_imgs

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_file = f"{cache_path}_{self._config.alpha_in}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump({
                "idxs": idxs,
                "safe_idxs": self.safe_idxs,
                "unsafe_idxs": self.unsafe_idxs,
                "theta_lin": self.theta_lin,
                "imgs": all_rgb_imgs,
                "heat": all_heat_imgs,
                "no_heat": all_no_heat_imgs,
                "labels": labels
            }, f)
        print(f"Cache saved to {cache_file}")

    def get_latent(self, thetas, heat_value, imgs, heat, no_heat, heat_bool=False):
        states = np.expand_dims(np.expand_dims(thetas,1),1)
        imgs = np.expand_dims(imgs, 1)
        heat = heat if heat_bool else no_heat
        heat = np.expand_dims(heat, 1)
        # print(f"[dreamer_offline/Dreamer/get_latent] heat: {heat.mean()}")
        # print(imgs.shape); quit()
        dummy_acs = np.zeros((np.shape(thetas)[0], 1))
        dummy_acs[np.arange(np.shape(thetas)[0]), :] = 0.
        firsts = np.ones((np.shape(thetas)[0], 1))
        lasts = np.zeros((np.shape(thetas)[0], 1))
        
        cos = np.cos(states)
        sin = np.sin(states)
        heat_values = np.ones_like(states) * heat_value
        if not heat_bool: heat_values *= 0
        
        if self._config.obs_priv_heat:
            states = np.concatenate([cos, sin, heat_values], axis=-1)
        else:
            states = np.concatenate([cos, sin], axis=-1)
            
        data = {'obs_state': states, 'image': imgs, 'heat': heat, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}
        # if self._config.include_no_heat_vis:
            # data = {'obs_state': states, 'image': imgs, 'heat': heat, 'no_heat': no_heat, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}
            
        data = self._wm.preprocess(data)
        embed = self._wm.encoder(data)

        post, prior = self._wm.dynamics.observe(
            embed, data["action"], data["is_first"]
            )
        feat = self._wm.dynamics.get_feat(post).detach()
        with torch.no_grad():  # Disable gradient calculation
            g_x = self._wm.heads["margin"](feat).detach().cpu().numpy().squeeze()
        feat = self._wm.dynamics.get_feat(post).detach().cpu().numpy().squeeze()

        return g_x, feat, post

    def get_eval_plot(self, heat_values=[0.2, 0.4, 0.6, 0.8]):
        self.eval()
        
        titles = ['w/ Heat', 'w/o Heat'] if self._config.include_no_heat_vis else ['w/ Heat']
        num_modes = 2 if self._config.include_no_heat_vis else 1
        num_heat = len(heat_values)

        fig, axes = plt.subplots(
            self.nz, num_heat * num_modes * 2,
            figsize=(6 * num_heat * num_modes, self.nz * 6)
        )

        if self.nz == 1:
            axes = np.expand_dims(axes, axis=0)

        metrics_summary = []

        def draw_circle(ax):
            circle = plt.Circle((0, 0), self._config.obs_r, fill=False, color='blue', label='GT boundary')
            ax.add_patch(circle)
            ax.set_aspect('equal')

        def draw_colorbar(image, ax, vmin, vmax):
            cbar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.05, shrink=0.95, ticks=[vmin, 0, vmax])
            cbar.ax.set_yticklabels([vmin, 0, vmax], fontsize=10)

        for h_idx, heat_val in enumerate(heat_values):
            # Get both modes: cold (no heat) and hot (heat)
            g_x_hot, _, _ = self.get_latent(self.theta_lin, heat_val, self.imgs[heat_val], self.heat_imgs[heat_val], self.no_heat_imgs.get(heat_val), heat_bool=True)
            g_x_list = [np.array(g_x_hot)]
            if self._config.include_no_heat_vis:
                g_x_cold, _, _ = self.get_latent(self.theta_lin, heat_val, self.imgs[heat_val], self.heat_imgs[heat_val], self.no_heat_imgs.get(heat_val), heat_bool=False)
                g_x_list = [np.array(g_x_hot), np.array(g_x_cold)]

            vmax_all = [round(max(np.max(gx), 0), 1) for gx in g_x_list]
            vmin_all = [round(min(np.min(gx), -v), 1) for gx, v in zip(g_x_list, vmax_all)]

            for mode_idx, g_x in enumerate(g_x_list):
                # Fill into full 3D grid: (x, y, θ)
                self.v[self.idxs[:, 0], self.idxs[:, 1], self.idxs[:, 2]] = g_x
                v = self.v

                # Classification metrics
                tp = np.where(g_x[self.safe_idxs] > 0)
                fn = np.where(g_x[self.safe_idxs] <= 0)
                fp = np.where(g_x[self.unsafe_idxs] > 0)
                tn = np.where(g_x[self.unsafe_idxs] <= 0)

                tp_g, fn_g = map(lambda x: x[0].shape[0], (tp, fn))
                fp_g, tn_g = map(lambda x: x[0].shape[0], (fp, tn))
                total = tp_g + fn_g + fp_g + tn_g
                metrics_summary.append((heat_val, titles[mode_idx], tp_g, tn_g, fp_g, fn_g, total))

                vmin = vmin_all[mode_idx]
                vmax = vmax_all[mode_idx]

                for i in range(self.nz):
                    col_base = h_idx * num_modes * 2 + mode_idx * 2
                    ax1 = axes[i, col_base]
                    ax2 = axes[i, col_base + 1]

                    im1 = ax1.imshow(
                        v[:, :, i].T, interpolation='none',
                        extent=[self._config.x_min, self._config.x_max, self._config.y_min, self._config.y_max],
                        origin="lower", cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
                    )
                    draw_colorbar(im1, ax1, vmin, vmax)
                    ax1.set_title(f"$g(x)$ {titles[mode_idx]}\nHeat={heat_val}", fontsize=12)
                    draw_circle(ax1)

                    im2 = ax2.imshow(
                        v[:, :, i].T > 0, interpolation='none',
                        extent=[self._config.x_min, self._config.x_max, self._config.y_min, self._config.y_max],
                        origin="lower", cmap="seismic", vmin=-1, vmax=1, zorder=-1
                    )
                    draw_colorbar(im2, ax2, -1, 1)
                    ax2.set_title(f"$v(x)$ {titles[mode_idx]}\nHeat={heat_val}", fontsize=12)
                    draw_circle(ax2)

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        summary_lines = [
            rf"$\text{{Heat={hv:.1f} {title}:  TP={tp/total:.0%}  TN={tn/total:.0%}  FP={fp/total:.0%}  FN={fn/total:.0%}}}$"
            for hv, title, tp, tn, fp, fn, total in metrics_summary
        ]
        fig.suptitle("\n".join(summary_lines), fontsize=14)

        from io import BytesIO
        from PIL import Image
        with BytesIO() as buf:
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

        self.train()
        return np.array(image), metrics_summary

    
def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def main(config):
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
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create environments") 
    action_space = gym.spaces.Box(
        low=-config.turnRate, high=config.turnRate, shape=(1,), dtype=np.float32
    )
    bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi], [0, 1]])
    low = bounds[:, 0]
    high = bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )
    # print(f"[dreamer_offline/main]: {gt_observation_space}")
    
    image_size = config.size[0] # 128
    
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
        obs_observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
    else:
        obs_observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
    
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

    # print(observation_space); quit()
        
    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    # expert episode buffer
    expert_eps = collections.OrderedDict()
    print("Expert Eps", expert_eps)
    
    config.dataset_path = f"{config.dataset_path}_{config.alpha_in}.pkl"
    tools.fill_expert_dataset_dubins(config, expert_eps)
    expert_dataset = make_dataset(expert_eps, config)
    
    # validation replay buffer
    expert_val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)
    eval_dataset = make_dataset(expert_eps, config)

    print("Length of training data:", len(expert_eps))
    print("Length of validation data:", len(expert_val_eps))

    print("Simulate agent.")
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        expert_dataset,
    ).to(config.device)
        
    step = logger.step
    agent.requires_grad_(requires_grad=False)
    
    if (logdir / "latest.pt").exists():
        print("Loading from checkpoint...")
        checkpoint = torch.load(logdir / "latest.pt", weights_only=False)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        
        agent._should_pretrain._once = False
        step = checkpoint.get("step")
        logger = tools.Logger(logdir, step)
        agent._logger = logger
        agent._step = agent._logger.step // config.action_repeat
        agent._wm._step = agent._step
        print("Done loading")
    # print(agent._wm._step); quit()
    
        try:
            print("Warming up model with one train batch to stabilize state...")
            agent.train()  # Ensure training mode
            warmup_batch = next(agent._dataset)
            agent._train(warmup_batch)
            print("Warmup step completed.")
            
        except Exception as e:
            print("[Warning] Warmup failed:", e)

    def log_plot(title, data):
        buf = BytesIO()
        plt.plot(np.arange(len(data)), data)
        plt.title(title)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        plot_arr = np.array(plot)
        logger.image("pretrain/" + title, np.transpose(plot_arr, (2, 0, 1)))
        
    def eval_obs_recon():
        recon_steps = 101
        obs_mlp, obs_opt = agent._wm._init_obs_mlp(config, 3)
        train_loss = []
        eval_loss = []
        for i in range(recon_steps):
            if i % int(recon_steps/4) == 0:
                new_loss = agent.pretrain_regress_obs(
                    next(eval_dataset), obs_mlp, obs_opt, eval=True
                )
                eval_loss.append(new_loss)
            else:
                new_loss = agent.pretrain_regress_obs(
                    next(expert_dataset), obs_mlp, obs_opt
                )
                train_loss.append(new_loss)
        log_plot("train_recon_loss", train_loss)
        log_plot("eval_recon_loss", eval_loss)
        logger.scalar("pretrain/train_recon_loss_min", np.min(train_loss))
        logger.scalar("pretrain/eval_recon_loss_min", np.min(eval_loss))
        # print(logger.step); quit()
        logger.write(step=logger.step)
        del obs_mlp, obs_opt  # dont need to keep these
        return np.min(eval_loss)
    
    def evaluate(other_dataset=None, eval_prefix=""):
        agent.eval()
        
        eval_policy = functools.partial(agent, training=False)

        # For Logging (1 episode)
        if config.video_pred_log:
            if config.multimodal:
                video_pred_rgb, video_pred_heat = agent._wm.video_pred_multimodal(next(eval_dataset))
                logger.video("eval_recon/openl_agent", to_np(video_pred_rgb))
                logger.video("eval_recon_heat/openl_agent", to_np(video_pred_heat))

                if other_dataset:
                    video_pred_rgb, video_pred_heat = agent._wm.video_pred_multimodal(next(other_dataset))
                    logger.video("train_recon/openl_agent", to_np(video_pred_rgb))
                    logger.video("train_recon_heat/openl_agent", to_np(video_pred_heat))
                    
            else:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_recon/openl_agent", to_np(video_pred))

                if other_dataset:
                    video_pred = agent._wm.video_pred(next(other_dataset))
                    logger.video("train_recon/openl_agent", to_np(video_pred))

        
        logger.write(step=logger.step)
        recon_eval = eval_obs_recon()  # testing observation reconstruction
        
        # if is_eval:
        #     quit() # TODO: make this more graceful

        agent.train()
        return recon_eval, recon_eval
    
    # ==================== Pretrain ====================
    total_train_steps = config.rssm_train_steps 
    print(total_train_steps)
    if total_train_steps > 0:
        
        cprint(
            f"Pretraining for {total_train_steps=}",
            color="cyan",
            attrs=["bold"],
        )
        ckpt_name = "rssm_ckpt" 
        best_pretrain_success = float("inf")
        for step in trange(
            agent._step,
            total_train_steps,
            desc="Training the RSSM",
            ncols=0,
            leave=False,
        ):
            if (
                ((step + 1) % config.eval_every) == 0
                or step == 1
            ):
                score, success = evaluate(
                    other_dataset=expert_dataset, eval_prefix="pretrain"
                )
                lx_plot, _ = agent.get_eval_plot()

                logger.image("pretrain/lx_plot", np.transpose(lx_plot, (2, 0, 1)))
                
                # print(step)
                best_pretrain_success = tools.save_checkpoint(
                    ckpt_name, step, success, best_pretrain_success, agent, logdir
                )
                
            exp_data = next(expert_dataset)
            agent.pretrain_model_only(exp_data, step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs.yaml", type=str)
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / f"../{args.config_path}").read_text()
    )

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
    main(parser.parse_args(remaining))