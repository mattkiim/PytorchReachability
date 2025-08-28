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
from generate_data_traj_cont import get_frame_eval, HeatFrameGenerator

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
        if config.rssm_train_steps > 0 or config.from_ckpt is not None:
            standard_kwargs = {
                "lr": config.model_lr,
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
                "opt": config.opt,
                "use_amp": use_amp,
            }

            # World model params (encoder, dynamics, decoder)
            model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
                + list(self._wm.heads["decoder"].parameters())
            }

            # Actor params
            actor_params = {
                "params": list(self._task_behavior.actor.parameters()),
                "lr": config.actor["lr"],
                "eps": config.actor["eps"],
                "clip": config.actor["grad_clip"],
            }

            # Margin head params (its own optimizer)
            margin_params = {
                "params": list(self._wm.heads["margin"].parameters()),
                "lr": config.margin_head.get("lr", config.model_lr),
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
            }
            self.margin_params = list(margin_params["params"])
            self.margin_opt = tools.Optimizer("margin_opt", [margin_params], **standard_kwargs)
            print(f"Optimizer margin_opt has {sum(p.numel() for p in self.margin_params)} variables.")

            # Pretrain optimizer (combined, if you want to toggle margin in/out)
            self.pretrain_params = list(model_params["params"]) + list(actor_params["params"]) + list(margin_params["params"])
            self.pretrain_opt = tools.Optimizer("pretrain_opt", [model_params, actor_params, margin_params], **standard_kwargs)

            # Model-only optimizer (excludes margin)
            excluded_params = set(self._wm.heads["margin"].parameters())
            model_only_params = [p for p in self._wm.parameters() if p not in excluded_params]
            self._model_opt = tools.Optimizer("model", model_only_params, config.model_lr, config.opt_eps,
                                            config.grad_clip, config.weight_decay, opt=config.opt, use_amp=self._wm._use_amp)

            # Extra bookkeeping
            self._opt = self.pretrain_opt._opt
            self._margin_pg_idx = 2
            self._margin_pg = self._opt.param_groups[self._margin_pg_idx]
            self._margin_warmup_step = config.margin_head["warmup_steps"]
            self._margin_active = False
            self.actor_params = list(self._task_behavior.actor.parameters())

            print(f"Optimizer pretrain has {sum(param.numel() for param in self.pretrain_params)} variables.")


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
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        wm = self._wm
        data = wm.preprocess(data)

        with tools.RequiresGrad(wm):
            with torch.amp.autocast(device_type='cuda', enabled=wm._use_amp):
                embed = wm.encoder(data)
                post, _ = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                    
        post = {k: v.detach() for k, v in post.items()}

        feat_detached = wm.dynamics.get_feat(post).detach()
        safe_data = torch.where(data["failure"] == 0.)
        unsafe_data = torch.where(data["failure"] == 1.)
        safe_dataset = feat_detached[safe_data]
        unsafe_dataset = feat_detached[unsafe_data]

        with tools.RequiresGrad(wm.heads["margin"]):
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                pos = wm.heads["margin"](safe_dataset)
                neg = wm.heads["margin"](unsafe_dataset)
                N = max(pos.numel(), neg.numel())
                gp_loss = torch.tensor(0., device=pos.device)

                if pos.numel() > 0 and neg.numel() > 0:
                    # balance safe and unsafe datasets
                    if N > safe_dataset.shape[0]:
                        repeat_times = (N + safe_dataset.shape[0] - 1) // safe_dataset.shape[0]
                        safe_repeated = safe_dataset.repeat((repeat_times,) + (1,) * (safe_dataset.dim() - 1))
                        indices = torch.randperm(safe_repeated.shape[0], device=safe_dataset.device)[:N]
                        pos_data = safe_repeated[indices]
                    else:
                        pos_data = safe_dataset

                    if N > unsafe_dataset.shape[0]:
                        repeat_times = (N + unsafe_dataset.shape[0] - 1) // unsafe_dataset.shape[0]
                        unsafe_repeated = unsafe_dataset.repeat((repeat_times,) + (1,) * (unsafe_dataset.dim() - 1))
                        indices = torch.randperm(unsafe_repeated.shape[0], device=unsafe_dataset.device)[:N]
                        neg_data = unsafe_repeated[indices]
                    else:
                        neg_data = unsafe_dataset

                    # gradient penalty
                    alpha = torch.rand(pos_data.shape[0], 1, device=pos_data.device)
                    interpolates = alpha * pos_data + (1 - alpha) * neg_data
                    interpolates.requires_grad_(True)
                    disc_interpolates = wm.heads["margin"](interpolates)

                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(disc_interpolates),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    gradients = gradients.view(pos_data.shape[0], -1)
                    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
                    gradient_thresh = 0.1
                    excess = (gradients_norm - gradient_thresh).clamp(min=0)
                    gp_loss = (excess ** 2).mean()

                gamma = self._config.gamma_lx
                relu_loss = torch.tensor(0., device=pos.device)
                zero_sum_loss = torch.tensor(0., device=pos.device)

                if pos.numel() > 0:
                    pos_mean = pos.mean()
                    zero_sum_loss -= pos_mean
                    relu_loss += torch.relu(gamma - pos).mean()

                if neg.numel() > 0:
                    neg_mean = neg.mean()
                    zero_sum_loss += neg_mean
                    relu_loss += torch.relu(gamma + neg).mean()

                relu_weight = 100
                gp_weight = 10
                print('losses', 
                    round(relu_weight * relu_loss.item(), 2),
                    round(gp_weight * gp_loss.item(), 2),
                    round(zero_sum_loss.item(), 2))

                loss = zero_sum_loss + relu_weight * relu_loss + gp_weight * gp_loss

                self._model_opt(torch.mean(loss), wm.parameters())
                metrics = self.margin_opt(loss, wm.heads["margin"].parameters())

                metrics["sign_loss"] = to_np(relu_loss)
                metrics["zero_sum_loss"] = to_np(zero_sum_loss)
                metrics["gp_loss"] = to_np(gp_loss)

        metrics = {f"model_only_pretrain/{k}": v for k, v in metrics.items()}
        self._update_running_metrics(metrics)
        self._maybe_log_metrics()
        self._step += 1
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
        # if self._config.include_no_heat:
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
    
    @torch.no_grad()
    def eval_full_metrics(self, data):
        """
        Held-out evaluation that logs:
          - Per-head reconstruction means (excluding 'margin' from recon_sum; 'cont' logged separately)
          - Failure-margin loss (lx_loss)
          - KL / dyn / rep, entropies
          - total_loss := recon_sum + lx_loss  (selection metric for HW expts)
        No gradients and no optimizer step.
        """
        wm = self._wm
        cfg = self._config
        data = wm.preprocess(data)

        with torch.amp.autocast("cuda", enabled=wm._use_amp):
            # Encode and infer posterior over the observed sequence
            embed = wm.encoder(data)
            post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])

            # KL terms (for logging)
            kl_free = cfg.kl_free
            dyn_scale = cfg.dyn_scale
            rep_scale = cfg.rep_scale
            kl_loss_bt, kl_value_bt, dyn_loss_bt, rep_loss_bt = wm.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )
            kl_loss = kl_loss_bt.mean()
            dyn_loss = dyn_loss_bt.mean()
            rep_loss = rep_loss_bt.mean()
            kl_value = kl_value_bt.mean()

            feat = wm.dynamics.get_feat(post)

            # ---------- Predict with all heads ----------
            preds = {}
            for name, head in wm.heads.items():
                pred = head(feat)
                if isinstance(pred, dict):
                    preds.update(pred)
                else:
                    preds[name] = pred

            # ---------- Per-head losses ----------
            per_head_means = {}
            recon_terms = []  # exclude 'margin'; keep 'cont' separate
            cont_loss = torch.tensor(0.0, device=feat.device)

            for name, pred in preds.items():
                if name == "margin":
                    continue  # margin is handled separately
                if name not in data:
                    continue  # skip heads with no target
                loss_bt = -pred.log_prob(data[name])  # [B, T]
                if name == "heat":
                    loss_bt *= 3
                mean_loss = loss_bt.mean()
                per_head_means[name] = mean_loss
                if name == "cont":
                    cont_loss = mean_loss
                else:
                    recon_terms.append(mean_loss)

            recon_sum = torch.stack(recon_terms).sum() if len(recon_terms) else torch.tensor(0.0, device=feat.device)

            # ---------- Failure-margin loss ----------
            lx_loss = torch.tensor(0.0, device=feat.device)
            if "failure" in data:
                failure = data["failure"]  # 1=unsafe, 0=safe
                safe_mask = (failure == 0)
                unsafe_mask = ~safe_mask

                safe_feat = feat[safe_mask]
                unsafe_feat = feat[unsafe_mask]

                gamma = cfg.gamma_lx
                if safe_feat.numel() > 0:
                    pos = wm.heads["margin"](safe_feat)
                    # lx_loss = lx_loss + torch.relu(gamma - pos).mean()
                    lx_loss = lx_loss + torch.nn.functional.softplus(gamma - pos).mean()
                if unsafe_feat.numel() > 0:
                    neg = wm.heads["margin"](unsafe_feat)
                    # lx_loss = lx_loss + torch.relu(gamma + neg).mean()
                    lx_loss = lx_loss + torch.nn.functional.softplus(gamma + neg).mean()
                lx_loss = lx_loss * cfg.margin_head["loss_scale"]

            # ---------- Selection metric ----------
            total_eval_loss = recon_sum + lx_loss

            # Entropies
            prior_ent = wm.dynamics.get_dist(prior).entropy().mean()
            post_ent = wm.dynamics.get_dist(post).entropy().mean()

        # Package numpy scalars
        out = {
            "recon_sum": to_np(recon_sum),
            "lx_loss": to_np(lx_loss),
            "total_loss": to_np(total_eval_loss),  # <-- use for checkpoint "success"
            "cont_loss": to_np(cont_loss),
            "kl_loss": to_np(kl_loss),
            "dyn_loss": to_np(dyn_loss),
            "rep_loss": to_np(rep_loss),
            "kl_value": to_np(kl_value),
            "prior_ent": to_np(prior_ent),
            "post_ent": to_np(post_ent),
        }
        # Per-head breakdowns
        for name, l in per_head_means.items():
            out[f"recon/{name}"] = to_np(l)
        return out

    @torch.no_grad()
    def evaluate_full_metrics(self, dataset, batches=10, prefix="eval"):
        """
        Run eval_full_metrics over 'batches' held-out batches, average, and log ALL metrics.
        Returns (recon_sum_mean, total_mean).
        """
        logs = {}

        for _ in range(batches):
            res = self.eval_full_metrics(next(dataset))
            for k, v in res.items():
                logs.setdefault(k, []).append(v)

        means = {k: float(np.mean(vs)) for k, vs in logs.items()}

        if self._logger is not None:
            for k, v in means.items():
                self._logger.scalar(f"{prefix}/{k}", v)
            self._logger.write(step=self._logger.step)

        recon_mean = means.get("recon_sum", 0.0)
        total_mean = means.get("total_loss", recon_mean)
        return recon_mean, total_mean

    
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
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
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
            low=-1, high=1, shape=(9,), dtype=np.float32
        )
    else:
        obs_observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
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
    
    config.dataset_path = config.dataset_path
    tools.fill_expert_dataset_dubins(config, expert_eps)
    expert_dataset = make_dataset(expert_eps, config)
    
    # validation replay buffer
    expert_val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)
    eval_dataset = make_dataset(expert_val_eps, config)

    print("Length of training data:", len(expert_eps)) # 32
    print("Length of validation data:", len(expert_val_eps)) # 10

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
        obs_mlp, obs_opt = agent._wm._init_obs_mlp(config, 8)
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

        # ---------- (UNCHANGED) optional visualization ----------
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

        # ---------- (KEEP) original auxiliary eval metrics ----------
        # Trains a small MLP and logs train/eval curves + min losses.
        recon_eval = eval_obs_recon()  # returns min eval MSE from the aux regressor

        # ---------- (ADD) new held-out metrics: recon + failure-margin + full logs ----------
        eval_batches = getattr(config, "eval_batches", 10)
        recon_mean, total_mean = agent.evaluate_full_metrics(
            eval_dataset, batches=eval_batches, prefix="eval"
        )

        agent.train()

        # Maintain (score, success) signature for your checkpoint code:
        #   - score   : original aux metric (min eval_obs_recon)
        #   - success : NEW selection metric (recon_sum + lx_loss)
        return recon_eval, total_mean

    
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