# eval_only.py
import argparse
import functools
import os
os.environ["WANDB_MODE"] = "disabled"
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

from termcolor import cprint
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import gym  # If you’ve migrated: `import gymnasium as gym`

to_np = lambda x: x.detach().cpu().numpy()
from generate_data_traj_cont import get_frame_eval, HeatFrameGenerator


class Dreamer(nn.Module):
    """Unchanged core; we only use its eval utilities."""
    def __init__(self, obs_space, act_space, config, logger, dataset):
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
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset

        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)

        if (config.compile and os.name != "nt"):
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

        # Only needed for exploration policy during rollouts; harmless for eval-only
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        # NOTE: We won’t train here, but keeping this call is harmless.
        self._make_pretrain_opt()

    def _make_pretrain_opt(self):
        # Safe to leave as-is; no training will use it.
        config = self._config
        use_amp = True if config.precision == 16 else False
        if (config.rssm_train_steps > 0):
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

    def pretrain_regress_obs(self, data, obs_mlp, obs_opt, eval=False):
        wm = self._wm
        data = wm.preprocess(data)
        if eval:
            obs_mlp.eval()
        with tools.RequiresGrad(obs_mlp):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
                feat = wm.dynamics.get_feat(prior).detach()  # use imagined prior
                target = torch.Tensor(data["privileged_state"]).to(self._config.device)
                pred_state = obs_mlp(feat)
                obs_loss = torch.mean((pred_state - target) ** 2)
            if not eval:
                obs_opt(torch.mean(obs_loss), obs_mlp.parameters())
            else:
                obs_mlp.train()
        return obs_loss.item()

    @torch.no_grad()
    def evaluate_full_metrics(self, dataset, batches=10, prefix="eval"):
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
    
    @torch.no_grad()
    def _decode_all_heads(self, feat):
        """
        Distribution heads -> .mode()
        Tensor heads (e.g., 'margin') -> raw tensor
        """
        outs = {}
        for name, head in self._wm.heads.items():
            pred = head(feat)
            if isinstance(pred, dict):
                for k, dist in pred.items():
                    outs[k] = dist.mode() if hasattr(dist, "mode") else dist
            else:
                if isinstance(pred, torch.Tensor):
                    outs[name] = pred
                elif hasattr(pred, "mode"):
                    outs[name] = pred.mode()
                else:
                    raise TypeError(
                        f"Head '{name}' returned unsupported type: {type(pred)}"
                    )
        return outs
    
    @torch.no_grad()
    def latent_state_test(self, windows: list, warm: int = 5):
        """
        Latent State Test (multi-batch):
        - Observe `warm` timesteps
        - Take posterior latent z at the last observed step
        - Evaluate l(z) using the margin head
        - Compare to ground-truth failure label at that same timestep
        - Aggregate results across `n_batches`
        """
        wm = self._wm
        agg = dict(TP=0, TN=0, FP=0, FN=0, N=0)

        for batch in windows:
            data = wm.preprocess(batch)
            B, T = data["action"].shape[:2]

            # Which latent to score
            idx = warm-1

            # Posterior latents
            embed = wm.encoder(data)
            post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
            feats = wm.dynamics.get_feat(post)  # [B, T, F]

            z = feats[:, idx]  # [B, F]
            margin = wm.heads["margin"](z).squeeze(-1)  # [B]
            pred_unsafe = (margin < 0)

            y = (data["failure"][:, idx] > 0.5)

            TP = (~pred_unsafe & ~y).sum().item()
            TN = ( pred_unsafe &  y).sum().item()
            FP = (~pred_unsafe &  y).sum().item()
            FN = ( pred_unsafe & ~y).sum().item()
            N  = int(y.numel())

            agg["TP"] += int(TP); agg["TN"] += int(TN)
            agg["FP"] += int(FP); agg["FN"] += int(FN)
            agg["N"]  += N
                
            agg["tpr"] = agg["TP"] / agg["N"]
            agg["tnr"] = agg["TN"] / agg["N"]
            agg["fpr"] = agg["FP"] / agg["N"]
            agg["fnr"] = agg["FN"] / agg["N"]

        agg["acc"] = (agg["TP"] + agg["TN"]) / max(1, agg["N"])
        return agg

    @torch.no_grad()
    def rollout_16_warm5(self, batch, mode="open", actor_mode=True):
        """
        Hybrid rollout of length 16:
        - first 5 steps: posterior updates with GT embeds + (GT actions)  -> 'warm'
        - next 11 steps: imagined steps
            * open:   use dataset actions
            * closed: use actor actions
        Returns per-head MSEs (all 16 and imag-only) and margin stats. NO videos.
        """
        assert mode in ("open", "closed")
        H, WARM = 16, 5
        FUT = H - WARM

        wm = self._wm
        cfg = self._config
        data = wm.preprocess(batch)  # dict of [B,T,...]
        B, T = data["action"].shape[:2]
        assert T >= H + 1, "Need at least H+1 timesteps in the batch."

        t0 = T - (WARM + FUT) - 1
        t0 = max(0, t0)

        # Encode whole sequence to get posterior states (for warm steps)
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])

        # Warm decoded reconstructions (posterior)
        warm_states = {k: v[:, t0 + 1 : t0 + 1 + WARM] for k, v in post.items()}
        warm_feat   = wm.dynamics.get_feat(warm_states)
        warm_dec    = self._decode_all_heads(warm_feat)  # dict of tensors [B,WARM,...]

        # Prepare starting state for imagination
        curr = {k: v[:, t0 + WARM] for k, v in post.items()}

        # Imagine FUT steps
        if mode == "open":
            a_seq   = data["action"][:, t0 + WARM + 1 : t0 + WARM + 1 + FUT]  # [B,FUT,A]
            prior   = wm.dynamics.imagine_with_action(a_seq, curr)            # dict of [B,FUT,...]
            imag_feat = wm.dynamics.get_feat(prior)                           # [B,FUT,F]
            imag_dec  = self._decode_all_heads(imag_feat)                     # dict of [B,FUT,...]
        else:
            # Closed-loop: actor picks actions step-by-step
            actor = self._task_behavior.actor
            feats = []
            c = curr
            for _ in range(FUT):
                feat_t = wm.dynamics.get_feat(c)            # [B, F]
                dist   = actor(feat_t)
                a_t    = dist.mode() if actor_mode else dist.sample()  # [B, A]
                c      = wm.dynamics.img_step(c, a_t, sample=True)     # next state
                feats.append(wm.dynamics.get_feat(c))       # features for next state

            # Stack across time and decode once (now shapes are [B, FUT, F])
            imag_feat = torch.stack(feats, dim=1)           # [B, FUT, F]
            imag_dec  = self._decode_all_heads(imag_feat)   # dict of [B, FUT, ...]
            
        # Concatenate warm + imag for outputs where it makes sense
        full_dec = {}
        for k in set(warm_dec.keys()).union(set(imag_dec.keys())):
            w = warm_dec.get(k, None)
            i = imag_dec.get(k, None)
            if w is not None and i is not None:
                full_dec[k] = torch.cat([w, i], dim=1)  # [B,16,...]
            elif w is not None:
                full_dec[k] = w
            elif i is not None:
                full_dec[k] = i

        # Targets for comparison (ground-truth slice [t0+1 .. t0+H])
        targets = {}
        for k in ("image", "heat", "obs_state"):
            if k in data:
                targets[k] = data[k][:, t0 + 1 : t0 + 1 + H]  # [B,16,...]

        # MSE over all 16 and imag-only (last FUT)
        mse_all, mse_imag = {}, {}
        for k, tgt in targets.items():
            if k in full_dec and full_dec[k] is not None:
                mse_all[k]  = torch.mean((full_dec[k] - tgt) ** 2).item()
                mse_imag[k] = torch.mean((full_dec[k][:, WARM:] - tgt[:, WARM:]) ** 2).item()

        # Safety margin statistics for imagined part
        margin_vals = self._wm.heads["margin"](imag_feat)  # [B,FUT,1] or [B,FUT]
        margin_mean = margin_vals.mean().item()
        frac_unsafe = (margin_vals < cfg.gamma_lx).float().mean().item()

        return {
            "mse_all": mse_all,
            "mse_imag": mse_imag,
            "margin_mean_imag": margin_mean,
            "frac_unsafe_imag": frac_unsafe,
        }
        
    @torch.no_grad()
    def confusion_16_warm5(self, batch, mode="open", actor_mode=True):
        """
        Like rollout_16_warm5 but only returns confusion counts over the 11 imagined steps.
        Positive class = 'unsafe' (failure == 1). Prediction = (margin < gamma_lx).
        Returns: dict(TP=..., TN=..., FP=..., FN=..., total=...)
        """
        H, WARM = 16, 5
        FUT = H - WARM

        wm = self._wm
        cfg = self._config
        data = wm.preprocess(batch)  # dict of [B, T, ...]
        B, T = data["action"].shape[:2]

        # choose start index so that [t0+1..t0+WARM] are warm steps, and [t0+WARM+1..WARM+FUT] are imagined steps
        t0 = max(0, T - (WARM + FUT) - 1)

        # Posterior from full sequence for warm steps
        embed = wm.encoder(data)
        post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])

        # Starting state at the boundary
        curr = {k: v[:, t0 + WARM] for k, v in post.items()}

        # Imagine FUT steps
        if mode == "open":
            a_seq = data["action"][:, t0 + WARM + 1 : t0 + WARM + 1 + FUT]
            prior = wm.dynamics.imagine_with_action(a_seq, curr)
            imag_feat = wm.dynamics.get_feat(prior)
        else:
            cl_states = {k: v[:, t0 + WARM + 1 : t0 + WARM + 1 + FUT] for k, v in post.items()}
            imag_feat = wm.dynamics.get_feat(cl_states)

        # Predicted (unsafe=1) if margin < gamma
        margin = self._wm.heads["margin"](imag_feat)
        margin = margin.squeeze(-1) if margin.dim() == 3 else margin
        pred_unsafe = (margin < cfg.gamma_lx)

        # Ground-truth (unsafe=1)
        gt = data["failure"][:, t0 + WARM + 1 : t0 + WARM + 1 + FUT]
        gt_unsafe = (gt > 0.5)
        
        # Confusion counts
        TP = torch.sum(~pred_unsafe & (~gt_unsafe)).item()
        TN = torch.sum(pred_unsafe & gt_unsafe).item()
        FP = torch.sum(~pred_unsafe & gt_unsafe).item()
        FN = torch.sum(pred_unsafe & (~gt_unsafe)).item()
        total = int(pred_unsafe.numel())
        return dict(TP=int(TP), TN=int(TN), FP=int(FP), FN=int(FN), total=total)

    @torch.no_grad()
    def eval_confusion_from_batches(self, batches, mode="open", actor_mode=True, log_prefix=None,
                                   fpr_over_total=True):
        """
        Use a pre-fetched list of batches so OL and CL evaluate on the *same* samples.
        Aggregates TP/TN/FP/FN over all batches, computes metrics, and (optionally) logs.
        """
        agg = dict(TP=0, TN=0, FP=0, FN=0, total=0)
        for batch in batches:
            res = self.confusion_16_warm5(batch, mode=mode, actor_mode=actor_mode)
            for k in agg: agg[k] += res[k]

        TP, TN, FP, FN, N = agg["TP"], agg["TN"], agg["FP"], agg["FN"], agg["total"]
        # Your requested definition: FPR = FP / total
        tpr = TP / N
        tnr = TN / N
        fpr = FP / N
        fnr = FN / N
            
        out = {**agg, "tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr}

        if self._logger is not None and log_prefix:
            for k, v in out.items():
                self._logger.scalar(f"{log_prefix}/{k}", float(v))
            self._logger.write(step=self._logger.step)

        return out

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



def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def main(config, ckpt_path=None, eval_batches=None):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    # Maintain logging schedule but we won't train
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    # Logger step reflects environment steps (kept for consistency)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    # ------------- Spaces (unchanged) -------------
    bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi], [0, 1]])
    low = bounds[:, 0]
    high = bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )
    action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
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
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
    else:
        obs_observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
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

    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    # ------------- Load dataset -------------
    expert_val_eps = collections.OrderedDict()
    config.dataset_path = f"{config.dataset_path}_{config.alpha_in}.pkl"
    tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)
    eval_dataset = make_dataset(expert_val_eps, config)

    print("Length of validation data:", len(expert_val_eps))

    # ------------- Build agent (no training) -------------
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        dataset=None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    # ------------- Load checkpoint -------------
    if ckpt_path is None:
        ckpt_path = config.rssm_ckpt_path
        # ckpt_path = "/data/mattkiim/runs/merged_5hz_fast_avg_gp/latest.pt"
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)

    agent.eval(); agent._wm.eval(); agent._task_behavior.eval()
    config.eval_disable_aug = True

    
    # recon_mean, total_mean = agent.evaluate_full_metrics(
    #     eval_dataset, batches=getattr(config, "eval_batches", 10), prefix="loaded"
    # )


    # Optims are irrelevant for eval

    # ------------- Optional: video predictions on eval set -------------
    # if config.video_pred_log:
    #     try:
    #         if config.multimodal:
    #             video_pred_rgb, video_pred_heat = agent._wm.video_pred_multimodal(next(eval_dataset))
    #             logger.video("eval_recon/openl_agent", to_np(video_pred_rgb))
    #             logger.video("eval_recon_heat/openl_agent", to_np(video_pred_heat))
    #         else:
    #             video_pred = agent._wm.video_pred(next(eval_dataset))
    #             logger.video("eval_recon/openl_agent", to_np(video_pred))
    #         logger.write(step=logger.step)
    #     except Exception as e:

    # ------------- Eval: probe MLP & full metrics -------------
    def log_plot(title, data):
        buf = BytesIO()
        plt.plot(np.arange(len(data)), data)
        plt.title(title)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        plot_arr = np.array(plot)
        logger.image("eval/" + title, np.transpose(plot_arr, (2, 0, 1)))

    def eval_obs_recon():
        recon_steps = 101
        obs_mlp, obs_opt = agent._wm._init_obs_mlp(config, 8)
        train_loss, eval_loss = [], []
        for i in range(recon_steps):
            if i % int(recon_steps / 4) == 0:
                new_loss = agent.pretrain_regress_obs(next(eval_dataset), obs_mlp, obs_opt, eval=True)
                eval_loss.append(new_loss)
            else:
                new_loss = agent.pretrain_regress_obs(next(expert_dataset), obs_mlp, obs_opt)
                train_loss.append(new_loss)
        log_plot("train_recon_loss", train_loss)
        log_plot("eval_recon_loss", eval_loss)
        logger.scalar("eval/train_recon_loss_min", float(np.min(train_loss)))
        logger.scalar("eval/eval_recon_loss_min", float(np.min(eval_loss)))
        logger.write(step=logger.step)
        del obs_mlp, obs_opt
        return float(np.min(eval_loss))

    # Run both eval passes
    print("Running evaluation ...")
    # probe_mse = eval_obs_recon()
    # batches = eval_batches if eval_batches is not None else getattr(config, "eval_batches", 10)
    # recon_mean, total_mean = agent.evaluate_full_metrics(eval_dataset, batches=batches, prefix="eval")
    
    # # ------------- Eval: OL and CL rollouts -------------
    # # ---- Hybrid 16-step eval (5 warm + 11 imagine) ----
    # batch_eval = next(eval_dataset)  # needs T >= 17 inside the batch

    # # Open-loop (uses dataset actions for imagined steps)
    # res_open = agent.rollout_16_warm5(batch_eval, mode="open")

    # # Closed-loop (actor picks actions for imagined steps)
    # res_close = agent.rollout_16_warm5(batch_eval, mode="closed", actor_mode=True)

    # # Log scalars
    # for k, v in res_open["mse_all"].items():
    #     logger.scalar(f"hybrid16_open/mse_all_{k}", float(v))
    # for k, v in res_open["mse_imag"].items():
    #     logger.scalar(f"hybrid16_open/mse_imag_{k}", float(v))
    # logger.scalar("hybrid16_open/margin_mean_imag", float(res_open["margin_mean_imag"]))
    # logger.scalar("hybrid16_open/frac_unsafe_imag", float(res_open["frac_unsafe_imag"]))

    # for k, v in res_close["mse_all"].items():
    #     logger.scalar(f"hybrid16_closed/mse_all_{k}", float(v))
    # for k, v in res_close["mse_imag"].items():
    #     logger.scalar(f"hybrid16_closed/mse_imag_{k}", float(v))
    # logger.scalar("hybrid16_closed/margin_mean_imag", float(res_close["margin_mean_imag"]))
    # logger.scalar("hybrid16_closed/frac_unsafe_imag", float(res_close["frac_unsafe_imag"]))
    
    # # Also print a short console summary
    
    # ------------- Eval: OL and CL safety confusion (imagined horizon only) -------------
    n_windows = 50  # number of windows you want to evaluate
    shared_batches = [next(eval_dataset) for _ in range(n_windows)]

    # # Open-loop on shared samples
    # open_stats  = agent.eval_confusion_from_batches(
    #     shared_batches, mode="open",  log_prefix="conf/open",  fpr_over_total=True
    # )
    # # Closed-loop on the exact same samples
    # closed_stats = agent.eval_confusion_from_batches(
    #     shared_batches, mode="closed", log_prefix="conf/closed", fpr_over_total=True
    # )

    # for k, v in open_stats.items():

    # for k, v in closed_stats.items():
        
    
    latent_state_test = agent.latent_state_test(shared_batches, warm=5)
    print("\n=== Latent State Test ===")
    for k, v in latent_state_test.items():
        print(f"{k}: {v}")
    
    quit()
    
    logger.write(step=logger.step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs.yaml", type=str)
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to a checkpoint (defaults to logdir/latest.pt)")
    parser.add_argument("--eval_batches", type=int, default=None, help="Override number of held-out eval batches")
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

    # Optional: disable training-specific setup
    # Ensures we don't even create/consider training loops
    cfg.rssm_train_steps = 0

    # Run eval-only
    main(cfg, ckpt_path=args.ckpt_path, eval_batches=args.eval_batches)