#!/usr/bin/env python3
# eval_ddpg.py
import os
import sys
import io
import cv2
import pickle
import argparse
import pathlib
import collections
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from datetime import datetime
from termcolor import cprint
from pathlib import Path

import numpy as np
import torch
import gymnasium
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- Repo paths --------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/PyHJ'))
sys.path.append(saferl_dir)

# --- Project imports ---------------------------------------------------------
import ruamel.yaml as yaml
import models
import tools
from dreamer import make_dataset

from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.exploration import GaussianNoise
from PyHJ.data import Batch
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
from generate_data_traj_cont import (
    get_frame_eval,
    get_frame_eval_pil,
    HeatFrameGenerator,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            recursive_update(base[k], v)
        else:
            base[k] = v

def get_args():
    parser = argparse.ArgumentParser("Eval-only script for Dreamer+DDPG safety visualizations")

    # config + experiment naming
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--config_path", default="configs.yaml", type=str)
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)

    # eval options
    parser.add_argument("--model_path", type=str,
                        help="Path to saved DDPG policy .pth (the actor/critic policy file).",
                        default="logs/dreamer_dubins_multimodal_v2plus3/PyHJ/0908/152442/PyHJ/dubins-wm/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_128_3_a1_128_3_gamma_0.9999/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_16/policy.pth")
    parser.add_argument("--rssm_ckpt_path", type=str, required=False,
                        help="Override Dreamer RSSM checkpoint (else read from YAML defaults).")
    parser.add_argument("--output_dir", type=str, default="eval",
                        help="Directory to save figures and video.")
    parser.add_argument("--make_cache_if_missing", action="store_true",
                        help="If set, creates the HJ cache when missing; otherwise raises.")
    parser.add_argument("--rollout_T", type=int, default=100)
    parser.add_argument("--vels", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--heat_values", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])

    # parse initial to locate config yaml section(s)
    config, remaining = parser.parse_known_args()

    # handle expt name
    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    # read yaml defaults + overlays
    yml = yaml.YAML(typ="safe", pure=True)
    configs = yml.load((pathlib.Path(sys.argv[0]).parent / f"../{config.config_path}").read_text())
    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    # materialize final argparse with defaults
    parser = argparse.ArgumentParser("Eval-only (resolved defaults)")
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    # re-inject eval-only args
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--rssm_ckpt_path_override", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=config.output_dir)
    parser.add_argument("--make_cache_if_missing", action="store_true", default=config.make_cache_if_missing)
    parser.add_argument("--rollout_T", type=int, default=config.rollout_T)
    parser.add_argument("--vels", type=float, nargs="+", default=config.vels)
    parser.add_argument("--heat_values", type=float, nargs="+", default=config.heat_values)

    final_args = parser.parse_args(remaining)
    # carry-through top-level inputs
    final_args.logdir = f"{final_args.logdir+'/PyHJ'}/{config.expt_name}"
    if config.model_path:
        final_args.model_path = config.model_path
    if config.rssm_ckpt_path and not final_args.rssm_ckpt_path_override:
        final_args.rssm_ckpt_path = config.rssm_ckpt_path
    if final_args.rssm_ckpt_path_override:
        final_args.rssm_ckpt_path = final_args.rssm_ckpt_path_override

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_args.task}", "cyan", attrs=["bold"])
    cprint(f"Output to: {final_args.output_dir}", "cyan", attrs=["bold"])
    print("---------------------")
    return final_args

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img.convert('RGB')

# -----------------------------------------------------------------------------
# Cache (same structure as in your training script)
# -----------------------------------------------------------------------------
def make_cache(config, vels, heat_values):
    nx, ny = config.nx, config.ny
    cache = {}
    xs = np.linspace(-1.5, 1.5, nx, endpoint=True)
    ys = np.linspace(-1.5, 1.5, ny, endpoint=True)

    for heat_value in heat_values:
        for vel in vels:
            theta = 0.0
            v = np.zeros((nx, ny))
            key = (vel, heat_value)
            print('creating cache for key', key)

            idxs, imgs_prev, heat_imgs, no_heat_imgs = [], [], [], []
            thetas_prev, vels_prev, heat_values_prev, states = [], [], [], []

            xs_prev = xs - config.dt * config.speed * np.cos(theta)
            ys_prev = ys - config.dt * config.speed * np.sin(theta)
            theta_prev = theta
            vel_prev = vel

            it = np.nditer(v, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                x_prev = xs_prev[idx[0]]
                y_prev = ys_prev[idx[1]]

                prev_state = torch.tensor([x_prev, y_prev, theta_prev, vel_prev])
                img = get_frame_eval_pil(prev_state, config) if config.use_pil else get_frame_eval(prev_state, config)
                gen = HeatFrameGenerator(config)
                gen._compute_geometry(img.shape)

                if config.heat_mode == 0:
                    heat = gen.get_heat_frame_v0(img, heat=True)
                    if config.include_no_heat_vis:
                        no_heat = gen.get_heat_frame_v0(img, heat=False)
                elif config.heat_mode == 1:
                    heat = gen.get_heat_frame_v1(img, heat=True)
                    if config.include_no_heat_vis:
                        no_heat = gen.get_heat_frame_v1(img, heat=False)
                elif config.heat_mode == 2:
                    img = gen.get_rgb_po(img, config, heat=True)
                    heat, _ = gen.get_heat_frame_po(img, config, heat=True, heat_value=heat_value)
                    if config.include_no_heat_vis:
                        no_heat, _ = gen.get_heat_frame_po(img, config, heat=False, heat_value=heat_value)
                elif config.heat_mode == 3:
                    img = gen.get_rgb_fo(img, config, heat=True, heat_value=heat_value)
                    heat, _ = gen.get_heat_frame_fo(img, config, heat=True, heat_value=heat_value)
                    if config.include_no_heat_vis:
                        no_heat, _ = gen.get_heat_frame_fo(img, heat=False, heat_value=heat_value)
                else:
                    raise ValueError(f"Unknown heat_mode: {config.heat_mode}")

                idxs.append(idx)
                imgs_prev.append(img)
                heat_imgs.append(heat)
                if config.include_no_heat_vis:
                    no_heat_imgs.append(no_heat)
                thetas_prev.append(theta_prev)
                vels_prev.append(vel_prev)
                heat_values_prev.append(heat_value)
                states.append(prev_state)
                it.iternext()

            cache[(vel, heat_value)] = [
                np.array(idxs),
                imgs_prev,
                heat_imgs,
                no_heat_imgs,
                np.array(thetas_prev),
                np.array(vels_prev),
                states,
            ]

    cache_path = f"{config.hj_cache_path}_{config.alpha_in}.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    return cache

def load_cache(config, allow_make=False, vels=None, heat_values=None):
    cache_path = f"{config.hj_cache_path}_{config.alpha_in}.pkl"
    if not os.path.exists(cache_path):
        if allow_make:
            print(f"[cache] not found, creating at {cache_path}")
            return make_cache(config, vels, heat_values)
        raise FileNotFoundError(f"Cache file not found at: {cache_path}")
    print(f"[cache] Loading {cache_path}")
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


import numpy as np
import torch
from generate_data_traj_cont import get_frame_eval_pil, HeatFrameGenerator
from PyHJ.data import Batch

# ---------- helpers: geometry / heat back-prop ----------

@torch.no_grad()
def _inside_obs_xy(cfg, x, y):
    dx, dy = x - cfg.obs_x, y - cfg.obs_y
    return (dx*dx + dy*dy) ** 0.5 < cfg.obs_r

def _backstep_state(cfg, xythetav):
    x, y, th, v = xythetav
    dt = cfg.dt
    return np.array([x - v*np.cos(th)*dt,
                     y - v*np.sin(th)*dt,
                     th,
                     v], dtype=np.float32)

def _backstep_heat(cfg, h_t, x_prev, y_prev):
    S = 255.0 / 1.1
    if _inside_obs_xy(cfg, x_prev, y_prev):
        h_prev = h_t - (cfg.alpha_in / S)
    else:
        h_prev = h_t + (cfg.alpha_out / S)
    return float(np.clip(h_prev, 0.0, 1.0))

def _build_backward_history(cfg, x0, heat0, K):
    xs = [np.array(x0, dtype=np.float32)]
    hs = [float(heat0)]
    cur_x, cur_h = xs[0].copy(), hs[0]
    for _ in range(K):
        xp = _backstep_state(cfg, cur_x)
        hp = _backstep_heat(cfg, cur_h, xp[0], xp[1])
        xs.append(xp); hs.append(hp)
        cur_x, cur_h = xp, hp
    return xs[::-1], hs[::-1]  # chronological: [x_-K ... x_0]

def _render_seq(cfg, xs, hs, use_no_heat=False):
    gen = HeatFrameGenerator(cfg)
    imgs, heats = [], []
    for s, h in zip(xs, hs):
        img_og = get_frame_eval_pil(s, cfg)
        gen._compute_geometry(img_og.shape)

        if cfg.heat_mode == 3:
            img = gen.get_rgb_fo(img_og, cfg, heat=True, heat_value=h)
            himg, _ = gen.get_heat_frame_fo(img_og, cfg, heat=not use_no_heat, heat_value=h)
        elif cfg.heat_mode == 2:
            img = gen.get_rgb_po(img_og, cfg, heat=True)
            himg, _ = gen.get_heat_frame_po(img_og, cfg, heat=not use_no_heat, heat_value=h)
        elif cfg.heat_mode == 1:
            img = img_og
            himg = gen.get_heat_frame_v1(img_og, heat=not use_no_heat)
        elif cfg.heat_mode == 0:
            img = img_og
            himg = gen.get_heat_frame_v0(img_og, heat=not use_no_heat)
        else:
            raise NotImplementedError("Unsupported heat_mode")
        imgs.append(img); heats.append(himg)
    return imgs, heats

# ---------- single-sample Dreamer posterior via obs_step chain ----------

@torch.no_grad()
def dreamer_posterior_obsstep(cfg, wm, x0, heat0, K=5, use_no_heat=False):
    """
    Build a K-step backward history under zero actions while back-propagating heat,
    render observations for each step, and forward-propagate Dreamer using obs_step.
    Returns posterior feature at the last (t=0) step and margin head output.
    """
    import numpy as np
    import torch
    from generate_data_traj_cont import HeatFrameGenerator, get_frame_eval_pil

    # ---------- helpers ----------
    def backstep_state(xyθv):
        x, y, th, v = xyθv
        dt = cfg.dt
        return np.array([
            x - v*np.cos(th)*dt,
            y - v*np.sin(th)*dt,
            th,  # zero angular accel in the backward model
            v    # zero linear accel in the backward model
        ], dtype=np.float32)

    def inside_obs_xy(x, y):
        dx, dy = x - cfg.obs_x, y - cfg.obs_y
        return (dx*dx + dy*dy) ** 0.5 < cfg.obs_r

    S = (255.0 / 1.1)
    def backstep_heat(h_t, x_prev, y_prev):
        if inside_obs_xy(x_prev, y_prev):
            h_prev = h_t - (cfg.alpha_in / S)
        else:
            h_prev = h_t + (cfg.alpha_out / S)
        return float(np.clip(h_prev, 0.0, 1.0))

    # ---------- build backward history (x_0, h_0), (x_-1, h_-1), ..., (x_-K, h_-K) ----------
    xs_hist = [np.array(x0, dtype=np.float32)]
    hs_hist = [float(heat0)]
    cur_x = xs_hist[0].copy()
    cur_h = hs_hist[0]
    for _ in range(K):
        x_prev = backstep_state(cur_x)
        h_prev = backstep_heat(cur_h, x_prev[0], x_prev[1])
        xs_hist.append(x_prev)
        hs_hist.append(h_prev)
        cur_x, cur_h = x_prev, h_prev

    # chronological order: (x_-K ... x_0)
    xs_seq = xs_hist[::-1]
    hs_seq = hs_hist[::-1]
    T = len(xs_seq)
    B = 1

    # ---------- render sequence ----------
    gen = HeatFrameGenerator(cfg)
    imgs_seq, heats_seq = [], []
    for s, h in zip(xs_seq, hs_seq):
        img_og = get_frame_eval_pil(s, cfg)
        gen._compute_geometry(img_og.shape)
        # produce RGB + heat planes matching your training setup
        if cfg.heat_mode == 3:
            rgb = gen.get_rgb_fo(img_og, cfg, heat=True, heat_value=h)
            heat_img, _ = gen.get_heat_frame_fo(img_og, cfg, heat=(not use_no_heat), heat_value=h)
        elif cfg.heat_mode == 2:
            rgb = gen.get_rgb_po(img_og, cfg, heat=True)
            heat_img, _ = gen.get_heat_frame_po(img_og, cfg, heat=(not use_no_heat), heat_value=h)
        elif cfg.heat_mode == 1:
            rgb = img_og
            heat_img = gen.get_heat_frame_v1(img_og, heat=(not use_no_heat))
        elif cfg.heat_mode == 0:
            rgb = img_og
            heat_img = gen.get_heat_frame_v0(img_og, heat=(not use_no_heat))
        else:
            raise NotImplementedError(f"Unsupported heat_mode {cfg.heat_mode}")

        imgs_seq.append(rgb)        # H,W,3 uint8
        heats_seq.append(heat_img)  # H,W,1 (uint8/float)

    # ---------- forward Dreamer via obs_step ----------
    # zero actions; first element of the chronological sequence is episode start
    actions  = np.zeros((B, 1, 2), dtype=np.float32)
    is_first = np.zeros((B, 1), dtype=bool)

    # initialize prior
    post_t = wm.dynamics.initial(B)
    prior_t = post_t  # prior and post have same structure for t=0 boot

    for t in range(T):
        x, y, th, v = xs_seq[t]
        cos_th = np.cos(th)
        sin_th = np.sin(th)
        if cfg.obs_priv_heat:
            obs_state = np.array([[ [cos_th, sin_th, hs_seq[t]] ]], dtype=np.float32)  # (B,1,3)
        else:
            obs_state = np.array([[ [cos_th, sin_th, v] ]], dtype=np.float32)          # (B,1,3)

        step = {
            "obs_state":  obs_state,
            "image":      np.expand_dims(imgs_seq[t], 0)[None, ...],   # (B,1,H,W,3)
            "heat":       np.expand_dims(heats_seq[t], 0)[None, ...],  # (B,1,H,W,1)
            "action":     actions,                                     # (B,1,2) zeros
            "is_first":   (is_first if t > 0 else np.array([[True]], dtype=bool)),
            "is_terminal": np.zeros((B, 1), dtype=bool),
        }
        step = wm.preprocess(step)
        # encoder expects (B,T,...) -> here T=1; take [:,0] after encoding
        embed_1 = wm.encoder(step)          # (B,1,De)
        embed_t = embed_1[:, 0]             # (B,De)
        act_t   = step["action"][:, 0]      # (B,2)
        first_t = step["is_first"][:, 0]    # (B,)

        post_t, prior_t = wm.dynamics.obs_step(prior_t, embed_t, act_t, first_t)

    feat_T = wm.dynamics.get_feat(post_t)               # (B, D_feat)
    lz_T   = torch.tanh(wm.heads["margin"](feat_T))     # (B, 1?) or (B,)

    return {
        "feat_T":   feat_T,                 # torch.Tensor (B, D_feat)
        "lz_T":     lz_T.squeeze(-1),       # torch.Tensor (B,)
        "post_T":   post_t,                 # dict of tensors
        "xs_seq":   np.stack(xs_seq, axis=0),  # (T, 4)
        "hs_seq":   np.array(hs_seq, dtype=np.float32),  # (T,)
        "heat_T":   float(hs_seq[-1]),
        "T_used":   T,
    }


# ---------- rollout from stacked Dreamer posts (vectorized) ----------

@torch.no_grad()
def rollout_from_posts(cfg, wm, policy, posts, feats, states_xythv, heats0, T=100, heat=True, batch_step=256):
    """
    posts: dict of tensors with leading batch dim B (stack of post_T)
    feats: np.ndarray or torch.Tensor of shape (B, D)
    states_xythv: torch.Tensor (B, 4) with [x,y,theta,v]
    heats0: torch.Tensor (B,)
    Returns: dict(TP,TN,FP,FN), lists of trajectories/failures
    """
    # prediction: min(V, l) > 0
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=cfg.device)
    # Critic V using actor_old head like in your pipeline
    tmp_batch = Batch(obs=feats_t.detach().cpu().numpy(), info=Batch())
    V_vals = policy.critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    if torch.is_tensor(V_vals): V_vals = V_vals.detach().cpu().numpy()
    lz_vals = []
    with torch.no_grad():
        lz_vals = torch.tanh(wm.heads["margin"](feats_t)).detach().cpu().numpy()
    lz_vals = lz_vals.squeeze()
    combined = np.minimum(V_vals, lz_vals)
    vf_binary = combined > 1e-6

    # ensure posts tensors have batch dim B
    B = states_xythv.shape[0]
    results = dict(TP=0, TN=0, FP=0, FN=0)
    trajectories = []
    failures = []
    fp_trajs, fn_trajs = [], []

    # slice-and-roll in chunks to save memory
    keys = list(posts.keys())
    for start in range(0, B, batch_step):
        end = min(start + batch_step, B)
        post_b = {k: v[start:end] for k, v in posts.items()}

        feat_b = feats_t[start:end].clone()
        st_b   = states_xythv[start:end].clone().to(cfg.device)  # (b,4)
        x, y, theta, v = st_b.t()
        heat_vals = heats0[start:end].clone().to(cfg.device)

        # initial failure (heat-threshold)
        if heat:
            failure = heat_vals >= (cfg.heat_threshold - 1e-6)
        else:
            failure = torch.zeros_like(heat_vals, dtype=torch.bool)

        xs_all = [x.detach().cpu().numpy()]
        ys_all = [y.detach().cpu().numpy()]

        for _ in range(T):
            # policy action in latent
            act = policy.actor(feat_b)[0]                 # (b,2)
            post_b = wm.dynamics.img_step(post_b, act)    # forward latent
            feat_b = wm.dynamics.get_feat(post_b).detach()

            angular_acc = act[:, 0]
            linear_acc  = act[:, 1]

            v     += linear_acc * cfg.dt
            v      = torch.clamp(v, 0.0, 1.0)
            x     += v * torch.cos(theta) * cfg.dt
            y     += v * torch.sin(theta) * cfg.dt
            theta += angular_acc * cfg.dt
            theta  = (theta + np.pi) % (2*np.pi) - np.pi

            if heat:
                dist = torch.sqrt((x - cfg.obs_x)**2 + (y - cfg.obs_y)**2)
                inside_obs = dist < cfg.obs_r
                heat_vals = torch.where(
                    inside_obs,
                    heat_vals + cfg.alpha_in  / (255/1.1),
                    heat_vals - cfg.alpha_out / (255/1.1),
                )
                heat_vals = torch.clamp(heat_vals, 0.0, 1.0)
                failure  |= heat_vals >= (cfg.heat_threshold - 1e-6)

            xs_all.append(x.detach().cpu().numpy())
            ys_all.append(y.detach().cpu().numpy())

        xs_all = np.stack(xs_all, axis=1)  # (b, T+1)
        ys_all = np.stack(ys_all, axis=1)

        # record per-sample
        for i in range(xs_all.shape[0]):
            trajectories.append((xs_all[i], ys_all[i]))
            f = bool(failure[i].item())
            failures.append(f)
            pred_safe = bool(vf_binary[start + i])

            if (not f) and pred_safe:      results["TP"] += 1
            elif f and (not pred_safe):    results["TN"] += 1
            elif f and pred_safe:          results["FP"] += 1; fp_trajs.append((xs_all[i], ys_all[i]))
            else:                           results["FN"] += 1; fn_trajs.append((xs_all[i], ys_all[i]))

    return results, np.array(trajectories, dtype=object), np.array(failures, dtype=bool), np.array(fp_trajs, dtype=object), np.array(fn_trajs, dtype=object)

# ---------- main API: sample N points & score ----------

@torch.no_grad()
def eval_random_points(cfg, wm, policy, N=500, K=5, heat0=0.0, use_no_heat=False, rollout_T=100, seed=0):
    """
    Uniformly samples N points over (x,y) domain, theta=0, v~U[0,1].
    Builds K-step backward histories, runs Dreamer obs_step chains,
    predicts with min(V,l) and validates via rollout.
    Returns: dict with counts and rates, plus the raw arrays.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(cfg.x_min, cfg.x_max, size=N).astype(np.float32)
    ys = rng.uniform(cfg.y_min, cfg.y_max, size=N).astype(np.float32)
    thetas = np.zeros(N, dtype=np.float32)  # fixed heading; adjust if desired
    vels   = rng.uniform(0.0, 1.0, size=N).astype(np.float32)

    posts_list = []
    feats_list = []
    lz_list = []
    states_list = []
    heats0_list = []

    for i in range(N):
        x0 = np.array([xs[i], ys[i], thetas[i], vels[i]], dtype=np.float32)
        h0 = np.random.rand()
        out = dreamer_posterior_obsstep(cfg, wm, x0, heat0=h0, K=K, use_no_heat=use_no_heat)
        posts_list.append(out["post_T"])
        feats_list.append(out["feat_T"].squeeze(0).detach().cpu().numpy())
        lz_list.append(out["lz_T"])
        states_list.append(x0)
        heats0_list.append(out["heat_T"])

    feats = np.stack(feats_list, axis=0)           # (N,D)
    states = torch.tensor(np.stack(states_list, 0), dtype=torch.float32)  # (N,4)
    heats0_t = torch.tensor(np.array(heats0_list, dtype=np.float32))      # (N,)

    # stack Dreamer post dicts along batch axis
    keys = list(posts_list[0].keys())
    posts = {k: torch.cat([p[k] for p in posts_list], dim=0) for k in keys}

    # rollout truth & confusion
    results, trajs, fails, fps, fns = rollout_from_posts(
        cfg, wm, policy,
        posts=posts,
        feats=feats,
        states_xythv=states,
        heats0=heats0_t,
        T=rollout_T,
        heat=not use_no_heat,
        batch_step=128,
    )

    TP, TN, FP, FN = results["TP"], results["TN"], results["FP"], results["FN"]
    total = max(1, TP + TN + FP + FN)
    metrics = {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "TPR_recall": TP / max(1, TP + FN),
        "TNR_specificity": TN / max(1, TN + FP),
        "FPR": FP / max(1, FP + TN),
        "FNR": FN / max(1, FN + TP),
        "Accuracy": (TP + TN) / total,
        "Total": total,
    }
    return metrics, dict(trajs=trajs, fails=fails, fps=fps, fns=fns)



# -----------------------------------------------------------------------------
# Latents + evaluation
# -----------------------------------------------------------------------------
def get_latent(config, wm, thetas, vels, heat_values, imgs, heat_imgs, no_heat_imgs, heat_bool=True):
    thetas = np.expand_dims(np.expand_dims(thetas, 1), 1)
    vels = np.expand_dims(np.expand_dims(vels, 1), 1)
    imgs = np.expand_dims(imgs, 1)
    heat_imgs = heat_imgs if heat_bool else no_heat_imgs
    heat_imgs = np.expand_dims(heat_imgs, 1)

    B = thetas.shape[0]
    dummy_acs = np.zeros((B, 1, 2), dtype=np.float32)
    firsts = np.ones((B, 1), dtype=bool)
    lasts = np.zeros((B, 1), dtype=bool)

    cos = np.cos(thetas)
    sin = np.sin(thetas)
    heat_values = np.ones_like(cos) * heat_values
    if not heat_bool:
        heat_values *= 0

    if config.obs_priv_heat:
        states = np.concatenate([cos, sin, heat_values], axis=-1)
    else:
        states = np.concatenate([cos, sin, vels], axis=-1)

    chunk = 128
    embed = []
    for i in range(0, B, chunk):
        s = slice(i, i + chunk)
        d = {'obs_state': states[s], 'image': imgs[s], 'heat': heat_imgs[s],
             'action': dummy_acs[s], 'is_first': firsts[s], 'is_terminal': lasts[s]}
        d = wm.preprocess(d)
        embed.append(wm.encoder(d))
    embed = torch.cat(embed, dim=0)

    d_full = {'obs_state': states, 'image': imgs, 'heat': heat_imgs,
              'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}
    d_full = wm.preprocess(d_full)

    post, _ = wm.dynamics.observe(embed, d_full["action"], d_full["is_first"])
    feat = wm.dynamics.get_feat(post).detach()
    lz = torch.tanh(wm.heads["margin"](feat)).detach()

    return feat.cpu().numpy(), lz.squeeze().cpu().numpy(), post

def evaluate_V(policy, state_batch_feat):
    tmp_obs = np.array(state_batch_feat)
    tmp_batch = Batch(obs=tmp_obs, info=Batch())
    out = policy.critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return out.cpu().detach().numpy().flatten()

@torch.no_grad()
def rollout_dubins(config, wm, policy, lz, feat, post, states, heat_value_init, T=100, rollout_batch_size=100, heat=True):
    EPS = 1e-6
    V_vals = evaluate_V(policy, feat)
    combined = np.minimum(lz, V_vals)
    vf_binary = combined > EPS

    post = {k: v.clone() for k, v in post.items()}
    for k in post:
        if post[k].ndim == 3 and post[k].shape[1] == 1:
            post[k] = post[k].squeeze(1)

    N = states.shape[0]
    trajectories, failures = [], []
    fp_trajs, fn_trajs = [], []
    results = dict(TP=0, TN=0, FP=0, FN=0)

    for start in range(0, N, rollout_batch_size):
        end = min(start + rollout_batch_size, N)
        post_b = {k: v[start:end] for k, v in post.items()}
        feat_b = torch.tensor(feat[start:end], dtype=torch.float32, device=config.device).clone()
        states_b = states[start:end].clone()
        x, y, theta, v = states_b.t()

        heat_vals = torch.full_like(x, heat_value_init)
        failure = heat_vals >= (config.heat_threshold - EPS) if heat else torch.zeros_like(x, dtype=torch.bool)

        xs_all = [x.cpu().numpy()]
        ys_all = [y.cpu().numpy()]

        for _ in range(T):
            act = policy.actor(feat_b)[0]
            post_b = wm.dynamics.img_step(post_b, act)
            feat_b = wm.dynamics.get_feat(post_b).detach()

            angular_acc = act[:, 0]
            linear_acc = act[:, 1]

            v += linear_acc * config.dt
            v = torch.clamp(v, min=0.0, max=1.0)
            x += v * torch.cos(theta) * config.dt
            y += v * torch.sin(theta) * config.dt
            theta += angular_acc * config.dt
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            if heat:
                dist = ((x - config.obs_x)**2 + (y - config.obs_y)**2).sqrt()
                inside = dist < config.obs_r
                heat_vals = torch.where(
                    inside,
                    heat_vals + config.alpha_in / (255 / 1.1),
                    heat_vals - config.alpha_out / (255 / 1.1)
                ).clamp_(0.0, 1.0)
                failure |= heat_vals >= (config.heat_threshold - EPS)

            xs_all.append(x.cpu().numpy())
            ys_all.append(y.cpu().numpy())

        xs_all = np.stack(xs_all, axis=1)
        ys_all = np.stack(ys_all, axis=1)

        for i in range(xs_all.shape[0]):
            trajectories.append((xs_all[i], ys_all[i]))
            f = bool(failure[i])
            failures.append(f)
            pred_safe = bool(vf_binary[start + i])
            if f and pred_safe:
                fp_trajs.append((xs_all[i], ys_all[i]))
            elif (not f) and (not pred_safe):
                fn_trajs.append((xs_all[i], ys_all[i]))

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

    return results, np.array(trajectories, dtype=object), np.array(failures, dtype=bool), np.array(fp_trajs, dtype=object), np.array(fn_trajs, dtype=object)

@torch.no_grad()
def single_rollout(config, wm, policy, initial_conditions, T=100, target=None):
    def fig_to_image(fig, dpi=200):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGB"))
        return img

    gen = HeatFrameGenerator(config)
    trajectories_rgb_obs = []
    trajectories_heat_obs = []

    for ic in initial_conditions:
        device = config.device
        state = torch.tensor(ic[:4], dtype=torch.float32, device=device)
        x, y, theta, vel = state
        omega = torch.tensor(0.0, dtype=torch.float32, device=device)
        dt = torch.tensor(config.dt, dtype=torch.float32, device=device)
        vehicle_heat = torch.tensor(ic[-1], dtype=torch.float32, device=device)

        traj_rgb, traj_heat = [], []

        for _ in range(T):
            img_og = get_frame_eval_pil(state.cpu().numpy(), config)
            gen._compute_geometry(img_og.shape)

            if config.heat_mode == 3:
                img = gen.get_rgb_fo(img_og, config, heat=True, heat_value=vehicle_heat.item())
                heat, _ = gen.get_heat_frame_fo(img_og, config, heat=True, heat_value=vehicle_heat.item())
            elif config.heat_mode == 2:
                img = gen.get_rgb_po(img_og, config, heat=True)
                heat, _ = gen.get_heat_frame_po(img_og, config, heat=True, heat_value=vehicle_heat.item())
            elif config.heat_mode == 1:
                img = img_og
                heat = gen.get_heat_frame_v1(img_og, heat=True)
            elif config.heat_mode == 0:
                img = img_og
                heat = gen.get_heat_frame_v0(img_og, heat=True)
            else:
                raise NotImplementedError("Unsupported heat_mode")

            img = cv2.putText(img, f"Heat: {vehicle_heat.item():.2f}", (5, 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
            ax.imshow(img)    # your rollout frame
            ax.axis("off")
            traj_rgb.append(fig_to_image(fig, dpi=200))
            plt.close(fig)
            traj_heat.append(heat)

            # one-step latent for policy
            feat, lz, post = get_latent(
                config, wm,
                thetas=np.array([theta.item()], dtype=np.float32),
                vels=np.array([vel.item()], dtype=np.float32),
                heat_values=vehicle_heat.item(),
                imgs=np.array([img]),
                heat_imgs=np.array([heat]),
                no_heat_imgs=None,
                heat_bool=True
            )
            feat_t = torch.tensor(feat, dtype=torch.float32, device=device)
            dreamer_action = policy.actor(feat_t.unsqueeze(0))[0][0]

            if target is not None:
                tx, ty = (torch.tensor(target[0], dtype=torch.float32, device=device),
                          torch.tensor(target[1], dtype=torch.float32, device=device))
                desired_heading = torch.atan2(ty - y, tx - x)
                heading_error = (desired_heading - theta + np.pi) % (2 * np.pi) - np.pi
                nominal_turn = torch.clamp(heading_error / dt, -config.turnRate, config.turnRate)
                nominal_accel = torch.tensor(0.0, device=device)
                nominal_action = torch.stack([nominal_turn, nominal_accel])
                value_nominal = policy.critic(feat_t.unsqueeze(0), nominal_action.unsqueeze(0))[0]
                action = nominal_action if value_nominal > 0 else dreamer_action
            else:
                action = dreamer_action

            ang_accel, lin_accel = action[0], action[1]
            omega += ang_accel * dt
            theta += omega * dt
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            vel = torch.clamp(vel + lin_accel * dt, 0.0, 1.0)
            x += vel * torch.cos(theta) * dt
            y += vel * torch.sin(theta) * dt
            state = torch.stack([x, y, theta, vel])

            obstacle_mask = gen._get_mask()
            vehicle_mask = ((img_og[..., 2:3] > 255/2) &
                            (img_og[..., 0:1] < 100) &
                            (img_og[..., 1:2] < 100))
            inside_obs = np.any(vehicle_mask & obstacle_mask)
            if inside_obs:
                vehicle_heat += config.alpha_in / (255 / 1.1)
            else:
                vehicle_heat -= config.alpha_out / (255 / 1.1)
            vehicle_heat = torch.clamp(vehicle_heat, 0.0, 1.0)

        trajectories_rgb_obs.append(traj_rgb)
        trajectories_heat_obs.append(traj_heat)

    return trajectories_rgb_obs, trajectories_heat_obs

def get_eval_plot(config, wm, policy, cache, vels, heat_values, rollout_T=100, boundary_eps=1e-3):
    from itertools import product
    from matplotlib.colors import ListedColormap
    from matplotlib import colors as mcolors

    vel_heat_pairs = list(product(vels, heat_values))
    nrows = 2 if config.include_no_heat_vis else 1
    ncols = len(vel_heat_pairs)
    figsize = (3 * ncols, 6)

    def _make_fig():
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, np.atleast_2d(ax)

    fig_lz, axes_lz = _make_fig()
    fig_lz_bin, axes_lz_bin = _make_fig()
    fig_v, axes_v = _make_fig()
    fig_v_bin, axes_v_bin = _make_fig()
    fig_combined, axes_combined = _make_fig()
    fig_combined_bin, axes_combined_bin = _make_fig()
    fig_rollout, axes_rollout = _make_fig()

    binary_cmap = ListedColormap(["#276fae", "#e6dc22"])

    gt = np.load(f"{config.ground_truth_path}_{config.nx}.npz")
    x_lin = np.linspace(config.x_min, config.x_max, config.nx)
    y_lin = np.linspace(config.y_min, config.y_max, config.ny)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

    plot_list = [(True, "heat")]
    if config.include_no_heat_vis:
        plot_list.append((False, "no_heat"))

    theta = 0.0
    traj_vid_path = None

    for col, (vel, heat_value) in enumerate(vel_heat_pairs):
        idxs, imgs_prev, heat_imgs_prev, no_heat_imgs_prev, thetas_prev, vels_prev, states_lst = cache[(vel, heat_value)]
        states_tensor = torch.stack(states_lst).float().to(config.device)

        for row, (heat_bool, lbl) in enumerate(plot_list):
            feat, lz, post = get_latent(
                config, wm,
                thetas_prev, vels_prev, heat_value,
                imgs_prev, heat_imgs_prev, no_heat_imgs_prev,
                heat_bool=heat_bool
            )
            vals = evaluate_V(policy, feat)
            combined = np.minimum(vals, lz)

            lz_img = lz.reshape(config.nx, config.ny).T
            v_img = vals.reshape(config.nx, config.ny).T
            comb_img = combined.reshape(config.nx, config.ny).T

            lz_bin = (lz_img > 0).astype(float)
            v_bin = (v_img > 0).astype(float)
            comb_bin = (comb_img > 0).astype(float)

            axes_lz[row, col].imshow(lz_img, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1, cmap="seismic")
            axes_v[row, col].imshow(v_img, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1, cmap="viridis")
            axes_combined[row, col].imshow(comb_img, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=-1, vmax=1, cmap="coolwarm")

            axes_lz_bin[row, col].imshow(lz_bin, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=0, vmax=1, cmap=binary_cmap)
            axes_v_bin[row, col].imshow(v_bin, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=0, vmax=1, cmap=binary_cmap)
            axes_combined_bin[row, col].imshow(comb_bin, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", vmin=0, vmax=1, cmap=binary_cmap)

            results, trajectories, failures, fp_trajs, fn_trajs = rollout_dubins(
                config, wm, policy, lz, feat, post, states_tensor,
                heat_value_init=heat_value, T=rollout_T, heat=heat_bool
            )

            ax = axes_rollout[row, col]
            for (xs, ys), is_failure in zip(trajectories, failures):
                base_rgb = mcolors.to_rgba('red' if is_failure else 'green')
                Tpts = xs.shape[0]
                alphas = np.linspace(0.15, 1.0, Tpts)
                colors_rgba = np.tile(base_rgb, (Tpts, 1))
                colors_rgba[:, 3] = alphas
                ax.scatter(xs, ys, s=3, marker='o', color=colors_rgba, linewidths=0)

            for xs, ys in fp_trajs:
                base_rgb = mcolors.to_rgba('dodgerblue')
                Tpts = xs.shape[0]
                alphas = np.linspace(0.05, 0.4, Tpts)
                rgba_arr = np.tile(base_rgb, (Tpts, 1)); rgba_arr[:, 3] = alphas
                ax.scatter(xs, ys, s=2.5, marker='o', color=rgba_arr, linewidths=0, zorder=3)

            for xs, ys in fn_trajs:
                base_rgb = mcolors.to_rgba('magenta')
                Tpts = xs.shape[0]
                alphas = np.linspace(0.05, 0.4, Tpts)
                rgba_arr = np.tile(base_rgb, (Tpts, 1)); rgba_arr[:, 3] = alphas
                ax.scatter(xs, ys, s=2.5, marker='o', color=rgba_arr, linewidths=0, zorder=3)

            key = f"theta_{theta:.4f}_{vel:.4f}_{heat_value:.4f}_rad"
            if key in gt:
                gt_slice = gt[key]
                for axg in [ax, axes_lz[row, col], axes_v[row, col], axes_combined[row, col],
                            axes_lz_bin[row, col], axes_v_bin[row, col], axes_combined_bin[row, col]]:
                    if axg is ax:
                        axg.contour(X, Y, gt_slice, levels=[0], colors="black", linewidths=1.0, zorder=4)
                    else:
                        axg.contour(X, Y, gt_slice, levels=[0], colors="black", linewidths=1.0)

                gt_flat = gt_slice.flatten()
                pred_flat = comb_img.flatten()
                safe_idx = gt_flat >= 1e-6
                unsafe_idx = ~safe_idx
                TP = np.logical_and(pred_flat > 0, safe_idx).sum()
                FN = np.logical_and(pred_flat <= 0, safe_idx).sum()
                FP = np.logical_and(pred_flat > 0, unsafe_idx).sum()
                TN = np.logical_and(pred_flat <= 0, unsafe_idx).sum()
                total = TP + TN + FP + FN
                if total > 0:
                    txt = f"TP:{TP/total:.2f}  TN:{TN/total:.2f}\nFP:{FP/total:.2f}  FN:{FN/total:.2f}"
                    axes_combined[row, col].text(
                        0.02, 0.02, txt, transform=axes_combined[row, col].transAxes,
                        fontsize=7, color="black", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
                    )

            ax.plot([], [], color='dodgerblue', linewidth=1.0, label='FP')
            ax.plot([], [], color='magenta', linewidth=1.0, label='FN')
            ax.plot([], [], color='green', linewidth=1.0, label='Safe')
            ax.plot([], [], color='red', linewidth=1.0, label='Unsafe')
            ax.legend(loc='lower right', fontsize=6, framealpha=0.6)

            ax.add_patch(patches.Circle((config.obs_x, config.obs_y), config.obs_r,
                                        edgecolor='black', facecolor='none',
                                        linestyle='--', linewidth=1.5))
            ax.set_xlim(config.x_min, config.x_max)
            ax.set_ylim(config.y_min, config.y_max)
            ax.set_aspect('equal')
            ax.set_title(f"Rollouts Θ={theta:.2f} H={heat_value:.2f} V={vel:.2f}", fontsize=8)
            ax.axis("off")

            for axg in [axes_lz, axes_lz_bin, axes_v, axes_v_bin, axes_combined, axes_combined_bin]:
                axc = axg[row, col]
                axc.add_patch(patches.Circle((config.obs_x, config.obs_y), config.obs_r,
                                             linewidth=1, edgecolor="red", facecolor="none", linestyle="--"))
                axc.axis("off")
                axc.set_title(f"Θ={theta:.2f}  H={heat_value:.2f}  ({lbl.upper()})", fontsize=8)

            # Example short rollout video from a single IC near origin
            if traj_vid_path is None:
                init_state = np.array([0.0, 0.0, 0.0, 0.0, heat_value], dtype=np.float32)
                traj_rgb, _ = single_rollout(config, wm, policy, [init_state], T=rollout_T)
                import imageio, tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_vid:
                    imageio.mimsave(temp_vid.name, traj_rgb[0], fps=8)
                    temp_vid.flush()
                    traj_vid_path = temp_vid.name

    labels = [
        (axes_lz, "l(x)"),
        (axes_lz_bin, "Binary l(x)"),
        (axes_v, "V(x)"),
        (axes_v_bin, "Binary V(x)"),
        (axes_combined, "min(V,l)"),
        (axes_combined_bin, "Binary min(V,l)"),
        (axes_rollout, "Trajectories"),
    ]
    for ax_arr, base in labels:
        ax_arr[0, 0].set_ylabel(f"{base} (HEAT)")
        if config.include_no_heat_vis:
            ax_arr[1, 0].set_ylabel(f"{base} (NO HEAT)")

    fig_lz.suptitle("Safety Margin l(x)", fontsize=14)
    fig_lz_bin.suptitle("Binary l(x) > 0", fontsize=14)
    fig_v.suptitle("Critic Value V(x)", fontsize=14)
    fig_v_bin.suptitle("Binary V(x) > 0", fontsize=14)
    fig_combined.suptitle("min(V(x), l(x))", fontsize=14)
    fig_combined_bin.suptitle("Binary min(V(x), l(x)) > 0", fontsize=14)
    fig_rollout.suptitle("Trajectories", fontsize=14)

    for fig in [fig_lz, fig_lz_bin, fig_v, fig_v_bin, fig_combined, fig_combined_bin, fig_rollout]:
        fig.tight_layout()

    return fig_lz, fig_lz_bin, fig_v, fig_v_bin, fig_combined, fig_combined_bin, fig_rollout, traj_vid_path

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Env / WM bootstrap ---------------------------------------------------
    env = gymnasium.make(args.task, params=[args])
    args.num_actions = env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
    args.dataset_path = f"{args.dataset_path}_{args.alpha_in}.pkl"

    if args.multimodal:
        env.observation_space_full['image'] = gymnasium.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        env.observation_space_full['heat']  = gymnasium.spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8)
        if args.obs_priv_heat:
            env.observation_space_full['obs_state'] = gymnasium.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            env.observation_space_full['obs_state'] = gymnasium.spaces.Box(
                low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32
            )

    wm = models.WorldModel(env.observation_space_full, env.action_space, 0, args)

    ckpt_path = args.rssm_ckpt_path
    if ckpt_path is None:
        raise ValueError("rssm_ckpt_path must be provided (either via YAML defaults or --rssm_ckpt_path_override).")
    checkpoint = torch.load(ckpt_path, weights_only=True)
    state_dict = {k[14:]: v for k, v in checkpoint['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict); wm.eval()

    # Offline dataset used by env.set_wm just like training
    offline_eps = collections.OrderedDict()
    args.batch_size = 1
    args.batch_length = 5
    tools.fill_expert_dataset_dubins(args, offline_eps)
    dataset = make_dataset(offline_eps, args)
    env.set_wm(wm, dataset, args)

    # --- Build policy nets & load weights -------------------------------------
    # activations
    act_map = dict(ReLU=torch.nn.ReLU, Tanh=torch.nn.Tanh, Sigmoid=torch.nn.Sigmoid, SiLU=torch.nn.SiLU)
    actor_activation = act_map.get(args.actor_activation, torch.nn.ReLU)
    critic_activation = act_map.get(args.critic_activation, torch.nn.ReLU)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0] if hasattr(env.action_space, "high") else 1.0

    if args.critic_net is None:
        raise ValueError("Please provide --critic-net in your configs (e.g., 128 128).")
    critic_net = Net(args.state_shape, args.action_shape, hidden_sizes=args.critic_net,
                     activation=critic_activation, concat=True, device=args.device)
    critic = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj)
    critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=critic_optim, gamma=0.995)

    actor_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
    actor = Actor(actor_net, args.action_shape, max_action=args.max_action, device=args.device).to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=actor_optim, gamma=1.0)

    policy = DDPGPolicy(
        critic, critic_optim, critic_scheduler=critic_scheduler,
        tau=args.tau, gamma=args.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=env.action_space,
        actor=actor, actor_optim=actor_optim, actor_scheduler=actor_scheduler,
        actor_gradient_steps=args.actor_gradient_steps,
    )

    if not args.model_path or not os.path.exists(args.model_path):
        raise FileNotFoundError(f"--model_path missing or not found: {args.model_path}")
    policy_state = torch.load(args.model_path, weights_only=False)
    policy.load_state_dict(policy_state, strict=True)
    policy.eval()

    # --- Cache ----------------------------------------------------------------
    cache = load_cache(args, allow_make=args.make_cache_if_missing, vels=args.vels, heat_values=args.heat_values)

    # --- CM -------------------------------------------------------------------
    metrics, blobs = eval_random_points(
        cfg=args,
        wm=wm,
        policy=policy,
        N=500,          # number of samples
        K=5,            # history length
        heat0=0.0,      # initial heat for all samples (tweak if you want)
        use_no_heat=False,
        rollout_T=args.rollout_T if hasattr(args, "rollout_T") else 100,
        seed=0,
    )

    print("Confusion (N=500):", metrics)
    # blobs["trajs"], blobs["fails"], blobs["fps"], blobs["fns"] are available if you want to visualize


    # --- Plots ----------------------------------------------------------------
    figs = get_eval_plot(args, wm, policy, cache, vels=args.vels, heat_values=args.heat_values, rollout_T=args.rollout_T)
    (fig_lz, fig_lz_bin, fig_v, fig_v_bin, fig_combined, fig_combined_bin, fig_rollout, traj_vid_path) = figs

    # Save outputs
    out = args.output_dir
    fig_lz.savefig(os.path.join(out, "lz_continuous.png"), dpi=200, bbox_inches="tight")
    fig_lz_bin.savefig(os.path.join(out, "lz_binary.png"), dpi=200, bbox_inches="tight")
    fig_v.savefig(os.path.join(out, "v_continuous.png"), dpi=200, bbox_inches="tight")
    fig_v_bin.savefig(os.path.join(out, "v_binary.png"), dpi=200, bbox_inches="tight")
    fig_combined.savefig(os.path.join(out, "min_v_l_continuous.png"), dpi=200, bbox_inches="tight")
    fig_combined_bin.savefig(os.path.join(out, "min_v_l_binary.png"), dpi=200, bbox_inches="tight")
    fig_rollout.savefig(os.path.join(out, "rollout_trajectories.png"), dpi=200, bbox_inches="tight")

    if traj_vid_path and os.path.exists(traj_vid_path):
        # move temp video to output_dir
        dest = os.path.join(out, "rollout_example.mp4")
        try:
            os.replace(traj_vid_path, dest)
        except Exception:
            import shutil
            shutil.copyfile(traj_vid_path, dest)

    print(f"[eval] Saved figures (and video if present) to: {os.path.abspath(out)}")

if __name__ == "__main__":
    main()
