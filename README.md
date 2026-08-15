# How Well Do Latent World Models Understand Partially Observable Safety Constraints?

**Paper:** [arXiv 2510.06492](https://arxiv.org/abs/2510.06492) — Matthew Kim, Kensuke Nakamura, Andrea Bajcsy (CoRL 2026)

This repository contains the Dubins car simulation code used to study how partial observability degrades latent-space safe control. We identify two failure modes — **estimation gaps** (safety state is unobservable) and **prediction gaps** (failures can't be anticipated) — and demonstrate mitigations using multimodal privileged supervision and conformal risk calibration.

---

## Setup

Requires Python 3.12 and conda.

```bash
git clone https://github.com/mattkiim/PytorchReachability
cd PytorchReachability
pip install -e .
conda install -c conda-forge ffmpeg
```

Dependencies: `torch==2.4.0`, `gymnasium==0.28.1`, `numpy==1.26.4`, `ruamel-yaml==0.17.40` (see `setup.py` for full list).

---

## Simulation Pipeline

The Dubins car environment simulates a vehicle navigating around a circular obstacle. The vehicle accumulates "heat" (a safety-relevant quantity) while inside the unsafe region. Two observability conditions are studied:

- **PO (partial observability):** heat is hidden from the RGB observation while the vehicle is inside the obstacle — an *estimation gap*
- **FO (full observability):** heat is always visible in the observation — the upper-bound baseline

The pipeline has four steps:

### 1. Generate Data

```bash
python scripts/generate_data_traj_cont.py --config_path configs/configs_po.yaml   # PO data
python scripts/generate_data_traj_cont.py --config_path configs/configs_fo.yaml   # FO data
```

This produces a `.pkl` dataset at the path specified by `dataset_path` in the config.

### 2. Train World Model

```bash
python scripts/dreamer_offline.py --config_path configs/configs_po.yaml   # PO world model
python scripts/dreamer_offline.py --config_path configs/configs_fo.yaml   # FO world model
```

Trains a DreamerV3-style world model on the offline dataset. Checkpoints are saved to `logdir`.

### 3. Train Safety Filter

```bash
python scripts/run_training_ddpg-wm.py --config_path configs/configs_po.yaml   # PO safety filter
python scripts/run_training_ddpg-wm.py --config_path configs/configs_fo.yaml   # FO safety filter
```

Trains a least-restrictive safety filter (DDPG) in the world model's latent space.

### 4. Evaluate

```bash
python scripts/eval_hj.py --config_path configs/configs_po.yaml   # main safety evaluation
```

Produces safety visualizations and metrics over the state space.

---

## Diagnostics

### Mutual Information (Section 4, Table 1)

Quantifies how much safety-relevant information is encoded in the latent state:

```bash
python scripts/eval_world_model.py --config_path configs/configs_po_info_theory.yaml
python scripts/dreamer_info_theory.py
```

### Open-Loop Rollout Prediction (Section 4, Table 2)

Evaluates whether the world model can predict downstream safety outcomes:

```bash
python scripts/eval_boundary.py --config_path configs/configs_po.yaml
```

---

## Configs

Each config corresponds to a specific experiment from the paper:

| Config | Experiment |
|---|---|
| `configs_po.yaml` | PO main experiment (estimation gap) |
| `configs_fo.yaml` | FO baseline (full observability); pass `--obs_priv_heat True` to enable privileged heat supervision (Section 5.1) |
| `configs_po_no_heat.yaml` | Ablation: no heat channel in observation |
| `configs_po_info_theory.yaml` | MI diagnostic (Section 4), smaller dataset |

Key parameters to set per run:
- `dataset_path` — path to generated trajectory data
- `logdir` — where world model checkpoints are saved
- `rssm_ckpt_path` — world model checkpoint used for safety filter training and eval
- `device` — CUDA device (default `cuda:0`)

---

## Vanilla Reachability (No World Model)

To run HJ reachability directly in state space (no world model, SAC-based):

```bash
python scripts/run_training_sac_RA_nodist.py \
  --control-net 512 512 512 512 \
  --critic-net 512 512 512 512 \
  --epoch 1 --total-episodes 80
```

---

## Repository Structure

```
configs/          experiment configs (one per paper experiment)
dreamerv3-torch/  world model (DreamerV3 implementation)
PyHJ/             reachability RL library and Dubins car environments
scripts/
  generate_data_traj_cont.py        data generation (main)
  generate_data_traj_cont_boundary.py  data generation near safe set boundary
  dreamer_offline.py                world model training
  run_training_ddpg-wm.py           latent safety filter training
  eval_hj.py                        safety filter evaluation
  eval_world_model.py               world model / MI diagnostic evaluation
  eval_boundary.py                  boundary evaluation for prediction gap analysis
  dreamer_info_theory.py            mutual information computation
  run_training_sac_RA_nodist.py     vanilla reachability (no world model)
  run_training_ddpg_franka-wm.py    hardware (Franka) deployment
```

---

## Citation

```bibtex
@inproceedings{kim2026latent,
  title={How Well Do Latent World Models Understand Partially Observable Safety Constraints?},
  author={Kim, Matthew and Nakamura, Kensuke and Bajcsy, Andrea},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```
