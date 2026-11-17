# How Well Do Latent World Models Understand Partially Observable Safety Constraints?

**Paper:** [arXiv 2510.06492](https://arxiv.org/abs/2510.06492) — Matthew Kim, Kensuke Nakamura, Andrea Bajcsy (CoRL 2026)

This repository contains the Thermal Unicycle simulation code (paper Appendix 8.2) used to study how partial observability degrades latent-space safe control. We identify two failure modes — **estimation gaps** (safety state is unobservable) and **prediction gaps** (failures can't be anticipated) — and demonstrate mitigations using multimodal privileged supervision and conformal risk calibration.

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

Training logs to Weights & Biases. To keep logs local without signing in, run `export WANDB_MODE=offline` before training.

---

## Simulation Pipeline

The Thermal Unicycle environment simulates a vehicle that can drive into a circular hot region. The vehicle accumulates heat (the safety-relevant quantity) while inside the region and is unsafe when heat exceeds 0.8. Each trajectory records paired RGB and IR observations. RGB hides the current heat while the vehicle is inside the region (an *estimation gap*); IR shows it directly.

The paper (Appendix 8.2, Figure 5, Tables 5–6) compares two world models trained on the same dataset:

- **RGB+IR (WM<sub>MM</sub>):** `configs_po.yaml`
- **RGB-only (WM<sub>RGB</sub>):** `configs_po_no_heat.yaml` (same data; the IR channel is withheld from the model)

`configs_fo.yaml` is an additional full-observability setting (heat always visible in RGB) that is not reported in the paper.

The pipeline has four steps:

### 1. Generate Data

```bash
python scripts/generate_data_traj_cont.py --config_path configs/configs_po.yaml   # shared RGB+IR / RGB-only data
python scripts/generate_data_traj_cont.py --config_path configs/configs_fo.yaml   # FO data
```

This produces a dataset at `<dataset_path>_<heat_rate>.pkl` (for example, `train_data/po_6.pkl`). Ensure the parent directory exists and is writable.

### 2. Train World Model

```bash
python scripts/dreamer_offline.py --config_path configs/configs_po.yaml          # RGB+IR world model
python scripts/dreamer_offline.py --config_path configs/configs_po_no_heat.yaml   # RGB-only world model
python scripts/dreamer_offline.py --config_path configs/configs_fo.yaml   # FO world model
```

Trains a DreamerV3-style world model on the offline dataset. Checkpoints are saved to `logdir`.

After the safety-head optimizer fix, retrain both RGB and MM world models in fresh
log directories, then train fresh safety policies against those checkpoints.
Earlier WM training excluded the margin and continuation head parameters from
the optimizer. Policy training also used a different acceleration scale from
evaluation; both now use the dataset's action units. Existing checkpoints are
not updated by these source changes, and already-running jobs must be restarted
to use them.

### 3. Train Safety Filter

```bash
python scripts/run_training_ddpg-wm.py --config_path configs/configs_po.yaml          # RGB+IR safety filter
python scripts/run_training_ddpg-wm.py --config_path configs/configs_po_no_heat.yaml   # RGB-only safety filter
python scripts/run_training_ddpg-wm.py --config_path configs/configs_fo.yaml   # FO safety filter
```

Trains a least-restrictive safety filter (DDPG) in the world model's latent space.

### 4. Evaluate

```bash
python scripts/eval_hj.py --config_path configs/configs_po.yaml \
  --model_path path/to/epoch_id_16/policy.pth
```

Produces safety visualizations and metrics over the state space.

Replace `path/to/epoch_id_16/policy.pth` with a checkpoint from the safety-filter training log directory. Use the matching config and world-model checkpoint; the script does not select a policy automatically. If the HJ cache is missing, add `--make_cache_if_missing`.

---

## Diagnostics

### World-model visual evaluation

Export an interactive HTML viewer, GIFs and contact sheets showing actual frames, reconstructions, open-loop predictions and absolute RGB errors:

```bash
python scripts/visualize_wm.py \
  --config_path configs/configs_po.yaml \
  --checkpoint logs/po/rssm_ckpt.pt \
  --dataset train_data/po_6.pkl \
  --modality mm --num_train_trajs 3800 \
  --history 5 --horizon 16 \
  --output_dir eval/po_wm_visual
```

Open `eval/po_wm_visual/index.html` in a browser. Use `--modality rgb` for a checkpoint trained with `--no_heat True`. Set `--num_train_trajs` to the split used during training; the viewer selects the first three sufficiently long held-out trajectories. It displays the recorded checkpoint step and uses the same action alignment as the training video diagnostic. These qualitative examples do not establish safety performance.

### Mutual Information (Section 4; simulation results in Appendix 8.2, Table 5)

Quantifies how much safety-relevant information is encoded in the latent state:

```bash
python scripts/eval_world_model.py --config_path configs/configs_po_info_theory.yaml
python scripts/dreamer_info_theory.py
```

### Safety Classification and Open-Loop Prediction (Appendix 8.2, Table 6)

Evaluates whether the world model can predict downstream safety outcomes:

```bash
python scripts/eval_boundary.py --config_path configs/configs_po.yaml
```

---

## Configs

Each config corresponds to a specific experiment from the paper:

| Config | Experiment |
|---|---|
| `configs_po.yaml` | RGB+IR world model (WM<sub>MM</sub>) |
| `configs_fo.yaml` | Full-observability setting, not reported in the paper; pass `--obs_priv_heat True` to enable privileged heat supervision |
| `configs_po_no_heat.yaml` | RGB-only world model (WM<sub>RGB</sub>); shares the `configs_po.yaml` dataset |
| `configs_po_info_theory.yaml` | MI diagnostic (Section 4, Table 5), smaller dataset |

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
PyHJ/             reachability RL library and Thermal Unicycle (Dubins) environments
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
