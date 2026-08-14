#!/bin/bash
set -e

# WM_MM (HW)
python scripts/eval.py \
  --config_path configs/configs_hw_merged_5hz_fast_avg_norm.yaml \
  --configs defaults \
  --ckpt_path /data/mattkiim/runs/merged_5hz_fast_avg_norm/dreamer.pt \
  --save_preds eval_preds/mm

# WM_RGB (HW)
python scripts/eval.py \
  --config_path configs/configs_hw_merged_5hz_fast_avg_rgb_only_norm.yaml \
  --configs defaults \
  --ckpt_path /data/mattkiim/runs/merged_5hz_fast_avg_rgb_only_norm/dreamer.pt \
  --save_preds eval_preds/rgb

# Compare
python scripts/compare_preds.py --mm_dir eval_preds/mm --rgb_dir eval_preds/rgb
