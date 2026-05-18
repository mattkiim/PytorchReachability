"""
Compare per-sample predictions from two models and report metrics on the
disagree subset.

Usage:
    python scripts/compare_preds.py \
        --mm_dir  eval_preds/mm  \
        --rgb_dir eval_preds/rgb

Each dir must contain preds.npy and gt.npy produced by eval.py --save_preds.
gt.npy should be identical between the two dirs (same eval windows); the
script asserts this.
"""

import argparse
import numpy as np


def confusion_metrics(pred, gt):
    TP = np.sum(~pred & ~gt)   # predicted safe, actually safe
    TN = np.sum(pred & gt)     # predicted unsafe, actually unsafe
    FP = np.sum(~pred & gt)    # predicted safe, actually unsafe
    FN = np.sum(pred & ~gt)    # predicted unsafe, actually safe
    N  = max(len(pred), 1)
    return dict(TP=int(TP), TN=int(TN), FP=int(FP), FN=int(FN), total=N,
                tpr=TP/N, tnr=TN/N, fpr=FP/N, fnr=FN/N)


def print_metrics(label, m):
    print(f"  {label}: TS={m['tpr']:.4f}  TU={m['tnr']:.4f}  "
          f"FS={m['fpr']:.4f}  FU={m['fnr']:.4f}  "
          f"(n={m['total']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mm_dir",  required=True, help="Dir with MM model preds.npy + gt.npy")
    parser.add_argument("--rgb_dir", required=True, help="Dir with RGB model preds.npy + gt.npy")
    args = parser.parse_args()

    preds_mm  = np.load(f"{args.mm_dir}/preds.npy")
    gt_mm     = np.load(f"{args.mm_dir}/gt.npy")
    preds_rgb = np.load(f"{args.rgb_dir}/preds.npy")
    gt_rgb    = np.load(f"{args.rgb_dir}/gt.npy")

    assert np.array_equal(gt_mm, gt_rgb), (
        "gt arrays differ — make sure both models ran on the same eval dataset/windows"
    )
    gt = gt_mm.astype(bool)
    preds_mm  = preds_mm.astype(bool)
    preds_rgb = preds_rgb.astype(bool)

    disagree = preds_mm != preds_rgb
    n_total    = len(gt)
    n_disagree = int(disagree.sum())

    print(f"\nTotal samples : {n_total}")
    print(f"Disagree      : {n_disagree} ({100*n_disagree/n_total:.1f}%)")
    print(f"  MM predicts unsafe, RGB predicts safe : "
          f"{int(np.sum(preds_mm & ~preds_rgb))}")
    print(f"  RGB predicts unsafe, MM predicts safe : "
          f"{int(np.sum(~preds_mm & preds_rgb))}")

    print("\n--- Overall ---")
    print_metrics("MM ", confusion_metrics(preds_mm,  gt))
    print_metrics("RGB", confusion_metrics(preds_rgb, gt))

    if n_disagree > 0:
        print("\n--- Disagree subset ---")
        print_metrics("MM ", confusion_metrics(preds_mm[disagree],  gt[disagree]))
        print_metrics("RGB", confusion_metrics(preds_rgb[disagree], gt[disagree]))

        print("\n--- Disagree subset: ground truth breakdown ---")
        gt_d = gt[disagree]
        print(f"  Actually safe   (gt=0): {int(np.sum(~gt_d))} ({100*np.mean(~gt_d):.1f}%)")
        print(f"  Actually unsafe (gt=1): {int(np.sum(gt_d))}  ({100*np.mean(gt_d):.1f}%)")
    else:
        print("\nModels agree on every sample — no disagree subset.")


if __name__ == "__main__":
    main()
