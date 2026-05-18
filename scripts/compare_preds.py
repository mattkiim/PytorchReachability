"""
Compare per-sample predictions from two models across open rollout, closed
rollout, and latent state eval modes.

Usage:
    python scripts/compare_preds.py --mm_dir eval_preds/mm --rgb_dir eval_preds/rgb
"""

import argparse
import numpy as np


def confusion_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    TP = np.sum(~pred & ~gt)
    TN = np.sum( pred &  gt)
    FP = np.sum(~pred &  gt)
    FN = np.sum( pred & ~gt)
    N  = max(len(pred), 1)
    return dict(TP=int(TP), TN=int(TN), FP=int(FP), FN=int(FN), n=N,
                tpr=TP/N, tnr=TN/N, fpr=FP/N, fnr=FN/N)


def print_metrics(label, m):
    print(f"  {label:6s}  TS={m['tpr']:.4f}  TU={m['tnr']:.4f}  "
          f"FS={m['fpr']:.4f}  FU={m['fnr']:.4f}  (n={m['n']})")


def analyze_mode(name, preds_mm, gt_mm, preds_rgb, gt_rgb):
    assert np.array_equal(gt_mm, gt_rgb), \
        f"[{name}] gt arrays differ — both models must use the same eval windows"
    gt        = gt_mm.astype(bool)
    preds_mm  = preds_mm.astype(bool)
    preds_rgb = preds_rgb.astype(bool)

    disagree   = preds_mm != preds_rgb
    n_total    = len(gt)
    n_disagree = int(disagree.sum())

    print(f"\n{'='*10} {name} {'='*10}")
    print(f"  Total: {n_total}  |  Disagree: {n_disagree} ({100*n_disagree/n_total:.1f}%)")
    if n_disagree:
        print(f"    MM unsafe / RGB safe : {int(np.sum( preds_mm & ~preds_rgb))}")
        print(f"    RGB unsafe / MM safe : {int(np.sum(~preds_mm &  preds_rgb))}")

    print("  --- Overall ---")
    print_metrics("MM",  confusion_metrics(preds_mm,  gt))
    print_metrics("RGB", confusion_metrics(preds_rgb, gt))

    if n_disagree:
        print("  --- Disagree subset ---")
        print_metrics("MM",  confusion_metrics(preds_mm[disagree],  gt[disagree]))
        print_metrics("RGB", confusion_metrics(preds_rgb[disagree], gt[disagree]))
        gt_d = gt[disagree]
        print(f"  GT in disagree: safe={int(np.sum(~gt_d))} ({100*np.mean(~gt_d):.1f}%)  "
              f"unsafe={int(np.sum(gt_d))} ({100*np.mean(gt_d):.1f}%)")
    else:
        print("  Models agree on every sample.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mm_dir",  required=True)
    parser.add_argument("--rgb_dir", required=True)
    args = parser.parse_args()

    modes = [
        ("Open rollout (warm=5)",  "open_preds.npy",        "open_gt.npy"),
        ("Open rollout (warm=3)",  "open_warm3_preds.npy",  "open_warm3_gt.npy"),
        ("Closed rollout",         "closed_preds.npy",      "closed_gt.npy"),
        ("Latent state",           "latent_preds.npy",      "latent_gt.npy"),
    ]

    for name, preds_file, gt_file in modes:
        try:
            preds_mm  = np.load(f"{args.mm_dir}/{preds_file}")
            gt_mm     = np.load(f"{args.mm_dir}/{gt_file}")
            preds_rgb = np.load(f"{args.rgb_dir}/{preds_file}")
            gt_rgb    = np.load(f"{args.rgb_dir}/{gt_file}")
        except FileNotFoundError as e:
            print(f"\n[{name}] Skipping — file not found: {e}")
            continue
        analyze_mode(name, preds_mm, gt_mm, preds_rgb, gt_rgb)


if __name__ == "__main__":
    main()
