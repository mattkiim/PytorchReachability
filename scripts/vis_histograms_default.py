#!/usr/bin/env python3
"""
Compute histogram of avg(nonzero hot_inner intensity)
at first timestep where brt ≥ 0.3.

Assumes each trajectory group in HDF5 contains:
  - "brt": (T,)
  - "hot_inner": (T, H, W, 1) or (T, H, W)
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def first_crossing_intensity(f, run, threshold=0.3):
    """
    Returns avg pixel intensity (nonzero only) at first timestep
    where brt >= threshold, else None.
    """

    brt = np.array(f[run]["brt"])
    heat = np.array(f[run]["hot_inner"])

    # remove channel if shape is (T, H, W, 1)
    if heat.ndim == 4 and heat.shape[-1] == 1:
        heat = np.squeeze(heat, -1)

    for t, val in enumerate(brt):
        if val <= threshold:
            frame = heat[t]
            nonzero = frame[frame > 0]
            return float(np.mean(nonzero)) if nonzero.size > 0 else 0.0

    # no crossing found
    return None


def run_histogram(datasets, out_dir="eval/histograms", threshold=0.3):
    """
    datasets: dict[str, str]
      name -> dataset path (.h5)
    """
    os.makedirs(out_dir, exist_ok=True)

    for name, path in datasets.items():
        print(f"\n=== Processing {name} ===")
        results = []

        with h5py.File(path, "r") as f:
            for run in f.keys():
                val = first_crossing_intensity(f, run, threshold=threshold)
                if val is not None:
                    results.append(val)

        vals = np.asarray(results, dtype=float)

        # --- stats ---
        mu, sigma = np.mean(vals), np.std(vals)
        print(f"[{name}] μ={mu:.4f}, σ={sigma:.4f}")

        # --- plotting ---
        bins = 100
        rng = (-0.1, 1.0)
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=bins, range=rng, density=True, color="C0",
                 alpha=0.5, edgecolor="white", label="Histogram")

        plt.xlabel("Avg pixel intensity at first brt ≥ 0.3")
        plt.ylabel("Density")
        plt.title(f"Hot-Inner Avg Intensity — {name}")
        plt.legend(frameon=False)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"histogram_brt03_{name}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"💾 Saved {out_path}")


if __name__ == "__main__":
    datasets = {
        "rgb": "/data/mattkiim/eval_rgb_v2.h5",
        "mm":  "/data/mattkiim/eval_multimodal_v2.h5",
    }

    run_histogram(datasets)
