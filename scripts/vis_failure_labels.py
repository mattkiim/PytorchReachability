#!/usr/bin/env python3
"""
Compare two failure labeling schemes on hot_inner frames, output as a video:
  1. Average over all pixels (threshold > 0.1)
  2. Average over nonzero pixels (threshold > 0.7)

For each trajectory in an HDF5 dataset:
  - Computes both label sequences
  - Reports % agreement
  - Saves a comparison video (MP4) showing:
      Row 0 = camera_0 (RGB), camera_2 (IR), hot_inner
      Row 1 = failure labels over time
"""

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio.v2 as iio
from io import BytesIO

failure_avg_2_threshold = 0.75

# ---------------- config ----------------
DATASET_PATH = "/data/mattkiim/merged_5hz.h5"   # change if needed
OUT_DIR = f"failure_label_videos/{failure_avg_2_threshold}"
FPS = 10
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- main ----------------
def main():
    with h5py.File(DATASET_PATH, "r") as f:
        for i, run in enumerate(f, start=1):
            if not all(k in f[run] for k in ["camera_0", "camera_2", "hot_inner"]):
                print(f"[{run}] missing camera or hot_inner, skipping")
                continue

            cam0       = f[run]["camera_0"][:]      # RGB
            cam2       = f[run]["camera_2"][:]      # IR
            hot_inner  = np.squeeze(f[run]["hot_inner"][:], -1)  # HxW
            T = hot_inner.shape[0]

            # --- Failure labels: Method 1 (avg over all pixels)
            failure_avg_1 = []
            for t in range(T):
                frame = hot_inner[t]
                heat_avg = np.mean(frame)
                failure_avg_1.append(heat_avg > 0.1)
            failure_avg_1 = np.asarray(failure_avg_1, dtype=np.float32)

            # --- Failure labels: Method 2 (avg over nonzero pixels)
            failure_avg_2 = []
            for t in range(T):
                frame = hot_inner[t]
                nonzero = frame[frame > 0]
                heat_avg = np.mean(nonzero) if nonzero.size > 0 else 0.0
                failure_avg_2.append(heat_avg > failure_avg_2_threshold)
            failure_avg_2 = np.asarray(failure_avg_2, dtype=np.float32)

            # --- Compare
            agreement = np.mean(failure_avg_1 == failure_avg_2)
            print(f"[{run}] agreement: {agreement*100:.2f}% "
                  f"({np.sum(failure_avg_1 == failure_avg_2)}/{T} frames)")

            # --- Save video
            out_path = pathlib.Path(OUT_DIR) / f"{run}_failure_compare.mp4"
            with iio.get_writer(out_path, fps=FPS, codec="libx264") as writer:
                x_full = np.arange(T)

                for t in range(T):
                    fig = plt.figure(figsize=(12, 6))
                    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

                    # Row 0: original observations
                    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(cam0[t]); ax0.axis("off"); ax0.set_title("Camera 0 (RGB)")
                    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(cam2[t][..., :1], cmap="plasma"); ax1.axis("off"); ax1.set_title("Camera 2 (IR)")
                    ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(hot_inner[t], cmap="plasma", vmin=0, vmax=1); ax2.axis("off"); ax2.set_title("Hot-Inner")

                    # Row 1: failure label comparison
                    axf = fig.add_subplot(gs[1, :])
                    axf.step(x_full[:t+1], failure_avg_1[:t+1], where="post", label="Average-all", color="C0")
                    axf.step(x_full[:t+1], failure_avg_2[:t+1], where="post", label="Average-nonzero", color="C1")
                    axf.set_ylim(-0.1, 1.1)
                    axf.set_xlim(0, T)
                    axf.set_xlabel("Frame index")
                    axf.set_ylabel("Failure label (0/1)")
                    axf.set_title(f"Failure Label Comparison — {run}\nAgreement: {agreement*100:.1f}%")
                    axf.legend()
                    axf.grid(True)

                    fig.tight_layout()
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    writer.append_data(iio.imread(buf))
                    plt.close(fig)

            print(f"[{run}] wrote {out_path}")

if __name__ == "__main__":
    main()
