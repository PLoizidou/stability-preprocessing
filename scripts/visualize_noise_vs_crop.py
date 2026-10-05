#!/usr/bin/env python3
"""
Visualize where the Miniscope ring-noise artifact (see detect_noise_frames.py)
sits spatially within the FOV, relative to a given animal's crop rectangle
from the CROPS dict in preproc_caiman_final.py. Helps answer "does the
existing crop already exclude most of the noise, or does the noise land
inside the region CaImAn actually sees?"

Computes a per-pixel high-frequency-energy map (mean |Laplacian|, the same
quantity detect_noise_frames.py reduces to a single number per frame) over a
sample of clean frames and a sample of artifact-flagged frames, and plots
their difference as a heatmap alongside example clean/noisy frames, all with
the crop rectangle overlaid.

Usage:
    python visualize_noise_vs_crop.py \
        /media/toor/Seagate3/Mouse944/RAW_DATA/.../Miniscope<ts>.avi \
        /path/to/Miniscope<ts>_noise_frames.csv \
        --crop 200 550 100 500 \
        --output results/mouse944_noise_vs_crop.png
"""
import argparse
import csv
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def resolve_crop(crop, h, w):
    row_start, row_stop, col_start, col_stop = crop
    row_stop_abs = h + row_stop if row_stop < 0 else row_stop
    col_stop_abs = w + col_stop if col_stop < 0 else col_stop
    return row_start, row_stop_abs, col_start, col_stop_abs


def load_noise_csv(path):
    clean_idx, bad_idx = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_idx"])
            (bad_idx if row["is_artifact"] == "1" else clean_idx).append(idx)
    return clean_idx, bad_idx


def sample(indices, n):
    if len(indices) <= n:
        return indices
    step = len(indices) / n
    return [indices[int(i * step)] for i in range(n)]


def mean_abs_laplacian_map(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    acc = None
    n = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        acc = lap if acc is None else acc + lap
        n += 1
    cap.release()
    return acc / max(n, 1), n


def grab_frame(video_path, idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path", help="Path to a Miniscope*.avi video")
    parser.add_argument("noise_csv", help="Path to that video's *_noise_frames.csv from detect_noise_frames.py")
    parser.add_argument("--crop", type=int, nargs=4, required=True, metavar=("ROW_START", "ROW_STOP", "COL_START", "COL_STOP"),
                         help="Crop to overlay, same convention as the CROPS dict in preproc_caiman_final.py")
    parser.add_argument("--n-sample", type=int, default=150, help="Frames to sample per group for the energy maps (default: 150)")
    parser.add_argument("--output", default=None, help="Output PNG path (default: results/<video_stem>_noise_vs_crop.png)")
    parser.add_argument("--title", default=None, help="Optional custom title prefix")
    args = parser.parse_args()

    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    output_path = args.output or os.path.join("results", f"{video_stem}_noise_vs_crop.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    clean_idx, bad_idx = load_noise_csv(args.noise_csv)
    if not bad_idx:
        raise ValueError(f"No artifact frames found in {args.noise_csv} -- nothing to visualize")

    clean_sample = sample(clean_idx, args.n_sample)
    bad_sample = sample(bad_idx, args.n_sample)

    print(f"Computing mean |Laplacian| map over {len(clean_sample)} clean frames...")
    clean_map, n_clean = mean_abs_laplacian_map(args.video_path, clean_sample)
    print(f"Computing mean |Laplacian| map over {len(bad_sample)} artifact frames...")
    bad_map, n_bad = mean_abs_laplacian_map(args.video_path, bad_sample)
    diff_map = bad_map - clean_map

    h, w = diff_map.shape
    row_start, row_stop, col_start, col_stop = resolve_crop(tuple(args.crop), h, w)

    crop_mask = np.zeros((h, w), dtype=bool)
    crop_mask[row_start:row_stop, col_start:col_stop] = True
    total_noise_energy = diff_map.clip(min=0).sum()
    in_crop_energy = diff_map.clip(min=0)[crop_mask].sum()
    pct_in_crop = 100 * in_crop_energy / total_noise_energy if total_noise_energy > 0 else float("nan")

    example_clean = grab_frame(args.video_path, clean_idx[len(clean_idx) // 3])
    example_bad = grab_frame(args.video_path, bad_idx[len(bad_idx) // 2])

    title_prefix = args.title or video_stem
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    def draw_crop_rect(ax, color="red"):
        rect = patches.Rectangle(
            (col_start, row_start), col_stop - col_start, row_stop - row_start,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

    axes[0].imshow(example_clean, cmap="gray")
    axes[0].set_title(f"Example clean frame (idx {clean_idx[len(clean_idx)//3]})")
    draw_crop_rect(axes[0])
    axes[0].axis("off")

    axes[1].imshow(example_bad, cmap="gray")
    axes[1].set_title(f"Example artifact frame (idx {bad_idx[len(bad_idx)//2]})")
    draw_crop_rect(axes[1])
    axes[1].axis("off")

    im = axes[2].imshow(diff_map, cmap="inferno")
    axes[2].set_title(f"Noise energy map (artifact - clean)\n{pct_in_crop:.1f}% of noise energy falls INSIDE crop")
    draw_crop_rect(axes[2], color="cyan")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{title_prefix} — noise location vs. crop {tuple(args.crop)}  "
                 f"(n_clean={n_clean}, n_artifact={n_bad})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    print(f"Fraction of noise energy inside crop rectangle: {pct_in_crop:.1f}%")


if __name__ == "__main__":
    main()
