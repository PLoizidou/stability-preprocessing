#!/usr/bin/env python3
"""
Sample N evenly-spaced frames from a Miniscope video and compute mean/max/std
projections to help pick a per-mouse FOV crop for the CROPS dict in
preproc_caiman_final.py. Optionally overlay a crop rectangle for review.

Usage:
    python sample_frames_for_crop.py /path/to/Miniscope<ts>.avi
    python sample_frames_for_crop.py /path/to/Miniscope<ts>.avi \
        --crop 200 550 100 500 \
        --output /path/to/results/mouseXXX_crop_check.png
"""
import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def sample_projections(video_path: str, n_sample: int):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_sample, total_frames)
    idxs = np.linspace(0, total_frames - 1, n, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
    cap.release()

    stack = np.stack(frames, axis=0)
    return stack, total_frames


def resolve_crop(crop, h, w):
    row_start, row_stop, col_start, col_stop = crop
    row_stop_abs = h + row_stop if row_stop < 0 else row_stop
    col_stop_abs = w + col_stop if col_stop < 0 else col_stop
    return row_start, row_stop_abs, col_start, col_stop_abs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path", help="Path to a Miniscope*.avi video")
    parser.add_argument("--n-frames", type=int, default=1000, help="Number of evenly-spaced frames to sample (default: 1000)")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("ROW_START", "ROW_STOP", "COL_START", "COL_STOP"),
                         help="Optional crop to overlay as a red rectangle, same (row_start, row_stop, col_start, col_stop) "
                              "convention as the CROPS dict in preproc_caiman_final.py (negative stop = from end)")
    parser.add_argument("--output", default=None, help="Output PNG path (default: <video_dir>/<video_stem>_crop_check.png)")
    parser.add_argument("--title", default=None, help="Optional custom title prefix for the figure")
    args = parser.parse_args()

    stack, total_frames = sample_projections(args.video_path, args.n_frames)
    h, w = stack.shape[1:]
    print(f"Sampled {stack.shape[0]} frames out of {total_frames} total, shape={h}x{w}")

    mean_img = stack.mean(axis=0)
    max_img = stack.max(axis=0)
    std_img = stack.std(axis=0)

    crop_abs = None
    if args.crop:
        crop_abs = resolve_crop(args.crop, h, w)
        print(f"Crop resolved on this {h}x{w} frame: rows {crop_abs[0]}:{crop_abs[1]}, cols {crop_abs[2]}:{crop_abs[3]}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [mean_img, max_img, std_img], ["Mean projection", "Max projection", "Std projection"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{title} (n={stack.shape[0]} frames)")
        ax.set_xticks(np.arange(0, img.shape[1], 50))
        ax.set_yticks(np.arange(0, img.shape[0], 50))
        ax.grid(color="yellow", alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=7)
        if crop_abs:
            row_start, row_stop_abs, col_start, col_stop_abs = crop_abs
            rect = patches.Rectangle(
                (col_start, row_start), col_stop_abs - col_start, row_stop_abs - row_start,
                linewidth=1.5, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)

    default_title = os.path.basename(args.video_path)
    title_prefix = args.title or default_title
    suptitle = f"{title_prefix} ({h}x{w} px)"
    if crop_abs:
        suptitle += f", red box = crop {tuple(args.crop)}"
    plt.suptitle(suptitle)
    plt.tight_layout()

    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.video_path))[0]
        output = os.path.join(os.path.dirname(args.video_path), f"{stem}_crop_check.png")
    plt.savefig(output, dpi=130)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
