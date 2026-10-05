#!/usr/bin/env python3
"""
Detect frames corrupted by the "ring" noise artifact seen in some Miniscope
recordings (a bright speckled ring overlaid on the FOV, usually near the end
of a recording — e.g. a failing/loose connector). Flags frames using
Laplacian variance (high-frequency energy), which jumps by ~30-60x on
affected frames relative to clean ones since the speckle is high-frequency
salt-and-pepper noise that a normal calcium-imaging frame doesn't have.

Writes a per-frame CSV (frame_idx, time_s, lap_var, is_artifact) and prints
the contiguous bad frame ranges found, so they can be excluded from
downstream preprocessing (e.g. by trimming before the artifact starts in
trim_videos_from_dlc.py, or masking them out before CaImAn).

Usage:
    python detect_noise_frames.py /path/to/Miniscope<ts>.avi

    # Tune the threshold or only scan the tail of a long recording
    python detect_noise_frames.py /path/to/Miniscope<ts>.avi \
        --threshold 300 --start-time 1500

    # Save a diagnostic plot of the metric over time
    python detect_noise_frames.py /path/to/Miniscope<ts>.avi --plot
"""
import argparse
import csv
import os

import cv2
import numpy as np


def compute_lap_var_per_frame(video_path: str, start_frame: int = 0, end_frame: int | None = None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total_frames if end_frame is None else min(end_frame, total_frames)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    lap_vars = []
    idx = start_frame
    while idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_vars.append(lap_var)
        idx += 1
    cap.release()

    return np.array(lap_vars), start_frame, fps, total_frames


def contiguous_ranges(flags: np.ndarray, offset: int = 0):
    ranges = []
    in_run = False
    run_start = None
    for i, f in enumerate(flags):
        if f and not in_run:
            in_run = True
            run_start = i
        elif not f and in_run:
            in_run = False
            ranges.append((run_start + offset, i - 1 + offset))
    if in_run:
        ranges.append((run_start + offset, len(flags) - 1 + offset))
    return ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path", help="Path to a Miniscope*.avi video")
    parser.add_argument("--threshold", type=float, default=200.0,
                         help="Laplacian-variance threshold above which a frame is flagged as artifact "
                              "(default: 200; clean frames typically <50, artifact frames typically >1000)")
    parser.add_argument("--start-time", type=float, default=0.0,
                         help="Only scan from this time (seconds) onward, e.g. to skip a known-clean head "
                              "and speed up scanning a long recording (default: 0, scan whole video)")
    parser.add_argument("--output", default=None,
                         help="Output CSV path (default: <video_dir>/<video_stem>_noise_frames.csv)")
    parser.add_argument("--plot", action="store_true", help="Also save a diagnostic plot of lap_var over time")
    args = parser.parse_args()

    video_path = args.video_path
    video_dir = os.path.dirname(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = args.output or os.path.join(video_dir, f"{video_stem}_noise_frames.csv")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    start_frame = int(args.start_time * fps)

    print(f"Scanning {video_path} from frame {start_frame} ({args.start_time:.1f}s) ...")
    lap_vars, offset, fps, total_frames = compute_lap_var_per_frame(video_path, start_frame=start_frame)
    flags = lap_vars > args.threshold

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "time_s", "lap_var", "is_artifact"])
        for i, (lv, fl) in enumerate(zip(lap_vars, flags)):
            frame_idx = i + offset
            writer.writerow([frame_idx, f"{frame_idx / fps:.3f}", f"{lv:.2f}", int(fl)])

    ranges = contiguous_ranges(flags, offset=offset)
    n_bad = int(flags.sum())
    print(f"Total frames scanned: {len(flags)} (of {total_frames} in video)")
    print(f"Flagged as artifact:  {n_bad} ({100 * n_bad / max(len(flags), 1):.1f}%)")
    print(f"Per-frame CSV written to: {output_csv}")
    print(f"Contiguous bad frame ranges (frame_idx, inclusive) [time_s]:")
    for r0, r1 in ranges:
        print(f"  frames {r0}-{r1}  ({r0 / fps:.1f}s - {r1 / fps:.1f}s)  n={r1 - r0 + 1}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = (np.arange(len(lap_vars)) + offset) / fps
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, lap_vars, lw=0.7)
        ax.axhline(args.threshold, color="red", ls="--", lw=1, label=f"threshold={args.threshold}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Laplacian variance")
        ax.set_title(f"Noise metric — {video_stem}")
        ax.legend()
        plot_path = os.path.join(video_dir, f"{video_stem}_noise_frames.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"Diagnostic plot written to: {plot_path}")


if __name__ == "__main__":
    main()
