#!/usr/bin/env python3
import argparse
import subprocess
import os
import pandas as pd
import numpy as np
import re
import csv
import shutil
from typing import Optional


likelihood_thresh = 0.9
TMaze_maximum_threshold = 45
Linear_minimum_threshold = 715  # anything smaller than this, catch it
min_frame = 0
frame_rate = 25


def pick_most_robust_bodypart(dlc_df: pd.DataFrame, thr: float = 0.9) -> str:
    """Return bodypart name with highest fraction of frames having likelihood > thr."""
    # find likelihood columns in the 3rd level of the MultiIndex
    lcols = [
        c
        for c in dlc_df.columns
        if len(c) == 3 and str(c[2]).strip().lower() == "likelihood"
    ]
    if not lcols:
        raise ValueError("No likelihood columns found in DLC data.")
    scores = {}
    for col in lcols:
        bp = col[1]
        s = pd.to_numeric(dlc_df[col], errors="coerce")
        total = s.notna().sum()
        if total == 0:
            continue
        score = (s > thr).sum() / total
        # keep the best score per bodypart in case of multiple scorers
        scores[bp] = max(scores.get(bp, 0.0), float(score))
    if not scores:
        raise ValueError("No valid likelihood data to compute robustness.")
    return max(scores, key=scores.get)


def col_of(dlc_df: pd.DataFrame, bodypart: str, coord: str):
    """Return MultiIndex column for given bodypart and coord in {'x','y','likelihood'}."""
    cs = [
        c
        for c in dlc_df.columns
        if len(c) == 3
        and c[1] == bodypart
        and str(c[2]).strip().lower() == coord
    ]
    if not cs:
        raise ValueError(f"Missing column for bodypart={bodypart}, coord={coord}.")
    return cs[0]


def extract_ts(fname: str) -> Optional[str]:
    """
    Extract YYYY-MM-DDTHH_MM_SS from a filename.
    Example match: behavior2024-10-03T14_16_18.avi
    """
    m = re.search(
        r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", os.path.basename(fname)
    )
    return m.group(1) if m else None


def _first_run_of_n(mask: pd.Series, n: int = 3) -> Optional[int]:
    """Return the first index where `mask` is True for n consecutive frames."""
    consecutive = 0
    run_start = None
    for idx, val in mask.items():
        if val:
            if consecutive == 0:
                run_start = idx
            consecutive += 1
            if consecutive >= n:
                return int(run_start)
        else:
            consecutive = 0
            run_start = None
    return None


def find_start_frame_tmaze(y: pd.Series, likelihood: pd.Series) -> Optional[int]:
    """Find first frame where mouse is above TMaze threshold for 10 consecutive frames with good likelihood."""
    mask = (
        (y.index >= min_frame)
        & (y > TMaze_maximum_threshold)
        & (likelihood > likelihood_thresh)
    )
    return _first_run_of_n(mask)


def find_start_frame_linear(y: pd.Series, likelihood: pd.Series) -> Optional[int]:
    """Find first frame where mouse is below Linear threshold for 10 consecutive frames with good likelihood."""
    mask = (
        (y.index >= min_frame)
        & (y < Linear_minimum_threshold)
        & (likelihood > likelihood_thresh)
    )
    return _first_run_of_n(mask)


def copy_dlc_folder(src_root: str, output_dir: str):
    """Copy the DLC folder tree into output_dir/dlc."""
    dlc_dir = os.path.join(src_root)
    if os.path.isdir(dlc_dir):
        dest_dlc_dir = os.path.join(output_dir, "dlc")

        # Remove if already exists (shutil.copytree requires this)
        if os.path.exists(dest_dlc_dir):
            shutil.rmtree(dest_dlc_dir)

        print(f"[COPY] Copying DLC folder: {dlc_dir} -> {dest_dlc_dir}")
        shutil.copytree(dlc_dir, dest_dlc_dir)
    else:
        print(f"[WARN] No 'dlc' folder found in: {src_root}")


def trim_videos(
    video_files, start_time_seconds: float, output_dir: str, prefix: str
):
    """Run ffmpeg to trim videos."""
    for video_file in video_files:
        out_name = f"{prefix}_{os.path.basename(video_file)}"
        output_file = os.path.join(output_dir, out_name)

        ffmpeg_command = [
            "ffmpeg",
            "-n",
            "-ss",
            f"{start_time_seconds:.3f}",
            "-i",
            video_file,
            "-t",
            "960",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            output_file,
        ]
        print(f"[RUN] {' '.join(ffmpeg_command)}")
        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f"[OK] Wrote: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"[ERR] Trimming failed for {video_file}: {e}")


def process_root(input_root: str, output_root: str, task: str, csv_only: bool):
    """
    Main processing loop.

    task: 'tmaze' or 'linear'
    csv_only: if True, only write dlc_trim_info.csv.
             In that case, the CSV is written in the directory ABOVE the DLC folder
             (the same directory that usually holds the behavior / miniscope videos),
             and no new folders are created and no DLC is copied.
    """
    # Only needed for non-csv-only mode, but harmless in all cases
    os.makedirs(output_root, exist_ok=True)

    for root, dirs, files in os.walk(input_root, topdown=False):
        # # Filter by task type if you want to be strict
        # if task == "tmaze" and "TMaze" not in root:
        #     continue
        # if task == "linear" and "Linear" not in root:
        #     continue

        dlc_file = None
        miniscope_file = None
        behavior_file = None
        start_time_seconds = None
        start_frame = None
        bp = None
        parent_dir = None  # directory above the DLC folder

        for file in files:
            # Identify DLC (CSV, filtered)
            if file.endswith(".csv") and "_filtered" in file:
                print(f"[INFO] Found filtered DLC file: {file}")
                dlc_file = os.path.join(root, file)
                dlc_data = pd.read_csv(
                    dlc_file, header=[0, 1, 2]
                )  # Load DLC CSV with 3 header rows (MultiIndex)
                bp = pick_most_robust_bodypart(dlc_data, thr=likelihood_thresh)

                # Grab its raw Y (no thresholding or interpolation)
                y_col = col_of(dlc_data, bp, "y")
                likelihood_col = col_of(dlc_data, bp, "likelihood")
                y = pd.to_numeric(dlc_data[y_col], errors="coerce")
                likelihood = pd.to_numeric(dlc_data[likelihood_col], errors="coerce")

                # Find first crossing index safely
                if task == "tmaze":
                    start_frame = find_start_frame_tmaze(y, likelihood)
                else:
                    start_frame = find_start_frame_linear(y, likelihood)

                if start_frame is None:
                    print(
                        f"[WARN] No valid crossing frame found in: {dlc_file}, skipping this folder."
                    )
                    dlc_file = None
                    break

                start_time_seconds = start_frame / frame_rate

                print(f"[INFO] DLC: {dlc_file}")
                print(f"[INFO] Selected bodypart by robustness: {bp}")
                print(
                    f"[INFO] First crossing at frame #{start_frame}, "
                    f"start_time_seconds={start_time_seconds:.3f}"
                )

                # --- look for videos ONE DIRECTORY ABOVE the DLC folder ---
                parent_dir = os.path.dirname(root)  # directory above the dlc folder
                try:
                    parent_files = os.listdir(parent_dir)
                except FileNotFoundError:
                    parent_files = []

                # Prefer a deterministic pick in case of multiple matches
                for pf in sorted(parent_files):
                    if pf.endswith(".avi") and "iniscope" in pf:  # 'iniscope' instead of  "miniscope" because sometimes capitalized 
                        miniscope_file = os.path.join(parent_dir, pf)
                        break

                for pf in sorted(parent_files):
                    if pf.endswith(".avi") and "behavior" in pf and "Linear" not in pf:
                        behavior_file = os.path.join(parent_dir, pf)
                    elif pf.endswith(".avi") and "behavior" not in pf and "Linear" in pf:  # new data naming convention is different 
                        behavior_file = os.path.join(parent_dir, pf)
                        break

                # Only one DLC file per folder is assumed, so break after processing it
                break

        # If everything needed is present, trim / write CSV
        if miniscope_file and behavior_file and dlc_file and start_time_seconds is not None:
            # CASE 1: CSV ONLY -> write dlc_trim_info.csv in parent_dir (above DLC folder)
            if csv_only:
                if parent_dir is None:
                    parent_dir = os.path.dirname(root)

                csv_path = os.path.join(parent_dir, "dlc_trim_info.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    if task == "tmaze":
                        writer.writerow(
                            ["start_frame", "start_time_seconds", "most_robust_bodypart"]
                        )
                        writer.writerow(
                            [start_frame, f"{start_time_seconds:.6f}", bp]
                        )
                    else:
                        writer.writerow(["start_frame", "start_time_seconds", "most_robust_bodypart"])
                        writer.writerow([start_frame, f"{start_time_seconds:.6f}", bp])

                print(f"[META] Saved trim info (CSV only) in parent dir: {csv_path}")
                print(f"[TRIM] start_time_seconds={start_time_seconds:.3f}")
                print(f"[TRIM] dlc:       {dlc_file}")
                print(f"[TRIM] miniscope: {miniscope_file}")
                print(f"[TRIM] behavior:  {behavior_file}")
                print("[INFO] --csv-only set, no new folders, no DLC copy, no ffmpeg.")
                continue

            # CASE 2: FULL MODE -> create output_root/ts, copy DLC, write CSV there, trim videos
            ts = extract_ts(behavior_file)
            if ts is None:
                # fallback: use folder name if timestamp missing
                ts = os.path.basename(os.path.dirname(behavior_file))
            output_dir = os.path.join(output_root, ts)
            os.makedirs(output_dir, exist_ok=True)

            # Copy DLC folder
            copy_dlc_folder(root, output_dir)

            # Save start_frame and start_time_seconds to a small CSV in the same folder
            csv_path = os.path.join(output_dir, "dlc_trim_info.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                if task == "tmaze":
                    writer.writerow(
                        ["start_frame", "start_time_seconds", "most_robust_bodypart"]
                    )
                    writer.writerow(
                        [start_frame, f"{start_time_seconds:.6f}", bp]
                    )
                else:
                    writer.writerow(["start_frame", "start_time_seconds"])
                    writer.writerow([start_frame, f"{start_time_seconds:.6f}"])

            print(f"[META] Saved trim info: {csv_path}")
            print(f"[TRIM] start_time_seconds={start_time_seconds:.3f}")
            print(f"[TRIM] dlc:       {dlc_file}")
            print(f"[TRIM] miniscope: {miniscope_file}")
            print(f"[TRIM] behavior:  {behavior_file}")

            prefix = "TMaze" if task == "tmaze" else "Linear"
            trim_videos(
                [miniscope_file, behavior_file],
                start_time_seconds,
                output_dir,
                prefix=prefix,
            )
        else:
            print(f"[MISS] Required files not all found in: {root}")
            if not miniscope_file:
                print("  - Miniscope .avi missing.")
            if not behavior_file:
                print("  - Behavior .avi missing.")
            if not dlc_file:
                print("  - DLC .csv missing.")
            if start_time_seconds is None:
                print("  - Crossing time not computed.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trim TMaze or Linear videos based on DLC crossing time, "
            "or only output dlc_trim_info.csv."
        )
    )
    parser.add_argument(
        "input_root",
        help="Root directory to search for DLC folders (e.g. /media/.../TMaze or /media/.../Linear)",
    )
    parser.add_argument(
        "output_root",
        help="Root directory where trimmed outputs will be written (ignored if --csv-only is used).",
    )
    parser.add_argument(
        "--task",
        choices=["tmaze", "linear"],
        required=True,
        help="Task type: 'tmaze' or 'linear'.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help=(
            "If set, only write dlc_trim_info.csv in the directory ABOVE each DLC folder, "
            "no new folders, no DLC copy, no video trimming."
        ),
    )

    args = parser.parse_args()

    print(f"[ARGS] input_root:  {args.input_root}")
    print(f"[ARGS] output_root: {args.output_root}")
    print(f"[ARGS] task:        {args.task}")
    print(f"[ARGS] csv_only:    {args.csv_only}")

    process_root(
        input_root=args.input_root,
        output_root=args.output_root,
        task=args.task,
        csv_only=args.csv_only,
    )


if __name__ == "__main__":
    main()
