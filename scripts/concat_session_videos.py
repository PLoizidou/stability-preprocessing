#!/usr/bin/env python3
"""
Strict concatenation for Miniscope, Home, Linear videos and timestamps CSVs.

Rules per calendar date (YYYY-MM-DD extracted from filenames):
  - Concatenate ONLY if that date has >= 2 files in EACH category:
        Miniscope, Home, Linear, timestamps
  - If any category has < 2 files that date:
        Skip the entire date (do not touch any files)
  - Concatenated output filenames use earliest datetime - 1 second
  - Output filenames DO NOT end with '_concat'
"""

import argparse
import os
import glob
import subprocess
import csv
import re
from datetime import datetime, timedelta
from collections import defaultdict

# ---------------------------------------
# Helpers
# ---------------------------------------

def sorted_glob(pattern, root_dir):
    return sorted(glob.glob(os.path.join(root_dir, pattern)))

def extract_datetime_from_name(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%dT%H_%M_%S")

def group_by_date(files):
    groups = defaultdict(list)
    for f in files:
        base = os.path.basename(f)
        dt = extract_datetime_from_name(base)
        if dt is None:
            continue
        date_key = dt.strftime("%Y-%m-%d")
        groups[date_key].append((dt, f))

    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x[0] or datetime.min)
    return groups

def make_output_name(prefix, group):
    dts = [dt for dt, _ in group if dt is not None]
    earliest = min(dts)
    adjusted = earliest - timedelta(seconds=1)
    return prefix + adjusted.strftime("%Y-%m-%dT%H_%M_%S")

def concat_videos(out_path, files):
    if len(files) < 2:
        return

    list_file = out_path + ".lst"
    with open(list_file, "w") as f:
        for v in files:
            f.write(f"file '{os.path.abspath(v)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.remove(list_file)

def concat_csv(out_path, files):
    if len(files) < 2:
        return

    with open(out_path, "w", newline="") as fout:
        writer = None
        for idx, fpath in enumerate(files):
            with open(fpath, "r", newline="") as fin:
                reader = csv.reader(fin)
                for j, row in enumerate(reader):
                    if idx > 0 and j == 0:
                        continue
                    if writer is None:
                        writer = csv.writer(fout)
                    writer.writerow(row)

# ---------------------------------------
# Duration filtering
# ---------------------------------------

def get_video_duration_seconds(video_path):
    """
    Returns video duration in seconds using ffprobe.
    Returns None if duration cannot be determined.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def filter_short_videos_and_csvs(video_group, ts_group, min_seconds=10):
    """
    Removes videos shorter than min_seconds.
    Any removed video also removes the corresponding timestamps CSV
    by index order within the date group.
    """
    kept_videos = []
    kept_csvs = []

    for idx, ((v_dt, v_path), (ts_dt, ts_path)) in enumerate(zip(video_group, ts_group)):
        dur = get_video_duration_seconds(v_path)
        if dur is not None and dur < min_seconds:
            print(f"    -> Excluding {os.path.basename(v_path)} ({dur}s)")
            continue
        kept_videos.append((v_dt, v_path))
        kept_csvs.append((ts_dt, ts_path))

    return kept_videos, kept_csvs


# ---------------------------------------
# Main
# ---------------------------------------

def process_session(
    session_dir,
    ms_pattern="miniscope*.avi",
    home_pattern="behaviorLinear*.avi",
    linear_pattern="behavior2*.avi",
    ts_pattern="timestamps*.csv",
):
    session_dir = os.path.abspath(session_dir)
    print("Session:", session_dir)

    # Collect files
    ms_files     = sorted_glob(ms_pattern, session_dir)
    home_files   = sorted_glob(home_pattern, session_dir)
    linear_files = sorted_glob(linear_pattern, session_dir)
    ts_files     = sorted_glob(ts_pattern, session_dir)

    ms_groups     = group_by_date(ms_files)
    home_groups   = group_by_date(home_files)
    linear_groups = group_by_date(linear_files)
    ts_groups     = group_by_date(ts_files)



    all_dates = (
        set(ms_groups.keys())
        | set(home_groups.keys())
        | set(linear_groups.keys())
        | set(ts_groups.keys())
    )

    for date in sorted(all_dates):
        ms_group     = ms_groups.get(date, [])
        home_group   = home_groups.get(date, [])
        linear_group = linear_groups.get(date, [])
        ts_group     = ts_groups.get(date, [])

        n_ms     = len(ms_group)
        n_home   = len(home_group)
        n_linear = len(linear_group)
        n_ts     = len(ts_group)

        print(f"\nDate {date}:")
        print(f"  Miniscope : {n_ms}")
        print(f"  Home      : {n_home}")
        print(f"  Linear    : {n_linear}")
        print(f"  timestamps: {n_ts}")

        if n_ms < 2 or n_home < 2 or n_linear < 2 or n_ts < 2:
            print("  -> Skipping (not all categories have 2+ files)")
            continue
        
                # ---------------------------------------
        # Remove videos < 10 seconds (and matching CSVs)
        # ---------------------------------------

        ms_group, ts_group = filter_short_videos_and_csvs(ms_group, ts_group)
        home_group, _      = filter_short_videos_and_csvs(home_group, ts_group)
        linear_group, _    = filter_short_videos_and_csvs(linear_group, ts_group)

        if len(ms_group) < 2 or len(home_group) < 2 or len(linear_group) < 2 or len(ts_group) < 2:
            print("  -> Skipping after duration filter (<10s videos removed)")
            continue

        # Miniscope
        _, ms_paths = zip(*ms_group)
        out_name = make_output_name("Miniscope", ms_group) + ".avi"
        concat_videos(os.path.join(session_dir, out_name), list(ms_paths))

        # Home
        _, home_paths = zip(*home_group)
        out_name = make_output_name("Home", home_group) + ".avi"
        concat_videos(os.path.join(session_dir, out_name), list(home_paths))

        # Linear
        _, linear_paths = zip(*linear_group)
        out_name = make_output_name("Linear", linear_group) + ".avi"
        concat_videos(os.path.join(session_dir, out_name), list(linear_paths))

        # timestamps
        _, ts_paths = zip(*ts_group)
        out_name = make_output_name("timestamps", ts_group) + ".csv"
        concat_csv(os.path.join(session_dir, out_name), list(ts_paths))

    print("\nDone\n")


# ---------------------------------------
# CLI
# ---------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Strict concat for all categories per date.")
    ap.add_argument("session_dir")
    ap.add_argument("--ms-pattern", default="Miniscope*.avi")
    ap.add_argument("--home-pattern", default="Home*.avi")
    ap.add_argument("--linear-pattern", default="Linear*.avi")
    ap.add_argument("--ts-pattern", default="timestamps*.csv")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_session(
        args.session_dir,
        ms_pattern=args.ms_pattern,
        home_pattern=args.home_pattern,
        linear_pattern=args.linear_pattern,
        ts_pattern=args.ts_pattern,
    )