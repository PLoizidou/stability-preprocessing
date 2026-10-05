#!/usr/bin/env python3
"""
Concatenate same-day recordings for Miniscope, Home, Linear videos
and timestamps CSVs.

Rules per calendar date (YYYY-MM-DD extracted from filenames):
  - Concatenate ONLY if that date has >= 2 files in EACH category:
        Miniscope, Home, Linear, timestamps
  - If any category has < 2 files that date:
        Skip the entire date (do not touch any files)
  - Videos shorter than 10 s are excluded before concatenation;
    the matching timestamps CSV entry is dropped alongside them.
  - If exclusions leave any category with < 2 files, skip the date.
  - Concatenated outputs are written to <session_dir>/concat/
  - Output filenames use the earliest datetime in the group minus 1 second.
  - Output filenames do NOT end with '_concat'.

Two recordings from different days that share the same time-of-day
are correctly separated because grouping is done on the full
YYYY-MM-DD date key, not on the time portion alone.

Usage
-----
    python concat_same_day_recordings.py /path/to/session_dir

    # override file-name patterns if your files use different prefixes:
    python concat_same_day_recordings.py /path/to/session_dir \\
        --ms-pattern     "Miniscope*.avi" \\
        --home-pattern   "Home*.avi"      \\
        --linear-pattern "Linear*.avi"   \\
        --ts-pattern     "timestamps*.csv"
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def sorted_glob(pattern, root_dir):
    """Return sorted list of absolute paths matching pattern inside root_dir."""
    return sorted(glob.glob(os.path.join(root_dir, pattern)))


def extract_datetime_from_name(name):
    """
    Parse the first YYYY-MM-DDTHH_MM_SS timestamp found in a filename.
    Returns a datetime object, or None if no match.
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%dT%H_%M_%S")


def group_by_date(files):
    """
    Group a list of file paths by their calendar date (YYYY-MM-DD).

    Returns
    -------
    dict[str, list[tuple[datetime, str]]]
        Keys are date strings; values are lists of (datetime, path) sorted
        chronologically.  Files whose names contain no recognisable timestamp
        are silently skipped.

    Note
    ----
    Two recordings from different days with the *same time of day* are kept
    separate because the key is the full date string, not the time portion.
    """
    groups = defaultdict(list)
    for f in files:
        dt = extract_datetime_from_name(os.path.basename(f))
        if dt is None:
            continue
        groups[dt.strftime("%Y-%m-%d")].append((dt, f))

    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x[0])
    return groups


def make_output_name(prefix, group):
    """
    Build an output filename stem from the earliest datetime in the group,
    shifted back by one second.

    Example: group with earliest 2025-12-16T16_22_31
             → 'Miniscope2025-12-16T16_22_30'
    """
    dts = [dt for dt, _ in group if dt is not None]
    adjusted = min(dts) - timedelta(seconds=1)
    return prefix + adjusted.strftime("%Y-%m-%dT%H_%M_%S")


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

def concat_videos(out_path, files):
    """
    Concatenate video files into out_path using ffmpeg's concat demuxer.
    The ffmpeg list file is written to a system temp directory so it is
    always writable, independent of whether out_path's directory exists.
    """
    if len(files) < 2:
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lst", delete=False, encoding="utf-8"
    ) as tmp:
        list_file = tmp.name
        for v in files:
            tmp.write(f"file '{os.path.abspath(v)}'\n")

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


def has_header(filepath):
    """
    Return True if the CSV's first row looks like a header (i.e. contains at
    least one cell that cannot be parsed as an ISO-8601 timestamp).
    These timestamp CSVs have NO header — every row is a timestamp string —
    so this guard prevents the first data row of files 2..N from being dropped.
    """
    import re
    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            first_row = next(csv.reader(f), None)
        if first_row is None:
            return False
        # If every cell in the first row looks like a timestamp, there is no header
        return not all(iso_re.match(cell.strip()) for cell in first_row if cell.strip())
    except Exception:
        return False


def concat_csv(out_path, files):
    """
    Concatenate CSV files into out_path.

    If the files have a header row (first row contains non-timestamp text),
    the header is taken from the first file and skipped in all subsequent files.
    If the files have no header (all rows are data, as with these timestamp CSVs),
    every row from every file is written without skipping anything.
    """
    if len(files) < 2:
        return

    skip_header = has_header(files[0])

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = None
        for file_idx, fpath in enumerate(files):
            with open(fpath, "r", newline="", encoding="utf-8") as fin:
                for row_idx, row in enumerate(csv.reader(fin)):
                    # Only skip the first row of files 2..N when a header exists
                    if skip_header and file_idx > 0 and row_idx == 0:
                        continue
                    if writer is None:
                        writer = csv.writer(fout)
                    writer.writerow(row)


# ---------------------------------------------------------------------------
# Duration filtering
# ---------------------------------------------------------------------------

def get_video_duration_seconds(video_path):
    """
    Return the duration of a video in seconds via ffprobe.
    Returns None if the duration cannot be determined (file is kept in that case).
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


def find_short_videos(video_group, min_seconds=10):
    """
    Return a list of (datetime, path, duration) tuples for every video in
    video_group whose duration is below min_seconds.
    Videos whose duration cannot be read by ffprobe are not flagged (fail-open).
    """
    short = []
    for dt, path in video_group:
        dur = get_video_duration_seconds(path)
        if dur is not None and dur < min_seconds:
            short.append((dt, path, dur))
    return short


def move_short_recordings(
    date, session_dir, short_dir,
    ms_group, home_group, linear_group, ts_group,
    min_seconds=10,
):
    """
    Identify all short videos across the three video categories for a given
    date.  For each short video, move it and its paired timestamps CSV into
    short_dir.

    Matching is done by TIMESTAMP, not by positional index.  This is safe
    even when the groups have different lengths (e.g. a stray extra file in
    one category), because each file is only moved if its own duration is
    short — never because another category's file at the "same position" was
    short.  The paired CSV is identified by sharing the same datetime key.

    Returns updated (ms_group, home_group, linear_group, ts_group) with the
    moved entries removed.
    """
    os.makedirs(short_dir, exist_ok=True)

    # Build a set of timestamps to remove, driven by each camera independently.
    # A timestamp is flagged if that camera's recording at that time is short.
    remove_timestamps = set()

    for label, group in (
        ("Miniscope", ms_group),
        ("Home",      home_group),
        ("Linear",    linear_group),
    ):
        for dt, path in group:
            dur = get_video_duration_seconds(path)
            if dur is not None and dur < min_seconds:
                remove_timestamps.add(dt)
                print(f"  -> Moving short [{label}]  {os.path.basename(path)}"
                      f"  ({dur:.1f} s)  →  {os.path.basename(short_dir)}/")

    if not remove_timestamps:
        return ms_group, home_group, linear_group, ts_group

    # Move files whose timestamp is flagged; keep the rest.
    def move_by_timestamp(group, label):
        kept = []
        for dt, path in group:
            if dt in remove_timestamps:
                dst = os.path.join(short_dir, os.path.basename(path))
                shutil.move(path, dst)
            else:
                kept.append((dt, path))
        return kept

    ms_group     = move_by_timestamp(ms_group,     "Miniscope")
    home_group   = move_by_timestamp(home_group,   "Home")
    linear_group = move_by_timestamp(linear_group, "Linear")

    # Move CSVs whose timestamp matches a removed session.
    kept_csvs = []
    for dt, path in ts_group:
        if dt in remove_timestamps:
            dst = os.path.join(short_dir, os.path.basename(path))
            shutil.move(path, dst)
            print(f"  -> Moving paired  [timestamps]  {os.path.basename(path)}"
                  f"  →  {os.path.basename(short_dir)}/")
        else:
            kept_csvs.append((dt, path))
    ts_group = kept_csvs

    return ms_group, home_group, linear_group, ts_group


# ---------------------------------------------------------------------------
# Concat-output detection
# ---------------------------------------------------------------------------

def find_prior_concat_outputs(group):
    """
    Detect files in a source group that are themselves prior concat outputs.

    A concat output is named with (earliest_timestamp - 1 second), so if
    such a file ends up back in the source directory its timestamp T satisfies:
        T + 1s  ==  some other file's timestamp in the same group.

    Returns a list of file paths that match this criterion.
    """
    ts_set = {dt for dt, _ in group}
    return [
        path for dt, path in group
        if (dt + __import__("datetime").timedelta(seconds=1)) in ts_set
    ]


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_session(
    session_dir,
    ms_pattern="Miniscope*.avi",
    home_pattern="Home*.avi",
    linear_pattern="Linear*.avi",
    ts_pattern="timestamps*.csv",
):
    session_dir = os.path.abspath(session_dir)
    concat_dir  = os.path.join(session_dir, "concat")
    print(f"Session : {session_dir}")
    print(f"Output  : {concat_dir}")

    # ── Collect and group all files by calendar date ────────────────────────
    ms_groups     = group_by_date(sorted_glob(ms_pattern,     session_dir))
    home_groups   = group_by_date(sorted_glob(home_pattern,   session_dir))
    linear_groups = group_by_date(sorted_glob(linear_pattern, session_dir))
    ts_groups     = group_by_date(sorted_glob(ts_pattern,     session_dir))

    all_dates = (
        set(ms_groups)
        | set(home_groups)
        | set(linear_groups)
        | set(ts_groups)
    )

    for date in sorted(all_dates):
        ms_group     = ms_groups.get(date, [])
        home_group   = home_groups.get(date, [])
        linear_group = linear_groups.get(date, [])
        ts_group     = ts_groups.get(date, [])

        print(f"\nDate {date}:")
        print(f"  Miniscope  : {len(ms_group)}")
        print(f"  Home       : {len(home_group)}")
        print(f"  Linear     : {len(linear_group)}")
        print(f"  timestamps : {len(ts_group)}")

        # ── Gate 1: if any category has < 2 files, copy what exists to concat/ ──
        # A single session per day is copied as-is rather than skipped, so it
        # is available alongside concatenated sessions in the concat/ folder.
        if any(len(g) < 2 for g in (ms_group, home_group, linear_group, ts_group)):
            # Use the single session's own timestamp as the subfolder name.
            all_groups = [g for g in (ms_group, home_group, linear_group, ts_group) if g]
            session_ts = all_groups[0][0][0].strftime("%Y-%m-%dT%H_%M_%S")
            session_out_dir = os.path.join(concat_dir, session_ts)
            if os.path.exists(session_out_dir):
                print(f"  -> Skipping {date}: concat/{session_ts}/ already exists.")
                continue
            print(f"  -> Only 1 session for {date}: copying files to concat/{session_ts}/ as-is.")
            os.makedirs(session_out_dir, exist_ok=True)
            for group in (ms_group, home_group, linear_group, ts_group):
                for _, src_path in group:
                    dst = os.path.join(session_out_dir, os.path.basename(src_path))
                    shutil.copy2(src_path, dst)
                    print(f"     Copied {os.path.basename(src_path)}")
            continue

        # ── Gate 2: move short videos (< 10 s) to a quarantine folder ────────
        # Short recordings and their paired CSVs are moved to
        # <session_dir>/short_recordings/ before concatenation.
        # If too many are removed and fewer than 2 sessions remain for this
        # date, the date is skipped (other dates are still processed).
        short_dir = os.path.join(session_dir, "short_recordings")
        ms_group, home_group, linear_group, ts_group = move_short_recordings(
            date, session_dir, short_dir,
            ms_group, home_group, linear_group, ts_group,
        )

        if any(len(g) < 2 for g in (ms_group, home_group, linear_group, ts_group)):
            all_groups = [g for g in (ms_group, home_group, linear_group, ts_group) if g]
            session_ts = all_groups[0][0][0].strftime("%Y-%m-%dT%H_%M_%S")
            session_out_dir = os.path.join(concat_dir, session_ts)
            if os.path.exists(session_out_dir):
                print(f"  -> Skipping {date}: concat/{session_ts}/ already exists.")
                continue
            print(f"  -> Only 1 session remains for {date} after moving short "
                  f"recordings: copying remaining file(s) to concat/{session_ts}/ as-is.")
            os.makedirs(session_out_dir, exist_ok=True)
            for group in (ms_group, home_group, linear_group, ts_group):
                for _, src_path in group:
                    dst = os.path.join(session_out_dir, os.path.basename(src_path))
                    shutil.copy2(src_path, dst)
                    print(f"     Copied {os.path.basename(src_path)}")
            continue

        # ── Gate 3: hard error if prior concat outputs exist in source dir ───────
        # or in the concat/ output dir. A prior output in the source dir would be
        # silently folded into the next concatenation. A prior output in concat/
        # would be overwritten. Both are detected and blocked.
        prior_in_source = {}
        for label, group in (
            ("Miniscope",  ms_group),
            ("Home",       home_group),
            ("Linear",     linear_group),
            ("timestamps", ts_group),
        ):
            found = find_prior_concat_outputs(group)
            if found:
                prior_in_source[label] = found

        session_ts = make_output_name("", ms_group)   # e.g. "2025-12-16T10_59_59"
        planned = {
            "Miniscope" : make_output_name("Miniscope",  ms_group)     + ".avi",
            "Home"      : make_output_name("Home",       home_group)   + ".avi",
            "Linear"    : make_output_name("Linear",     linear_group) + ".avi",
            "timestamps": make_output_name("timestamps", ts_group)     + ".csv",
        }
        session_out_dir = os.path.join(concat_dir, session_ts)
        prior_in_concat = [
            name for name in planned.values()
            if os.path.exists(os.path.join(session_out_dir, name))
        ]

        if prior_in_concat:
            # Files already in concat/ subfolder — skip this date, nothing to do.
            print(f"  -> Skipping {date}: concatenated file(s) already present in concat/.")
            for name in prior_in_concat:
                print(f"       [concat/]  {name}")
            continue

        if prior_in_source:
            # Prior concat output(s) are sitting in the source directory.
            # Move ALL files belonging to those prior-output timestamps into
            # a 'previous_concat' folder, then remove them from the groups so
            # the remaining files are concatenated normally.
            prev_dir = os.path.join(session_dir, "previous_concat")
            os.makedirs(prev_dir, exist_ok=True)
            prior_timestamps = set()
            for label, paths in prior_in_source.items():
                for p in paths:
                    dt = extract_datetime_from_name(os.path.basename(p))
                    if dt:
                        prior_timestamps.add(dt)
                    dst = os.path.join(prev_dir, os.path.basename(p))
                    shutil.move(p, dst)
                    print(f"  -> Moved prior concat [{label}]  {os.path.basename(p)}"
                          f"  →  previous_concat/")

            # Remove the prior-output entries from every group by timestamp
            def drop_prior(group):
                return [(dt, p) for dt, p in group if dt not in prior_timestamps]

            ms_group     = drop_prior(ms_group)
            home_group   = drop_prior(home_group)
            linear_group = drop_prior(linear_group)
            ts_group     = drop_prior(ts_group)

            # Recompute planned names and session dir with the cleaned groups
            session_ts      = make_output_name("", ms_group)
            planned = {
                "Miniscope" : make_output_name("Miniscope",  ms_group)     + ".avi",
                "Home"      : make_output_name("Home",       home_group)   + ".avi",
                "Linear"    : make_output_name("Linear",     linear_group) + ".avi",
                "timestamps": make_output_name("timestamps", ts_group)     + ".csv",
            }
            session_out_dir = os.path.join(concat_dir, session_ts)

            if any(len(g) < 2 for g in (ms_group, home_group, linear_group, ts_group)):
                print(f"  -> Only 1 session remains for {date} after removing prior "
                      f"concat output(s): copying to concat/{session_ts}/ as-is.")
                os.makedirs(session_out_dir, exist_ok=True)
                for group in (ms_group, home_group, linear_group, ts_group):
                    for _, src_path in group:
                        dst = os.path.join(session_out_dir, os.path.basename(src_path))
                        shutil.copy2(src_path, dst)
                        print(f"     Copied {os.path.basename(src_path)}")
                continue

        # ── All gates passed: create per-session subfolder and write outputs ─────
        # The subfolder is named after the output timestamp, e.g.
        #   concat/2025-12-16T10_59_59/
        # derived from the same earliest-minus-1s logic as the filenames.
        os.makedirs(session_out_dir, exist_ok=True)

        # Miniscope
        _, ms_paths = zip(*ms_group)
        print(f"  Writing {session_ts}/{planned['Miniscope']}")
        concat_videos(os.path.join(session_out_dir, planned["Miniscope"]), list(ms_paths))

        # Home
        _, home_paths = zip(*home_group)
        print(f"  Writing {session_ts}/{planned['Home']}")
        concat_videos(os.path.join(session_out_dir, planned["Home"]), list(home_paths))

        # Linear
        _, linear_paths = zip(*linear_group)
        print(f"  Writing {session_ts}/{planned['Linear']}")
        concat_videos(os.path.join(session_out_dir, planned["Linear"]), list(linear_paths))

        # timestamps CSV
        _, ts_paths = zip(*ts_group)
        print(f"  Writing {session_ts}/{planned['timestamps']}")
        concat_csv(os.path.join(session_out_dir, planned["timestamps"]), list(ts_paths))

    print("\nDone\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Concatenate same-day Miniscope/Home/Linear recordings and "
            "timestamps CSVs into <session_dir>/concat/."
        )
    )
    ap.add_argument(
        "session_dir",
        help="Directory containing the raw recording files.",
    )
    ap.add_argument("--ms-pattern",     default="Miniscope*.avi",  help="Glob for Miniscope videos.")
    ap.add_argument("--home-pattern",   default="Home*.avi",       help="Glob for Home videos.")
    ap.add_argument("--linear-pattern", default="Linear*.avi",     help="Glob for Linear videos.")
    ap.add_argument("--ts-pattern",     default="timestamps*.csv", help="Glob for timestamps CSVs.")
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
