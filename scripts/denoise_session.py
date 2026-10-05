#!/usr/bin/env python3
"""
Remove frames affected by the Miniscope ring-noise artifact (detected via
detect_noise_frames.py's Laplacian-variance metric) from a session's
Miniscope, Home, and Linear recordings plus timestamps.csv, keeping all
four in lockstep by frame index (they share one synchronized frame clock
and one timestamps.csv row per frame). Writes cleaned copies to a new
DENOISED/<TIMESTAMP>/ folder -- never touches the raw files.

Hybrid strategy (see CLAUDE.md pipeline step 3.5 for the full story of how
this was arrived at):
  - Home/Linear are mpeg4/FMP4 with a strictly regular GOP (empirically
    verified fixed at 12 frames for Mouse944, with zero deviation across a
    full-file scan) -- they get cut via lossless stream copy (ffmpeg concat
    demuxer, like concat_session_videos.py) at GOP-aligned boundaries. Zero
    re-encoding, so kept frames stay bit-identical and output size merely
    shrinks by the dropped fraction.
  - Miniscope's own keyframes are NOT reliably on that same grid -- the
    encoder appears to insert extra keyframes at the noise bursts
    themselves (scene-cut detection reacting to the artifact) and this
    resets its periodic keyframe counter, permanently phase-shifting later
    "regular" keyframes off the original grid. (Verified: of 3498 nominal
    grid positions, only 439 were still real Miniscope keyframes in a real
    session -- requiring a grid position that's ALSO a real Miniscope
    keyframe would have forced dropping 73% of the video.) Since Miniscope
    is by far the smallest stream (608x608 vs 1936x1464), it's cheap to
    re-encode losslessly (libx264 -qp 0) with frame-exact selection instead
    -- no GOP constraint needed for a re-encode, so it can drop exactly the
    same frame set as Home/Linear regardless of its own keyframe layout.
  - timestamps.csv drops the same frame-index set as the videos.

An earlier version re-encoded ALL FOUR videos losslessly to get frame-exact
cuts, which inflated a 4.2GB session to 35-43GB since lossless coding of
noisy 1p sensor data barely compresses. Restricting the lossless re-encode
to just the small Miniscope stream avoids that entirely.

Currently only applicable to Mouse944 sessions where this artifact has been
observed (see CLAUDE.md pipeline step 3.5).

Usage:
    python denoise_session.py /media/toor/Seagate3/Mouse944/RAW_DATA/awaiting_dlc/TMaze/2026-07-22T15_08_41/Miniscope2026-07-22T15_08_41.avi
"""
import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_noise_frames import compute_lap_var_per_frame, contiguous_ranges

TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}")


def find_session_files(miniscope_path):
    session_dir = os.path.dirname(os.path.abspath(miniscope_path))
    m = TIMESTAMP_RE.search(os.path.basename(miniscope_path))
    if not m:
        raise ValueError(f"Could not extract a timestamp from {miniscope_path}")
    ts = m.group(0)

    candidates = glob.glob(os.path.join(session_dir, f"*{ts}*.avi"))
    behavior_videos = sorted(
        p for p in candidates
        if os.path.abspath(p) != os.path.abspath(miniscope_path)
        and not os.path.basename(p).lower().startswith("miniscope")
    )

    timestamps_csv = os.path.join(session_dir, f"timestamps{ts}.csv")
    if not os.path.isfile(timestamps_csv):
        matches = glob.glob(os.path.join(session_dir, f"timestamps*{ts}*.csv"))
        timestamps_csv = matches[0] if matches else None

    return ts, session_dir, behavior_videos, timestamps_csv


def resolve_output_dir(miniscope_path, ts, output_root):
    if output_root:
        return os.path.join(output_root, ts)
    parts = os.path.abspath(miniscope_path).split(os.sep)
    if "RAW_DATA" in parts:
        idx = parts.index("RAW_DATA")
        mouse_dir = os.sep.join(parts[:idx + 1])
        return os.path.join(mouse_dir, "DENOISED", ts)
    return os.path.join(os.path.dirname(os.path.abspath(miniscope_path)), "DENOISED", ts)


def get_keyframe_indices(video_path, n_probe_frames):
    """pict_type of the first n_probe_frames frames -> indices of 'I' frames."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-read_intervals", f"%+#{n_probe_frames}",
         "-show_entries", "frame=pict_type", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    types = out.stdout.split()
    return [i for i, t in enumerate(types) if t == "I"]


def detect_shared_gop(video_paths, n_probe_frames=300):
    """Largest step size at which a keyframe exists at the same frame index in
    every one of video_paths, found from the intersection of each video's
    early keyframe positions."""
    keyframe_sets = [set(get_keyframe_indices(p, n_probe_frames)) for p in video_paths]
    common = sorted(set.intersection(*keyframe_sets))
    if len(common) < 2:
        raise RuntimeError(
            f"Could not establish a shared keyframe grid across {video_paths} "
            f"from the first {n_probe_frames} frames (common keyframe positions: {common}). "
            "These videos may not share a fixed GOP -- frame-accurate stream-copy cutting "
            "isn't safe; inspect manually before proceeding."
        )
    diffs = [b - a for a, b in zip(common, common[1:])]
    gop = diffs[0]
    for d in diffs[1:]:
        gop = math.gcd(gop, d)
    return gop


def verify_keyframe(video_path, frame_idx):
    """Frame-exact (not timestamp-seek-snapped) pict_type check via ffprobe's
    lavfi movie+select source, which decodes from the nearest preceding
    keyframe forward and counts frames -- unlike -read_intervals with a time
    offset, which snaps to whatever keyframe precedes that time and is NOT
    frame-exact (this bit us once already: it silently reported the wrong
    frame as a keyframe)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-f", "lavfi",
         "-i", f"movie={video_path},select=eq(n\\,{frame_idx})",
         "-show_entries", "frame=pict_type", "-of", "csv=p=0"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip() == "I"


def expand_and_merge_ranges(bad_ranges, total_frames, gop):
    """Extend each bad range's end to the next shared keyframe boundary (so the
    next kept segment can resume there via stream copy), merging any ranges
    that end up overlapping as a result. Returns (drop_start_inclusive,
    drop_end_exclusive) windows."""
    merged = []
    cur_start = cur_end = None
    for b0, b1 in bad_ranges:
        resume = min(((b1 + 1 + gop - 1) // gop) * gop, total_frames)
        if cur_start is None:
            cur_start, cur_end = b0, resume
        elif b0 <= cur_end:
            cur_end = max(cur_end, resume)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = b0, resume
    if cur_start is not None:
        merged.append((cur_start, cur_end))
    return merged


def keep_segments(merged_drops, total_frames):
    """Complement of the drop windows: (start_inclusive, end_exclusive) segments to keep."""
    segs = []
    prev = 0
    for d0, d1 in merged_drops:
        if d0 > prev:
            segs.append((prev, d0))
        prev = d1
    if prev < total_frames:
        segs.append((prev, total_frames))
    return segs


def write_kept_segments_streamcopy(input_path, output_path, segs, total_frames, fps):
    """For videos with a verified regular GOP: cut via lossless stream copy + concat demuxer."""
    if len(segs) == 1 and segs[0] == (0, total_frames):
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path], check=True)
        return

    list_path = output_path + ".concat.txt"
    with open(list_path, "w") as f:
        for s0, s1 in segs:
            f.write(f"file '{os.path.abspath(input_path)}'\n")
            f.write(f"inpoint {s0 / fps:.6f}\n")
            f.write(f"outpoint {s1 / fps:.6f}\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path]
    subprocess.run(cmd, check=True)
    os.remove(list_path)


def build_select_expr(merged_drops):
    """Keep everything NOT covered by any (inclusive start, exclusive end) drop window."""
    if not merged_drops:
        return None
    terms = [f"between(n\\,{d0}\\,{d1 - 1})" for d0, d1 in merged_drops]
    return "not(" + "+".join(terms) + ")"


def write_kept_frames_reencode(input_path, output_path, merged_drops, fps):
    """For Miniscope (small, irregular keyframes): frame-exact select + lossless
    re-encode. No GOP constraint since this fully re-encodes -- can drop
    exactly the same frame set chosen for the stream-copied videos regardless
    of Miniscope's own (possibly phase-shifted) keyframe layout."""
    select_expr = build_select_expr(merged_drops)
    if select_expr is None:
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path], check=True)
        return
    vf = f"select='{select_expr}',setpts=N/FRAME_RATE/TB"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-qp", "0",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def write_filtered_timestamps(input_csv, output_csv, bad_idx_set):
    with open(input_csv, newline="") as fin, open(output_csv, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for i, row in enumerate(reader):
            if i not in bad_idx_set:
                writer.writerow(row)


def count_frames(video_path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("miniscope_path", help="Path to the session's Miniscope*.avi (raw, untouched)")
    parser.add_argument("--threshold", type=float, default=200.0, help="Laplacian-variance threshold (default: 200)")
    parser.add_argument("--output-root", default=None,
                         help="Root dir to write <TIMESTAMP>/ under (default: <mouse_dir>/RAW_DATA/DENOISED)")
    args = parser.parse_args()

    ts, session_dir, behavior_videos, timestamps_csv = find_session_files(args.miniscope_path)
    if timestamps_csv is None:
        raise FileNotFoundError(f"No timestamps*.csv found for session {ts} in {session_dir}")
    if not behavior_videos:
        raise FileNotFoundError(f"No behavior (Home/Linear) videos found for session {ts} in {session_dir}")

    print(f"Session {ts} in {session_dir}")
    print(f"  Miniscope:  {args.miniscope_path}")
    for v in behavior_videos:
        print(f"  Behavior:   {v}")
    print(f"  Timestamps: {timestamps_csv}")

    print("Scanning Miniscope video for noise frames (full length)...")
    lap_vars, offset, fps, total_frames = compute_lap_var_per_frame(args.miniscope_path)
    flags = lap_vars > args.threshold
    bad_ranges = contiguous_ranges(flags, offset=offset)
    n_bad = sum(r1 - r0 + 1 for r0, r1 in bad_ranges)
    print(f"Flagged {n_bad}/{total_frames} frames across {len(bad_ranges)} range(s).")

    if not bad_ranges:
        print("No artifact frames found -- nothing to drop.")
        merged_drops = []
    else:
        print("Detecting shared keyframe (GOP) grid across behavior videos...")
        gop = detect_shared_gop(behavior_videos)
        print(f"  Shared GOP = {gop} frames (behavior videos only -- Miniscope is re-encoded, "
              f"so its own keyframe layout doesn't need to match)")
        merged_drops = expand_and_merge_ranges(bad_ranges, total_frames, gop)
        n_dropped = sum(d1 - d0 for d0, d1 in merged_drops)
        print(f"After expanding to keyframe boundaries: {n_dropped}/{total_frames} frames across "
              f"{len(merged_drops)} window(s).")
        for d0, d1 in merged_drops:
            print(f"  drop frames {d0}-{d1 - 1} ({d0 / fps:.1f}s-{(d1 - 1) / fps:.1f}s) n={d1 - d0}")
            if d1 < total_frames:
                for video_path in behavior_videos:
                    if not verify_keyframe(video_path, d1):
                        raise RuntimeError(
                            f"Expected frame {d1} to be a keyframe in {video_path} but it isn't -- "
                            "shared-GOP assumption broke down for this session. Aborting before writing "
                            "any output; inspect this session's encoding manually."
                        )

    segs = keep_segments(merged_drops, total_frames)
    expected_kept = total_frames - sum(d1 - d0 for d0, d1 in merged_drops)

    out_dir = resolve_output_dir(args.miniscope_path, ts, args.output_root)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing denoised session to: {out_dir}")

    mini_out = os.path.join(out_dir, os.path.basename(args.miniscope_path))
    print(f"  Writing {mini_out} (frame-exact lossless re-encode) ...")
    write_kept_frames_reencode(args.miniscope_path, mini_out, merged_drops, fps)
    kept = count_frames(mini_out)
    status = "OK" if kept == expected_kept else "MISMATCH"
    print(f"    -> {kept} frames (expected {expected_kept}) [{status}]")

    for video_path in behavior_videos:
        out_path = os.path.join(out_dir, os.path.basename(video_path))
        print(f"  Writing {out_path} (lossless stream copy) ...")
        write_kept_segments_streamcopy(video_path, out_path, segs, total_frames, fps)
        kept = count_frames(out_path)
        status = "OK" if kept == expected_kept else "MISMATCH"
        print(f"    -> {kept} frames (expected {expected_kept}) [{status}]")

    bad_idx_set = set()
    for d0, d1 in merged_drops:
        bad_idx_set.update(range(d0, d1))

    out_csv = os.path.join(out_dir, os.path.basename(timestamps_csv))
    write_filtered_timestamps(timestamps_csv, out_csv, bad_idx_set)
    with open(out_csv) as f:
        n_rows = sum(1 for _ in f)
    status = "OK" if n_rows == expected_kept else "MISMATCH"
    print(f"  Writing {out_csv} ... -> {n_rows} rows (expected {expected_kept}) [{status}]")

    log_path = os.path.join(out_dir, "denoise_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dropped_range_start", "dropped_range_end_exclusive", "n_frames", "start_time_s", "end_time_s"])
        for d0, d1 in merged_drops:
            writer.writerow([d0, d1, d1 - d0, f"{d0 / fps:.3f}", f"{(d1 - 1) / fps:.3f}"])
    print(f"Dropped-frame log written to: {log_path}")


if __name__ == "__main__":
    main()
