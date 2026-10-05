#!/usr/bin/env python3
"""Recover each trimmed session's TRUE first raw frame and record it in dlc_trim_info.csv.

Why this exists
---------------
`trim_videos_from_dlc.py` cuts with input seeking plus stream copy:

    ffmpeg -ss <start_time_seconds> -i raw.avi -t 960 -c:v copy -c:a copy out.avi

`-c copy` cannot re-encode, so ffmpeg is unable to begin mid-GOP: it snaps back to the
nearest keyframe at or before the requested time. The clip therefore starts a few frames
EARLIER than asked. `dlc_trim_info.csv` records the *requested* frame, so every downstream
consumer that slices the DLC table as

    dlc[start_frame : start_frame + neural_len]

pairs pose row i (raw frame start_frame + i) with neural frame i (raw frame
start_frame - k + i). Behaviour therefore runs k frames AHEAD of the neural data, in the
same direction every session, by a session-specific amount (observed 0-57 frames).

Because the trim was a stream copy, the kept frames are BIT-IDENTICAL to the raw ones, so
the true start can be recovered exactly, after the fact, by finding which raw frame the
trimmed video's frame 0 is. No re-trim, no re-encode, no CaImAn re-run.

What it writes
--------------
Two new columns in each dlc_trim_info.csv; `start_frame` is left untouched so the original
requested value stays on the record:

    start_frame_actual      - raw frame index the trimmed video really begins at
    keyframe_offset_frames  - start_frame - start_frame_actual (>= 0)

Videos are only ever opened for reading. Nothing else in the session folder is modified.

Usage
-----
    python scripts/data_audit/measure_trim_offset.py --dry-run          # measure, write nothing
    python scripts/data_audit/measure_trim_offset.py                    # measure and write
    python scripts/data_audit/measure_trim_offset.py --report out.csv   # also dump a summary table
    python scripts/data_audit/measure_trim_offset.py --roots /media/toor/SeagateClean
    python scripts/data_audit/measure_trim_offset.py --force            # redo already-measured rows
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)

DEFAULT_ROOTS = [f"/media/toor/{d}" for d in
                 ("Seagate2", "Seagate3", "Seagate5", "SeagateClean", "SeagatePortableDrive")]

TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})")
# Raw miniscope: no task prefix. Aging cohort uses "miniscope", young uses "Miniscope".
RAW_RE = re.compile(r"^[mM]iniscope(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})\.avi$")
# Trimmed miniscope: a task prefix ("Linear_"/"TMaze_") in front.
TRIM_RE = re.compile(r"^(?:Linear|TMaze)_[mM]iniscope(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})\.avi$")

# Directories that never contain a raw recording we trimmed from, and are expensive to walk.
SKIP_DIRS = {"caiman", "caiman_final", "caiman_aging", "caiman_final_young", "dlc",
             "MEMMAPS", "caiman_data", "__pycache__", "Bpod Local", "Bpod RAW",
             # Recycle bins hold deleted copies that are byte-identical to live files; indexing
             # them lets a match resolve against a file the user already threw away.
             "$RECYCLE.BIN", "RECYCLER", ".Trash-1000", "lost+found"}


def build_raw_index(roots, verbose=True):
    """timestamp -> [paths of raw miniscope files], across every mounted drive."""
    index = defaultdict(list)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in files:
                m = RAW_RE.match(f)
                if m:
                    index[m.group(1)].append(os.path.join(dirpath, f))
    if verbose:
        print(f"[index] {len(index)} distinct raw miniscope timestamps "
              f"({sum(len(v) for v in index.values())} files)", flush=True)
    return index


def find_trim_infos(roots):
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if "dlc_trim_info.csv" in files:
                out.append(os.path.join(dirpath, "dlc_trim_info.csv"))
    return sorted(out)


def first_frame_gray(path):
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)


def frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _read_gray(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)


def corroborate(raw_path, trim_path, raw_start, probes=(1000, 5000)):
    """A single matching frame is not proof. 1p miniscope frames are near-static, and encoders
    emit duplicate frames, so one frame can match by coincidence (this produced false hits at
    the search-window edge). Require that trimmed frame N is ALSO bit-identical to raw frame
    raw_start + N for at least one probe N well inside both clips."""
    cap_t, cap_r = cv2.VideoCapture(trim_path), cv2.VideoCapture(raw_path)
    tn = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
    rn = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    checked = ok_any = 0
    for n in probes:
        if n >= tn or raw_start + n >= rn:
            continue
        a, b = _read_gray(cap_t, n), _read_gray(cap_r, raw_start + n)
        if a is None or b is None or a.shape != b.shape:
            continue
        checked += 1
        if not np.any(a - b):
            ok_any += 1
    cap_t.release(); cap_r.release()
    if checked == 0:
        return "unverifiable"          # clip too short to probe; fall back to arithmetic
    return "ok" if ok_any else "failed"


def iter_exact_matches(raw_path, target, start_hint, window):
    """Yield every raw frame index that is bit-identical to `target`, in order.

    The first hit is not necessarily the right one: 1p miniscope frames are near-static and
    encoders emit duplicates, so a coincidental match can precede the true start. Callers
    corroborate each candidate and move on to the next when it fails.
    """
    cap = cv2.VideoCapture(raw_path)
    lo = max(0, start_hint - window)
    for i in range(lo, start_hint + 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if g.shape == target.shape and not np.any(g - target):
            yield i
    cap.release()


def locate_first_frame(raw_path, target, start_hint, window, check_unique=False):
    """Find the raw frame index whose decoded content is identical to `target`.

    Searches [start_hint - window, start_hint + 2]; the true start can only be at or
    before the requested frame, the +2 is slack for off-by-one in the request itself.

    Returns (index, n_exact_matches). index is None when nothing matches exactly -- that
    is a real signal (e.g. the clip came from a denoised/re-encoded copy, not the raw),
    not something to paper over with a nearest-match guess.
    """
    cap = cv2.VideoCapture(raw_path)
    lo = max(0, start_hint - window)
    hits = []
    for i in range(lo, start_hint + 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if g.shape == target.shape and not np.any(g - target):
            hits.append(i)
            if not check_unique:
                break
    cap.release()
    if not hits:
        return None, 0
    return hits[0], len(hits)


def process(ti_path, raw_index, window, dry_run, force, check_unique):
    """Returns a result dict for one session."""
    sess = os.path.dirname(ti_path)
    res = {"trim_info": ti_path, "session": sess, "status": "", "start_frame": None,
           "start_frame_actual": None, "keyframe_offset_frames": None,
           "n_exact": None, "trimmed_frames": None, "raw_frames": None,
           "arithmetic_check": ""}
    try:
        df = pd.read_csv(ti_path)
    except Exception as e:
        res["status"] = f"unreadable_csv:{type(e).__name__}"
        return res
    if "start_frame" not in df.columns or df.empty:
        res["status"] = "no_start_frame_column"
        return res
    if "start_frame_actual" in df.columns and df["start_frame_actual"].notna().all() and not force:
        res["status"] = "already_measured"
        res["start_frame"] = int(df["start_frame"].iloc[0])
        res["start_frame_actual"] = int(df["start_frame_actual"].iloc[0])
        res["keyframe_offset_frames"] = res["start_frame"] - res["start_frame_actual"]
        return res

    start = int(df["start_frame"].iloc[0])
    res["start_frame"] = start

    trimmed = [f for f in os.listdir(sess) if TRIM_RE.match(f)]
    if not trimmed:
        res["status"] = "no_trimmed_miniscope"
        return res
    trim_path = os.path.join(sess, sorted(trimmed)[0])
    ts = TRIM_RE.match(os.path.basename(trim_path)).group(1)

    candidates = raw_index.get(ts, [])
    if not candidates:
        res["status"] = "no_raw_found"
        return res

    target = first_frame_gray(trim_path)
    if target is None:
        res["status"] = "unreadable_trimmed_video"
        return res

    # Evaluate EVERY candidate rather than taking the first hit: a timestamp can resolve to
    # several files (duplicated trees, plus the pre-concat sub-recordings alongside the
    # concatenated file the trim was actually cut from). Rank by evidence, not by scan order.
    tn = frame_count(trim_path)
    scored = []
    for raw_path in candidates:
        rn = frame_count(raw_path)
        # Try successive exact matches within this candidate until one corroborates, rather
        # than committing to the first (which may be a coincidental static-frame match).
        idx = None
        for cand_idx in iter_exact_matches(raw_path, target, start, window):
            idx = cand_idx
            if corroborate(raw_path, trim_path, cand_idx) != "failed":
                break
        if idx is None:
            continue
        n = 1
        feasible = (rn - idx) >= tn          # the clip cannot be longer than what follows idx
        lo = max(0, start - window)
        # Only suspect when the WINDOW truncated the search (lo > 0). If start <= window then
        # lo == 0 simply because frame 0 is the start of the recording, and a hit there is real.
        at_edge = (idx == lo) and (lo > 0)
        corr = corroborate(raw_path, trim_path, idx)
        exact_len = (rn - idx) == tn         # truncated clip: strongest possible confirmation
        score = (corr == "ok", feasible, exact_len, not at_edge)
        scored.append((score, idx, n, raw_path, rn, corr, feasible, at_edge))
    if not scored:
        res["status"] = "no_exact_match"
        return res
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    _, found, n_exact, used_raw, rn_used, corr, feasible, at_edge = best
    res["corroboration"] = corr
    res["n_candidates"] = len(candidates)
    if corr == "failed":
        res["status"] = "corroboration_failed"
        return res
    if not feasible:
        res["status"] = "infeasible_length"
        return res
    if at_edge:
        res["status"] = "match_at_window_edge"
        return res

    res["start_frame_actual"] = found
    res["keyframe_offset_frames"] = start - found
    res["n_exact"] = n_exact
    res["raw_path"] = used_raw

    # Independent cross-check that uses only frame COUNTS, never pixels: when the trim ran
    # past the end of the recording, trimmed_frames must equal raw_frames - true_start.
    rn = rn_used
    res["trimmed_frames"], res["raw_frames"] = tn, rn
    if rn - found == tn:
        res["arithmetic_check"] = "CONFIRMS (truncated clip)"
    elif tn < rn - found:
        res["arithmetic_check"] = "n/a (full-length clip)"
    else:
        res["arithmetic_check"] = f"INCONSISTENT (raw {rn} - start {found} < trimmed {tn})"

    res["status"] = "measured"

    if not dry_run:
        df["start_frame_actual"] = found
        df["keyframe_offset_frames"] = start - found
        df.to_csv(ti_path, index=False)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--window", type=int, default=300,
                    help="how many frames before start_frame to search (default 300; the "
                         "largest offset seen in practice is 57, but GOP length varies)")
    ap.add_argument("--dry-run", action="store_true", help="measure and report, write nothing")
    ap.add_argument("--force", action="store_true", help="re-measure rows already carrying start_frame_actual")
    ap.add_argument("--check-unique", action="store_true",
                    help="scan the whole window and count every exact match, to prove the "
                         "match is unique (slower; used for validation)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N sessions")
    ap.add_argument("--only-from", type=str, default=None,
                    help="path to a file listing session directories (one per line); only those "
                         "are processed. The raw index is still built from the full --roots, which "
                         "is why this is a filter rather than just narrowing --roots.")
    ap.add_argument("--report", type=str, default=None, help="write a per-session summary CSV here")
    args = ap.parse_args()

    raw_index = build_raw_index(args.roots)
    tis = find_trim_infos(args.roots)
    if args.only_from:
        with open(args.only_from) as fh:
            wanted = {line.strip().rstrip("/") for line in fh if line.strip()}
        tis = [t for t in tis if os.path.dirname(t).rstrip("/") in wanted]
        print(f"[filter] --only-from matched {len(tis)}/{len(wanted)} listed sessions", flush=True)
    if args.limit:
        tis = tis[:args.limit]
    print(f"[scan] {len(tis)} dlc_trim_info.csv to process"
          f"{' (DRY RUN - nothing will be written)' if args.dry_run else ''}", flush=True)

    rows, counts = [], defaultdict(int)
    for n, ti in enumerate(tis, 1):
        r = process(ti, raw_index, args.window, args.dry_run, args.force, args.check_unique)
        rows.append(r)
        counts[r["status"]] += 1
        if n % 25 == 0 or n == len(tis):
            print(f"  [{n}/{len(tis)}] " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                  flush=True)

    print("\n=== summary ===")
    for k, v in sorted(counts.items()):
        print(f"  {k:28s} {v}")
    ks = [r["keyframe_offset_frames"] for r in rows
          if r["status"] in ("measured", "already_measured") and r["keyframe_offset_frames"] is not None]
    if ks:
        ks = np.array(ks)
        print(f"\n  offset k (frames): n={len(ks)}  min={ks.min()}  median={int(np.median(ks))}  "
              f"mean={ks.mean():.1f}  max={ks.max()}")
        print(f"  in seconds @25fps: median={np.median(ks)/25:.2f}s  mean={ks.mean()/25:.2f}s  "
              f"max={ks.max()/25:.2f}s")
        print(f"  k == 0 (already aligned): {(ks == 0).sum()} / {len(ks)}")
    checks = defaultdict(int)
    for r in rows:
        if r["arithmetic_check"]:
            checks[r["arithmetic_check"].split(" ")[0]] += 1
    if checks:
        print("\n  independent frame-count cross-check: " +
              "  ".join(f"{k}={v}" for k, v in sorted(checks.items())))

    if args.report:
        cols = ["session", "status", "start_frame", "start_frame_actual",
                "keyframe_offset_frames", "n_exact", "trimmed_frames", "raw_frames",
                "arithmetic_check", "corroboration", "n_candidates", "raw_path"]
        with open(args.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n  report written: {args.report}")
    bad = [r for r in rows if r["status"] not in ("measured", "already_measured")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
