#!/usr/bin/env python
"""Inventory every session and resolve the files the reward pipeline needs.

Writes one row per session naming the raw arena video, ``timestamps<TS>.csv``, Bpod log, trim info
and behaviour trial tables, plus which reward route applies. Runs no video decoding, so it is cheap
to re-run and is the right place to discover missing pieces before committing to a batch.

Camera naming differs by cohort, and not the way the filenames suggest:

* aging cohort (2024) -- the ARENA camera is ``behavior<TS>.avi``; ``behaviorLinear<TS>.avi`` is the
  home cage. Confirmed by the trimmer's own output: 207 ``Linear_behavior*`` / 146 ``TMaze_behavior*``
  files exist and zero ``*_behaviorLinear*``.
* young and post-May cohorts -- the arena camera is ``Linear<TS>.avi``, home is ``Home<TS>.avi``.

Usage
-----
    python build_reward_manifest.py -o docs/reward_session_manifest.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

import pandas as pd

DRIVES = ["/media/toor/Seagate2", "/media/toor/Seagate3", "/media/toor/Seagate5",
          "/media/toor/SeagateClean", "/media/toor/SeagatePortableDrive"]
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})")
# Mouse905 is deliberately absent: CLAUDE.md states it is not part of either cohort and must not be
# processed further (its FOV degraded partway through the linear-track stage).
MOUSE_RE = re.compile(r"Mouse ?(163|1636|1637|1639|847|943|944|945|946)(?!\d)")
EXCLUDED_MICE = {"Mouse905"}
BPOD_ROOT = "/media/toor/SeagateClean/Bpod Local/Data"
BPOD_RE = re.compile(r"^Mouse(\d+)_(LinearTrack|TMaze\d?)_(\d{8})_(\d{6})\.mat$")

# Reward route per arena. Linear reads rewards off the video; the T-maze cannot (the mouse occludes
# the LED at reward and the same LED also signals cues and error returns), so it fits the clock from
# error-episode LED onsets and takes the reward list from the Bpod log.
ROUTE = {"Linear": "detect", "TMaze": "align"}


def era_for(date: str, task: str) -> str:
    """Camera/apparatus era. The linear track is fixed, so 2024 shares the pre-May boxes; the T-maze
    is carried in and out of the room, so it moved between 2024 and 2026 and needs its own set."""
    if date >= "2026-05-01":
        return "post2026-05"
    if task == "TMaze" and date < "2025-01-01":
        return "aging2024"
    return "pre2026-05"


def index_files(patterns: list[str]) -> dict[str, list[str]]:
    """Map session timestamp -> paths, for a set of filename globs across all drives."""
    idx = defaultdict(list)
    for pat in patterns:
        out = subprocess.run(["find", *[d for d in DRIVES if os.path.isdir(d)],
                              "-name", pat], capture_output=True, text=True).stdout
        for p in out.split("\n"):
            if not p:
                continue
            m = TS_RE.search(os.path.basename(p))
            if m:
                idx[m.group(0)].append(p)
    return idx


def pick_arena_video(cands: list[str], stamp: str, aging: bool) -> str:
    """Choose the raw arena video, preferring an untrimmed copy over a trimmed one."""
    want = f"behavior{stamp}.avi" if aging else f"Linear{stamp}.avi"
    # The exact-basename match is what actually excludes trimmed clips today, because the trimmer
    # writes them as "<Task>_<basename>". The dlc_trim_info.csv check below is a second line of
    # defence in case that convention ever changes -- returning a trimmed clip would offset every
    # frame index by start_frame with nothing to catch it.
    exact = [p for p in cands if os.path.basename(p) == want]
    raw = [p for p in exact if not os.path.exists(os.path.join(os.path.dirname(p), "dlc_trim_info.csv"))]
    return (raw or exact or [""])[0]


def pick_timestamps(cands: list[str], stamp: str, raw_video: str) -> tuple[str, bool]:
    """Timestamps for the chosen video copy, preferring the file sitting beside it.

    Sessions exist in several copies across the drives. Choosing the video and the timestamps
    independently can pair a video with another copy's CSV, which would shift every ``frame_raw``
    with no error. Returns (path, same_directory).
    """
    want = f"timestamps{stamp}.csv"
    exact = [p for p in cands if os.path.basename(p) == want]
    if raw_video:
        beside = [p for p in exact if os.path.dirname(p) == os.path.dirname(raw_video)]
        if beside:
            return beside[0], True
    untrimmed = [p for p in exact
                 if not os.path.exists(os.path.join(os.path.dirname(p), "dlc_trim_info.csv"))]
    return ((untrimmed or exact or [""])[0], False)


def pick_bpod(logs: list[str], stamp: str, n_sessions_that_day: int) -> str:
    """Choose the Bpod log for this session.

    When a mouse ran one session that day, several logs mean the protocol was restarted and the
    largest is the real attempt. When two genuine sessions share a day -- which happens -- picking by
    size would hand both sessions the same log, so fall back to the log starting nearest this
    session's recording instead. The Bpod filename's own start time is what disambiguates.
    """
    if not logs:
        return ""
    if n_sessions_that_day <= 1:
        return max(logs, key=os.path.getsize)
    ref = pd.Timestamp(stamp.replace("T", " ").replace("_", ":"))
    def start(f):
        m = BPOD_RE.match(os.path.basename(f))
        return pd.Timestamp(f"{m.group(3)} {m.group(4)}")
    return min(logs, key=lambda f: abs((start(f) - ref).total_seconds()))


def pick_trials(cands: list[str]) -> str:
    """Prefer the project's N_BINS=70 convention, then the most recently written file."""
    if not cands:
        return ""
    preferred = [c for c in cands if "70nbins" in os.path.basename(c)]
    pool = preferred or cands
    return max(pool, key=os.path.getmtime)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    print("indexing bpod logs ...")
    bpod = defaultdict(list)
    for f in glob.glob(os.path.join(BPOD_ROOT, "**", "*.mat"), recursive=True):
        m = BPOD_RE.match(os.path.basename(f))
        if not m:
            continue
        task = "Linear" if m.group(2) == "LinearTrack" else "TMaze"
        d = m.group(3)
        key = (f"Mouse{m.group(1)}", task, f"{d[:4]}-{d[4:6]}-{d[6:]}")
        if not any(os.path.basename(x) == os.path.basename(f) for x in bpod[key]):
            bpod[key].append(f)

    print("indexing videos and timestamps ...")
    vids = index_files(["behavior*.avi", "Linear*.avi"])
    tstamps = index_files(["timestamps*.csv"])

    print("scanning sessions ...")
    trims = subprocess.run(["find", *[d for d in DRIVES if os.path.isdir(d)],
                            "-name", "dlc_trim_info.csv"], capture_output=True, text=True).stdout
    rows, seen = [], set()
    for p in trims.split("\n"):
        if not p:
            continue
        d = os.path.dirname(p)
        mm, tm = MOUSE_RE.search(d), TS_RE.search(os.path.basename(d))
        if not (mm and tm):
            continue
        mouse, stamp, date = f"Mouse{mm.group(1)}", tm.group(0), tm.group(1)
        try:
            files = os.listdir(d)
        except OSError:
            continue
        low = (d + " " + " ".join(files)).lower()
        task = "TMaze" if "tmaze" in low else ("Linear" if "linear" in low else "?")
        if task == "?" or (mouse, stamp, task) in seen:
            continue
        seen.add((mouse, stamp, task))
        aging = date < "2025-01-01"

        dlc_dir = os.path.join(d, "dlc")
        trials = [t for t in glob.glob(os.path.join(dlc_dir, "*_trials.csv"))
                  if "locations" not in os.path.basename(t)]
        binned = glob.glob(os.path.join(dlc_dir, "*binned_trials*.csv"))

        raw_video = pick_arena_video(vids.get(stamp, []), stamp, aging)
        ts_path, ts_beside = pick_timestamps(tstamps.get(stamp, []), stamp, raw_video)
        rows.append(dict(
            mouse=mouse, task=task, stamp=stamp, date=date,
            era=era_for(date, task), route=ROUTE[task], cohort="aging" if aging else "young",
            session_dir=d,
            raw_video=raw_video,
            timestamps=ts_path, timestamps_beside_video=ts_beside,
            bpod="", n_bpod_logs=len(bpod.get((mouse, task, date), [])),
            trim_info=p,
            trim_video=next((os.path.join(d, f) for f in files
                             if re.match(r"(Linear|TMaze)_(behavior|Linear)", f) and f.endswith(".avi")), ""),
            trials=pick_trials(trials),
            binned=pick_trials(binned)))

    df = pd.DataFrame(rows).sort_values(["mouse", "task", "stamp"])
    df = df[~df.mouse.isin(EXCLUDED_MICE)]
    # Bpod logs are keyed by calendar date, so two genuine sessions on one day would otherwise be
    # handed the same log. Resolve per session now that we know how many share each day.
    per_day = df.groupby(["mouse", "task", "date"]).size()
    df["bpod"] = [pick_bpod(bpod.get((r.mouse, r.task, r.date), []), r.stamp,
                            int(per_day.get((r.mouse, r.task, r.date), 1)))
                  for r in df.itertuples()]
    dupes = df[df.bpod != ""].groupby("bpod").size()
    shared = int((dupes > 1).sum())
    df["ready"] = (df.raw_video != "") & (df.timestamps != "")
    df.loc[df.route == "align", "ready"] &= df.bpod != ""   # the T-maze route needs the log
    df["can_assign"] = df.ready & (df.trials != "")
    df.to_csv(args.output, index=False)

    print(f"\n{len(df)} sessions\n")
    print(df.groupby(["cohort", "task", "era"]).size().to_string())
    print("\nresolvable pieces:")
    for col in ["raw_video", "timestamps", "bpod", "trials", "binned"]:
        print(f"  {col:11s} present for {int((df[col] != '').sum()):4d}/{len(df)}")
    print(f"\n  timestamps taken from the video's own directory: "
          f"{int(df.timestamps_beside_video.sum())}/{int((df.timestamps != '').sum())}")
    if shared:
        print(f"  !! {shared} Bpod logs are still assigned to more than one session - review these")
    print(f"\n  ready to run reward extraction : {int(df.ready.sum())}/{len(df)}")
    print(f"  ...and joinable to trials      : {int(df.can_assign.sum())}/{len(df)}")
    print("\nnot ready, by reason:")
    nr = df[~df.ready]
    if len(nr):
        why = nr.apply(lambda r: "no raw video" if not r.raw_video else
                       ("no timestamps" if not r.timestamps else "no bpod log (T-maze needs it)"), axis=1)
        print(pd.crosstab(nr.task, why).to_string())
    print(f"\nwrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
