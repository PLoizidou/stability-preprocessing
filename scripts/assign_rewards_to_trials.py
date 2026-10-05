#!/usr/bin/env python
"""Attach detected/aligned rewards to the behaviour-derived trials of a session.

Consumes the CSV written by ``detect_reward_flushes.py`` and the trial table written by
``Linear_combined_pipeline.ipynb`` / ``TMaze_combined_pipeline.ipynb``, and adds reward columns to
the trial table (and, optionally, to the spatially-binned per-frame table that feeds the megadata
builder in ``stability-analysis``).

Both sides already speak raw-video frame indices -- the detector reports ``frame_raw`` and the trial
tables carry ``start_frame_global`` / ``end_frame_global`` -- so no conversion is needed and there is
no opportunity for an off-by-one.

Matching rule
-------------
A reward is attached to a trial whose ``destination`` equals the reward's port and whose frame span
brackets the reward, allowing ``--tolerance`` frames past the end. A trial containing the reward
outright is preferred over one that merely ended shortly before; a trial never receives more than
one reward.

The two arenas differ here and the rule has to cover both:

* **Linear track** -- the traversal is closed on arrival and the poke follows, so rewards land
  *after* ``end_frame_global`` (median ~18 frames later) and never inside the trial.
* **T-maze** -- traversals are longer and run through the time spent at the port, so rewards
  typically land *inside* ``[start_frame_global, end_frame_global]``.

The destination constraint does the real disambiguating; the window only guards against attaching a
reward to a trial many seconds earlier.

Rewards falling outside the frame range the behaviour pipeline covers (the trimmed clip) cannot be
attached to anything and are reported separately from genuine failures -- the Bpod session usually
runs longer than the 960 s analysis window.

Usage
-----
    python assign_rewards_to_trials.py --rewards rewards.csv --trials <...>_trials.csv \\
        --task linear -o trials_with_rewards.csv [--binned <...>_binned_trials_70nbins.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

# Reward port/arm -> the `destination` label the behaviour pipelines use.
PORT_TO_DESTINATION = {
    "linear": {"Port4": "left", "Port2": "right"},
    # T-maze trial tables call the centre/stem arm "base".
    "tmaze": {"left": "left", "right": "right", "centre": "base"},
}

DEFAULT_TOLERANCE = 400  # frames (~16 s); the destination match is what actually disambiguates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rewards", required=True, help="CSV from detect_reward_flushes.py")
    ap.add_argument("--trials", required=True, help="*_trials.csv from the behaviour pipeline")
    ap.add_argument("--task", choices=["linear", "tmaze"], required=True)
    ap.add_argument("-o", "--output", required=True, help="trial table with reward columns added")
    ap.add_argument("--binned", help="optional *_binned_trials_*nbins.csv to propagate columns into")
    ap.add_argument("--binned-output", help="where to write the annotated binned table")
    ap.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE,
                    help=f"max frames after trial end to still attach a reward (default {DEFAULT_TOLERANCE})")
    ap.add_argument("--report", help="write a JSON QC report here")
    ap.add_argument("--trim-info", help="dlc_trim_info.csv, used to derive the *_global frame "
                                        "columns for older trial tables that lack them")
    args = ap.parse_args()

    rw = pd.read_csv(args.rewards).sort_values("frame_raw").reset_index(drop=True)
    tr = pd.read_csv(args.trials)
    if "trial_id" not in tr.columns:
        tr = tr.reset_index().rename(columns={"index": "trial_id"})
    # Older trial tables predate the *_global columns. They can be reconstructed exactly, since the
    # global index is just the local one shifted by the trim start frame.
    if ("start_frame_global" not in tr.columns or "end_frame_global" not in tr.columns):
        if args.trim_info and {"start_frame", "end_frame"} <= set(tr.columns):
            with open(args.trim_info) as fh:
                off = int(next(csv.DictReader(fh))["start_frame"])
            tr["start_frame_global"] = tr.start_frame + off
            tr["end_frame_global"] = tr.end_frame + off
            print(f"note   : trial table had no *_global columns; derived them from "
                  f"{os.path.basename(args.trim_info)} (start_frame {off})")
        else:
            print("ERROR: trial table lacks the *_global frame columns and no --trim-info was given "
                  "to derive them.", file=sys.stderr)
            return 2
    if "destination" not in tr.columns:
        print("ERROR: trial table lacks 'destination'.", file=sys.stderr)
        return 2

    mapping = PORT_TO_DESTINATION[args.task]
    unknown = sorted(set(rw.port) - set(mapping))
    if unknown:
        print(f"ERROR: reward ports {unknown} have no destination mapping for task "
              f"'{args.task}' (know {sorted(mapping)}).", file=sys.stderr)
        return 2
    rw["destination"] = rw.port.map(mapping)

    print(f"trials : {len(tr)}   rewards: {len(rw)}   tolerance: {args.tolerance} frames")
    print(f"trial frame range {tr.start_frame_global.min()}..{tr.end_frame_global.max()}   "
          f"reward frame range {rw.frame_raw.min()}..{rw.frame_raw.max()}")

    covered_lo = int(tr.start_frame_global.min())
    covered_hi = int(tr.end_frame_global.max()) + args.tolerance
    outside = [{"frame_raw": int(r.frame_raw), "port": r.port} for _, r in rw.iterrows()
               if not (covered_lo <= r.frame_raw <= covered_hi)]
    inrange = rw[(rw.frame_raw >= covered_lo) & (rw.frame_raw <= covered_hi)].reset_index(drop=True)

    # Globally optimal one-to-one matching rather than a greedy pass. Greedy in time order lets an
    # earlier reward claim the trial that chronologically belongs to a later one whenever two trials
    # to the same destination fall inside the tolerance window -- which happens readily on the
    # T-maze, where traversals are short and frequent. That is a silent wrong-result path.
    BIG = 10 ** 9
    cost = np.full((len(inrange), len(tr)), BIG, dtype=float)
    for i, r in inrange.iterrows():
        ok = ((tr.start_frame_global <= r.frame_raw)
              & (r.frame_raw <= tr.end_frame_global + args.tolerance)
              & (tr.destination == r.destination)).to_numpy()
        # Prefer containment, then the smallest gap.
        gap = (r.frame_raw - tr.end_frame_global).to_numpy()
        cost[i, ok] = np.where(gap[ok] <= 0, np.abs(gap[ok]), gap[ok] + args.tolerance)
    from scipy.optimize import linear_sum_assignment

    ri, ti = linear_sum_assignment(cost)
    assigned: dict[int, dict] = {}
    matched_rewards = set()
    for i, j in zip(ri, ti):
        if cost[i, j] >= BIG:
            continue
        r = inrange.iloc[i]
        t = tr.iloc[j]
        matched_rewards.add(i)
        assigned[int(t.trial_id)] = {"frame": int(r.frame_raw), "port": r.port,
                                     "gap": int(r.frame_raw - t.end_frame_global),
                                     "inside": bool(r.frame_raw <= t.end_frame_global)}
    unassigned = [{"frame_raw": int(inrange.iloc[i].frame_raw), "port": inrange.iloc[i].port}
                  for i in range(len(inrange)) if i not in matched_rewards]

    tr["rewarded"] = tr.trial_id.map(lambda i: i in assigned)
    tr["reward_frame_global"] = tr.trial_id.map(lambda i: assigned[i]["frame"] if i in assigned else pd.NA)
    tr["reward_port"] = tr.trial_id.map(lambda i: assigned[i]["port"] if i in assigned else pd.NA)
    tr["reward_gap_frames"] = tr.trial_id.map(lambda i: assigned[i]["gap"] if i in assigned else pd.NA)
    if "start_frame" in tr.columns:
        off = tr.start_frame_global - tr.start_frame  # trim start_frame, identical for all rows
        tr["reward_frame_local"] = tr.reward_frame_global - off

    tr.to_csv(args.output, index=False)
    gaps = np.array([a["gap"] for a in assigned.values()]) if assigned else np.array([0])
    print(f"\nassigned {len(assigned)}/{len(rw)} rewards to {int(tr.rewarded.sum())} of {len(tr)} trials")
    print(f"  unrewarded trials: {int((~tr.rewarded).sum())}")
    n_inside = sum(1 for a in assigned.values() if a["inside"])
    print(f"  rewards inside the trial span: {n_inside}; after the trial end: {len(assigned) - n_inside}")
    print(f"  offset trial-end -> reward: median {np.median(gaps):+.0f}, "
          f"range {gaps.min():+.0f}..{gaps.max():+.0f} frames (negative = inside the trial)")
    if outside:
        print(f"  {len(outside)} rewards fall outside the frames the behaviour pipeline covers "
              f"({covered_lo}..{covered_hi}) - the Bpod session outruns the trimmed window")
    if unassigned:
        print(f"  !! {len(unassigned)} rewards could not be attached to any trial:")
        for u in unassigned[:10]:
            print(f"       frame {u['frame_raw']} ({u['port']})")
    print(f"wrote  : {args.output}")

    report = {"task": args.task, "n_trials": len(tr), "n_rewards": len(rw),
              "n_assigned": len(assigned), "n_rewarded_trials": int(tr.rewarded.sum()),
              "n_unrewarded_trials": int((~tr.rewarded).sum()),
              "n_unassigned_rewards": len(unassigned), "unassigned": unassigned,
              "n_rewards_outside_covered_range": len(outside), "outside": outside,
              "n_inside_trial": n_inside,
              "gap_median": float(np.median(gaps)), "gap_max": int(gaps.max()),
              "tolerance": args.tolerance}

    if args.binned:
        bn = pd.read_csv(args.binned)
        if "trial_id" not in bn.columns:
            print("ERROR: binned table lacks 'trial_id'.", file=sys.stderr)
            return 2
        keep = ["trial_id", "rewarded", "reward_frame_global", "reward_port"]
        if "reward_frame_local" in tr.columns:
            keep.append("reward_frame_local")
        before = len(bn)
        bn = bn.merge(tr[keep], on="trial_id", how="left")
        if len(bn) != before:
            print(f"ERROR: merge changed row count {before} -> {len(bn)}; trial_id is not unique.",
                  file=sys.stderr)
            return 2
        out = args.binned_output or args.binned.replace(".csv", "_rewarded.csv")
        bn.to_csv(out, index=False)
        n_missing = int(bn.rewarded.isna().sum())
        print(f"binned : {len(bn)} rows, {bn.trial_id.nunique()} trials, "
              f"{int(bn.groupby('trial_id').rewarded.first().sum())} rewarded"
              + (f", {n_missing} rows with no trial match" if n_missing else ""))
        print(f"wrote  : {out}")
        report["binned_rows"] = len(bn)
        report["binned_rows_unmatched"] = n_missing

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote  : {args.report}")
    print("\n" + json.dumps({k: v for k, v in report.items() if k not in ("unassigned", "outside")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
