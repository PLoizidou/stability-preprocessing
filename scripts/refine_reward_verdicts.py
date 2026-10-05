#!/usr/bin/env python
"""Re-score a reward batch without re-decoding any video.

``run_reward_batch.py`` compares the number of detections across the WHOLE raw video against a Bpod
log that only covers its own session. Those are different windows: the recording routinely starts
before the protocol is launched and runs after it stops, and on restart days an earlier aborted run
delivers real rewards the surviving log knows nothing about. A detection outside the log's window is
therefore expected, not an error -- but the batch gate counted it as one.

This reads each session's saved ``rewards.csv`` plus its Bpod log and ``timestamps<TS>.csv``, fits
the clock from the detections themselves, and splits the surplus into

* **inside** the Bpod reward window  -> a genuine false positive, still a failure;
* **outside** it                     -> a reward the log does not cover, reported but not a failure.

Nothing is re-detected and no session output is modified; only the verdict changes.

Usage
-----
    python refine_reward_verdicts.py --batch results/reward_linear_all \\
        --manifest docs/reward_session_manifest.csv -o results/reward_linear_all/verdicts_refined.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("drf", os.path.join(HERE, "detect_reward_flushes.py"))
drf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drf)

# How far outside the Bpod window a detection must fall to count as "not covered by this log".
# Measured alignment residuals are 20-30 ms SD, so 0.25 s is ~10 sigma -- generous enough to absorb
# fit jitter without excusing a detection that genuinely sits inside the session.
EDGE_MARGIN_S = 0.25

# Tolerance for pairing one detection with one Bpod reward once the clock is fitted.
MATCH_TOL_S = 0.3

# A fit worse than this is not trustworthy enough to decide which individual detections are spurious.
# Good sessions land at 20-30 ms; the handful that fail come in at 78-264 ms, and on those the
# pairing collapses (4-7 of ~45 events matched) even though the session's total count matches Bpod
# exactly. Reporting those as "missing rewards" would invent failures that are not there.
MAX_TRUSTED_RESID_MS = 60.0


def bpod_reward_times(mat_path: str, task: str) -> tuple[np.ndarray, list[str]]:
    from scipy.io import loadmat

    sd = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["SessionData"]
    tst = np.atleast_1d(sd.TrialStartTimestamp).astype(float)
    types = np.atleast_1d(sd.TrialTypes).astype(int)
    table = drf.TRIALTYPE_TO_PORT if task == "linear" else drf.TMAZE_TRIALTYPE_TO_ARM
    times, ports = [], []
    for i, t in enumerate(np.atleast_1d(sd.RawEvents.Trial)):
        r = np.array(t.States.Reward, dtype=float).ravel()
        if np.isnan(r).all():
            continue
        times.append(tst[i] + r[0])
        ports.append(table[int(types[i])])
    order = np.argsort(times)
    return np.array(times)[order], [ports[i] for i in order]


def video_seconds_rel_bpod(timestamps_csv: str, mat_path: str) -> np.ndarray:
    raw = pd.read_csv(timestamps_csv, header=None, names=["t"], dtype=str)["t"]
    t = pd.to_datetime(raw.str[:26], format="mixed")
    m = re.search(r"_(\d{8})_(\d{6})\.mat$", os.path.basename(mat_path))
    b0 = pd.Timestamp(f"{m.group(1)} {m.group(2)}")
    return (t - b0).dt.total_seconds().to_numpy()


def refine(session_dir: str, row: pd.Series, task: str) -> dict:
    out = {"n_extra_inside": None, "n_extra_outside": None, "n_missing": None,
           "n_matched": None, "clock_offset_s": None, "resid_sd_ms": None, "note": ""}
    rewards_csv = os.path.join(session_dir, "rewards.csv")
    if not (os.path.exists(rewards_csv) and row.bpod and row.timestamps):
        out["note"] = "no rewards.csv or no Bpod log"
        return out
    rw = pd.read_csv(rewards_csv).sort_values("frame_raw")
    if rw.empty:
        out["note"] = "no detections"
        return out
    try:
        vt = video_seconds_rel_bpod(row.timestamps, row.bpod)
        bt, _ = bpod_reward_times(row.bpod, task)
    except Exception as e:  # noqa: BLE001 - a bad log should not stop the sweep
        out["note"] = f"could not read log/timestamps: {e}"
        return out
    if len(bt) < 4:
        out["note"] = f"only {len(bt)} Bpod rewards; too few to fit"
        return out

    idx = np.clip(rw.frame_raw.to_numpy(), 0, len(vt) - 1)
    det_t = vt[idx]
    try:
        fit = drf.fit_clock(np.sort(det_t), bt)
    except Exception as e:  # noqa: BLE001
        out["note"] = f"clock fit failed: {e}"
        return out
    out["clock_offset_s"] = round(fit["a"], 3)
    out["resid_sd_ms"] = round(fit["resid_sd_s"] * 1000, 1)
    if out["resid_sd_ms"] > MAX_TRUSTED_RESID_MS:
        out["note"] = (f"clock fit unreliable (residual SD {out['resid_sd_ms']:.0f} ms); "
                       "per-event pairing not attempted, original verdict kept")
        return out

    # Pair individual detections with individual Bpod rewards rather than differencing the counts.
    # Counting alone conflates the two error directions: a session with one benign extra outside the
    # window AND one genuine false positive inside it nets out to a surplus of two, and attributing
    # both to the benign bucket would hide the real one. It also cannot see a miss at all.
    pred = fit["a"] + fit["b"] * bt
    det_sorted = np.sort(det_t)
    used = np.zeros(len(det_sorted), bool)
    matched_bpod = np.zeros(len(pred), bool)
    for k, p in enumerate(pred):
        free = np.flatnonzero(~used)
        if not len(free):
            break
        j = free[np.argmin(np.abs(det_sorted[free] - p))]
        if abs(det_sorted[j] - p) <= MATCH_TOL_S:
            used[j] = True
            matched_bpod[k] = True

    lo = pred.min() - EDGE_MARGIN_S
    hi = pred.max() + EDGE_MARGIN_S
    extras = det_sorted[~used]
    out["n_matched"] = int(used.sum())
    out["n_missing"] = int((~matched_bpod).sum())
    out["n_extra_outside"] = int(((extras < lo) | (extras > hi)).sum())
    out["n_extra_inside"] = int(len(extras) - out["n_extra_outside"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="directory containing batch_summary.csv")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    summary = pd.read_csv(os.path.join(args.batch, "batch_summary.csv")).fillna("")
    man = pd.read_csv(args.manifest).fillna("")
    key = ["mouse", "task", "stamp"]
    man = man.set_index(key)

    rows = []
    for _, s in summary.iterrows():
        try:
            m = man.loc[(s.mouse, s.task, s.stamp)]
        except KeyError:
            rows.append({**s.to_dict(), "note": "not in manifest"})
            continue
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        task = "linear" if s.task == "Linear" else "tmaze"
        info = refine(s.outdir, m, task) if s.route == "detect" else {
            "n_extra_inside": None, "n_extra_outside": None, "n_missing": None, "n_matched": None,
            "clock_offset_s": None, "resid_sd_ms": None,
            "note": "align mode: rewards come from the log, no surplus possible"}

        verdict, reasons = s.verdict, s.reasons
        if s.route == "detect" and info["n_extra_inside"] is not None:
            # Replace the whole-video count/order complaints with per-event ones. Both directions are
            # re-stated: a MISSED reward is a worse error than a spurious one and must never be
            # excused just because this script's job is to forgive out-of-window surplus.
            kept = [r for r in str(reasons).split("; ")
                    if r and not r.startswith("detected ") and not r.startswith("order check")]
            if info["n_missing"]:
                kept.insert(0, f"{info['n_missing']} Bpod reward(s) with no matching detection")
            if info["n_extra_inside"]:
                kept.insert(0, f"{info['n_extra_inside']} extra detection(s) inside the Bpod window")
            reasons = "; ".join(kept)
            if s.verdict == "FAIL" and not kept:
                verdict = "PASS"
        rows.append({**s.to_dict(), **info, "verdict_refined": verdict, "reasons_refined": reasons})

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    before = summary.verdict.value_counts().to_dict()
    after = out.verdict_refined.value_counts().to_dict() if "verdict_refined" in out else {}
    print("verdicts before:", before)
    print("verdicts after :", after)
    changed = out[out.verdict != out.get("verdict_refined", out.verdict)]
    print(f"\n{len(changed)} sessions reclassified")
    if len(changed):
        print(changed[["mouse", "task", "stamp", "verdict", "verdict_refined",
                       "n_extra_outside", "n_extra_inside", "n_missing"]].head(20).to_string(index=False))
    if "n_extra_outside" in out:
        tot = pd.to_numeric(out.n_extra_outside, errors="coerce")
        print(f"\ndetections outside the Bpod window: {int(tot.fillna(0).sum())} across "
              f"{int((tot > 0).sum())} sessions")
    print(f"\nwrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
