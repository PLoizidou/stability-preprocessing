#!/usr/bin/env python
"""Run reward extraction (and the trial join) over many sessions, with an explicit pass/fail gate.

Reads the manifest from ``build_reward_manifest.py``, runs ``detect_reward_flushes.py`` and then
``assign_rewards_to_trials.py`` per session, and writes one summary row each. Every session gets a
verdict rather than output that has to be read by eye -- the failure mode of this pipeline is a
silent wrong answer with a zero exit code, so nothing counts as done unless it passes the gate.

Each session runs in its own subprocess, so a crash or hang in one does not take the batch with it.

Usage
-----
    python run_reward_batch.py --manifest docs/reward_session_manifest.csv \\
        --outdir results/reward_batch --sample 20
    python run_reward_batch.py --manifest ... --outdir ... --mouse Mouse945 --task TMaze
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Gate thresholds. Deliberately strict: a session that trips one of these is set aside for review,
# not silently accepted.
MAX_RESID_SD_S = 0.15      # ~3-4 frames; measured 0.06-0.07 s on good sessions
MIN_ANCHORS_MATCHED = 8
MAX_ABS_DRIFT_PPM = 1000.0  # measured +136..+200 ppm; an order of magnitude beyond that is a bad fit
# Healthy linear sessions separate genuine flushes from the noise floor by a wide margin (median
# accepted amplitude 68-126 against a threshold of 20). A session sitting close to threshold is
# detecting noise: Mouse943 2025-12-18 reported 929 rewards against 35 expected with a median of 30.
MIN_AMPLITUDE_RATIO = 2.5


def gate(det: dict | None, asg: dict | None, route: str) -> tuple[str, list[str]]:
    """Turn the two summaries into PASS/FAIL plus the specific reasons."""
    why: list[str] = []
    unverified = False
    if det is None:
        return "FAIL", ["detection did not produce a summary"]

    if route == "detect":
        for port, sep in (det.get("amplitude_separation") or {}).items():
            med, thr = sep.get("median_accepted"), sep.get("threshold")
            if med is not None and thr and med < MIN_AMPLITUDE_RATIO * thr:
                why.append(f"{port} amplitudes near the noise floor (median {med:.0f} vs "
                           f"threshold {thr:.0f})")
        bp = det.get("bpod")
        if bp:
            if det["n_detected"] != bp["n_rewards_expected"]:
                why.append(f"detected {det['n_detected']} vs {bp['n_rewards_expected']} expected")
            oc = det.get("order_check")
            if isinstance(oc, dict) and (oc["extra_detected"] or oc["missing"] or oc["substituted"]):
                why.append(f"order check: {oc['extra_detected']} extra, {oc['missing']} missing, "
                           f"{oc['substituted']} wrong port")
        else:
            unverified = True
    else:
        fit = det.get("clock_fit")
        if not fit:
            why.append("no clock fit")
        else:
            if fit["n_matched"] < MIN_ANCHORS_MATCHED:
                why.append(f"only {fit['n_matched']} anchors matched")
            if fit["resid_sd_s"] > MAX_RESID_SD_S:
                why.append(f"alignment residual SD {fit['resid_sd_s'] * 1000:.0f} ms")
            if abs(fit["drift_ppm"]) > MAX_ABS_DRIFT_PPM:
                why.append(f"implausible drift {fit['drift_ppm']:+.0f} ppm")

    if asg is not None and asg.get("n_unassigned_rewards", 0) > 0:
        why.append(f"{asg['n_unassigned_rewards']} rewards unattached to any trial")
    if why:
        return "FAIL", why
    # A session with no Bpod log cannot be count-verified, but the detection itself may be perfectly
    # good. Calling that FAIL would condemn all 26 no-log linear sessions; it needs its own verdict.
    return ("UNVERIFIED" if unverified else "PASS"), (["no Bpod log: count unverified"] if unverified else [])


def run(cmd: list[str], log_path: str, timeout: int) -> int:
    with open(log_path, "w") as fh:
        try:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\nTIMEOUT after {timeout}s\n")
            return 124


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", type=int, help="take up to N sessions spread across every "
                                               "(cohort, task, era) combination")
    ap.add_argument("--mouse")
    ap.add_argument("--task", choices=["Linear", "TMaze"])
    ap.add_argument("--timeout", type=int, default=900, help="per-session seconds (default 900)")
    ap.add_argument("--require-trials", action="store_true",
                    help="only run sessions that also have a behaviour trial table")
    ap.add_argument("--jobs", type=int, default=1,
                    help="sessions to process concurrently. The work is ffmpeg decoding plus reads "
                         "from external drives, so a handful of workers helps; too many will thrash "
                         "a single drive (default 1)")
    ap.add_argument("--resume", action="store_true",
                    help="skip sessions that already have a detect_summary.json in --outdir")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest).fillna("")
    df = df[df.ready == True]  # noqa: E712  (column is a real bool from to_csv round-trip)
    if args.mouse:
        df = df[df.mouse == args.mouse]
    if args.task:
        df = df[df.task == args.task]
    if args.require_trials:
        df = df[df.can_assign == True]  # noqa: E712
    if args.sample:
        groups = list(df.groupby(["cohort", "task", "era"]))
        per = max(1, args.sample // max(len(groups), 1))
        picked = []
        for _, g in groups:
            # Prefer sessions that also exercise the trial join.
            g = g.sort_values("can_assign", ascending=False)
            picked.append(g.head(per))
        df = pd.concat(picked).head(args.sample)

    os.makedirs(args.outdir, exist_ok=True)
    summary_csv = os.path.join(args.outdir, "batch_summary.csv")
    prior = pd.DataFrame()
    if args.resume:
        # A session counts as done only if it has a verdict row from a previous run. Keying off
        # detect_summary.json alone would permanently skip a session that was killed during the
        # assign step: the detect summary exists, but the session never got a verdict.
        if os.path.exists(summary_csv):
            prior = pd.read_csv(summary_csv).fillna("")
            done_keys = set(zip(prior.mouse, prior.task, prior.stamp))
            before = len(df)
            df = df[[(r.mouse, r.task, r.stamp) not in done_keys for r in df.itertuples()]]
            print(f"resume: skipping {before - len(df)} sessions that already have a verdict")
        else:
            print("resume: no previous batch_summary.csv found; running everything")
    print(f"{len(df)} sessions to run on {args.jobs} worker(s)\n")
    print(df.groupby(["cohort", "task", "era"]).size().to_string(), "\n")

    done = [0]
    lock = threading.Lock()

    def process(r):
        tag = f"{r.mouse}_{r.task}_{r.stamp}"
        sub = os.path.join(args.outdir, tag)
        os.makedirs(sub, exist_ok=True)
        rewards = os.path.join(sub, "rewards.csv")
        det_json = os.path.join(sub, "detect_summary.json")
        asg_json = os.path.join(sub, "assign_report.json")
        t0 = time.time()

        cmd = [sys.executable, os.path.join(HERE, "detect_reward_flushes.py"), r.raw_video,
               "--task", "linear" if r.task == "Linear" else "tmaze",
               "--mode", r.route, "-o", rewards, "--timestamps", r.timestamps,
               "--summary-json", det_json, "--qc-plot", os.path.join(sub, "qc.png")]
        if r.bpod:
            cmd += ["--bpod", r.bpod]
        if r.trim_info:
            cmd += ["--trim-info", r.trim_info]
        if r.trim_video:
            cmd += ["--trim-video", r.trim_video]
        rc = run(cmd, os.path.join(sub, "detect.log"), args.timeout)
        det = json.load(open(det_json)) if os.path.exists(det_json) else None

        asg, rc2 = None, None
        if det is not None and r.trials:
            cmd2 = [sys.executable, os.path.join(HERE, "assign_rewards_to_trials.py"),
                    "--rewards", rewards, "--trials", r.trials,
                    "--task", "linear" if r.task == "Linear" else "tmaze",
                    "-o", os.path.join(sub, "trials_rewarded.csv"), "--report", asg_json]
            if r.trim_info:
                cmd2 += ["--trim-info", r.trim_info]
            if r.binned:
                cmd2 += ["--binned", r.binned,
                         "--binned-output", os.path.join(sub, "binned_rewarded.csv")]
            rc2 = run(cmd2, os.path.join(sub, "assign.log"), args.timeout)
            asg = json.load(open(asg_json)) if os.path.exists(asg_json) else None

        verdict, why = gate(det, asg, r.route)
        if rc != 0:
            verdict, why = "FAIL", [f"detect exited {rc}"] + why
        if rc2 not in (None, 0):
            verdict, why = "FAIL", [f"assign exited {rc2}"] + why

        fit = (det or {}).get("clock_fit") or {}
        row = dict(
            mouse=r.mouse, task=r.task, stamp=r.stamp, era=r.era, cohort=r.cohort, route=r.route,
            verdict=verdict, reasons="; ".join(why),
            n_rewards=(det or {}).get("n_detected"),
            n_expected=((det or {}).get("bpod") or {}).get("n_rewards_expected"),
            anchors_matched=fit.get("n_matched"), anchors_total=fit.get("n_anchors"),
            resid_sd_ms=round(fit["resid_sd_s"] * 1000, 1) if fit.get("resid_sd_s") is not None else None,
            drift_ppm=round(fit["drift_ppm"]) if fit.get("drift_ppm") is not None else None,
            n_trials=(asg or {}).get("n_trials"), n_rewarded=(asg or {}).get("n_rewarded_trials"),
            n_unassigned=(asg or {}).get("n_unassigned_rewards"),
            n_outside=(asg or {}).get("n_rewards_outside_covered_range"),
            secs=round(time.time() - t0), outdir=sub)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(df)}] {verdict:10s} {tag}  ({row['secs']}s)"
                  + (f"  -- {row['reasons']}" if why else ""), flush=True)
        return row

    if args.jobs > 1:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(process, [r for r in df.itertuples()]))
    else:
        rows = [process(r) for r in df.itertuples()]

    out = pd.DataFrame(rows)
    # Carry forward verdicts from earlier invocations, otherwise a resumed run would overwrite the
    # summary with only the sessions it happened to process this time.
    if len(prior):
        out = pd.concat([prior, out], ignore_index=True)
        out = out.drop_duplicates(subset=["mouse", "task", "stamp"], keep="last")
    out.to_csv(summary_csv, index=False)
    print(f"\n=== {int((out.verdict == 'PASS').sum())} PASS, "
          f"{int((out.verdict == 'UNVERIFIED').sum())} UNVERIFIED, "
          f"{int((out.verdict == 'FAIL').sum())} FAIL of {len(out)} ===")
    print(out.groupby(["cohort", "task", "era", "verdict"]).size().to_string())
    if (out.verdict == "FAIL").any():
        print("\nfailures:")
        print(out[out.verdict == "FAIL"][["mouse", "task", "stamp", "era", "reasons"]]
              .to_string(index=False, max_colwidth=70))
    print(f"\nwrote: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
