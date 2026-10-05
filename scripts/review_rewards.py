#!/usr/bin/env python
"""Human-in-the-loop review of individual reward events, to locate where a discrepancy comes from.

The automated gate can tell us a session disagrees with its Bpod log, but not *why*. Each
disagreement has two possible causes and they need opposite fixes:

===================  ==========================================  =========================
candidate            you see a reward flush in the frames        you see nothing
===================  ==========================================  =========================
``unassigned``       the reward is real -> TRIAL SEGMENTATION     reward detection false positive
                     failed to emit a trial for it
``missing``          -- (nothing was detected here) --            Bpod log is right and we
                                                                  missed it -> DETECTION gap
``extra_inside``     a real reward the Bpod log lacks             detection false positive
===================  ==========================================  =========================

So one human judgement per event -- "is there a water flush here, yes or no" -- separates a
segmentation problem from a detection problem. This script makes that judgement cheap.

Two steps
---------
``build``      picks candidate events, cuts a montage image for each (zoomed port ROI frame by frame,
               plus wide arena context), and writes ``review_sheet.csv`` with a blank ``verdict``
               column.
``summarize``  reads the sheet back once you have filled it in and reports the diagnosis.

Fill ``verdict`` with one of: ``reward``, ``no_reward``, ``unsure``. Leave blank to skip.

Usage
-----
    python review_rewards.py build --batch results/reward_linear_all \\
        --manifest docs/reward_session_manifest.csv --source unassigned \\
        --outdir results/reward_review --per-session 3 --max-events 40

    # ... open results/reward_review/*.png, fill in review_sheet.csv ...

    python review_rewards.py summarize --outdir results/reward_review
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("drf", os.path.join(HERE, "detect_reward_flushes.py"))
drf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drf)

VERDICTS = ("reward", "no_reward", "unsure")
N_ZOOM_FRAMES = 9      # frames shown around the event; the flush lasts 2-3
ZOOM = 6               # magnification of the port ROI crop
CONTEXT_PAD = 170      # px of arena around the port in the wide view


def rois_for(mouse_task: str, stamp: str) -> dict:
    """The committed ROI boxes for this session's arena and camera era."""
    if mouse_task == "Linear":
        return drf.PORT_ROIS[drf.epoch_for_session(f"x{stamp}.avi")]
    return drf.TMAZE_ARM_ROIS[drf.tmaze_era_for_session(f"x{stamp}.avi")]


def read_window(video: str, first: int, n: int) -> tuple[list, int]:
    """Decode ``n`` frames starting at ``first``, returning them and the index actually reached.

    OpenCV's ``CAP_PROP_POS_FRAMES`` seek is not exact on every mpeg4/AVI source, and this whole
    tool exists to let a human judge a specific frame -- silently showing the wrong frames would
    record a verdict against the wrong event. So the landed position is read back rather than
    assumed, and the caller labels tiles with the real index. ``verify_seek()`` cross-checks the
    seek against a sequential decode once per run.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return [], -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, (landed if landed >= 0 else first)


def verify_seek(video: str, frame: int) -> tuple[bool, str]:
    """Confirm the seek-based read returns the same pixels as a sequential decode to ``frame``."""
    seeked, landed = read_window(video, frame, 1)
    if not seeked:
        return False, "could not read the frame at all"
    cap = cv2.VideoCapture(video)
    ok, seq = True, None
    for _ in range(frame + 1):
        ok, seq = cap.read()
        if not ok:
            break
    cap.release()
    if not ok or seq is None:
        return False, "sequential decode ran out of frames"
    if landed != frame:
        return False, f"seek requested frame {frame} but landed on {landed}"
    diff = float(np.abs(seq.astype(np.int16) - seeked[0].astype(np.int16)).mean())
    return (diff < 1.0), f"mean |seek - sequential| = {diff:.2f} grey levels"


def montage(video: str, frame: int, box: tuple[int, int, int, int], title: str,
            subtitle: str, out_png: str) -> bool:
    """Frames around ``frame``: zoomed port ROI on top, wide arena context below."""
    first = max(frame - N_ZOOM_FRAMES // 2, 0)
    frames, landed = read_window(video, first, N_ZOOM_FRAMES)
    if not frames:
        return False
    first = landed

    x0, y0, x1, y1 = box
    H, W = frames[0].shape[:2]
    tiles = []
    for i, f in enumerate(frames):
        crop = f[max(y0 - 6, 0):min(y1 + 6, H), max(x0 - 6, 0):min(x1 + 6, W)]
        z = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                       interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(z, (6 * ZOOM, 6 * ZOOM),
                      (6 * ZOOM + (x1 - x0) * ZOOM, 6 * ZOOM + (y1 - y0) * ZOOM), (0, 0, 255), 2)
        # The number under each tile is what the detector actually thresholds on.
        val = np.percentile(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), 95)
        lab = np.zeros((26, z.shape[1], 3), np.uint8)
        n = first + i
        col = (0, 255, 255) if n == frame else (200, 200, 200)
        cv2.putText(lab, f"{n}  p95={val:.0f}", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        tiles.append(np.vstack([lab, z]))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for t in tiles]
    top = np.hstack(tiles)

    wide = []
    for i in (0, len(frames) // 2, len(frames) - 1):
        c = frames[i][max(y0 - CONTEXT_PAD, 0):min(y1 + CONTEXT_PAD, H),
                      max(x0 - CONTEXT_PAD, 0):min(x1 + CONTEXT_PAD, W)].copy()
        ox, oy = x0 - max(x0 - CONTEXT_PAD, 0), y0 - max(y0 - CONTEXT_PAD, 0)
        cv2.rectangle(c, (ox, oy), (ox + (x1 - x0), oy + (y1 - y0)), (0, 0, 255), 2)
        cv2.putText(c, f"frame {first + i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        wide.append(cv2.resize(c, (top.shape[1] // 3 - 4, int(c.shape[0] * (top.shape[1] / 3 - 4) / c.shape[1]))))
    hh = max(w.shape[0] for w in wide)
    wide = [cv2.copyMakeBorder(w, 0, hh - w.shape[0], 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for w in wide]
    bottom = np.hstack(wide)
    width = max(top.shape[1], bottom.shape[1])
    top = cv2.copyMakeBorder(top, 0, 0, 0, width - top.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    bottom = cv2.copyMakeBorder(bottom, 0, 0, 0, width - bottom.shape[1], cv2.BORDER_CONSTANT,
                               value=(0, 0, 0))
    head = np.zeros((66, width, 3), np.uint8)
    cv2.putText(head, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(head, subtitle, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1)
    cv2.imwrite(out_png, np.vstack([head, top, bottom]))
    return True


def trial_context(trials_csv: str, frame: int, trim_info: str = "") -> str:
    """What the behaviour pipeline thought was happening at this frame."""
    if not (trials_csv and os.path.exists(trials_csv)):
        return "no trial table"
    t = pd.read_csv(trials_csv)
    if not {"start_frame_global", "end_frame_global"} <= set(t.columns):
        # Older trial tables predate the *_global columns; they are just the local ones shifted by
        # the trim start frame, so reconstruct rather than giving up on the context.
        if trim_info and os.path.exists(trim_info) and {"start_frame", "end_frame"} <= set(t.columns):
            with open(trim_info) as fh:
                off = int(next(csv.DictReader(fh))["start_frame"])
            t["start_frame_global"] = t.start_frame + off
            t["end_frame_global"] = t.end_frame + off
        else:
            return "trial table lacks global frame columns"
    inside = t[(t.start_frame_global <= frame) & (frame <= t.end_frame_global)]
    if len(inside):
        r = inside.iloc[0]
        return f"INSIDE trial {r.origin}->{r.destination} ({r.start_frame_global}-{r.end_frame_global})"
    prev = t[t.end_frame_global <= frame]
    if not len(prev):
        return "before the first trial"
    r = prev.iloc[-1]
    return (f"{frame - int(r.end_frame_global)} frames after trial {r.origin}->{r.destination} "
            f"(ends {int(r.end_frame_global)})")


def unmatched_detections(summary_row, man_row, task: str) -> list[tuple[int, str, str]]:
    """Detections that pair with no Bpod reward, tagged inside/outside the log's window.

    Reuses the same fit-and-pair logic that produced the counts being reviewed, so the events shown
    are exactly the ones flagged rather than a stand-in for them.
    """
    spec = importlib.util.spec_from_file_location("rv", os.path.join(HERE, "refine_reward_verdicts.py"))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)

    rw = pd.read_csv(os.path.join(str(summary_row.outdir), "rewards.csv")).sort_values("frame_raw")
    vt = rv.video_seconds_rel_bpod(man_row.timestamps, man_row.bpod)
    bt, _ = rv.bpod_reward_times(man_row.bpod, task)
    if len(bt) < 4 or rw.empty:
        return []
    idx = np.clip(rw.frame_raw.to_numpy(), 0, len(vt) - 1)
    det_t = vt[idx]
    fit = drf.fit_clock(np.sort(det_t), bt)
    if fit["resid_sd_s"] * 1000 > rv.MAX_TRUSTED_RESID_MS:
        return []
    pred = fit["a"] + fit["b"] * bt
    order = np.argsort(det_t)
    used = np.zeros(len(det_t), bool)
    for p in pred:
        free = np.flatnonzero(~used[order])
        if not len(free):
            break
        j = free[np.argmin(np.abs(det_t[order][free] - p))]
        if abs(det_t[order][j] - p) <= rv.MATCH_TOL_S:
            used[order[j]] = True
    lo, hi = pred.min() - rv.EDGE_MARGIN_S, pred.max() + rv.EDGE_MARGIN_S
    out = []
    for i in np.flatnonzero(~used):
        r = rw.iloc[i]
        out.append((int(r.frame_raw), r.port,
                    "outside" if (det_t[i] < lo or det_t[i] > hi) else "inside"))
    return out


def collect(args) -> list[dict]:
    man = pd.read_csv(args.manifest).fillna("").set_index(["mouse", "task", "stamp"])
    summary = pd.read_csv(os.path.join(args.batch, "batch_summary.csv")).fillna("")
    refined_path = os.path.join(args.batch, "verdicts_refined.csv")
    refined = pd.read_csv(refined_path).fillna("") if os.path.exists(refined_path) else None

    out = []
    for _, s in summary.iterrows():
        try:
            m = man.loc[(s.mouse, s.task, s.stamp)]
        except KeyError:
            continue
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        if args.mouse and s.mouse != args.mouse:
            continue
        if args.task and s.task != args.task:
            continue
        picked: list[tuple[int, str, str]] = []  # (frame, port, why)

        if args.source == "unassigned":
            rep = os.path.join(str(s.outdir), "assign_report.json")
            if not os.path.exists(rep):
                continue
            for u in (json.load(open(rep)).get("unassigned") or []):
                picked.append((int(u["frame_raw"]), u["port"], "detected but attached to no trial"))

        elif args.source == "extra_inside":
            # Recompute the pairing so the human sees the events actually flagged as spurious.
            # Sampling the session's weakest detections instead would be a proxy: the pairing is by
            # time, not amplitude, so a low-amplitude event is just as likely to be a matched, real
            # reward -- and a verdict on it would answer a different question than the one asked.
            try:
                extras = unmatched_detections(s, man_row=m, task="linear" if s.task == "Linear"
                                              else "tmaze")
            except Exception:  # noqa: BLE001 - a session we cannot pair is simply not reviewable
                continue
            for frame, port, where in extras:
                if where == "inside":
                    picked.append((frame, port, "detected inside the Bpod window with no log entry"))

        elif args.source == "lowamp":
            rw = os.path.join(str(s.outdir), "rewards.csv")
            if not os.path.exists(rw) or "noise floor" not in str(s.reasons):
                continue
            df = pd.read_csv(rw)
            for _, e in df.nsmallest(args.per_session, "amplitude").iterrows():
                picked.append((int(e.frame_raw), e.port,
                               f"low-amplitude session (amp {e.amplitude:.0f})"))

        if not picked:
            continue
        rois = rois_for(s.task, s.stamp)
        for frame, port, why in picked[:args.per_session]:
            if port not in rois:
                continue
            era = (drf.epoch_for_session if s.task == "Linear" else drf.tmaze_era_for_session)(
                f"x{s.stamp}.avi")
            out.append(dict(mouse=s.mouse, task=s.task, stamp=s.stamp, era=era, source=args.source,
                            frame_raw=frame, port=port, why=why,
                            video=m.raw_video, trials=m.trials, trim_info=m.trim_info,
                            box=rois[port], outdir=str(s.outdir)))
    if not args.max_events or len(out) <= args.max_events:
        return out
    # Round-robin across (mouse, era) rather than truncating in batch order. Truncating alphabetically
    # gave 42% of a 36-event sample to one mouse and zero events to Mouse946 and to the post-2026-05
    # camera era entirely -- a "% segmentation vs % detection" figure read off that would describe two
    # aging-cohort mice, not the dataset.
    buckets: dict[tuple, list] = {}
    for e in out:
        buckets.setdefault((e["mouse"], e["era"]), []).append(e)
    picked, keys = [], list(buckets)
    while len(picked) < args.max_events and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < args.max_events:
                picked.append(buckets[k].pop(0))
    return picked


def cmd_build(args) -> int:
    os.makedirs(args.outdir, exist_ok=True)
    events = collect(args)
    print(f"{len(events)} candidate events to review")
    if not events:
        print("nothing matched; check --source / --batch")
        return 1
    # One-off check that seeking returns the same pixels as decoding from the start. If it does not,
    # every frame label in every montage is suspect and a human verdict would be recorded against the
    # wrong event, so stop rather than produce a sheet that looks fine.
    probe = events[0]
    ok, detail = verify_seek(probe["video"], probe["frame_raw"])
    print(f"seek self-test on {os.path.basename(probe['video'])} frame {probe['frame_raw']}: "
          f"{'OK' if ok else 'FAILED'} ({detail})")
    if not ok and not args.ignore_seek_check:
        print("ERROR: frame seeking is not exact on this source; montage labels cannot be trusted.\n"
              "       Re-run with --ignore-seek-check only if you accept that risk.", file=sys.stderr)
        return 2

    rows = []
    for i, e in enumerate(events, 1):
        png = os.path.join(args.outdir,
                           f"{i:03d}_{e['mouse']}_{e['stamp']}_f{e['frame_raw']}_{e['port']}.png")
        ctx = trial_context(e["trials"], e["frame_raw"], e.get("trim_info", ""))
        ok = montage(e["video"], e["frame_raw"], e["box"],
                     f"{e['mouse']} {e['task']} {e['stamp']}  frame {e['frame_raw']}  {e['port']}",
                     f"{e['why']}  |  behaviour pipeline: {ctx}", png)
        if not ok:
            print(f"  [{i}] could not read {e['video']}")
            continue
        rows.append({k: e[k] for k in ("mouse", "task", "stamp", "era", "source", "frame_raw",
                                       "port", "why")}
                    | {"trial_context": ctx, "image": os.path.basename(png),
                       "verdict": "", "notes": ""})
        print(f"  [{i}/{len(events)}] {os.path.basename(png)}")

    sheet = os.path.join(args.outdir, "review_sheet.csv")
    pd.DataFrame(rows).to_csv(sheet, index=False)
    print(f"\nwrote {len(rows)} rows -> {sheet}")
    print(f"images in {args.outdir}/")
    print(f"\nOpen the PNGs, then fill the 'verdict' column with one of: {', '.join(VERDICTS)}")
    print("Top row = the port ROI frame by frame (yellow label marks the event frame, p95 is the")
    print("number the detector thresholds on). Bottom row = wider arena view for context.")
    print(f"\nThen: python {os.path.basename(__file__)} summarize --outdir {args.outdir}")
    return 0


def cmd_summarize(args) -> int:
    sheet = os.path.join(args.outdir, "review_sheet.csv")
    if not os.path.exists(sheet):
        print(f"no review sheet at {sheet}", file=sys.stderr)
        return 2
    d = pd.read_csv(sheet).fillna("")
    d["verdict"] = d.verdict.astype(str).str.strip().str.lower()
    bad = sorted(set(d.verdict) - set(VERDICTS) - {""})
    if bad:
        print(f"WARNING: unrecognised verdicts ignored: {bad}\n")
    done = d[d.verdict.isin(VERDICTS)]
    print(f"reviewed {len(done)} of {len(d)} events\n")
    if not len(done):
        return 0
    print(pd.crosstab(done.source, done.verdict).to_string(), "\n")

    for source, grp in done.groupby("source"):
        n_rew = int((grp.verdict == "reward").sum())
        n_no = int((grp.verdict == "no_reward").sum())
        tot = n_rew + n_no
        if not tot:
            continue
        print(f"--- {source} ({tot} decided) ---")
        if source == "unassigned":
            print(f"  {n_rew} real rewards the trial table had no trial for  -> SEGMENTATION problem")
            print(f"  {n_no} detections with no visible flush                -> DETECTION false positives")
            print(f"  => {100 * n_rew / tot:.0f}% of the unattached rewards are segmentation, "
                  f"{100 * n_no / tot:.0f}% are detection")
        elif source == "lowamp":
            print(f"  {n_rew} genuine flushes among this session's WEAKEST detections -> the "
                  "amplitude gate is too strict")
            print(f"  {n_no} with no visible flush                                    -> real "
                  "false positives")
            print("  note: these are each session's lowest-amplitude detections, so this is the "
                  "harshest\n        test of the threshold, not a random sample of its detections.")
        elif source == "extra_inside":
            print(f"  {n_rew} real rewards the Bpod log has no entry for -> the log is incomplete")
            print(f"  {n_no} with no visible flush                       -> detection false positives")
        print()
    per = done.groupby("mouse").verdict.value_counts().unstack(fill_value=0)
    print("by mouse:"); print(per.to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="cut montages and write a blank review sheet")
    b.add_argument("--batch", required=True, help="a reward batch dir with batch_summary.csv")
    b.add_argument("--manifest", required=True)
    b.add_argument("--outdir", required=True)
    b.add_argument("--source", required=True,
                   choices=["unassigned", "lowamp", "extra_inside"],
                   help="which kind of questionable event to review")
    b.add_argument("--per-session", type=int, default=3, help="events per session (default 3)")
    b.add_argument("--max-events", type=int, default=60, help="stop after this many (default 60)")
    b.add_argument("--mouse")
    b.add_argument("--task", choices=["Linear", "TMaze"])
    b.add_argument("--ignore-seek-check", action="store_true",
                   help="build montages even if the seek self-test fails (labels may be wrong)")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("summarize", help="read the filled sheet and report the diagnosis")
    s.add_argument("--outdir", required=True)
    s.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
