#!/usr/bin/env python
"""Detect reward deliveries in a linear-track behaviour video from the water-port LED flush.

Every time the Bpod ``Reward`` state fires, the valve opens and the port LED lights for the valve
duration (~70-120 ms, i.e. 2-3 frames at 25 fps).  On the linear track the LED is off at all other
times, so a brief saturating transient inside a port ROI is a reward.

Detection runs on the **raw** behaviour video (before trimming and before any denoising), so the
reported frame indices are raw-video frame indices -- the same indexing as ``timestamps<TS>.csv`` and
the DLC csvs, and stable against re-trimming.  Convert to the trimmed clip with
``frame_local = frame_raw - dlc_trim_info.start_frame``.

The Bpod ``.mat`` log is used **only** to state how many rewards were expected (total and per port);
it is never used to time an individual event.

Usage
-----
    python detect_reward_flushes.py <Linear*.avi> --epoch post2026-05 -o rewards.csv
    python detect_reward_flushes.py <Linear*.avi> --epoch post2026-05 -o rewards.csv \\
        --bpod ".../Mouse944_LinearTrack_20260713_160106.mat" \\
        --trim-info ".../dlc_trim_info.csv" --qc-plot qc.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np

# --------------------------------------------------------------------------------------------
# Port ROIs, full-resolution (1936x1464) pixel coordinates of the LED, per camera epoch.
#
# The camera is fixed between recordings, so one ROI set covers every mouse within an epoch.  It was
# re-aimed around May 2026, hence the second entry (used by Mouse944 and Mouse945-Round2).
#
# Each entry is  port_name -> (x0, y0, x1, y1), a half-open box in full-resolution pixels.
# Boxes are deliberately tight around the LED rather than around the whole port box: a wider ROI
# dilutes the flush against the port walls and the mouse's fur.
# Port names must match the Bpod port numbers in LinearTrack.m: trial type 1 -> Port4, type 2 -> Port2.
# --------------------------------------------------------------------------------------------
PORT_ROIS = {
    # Post-May-2026 camera position (Mouse944, Mouse945-Round2). Upper track. Boxes supplied by the
    # project owner off results/reward_port_rois/Mouse944_2026-07-13_ports_zoom_grid.png.
    # Validated on Mouse944 2026-07-13: 78/78 rewards, 39/39 per port, strictly alternating, port
    # identity matching Bpod on every event, lag SD 0.04 s.
    "post2026-05": {"Port4": (190, 695, 215, 715), "Port2": (1785, 640, 1805, 660)},
    # Pre-May-2026 camera position. Boxes supplied by the project owner off the marked-up reference
    # frame (results/reward_port_rois/Mouse945_2026-02-03_ports_zoom.png).
    "pre2026-05": {"Port4": (180, 650, 210, 670), "Port2": (1775, 585, 1795, 605)},
}

# Every port box seen in the pre-May frame, for the survey run that decides which of the two linear
# tracks a mouse ran. Not used for detection once the track is known.
PRE2026_05_ALL_PORTS = {
    "upper_left":  (180, 650, 210, 670),
    "upper_right": (1775, 585, 1795, 605),
    "lower_left":  (225, 1020, 245, 1040),
    "lower_right": (1765, 960, 1785, 980),
}

# The 2024 aging-cohort linear frames were checked against both sets (see
# results/reward_port_rois/Linear_aging2024_vs_committed_ROIs.png): the pre-May boxes land on the
# spouts and the post-May boxes sit below them, so 2024 linear sessions use "pre2026-05". The linear
# track is a fixed installation, so it has only these two camera epochs.

# --------------------------------------------------------------------------------------------
# T-MAZE arm ROIs. NOT YET WIRED INTO DETECTION -- this script is linear-track only. Recorded here
# so the coordinates live with the code rather than in a chat log.
#
# Unlike the linear track, the T-maze apparatus is swapped in and out of the room, so its ports move
# independently of the camera: 2024 needs its own set even though 2024 *linear* does not.
#
# The maze has four arms but only three ports. The TOP-CENTRE arm is the home cage and is not a
# reward port -- never place an ROI there.
#
# Boxes supplied by the project owner off the numbered zoom grids in results/reward_port_rois/.
# --------------------------------------------------------------------------------------------
TMAZE_ARM_ROIS = {
    "aging2024":   {"left": (145, 470, 165, 490), "right": (1720, 510, 1740, 530),
                    "centre": (920, 1320, 940, 1340)},
    "pre2026-05":  {"left": (135, 440, 155, 460), "right": (1710, 490, 1740, 510),
                    "centre": (910, 1290, 930, 1310)},
    "post2026-05": {"left": (125, 440, 145, 460), "right": (1725, 460, 1745, 480),
                    "centre": (940, 1265, 960, 1285)},
}

# Both T-maze protocols map trial type to the same physical arm; only the Bpod port *numbers* differ
# (TMaze2: middle=1 left=2 right=3; TMaze4: middle=1 left=3 right=5). So an arm-based audit needs no
# protocol lookup. Which physical arm the protocol calls "left" still has to be confirmed against the
# video, exactly as it was for the linear track -- do not assume the labels match.
TMAZE_TRIALTYPE_TO_ARM = {1: "centre", 2: "left", 3: "right"}


EPOCH_CUTOFF = "2026-05-01"

# The MAD-based automatic threshold underestimates this signal's heavy tail, so a floor is applied.
# With 64 px ROIs the residual noise excursions reach ~13 while genuine flushes are >=40 on the
# dimmer port and >=119 on the brighter one, so 20 sits inside that gap. Validated on two sessions
# (Mouse944 2026-07-13, Mouse945 2026-02-03); check `amplitude_separation` in the run summary before
# trusting it on a new mouse or a new camera position.
DEFAULT_THRESHOLD_FLOOR = 20.0

# Resolution the PORT_ROIS coordinates were measured on. A video of any other size would still crop
# successfully but would point at the wrong physical location, so it is checked rather than assumed.
REF_WIDTH, REF_HEIGHT = 1936, 1464

# Minimum anchor events needed before a clock fit is trusted in align mode.
MIN_ANCHORS = 8

# Guards on the anchor LED trace used by align mode.
MIN_LED_CONTRAST = 30.0     # grey levels between the off and on states
MAX_LED_ON_FRACTION = 0.60  # above this the percentile threshold is not trustworthy

# Video frames may exceed timestamps.csv rows by this much before it is treated as an error.
# Concatenated sessions can lose a trailing timestamp row per spliced segment, so a handful of
# surplus frames is normal. Only a SURPLUS is tolerated, never a deficit, and the actual figure is
# printed -- a genuinely mismatched pairing differs by far more than this.
TIMESTAMP_SURPLUS_TOLERANCE = 32

# Bpod LinearTrack.m: case 1 -> CorrectPort Port4In ("left"), case 2 -> Port2In ("right").
TRIALTYPE_TO_PORT = {1: "Port4", 2: "Port2"}


def epoch_for_session(video_path: str) -> str:
    """Pick the camera epoch from the ISO timestamp embedded in the filename."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})T\d{2}_\d{2}_\d{2}", os.path.basename(video_path))
    if not m:
        raise ValueError(f"no timestamp in filename: {video_path}")
    return "pre2026-05" if m.group(1) < EPOCH_CUTOFF else "post2026-05"


def tmaze_era_for_session(video_path: str) -> str:
    """Camera/apparatus era for a T-MAZE session -- three eras, unlike the linear track's two.

    Deliberately separate from :func:`epoch_for_session`. The linear track is a fixed installation
    with two camera positions, and its 2024 sessions share the pre-May-2026 boxes; the T-maze is
    swapped in and out of the room, so its ports moved between 2024 and 2026 as well. Using the
    linear resolver for a T-maze session would silently map a 2024 recording to "pre2026-05" and
    read the wrong arm boxes.
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})T\d{2}_\d{2}_\d{2}", os.path.basename(video_path))
    if not m:
        raise ValueError(f"no timestamp in filename: {video_path}")
    date = m.group(1)
    if date < "2025-01-01":
        return "aging2024"
    return "pre2026-05" if date < EPOCH_CUTOFF else "post2026-05"


def probe_video(video: str) -> tuple[int, int, int]:
    """Return (width, height, nb_frames) from the container metadata."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h, n = out.split(",")[:3]
    return int(w), int(h), int(n)


def _even_down(v: int) -> int:
    return v - (v % 2)


def _even_up(v: int) -> int:
    return v + (v % 2)


def extract_roi_stack(video: str, rois: dict[str, tuple[int, int, int, int]],
                      width: int, height: int) -> dict[str, np.ndarray]:
    """Decode the video once, cropping every ROI in ffmpeg and returning one uint8 stack per port.

    Cropping server-side avoids moving full 1936x1464 frames into Python; this runs at ~1200 fps
    versus ~175 fps for a per-frame OpenCV loop, so a 42k-frame session takes well under a minute.

    ROI boxes may differ in size, but ``hstack`` needs a common shape, so each is decoded inside a
    uniform tile anchored near its own top-left corner and the exact box is sliced out afterwards.

    **All tile sizes and offsets are forced even.** ffmpeg silently rounds odd crop dimensions down
    on chroma-subsampled input (asking for 25 px wide yields 24), which would misalign the reshape
    and corrupt every frame index without any error. The real bytes-per-frame is probed from a
    one-frame decode rather than assumed, so a future surprise fails loudly instead of silently.
    """
    names = sorted(rois)
    for n in names:
        x0, y0, x1, y1 = rois[n]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"ROI {n}={rois[n]} falls outside the {width}x{height} frame")
    # +2 absorbs the shift from rounding each anchor down to an even coordinate.
    tile_w = _even_up(max(x1 - x0 for x0, _, x1, _ in rois.values()) + 2)
    tile_h = _even_up(max(y1 - y0 for _, y0, _, y1 in rois.values()) + 2)

    anchors = {}
    for n in names:
        x0, y0, x1, y1 = rois[n]
        ax = min(_even_down(x0), _even_down(width - tile_w))
        ay = min(_even_down(y0), _even_down(height - tile_h))
        if ax < 0 or ay < 0 or ax + tile_w > width or ay + tile_h > height:
            raise ValueError(f"ROI {n} tile {tile_w}x{tile_h} does not fit the frame")
        if not (ax <= x0 and x1 <= ax + tile_w and ay <= y0 and y1 <= ay + tile_h):
            raise ValueError(f"ROI {n} box {rois[n]} not contained in its tile at ({ax},{ay})")
        anchors[n] = (ax, ay)

    fc = ";".join(f"[0:v]crop={tile_w}:{tile_h}:{anchors[n][0]}:{anchors[n][1]},format=gray[c{i}]"
                  for i, n in enumerate(names))
    fc += ";" + "".join(f"[c{i}]" for i in range(len(names))) + f"hstack=inputs={len(names)}[o]"
    base = ["ffmpeg", "-v", "error", "-i", video, "-filter_complex", fc, "-map", "[o]",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]

    expected = tile_h * tile_w * len(names)
    probe = subprocess.run(base[:-1] + ["-frames:v", "1", "-"], stdout=subprocess.PIPE)
    if len(probe.stdout) != expected:
        raise RuntimeError(
            f"ffmpeg produced {len(probe.stdout)} bytes/frame but {expected} was expected "
            f"({len(names)} tiles of {tile_w}x{tile_h}); the crop geometry was altered by ffmpeg.")

    proc = subprocess.run(base, stdout=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video}")
    buf = np.frombuffer(proc.stdout, np.uint8)
    if len(buf) % expected:
        raise RuntimeError(f"ffmpeg returned a partial frame for {video} "
                           f"({len(buf)} bytes is not a multiple of {expected})")
    n_frames = len(buf) // expected
    stack = buf.reshape(n_frames, tile_h, tile_w * len(names))

    out = {}
    for i, n in enumerate(names):
        x0, y0, x1, y1 = rois[n]
        ax, ay = anchors[n]
        tile = stack[:, :, i * tile_w:(i + 1) * tile_w]
        out[n] = tile[:, y0 - ay:y0 - ay + (y1 - y0), x0 - ax:x0 - ax + (x1 - x0)]
    return out


def roi_signal(stack: np.ndarray, baseline_win: int = 151) -> np.ndarray:
    """Per-frame brightness of the ROI with slow lighting drift removed.

    A high percentile rather than the mean, because the LED is a small bright blob inside a much
    larger ROI; a rolling median baseline removes ambient drift and the mouse slowly occluding the
    port, while leaving the 2-3 frame flush intact.
    """
    from scipy.ndimage import median_filter

    # Boxes are tight around the LED (a few hundred px), so a high percentile is effectively the
    # brightest few pixels; the mean would be dragged down by the surrounding port wall.
    v = np.percentile(stack.reshape(len(stack), -1), 95.0, axis=1).astype(np.float64)
    return v - median_filter(v, size=baseline_win)


def find_events(sig: np.ndarray, threshold: float, min_len: int = 1, max_len: int = 8) -> list[dict]:
    """Group supra-threshold frames into runs and keep the flush-length ones.

    ``min_len`` defaults to 1 deliberately. The valve is open 70-120 ms and the camera runs at
    ~24.85 fps, so the shortest flush spans only ~1.7 frames -- whether it lands on one frame or two
    depends on where it falls relative to the frame clock. Requiring two frames would therefore
    discard real rewards. Raise it only for a session where single-frame noise is a demonstrated
    problem, and check the count against the Bpod expectation before and after.
    """
    above = sig > threshold
    events, i, n = [], 0, len(sig)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if min_len <= (j - i) <= max_len:
            peak = i + int(np.argmax(sig[i:j]))
            events.append({"frame_raw": peak, "n_frames": j - i, "amplitude": float(sig[i:j].max())})
        i = j
    return events


def enforce_alternation(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop detections that violate the linear track's strict port alternation.

    ``LinearTrack.m`` alternates the baited port every trial, and the saved ``TrialTypes`` confirm it
    in 486 of 486 sessions -- so two consecutive detections at the same port mean one of them is
    spurious. Where that happens the weaker (lower-amplitude) event is dropped.

    This is a protocol invariant rather than a tuned threshold, and it needs no Bpod log, so it also
    cleans up the 26 linear sessions that have none. It can only remove events, never invent one, so
    it cannot mask a genuine miss: a missing reward still shows as an alternation gap.
    """
    kept = list(events)
    dropped: list[dict] = []
    changed = True
    while changed:
        changed = False
        for i in range(len(kept) - 1):
            if kept[i]["port"] == kept[i + 1]["port"]:
                weaker = i if kept[i]["amplitude"] <= kept[i + 1]["amplitude"] else i + 1
                dropped.append(kept.pop(weaker))
                changed = True
                break
    return kept, dropped


def auto_threshold(sig: np.ndarray, k: float, floor: float) -> float:
    """Robust noise floor of the ROI signal; the flush is orders of magnitude above it."""
    mad = np.median(np.abs(sig - np.median(sig)))
    return max(floor, k * 1.4826 * mad)


def read_bpod_expectation(mat_path: str, task: str = "linear") -> dict:
    """Expected reward counts and ORDER from the Bpod log -- never individual event times.

    The order is the useful part for the T-maze: the detected sequence of lit arms can be checked
    against the sequence the protocol baited, with no clock alignment involved at all.
    """
    from scipy.io import loadmat

    sd = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["SessionData"]
    trials = np.atleast_1d(sd.RawEvents.Trial)
    types = np.atleast_1d(sd.TrialTypes).astype(int)
    table = TRIALTYPE_TO_PORT if task == "linear" else TMAZE_TRIALTYPE_TO_ARM

    per_port: dict[str, int] = {}
    order: list[str] = []
    n_errors = 0
    for i, t in enumerate(trials):
        reward = np.atleast_2d(np.array(t.States.Reward, dtype=float))
        if "ReturnMiddle" in t.States._fieldnames:
            rm = np.atleast_2d(np.array(getattr(t.States, "ReturnMiddle"), dtype=float))
            if not np.isnan(rm).all():
                n_errors += rm.shape[0]
        if np.isnan(reward).all():
            continue  # linear-track session-timeout trial: no water delivered
        ttype = int(types[i])
        if ttype not in table:
            raise ValueError(
                f"{os.path.basename(mat_path)} trial {i + 1} has TrialType {ttype}, not valid for "
                f"task '{task}' ({sorted(table)}). Check the log matches the video's task.")
        name = table[ttype]
        per_port[name] = per_port.get(name, 0) + 1
        order.append(name)
    return {"n_trials": int(np.atleast_1d(sd.nTrials).ravel()[0]),
            "n_rewards_expected": len(order), "per_port_expected": per_port,
            "expected_order": order, "n_error_entries": n_errors}


def read_bpod_events(mat_path: str, task: str) -> dict:
    """Reward times and error-episode intervals, in seconds since the Bpod session started."""
    from scipy.io import loadmat

    sd = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["SessionData"]
    tst = np.atleast_1d(sd.TrialStartTimestamp).astype(float)
    types = np.atleast_1d(sd.TrialTypes).astype(int)
    table = TRIALTYPE_TO_PORT if task == "linear" else TMAZE_TRIALTYPE_TO_ARM
    rewards, errors = [], []
    for i, t in enumerate(np.atleast_1d(sd.RawEvents.Trial)):
        if "ReturnMiddle" in t.States._fieldnames:
            rm = np.atleast_2d(np.array(getattr(t.States, "ReturnMiddle"), dtype=float))
            if not np.isnan(rm).all():
                for a, b in rm:
                    errors.append((tst[i] + a, tst[i] + b))
        r = np.array(t.States.Reward, dtype=float).ravel()
        if not np.isnan(r).all():
            rewards.append((tst[i] + r[0], table[int(types[i])]))
    return {"rewards": rewards, "errors": errors}


def led_on_edges(sig_raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Frames where a steadily-lit LED switches on, plus the threshold used and the on-fraction.

    The T-maze LED is a square wave held on for seconds at a time, not the brief flash the linear
    track produces, so the trace is bimodal (dim baseline vs saturated). Threshold midway between.
    """
    lo, hi = np.percentile(sig_raw, 25), np.percentile(sig_raw, 99.5)
    thr = 0.5 * (lo + hi)
    if hi - lo < MIN_LED_CONTRAST:
        raise RuntimeError(
            f"anchor ROI is not bimodal (p25={lo:.0f}, p99.5={hi:.0f}, contrast {hi - lo:.0f} < "
            f"{MIN_LED_CONTRAST}); there is no clear LED on/off state to take anchors from.")
    on = (sig_raw > thr).astype(np.int8)
    frac = float(on.mean())
    if frac > MAX_LED_ON_FRACTION:
        # The threshold is the midpoint of p25 and p99.5, which assumes the LED is off for a good
        # part of the session. If it is lit most of the time, that midpoint can land inside the
        # "on" cluster and the edges become meaningless.
        raise RuntimeError(
            f"anchor LED is on for {frac * 100:.0f}% of frames (> {MAX_LED_ON_FRACTION * 100:.0f}%); "
            "the percentile threshold cannot be trusted on this session.")
    return np.flatnonzero(np.diff(on) == 1) + 1, float(thr), frac


def fit_clock(anchor_video_t: np.ndarray, anchor_bpod_t: np.ndarray,
              tol: float = 0.4, iters: int = 6) -> dict:
    """Robustly fit video_time = a + b * bpod_time from anchor events.

    ``b`` absorbs the drift between the Bpod device oscillator and the acquisition PC clock, which
    is real and worth ~3-4 frames over a 16-minute session, so a constant offset is not enough.
    """
    best = None
    for a0 in np.arange(-120.0, 120.0, 0.05):
        pred = anchor_bpod_t + a0
        idx = np.searchsorted(anchor_video_t, pred).clip(1, len(anchor_video_t) - 1)
        near = np.minimum(np.abs(anchor_video_t[idx] - pred), np.abs(anchor_video_t[idx - 1] - pred))
        n = int((near < tol).sum())
        if best is None or n > best[0]:
            best = (n, a0)
    a, b = best[1], 1.0
    matched = None
    for _ in range(iters):
        pred = a + b * anchor_bpod_t
        idx = np.searchsorted(anchor_video_t, pred).clip(1, len(anchor_video_t) - 1)
        pick = np.where(np.abs(anchor_video_t[idx] - pred) < np.abs(anchor_video_t[idx - 1] - pred),
                        idx, idx - 1)
        dist = np.abs(anchor_video_t[pick] - pred)
        keep = dist < tol
        # One video edge may not serve two anchors. Without this, two closely-spaced error episodes
        # can both snap to a single merged LED edge, inflating n_matched and flattering the residual
        # exactly where a merged/missing edge should instead be flagged.
        order = np.argsort(np.where(keep, dist, np.inf))
        seen, unique = set(), np.zeros_like(keep)
        for i in order:
            if not keep[i]:
                break
            if pick[i] not in seen:
                seen.add(pick[i])
                unique[i] = True
        n_dropped = int(keep.sum() - unique.sum())
        keep = unique
        if keep.sum() < 4:
            break
        b, a = np.polyfit(anchor_bpod_t[keep], anchor_video_t[pick][keep], 1)
        matched = (keep, pick, n_dropped)
    if matched is None:
        raise RuntimeError("clock fit failed: fewer than 4 anchor events matched uniquely")
    keep, pick, n_dropped = matched
    resid = anchor_video_t[pick][keep] - (a + b * anchor_bpod_t[keep])
    return {"a": float(a), "b": float(b), "n_anchors": int(len(anchor_bpod_t)),
            "n_matched": int(keep.sum()), "n_dropped_duplicate_edge": n_dropped,
            "n_video_edges": int(len(anchor_video_t)),
            "n_edges_unused": int(len(anchor_video_t) - keep.sum()),
            "resid_sd_s": float(resid.std()), "resid_max_s": float(np.abs(resid).max()),
            "drift_ppm": float((b - 1.0) * 1e6)}


def bpod_session_start(mat_path: str):
    """Wall-clock start of a Bpod session, as naive local time.

    Taken from the filename, which equals ``Info.SessionStartTime_UTC`` -- itself local time despite
    the name. Bpod state timestamps are seconds from this instant.
    """
    import pandas as pd

    m = re.search(r"_(\d{8})_(\d{6})\.mat$", os.path.basename(mat_path))
    if not m:
        raise ValueError(f"no timestamp in Bpod filename: {mat_path}")
    return pd.Timestamp(f"{m.group(1)} {m.group(2)}", tz=None)


def _video_seconds(timestamps_csv: str, n_frames: int, reference=None) -> np.ndarray:
    """Per-frame wall-clock seconds relative to ``reference`` (default: the first frame).

    Passing the Bpod session start puts video time and Bpod time on the same origin, so the fitted
    offset is just the small difference between the two machines' clocks rather than the several
    minutes between the recording starting and the protocol being launched.
    """
    import pandas as pd

    raw = pd.read_csv(timestamps_csv, header=None, names=["t"], dtype=str)["t"]
    t = pd.to_datetime(raw.str[:26], format="mixed")
    origin = t.iloc[0] if reference is None else reference
    v = (t - origin).dt.total_seconds().to_numpy()
    if len(v) < n_frames:  # tolerated trailing frames with no timestamp row
        v = np.concatenate([v, v[-1] + np.arange(1, n_frames - len(v) + 1) / 25.0])
    return v[:n_frames]


def _compact(seq: list[str], mapping: dict[str, str]) -> str:
    """One character per event, so long sequences stay readable side by side."""
    return "".join(mapping.get(x, "?") for x in seq)


def _label_chars(labels) -> dict[str, str]:
    """Distinct single characters for a set of port/arm labels, shared by both sequences.

    First letters are preferred but collide for the linear track ("Port4"/"Port2"), so fall back to
    last characters and then to arbitrary letters rather than rendering both as "P".
    """
    labels = sorted(set(labels))
    for pick in (lambda l: l[0].upper(), lambda l: l[-1].upper()):
        chars = [pick(l) for l in labels]
        if len(set(chars)) == len(labels):
            return dict(zip(labels, chars))
    return {l: chr(65 + i) for i, l in enumerate(labels)}


def _align(detected: list[str], expected: list[str]) -> list[tuple[str, str]]:
    """Needleman-Wunsch alignment of two label sequences (unit costs).

    Used instead of a positional comparison so one extra or missing detection shows up as a single
    edit rather than shifting -- and thus invalidating -- every comparison after it.
    """
    n, m = len(detected), len(expected)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (detected[i - 1] != expected[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (detected[i - 1] != expected[j - 1]):
            ops.append(("match" if detected[i - 1] == expected[j - 1] else "sub", detected[i - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("extra", detected[i - 1])); i -= 1
        else:
            ops.append(("missing", expected[j - 1])); j -= 1
    return ops[::-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="raw linear-track behaviour video (Linear<TIMESTAMP>.avi)")
    ap.add_argument("-o", "--output", required=True, help="output CSV of detected reward events")
    ap.add_argument("--mode", choices=["detect", "align"], default="detect",
                    help="detect: read reward times off the video (linear track). "
                         "align: fit the clock from LED anchor events and take reward times from "
                         "the Bpod log (T-maze, where the reward LED is occluded by the mouse)")
    ap.add_argument("--anchor-roi", default="centre",
                    help="align mode: which ROI carries the anchor LED (default centre)")
    ap.add_argument("--task", choices=["linear", "tmaze"], default="linear",
                    help="which arena's ROI set and era rules to use (default linear)")
    ap.add_argument("--epoch", help="camera epoch/era override (default: from the filename date)")
    ap.add_argument("--roi", action="append", metavar="PORT:X0:X1:Y0:Y1",
                    help="override/define a port ROI box, e.g. Port4:180:210:650:670 (repeatable)")
    ap.add_argument("--survey-pre-ports", action="store_true",
                    help="detect on all four pre-May-2026 port boxes at once, to identify which "
                         "linear track a mouse ran (counts only; no Bpod port mapping)")
    ap.add_argument("--threshold", type=float, help="fixed detection threshold (default: automatic)")
    ap.add_argument("--threshold-k", type=float, default=8.0, help="automatic threshold in robust SDs")
    ap.add_argument("--threshold-floor", type=float, default=DEFAULT_THRESHOLD_FLOOR,
                    help=f"minimum automatic threshold (default {DEFAULT_THRESHOLD_FLOOR})")
    ap.add_argument("--min-frames", type=int, default=1, help="minimum event length in frames (see find_events)")
    ap.add_argument("--bpod", help="Bpod .mat log, for the expected-count audit only")
    ap.add_argument("--trim-info", help="dlc_trim_info.csv, to also report frame_local")
    ap.add_argument("--trim-video", help="the trimmed clip, so the trim window length is read rather than assumed")
    ap.add_argument("--timestamps", help="timestamps<TS>.csv, to assert one decoded frame per row")
    ap.add_argument("--qc-plot", help="write a QC figure showing both ROI traces and detections")
    ap.add_argument("--summary-json", help="write the run summary to this path as JSON")
    ap.add_argument("--no-alternation-filter", action="store_true",
                    help="keep detections that break the linear track's strict port alternation")
    args = ap.parse_args()

    if args.task == "tmaze":
        epoch = args.epoch or tmaze_era_for_session(args.video)
        table = TMAZE_ARM_ROIS
    else:
        epoch = args.epoch or epoch_for_session(args.video)
        table = PORT_ROIS
    if epoch not in table:
        print(f"ERROR: no ROI set '{epoch}' for task '{args.task}' "
              f"(have {sorted(table)})", file=sys.stderr)
        return 2
    if args.survey_pre_ports and epoch != "pre2026-05":
        print(f"ERROR: --survey-pre-ports uses the pre-May-2026 port boxes, but this video resolves "
              f"to epoch '{epoch}'. The camera was re-aimed, so those boxes would sample the wrong "
              "place -- and the resolution check cannot catch it, since only the aim changed.",
              file=sys.stderr)
        return 2
    rois = dict(PRE2026_05_ALL_PORTS) if args.survey_pre_ports else dict(table[epoch])
    for spec in args.roi or []:
        name, x0, x1, y0, y1 = spec.split(":")
        rois[name] = (int(x0), int(y0), int(x1), int(y1))
    if not rois:
        print(f"ERROR: no ROIs defined for epoch '{epoch}'. Pass --roi PORT:X:Y or fill in PORT_ROIS.",
              file=sys.stderr)
        return 2

    width, height, declared_frames = probe_video(args.video)
    print(f"video  : {args.video}")
    print(f"geometry: {width}x{height}, {declared_frames} frames declared")
    if (width, height) != (REF_WIDTH, REF_HEIGHT):
        print(f"ERROR: PORT_ROIS coordinates were measured on {REF_WIDTH}x{REF_HEIGHT}. This video is "
              f"{width}x{height}, so the ROIs would land in the wrong place.", file=sys.stderr)
        return 2
    print(f"task   : {args.task}")
    print(f"epoch  : {epoch}   ROIs: " + ", ".join(f"{k}@{v}" for k, v in sorted(rois.items())))

    stacks = extract_roi_stack(args.video, rois, width, height)
    n_frames = len(next(iter(stacks.values())))
    print(f"frames : {n_frames} decoded")

    # The whole premise is that frame_raw indexes timestamps.csv and the DLC csv identically, so a
    # silently dropped frame would shift every later index. Check it rather than trust it.
    if n_frames != declared_frames:
        print(f"ERROR: decoded {n_frames} frames but the container declares {declared_frames}; "
              "frame indices would not line up with timestamps.csv / DLC.", file=sys.stderr)
        return 2
    if args.timestamps:
        with open(args.timestamps) as fh:
            n_rows = sum(1 for line in fh if line.strip())
        # A trailing surplus of video frames is benign and does occur: acquisition can stop after
        # writing a frame but before flushing its timestamp row, so the last frame(s) have no row.
        # Frame i still maps to row i for every row that exists. A deficit, or a large surplus,
        # means the two are genuinely out of step and every index would be wrong.
        surplus = n_frames - n_rows
        if surplus < 0 or surplus > TIMESTAMP_SURPLUS_TOLERANCE:
            print(f"ERROR: {os.path.basename(args.timestamps)} has {n_rows} rows but the video has "
                  f"{n_frames} frames (difference {surplus}).", file=sys.stderr)
            return 2
        if surplus:
            print(f"       timestamps.csv has {n_rows} rows for {n_frames} frames; the last "
                  f"{surplus} frame(s) have no timestamp (tolerated)")
        else:
            print(f"       timestamps.csv rows match ({n_rows})")

    start_frame = None
    if args.trim_info:
        with open(args.trim_info) as fh:
            start_frame = int(next(csv.DictReader(fh))["start_frame"])
    # Length of the trimmed clip. Read it from the clip itself when available rather than assuming
    # 960 s x 25 fps -- true capture rate is ~24.8-24.9 fps, so 24000 is only nominal.
    trim_len = 24000
    trim_len_source = "assumed 960 s x 25 fps (nominal)"
    if args.trim_video:
        trim_len = probe_video(args.trim_video)[2]
        trim_len_source = f"measured from {os.path.basename(args.trim_video)}"
    if start_frame is not None:
        print(f"trim   : start_frame {start_frame}, window {trim_len} frames ({trim_len_source})")

    if args.mode == "align":
        if not (args.bpod and args.timestamps):
            print("ERROR: align mode needs --bpod and --timestamps.", file=sys.stderr)
            return 2
        if args.anchor_roi not in stacks:
            print(f"ERROR: --anchor-roi '{args.anchor_roi}' is not one of {sorted(stacks)}",
                  file=sys.stderr)
            return 2
        vt = _video_seconds(args.timestamps, n_frames, bpod_session_start(args.bpod))
        raw = np.percentile(stacks[args.anchor_roi].reshape(n_frames, -1), 95.0, axis=1)
        edges, thr, on_frac = led_on_edges(raw)
        bp = read_bpod_events(args.bpod, args.task)
        print(f"align  : anchor ROI '{args.anchor_roi}' threshold {thr:.0f}, "
              f"LED on {on_frac * 100:.0f}% of frames, {len(edges)} rising edges")
        print(f"         Bpod error episodes (anchors): {len(bp['errors'])}")
        if len(bp["errors"]) < MIN_ANCHORS:
            print(f"ERROR: only {len(bp['errors'])} anchor episodes; need >= {MIN_ANCHORS} to fit a "
                  "clock. This session cannot be aligned.", file=sys.stderr)
            return 2
        fit = fit_clock(vt[edges], np.array([a for a, _ in bp["errors"]]))
        fps_eff = n_frames / (vt[-1] - vt[0])
        print(f"         (video spans {vt[0]:+.0f}s to {vt[-1]:+.0f}s relative to Bpod session start)")
        print(f"         fit: offset {fit['a']:+.3f}s  drift {fit['drift_ppm']:+.0f} ppm  "
              f"matched {fit['n_matched']}/{fit['n_anchors']} anchors")
        print(f"         residual SD {fit['resid_sd_s'] * 1000:.0f} ms "
              f"({fit['resid_sd_s'] * fps_eff:.2f} frames), max {fit['resid_max_s'] * 1000:.0f} ms "
              f"({fit['resid_max_s'] * fps_eff:.1f} frames)")
        if fit["n_matched"] < MIN_ANCHORS:
            print(f"ERROR: only {fit['n_matched']} anchors matched the video; fit not trustworthy.",
                  file=sys.stderr)
            return 2
        events = []
        for t, name in bp["rewards"]:
            f = int(np.searchsorted(vt, fit["a"] + fit["b"] * t))
            e = {"frame_raw": min(f, n_frames - 1), "port": name, "amplitude": float("nan"),
                 "n_frames": 0, "threshold": round(thr, 2), "source": "bpod_aligned"}
            if start_frame is not None:
                e["frame_local"] = e["frame_raw"] - start_frame
                e["in_trim_window"] = 0 <= e["frame_local"] < trim_len
            events.append(e)
        events.sort(key=lambda e: e["frame_raw"])
        signals, thresholds = {args.anchor_roi: raw}, {args.anchor_roi: thr}
        for port in sorted(rois):
            print(f"  {port}: {sum(1 for e in events if e['port'] == port)} rewards placed")
        summary_extra = {"clock_fit": fit, "anchor_roi": args.anchor_roi}
    else:
        summary_extra = {}

    events, signals, thresholds = (events, signals, thresholds) if args.mode == "align" else ([], {}, {})
    for port, stack in (sorted(stacks.items()) if args.mode == "detect" else []):
        sig = roi_signal(stack)
        signals[port] = sig
        thr = args.threshold if args.threshold is not None else auto_threshold(
            sig, args.threshold_k, args.threshold_floor)
        thresholds[port] = thr
        found = find_events(sig, thr, min_len=args.min_frames)
        print(f"  {port}: threshold {thr:6.1f}  -> {len(found)} events")
        for e in found:
            e["port"] = port
            e["threshold"] = round(thr, 2)
            if start_frame is not None:
                e["frame_local"] = e["frame_raw"] - start_frame
                e["in_trim_window"] = 0 <= e["frame_local"] < trim_len
            events.append(e)
    events.sort(key=lambda e: e["frame_raw"])

    n_alt_dropped = 0
    if args.mode == "detect" and args.task == "linear" and not args.no_alternation_filter:
        events, alt_dropped = enforce_alternation(events)
        n_alt_dropped = len(alt_dropped)
        if n_alt_dropped:
            amps = sorted(round(e["amplitude"], 1) for e in alt_dropped)
            print(f"  alternation filter: dropped {n_alt_dropped} event(s) breaking strict port "
                  f"alternation (amplitudes {amps[:8]}{' ...' if len(amps) > 8 else ''})")

    cols = ["frame_raw", "port", "amplitude", "n_frames", "threshold"]
    if args.mode == "align":
        cols.append("source")
    if start_frame is not None:
        cols += ["frame_local", "in_trim_window"]
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in events:
            w.writerow({c: e[c] for c in cols})
    print(f"wrote  : {args.output}  ({len(events)} events)")

    summary = {"video": args.video, "epoch": epoch, "rois": {k: list(v) for k, v in rois.items()},
               "n_frames": n_frames, "mode": args.mode, "task": args.task,
               "n_alternation_dropped": n_alt_dropped,
               "thresholds": {k: round(v, 2) for k, v in thresholds.items()},
               **summary_extra,
               "n_detected": len(events),
               "detected_per_port": {p: sum(1 for e in events if e["port"] == p) for p in rois},
               "amplitude_separation": {
                   p: {"threshold": round(thresholds[p], 2),
                       "min_accepted": round(min((e["amplitude"] for e in events if e["port"] == p),
                                                 default=float("nan")), 2),
                       "median_accepted": round(float(np.median(
                           [e["amplitude"] for e in events if e["port"] == p])), 2)
                       if any(e["port"] == p for e in events) else None}
                   for p in thresholds}}
    if start_frame is not None:
        inwin = [e for e in events if e["in_trim_window"]]
        summary["n_detected_in_trim_window"] = len(inwin)
        summary["detected_per_port_in_trim_window"] = {
            p: sum(1 for e in inwin if e["port"] == p) for p in rois}

    if args.bpod:
        exp = read_bpod_expectation(args.bpod, args.task)
        summary["bpod"] = exp
        print(f"\nBpod audit ({os.path.basename(args.bpod)}):")
        print(f"  expected {exp['n_rewards_expected']} rewards from {exp['n_trials']} trials"
              f"  {exp['per_port_expected']}")
        print(f"  detected {summary['n_detected']} " + str(summary["detected_per_port"]))
        for port, k in sorted(exp["per_port_expected"].items()):
            if port not in summary["detected_per_port"]:
                print(f"    {port}: no ROI named '{port}' in this run")
                continue
            got = summary["detected_per_port"][port]
            print(f"    {port}: {got}/{k} = {100.0 * got / k:5.1f}% of expected")
        unmapped = sorted(set(summary["detected_per_port"]) - set(exp["per_port_expected"]))
        if unmapped:
            print("  ROIs with no Bpod port mapping (survey): "
                  + ", ".join(f"{u}={summary['detected_per_port'][u]}" for u in unmapped))

        # ORDER CHECK. In DETECT mode this is the strongest validation available without any clock
        # alignment: the sequence of lit ports observed in the video against the sequence the
        # protocol baited. Reported as an edit distance so one spurious or missing event shows up as
        # a single edit rather than desynchronising everything after it.
        #
        # In ALIGN mode it is CIRCULAR and validates nothing about the alignment: the events were
        # generated from this same .mat file, so both sequences have the same source and can only
        # ever disagree if the two Bpod readers disagree. Say so rather than printing a perfect
        # score that looks like confirmation.
        if args.mode == "align":
            print("\n  ORDER CHECK skipped: in align mode the reward list comes from the Bpod log "
                  "itself, so comparing it back to that log would be circular.")
            print("  Alignment quality is the clock fit above: "
                  f"{summary_extra['clock_fit']['n_matched']}/"
                  f"{summary_extra['clock_fit']['n_anchors']} anchors, residual SD "
                  f"{summary_extra['clock_fit']['resid_sd_s'] * 1000:.0f} ms.")
            summary["order_check"] = "skipped (circular in align mode)"
        else:
            detected_order = [e["port"] for e in events]
            expected_order = exp["expected_order"]
            ops = _align(detected_order, expected_order)
            summary["order_check"] = {
            "n_detected": len(detected_order), "n_expected": len(expected_order),
            "matches": sum(1 for o in ops if o[0] == "match"),
            "extra_detected": sum(1 for o in ops if o[0] == "extra"),
            "missing": sum(1 for o in ops if o[0] == "missing"),
            "substituted": sum(1 for o in ops if o[0] == "sub")}
            oc = summary["order_check"]
            print(f"\n  ORDER CHECK  detected {oc['n_detected']} vs expected {oc['n_expected']}: "
              f"{oc['matches']} in order, {oc['extra_detected']} extra, {oc['missing']} missing, "
              f"{oc['substituted']} wrong port")
            chars = _label_chars(list(expected_order) + list(detected_order))
            print("    key: " + ", ".join(f"{v}={k}" for k, v in sorted(chars.items())))
            print("    expected: " + _compact(expected_order, chars))
            print("    detected: " + _compact(detected_order, chars))

    print("\n" + json.dumps(summary, indent=2))
    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            json.dump(summary, fh, indent=2)

    if args.qc_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ports = sorted(signals)
        fig, axes = plt.subplots(len(ports), 1, figsize=(14, 3 * len(ports)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, port in zip(axes, ports):
            sig = signals[port]
            ax.plot(sig, lw=0.4, color="0.3")
            ev = [e for e in events if e["port"] == port] if args.mode == "detect" else events
            # In align mode events carry no amplitude (they came from the log, not the trace), so
            # mark them against the anchor trace instead of at y=NaN, which renders nothing at all.
            ys = ([e["amplitude"] for e in ev] if args.mode == "detect"
                  else [sig[min(e["frame_raw"], len(sig) - 1)] for e in ev])
            ax.plot([e["frame_raw"] for e in ev], ys, "v", color="tab:red", ms=5,
                    label=f"{len(ev)} " + ("detected" if args.mode == "detect" else "rewards (aligned)"))
            # Plot the computed threshold, not one read off a detected event -- a port with zero
            # detections is exactly when seeing where the threshold sat matters most.
            ax.axhline(thresholds[port], color="tab:blue", ls="--", lw=0.8,
                       label=f"threshold {thresholds[port]:.1f}")
            if start_frame is not None:
                ax.axvspan(start_frame, start_frame + trim_len, color="tab:green", alpha=0.08,
                           label="trim window")
            ax.set_ylabel(f"{port}\nROI brightness")
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("raw video frame")
        axes[0].set_title(os.path.basename(args.video))
        fig.tight_layout()
        fig.savefig(args.qc_plot, dpi=110)
        print(f"wrote  : {args.qc_plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
