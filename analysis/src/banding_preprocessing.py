"""banding_preprocessing.py — Horizontal banding (row-stripe) detection and removal.

V4 Miniscope frames occasionally contain transient horizontal bands: a run of
consecutive sensor rows that is uniformly brighter (or darker) than its
neighbours for one frame.  The artefact is additive and constant along x, so it
shows up cleanly in the per-row mean profile and is invisible to a spatial
band-pass filter tuned to soma size.

Credits: Aleksandar Marinkovic.
"""

import os
import sys

# ── path bootstrap ────────────────────────────────────────────────────────────
# Adds analysis/src/ so utils.analysis_utils is importable from here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itertools import product

from utils.analysis_utils import *
from scipy.ndimage import median_filter, binary_dilation

# ── defaults ─────────────────────────────────────────────────────────────────
NOMINAL_FPS = 25.0    # Hz  — expected GCaMP6s miniscope rate
MAX_PIXEL   = 255     # 8-bit AVI

# Defaults tuned on Mouse944 Round1 Linear; check them per recording with
# plot_score_distribution.  A stripe event there spans ~50 rows, so a narrow
# baseline window sits inside the artefact and under-corrects it; the flagged
# frame count plateaus above z ~ 10, which is where noise stops and bands start.
DEFAULT_WIN     = 31    # rows; median-filter width of the row-profile baseline
DEFAULT_Z       = 10.0  # robust z threshold for calling a row banded
DEFAULT_DILATE  = 3     # rows; dilation of the flagged mask along y
# V4 banding arrives as a stripe pattern with bright *and* dark lines in the same
# frame, so both signs are flagged by default.
DEFAULT_POLARITY = "both"

# Outputs go to <session_dir>/denoised/<MMDDYYYY-HHMMSS>/, one folder per run, so
# re-running with different parameters never overwrites an earlier result.
OUT_ROOT      = "denoised"
RUN_STAMP_FMT = "%m%d%Y-%H%M%S"

# Encodings offered by destripe_video, as name -> FourCC.  FFV1 is lossless and the
# default: banding is only a few grey levels deep in most frames, so a lossy
# re-encode is the same order of magnitude as the artefact being removed.  MPEG4
# (FMP4, the codec the Miniscope itself records in) is several times smaller and
# faster to write — use it for previews, not for anything that feeds CNMF-E.
CODECS        = {"FFV1": "FFV1", "MPEG4": "FMP4"}
DEFAULT_CODEC = "FFV1"

# Normal-consistency constants: for x ~ N(0, sigma),
#   median(|x|) = 0.6745 sigma  ->  sigma = 1.4826 * MAD
#   P90(|x|)    = 1.6449 sigma
_MAD_TO_SIGMA = 1.4826
_P90_TO_SIGMA = 1.0 / 1.6449


# ─────────────────────────────────────────────────────────────────────────────
# Row profiles
# ─────────────────────────────────────────────────────────────────────────────

def compute_row_profiles(path, subsample=1, max_frames=None, progress=True):
    """Stream a video and return the mean intensity of every row of every frame.

    Only the (H,) row profile of each frame is retained, so a full-length
    recording costs ~T x H floats (a few MB) instead of the tens of GB the movie
    itself would need.  Decoding is skipped for frames that are sub-sampled out.
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {path}")

    fps            = float(cap.get(cv2.CAP_PROP_FPS) or NOMINAL_FPS)
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if abs(fps - NOMINAL_FPS) / NOMINAL_FPS > 0.05:
        warnings.warn(
            f"Container FPS ({fps:.2f} Hz) differs from nominal "
            f"({NOMINAL_FPS} Hz) by >5 %.  Check acquisition hardware log."
        )

    # cv2.grab() advances without decoding (fast); cv2.read() decodes (slow).
    n_expected = len(range(0, n_frames_total, subsample))
    if max_frames is not None:
        n_expected = min(n_expected, max_frames)
    bar = tqdm(total=n_expected, desc=f"Row profiles {os.path.basename(path)}",
               disable=not progress)

    profiles, indices, idx = [], [], 0
    while True:
        if idx % subsample == 0:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            profiles.append(gray.mean(axis=1, dtype=np.float64).astype(np.float32))
            indices.append(idx)
            bar.update(1)
            if max_frames is not None and len(profiles) >= max_frames:
                break
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()
    bar.close()

    if not profiles:
        raise IOError(f"No frames decoded from: {path}")

    rows = np.stack(profiles, axis=0)
    meta = dict(
        path=path,
        n_frames_total=n_frames_total,
        n_frames_used=int(rows.shape[0]),
        frame_indices=np.array(indices, dtype=np.int64),
        height=height,
        width=width,
        fps=fps,
        subsample=subsample,
        max_pixel=MAX_PIXEL,
    )
    return rows, meta


def row_profiles(movie):
    """Return the (T, H) row-mean profiles of an in-memory ``(T, H, W)`` movie."""
    movie = np.asarray(movie)
    if movie.ndim != 3:
        raise ValueError(f"Expected a (T, H, W) movie, got shape {movie.shape}")
    return movie.mean(axis=2, dtype=np.float64).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

def _robust_sigma(resid):
    """Estimate the noise scale of the row residuals, tolerating quantisation.

    The MAD is the natural choice, but miniscope AVIs are lossy-compressed and
    frequently reproduce a row exactly from its neighbour, so more than half of
    the residuals can be identically zero and the MAD collapses to 0.  In that
    case the 90th percentile of |resid| is used instead: bands occupy well under
    1 % of all rows, so P90 still reflects noise rather than the artefact.
    """
    mad = float(np.median(np.abs(resid - np.median(resid))))
    if mad > 0:
        return _MAD_TO_SIGMA * mad
    p90 = float(np.percentile(np.abs(resid), 90))
    if p90 > 0:
        return _P90_TO_SIGMA * p90
    raise ValueError("Row residuals are identically zero; the video has no "
                     "row-to-row variation to threshold against.")


def _flag_rows(stat, z, dilate):
    """Threshold the detection statistic and widen each hit along y."""
    bad = stat > z
    if dilate > 1:
        bad = binary_dilation(bad, np.ones((1, dilate), dtype=bool))
    return bad


def detect_bands(rows, win=DEFAULT_WIN, z=DEFAULT_Z, dilate=DEFAULT_DILATE,
                 polarity=DEFAULT_POLARITY):
    """Flag rows whose mean intensity departs from the local row-profile baseline.

    The baseline is a median filter applied along y only, so it follows the
    vignette and any slow illumination gradient but not a band a few rows wide.
    Residuals are scaled by a single robust sigma pooled over the whole recording
    (see :func:`_robust_sigma`), which keeps the z-score comparable across frames.
    """
    rows = np.asarray(rows)
    if rows.ndim == 3:
        rows = row_profiles(rows)
    if rows.ndim != 2:
        raise ValueError(f"Expected (T, H) row profiles or a (T, H, W) movie, "
                         f"got shape {rows.shape}")
    if polarity not in ("bright", "dark", "both"):
        raise ValueError(f"polarity must be 'bright', 'dark' or 'both', got {polarity!r}")
    if win < 3 or win % 2 == 0:
        raise ValueError(f"win must be an odd integer >= 3, got {win}")

    rows = rows.astype(np.float32, copy=False)

    # Smooth along y only: size=(1, win) leaves each frame independent.
    baseline = median_filter(rows, size=(1, win), mode="nearest")
    resid    = rows - baseline

    sigma = _robust_sigma(resid)
    score = resid / sigma

    stat = {"bright": score, "dark": -score, "both": np.abs(score)}[polarity]
    bad = _flag_rows(stat, z, dilate)

    return dict(
        rows=rows,
        baseline=baseline.astype(np.float32),
        resid=resid.astype(np.float32),
        score=score.astype(np.float32),
        stat=stat.astype(np.float32),
        bad=bad,
        trace=stat.max(axis=1).astype(np.float32),
        sigma=sigma,
        win=win,
        z=z,
        dilate=dilate,
        polarity=polarity,
    )


def band_summary(res):
    """Summarise a :func:`detect_bands` result.
    """
    bad, stat, resid = res["bad"], res["stat"], res["resid"]
    n_frames   = int(bad.shape[0])
    bad_frames = np.flatnonzero(bad.any(axis=1))
    n_bad      = int(bad_frames.size)
    rows_per   = float(bad[bad_frames].sum(axis=1).mean()) if n_bad else 0.0
    amp        = float(np.abs(resid[bad]).max()) if bad.any() else 0.0
    return dict(
        n_frames=n_frames,
        bad_frames=bad_frames,
        n_bad_frames=n_bad,
        frac_bad_frames=n_bad / n_frames if n_frames else 0.0,
        frac_bad_rows=float(bad.mean()),
        rows_per_bad_frame=rows_per,
        max_stat=float(stat.max()),
        max_amplitude=amp,
    )


def print_band_summary(res, meta=None):
    """Pretty-print the banding statistics for one recording.
    """
    s = band_summary(res)
    bar = "=" * 58
    print(bar)
    print("  Horizontal banding summary")
    print(bar)
    if meta is not None:
        print(f"  File             : {os.path.basename(meta['path'])}")
        print(f"  Frames analysed  : {s['n_frames']} / {meta['n_frames_total']}  "
              f"(subsample = {meta['subsample']}x)")
    else:
        print(f"  Frames analysed  : {s['n_frames']}")
    print(f"  Detection        : polarity={res['polarity']}  win={res['win']}  "
          f"z>{res['z']}  dilate={res['dilate']}")
    print(f"  Residual sigma   : {res['sigma']:.3f} pixel units")
    print(f"  Banded frames    : {s['n_bad_frames']}  "
          f"({100 * s['frac_bad_frames']:.2f} % of frames)")
    print(f"  Banded rows      : {100 * s['frac_bad_rows']:.3f} % of all rows  "
          f"({s['rows_per_bad_frame']:.1f} rows per banded frame)")
    print(f"  Peak statistic   : z = {s['max_stat']:.1f}  "
          f"(max amplitude {s['max_amplitude']:.2f} pixel units)")
    print(bar)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweeps
# ─────────────────────────────────────────────────────────────────────────────

def _as_list(v):
    """Wrap a scalar (including a string) in a list; leave a sequence alone."""
    if isinstance(v, (str, bytes)) or np.isscalar(v):
        return [v]
    return list(v)


def param_grid(win=DEFAULT_WIN, z=DEFAULT_Z, dilate=DEFAULT_DILATE,
               polarity=DEFAULT_POLARITY):
    """Every combination of the detection parameters, as a list of dicts.

    Each argument takes a scalar or a list, so ``param_grid(win=[31, 101], z=[6, 8])``
    gives the four combinations while all-scalar arguments give a single one — the
    ordinary single-parameter run is just a sweep of length 1.  The dicts are
    keyword arguments for :func:`detect_bands`.
    """
    axes = dict(win=_as_list(win), z=_as_list(z),
                dilate=_as_list(dilate), polarity=_as_list(polarity))
    return [dict(zip(axes, values)) for values in product(*axes.values())]


def param_tag(params):
    """Short filesystem-safe name for one parameter combination.

    Used as the folder name for that combination's outputs, so a video is always
    identifiable from its path alone: ``win101_z8_dil3_both``.
    """
    return (f"win{params['win']}_z{params['z']:g}"
            f"_dil{params['dilate']}_{params['polarity']}")


def sweep_bands(rows, params=None, progress=True, **grid):
    """Run :func:`detect_bands` under every parameter combination; return a table.

    Row profiles are the expensive part — one full decode of the video — and they
    do not depend on win/z/dilate/polarity, so a sweep computes them once (with
    :func:`compute_row_profiles`) and only repeats the median filter, a few
    seconds per combination.  Only summary numbers are kept: one full
    detect_bands result is ~0.2 GB, so a dozen of them would not fit in memory.
    Recompute the one combination you settle on rather than caching them all.
    """
    params = param_grid(**grid) if params is None else params
    table = []
    for p in tqdm(params, desc="Parameter sweep", disable=not progress):
        res = detect_bands(rows, **p)
        s = band_summary(res)
        table.append(dict(
            tag=param_tag(p),
            **p,
            sigma=res["sigma"],
            banded_frames=s["n_bad_frames"],
            pct_frames=100.0 * s["frac_bad_frames"],
            pct_rows=100.0 * s["frac_bad_rows"],
            rows_per_bad_frame=s["rows_per_bad_frame"],
            max_amplitude=s["max_amplitude"],
        ))
    return pd.DataFrame(table)


# ─────────────────────────────────────────────────────────────────────────────
# Correction
# ─────────────────────────────────────────────────────────────────────────────

def correct_bands(frames, resid, bad=None, clip=(0, MAX_PIXEL)):
    """Subtract the per-row band offset from one or more frames.
    """
    frames = np.asarray(frames, dtype=np.float32)
    offset = np.asarray(resid, dtype=np.float32)
    if bad is not None:
        offset = np.where(np.asarray(bad, dtype=bool), offset, 0.0)
    if frames.ndim - 1 != offset.ndim:
        raise ValueError(f"resid shape {offset.shape} does not match frames "
                         f"shape {frames.shape}")
    if frames.shape[-2] != offset.shape[-1]:
        raise ValueError(f"resid covers {offset.shape[-1]} rows but frames have "
                         f"{frames.shape[-2]}")

    out = frames - offset[..., None]
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


def read_frames(path, indices):
    """Read specific frames from a video by index.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open: {path}")
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Could not read frame {i} from {path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        frames.append(gray.astype(np.float32))
    cap.release()
    return np.stack(frames, axis=0)


def make_run_dir(path, stamp=None, tag=None, create=True):
    """Return ``<session_dir>/denoised/<MMDDYYYY-HHMMSS>/[<tag>/]`` for one output run.

    Pass one ``stamp`` for a whole sweep and a ``tag`` per parameter combination
    (from :func:`param_tag`), so every combination of one run sits side by side
    under a single timestamp and nothing is overwritten.
    """
    stamp = stamp or datetime.now().strftime(RUN_STAMP_FMT)
    parts = [os.path.dirname(os.path.abspath(str(path))), OUT_ROOT, stamp]
    if tag:
        parts.append(tag)
    run_dir = os.path.join(*parts)
    if create:
        os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _resolve_codec(codec):
    """Map a codec name from :data:`CODECS` to its FourCC.

    A raw 4-character FourCC is passed through, so any encoder the local OpenCV
    build supports is still reachable.
    """
    key = str(codec).upper()
    if key in CODECS:
        return CODECS[key]
    if len(key) == 4:
        return key
    raise ValueError(f"Unknown codec {codec!r}; use one of {sorted(CODECS)} or a "
                     f"4-character FourCC.")


def destripe_video(path, res, out_path=None, out_dir=None, codec=DEFAULT_CODEC,
                   restrict_to_flagged=True, progress=True):
    """Write a banding-corrected copy of a video.

    Frames are streamed one at a time, corrected with :func:`correct_bands`, and
    written as 8-bit greyscale.  ``res`` must have been computed on the same
    video at ``subsample=1`` and cover every frame.

    ``codec`` selects the encoding: ``"FFV1"`` (lossless, the default) or
    ``"MPEG4"`` (lossy, smaller and faster — previews only).  See :data:`CODECS`.
    Both are written into the AVI container the sources already use.
    """
    path = str(path)
    fourcc = _resolve_codec(codec)      # validated before anything is opened
    resid, bad = res["resid"], res["bad"]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {path}")
    fps            = float(cap.get(cv2.CAP_PROP_FPS) or NOMINAL_FPS)
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resid.shape[0] != n_frames_total:
        cap.release()
        raise ValueError(
            f"detect_bands result covers {resid.shape[0]} frames but the video has "
            f"{n_frames_total}. Re-run compute_row_profiles with subsample=1."
        )
    if resid.shape[1] != height:
        cap.release()
        raise ValueError(f"detect_bands result covers {resid.shape[1]} rows but the "
                         f"video is {height} px tall.")

    if out_path is None:
        out_dir = out_dir or make_run_dir(path)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "debanded_" + os.path.basename(path))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps,
                             (width, height), isColor=False)
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open a VideoWriter for {out_path} with codec "
                      f"{codec!r} (FourCC {fourcc!r})")

    try:
        for t in tqdm(range(n_frames_total), desc=f"Debanding {os.path.basename(path)}",
                      disable=not progress):
            ret, frame = cap.read()
            if not ret:
                warnings.warn(f"Video ended at frame {t} of {n_frames_total}.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            fixed = correct_bands(gray, resid[t],
                                  bad[t] if restrict_to_flagged else None)
            writer.write(np.rint(fixed).astype(np.uint8))
    finally:
        writer.release()
        cap.release()

    print(f"Wrote {out_path}")
    return out_path


def save_band_mask(res, path=None, out_path=None, out_dir=None):
    """Save the flagged-row mask (and residuals) for downstream steps.
    """
    if out_path is None:
        if out_dir is None:
            if path is None:
                raise ValueError("Provide `out_dir`, `out_path`, or `path` "
                                 "(the source video).")
            out_dir = make_run_dir(path)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "band_mask.npz")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    np.savez_compressed(out_path, bad=res["bad"], resid=res["resid"],
                        score=res["score"])
    print(f"Wrote {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_band_map(res, meta=None, vmax=None, label="", plot=True):
    """Row-vs-frame map of the detection statistic, plus the per-frame peak.

    **Top** — the statistic for every (frame, row); a band appears as a short
    bright vertical dash.  **Bottom** — the per-frame maximum with the threshold
    drawn in red and flagged frames marked.
    """
    setup_style()
    s = band_summary(res)
    stat = res["stat"]
    x = (meta["frame_indices"] if meta is not None
         else np.arange(stat.shape[0], dtype=np.int64))
    if vmax is None:
        vmax = 1.5 * res["z"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)

    im = axes[0].imshow(stat.T, aspect="auto", vmin=0, vmax=vmax, cmap="inferno",
                        origin="lower",
                        extent=(x[0], x[-1], 0, stat.shape[1]))
    fig.colorbar(im, ax=axes[0], pad=0.01, label="z")
    axes[0].set_ylabel("Row (y)")
    title = "Horizontal Banding — Row Deviation Map"
    axes[0].set_title(f"{title}  —  {label}" if label else title)

    axes[1].plot(x, res["trace"], lw=0.6, color="#1f77b4")
    axes[1].axhline(res["z"], color="#d62728", ls="--", lw=1.2,
                    label=f"Threshold (z = {res['z']})")
    if s["n_bad_frames"]:
        bf = s["bad_frames"]
        axes[1].plot(x[bf], res["trace"][bf], ".", color="#d62728", ms=3,
                     label=f"{s['n_bad_frames']} banded frames "
                           f"({100 * s['frac_bad_frames']:.2f} %)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Max z")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if plot:
        plt.show()

    return dict(fig=fig, axes=axes, **s)


def plot_score_distribution(res, candidates=(4, 6, 8, 10, 15, 20), bins=200,
                            plot=True):
    """Distribution of the detection statistic, for choosing the threshold.

    **Left** — histogram of the per-row statistic on a log count axis.  Noise
    forms the bulk near zero; banded rows form a long right tail.  A good
    threshold sits in the gap between them.  **Right** — how many frames each
    candidate threshold would flag.
    """
    setup_style()
    stat = res["stat"]
    rows_table = []
    for c in candidates:
        bad = _flag_rows(stat, c, res["dilate"])   # same dilation as detect_bands
        rows_table.append(dict(
            z=float(c),
            banded_frames=int(bad.any(axis=1).sum()),
            pct_frames=100.0 * float(bad.any(axis=1).mean()),
            pct_rows=100.0 * float(bad.mean()),
        ))
    table = pd.DataFrame(rows_table)

    # Clip into a window around the threshold: the tail runs to z ~ 100 and
    # would otherwise squash the noise bulk into a single bin.
    hi = 1.5 * max(max(candidates), res["z"])
    lo = 0.0 if res["polarity"] == "both" else -hi

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].hist(np.clip(stat.ravel(), lo, hi), bins=bins, range=(lo, hi),
                 color="#1f77b4")
    axes[0].axvline(res["z"], color="#d62728", ls="--", lw=1.2,
                    label=f"Current threshold (z = {res['z']})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(f"Row statistic (z, clipped to {hi:g})")
    axes[0].set_ylabel("Rows (count)")
    axes[0].set_title("Detection Statistic — Noise Bulk vs. Band Tail")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(table["z"], table["pct_frames"], "o-", color="#1f77b4")
    axes[1].axvline(res["z"], color="#d62728", ls="--", lw=1.2)
    axes[1].set_xlabel("Threshold (z)")
    axes[1].set_ylabel("Frames flagged (%)")
    axes[1].set_title("Threshold Sensitivity")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if plot:
        plt.show()

    return dict(fig=fig, axes=axes, table=table)


def plot_row_profile(res, frame, meta=None, plot=True):
    """Row-mean profile of one frame against its baseline, with bands marked.
    """
    setup_style()
    y = np.arange(res["rows"].shape[1])
    orig = (meta["frame_indices"][frame] if meta is not None else frame)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(y, res["rows"][frame], lw=0.9, color="#1f77b4", label="Row mean")
    ax.plot(y, res["baseline"][frame], lw=1.4, color="#ff7f0e",
            label=f"Baseline (median filter, win = {res['win']})")
    flagged = np.flatnonzero(res["bad"][frame])
    if flagged.size:
        ax.plot(y[flagged], res["rows"][frame][flagged], ".", color="#d62728",
                ms=6, label=f"{flagged.size} flagged rows")
    ax.set_xlabel("Row (y)")
    ax.set_ylabel("Mean pixel value")
    ax.set_title(f"Row Profile — frame {orig}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if plot:
        plt.show()

    return dict(fig=fig, ax=ax)


def plot_worst_frames(path, res, meta=None, n=3, plot=True):
    """Raw / corrected / removed-signal images for the most strongly banded frames.

    The third panel shows exactly what the correction took out, on a symmetric
    diverging scale in pixel units.  A clean result shows horizontal lines there
    and nothing resembling neurons or vignette.
    """
    setup_style()
    s = band_summary(res)
    if not s["n_bad_frames"]:
        print("No banded frames were detected; nothing to show.")
        return dict(frames=np.array([], dtype=np.int64), raw=None,
                    corrected=None, fig=None)

    order = res["trace"].argsort()[::-1]
    sel   = order[:min(n, s["n_bad_frames"])]
    orig  = (meta["frame_indices"][sel] if meta is not None else sel)

    raw   = read_frames(path, orig)
    fixed = correct_bands(raw, res["resid"][sel], res["bad"][sel])
    diff  = raw - fixed

    fig, axes = plt.subplots(len(sel), 3, figsize=(13.5, 4.3 * len(sel)),
                             squeeze=False)
    for i, (t_prof, t_orig) in enumerate(zip(sel, np.atleast_1d(orig))):
        vmin, vmax = np.percentile(raw[i], [1, 99.5])
        amp = float(np.abs(diff[i]).max()) or 1.0
        panels = [
            (raw[i], f"Raw — frame {t_orig}  (z = {res['trace'][t_prof]:.0f})",
             "gray", dict(vmin=vmin, vmax=vmax), False),
            (fixed[i], "Corrected", "gray", dict(vmin=vmin, vmax=vmax), False),
            (diff[i], f"Removed (peak {amp:.1f} px)", "RdBu_r",
             dict(vmin=-amp, vmax=amp), True),
        ]
        for ax, (img, title, cmap, kw, cbar) in zip(axes[i], panels):
            im = ax.imshow(img, cmap=cmap, aspect="equal", **kw)
            if cbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                             label="Pixel units")
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    if plot:
        plt.show()

    return dict(frames=np.atleast_1d(orig), raw=raw, corrected=fixed, fig=fig)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run(paths, labels=None, subsample=1, win=DEFAULT_WIN, z=DEFAULT_Z,
        dilate=DEFAULT_DILATE, polarity=DEFAULT_POLARITY, plot=True, n_worst=2,
        save_mask=False, write_video=False, stamp=None, codec=DEFAULT_CODEC):
    """Detect banding in one or more videos, under one or more parameter sets.

    ``win``, ``z``, ``dilate`` and ``polarity`` each take a scalar or a list;
    lists are expanded into every combination (:func:`param_grid`) and each one is
    processed in turn.  Every video is decoded once for its row profiles and the
    combinations then reuse them.

    With ``save_mask`` or ``write_video``, each combination writes into its own
    folder, ``<session_dir>/denoised/<MMDDYYYY-HHMMSS>/<tag>/``; a single ``stamp``
    covers the whole call, so one sweep is one timestamped folder.  ``codec`` picks
    the encoding of those videos — ``"FFV1"`` (lossless) or ``"MPEG4"``.

    Returns a DataFrame with one row per (video, parameter combination) — the
    arrays are not kept, since a full result is ~0.2 GB.  Rebuild the combination
    you settle on with :func:`detect_bands`.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    if labels is None:
        labels = [os.path.basename(str(p)) for p in paths]
    if len(labels) != len(paths):
        raise ValueError(f"len(labels)={len(labels)} != len(paths)={len(paths)}")

    params = param_grid(win=win, z=z, dilate=dilate, polarity=polarity)
    stamp = stamp or datetime.now().strftime(RUN_STAMP_FMT)
    writing = save_mask or write_video

    table = []
    for i, (path, label) in enumerate(zip(paths, labels)):
        print(f"\n[{i + 1}/{len(paths)}] {label}  —  {len(params)} parameter set(s)")
        rows, meta = compute_row_profiles(path, subsample=subsample)

        for p in params:
            tag = param_tag(p)
            print(f"\n-- {tag}")
            res = detect_bands(rows, **p)
            summary = print_band_summary(res, meta)

            if plot:
                plot_band_map(res, meta, label=f"{label} — {tag}")
                if n_worst:
                    plot_worst_frames(path, res, meta, n=n_worst)

            out_dir = make_run_dir(path, stamp=stamp, tag=tag) if writing else None
            mask_path = save_band_mask(res, out_dir=out_dir) if save_mask else None
            video_path = (destripe_video(path, res, out_dir=out_dir, codec=codec)
                          if write_video else None)

            table.append(dict(
                label=label, path=str(path), tag=tag, **p,
                sigma=res["sigma"],
                banded_frames=summary["n_bad_frames"],
                pct_frames=100.0 * summary["frac_bad_frames"],
                pct_rows=100.0 * summary["frac_bad_rows"],
                rows_per_bad_frame=summary["rows_per_bad_frame"],
                max_amplitude=summary["max_amplitude"],
                mask_path=mask_path,
                video_path=video_path,
            ))
    return pd.DataFrame(table)
