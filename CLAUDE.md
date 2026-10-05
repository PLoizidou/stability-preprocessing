# CLAUDE.md — stability-preprocessing

## Project Overview

This pipeline preprocesses one-photon miniscope calcium imaging data and paired behavioral video recordings for a study of hippocampal activity stability. Raw `.avi` files from miniscope and behavior cameras are organized into per-session folders, optionally concatenated across same-day recordings, trimmed to task-relevant periods using DeepLabCut pose estimates, and then processed through CaImAn (CNMF-E algorithm) to extract spatial and temporal neural components. Behavior trajectories (linear track and T-maze) are separately processed to extract trial structure. The mice in this dataset across two cohorts are labeled AGING COHORT = `Mouse163`, `Mouse1636`, `Mouse1637`, `Mouse1639`, and YOUNG COHORT: 'Mouse945', 'Mouse946', 'Mouse847', and 'Mouse943', 'Mouse944'. From the aged cohort, Mouse1639 and Mouse1637 also have rounds 3 and 3.5 which were experiments done approximately four months after rounds 1 and 1.5 were done. Round 3 consisted of both linear and TMaze track recordings. Round 3.5 only consisted of linear track recordings. From the young cohort, only Mouse 945 has a second round (usually named Mouse 945-Round2) of experiments which again took place approximately four months the first round of experiments. 

**Mouse905** is not part of either cohort and should not be processed further — it went through the Linear track portion of the experiment for 10 days before its field of view degraded too badly to continue.

---

## Environment

```bash
conda env create -f environment.yml
conda activate stability-preprocessing   # for all scripts except pose estimation
conda activate dlc                        # only for estimate_pose.py
```

**Conda env name:** `stability-preprocessing`  
**Python:** 3.11.8  
**Key packages:** `caiman=1.11`, `numpy=1.26.4`, `pandas=2.2.1`, `scipy=1.13`, `scikit-learn=1.4.1`, `matplotlib=3.8.4`, `seaborn=0.13.0`  
**Not in environment.yml but used at runtime:** `pynwb`, `neuroconv`, `opencv-python` (`cv2`), `cupy`, `holoviews`, `bokeh`, `plotly`, `ffmpeg` (system binary)

---

## Data Structure

### Raw data (before processing)

Raw data is spread across whichever external hard drives happen to be plugged in — there is no single canonical drive or path. Layout also varies by drive: some group mice under a cohort folder (e.g. `AGING MICE`), while others have mouse folders directly at the drive root with no wrapping folder. Always check all currently mounted drives under `/media/toor/` when looking for a mouse's raw data.

```
# Cohort-folder layout
/media/toor/Seagate Portable Drive/AGING MICE/
    Mouse163/
        miniscope2024-05-28T13_15_08.avi
        timestamps2024-05-28T13_15_08.csv
        behavior2024-05-28T13_15_08.avi
        behaviorLinear2024-05-28T13_15_08.avi   # optional, if linear track session
        ...
    Mouse1636/
        ...

# Flat layout — mouse folder directly at drive root, no cohort wrapper
/media/toor/Seagate2/
    Mouse847/
        miniscope2025-11-21T18_22_12.avi
        timestamps2025-11-21T18_22_12.csv
        Linear2025-11-21T18_22_12.avi
        ...
    Mouse943/
        ...

/media/toor/SeagatePortableDrive/
    Mouse943/
        ...
```

### DLC-based trimming (`trim_videos_from_dlc.py`)

```
<output_root>/<TIMESTAMP>/
    TMaze_miniscope<TIMESTAMP>.avi      # trimmed to 960 s from task start
    TMaze_behavior<TIMESTAMP>.avi
    dlc_trim_info.csv                   # columns: start_frame, start_time_seconds, most_robust_bodypart
    dlc/
        *_filtered.csv                  # DLC pose predictions (MultiIndex header)
```

### Session folder lifecycle

`awaiting_dlc/`, `awaiting_caiman/`, and `pre_concat/` are **temporary staging folders** — they only exist to show at a glance which sessions have gone through which preprocessing stage. `RAW_DATA/` and `Round<N>/<Task>/` are the permanent locations. A session folder moves through these locations under `<mouse_dir>/` (e.g. `Seagate3/Mouse944/`) as it progresses:

```
<mouse_dir>/RAW_DATA/awaiting_dlc/<TIMESTAMP>/     # organized, DLC not yet run
    Home/Linear/Miniscope<TIMESTAMP>.avi + timestamps<TIMESTAMP>.csv
    no_miniscope/<TIMESTAMP>/                      # sessions with no Miniscope recording (see step 3) — dead end,
                                                    #   doesn't proceed further down this lifecycle

    # -- DLC runs (step 4) --> move the whole session folder out of RAW_DATA/awaiting_dlc/ into:

<mouse_dir>/RAW_DATA/<TIMESTAMP>/                  # DLC done — PERMANENT home for raw + dlc/, regardless of downstream stage
    Home/Linear/Miniscope<TIMESTAMP>.avi + timestamps<TIMESTAMP>.csv
    dlc/*_filtered.csv

    # -- trim_videos_from_dlc.py runs (step 5), reading from RAW_DATA/<TIMESTAMP>/ --> writes to:

<mouse_dir>/Round<N>/<Task>/awaiting_caiman/<TIMESTAMP>/   # trimmed, dlc/ copied, not yet run through CaImAn
    Linear_Miniscope<TIMESTAMP>.avi / Linear_Linear<TIMESTAMP>.avi  (or TMaze_ prefix)
    dlc_trim_info.csv
    dlc/*_filtered.csv

    # -- CaImAn runs (step 6), writing caiman_final/ into the same awaiting_caiman/<TIMESTAMP>/ folder -->
    # -- once done, move the whole session folder out of awaiting_caiman/ into:

<mouse_dir>/Round<N>/<Task>/<TIMESTAMP>/           # FINAL location — fully preprocessed
    Linear_Miniscope<TIMESTAMP>.avi / Linear_Linear<TIMESTAMP>.avi
    dlc_trim_info.csv
    dlc/*_filtered.csv
    caiman_final/caiman_results.hdf5
```

`<Task>` is `Linear` or `TMaze` depending on the arena; `Round<N>` (e.g. `Round1`, `Round1.5`, `Round3`) identifies which recording round, matching the convention already used for other mice (e.g. `Mouse1639/Round3/TMaze`, `Mouse163/Round1.5/Linear`).

**Pre-concatenation raw sub-recordings** (the individual same-day recordings merged by `concat_session_videos.py`, see step 2) go into a flat `<mouse_dir>/RAW_DATA/pre_concat/` folder — not per-session subfolders, just the loose files — once concatenation is done, so they don't clutter the working raw-data listing while still being kept for audit.

Sessions missing a Miniscope recording use a `no_miniscope/` sibling at whichever staging level applies (see step 3) instead of following the rest of this lifecycle, since they can't be trimmed/run through CaImAn.

### CaImAn output

```
<session_dir>/caiman_final/caiman_results.hdf5       # young cohort
<session_dir>/caiman_aging/caiman_results.hdf5       # aging cohort (2026-08 re-run)
```

⚠️ **The aging-cohort directory is `caiman_aging/`, not `caiman_final_aging/`.** An earlier version
of this section named a directory that **does not exist anywhere on disk**; code following it
silently found nothing. Verified counts of directories actually containing `caiman_results.hdf5`:
`caiman_final` 855, `caiman_aging` 255, `caiman` 179, `caiman_final_young` 4, plus one-off
parameter-sweep variants (`caiman00`-`caiman03`, `caiman2`, `caiman_final_v2`, ...).

**Legacy naming:** a small number of sessions processed before this convention was adopted still use `caiman_final_young/` (2 known sessions: `Mouse943/Round1/Linear/2025-11-17T17_15_35` and `2025-11-18T17_44_50`). Code that locates CaImAn output for a session should check `caiman_final/`, then `caiman_aging/`, then fall back to `caiman_final_young/`.

### Temporary files (deleted after preprocessing by default)

```
/tmp/caiman_data/                          # memmap files for adult mice (SSD, ~115 GB)
/media/toor/Seagate2/MEMMAPS/             # memmap files for young mice
/media/toor/Seagate3/caiman_data/         # source CaImAn data dir (copied to /tmp)
```

---

## Naming Conventions

### Timestamps

All filenames use ISO 8601 with underscores replacing colons: `YYYY-MM-DDTHH_MM_SS`  
Timestamps inside CSV files use colons: `YYYY-MM-DDTHH:MM:SS`

### Video file types

| Prefix | Content |
|--------|---------|
| `miniscope<TIMESTAMP>.avi` / `Miniscope<TIMESTAMP>.avi` | Raw calcium imaging |
| `Home<TIMESTAMP>.avi` (young/post-May cohorts) | Home box camera |
| `Linear<TIMESTAMP>.avi` (young/post-May cohorts) | **Arena** camera — the track or maze, whichever is in the room |
| `behavior<TIMESTAMP>.avi` (aging cohort, 2024) | **Arena** camera |
| `behaviorLinear<TIMESTAMP>.avi` (aging cohort, 2024) | Home box camera |
| `homecage<TIMESTAMP>.avi` (Baseline recordings only) | Home box camera |
| `miniscopeBaseline<TIMESTAMP>.avi` (Baseline recordings only) | Raw calcium imaging |

⚠️ **The aging cohort's two behaviour filenames mean the opposite of what they look like.**
`behavior<TIMESTAMP>.avi` is the arena camera (both linear tracks / the T-maze, with the water ports
visible) and `behaviorLinear<TIMESTAMP>.avi` is a close-up of the home cage. Verified two ways:
frames extracted from each, and the trimmer's own output — the aging cohort has 207
`Linear_behavior*` and 146 `TMaze_behavior*` trimmed arena videos and **zero** `*_behaviorLinear*`.
An earlier version of this table had these two reversed; anything written against it should be
re-checked. The young and post-May cohorts follow the intuitive `Home`/`Linear` naming.

⚠️ **Baseline recordings use their own filename scheme.** `Mouse1637/BaselineRecordings/` (the only
Baseline set) pairs `homecage<TIMESTAMP>.avi` with `miniscopeBaseline<TIMESTAMP>.avi` and has **no
arena video at all** — it is a home-cage-only recording made before sacrifice. Any scan that matches
only the seven prefixes above will report these sessions as "timestamps, no video". There is also one
known typo, `Liner2025-11-10T17_24_58.avi`, in a Mouse847 denoising-test folder.
| `TMaze_miniscope...` / `Linear_miniscope...` | Trimmed, task-labeled versions |

### Session directories

After curation: `ses-<YYYYMMDDTHHMMSS>` (no hyphens/colons in the timestamp part)

### Mouse IDs

Used as directory names and as keys in per-mouse parameter dicts. Known IDs: `Mouse163`, `Mouse1636`, `Mouse1637`, `Mouse1639`. The CROPS dict in preprocessing scripts keys on the numeric suffix only (e.g., `"946"`, `"847"`, `"945"`, `"943"`, `"944"`).

---

## Pipeline Order

### 1. Sort recordings by environment (if needed)

Not always required — only run this when a mouse's behavior videos still need to be classified into Home/Linear/TMaze (e.g. filenames don't already indicate the arena).

```bash
# (base) env
python sort_recordings_by_environment.py <mouse_dir>
```

Classifies each behavior video as Home, Linear, or TMaze based on image brightness and aspect ratio. **Each mouse requires individually tuned `BRIGHTNESS_THRESHOLD` and `ASPECT_RATIO_CUTOFF`** — see "Sorting Parameters" section below.

---

### 2. Concatenate same-day recordings (if needed)

Only required when multiple recordings exist for a single session. Probably needed for all young mice (i.e. `Mouse945`, `Mouse946`, `Mouse847`, `Mouse943`, 'Mouse944' need to be stitched together on days with multiple recordings). Runs before organizing into per-session folders.

```bash
# (base) env
python concat_session_videos.py /media/toor/SeagatePortableDrive/Mouse943
```

Merges same-day recordings per modality. Output filenames use the earliest timestamp minus 1 second. This produces a new, separate output file — the original raw recordings are never modified or deleted (see "Important Notes for New Collaborators" #1).

**⚠️ A same-day group of recordings is not guaranteed to be one arena.** `concat_session_videos.py` groups purely by calendar date — it has no idea what arena a recording is, and the camera's fixed `Linear<TIMESTAMP>.avi` filename (see "Naming Conventions") gives no hint either, since it's written regardless of true arena. Confirmed on Mouse944/Mouse945-Round2 data from 2026-07-23 through 2026-08-04: **every single recording day for both mice had at least one arena switch mid-day** (e.g. Linear → TMaze → Linear across three separate recordings on Mouse944's 2026-08-01), driven by same-day back-to-back Linear+TMaze sessions rather than incidental restarts. Blindly concatenating a whole day's recordings together — or even just visually spot-checking a single frame from the *merged* output — will silently splice two different arenas' footage into one file. Before grouping same-day recordings for concatenation:
1. Classify **every individual sub-recording's** arena separately (visually — a single sampled frame per file is enough given how visually distinct TMaze's cross-shaped apparatus is from the Linear track, but sample more than one frame per file if unsure, since a switch could in principle happen *within* one recording, not just between them).
2. Only merge **contiguous runs of the same arena** within a day — a day with Linear→TMaze→Linear splits into three separate sessions, not one 3-way merge.
3. Do not trust automated aspect-ratio/brightness heuristics (`sort_recordings_by_environment.py`'s `BRIGHTNESS_THRESHOLD`/`ASPECT_RATIO_CUTOFF`) as the sole classifier near their cutoff boundaries — they have been observed to misclassify TMaze as Linear (and vice versa) even on frames that are visually unambiguous once viewed directly.

**After concatenating, move the pre-concat raw sub-recordings into `<mouse_dir>/RAW_DATA/pre_concat/`** (flat — just the loose files, no per-session subfolders) rather than leaving them next to the concatenated output — keeps the working directory limited to the files actually used downstream while still preserving the originals for audit. Only the sub-recordings that were merged move here; single-recording days are untouched and proceed as normal sessions.

---

### 3. Organize into per-session folders

Run `scripts/organizing_to_folders.ipynb` — straightforward notebook, follow cells in order.

Organize new sessions into `<mouse_dir>/RAW_DATA/awaiting_dlc/<TIMESTAMP>/` (see "Session folder lifecycle" above).

**Sessions missing a Miniscope recording:** some sessions have Home/Linear/timestamps but no paired `Miniscope*.avi` (e.g. the camera wasn't recording that day). Organize these into a sibling `RAW_DATA/awaiting_dlc/no_miniscope/<TIMESTAMP>/` folder instead of the normal `RAW_DATA/awaiting_dlc/<TIMESTAMP>/` — keeps them out of scripts that assume all three modalities are present (DLC itself only needs the Linear video, but downstream trimming/CaImAn steps need Miniscope), while still preserving the behavior data for later use.

---

### 4. Run DeepLabCut pose estimation

Requires the `dlc` conda environment.

```bash
conda activate dlc
python ~/Desktop/stability-preprocessing/scripts/estimate_pose.py \
    /home/toor/Desktop/linear-abrotman-2024-05-29/config.yaml \
    /media/toor/Seagate2/Mouse847/RAW_DATA/awaiting_dlc/2025-11-21T18_22_12/Linear2025-11-21T18_22_12.avi \
    --gpu_id 0
```

Which DLC project config to pass depends on which environment (arena) the recordings were done in:

| Environment | Config path |
|---|---|
| T-Maze | `/home/toor/Desktop/T-Maze-PLoizidou-2025-06-16/config.yaml` |
| Linear track | `/home/toor/Desktop/linear-abrotman-2024-05-29/config.yaml` |
| Home | `/home/toor/Desktop/home-abrotman-2024-05-29/config.yaml` |

If the command fails to run, try unsetting `LD_LIBRARY_PATH` first (library conflict):

```bash
unset LD_LIBRARY_PATH
```

Outputs `*_filtered.csv` pose predictions into a `dlc/` subdirectory alongside the video.

**Once DLC is done for a session, move its whole folder out of `RAW_DATA/awaiting_dlc/` into `<mouse_dir>/RAW_DATA/<TIMESTAMP>/`** (up one level, out of `awaiting_dlc/`) — see "Session folder lifecycle" above. `RAW_DATA/awaiting_dlc/` should only ever contain sessions still pending DLC.

**Linear track snapshot versions:** some linear track sessions were processed with two different DLC model snapshots living side by side in the same session's `dlc/` folder (example: `/media/toor/SeagatePortableDrive/AgingMiceNWB/sub-Mouse1637/Round1/Linear/ses-20240524T184216/dlc`). **Always prefer the output produced with snapshot `160000`** over any other snapshot. If a session's `dlc/` folder does not contain a `160000` snapshot, add that session to the "Pending DLC re-processing" list below so it gets re-run with the correct snapshot.

---

### 5. Trim videos to task period

```bash
# (base) env — T-maze example
python trim_videos_from_dlc.py \
    /media/toor/Seagate2/Mouse1639/Round3/TMaze \
    /media/toor/Seagate2/Mouse1639/Round3/TMaze_trimmed \
    --task tmaze

# Linear track example
python trim_videos_from_dlc.py \
    /media/toor/Seagate2/Mouse163/Round1.5/Linear \
    /media/toor/Seagate2/Mouse163/Round1.5/Linear_trimmed \
    --task linear

# Generate only dlc_trim_info.csv without copying videos
python trim_videos_from_dlc.py \
    /media/toor/Seagate2/Mouse163/Round1.5/Linear \
    /tmp/unused \
    --task linear --csv-only
```

Crops videos to 960 s (=16 mins) from task start (tmaze: first frame `y > 45 px`; linear: first frame `y < 715 px`, both requiring `likelihood > 0.9` for 10 consecutive frames).

⚠️ **960 s is the requested duration, not the delivered one — session length varies a lot and a
short session is usually fine, not broken.** `ffmpeg -t 960` stops early when the raw recording
runs out after the task-start frame, so the real clip length depends on how much footage was left.
Measured across the aging cohort (frame count of `estimates.F_dff` ÷ 25, 2026-09-28):

| Mouse / round / task | n | full ~960 s | short | shortest |
|---|---|---|---|---|
| Mouse163 — Round1 + Round1.5 Linear | 39 | 0 | **39** | 900 s |
| Mouse1637 — Round1 + Round1.5, both tasks | 58 | 0 | **58** | 895 s |
| Mouse1636 — Round1 Linear, Round1.5 TMaze | 30 | 30 | 0 | 960 s |
| Mouse1636 — Round1.5 Linear | 14 | 12 | 2 | 790 s |
| Mouse1639 — Round1.5 Linear | 17 | 9 | **8** | 669 s |
| Mouse1639 — Round1.5 TMaze | 17 | 15 | 2 | 751 s |

Two distinct causes, don't conflate them:
* **Mouse163 and Mouse1637 are uniformly ~900 s** across *every* round and task — a flat 60 s
  short of nominal with essentially zero spread. That is a per-animal constant, not truncation.
* **Mouse1639's Round1.5 Linear (and Mouse1636's two short ones) are genuinely truncated** — the
  recording ended before 960 s of task had elapsed, so lengths scatter (669–950 s).

So **do not treat "< 960 s" as a QC failure or a reason to exclude a session.** Round1.5 Linear is
the worst affected, but the variation is cohort-wide. Anything that assumes a fixed 24 000-frame
clip (pre-allocated arrays, hard-coded bin counts, cross-session stacking) must read the real
frame count per session instead. Shortest currently in use is **480 s**
(`Mouse1639/Round1/Linear/2024-05-25T19_15_57`).

**Per the session folder lifecycle above:** `input_root` should be the mouse's `RAW_DATA/` folder (sessions that finished DLC, excluding the `awaiting_dlc/` and `pre_concat/` subfolders), and `output_root` should be `<mouse_dir>/Round<N>/<Task>/awaiting_caiman/` — this is itself the staging folder for step 6, not a final location (older examples above predate this convention and use other output_root names, e.g. `..._trimmed`).

---

### 5.5. Denoise sessions with the Miniscope ring-noise artifact (Mouse944 only, as of 2026-07-22)

**Only needed for Mouse944 so far.** Some Mouse944 Miniscope recordings develop a bright speckled "ring" artifact partway through (looks like a torus/donut of noise overlaid on the FOV) that completely destroys the signal for the affected frames — cause not yet root-caused (suspected loose connector).

**Run this after trimming (step 5), not before DLC** — as of 2026-07-25, denoising moved from pre-DLC to post-trim: DLC and trimming both tolerate the noisy frames fine (DLC just gets a few bad-pose frames wherever the artifact interrupts the video, and trimming's crossing detection ignores them per-frame), so running denoise on the full raw recording was wasted work on footage that gets discarded at the trim step anyway. Denoising the trimmed 960s clip instead of the full raw session cuts the amount of (partial) re-encoding down to only what CaImAn will actually see.

```bash
# (stability-preprocessing env)
python detect_noise_frames.py <mouse_dir>/Round<N>/<Task>/awaiting_caiman/<TIMESTAMP>/Linear_Miniscope<TIMESTAMP>.avi --plot   # inspect first
python denoise_session.py <mouse_dir>/Round<N>/<Task>/awaiting_caiman/<TIMESTAMP>/Linear_Miniscope<TIMESTAMP>.avi
```

(`TMaze_Miniscope<TIMESTAMP>.avi` for T-maze sessions.)

- `detect_noise_frames.py` flags affected frames via Laplacian variance (high-frequency speckle energy) — clean frames score ~20-40, artifact frames ~1300+, so the default `--threshold 200` has a wide safety margin. Writes a per-frame CSV and, with `--plot`, a diagnostic plot of the metric over time; use these to sanity-check before deleting anything.
- `denoise_session.py` re-detects the bad frame ranges the same way, then drops those exact frame indices from the Miniscope video **and** the paired Home/Linear behavior videos **and** `timestamps.csv`, keeping all four in lockstep (they share one synchronized frame clock, one row per frame). Uses a **hybrid strategy** rather than re-encoding everything losslessly (an earlier version did this and inflated a 4.2GB session to 35-43GB, since lossless coding of noisy 1p sensor data barely compresses — doesn't scale across ~40+ sessions):
  - Home/Linear are `mpeg4`/`FMP4` with a strictly regular GOP (empirically verified fixed at 12 frames for Mouse944, checked via a full-file keyframe scan with zero deviation) — these get cut via lossless **stream copy** (ffmpeg concat demuxer, same idiom as `concat_session_videos.py`) at GOP-aligned boundaries, so kept frames stay bit-identical and output size merely shrinks by the dropped fraction (no re-encode at all).
  - Miniscope's own keyframes are **not** reliably on that same grid — the encoder inserts extra keyframes at the noise bursts themselves (scene-cut detection reacting to the artifact), which resets its periodic keyframe counter and permanently phase-shifts later "regular" keyframes off the original grid. (Verified on a real session: of 3498 nominal grid positions, only 439 were still real Miniscope keyframes — requiring a grid position that's *also* a real Miniscope keyframe would have forced dropping 73% of the video.) Since Miniscope is by far the smallest stream (608×608 vs 1936×1464), it's cheap to **re-encode losslessly** (`libx264 -qp 0`) with frame-exact selection instead — a re-encode has no GOP constraint, so it can drop exactly the same frame set as Home/Linear regardless of its own keyframe layout.
  - Each drop window's keyframe-boundary assumption is verified against the real bitstream (via ffprobe's `lavfi movie+select`, which decodes by frame count rather than timestamp-snapped seeking) before any output is written, aborting loudly rather than silently producing a corrupted splice if a future session's encoding doesn't match this pattern.
  - On the real test session this brought total output size for one recording from ~49GB (all-four-lossless) down to ~6.8GB (vs. ~5.7GB raw) — Home/Linear end up *smaller* than raw (stream-copy trimming), Miniscope ends up ~2GB (lossless re-encode of the small stream only).
- **Never touches the trimmed input files** — writes cleaned copies alongside a `denoise_log.csv` recording exactly which frame ranges were dropped (for audit). Since the input path is now the trimmed clip (no longer under `RAW_DATA/`), the script's default output falls back to a sibling `DENOISED/` folder next to the trimmed clip, i.e. `<mouse_dir>/Round<N>/<Task>/awaiting_caiman/<TIMESTAMP>/DENOISED/<TIMESTAMP>/` (override with `--output-root` if a different location is wanted). Point step 6 (CaImAn) at this `DENOISED/<TIMESTAMP>/` folder instead of `awaiting_caiman/<TIMESTAMP>/` for affected sessions.
- If future mice show the same artifact, extend this step to their trimmed sessions too and update this note.

---

### 5.6. Denoise sessions with the Miniscope banding (row-stripe) artifact (Mouse944 Round1, as of 2026-08-24)

**Distinct from the ring-noise artifact above.** V4 Miniscope frames occasionally show horizontal banding: a run of consecutive sensor rows uniformly brighter or darker than its neighbors for one frame — additive, constant along x, and invisible to a spatial band-pass filter tuned to soma size (so it isn't caught by anything upstream). Credit: Aleksandar Marinkovic.

- **Script:** `analysis/src/banding_preprocessing.py` (also `scripts/banding_preprocessing.ipynb`). Detects banded rows via a robust z-score of each frame's row-mean profile against a median-filtered baseline (tunable `--win`/`--z`/`--dilate`/`--polarity`; defaults `win=31, z=10.0, dilate=3, polarity=both`, but check per-recording with the notebook's `plot_score_distribution` — the params actually used for Mouse944 Round1 were `win=101, z=6, dilate=3, polarity=both`, tuned looser than the defaults because that mouse's stripe events span ~50 rows, wider than the default window can baseline correctly).
- **Output location:** `<session_dir>/denoised/<MMDDYYYY-HHMMSS>/<param_tag>/debanded_<Arena>_Miniscope<TIMESTAMP>.avi` — one timestamped run folder per invocation (never overwrites an earlier run), with a `<param_tag>` subfolder per parameter combination (e.g. `win101_z6_dil3_both`) so a parameter sweep can write multiple candidate outputs side by side. `<session_dir>` here is wherever the input video already lives (e.g. `Round<N>/<Task>/awaiting_caiman/<TIMESTAMP>/`), matching the "denoise in place, alongside the input" convention from step 5.5. **Never touches the input video.**
- **Codec: FFV1 lossless by default, and deliberately so** — banding is only a few grey levels deep in most frames, so a lossy re-encode would be the same order of magnitude as the artifact being removed. This makes debanded output **much larger than the source**: a 960s/25fps session went from ~420 MB (source `mpeg4`) to ~2.9 GB (debanded `FFV1`), roughly 7x. The script explicitly calls out `MPEG4` as an option too, but only for previews — **"use it for previews, not for anything that feeds CNMF-E."**
- **⚠️ CaImAn memory footprint on debanded input:** `preproc_caiman_final.py` auto-scales to a large worker pool (up to 32 processes on a 64-core box) sized for the *original* mpeg4 file sizes. Feeding it the ~7x larger FFV1 debanded video at that same parallelism has caused repeated OOM kills (`dmesg` confirms `oom-kill`, process exit code 137) on a 94GB-RAM machine — even with 50+ GB reported free beforehand — especially when other memory-heavy jobs (Jupyter kernels, QC scan scripts, etc.) are running concurrently on the same machine. There's no fix committed yet; until there is, expect CaImAn runs on debanded video to be flaky under concurrent load and plan to retry OOM'd sessions once the machine is quieter (a killed run is always safely retryable — the batch script pattern in step 6 only moves a session to its final location after confirming valid output, so nothing is lost, just re-attempted).
- **A `FAILED`/non-zero exit from `preproc_caiman_final.py` does not necessarily mean no usable output was produced.** Observed once: the full CNMF-E pipeline completed (motion correction, fit, component evaluation all finished, `Num accepted/rejected` printed) and `cnmf_fit.save()` wrote a complete, valid `caiman_results.hdf5` (confirmed via `h5py` — all expected `estimates` keys present, including the one the traceback complained about), but the process still raised `ValueError: Error while saving ndarray cnn_preds of dtype float32` and exited non-zero, apparently after the data was already durably written. **Before treating a failed run as a total loss, check whether `caiman_final/caiman_results.hdf5` exists and opens cleanly** (`h5py.File(path, 'r')`, check `list(f['estimates'].keys())`) rather than trusting the exit code alone.

---

### 6. Run CaImAn

```bash
# (base) env
./run_all_caiman.sh /media/toor/Seagate2/Mouse847/awaiting_caiman
# or for a specific round
./run_all_caiman.sh /media/toor/Seagate3/Mouse1637_testDLC/Round1/Linear
```

For now, `scripts/run_all_caiman.sh` (calls `preproc_caiman_final.py`) is the only shell wrapper needed — used for all mice, adult and young. Point it at `<mouse_dir>/Round<N>/<Task>/awaiting_caiman/` per the current lifecycle convention (the Mouse847 example above predates the `Round<N>/<Task>/` nesting).

Motion correction → CNMF-E → component evaluation → ΔF/F → saves `caiman_results.hdf5` directly inside the session's `awaiting_caiman/<TIMESTAMP>/` folder (as `caiman_final/caiman_results.hdf5`).

**Once CaImAn is done for a session, move its whole folder out of `awaiting_caiman/` into `<mouse_dir>/Round<N>/<Task>/<TIMESTAMP>/` directly** (up one level, out of `awaiting_caiman/`) — this is the final, permanent location per the "Session folder lifecycle" above.

---

### 7. CaImAn QC

- `visualize_caiman_results.ipynb`
- `pick_corr_pnr_thresholds.ipynb`

##### USed for selecting *optimal* caiman parameters
- `compare_caiman_runs.ipynb`
- `caiman_param_sweep_report.ipynb` — iterates over every `caiman_final*` run directory for a mouse, loads each `caiman_results.hdf5`, and saves a `parameter_report.png` (correlation image with accepted/rejected contours + r_value/SNR/CNN distributions) plus `run_parameters.json`/`.txt` into each run directory; also writes a cross-run `parameter_sweep_summary.csv`. Used for comparing CaImAn parameter choices across runs (e.g. the Mouse946 adult-vs-young parameter comparison referenced in "CaImAn Parameters" below).

---

### 8. Register across sessions

- `multisession_registration.ipynb` — matches components across a mouse's sessions using CaImAn's Hungarian-matching registration (`A_union`, `assignments`, `matchings`), at the default `thresh_cost = 0.7` (IoU > 0.3) and also re-run at a stricter `thresh_cost = 0.5` (IoU > 0.5) for comparison; saves the confidence metrics (`iou_scores`, `centroid_dist_scores`, per-component `confidence_df`) plus `.pkl`/`.npy` outputs consumed by the group-level notebook below. Plots include: a grid of session templates (correlation images) side by side; contour overlays of components active across all sessions on the first session's correlation image; a 3-panel figure of mean-IoU histogram / IoU-vs-threshold / other coregistration-confidence summaries; per-mouse bar/heatmap of the fraction of components active in each session; and CaImAn QC scatter/histograms (SNR, r_value, CNN score) for a chosen session's `Estimates`.

  **Coregistration scopes.** All coregistration is within-animal. Four scopes are supported, selected entirely by which `rounds`/`tasks` you list in the notebook's config cell — the matching/confidence-metric code itself is agnostic to which scope is running:

  | # | Scope | Example | `rounds` / `tasks` |
  |---|-------|---------|---------------------|
  | 1 | Within round + task | Mouse1637, Round1, Linear | `rounds=[1]`, `tasks=["Linear"]` |
  | 2 | Cross-task, one round | Mouse1637, Round1, Linear **and** TMaze | `rounds=[1]`, `tasks=["Linear", "TMaze"]` |
  | 3 | Cross-round, one task | Mouse1637, Round1 **and** Round1.5, Linear | `rounds=[1, 1.5]`, `tasks=["Linear"]` |
  | 4 | Pooled — all rounds, all tasks | Mouse1637, every round, every task | `rounds=[1, 1.5, ...]`, `tasks=["Linear", "TMaze"]` |

  Sessions from every listed `(round, task)` combination are pooled and globally re-sorted by timestamp before registration, so listing them out of order is fine, and a `(round, task)` pair that doesn't exist (e.g. `TMaze` wasn't run in `Round1`) is skipped rather than erroring.

  **Naming/location convention**, computed by `build_scope_tag()` / `build_save_dir()` / `build_strict_save_dir()` in the notebook:
  - **Scope tag** — the identifier used in both the folder name and every output filename:
    - Type 1: `Round<N>` (e.g. `Round1`) — unchanged from the original single-round+task convention.
    - Type 2: `Round<N>_CrossTask` (e.g. `Round1_CrossTask`).
    - Type 3: `<Task>_CrossRound_Round<N1>-Round<N2>[-Round<N3>...]` (e.g. `Linear_CrossRound_Round1-Round1.5`).
    - Type 4: `AllSessions`.
  - **Location** — Type 1 outputs stay exactly where they've always lived: `<mouse_dir>/Round<N>/<Task>/<caiman_v>_Registration_Round<N>/` (this is deliberate — `group_coregistration_quality_comparison.ipynb` already globs `confidence_metrics_Round*.pkl` from that path, and the Type-1 scope tag was chosen to match it exactly). Types 2–4 have no single owning `Round/Task` folder, so they go under a mouse-level folder instead: `<mouse_dir>/Coregistration/<caiman_v>_Registration_<scope_tag>/`.
  - **Filenames** inside either location follow `<thing>_<scope_tag>.<ext>`, e.g. `spatial_union_Round1_CrossTask.npy`, `confidence_metrics_AllSessions.pkl`, `session_dirs_Linear_CrossRound_Round1-Round1.5.pkl`. A `scope_info_<scope_tag>.json` (mouse id, hard disk, `rounds`, `tasks`, resolved session dirs) is also saved alongside, for reproducibility of exactly which sessions went into a given cross-task/cross-round/pooled run.
  - **Stricter-threshold reruns** — `build_strict_save_dir(save_dir, thresh_cost)` nests a `stricter_coregistration_thresh_cost_<thresh_tag>/` subfolder underneath whichever `save_dir` the scope resolved to (`<thresh_tag>` = `thresh_cost` with `.` → `_`, e.g. `thresh_cost=0.5` → `stricter_coregistration_thresh_cost_0_5/`). It's parameterized by the exact `thresh_cost` value rather than hardcoded to `0.5`, so reruns at different thresholds for the same scope (e.g. `0.5` and `0.4`) coexist side by side instead of overwriting each other. Files inside follow the same `<thing>_<scope_tag>.<ext>` convention as the default run (scope tag only — the threshold is already captured by the subfolder name, so it isn't repeated in the filenames).

  `group_coregistration_quality_comparison.ipynb` currently only consumes Type-1 outputs (per its `MOUSE_REGISTRY`); it has not yet been extended to load Type 2–4 scopes.
- `group_coregistration_quality_comparison.ipynb` — loads every mouse's `confidence_metrics_Round*.pkl` (default and strict threshold) and produces group-level, publication-style comparison figures: (1) per-mouse violin plots of mean IoU and mean centroid distance; (2) group bar chart of mean IoU / IoU-stability / centroid-distance metrics with individual mouse dots + SEM; (3) default-vs-strict threshold comparison panel; (4) longitudinal stability plot of median IoU per session index, colored by young vs. old cohort; (5) soma-radius-calibration plot (pixel radius from footprint area vs. expected ~15–25 µm CA1 soma diameter), also split young vs. old.
Old and young mice were recorded using different types of microscopes which warrants the necesity for tests that the quality of coregistration is comparable. 

---

### 9. Generate trialwise binned location

- `Linear_combined_pipeline.ipynb` — linear track (trajectory cleaning, movement/port-crossing detection, trial segmentation, spatial binning). Produces per-session frame-wise and trial-wise CSVs (`process_lineartrack_session`), plus sanity-check plots: raw XY scatter with the port ROIs drawn as circles (`plot_with_rois`, used once to calibrate `PORT_ROIS_DEFAULT`), a grid of per-trial X-vs-Y trajectory panels (`plot_trials_xy_grid`), and a grid of per-trial spatial-bin-vs-frame panels (`plot_trials_bins_grid`).
- `TMaze_combined_pipeline.ipynb` — combines TMaze DLC coordinate processing, trajectory cleaning, movement detection, and trial segmentation (can also be exported as a `.py` script for batch processing). Sanity-check plots include: a two-panel session overview (raw vs. cleaned trajectory, ROIs, bounding boxes, movement mask) via `plot_session_overview`; a maze-geometry plot showing cleaned positions, port centers, the branch point, and BL/BR/LR polylines (`plot_tmaze_branch_and_polylines`); a quick per-trial 2D line plot colored by trial id (`plot_trials_quick`); a mid-pipeline check plot of cleaned trajectory/ROIs/bounding boxes/moving-vs-not (`plot_session_midpoint_check`); a 2D linearization-check plot colored by `linear_pos` per corridor (`plot_linearization_2d`); and a linear-position-vs-frame plot per example trial with the corresponding spatial bin (`plot_linear_pos_per_trial`).

---

### 9.5. Reward outcome (Bpod) — **planned, not yet implemented**

Reward delivery is driven by a Bpod state machine running a custom protocol, which writes one
MATLAB `.mat` log per session. As of 2026-09-04 this is **not yet wired into the pipeline** — the
conceptual plan lives in [docs/reward_extraction_plan.md](docs/reward_extraction_plan.md) and the
known data problems in [docs/bpod_reward_log_anomalies.md](docs/bpod_reward_log_anomalies.md).
Read both before writing any reward code. The facts below are the ones that constrain how reward
can be recovered.

#### Where the Bpod data lives

```
/media/toor/SeagateClean/Bpod Local/Protocols/LinearTrack/LinearTrack.m
/media/toor/SeagateClean/Bpod Local/Protocols/TMaze4/TMaze4.m
/media/toor/SeagateClean/Bpod Local/Protocols/TMaze2/TMaze2.m      # earlier T-maze protocol, different semantics
/media/toor/SeagateClean/Bpod Local/Data/Mouse<ID>/<Protocol>/Session Data/Mouse<ID>_<Protocol>_<YYYYMMDD>_<HHMMSS>.mat
```

`/media/toor/SeagateClean/Bpod Local` is the canonical copy — a strict superset of `Seagate3/Bpod
Local` and the two `Bpod RAW/Bpod Local` trees. Files are MATLAB v5, so `scipy.io.loadmat` reads
them. **Glob recursively, take the protocol from the filename (not the containing folder), and
de-duplicate on `(mouse, protocol, timestamp)`** — `TMaze4/Session Data/` contains TMaze2 files and
nested `Round2`/`Round3`/`Round1.5` subfolders (see the anomalies doc, §4).

Key `SessionData` fields: `Info.SessionStartTime_UTC` (**actually local time**, equal to the
filename timestamp), `TrialStartTimestamp`/`TrialEndTimestamp` (s since session start),
`RawEvents.Trial{i}.States.Reward` = `[t_on, t_off]` within-trial (the valve window),
`States.ReturnMiddle` (T-maze error), `Events.Port<N>In/Out`, `TrialTypes`, and — TMaze4 only —
`LightOnTrials`.

#### Every trial in a Bpod log is a rewarded trial

In both protocols `WaitForPoke` can only be left by poking the correct port (→ `Reward`) or, in the
T-maze, a wrong port (→ `ReturnMiddle`, which loops back), and a trial is only appended once
`RunStateMachine` returns. The only unrewarded trials come from the linear track's 30-min session
timeout (`Tup → exit`, leaving `States.Reward = NaN`). 31 sessions have them, all `LinearTrack`, all
at `dur ≈ 1800 s`: 23 with one such trial, and 8 with two — in the latter the loop runs one more
iteration whose `Timer = totalSessionTime - timeElapsed` is already ≤ 0, producing a **zero-duration
trial that should be dropped** (verified on `Mouse944_LinearTrack_20260627_151928`: trial 15 is a
real 441 s timeout, trial 16 lasts 0.0001 s).

**So the Bpod log by itself is not a rewarded/unrewarded label.** The reward contrast has to come
from joining Bpod's rewarded events onto the behaviour-derived trials from step 9, which segment
*every* port-to-port traversal including unrewarded ones (linear track: re-poking the port you were
just rewarded at does nothing; T-maze: each `ReturnMiddle` entry is a wrong-arm visit).

#### Port numbering differs between protocols

| Protocol | Port 1 | Port 2 | Port 3 | Port 4 | Port 5 |
|---|---|---|---|---|---|
| `LinearTrack` | — | "right" end | — | "left" end | — |
| `TMaze2` | middle | left | right | — | — |
| `TMaze4` | middle | — | left | — | right |

Do not hard-code one mapping. Mouse1637/Mouse1639 `Round3` mixes both T-maze protocols *within* the
round (TMaze2 on 2024-10-02…10-04, TMaze4 from 2024-10-29).

#### Water-port illumination patterns — what is visible in the behaviour video

Reward is visible in the behaviour video because the valve opens and the port LED lights. The valve
window is `GetValveTimes(20 µl)` ≈ **70–120 ms**, i.e. only **2–3 frames at 25 fps**. What surrounds
that flush differs by protocol:

| Protocol | LED while the mouse is running the trial | At reward |
|---|---|---|
| `LinearTrack` | **off** — clean, the LED is only ever on at reward | LED + valve on the rewarded port, 70–120 ms |
| `TMaze2` | **always on**, on the port to be visited next, for the whole trial | LED stays on; valve flushes ~85–115 ms; LED goes **off** at the end of `Reward` |
| `TMaze4` | on **only** for trials listed in `LightOnTrials` | same |
| both T-mazes | middle-port LED on for the entire `ReturnMiddle` (error) state — **no water delivered** | — |

Consequences:
* Linear track — brightness onset at a port ⇒ reward. Straightforward.
* T-maze — a lit port does **not** mean reward. Ports stay lit for long stretches as a "go here
  next" cue, and the middle port is lit throughout every error return. Detect the *edges*: LED
  **onset** on an uncued trial, LED **offset** at the end of `Reward` on a cued one (in `TMaze2`,
  where the LED is always on, the offset edge is the only usable one), plus the water-meniscus
  movement, which is LED-independent.
* `TMaze4`'s `LightOnTrials = randperm(90, 45)` is drawn over the *maximum* 90 trials while sessions
  end after ~15–35, so only a small random fraction of completed trials are actually cued (e.g. 5 of
  34 in `Mouse944_TMaze4_20260723_153813`). Always intersect with `1:nTrials`; never assume 50%.

#### There is no hardware sync — alignment must be inferred

Across the whole dataset the only Bpod event channels are `Port<N>In/Out` and `Tup`. No BNC/TTL from
the miniscope exists. Bpod filename timestamps are only good to ±30 s for 56% of sessions (±300 s for
91%), so **pairing a log to a video by filename alone is not safe**.

The alignment substrate is `timestamps<TIMESTAMP>.csv`, which is one row per frame of absolute
wall-clock at sub-ms precision (`2026-06-30T15:05:24.0566784-07:00`), with row count exactly equal to
the frame count of the Miniscope, Home *and* Linear videos. **Never convert frames to seconds with
`frame / 25`** — real capture rate is ~24.8–24.9 fps, and for concatenated young-cohort sessions the
CSV also preserves the real (sometimes hour-long) gaps between sub-recordings that the spliced video
does not. See the anomalies doc §6.

#### Feasibility is confirmed

On `Mouse944 / Round1 / Linear / 2026-07-13T15_55_27`, a rolling-median-detrended high-percentile
intensity trace in a ROI at each track end, cross-correlated against the Bpod reward train split by
`TrialTypes`, peaks at **z = 15.1** (left port ↔ trial type 1) and **z = 14.9** (right ↔ type 2) at
the *same* residual lag of +0.9 s, versus z = 6.4 / 3.9 for the swapped pairings. That single test
recovers the port→side mapping, the clock offset, and per-event detectability (median transient 106
and 36 grey levels above baseline) at once. The +0.9 s residual is the Bpod-PC-vs-acquisition-PC
clock offset and should be fitted per session, not assumed.

---

### 10. Analysis

After generating the trialwise binned location files, we move from preprocessing to the analysis side — this lives in the separate **`stability-analysis`** repo (sibling to `stability-preprocessing`) /home/toor/Desktop/stability-analysis. The environment remains the same. 

1. **Generate the megadata table:** `Linear_spatial_binned_activity_multisession_raw_trials.ipynb` (in `stability-analysis`). *Despite being named "Linear", it works for both Linear and TMaze sessions* (set via the `task = 'Linear'` / `'TMaze'` config variable in the notebook).

   For a mouse/round, it joins the per-session registration output (`assignments.npy` from "Register across sessions" above) with each session's binned behavior CSV (`*_linearized_binned_trials_*nbins.csv`, requires `frame_local`, `spatial_bin`, `trial_type`, `trial_id`) and that session's `caiman_results.hdf5`, and builds one long-format table with one row per `(session, global_neuron_id, trial_type, trial_id, spatial_bin)`, storing the raw (un-averaged) per-frame activity arrays for that bin: `frames`, `F_dff`, `denoised_z` (z-scored trace), `S_cnmf` (deconvolved), `S_oasis` (OASIS spike estimate).

   Each session is written to a temporary per-session Parquet file, then streamed together into one output file via `pyarrow.parquet.ParquetWriter` (to avoid holding the full concatenated table in memory). **Always save results in Parquet**, not CSV

2. **Reference/tutorial notebooks** (not run as a per-session pipeline step — kept for reference on methodology):
   - `scripts/normalize_1p_traces.ipynb` (in `stability-preprocessing`) — walks through why CaImAn's `cnm.estimates.F_dff`/`detrend_df_f()` is only *detrended*, not truly normalized, and how to fix that: feed a raw trace into CaImAn's `GetSn()` (noise estimate from the power spectral density) and use it to z-score the trace, versus the cruder alternative of normalizing by the trace's plain standard deviation (which overestimates noise because it includes real signal fluctuations). Ends with a "Make a ridge plot" section built on top of these normalized traces. This is the origin/tutorial for the `zscore_trace()` function actually used downstream (see below) — it is not itself imported anywhere.
   - `scripts/demos/visualize_ridge_plot.ipynb` — this is a stock CaImAn library demo notebook, not project-specific (see "Important Notes for New Collaborators" #6 — do not modify). Shows how to build a ridge plot (each trace plotted with a small y-offset from its neighbor, for a quick big-picture view of a whole session's traces at once) from a demo `cnmf` object, plus utility functions for highlighting a stimulus window or animating a scrolling ridge plot.

**Where `zscore_trace()` is actually used:** there is no shared `.py` module or importable script for it — the identical `zscore_trace()` function body (from `normalize_1p_traces.ipynb` above, "Adapted from code by Zach Barry") is copy-pasted directly into each `stability-analysis` notebook that builds a spatially-binned megadata table, both Linear and TMaze, single-session and multisession variants:
- `Linear_spatial_binned_activity_multisession_raw_trials.ipynb` (documented above)
- `TMaze_spatial_binned_activity_multisession_raw_trials.ipynb`
- `TMaze_spatial_binned_activity_multisession.ipynb`
- `TMaze_spatial_binned_activity.ipynb`

Each of these calls it to produce that notebook's `denoised_z` column. Since it's duplicated rather than shared, if the normalization logic ever needs to change, it has to be updated in all of these places by hand.

---

## Key Variables & Shapes

All CaImAn outputs are loaded via:
```python
cnmf_fit = cnmf.load_CNMF('caiman_results.hdf5', n_processes=1, dview=None)
```

| Variable | Shape | Description |
|----------|-------|-------------|
| `cnmf_fit.estimates.A` | `(n_pixels, n_neurons)` sparse | Spatial components; `n_pixels = height × width` |
| `cnmf_fit.estimates.C` | `(n_neurons, n_frames)` | Raw temporal traces |
| `cnmf_fit.estimates.S` | `(n_neurons, n_frames)` | Deconvolved spike estimates |
| `cnmf_fit.estimates.F_dff` | `(n_neurons, n_frames)` | ΔF/F (quantile min baseline, 250-frame window) |
| `cnmf_fit.estimates.b` | `(n_pixels, n_bg)` | Background spatial |
| `cnmf_fit.estimates.f` | `(n_bg, n_frames)` | Background temporal |
| `cnmf_fit.estimates.Cn` | `(height, width)` | Correlation image |
| `cnmf_fit.estimates.idx_components` | `(n_accepted,)` | Indices of accepted neurons |
| `cnmf_fit.estimates.idx_components_bad` | `(n_rejected,)` | Indices of rejected neurons |
| `cnmf_fit.estimates.r_values` | `(n_neurons,)` | Spatial correlation score (0–1) |
| `cnmf_fit.estimates.SNR_comp` | `(n_neurons,)` | Signal-to-noise ratio |
| `cnmf_fit.estimates.cnn_preds` | `(n_neurons,)` | CNN classifier score (0–1) |
| `cnmf_fit.estimates.coordinates` | list of dicts | Contour coords: `coords[i]['coordinates']` → `(N, 2)` |
| `cnmf_fit.estimates.dims` | `(height, width)` | FOV dimensions |

**DLC outputs** (`*_filtered.csv`): MultiIndex columns with levels `(scorer, bodypart, coordinate)`. Coordinates are `'x'`, `'y'`, `'likelihood'`. Load with `pd.read_csv(..., header=[0,1,2], index_col=0)`.

---

## Environment Sorting Parameters (`sort_recordings_by_environment.py`)

Each mouse requires individually tuned values for:

| Parameter | Description |
|-----------|-------------|
| `BRIGHTNESS_THRESHOLD` | Mean pixel intensity cutoff to distinguish arena types |
| `ASPECT_RATIO_CUTOFF` | Frame aspect ratio cutoff to distinguish arena types |

These are set per-mouse inside the script. When adding a new mouse, add its entry to the per-mouse parameter dict and verify classification visually before continuing the pipeline.

---

## CaImAn Parameters

### All mice (as of 2026-08-23) (`preproc_caiman_final.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `fr` | 25 | fps |
| `decay_time` | 0.56 s | calcium transient duration |
| `dxy` | [0.83, 0.83] μm/px | spatial resolution |
| `gSig_filt` | see below — cohort-dependent | Gaussian filter for motion correction |
| `strides` | [64, 64] | MC patch size |
| `overlaps` | [32, 32] | MC patch overlap |
| `max_shifts` | [25, 25] | max allowed pixel shift |
| `gSig` | see below — cohort-dependent | CNMF-E spatial kernel |
| `min_corr` | 0.8 | initialization correlation threshold |
| `min_pnr` | 6.5 | initialization peak-to-noise threshold |
| `merge_thr` | 0.65 | component merge threshold |
| `min_SNR` | 2.5 | evaluation SNR threshold |
| `rval_thr` | 0.75 | evaluation spatial correlation threshold |
| `min_cnn_thr` | 0.9 | CNN upper threshold |
| `cnn_lowest` | 0.4 | CNN lower threshold |

### Cohort-dependent `gSig`/`gSig_filt` (as of 2026-08-23, superseding the 2026-07-09 "same for all mice" note)

A parameter sweep across all 4 aging-cohort mice on 2026-08-23 found that bigger `gSig`/`gSig_filt` consistently outperformed the young-mouse defaults for the aging cohort specifically (`gSig=7`/`gSig_filt=[10,10]` beat both `gSig=5`/`gSig_filt=[8,8]` and `gSig=9+`, the latter overshooting). `preproc_caiman_final.py` now resolves these automatically based on which mouse ID appears in the input path (same `OLD_MICE_IDS` matching logic as the `CROPS` dict below) unless overridden via `--gSig`/`--gSig_filt`:

| Cohort | `gSig` | `gSig_filt` |
|--------|--------|-------------|
| Aging (`OLD_MICE_IDS`) | `[7, 7]` | `[10, 10]` |
| Young (default) | `[5, 5]` | `[8, 8]` |

The script prints the resolved values at startup (`Resolved gSig=... gSig_filt=... (aging cohort: True/False)`) — check this line rather than assuming, especially when running on a new/renamed path.

### Per-mouse FOV crop (CROPS dict, keyed by numeric mouse ID suffix)

```python
CROPS = {
    "946": (70, -200, 70, -70),    # (row_start, row_stop, col_start, col_stop)
    "847": (90, -150, 150, -20),
    "945": (20, -180, 150, -1),
    "943": (120, -250, 150, -1),
    "944": (200, 550, 100, 500),
    "163": (0, -50, 70, -50),
    "1636": (100, -1, 80, -20),
    "1637": (30, -1, 130, -10),
    "1639": (10, -10, 70, -20),
}
```

Lookup matches `Mouse<id>` as a whole ID via regex (`Mouse{id}` not followed by another digit), not a plain substring — `"163"` is a numeric prefix of `"1636"`/`"1637"`/`"1639"`, so substring matching would be ambiguous between them.

Applied after motion correction to remove border artifacts. Must be extended for new animals.

**Deciding/checking a crop (`sample_frames_for_crop.py`):** samples N evenly-spaced frames (default 1000) from a `Miniscope*.avi` and saves a mean/max/std-projection PNG — the max/std projections show active cell locations across the recording, which is far more informative for picking crop bounds than a single frame. Use it both when tuning a brand-new animal's crop and when spot-checking whether an existing animal's crop is still valid for a new round of recordings (e.g. camera position drift).

```bash
# (stability-preprocessing env)
python sample_frames_for_crop.py /path/to/Miniscope<TIMESTAMP>.avi

# With an existing/proposed crop overlaid as a red rectangle for visual review
python sample_frames_for_crop.py /path/to/Miniscope<TIMESTAMP>.avi \
    --crop 200 550 100 500 \
    --output results/mouseXXX_crop_check.png
```

`--crop` takes the same `(row_start, row_stop, col_start, col_stop)` convention as the `CROPS` dict (negative stop = from the end). Save review images to `results/` so they persist for visual inspection.

---

## Behavior Analysis Parameters

### Linear track (`Linear_combined_pipeline.ipynb`)

```python
FPS = 25
LIKELIHOOD_THRESHOLD = 0.9
MAX_JUMP = 5.0          # pixels/frame — filters tracking errors
SMOOTHING_SIGMA = 1.0   # Gaussian smoothing (samples)
SPEED_THRESHOLD = 2.0   # pixels/frame — minimum speed for valid traversal
MIN_TRIAL_DURATION_S = 0.5
N_BINS = 70             # spatial bins along track
PORT_ROIS_DEFAULT = {
    "left":  (250, 665, 60),   # (x, y, radius)
    "right": (1740, 610, 60),
}
```

### DLC trim thresholds (`trim_videos_from_dlc.py`)

- **T-maze start:** first frame where `y > 45` px and `likelihood > 0.9` for 10 consecutive frames
- **Linear start:** first frame where `y < 715` px and `likelihood > 0.9`
- **Trim duration:** 960 seconds (16 minutes) from start frame

---

## Code Conventions

### Plotting

- **Library:** `matplotlib` for static/saved figures, `bokeh`/`holoviews` for interactive notebook views, `plotly` for 3D threshold exploration
- **Accepted components:** lime green; **Rejected:** red; **Gained across sessions:** cyan; **Lost across sessions:** yellow
- **Correlation image colormap:** `'gray'`
- **Typical figure sizes:** contour overlays `(8, 8)`, trace panels `(7, 2)`, animated figures `(10, 7)` with `gridspec`

### Saving figures

Notebooks save figures to local subdirectories (e.g., `figs_caiman_report_no_memmap/`). No global convention — check each notebook's save cell.

### Video processing

Uses `ffmpeg` via `subprocess` with concat demuxer (list-file method). Never re-encodes — always stream copy (`-c copy`) to avoid quality loss. Short videos (< 10 s) must be removed before concatenation or ffmpeg will error.

### CaImAn data loading

```python
import caiman as cm
from caiman.source_extraction import cnmf

# Load results
cnmf_fit = cnmf.load_CNMF('caiman_results.hdf5', n_processes=1, dview=None)

# Load a video (subindices for memory management)
movie = cm.load(video_path, subindices=slice(2000))
```

### Memmap files

Stored in `/tmp/caiman_data/` during processing. Pass `--delete_memmaps` flag to `preproc_caiman_final.py` to clean up automatically. Do not delete manually while CaImAn is running.

---

## Important Notes for New Collaborators

1. **Never modify or overwrite raw data in place.** Any script or operation that touches a raw `.avi`/`.csv` file (concatenation, trimming, renaming, etc.) must write its output to a new file/copy — never delete or replace the original raw file. All such manipulations must be reviewed by the project owner before the originals are considered safe to clean up.

2. **Clear notebook outputs before committing.** Notebook cell outputs contain large images that will bloat the repo. The README explicitly warns about this.

3. **Per-mouse parameters must be manually set.** There is no config file system — CaImAn parameters are *for now* the same for all animals/cohorts but FOV crops are tuned per animal. When adding a new mouse, update the `CROPS` dict in `preproc_caiman_final.py` and document its parameters.

4. **Two separate conda environments.** `stability-preprocessing` for everything except pose estimation. `dlc` environment only for `estimate_pose.py`. If `estimate_pose.py` fails to run after activating `dlc`, try `unset LD_LIBRARY_PATH` (library conflict). Do not mix environments.


5. **`scripts/archive/` and `archive/` contain old but potentially reference-worthy code.** Do not delete. `scripts/demos/` contains demo notebooks from the CaImAn library — do not modify.

6. **Video file naming is load-bearing.** The timestamp in the filename is used to group files by date/session. Renaming files will break concatenation and curation scripts.

7. **The `timestamps*.csv` files have no header row** — every row is data. DLC `*_filtered.csv` files have a 3-row MultiIndex header. The concat script auto-detects this via regex on the first row.

8. **Never assume a same-day batch of recordings shares one arena.** See the warning in step 2 — this has caused real data corruption via `concat_session_videos.py` on Mouse944/Mouse945-Round2 data (2026-08). Classify every individual sub-recording's arena before grouping for concatenation; don't rely on the filename or a single frame from an already-merged output.

9. **A non-zero exit from `preproc_caiman_final.py` doesn't always mean the run produced nothing.** See step 5.6's note on `cnn_preds`/HDF5 save errors — check for a valid `caiman_final/caiman_results.hdf5` before discarding a "failed" run.

10. **`timestamps<TIMESTAMP>.csv` is the only absolute clock in the dataset — treat it as primary data, not a byproduct.** One row per frame, wall-clock to sub-ms, row count exactly equal to the frame count of all three videos, and it stays correct across concatenation (real inter-recording gaps preserved). Real capture rate is ~24.8–24.9 fps, so anything that computes seconds as `frame / 25` (including `dlc_trim_info.csv:start_time_seconds`) drifts by seconds over a 960 s clip. Only 60 of 646 session folders currently keep their own copy, and the same filename timestamp can also match a short aborted-recording stub — resolve from `RAW_DATA/<TIMESTAMP>/` and assert `n_rows == n_frames` before using it. See step 9.5 and [docs/bpod_reward_log_anomalies.md](docs/bpod_reward_log_anomalies.md) §6.

11. **Bpod protocol folders do not identify the protocol.** `Data/Mouse<ID>/TMaze4/Session Data/` contains `TMaze2`-named sessions (all byte-identical duplicates of the sibling `TMaze2` folder) plus nested `Round2`/`Round3`/`Round1.5` subfolders that a flat glob misses. Glob recursively, read the protocol from the filename, and de-duplicate on `(mouse, protocol, timestamp)`. TMaze2 and TMaze4 differ in port numbering and LED cueing, so this matters for reward decoding. See step 9.5.

---

## Steps Only Sometimes Needed

These are not part of the standard pipeline run for every session — only run them when the described condition applies.

- **Remove short videos (video QC)** (`remove_short_videos.py`) — only needed if a session contains recordings under 10 s (e.g. an aborted/restarted recording). Run before concatenation, since `ffmpeg` will error on same-day concatenation if a short video is still present.

  ```bash
  # (base) env
  python remove_short_videos.py <session_dir>
  ```

  Deletes videos < 10 s and their paired timestamps CSV.

---

## Pending DLC Re-processing

Linear-track sessions whose `dlc/` folder does not contain a `160000` snapshot (see "Run DeepLabCut pose estimation" above). These need to be re-run with the `160000` snapshot before their DLC output can be trusted.

Generated by `scripts/data_audit/build_headcount.py` — see the `dlc_sessions_missing_160000`
column of `docs/headcount_<Mouse>.csv` for the authoritative list. **48 recordings** lack a
`160000` snapshot.

**This applies to the `linear` DLC project only.** That project has two snapshots on disk —
`160000` (preferred, 1104 files) and `100000` (stale, 350 files). The `T-Maze` project only ever
had `1100000`, so a T-maze session lacking `160000` is **not** a defect and must not be listed
here. Tracked per recording, not per day: an aging-cohort day holds a Linear and a TMaze session.

- **Mouse163 / Round1** — 4 sessions: `2024-05-23T18_44_50`, `2024-05-29T16_08_20`, `2024-06-05T09_57_40`, `2024-06-05T10_11_50`
- **Mouse163 / Round2** — 2 sessions: `2024-07-18T17_19_51`, `2024-07-19T14_10_38`
- **Mouse1636 / Round1.5** — 1 sessions: `2024-06-10T16_43_15`
- **Mouse1637 / Round3** — 26 sessions: `2024-10-02T13_38_21`, `2024-10-03T14_16_18`, `2024-10-04T13_17_47`, `2024-10-05T15_02_39`, `2024-10-06T12_48_27`, `2024-10-07T14_02_49`, `2024-10-08T13_45_20`, `2024-10-09T14_12_40`, `2024-10-10T12_58_46`, `2024-10-11T15_17_23`, `2024-10-12T14_58_08`, `2024-10-16T13_38_56`, `2024-10-17T15_02_01`, `2024-10-18T13_46_24`, `2024-10-19T16_09_05`, `2024-10-20T14_33_04`, `2024-10-21T15_00_44`, `2024-10-22T12_51_41`, `2024-10-23T13_53_18`, `2024-10-24T14_09_39`, `2024-10-25T13_40_17`, `2024-10-26T13_14_59`, `2024-10-27T15_22_15`, `2024-10-28T14_54_19`, `2024-10-29T15_19_08`, `2024-10-30T14_36_05`
- **Mouse1637 / Round3.5** — 14 sessions: `2024-10-31T13_40_30`, `2024-11-01T13_08_11`, `2024-11-02T13_06_59`, `2024-11-03T13_10_29`, `2024-11-04T14_19_30`, `2024-11-05T14_39_19`, `2024-11-06T14_59_26`, `2024-11-07T14_12_14`, `2024-11-08T14_14_47`, `2024-11-09T14_30_28`, `2024-11-10T14_57_58`, `2024-11-11T14_09_51`, `2024-11-12T11_52_38`, `2024-11-13T13_14_55`
- **Mouse1639 / Round1** — 1 sessions: `2024-06-09T13_21_01`
