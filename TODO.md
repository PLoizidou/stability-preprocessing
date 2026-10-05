# TODO — pending re-runs

Running list of things that need to be re-run, so they can be batched together instead of
re-running the same expensive step once per discovery. Add new items as they come up; move
finished ones to **Done** at the bottom with the date.

When adding an item, say **what** needs re-running, **why** (what changed underneath it), and
**what else is downstream of it** — that's what makes batching possible.

---

## Open

### 1. Multisession registration — Mouse945-Round2 / Round1.5 / **both tasks**, all scope types

**Re-run:** `scripts/multisession_registration.ipynb` for **every scope that touches Round1.5**,
at **both** `thresh_cost = 0.7` and `0.5`:

| Type | `rounds` / `tasks` | scope tag | output dir |
|---|---|---|---|
| 1 | `[1.5]` / `["Linear"]` | `Round1.5` | `<mouse_dir>/Round1.5/Linear/caiman_final_Registration_Round1.5/` |
| 1 | `[1.5]` / `["TMaze"]` | `Round1.5` | `<mouse_dir>/Round1.5/TMaze/caiman_final_Registration_Round1.5/` |
| 2 | `[1.5]` / `["Linear","TMaze"]` | `Round1.5_CrossTask` | `<mouse_dir>/Coregistration/caiman_final_Registration_Round1.5_CrossTask/` |
| 3 | `[1, 1.5]` / `["Linear"]` | `Linear_CrossRound_Round1-Round1.5` | `<mouse_dir>/Coregistration/caiman_final_Registration_Linear_CrossRound_Round1-Round1.5/` |
| 4 | `[1, 1.5]` / `["Linear","TMaze"]` | `AllSessions` | `<mouse_dir>/Coregistration/caiman_final_Registration_AllSessions/` |

`<mouse_dir>` = `/media/toor/Seagate3/Mouse945 - Round2`. Config: `mouse_id = "945 - Round2"`
(a *string*, so the notebook's `f"Mouse{mouse_id}"` resolves the non-standard folder name),
`hard_disk = "Seagate3"`, `caiman_v = "caiman_final"`.

Round1 has no TMaze, so there is no cross-round TMaze scope.

**Why:** two Round1.5 sessions were found unusable on 2026-09-25 and dropped, changing the
session set every Round1.5 registration is built from:

- `Round1.5/Linear/2026-07-16T15_04_55` — CaImAn accepted **0 components** (every component
  rejected by the evaluation step). This made the Type-1 Linear registration crash outright with
  `ValueError: need at least one array to stack` (empty footprint matrix breaks the optical-flow
  remap in `register_ROIs_with_costs`). Session has been moved out of the directory: Linear
  **16 → 15 sessions**.
- `Round1.5/TMaze/2026-07-16T14_32_41` — **behaviourally unusable**: the mouse is in the home cage
  for the vast majority of the session rather than on the T-maze. (Separately, the miniscope also
  died at frame 10,858 / t ≈ 434 s; the video was trimmed to 7 min on 2026-09-25 before the
  behavioural problem was found, and that CaImAn re-run was cancelled — see the session's own
  `README.md`.) TMaze **18 → 17 sessions** once excluded.

**State of existing registrations for this mouse:**
- `Round1/Linear` Type 1 — ran fine 2026-09-23, **still valid** (neither dropped session is in
  Round1). Do not need to redo it on its own, but it *is* an input to the Type 3 and Type 4 scopes
  above, which do need redoing.
- `Round1.5/TMaze` Type 1 — ran fine 2026-09-23 but is **now stale**: it includes the unusable
  session.
- `Round1.5/Linear` Type 1 — **never succeeded** (the crash above).
- Types 2, 3, 4 — **never run at all** for this mouse.

**Decide first: how to exclude the two sessions.** `session_exclusions.csv` currently has **zero
rows for mouse 945**, and the notebook matches exclusions on `str(mouse_id)` — which here is
`"945 - Round2"`, so a plain `945` row would *not* match anyway. Right now the Linear session is
excluded only by having been physically moved out of the folder, and the TMaze session is **still
physically present** and would be picked up by a re-run as-is. Either move the TMaze session out
the same way, or add both to `session_exclusions.csv` under a mouse value that matches what the
config actually passes. Moving-out-only leaves no durable record of *why*, so the CSV is preferable.

**Downstream of this:**
- `scripts/group_coregistration_quality_comparison.ipynb` — consumes Type-1
  `confidence_metrics_Round*.pkl`; this mouse's entries are stale/missing.
- `stability-analysis` → `Linear_spatial_binned_activity_multisession_raw_trials.ipynb` (the
  task-agnostic megadata notebook, run with `task='Linear'` / `'TMaze'`) — Mouse945-Round2 has
  **no megadata generated yet at all**, because it had no registration when the young-cohort
  megadata batch ran. Generate it for Round1.5 Linear + TMaze (both thresholds) *after* the
  registrations land, not before.
- Trial-segmentation (step 9) outputs for the two dropped sessions are moot — they should not feed
  any downstream table.
- Anything already computed off a Mouse945-Round2 megadata table (PV correlation / participation
  variability notebooks) — nothing known to exist yet, but re-check before reusing any such output.

---

### 2. Regenerate the `stability-analysis` megadata tables — after the trim-offset fix

**Re-run:** in `stability-analysis`, the task-agnostic megadata notebook
`Linear_spatial_binned_activity_multisession_raw_trials.ipynb` (with `task='Linear'` and
`task='TMaze'`), plus `TMaze_spatial_binned_activity_multisession_raw_trials.ipynb`. Output is
Parquet, per CLAUDE.md step 10.

**Why:** every megadata table joins a session's binned behaviour CSV to its neural traces by frame
index. Those binned CSVs were built against the WRONG alignment — `dlc_trim_info.csv` recorded the
frame the trim was *requested* at, but `ffmpeg -ss ... -c copy` backs off to an earlier keyframe, so
each clip actually begins k frames earlier (k measured per session, 0-57 frames ≈ 0-2.3 s). Pose was
therefore paired with neural frames k earlier than it should have been — behaviour running AHEAD of
neural data, the same direction in every session, so it does not average out across trials. At ~20
cm/s a typical k is roughly one 3 cm spatial bin; the worst cases are several bins.

`scripts/data_audit/measure_trim_offset.py` records the true start as `start_frame_actual` in every
`dlc_trim_info.csv`, and both pipeline notebooks now read it (falling back to `start_frame` when a
session has not been measured). Segmentation has been re-run on that basis, so the binned CSVs are
now correct — the megadata tables built from the OLD ones are the remaining stale layer.

**Order:** segmentation (done) → megadata (this item) → anything built on megadata.

**Not affected, do NOT redo:** DLC, trimming, CaImAn, and all multisession registrations —
registration reads only `A` and `Cn` from the CaImAn files and never touches behaviour or
`start_frame`.

**Also still to do, separately:** `trim_videos_from_dlc.py` still *writes* only the requested
`start_frame`, so newly trimmed sessions will need `measure_trim_offset.py` run on them too. Better
fix is to have the trimmer record the true start itself at trim time.

**Reward code is affected and not yet wired in** (CLAUDE.md step 9.5): `detect_reward_flushes.py`,
`assign_rewards_to_trials.py` and `review_rewards.py` all compute
`frame_local = frame_raw - start_frame`. This matters more there than anywhere else — a reward valve
window is only 2-3 frames at 25 fps, i.e. smaller than the offset being corrected. Point them at
`start_frame_actual` before any reward extraction is trusted.

---

### 3. Mouse1639 — two incomplete coregistrations

Audit 2026-09-29. All **98** canonical sessions under `/media/toor/SeagateClean/Mouse1639`
(Round1/Linear 16, Round1.5/Linear 17, Round1.5/TMaze 17, Round3/Linear 24, Round3/TMaze 24) are
complete: `caiman_aging`, trial CSVs, binned CSVs, and `160000` DLC on every Linear session. No
misfiling — zero session-timestamp overlap with the other three aging mice, and every canonical
session is backed by Mouse1639 raw. Type-1 and most Type-2/3 scopes are current (built 2026-09-08/09)
and their counts match the session counts exactly.

**Two scopes are broken, both silently — the folder exists so nothing flags them as missing:**

| Scope | State |
|---|---|
| `Coregistration/caiman_aging_Registration_AllSessions` | **completely empty** (0 files). Type-4 pooled scope never produced. Should be 98 sessions. |
| `Coregistration/caiman_aging_Registration_Linear_CrossRound_Round1-Round1.5-Round3` | default run fine (57 sessions), but its `stricter_coregistration_thresh_cost_0_5/` exists and is **empty** — strict rerun never completed. Every other scope has 16-17 files + 3 pkl there. |

**Re-run:** `scripts/multisession_registration.ipynb`, `mouse_id = 1639`,
`hard_disk = "SeagateClean"`, `caiman_v = "caiman_aging"`, for
`rounds=[1, 1.5, 3], tasks=["Linear","TMaze"]` (AllSessions) and
`rounds=[1, 1.5, 3], tasks=["Linear"]` (the cross-round strict half). One execution per scope covers
both thresholds.

**Do this only after item 4 below is settled** — anything processed from item 4 changes the session
set these two scopes are built from, and re-registering twice is wasted.

---

### 4. Mouse1639 — 11 usable recordings never processed (4 more are blank, do not chase)

Audit 2026-09-29. Compared every raw miniscope recording on
`SeagatePortableDrive/AGING_MICE/Mouse1639` (138) against the 98 canonical sessions, then measured
duration, pixel statistics and FOV correlation on every unmatched one. 19 are short aborted stubs
(<300 s) and are correctly ignored. Of the rest:

**Blank — miniscope recorded nothing, correctly skipped. Recorded here so nobody re-investigates:**

| Recording | Duration | Pixels |
|---|---|---|
| `Round1/2024-06-08T17_23_02` | 1545 s | uniform grey, mean 55, **std 0** |
| `Round1/2024-06-09T13_21_01` | 1602 s | all black, mean 0, **std 0** |
| `Round1.5/2024-06-10T14_19_09` | 1567 s | all black, mean 0, **std 0** |
| `Round1.5/2024-06-11T12_38_18` | 1536 s | all black, mean 0, **std 0** |

Note `2024-06-09T13_21_01` is in CLAUDE.md's "Pending DLC Re-processing" list — it is blank, so it
can never be re-processed and should be dropped from that list rather than re-run.

**Real, usable, and unprocessed** (all FOV-verified as Mouse1639: 0.92-0.99 vs 0.72-0.81 against
other aging mice). Arena taken from the DLC model already run on each, since a single video frame
cannot classify it — **both the linear track and the T-maze are in the room at once**, visible in
every frame, so the usual "sample a frame and look" rule from CLAUDE.md step 2 does not work for
this mouse's room:

| Round | Recording | Duration | Fills gap | DLC model present |
|---|---|---|---|---|
| Round3.5 | `2024-10-31T13_12_23` | 1573 s | whole round | `linear...160000` |
| Round3.5 | `2024-11-01T12_39_19` | 1609 s | whole round | `linear...160000` |
| Round3.5 | `2024-11-02T13_35_58` | 451 s | whole round | `linear...160000` |
| Round3 | `2024-10-06T13_17_01` | 508 s | TMaze on 10-06 | `T-Maze...1100000` |
| Round3 | `2024-10-12T15_23_44` | 1663 s | TMaze on 10-12 | `T-Maze...1100000` |
| Round3 | `2024-10-12T15_53_22` | 774 s | Linear on 10-12 | **none — needs DLC** |
| Round3 | `2024-10-26T13_41_52` | 1384 s | TMaze on 10-26 | `T-Maze...1100000` |
| Round3 | `2024-10-30T15_31_42` | 1376 s | Linear on 10-30 | **none — needs DLC** |
| Round1.5 | `2024-06-12T16_38_30` | 1596 s | 06-12 (no coverage at all) | `linear...` |
| Round1 | `2024-05-21T17_25_56` | 917 s | 05-21 (no coverage) | — |
| Round1 | `2024-05-22T18_43_07` | 718 s | 05-22 (no coverage) | — |

**Round3.5 is the biggest item: the entire round is unprocessed for this mouse** (3 raw sessions,
DLC already done with the correct `160000` snapshot, nothing else). Mouse1637's Round3.5 has 14
sessions fully processed, so this is a real structural gap, not a round that was never run.
CLAUDE.md confirms Round3.5 is Linear-only, matching the DLC model on all three.

`2024-05-21T17_25_56` is also the recording sitting in `Seagate2/Mouse1639/Baseline/` — decide
whether it is a Baseline recording (excluded by design) or a Round1 session before processing it.

**Round2** (`2024-07-18T14_13_22` 1986 s, `2024-07-18T14_48_39` 1369 s, the latter with T-Maze DLC)
is also unprocessed, but Round2 is not one of the analysis rounds in CLAUDE.md for the aging cohort
and no mouse has a processed Round2 — treat as deliberately out of scope unless you say otherwise.

**Downstream:** anything processed here feeds item 3's registrations, then the `stability-analysis`
megadata tables.

---

### 5. Mouse1639 — stale duplicate tree on Seagate2, and a legacy registration folder

- `Seagate2/Mouse1639` holds a second copy of the same session timestamps, but with `caiman_final`
  (not `caiman_aging`) and **no trial-segmentation outputs**. The canonical tree the registrations
  actually point at is `SeagateClean/Mouse1639` (confirmed from `scope_info_*.json`:
  `hard_disk: SeagateClean`). The Seagate2 copy is stale — worth deleting or clearly marking so it
  stops being mistaken for current, since it is the copy that `measure_trim_offset.py` landed on.
- `Round3/TMaze/caiman_Registration_Round3_RERUN BECAUSE PROBLEMATIC SESSIONS IN` (Nov 2025) — the
  folder name is truncated mid-sentence and states a problem without saying which sessions. It holds
  an old `caiman_`-era registration plus a `TMaze_spatial_binned_activity_multisession.parquet`.
  Either record what the problematic sessions were or delete it; as-is it is a warning nobody can act on.

---

## Done

### Done 1 — Mouse1636 / Round1.5 / TMaze / `2024-06-27T14_09_07` (2026-09-23)
Was missing everything downstream of DLC. Ran `caiman_aging` (417 accepted / 776 rejected,
`gSig=[7,7]`, crop `(100,-1,80,-20)`, 24 000 frames) + TMaze trial segmentation (53 trials,
11 680 binned rows, 0.4 % unbinned). Also deleted 5 stale Nov-2025 pipeline CSVs that disagreed
with the new outputs (59 vs 53 trials). Ownership had to be established first: the session was
filed under Mouse1639 on two drives; FOV correlation gave 0.9996 vs Mouse1636, 0.80 vs Mouse1639.

### Done 2 — Mouse1636 / Round1.5 / TMaze / `2024-06-10T16_43_15` (2026-09-28)
Old clip started 95 s too early. Re-trimmed **from the original raw** at 375.0 s (old
`start_frame` 7000 = 280 s, + 95 s) for 960 s → 24 003 frames, both streams in lockstep, into a
**new** folder `2024-06-10T16_43_15/`. Originals untouched and verified; the bad
`... - TMaze DLC problems/` folder left in place for you to delete. DLC did **not** need re-running
— it was run on the full 38 014-frame raw video, so it still covers the new window
(9375 + 24003 = 33 378 ≤ 38 014). CaImAn + segmentation followed.

### Done 3 — `session_exclusions.csv` gains a `status` column (2026-09-28)
`1636 / Linear / 1.5 / 2024-06-19T13_09_38` was excluded on 2026-08-27 with *"have not checked but
PV corr is very off"*. Checked — nothing wrong. Rather than delete the row, the CSV now has a
`status` column (`exclude` | `caution`); that row is `caution`, so it is kept as a note but **not**
excluded. All 15 other rows are `exclude`, behaviour unchanged.

**Four** separate copies of the exclusion logic had to be patched — each treated every row as a
hard exclusion, so the un-exclusion would silently not have propagated:
`stability-preprocessing/scripts/multisession_registration.ipynb`, and in `stability-analysis`
`Linear_spatial_binned_activity_multisession_raw_trials.ipynb`,
`Within_Day_Participation_Variability.ipynb`, and `PVcorr_GlobalSim_AllCells crossRound.ipynb`
(this last one has the filter inlined rather than in a `load_session_exclusions()` function, which
is why it is easy to miss). All four default a missing `status` column to `exclude`, so older
copies of the CSV behave exactly as before.

The 11 notebooks under `stability-analysis/PV_correlation_results/notebooks_*/` also contain the
old logic but were **deliberately left alone**: they are papermill *output records* of past runs
(`run_pvcorr_crossround_all.py` / `run_participation_aging.py` execute the source notebooks above
into that folder). Editing them would falsify the record of what those runs actually did. They
will pick up the new behaviour when regenerated.

### Done 4 — Mouse1636 fully preprocessed + all registrations current (2026-09-28)
`2024-06-29T17_56_53` (Linear) finished: DLC re-run at snapshot `160000` (better on every bodypart,
e.g. bodypart2 frac>0.9 0.647 → 0.889; detected start unchanged at frame 9480, so the existing trim
was already correct and was kept), `caiman_aging` **367 accepted** (vs 98 under the old young-param
`caiman_final`, which is likely why it had been dropped), Linear segmentation 45 trials. Its raw had
been misfiled under Mouse163 and was moved, md5-verified, as was its same-day pair
`2024-06-29T18_18_00`; ownership settled by miniscope FOV correlation (0.997 vs Mouse1636, 0.65 vs
Mouse163). Mouse1636 ended at **44/44 sessions, zero gaps**.

All five affected registration scopes then re-run at both thresholds, counts matching predictions
exactly: Round1.5/Linear 12→**14**, Round1.5/TMaze 11→**13**, CrossTask 23→**27**, Linear CrossRound
29→**31**, AllSessions 40→**44**. Mean IoU 0.67-0.68 across scopes. `Round1/Linear` deliberately NOT
re-run — unchanged sessions and exclusions, so its existing registration is still valid.

### Done 5 — trim keyframe-offset measured and corrected (2026-09-28)
See open item 2 for the mechanism. `scripts/data_audit/measure_trim_offset.py` added; it recovers
each clip's true first raw frame by exact pixel match (the trim was a stream copy, so kept frames are
bit-identical to the raw) and writes `start_frame_actual` + `keyframe_offset_frames` into
`dlc_trim_info.csv` without touching any video. Validated against frame-count arithmetic
(`raw_frames - true_start == trimmed_frames`), which shares no logic with the pixel matcher: 3/3
agreement on constructed k=0/2/8 cases, match unique in a 300-frame window, idempotent, videos
md5-unchanged, and it refuses to guess (`no_exact_match`) on a re-encoded clip rather than returning
a nearest match. Both pipeline notebooks now read `start_frame_actual`, and segmentation was re-run
across all sessions.

### Not needed — the `160000` pending-DLC list was already correct
`build_headcount.py` already restricts the `160000` rule to linear-model DLC files
(`DLC_resnet50_linear`, line ~527, with the reasoning in the comment at ~371-377), the generated
`docs/headcount_*.csv` are correct, and CLAUDE.md's "Pending DLC Re-processing" section already
reflects it: **48** recordings, one of them Mouse1636's. No fix required — an earlier note in this
file describing a 287-recording list full of T-maze false positives was reading a stale copy.
