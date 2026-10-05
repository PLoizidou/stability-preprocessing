# Head-count CSVs

Per-mouse recording inventory with gaps flagged, generated from a full file-level scan of every
mounted data drive. **Read-only audit** — nothing here moves or modifies data.

## Regenerating

```bash
python scripts/data_audit/scan_drives.py          # cache drive listings (~2 min; --force to rescan)
python scripts/data_audit/build_headcount.py      # emit docs/headcount_*.csv
```

`scan_drives.py` caches a `find` listing per drive under `scripts/data_audit/cache/`, so rebuilding
the CSVs costs seconds and does not compete for disk I/O with running CaImAn/megadata jobs.

## Files

- `headcount_<Mouse>.csv` — one row per **recording day**, sorted by date.
- `headcount_summary.csv` — one row per (mouse, round): expected/raw/CaImAn day counts and the
  dates of any genuinely missing days.

## Two rules the script enforces

1. **Gaps are computed from RAW presence**, with processing state in separate columns. Computing
   gaps from processed sessions conflates "not recorded" with "not yet processed" and massively
   overstates missing data.
2. **Arena is only knowable from a processed session's `Round<N>/<Task>/` path.** Unprocessed raw
   sits flat in `AGING_MICE/<Mouse>/ThirdRound/`-style folders with no arena label, and the camera
   writes `Linear<TIMESTAMP>.avi` regardless of the true arena. So `TASK_UNLABELLED` means "needs
   arena classification" — the script never asserts that a task is missing.

## Gap flags

| Flag | Meaning |
|---|---|
| `MISSING_RAW` | Expected by the round window, no raw anywhere. Emitted as a placeholder row. |
| `NO_VIDEO` | Timestamps written but zero video bytes — a recording-system failure. |
| `INCOMPLETE_SESSION` | A required modality is absent (e.g. Home video with no timestamps). |
| `NO_MINISCOPE` | Behaviour + timestamps but no Miniscope; can never go through CaImAn. |
| `BLOCKED_ON_DENOISE` | Unprocessed, cause known — Mouse944 ring-noise/banding (CLAUDE.md 5.5, 5.6). |
| `UNPROCESSED` | Raw present, no CaImAn, no known cause. |
| `TASK_UNLABELLED` | Arena not determinable from the path — needs classification, **not** a missing task. |
| `ID_MIXUP` | Day affected by a known mouse-ID mix-up; see `notebook_note`. |
| `DISPUTED_ATTRIBUTION` | Data in a "looking for mouse owner" folder — ownership never resolved. |
| `MULTI_FILE_DAY` | Several recordings that day (restarts/splits) — **not** duplicates. |
| `DLC_WRONG_SNAPSHOT` | At least one **linear-project** recording lacks the `160000` snapshot; see `dlc_sessions_missing_160000`. T-Maze is exempt — that project only ever had snapshot `1100000`, so it can never have `160000`. |
| `ORPHAN` | Exists only in a scratch/temp location, no primary copy. |
| `OWNER_FLAGGED` | The session folder name carries an owner annotation (see `owner_annotation`). |
| `PROCESSED_DESPITE_OWNER_FLAG` | **The owner marked this data unusable or misattributed, but it has CaImAn output** — so it may be feeding the analysis tables. |

## Columns worth knowing about

- `raw_gb` — video bytes. Several "present" days are aborted stubs (Mouse847 2026-01-16 is 0.04 GB),
  so presence alone overstates usable data.
- `dlc_sessions_missing_160000` — the exact recordings needing a DLC re-run. Tracked **per
  recording, not per day**: an aging-cohort day holds a Linear and a TMaze session and typically only
  one has the correct snapshot, so a day-level boolean hides most of them.
- `copies_on_mounted_drives` — deliberately *not* a global copy count. A third raw copy exists on an
  `Elements` WD drive (office shelf) that is not mounted and was not inventoried.
- `owner_annotation` — text the owner appended to the session folder name, captured verbatim
  (e.g. `linear do not use mouse sick`, `LINEAR RECORDED AT 6 HZ`,
  `LINEAR DOES NOT BELONG TO THIS MOUSE`). These are real quality judgements that would
  otherwise be invisible to any code that parses only the timestamp out of a folder name.
- `notebook_note` — pre-filled from the lab notebook where it explains the row. The notebook's
  `0712`–`0724` headings are misdated and are really `1012`–`1024`; notes here use the corrected dates.

## Provenance of the expected-day windows

Aging cohort round windows come from the Data Storage doc (Round1 20240521–0609, Round1.5
20240610–0629, Round2 20240718–0719, Round3 20241002–1030, Round3.5 20241031–1113, Baseline
20241213 + 20241216). Mouse1639's Round3.5 is truncated to 3 days. Baseline is two explicit days,
not a range.

The young cohort has **no plan document**, so its rounds are inferred: recording days are segmented
into blocks on breaks > 7 days, and an explicit `Round<N>/` folder label overrides the inference
wherever one exists.
