# Bpod reward-log anomalies — background inventory

Compiled 2026-09-04 from a full scan of `/media/toor/SeagateClean/Bpod Local/Data` (all `Mouse*.mat`,
recursive) cross-referenced against every session folder on the mounted drives that has a
`dlc_trim_info.csv`. Companion to [reward_extraction_plan.md](reward_extraction_plan.md).

**Nothing here blocks the current pipeline.** It is a to-fix / to-watch list for when reward
extraction is implemented. Items marked ⚠️ need a human decision or a look at the acquisition PC.

Inventory: **695 unique Bpod sessions** (after de-duplication) against **646 trimmed imaging
sessions**; 601 imaging sessions have a task-matched Bpod log on the same day.

---

## ⚠️ 1. Bpod logging stopped on 2026-07-25 while recording continued to 2026-08-04

The largest single gap. Both mice, both tasks, no log at all:

| Mouse | Sessions with no Bpod log | Last Bpod log | Missing dates |
|---|---|---|---|
| Mouse944 | 17 (9 Linear + 8 TMaze) | `2026-07-25 16:13:46` | 07-27 … 08-04 |
| Mouse945 (Round2) | 17 (8 Linear + 9 TMaze) | `2026-07-25 15:33:16` | 07-26 … 08-04 |

Both stop on the same day, which points at one event on the Bpod PC (data directory changed, or the
drive it saved to was swapped) rather than 34 independent failures. **Worth checking whether these
logs still exist on the acquisition PC before treating those sessions as video-only.**

Confirmed absent everywhere: `/media/toor/SeagateClean/Bpod Local` is a strict superset of the other
three copies (`Seagate3/Bpod Local`, and the two `Bpod RAW/Bpod Local` trees) — no Mouse944/945 logs
exist there that aren't already in the canonical copy.

## 2. Other sessions with no Bpod log (13, scattered)

| Mouse | Dates |
|---|---|
| Mouse163 | 2024-05-22, 2024-05-25 |
| Mouse1636 | 2024-06-14, 2024-06-21, 2024-06-27, 2024-06-28 |
| Mouse1637 | 2024-05-23, 2024-10-24 |
| Mouse1639 | 2024-05-22 |
| Mouse847 | 2025-11-10 |
| Mouse943 | 2025-11-17 |

Several are the very first recording days for that mouse (habituation, protocol probably not run).

## 3. Restarts — more than one Bpod log for one recording day

28 mouse-task-days. Most are "short aborted attempt, then the real session"; a few are two genuinely
full sessions hours apart, which pair with two *different* videos and must not be merged.

| Mouse | Task | Date | Logs (start, nTrials, duration) |
|---|---|---|---|
| Mouse163 | Linear | 2024-05-29 | `16:14:51` nT=-1 dur=nans → `18:51:05` nT=42 dur=1276s |
| Mouse163 | Linear | 2024-06-14 | `16:50:52` nT=1 dur=4s → `16:51:38` nT=31 dur=769s |
| Mouse1636 | Linear | 2024-05-28 | `13:20:38` nT=1 dur=28s → `13:29:35` nT=63 dur=1290s |
| Mouse1637 | Linear | 2024-11-04 | `14:03:57` nT=9 dur=136s → `14:25:12` nT=56 dur=663s |
| Mouse1637 | TMaze | 2024-10-24 | `13:30:16` nT=11 dur=207s → `13:44:42` nT=37 dur=927s |
| Mouse1639 | Linear | 2024-05-23 | `16:44:44` nT=4 dur=778s → `17:12:38` nT=6 dur=955s |
| Mouse1639 | Linear | 2024-06-12 | `16:43:58` nT=78 dur=753s → `16:57:25` nT=8 dur=225s |
| Mouse1639 | Linear | 2024-06-27 | `13:53:29` nT=40 dur=409s → `15:34:42` nT=43 dur=345s |
| Mouse1639 | TMaze | 2024-06-14 | `13:24:27` nT=36 dur=911s → `18:21:58` nT=25 dur=952s |
| Mouse1639 | TMaze | 2024-06-18 | `14:24:51` nT=4 dur=53s → `14:39:18` nT=53 dur=917s |
| Mouse1639 | TMaze | 2024-06-27 | `14:09:29` nT=39 dur=908s → `15:49:23` nT=38 dur=900s |
| Mouse1639 | TMaze | 2024-06-28 | `15:13:48` nT=42 dur=568s → `17:43:24` nT=40 dur=883s |
| Mouse1639 | TMaze | 2024-10-06 | `13:22:48` nT=1 dur=5s → `13:24:41` nT=1 dur=35s |
| Mouse1639 | TMaze | 2024-10-23 | `12:18:48` nT=3 dur=92s → `12:39:18` nT=24 dur=893s |
| Mouse847 | Linear | 2025-11-19 | `17:30:00` nT=14 dur=1096s → `18:04:41` nT=44 dur=952s |
| Mouse847 | Linear | 2025-11-23 | `17:42:52` nT=38 dur=852s → `17:59:01` nT=2 dur=61s |
| Mouse847 | TMaze | 2025-12-02 | `18:45:40` nT=1 dur=78s → `18:52:41` nT=1 dur=4s → `18:55:55` nT=16 dur=993s |
| Mouse905 | Linear | 2026-07-09 | `14:27:04` nT=1 dur=251s → `14:40:09` nT=31 dur=833s |
| Mouse943 | Linear | 2025-12-09 | `18:09:09` nT=10 dur=525s → `18:43:39` nT=1 dur=34s → `18:44:38` nT=7 dur=485s |
| Mouse943 | TMaze | 2025-12-15 | `17:04:09` nT=3 dur=294s → `17:09:39` nT=23 dur=611s |
| Mouse944 | Linear | 2026-06-26 | `15:25:44` nT=1 dur=276s → `15:31:00` nT=11 dur=1800s |
| Mouse944 | Linear | 2026-07-09 | `13:55:22` nT=2 dur=74s → `13:58:41` nT=53 dur=935s |
| Mouse944 | TMaze | 2026-07-18 | `15:27:28` nT=1 dur=201s → `15:35:38` nT=16 dur=1238s |
| Mouse945 | Linear | 2026-01-22 | `16:14:43` nT=2 dur=7s → `16:15:07` nT=19 dur=1801s |
| Mouse945 | TMaze | 2026-02-18 | `16:00:55` nT=1 dur=65s → `16:02:16` nT=8 dur=820s |
| Mouse946 | Linear | 2026-02-23 | `16:54:04` nT=3 dur=232s → `16:58:07` nT=40 dur=422s |
| Mouse946 | TMaze | 2026-02-14 | `18:27:44` nT=6 dur=172s → `18:35:04` nT=33 dur=922s |
| Mouse946 | TMaze | 2026-02-15 | `17:22:05` nT=3 dur=386s → `18:57:37` nT=48 dur=978s |

Rule of thumb from the table: an `nT ≤ 3` / `dur < 300 s` log immediately followed by a long one is
an aborted start — use the long one. Two long logs hours apart (Mouse1639 TMaze 2024-06-14,
2024-06-27, 2024-06-28; Mouse946 TMaze 2026-02-15) are two real sessions.

⚠️ `Mouse163_LinearTrack_20240529_161451.mat` fails to load (`buffer is too small for requested
array`) — truncated file. A second, healthy log exists for that day (`18:51:05`, 42 trials).

## ⚠️ 4. `TMaze4/Session Data/` contains `TMaze2` sessions, and nested `Round*` subfolders

The protocol folder does **not** identify the protocol, and a flat `Session Data/*.mat` glob misses
files:

* `Mouse163/TMaze4/Session Data/` holds **20 `Mouse163_TMaze2_*.mat` files and zero TMaze4 files**.
  Same pattern (to a lesser degree) for Mouse1636/1637/1639.
* All 164 such copies are **byte-identical** duplicates of the file in the sibling `TMaze2` folder —
  harmless, but they double-count if you glob recursively without de-duplicating.
* Nested subfolders that a flat glob misses entirely:
  `Mouse1637/TMaze4/Session Data/Round2` (2), `…/Round3` (29),
  `Mouse1639/TMaze4/Session Data/Round2` (1), `…/Round3` (33),
  `Mouse163/TMaze4/Session Data/Round2` (2),
  `Mouse847/LinearTrack/Session Data/Round1.5` (18),
  `Mouse943/LinearTrack/Session Data/Round1.5/FirstTask` (5) and `…/SecondTask` (9).

**Rules:** glob recursively; take the protocol from the *filename*, never the folder; de-duplicate on
`(mouse, protocol, timestamp)`.

Mouse1637/Mouse1639 `Round3` mixes protocols within the round — `TMaze2` files on 2024-10-02…10-04,
`TMaze4` from 2024-10-29 onward. Reward/LED semantics differ between the two (see the plan doc), so
the protocol must be resolved per session, not per round.

## 5. Filename timestamps are not reliable enough to pair a log with a video

Comparing Bpod session start against the DLC-detected task start (`dlc_trim_info.csv`, converted
through the real per-frame clock) over the 601 matched sessions:

| | |
|---|---|
| median | −18 s (Bpod launched shortly before the mouse steps onto the track — expected) |
| within ±30 s | 56% |
| within ±60 s | 69% |
| within ±300 s | 91% |
| range | −1369 s … +2697 s |

Worst offenders, where the Bpod session ends **before** the imaged task window even starts, i.e. the
pairing is certainly wrong or the two PC clocks disagree badly:

| Mouse | Task | Session | Bpod log | offset |
|---|---|---|---|---|
| Mouse163 | Linear | 2024-06-29 17:56:53 | `20240629_184812` | +2697 s |
| Mouse944 | Linear | 2026-07-01 14:42:03 | `20260701_142515` | −1369 s |
| Mouse944 | Linear | 2026-07-21 14:32:10 | `20260721_141951` | −1090 s |
| Mouse1639 | Linear | 2024-05-25 19:15:57 | `20240525_190143` | −1058 s |
| Mouse847 | Linear | 2025-11-12 17:19:09 | `20251112_171436` | −629 s |
| Mouse946 | TMaze | 2026-02-12 17:23:14 | `20260212_172…` | −565 s |
| Mouse945 | TMaze | 2026-07-23 14:58:12 | `20260723_145002` | −532 s |

This is the concrete reason the plan fits the offset per session from the video rather than trusting
filenames.

## 6. Frame-clock issues that affect reward timing

* Real capture rate is ~**24.8–24.9 fps**, not 25. `dlc_trim_info.csv:start_time_seconds` is computed
  as `start_frame / 25` and is therefore ~1–2 s optimistic for aging-cohort sessions; the nominal
  960 s trim window is really 964.8 s of wall-clock at the median (min 960.1, max 1019.1 over 497
  clean sessions).
* **83 of 642** trimmed windows contain a >2 s wall-clock gap — a concatenation splice or a dropped
  block. Worst cases:

  | Mouse | Task | Session | max gap in window |
  |---|---|---|---|
  | Mouse945 | TMaze | 2026-07-16 14:32:41 | 702 s |
  | Mouse944 | Linear | 2026-07-15 15:30:22 | 268 s |
  | Mouse945 | Linear | 2026-02-07 17:05:17 | 247 s |
  | Mouse945 | Linear | 2026-02-12 16:55:14 | 239 s |
  | Mouse946 | Linear | 2026-02-03 16:05:44 | 214 s |
  | Mouse944 | Linear | 2026-07-04 15:05:50 | 211 s |

  Mostly young-cohort concatenated sessions. `timestamps*.csv` handles this correctly (row *i* is
  still frame *i*, with the true wall-clock including the gap) — but anything that assumes a constant
  frame rate does not. One extreme: `Mouse847 Linear 2025-12-06T16_10_37` is a single "session" whose
  timestamps span 3.3 h with a **1.85 h** gap between sub-recordings.

* Only **60 of 646** session folders keep their own `timestamps*.csv`, and the same filename
  timestamp can match a short aborted stub as well as the real file — e.g. `2025-12-17T17_05_56`
  matches a **14-row** stub in `Mouse943/TMaze/short_recordings/` and the real **42612-row**
  concatenated file. Resolve from `RAW_DATA/<TIMESTAMP>/` and assert `n_rows == n_frames`.

## 7. Linear-track sessions with an unrewarded final trial (expected, not a bug)

31 sessions have `nReward < nTrials` — all `LinearTrack`, all at `dur ≈ 1800 s`: **23** with one
unrewarded trial and **8** with two. That is the `totalSessionTime` timeout firing in `WaitForPoke1`
(`Tup → exit`), which appends a trial with `States.Reward = NaN`. In the 8 two-trial cases the loop
runs one further iteration whose `Timer = totalSessionTime - timeElapsed` is already ≤ 0, so it exits
instantly — verified on `Mouse944_LinearTrack_20260627_151928`, where trial 15 is a genuine 441 s
timeout and trial 16 lasts 0.0001 s.

Treat the real timeout trial as a genuine unrewarded trial (not a parse failure), and **drop
zero-duration trials** (`TrialEndTimestamp - TrialStartTimestamp < ~0.01 s`).

## 8. `LightOnTrials` indexes the maximum trial count, not the completed one

`TMaze4` draws `LightOnTrials = randperm(90, 45)` before the session runs, so it contains indices far
above the number of trials actually completed. Example: `Mouse944_TMaze4_20260723_153813` has 34
trials and `LightOnTrials = [5, 7, 14, 29, 34, 75, 82, 84, 87]` → only **5 of 34** trials were cued,
not "50%". Always intersect with `1:nTrials`.

## 9. No hardware sync exists

Across the whole dataset the only Bpod event channels present are `Port<N>In`, `Port<N>Out` and
`Tup` — no `BNC*` or `Wire*` TTL from the miniscope. Alignment is inference-only; there is no ground
truth to fall back on beyond the video itself.

---

## Machine-readable companions (in `docs/`)

| File | Contents |
|---|---|
| `bpod_session_inventory.csv` | 695 unique Bpod sessions: mouse, protocol, start, path, `nTrials`, duration, `nRew`, T-maze error-state entries, `LightOnTrials` presence/in-range count, duplicate-copy count |
| `bpod_video_pairing.csv` | 646 trimmed imaging sessions → best-matching Bpod log, offset in seconds against the true frame clock, or a `NO_BPOD_*` note |
| `session_frameclock_gaps.csv` | per session: `timestamps*.csv` row count, frames available after `start_frame`, number and size of >2 s wall-clock gaps inside the 960 s window, true wall-clock duration of that window |

Caveat on the last two: only 60 of 646 sessions keep a `timestamps*.csv` in their own folder, so for
the rest the file was resolved by filename match across all drives. That resolution is ambiguous
where a short aborted-recording stub shares the timestamp (§6), so a handful of rows report
implausibly small `n_rows`. Re-derive from `RAW_DATA/<TIMESTAMP>/` when this is turned into code.
