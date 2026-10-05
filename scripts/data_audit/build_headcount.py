#!/usr/bin/env python3
"""Build the per-mouse head-count CSVs with gaps flagged.

Reads the cached drive listings from `scan_drives.py` and emits, per mouse, one
row per recording day with what exists on disk, what stage it reached, and which
gap flags apply.

Two rules this script exists to enforce:

1. **Gap detection runs on RAW presence.** Processing state is a separate column.
   Computing gaps from processed sessions conflates "not recorded" with "not yet
   processed" and massively overstates missing data.
2. **Arena is only knowable from a processed session's `Round<N>/<Task>/` path.**
   Unprocessed raw sits flat in `AGING_MICE/<Mouse>/ThirdRound/`-style folders with
   no arena label, and the camera writes `Linear<TS>.avi` regardless of true arena.
   So a day with only one known task is flagged TASK_UNLABELLED ("needs
   classification"), never "task missing".

Usage:
    python scripts/data_audit/build_headcount.py [--cache-dir DIR] [--out-dir DIR]
"""
import argparse
import csv
import datetime as dt
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "cache"
DEFAULT_OUT = HERE.parent.parent / "docs"

DRIVES = ["Seagate2", "Seagate3", "SeagateClean", "SeagatePortableDrive", "Seagate5"]

# ---------------------------------------------------------------------------
# Filename prefixes. Anything matching <prefix><TIMESTAMP>.avi whose prefix is
# not listed here is a hard error: silent skipping is what hid the Baseline
# videos (homecage/miniscopeBaseline) in the first audit pass.
# ---------------------------------------------------------------------------
RAW_VIDEO = {
    "miniscope": "miniscope", "Miniscope": "miniscope",
    "miniscopeBaseline": "miniscope",          # Baseline scheme
    "behavior": "arena",                       # aging cohort: ARENA camera
    "behaviorLinear": "home",                  # aging cohort: HOME box (yes, inverted)
    "Linear": "arena",                         # young cohort: arena, whatever the arena is
    "Liner": "arena",                          # known typo, Mouse847 test folder
    "Home": "home",
    "homecage": "home",                        # Baseline scheme
}
RAW_CSV = {"timestamps": "timestamps"}
TRIMMED_PREFIXES = {
    "Linear_behavior", "Linear_miniscope", "Linear_Miniscope", "Linear_Linear",
    "Linear_Home", "TMaze_miniscope", "TMaze_behavior", "TMaze_Miniscope",
    "TMaze_Linear", "TMaze_Home", "TMaze_behaviorLinear", "Linear_behaviorLinear",
}
DEBANDED_PREFIXES = {"debanded_Linear_Miniscope", "debanded_TMaze_Miniscope",
                     "debanded_Miniscope"}   # untrimmed debanded output (Seagate5 archive)

STAMPED = re.compile(r"^([A-Za-z_]+?)(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})\.(avi|csv)$")

# ---------------------------------------------------------------------------
# Mouse resolution
# ---------------------------------------------------------------------------
MOUSE_RE = re.compile(r"(?:^|[/_ -])(?:sub-)?Mouse[_ ]?(\d{3,4})(?!\d)", re.I)
COHORT_AGING = ["Mouse163", "Mouse1636", "Mouse1637", "Mouse1639"]
COHORT_YOUNG = ["Mouse847", "Mouse943", "Mouse944", "Mouse945", "Mouse946"]
OUT_OF_COHORT = ["Mouse905"]

# Scratch/aborted/test locations: real data, but not the primary copy.
# NB: "looking for mouse owner" folders are NOT scratch - they hold real data whose
# owner the project owner could not determine. They get DISPUTED_ATTRIBUTION instead.
SCRATCH_TOKENS = ("test", "denois", "archive", "temp", "trash", "unusable",
                  "aborted", "trimmed_wrong", "trim3", "mousetest_del")

# ---------------------------------------------------------------------------
# Owner rulings on mouse-ID attribution (see plan; corroborated by lab notebook)
# ---------------------------------------------------------------------------
# Keyed by EXACT TIMESTAMP, not by date. On every disputed day both mice really did
# record; only specific recordings were saved under the wrong mouse. Re-attributing a
# whole day wrongly erases the other mouse's own session for that day.
MOUSE_ALIASES = {"Mouse846": "Mouse847"}     # Mouse846 is Mouse847 throughout
ID_RULINGS = {
    # (mouse_from_path, timestamp) -> corrected mouse
    ("Mouse1639", "20240627T134746"): "Mouse1636",   # notebook: "Mistaken for 1639"
    ("Mouse1639", "20240627T140907"): "Mouse1636",   # same session (Linear then TMaze)
    ("Mouse163",  "20240629T181800"): "Mouse1636",   # notebook: "wrongly under 163"
    # Folders on SeagateClean are annotated "DOES NOT BELONG TO THIS MOUSE"; owner
    # ruled both are Mouse1636. (Slot timing would suggest 163, but the owner's
    # attribution is authoritative - the recording order that day is not reliable.)
    ("Mouse163",  "20240628T154956"): "Mouse1636",   # Linear
    ("Mouse163",  "20240628T161516"): "Mouse1636",   # TMaze
    # All five 2025-11-25 recordings from 18:05 onward are byte-identical mis-copies
    # filed under both mice. Mouse943's own session that day runs 17:31-17:55.
    ("Mouse943",  "20251125T180549"): "Mouse847",
    ("Mouse943",  "20251125T180550"): "Mouse847",
    ("Mouse943",  "20251125T181205"): "Mouse847",
    ("Mouse943",  "20251125T182027"): "Mouse847",
    ("Mouse943",  "20251125T183014"): "Mouse847",
}
# The owner's own annotation for a recording they could not attribute.
DISPUTED_MARKER = "looking for mouse owner"
# The owner annotates session folders in place, e.g.
# "2024-06-20T15_41_54 - linear do not use mouse sick". These are real quality
# judgements and several mark data as unusable or misattributed, so they are
# captured verbatim rather than discarded with the rest of the folder name.
ANNOTATION_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\s*-\s*(\S.*)$")
ID_MIXUP_DATES = {
    ("Mouse1636", "20240627"): "Notebook 06-27: 'Mouse 1636: Linear track first... Mistaken for 1639 "
                               "when the recordings and Bpod data were saved.' Re-attributed to 1636.",
    ("Mouse1636", "20240629"): "Notebook 06-29: 'Mouse 1636: Recordings wrongly under 163.' "
                               "Re-attributed to 1636.",
    ("Mouse1636", "20240628"): "Folders on SeagateClean were annotated 'LINEAR/TMAZE DOES NOT BELONG "
                               "TO THIS MOUSE' under Mouse163; owner ruled both recordings "
                               "(15:49:56, 16:15:16) are Mouse1636.",
    ("Mouse163",  "20240628"): "LOST. The notebook records a 163 session at 3:45 PM, but the only two "
                               "recordings that existed under 163 that day (15:49:56, 16:15:16) belong "
                               "to Mouse1636 per the owner. An exhaustive search of all four drives "
                               "found exactly 11 recordings on 2024-06-28, all attributed "
                               "(1637 14:00-14:39, 1639 15:07/15:31, 1636 15:49/16:15/17:36/17:37/18:08) "
                               "- no unattributed copy exists. Owner confirmed it cannot be found.",
    ("Mouse1636", "20240530"): "Notebook 05-30: 'copy 163 files over to 1636 BPOD' - affects BPOD "
                               "attribution only, not the video files.",
    ("Mouse163", "20240530"): "Notebook 05-30: 'Mistakenly thought it was Mouse 163 at the beginning "
                              "but copied files over' - Bpod attribution only.",
    ("Mouse847", "20251125"): "2025-11-25 sub-recordings also filed under Mouse943; owner ruled these "
                              "are Mouse847. The Mouse943 copies are mis-copies.",
    ("Mouse847", "20251119"): "Sessions 2025-11-19..23 were filed under 'Mouse846'; owner ruled "
                              "Mouse846 IS Mouse847. Merged.",
}

# ---------------------------------------------------------------------------
# Experimental plan (aging cohort), from the Data Storage doc
# ---------------------------------------------------------------------------
AGING_WINDOWS = {
    "Round1":   ("20240521", "20240609"),
    "Round1.5": ("20240610", "20240629"),
    "Round2":   ("20240718", "20240719"),
    "Round3":   ("20241002", "20241030"),
    "Round3.5": ("20241031", "20241113"),
    "Baseline": ("20241213", "20241216"),
}
AGING_EXPECTED = {
    "Mouse163":  ["Round1", "Round1.5", "Round2"],
    "Mouse1636": ["Round1", "Round1.5", "Round2"],
    "Mouse1637": ["Round1", "Round1.5", "Round2", "Round3", "Round3.5", "Baseline"],
    "Mouse1639": ["Round1", "Round1.5", "Round2", "Round3", "Round3.5"],
}
# Mouse1639 ran only the first 3 days of the Round3.5 window before dying.
ROUND_TRUNCATION = {("Mouse1639", "Round3.5"): 3}
# Baseline is two specific recording days, not a contiguous window - 12-14/12-15
# were never recording days and must not be counted as missing.
ROUND_EXPLICIT_DAYS = {"Baseline": ["20241213", "20241216"]}


def round_days(mouse, rnd):
    """The days a round is expected to cover for this mouse."""
    if rnd in ROUND_EXPLICIT_DAYS:
        return list(ROUND_EXPLICIT_DAYS[rnd])
    days = dayrange(*AGING_WINDOWS[rnd])
    trunc = ROUND_TRUNCATION.get((mouse, rnd))
    return days[:trunc] if trunc else days

# ---------------------------------------------------------------------------
# Notebook explanations, pre-filled into `notebook_note`.
# Dates already corrected: the notebook's 0712..0724 headings are really 1012..1024.
# ---------------------------------------------------------------------------
WEDDING = ("Notebook 07-19: 'Only have time to do two mice (Andrea's wedding) so I will do the two "
           "that have seemingly the best imaging quality; mice 163 and 1637.'")
NO_MINISCOPE_1639 = ("Notebook 06-10: '1639 will stay on the linear track without miniscope recordings "
                     "as I cannot enter the sx room to fix his connector. Also the room is quarantined.' "
                     "Raw exists but is correctly excluded from Round1.5.")
NOTEBOOK_NOTES = {
    ("Mouse1636", "20240620"): "Notebook 06-20: 'Not doing well... Didn't go through with the recording "
                               "due to the mouse not moving well. Back on ad lib water.'",
    ("Mouse1636", "20240622"): "Notebook 06-22: session lists only 1639/163/1637 - 1636 not run.",
    ("Mouse1636", "20240623"): "Notebook 06-23: 'Mouse 1636 seems to be doing much better' - not run.",
    ("Mouse1636", "20240624"): "Notebook 06-24: session lists only 163/1639/1637 - 1636 not run.",
    ("Mouse1636", "20240718"): "Notebook 07-18: 'Very bad health. will be euthanized today. but runs ok.'",
    ("Mouse1636", "20240719"): WEDDING + " 1636 euthanized 07-18/19.",
    ("Mouse1639", "20240719"): WEDDING,
    ("Mouse1639", "20240610"): NO_MINISCOPE_1639,
    ("Mouse1639", "20240611"): NO_MINISCOPE_1639,
    ("Mouse1639", "20240612"): NO_MINISCOPE_1639,
    ("Mouse1639", "20240523"): "Notebook 05-23: linear track video corrupted, Bonsai stopped; restarted.",
    ("Mouse1639", "20240525"): "Notebook 05-25: 'Linear track video did not save' - disk full.",
    ("Mouse1637", "20240521"): "Notebook 05-21: Bonsai froze repeatedly; mouse returned to cage. "
                               "Round1 effectively starts 05-23.",
    ("Mouse1637", "20240522"): "Notebook 05-21/22: aborted start attempts; raw exists, unprocessed.",
    ("Mouse163",  "20240529"): "Notebook 05-29: 'Decided to rerun mouse 163 since I lost the previous "
                               "data.' Multiple files that day are reruns, not duplicates.",
    ("Mouse163",  "20240605"): "Notebook 06-05: servodoor failure; session split across several files.",
    # Re-dated Round3 entries (notebook headings said 07-xx)
    ("Mouse1639", "20241012"): "Notebook 10-12 (re-dated from 07-12): 'miniscope is not working after a "
                               "few minutes.'",
    ("Mouse1637", "20241013"): "Notebook 10-13 (re-dated from 07-13): 'linear track of mouse 1637 was "
                               "not recorded due to low memory.'",
    ("Mouse1637", "20241016"): "Notebook 10-16 (re-dated): first day of TMaze4 for Mouse 1637.",
    ("Mouse1639", "20241017"): "Notebook 10-17 (re-dated): blood in cage; switched to TMaze4 a day early.",
    ("Mouse1639", "20241019"): "Notebook 10-19 (re-dated): 'Very weak. Maybe I should skip it for "
                               "tomorrow and put it back on ad lib.'",
    ("Mouse1639", "20241020"): "Not recorded - notebook 10-19 (re-dated) planned to skip this day.",
    ("Mouse1639", "20241007"): "No notebook entry; owner confirms nothing notable happened.",
    ("Mouse1639", "20241008"): "No notebook entry; owner confirms nothing notable happened.",
    ("Mouse1639", "20241023"): "Notebook 10-23 (re-dated): TMaze4 water port 1 failed; restarted after "
                               "replacement. Extra file is a restart, not a duplicate.",
    ("Mouse1637", "20241024"): "Notebook 10-24 (re-dated): 'Port 2 doesn't work. Fixed it and restarted "
                               "TMaze4.' Extra file is a restart, not a duplicate.",
    ("Mouse1637", "20241022"): "Notebook 10-22 (re-dated): blood in the cage.",
    # Young cohort - data-derived, no notebook supplied
    ("Mouse847",  "20251128"): "No video written at all - only timestamps CSVs. Recording-system "
                               "failure; affected Mouse943 the same day.",
    ("Mouse943",  "20251128"): "No video written at all - only timestamps CSVs. Recording-system "
                               "failure; affected Mouse847 the same day.",
    ("Mouse946",  "20260222"): "Not recorded. Mouse945 managed only a 0.44 GB fragment the same day.",
    ("Mouse945",  "20260222"): "Only a 0.44 GB fragment recorded; unprocessed.",
    ("Mouse945",  "20260731"): "Not recorded. No notebook or plan document covers the young cohort, "
                               "so no explanation is available - genuinely unexplained.",
}
MOUSE944_DENOISE = ("Unprocessed - Miniscope ring-noise / banding artifact (CLAUDE.md steps 5.5, 5.6). "
                    "Needs denoise before CaImAn.")
MOUSE944_BLOCKED = {"20260710", "20260717", "20260718", "20260719", "20260720",
                    "20260722", "20260723", "20260724", "20260725", "20260726", "20260731"}


def dayrange(a, b):
    A = dt.datetime.strptime(a, "%Y%m%d").date()
    B = dt.datetime.strptime(b, "%Y%m%d").date()
    return [(A + dt.timedelta(d)).strftime("%Y%m%d") for d in range((B - A).days + 1)]


def is_scratch(relpath):
    p = relpath.lower()
    return any(t in p for t in SCRATCH_TOKENS)


def resolve_mouse(relpath, drive):
    """Mouse ID from a path, handling every aliasing convention on these drives."""
    up = relpath.upper()
    if "MOUSE944" in up:                       # ALEKS_TEMP_OLD_MOUSE944_FILES
        base = "Mouse944"
    elif re.match(r"^94[56](/|$)", relpath) and drive == "Seagate3":
        base = "Mouse" + relpath[:3]           # bare 945/ and 946/ dirs
    else:
        found = MOUSE_RE.findall(relpath)
        base = "Mouse" + found[-1] if found else None
    return base


def apply_ruling(mouse, key):
    """Resolve a mouse alias, then any per-timestamp attribution ruling."""
    mouse = MOUSE_ALIASES.get(mouse, mouse)
    return ID_RULINGS.get((mouse, key), mouse)


def load_files(cache_dir):
    rows = []
    for drive in DRIVES:
        f = cache_dir / f"ls_{drive}.tsv"
        if not f.exists():
            print(f"missing cache file {f}; run scan_drives.py first", file=sys.stderr)
            return None
        prefix = f"/media/toor/{drive}/"
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t", 2)
                if len(parts) != 3:
                    continue
                typ, size, path = parts
                if typ != "f" or not path.startswith(prefix):
                    continue
                rows.append((drive, path[len(prefix):], int(size) if size.isdigit() else 0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    files = load_files(args.cache_dir)
    if files is None:
        return 1
    print(f"Loaded {len(files):,} files from {len(DRIVES)} drives")

    # ---- pass 0: attribute paths that do not name a mouse ------------------
    # Some real data lives in folders with no mouse in the path at all: a session
    # folder at a drive root, CaimanTesting/, ARCHIVE/ trees. Dropping those silently
    # loses data, so resolve them by timestamp against paths that DO name a mouse.
    ANYKEY = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})")
    key_to_mouse = defaultdict(set)
    for drive, rel, _ in files:
        km = ANYKEY.search(rel)
        mouse0 = resolve_mouse(rel, drive)
        if km and mouse0:
            k = "".join(km.groups()[:3]) + "T" + "".join(km.groups()[3:])
            key_to_mouse[k].add(apply_ruling(mouse0, k))

    def resolve(rel, drive, key):
        """Mouse from the path; else from the timestamp; else None (unattributed)."""
        m = resolve_mouse(rel, drive)
        if m:
            return apply_ruling(m, key)
        cands = key_to_mouse.get(key, set())
        return next(iter(cands)) if len(cands) == 1 else None

    unattributed = defaultdict(lambda: [0, 0])

    # ---------------- classify every timestamped file ----------------
    unknown_prefixes = defaultdict(list)
    # day[(mouse, date)] -> aggregates
    raw = defaultdict(lambda: {"kinds": set(), "ts": set(), "disputed": set(),
                               "locs": set(), "primary": False, "video_bytes": 0})
    for drive, rel, size in files:
        base = os.path.basename(rel)
        m = STAMPED.match(base)
        if not m:
            continue
        pfx, Y, Mo, D, h, mi, s = m.group(1), *m.groups()[1:7]
        ext = m.group(8)
        date = f"{Y}{Mo}{D}"
        key = f"{date}T{h}{mi}{s}"

        if pfx in TRIMMED_PREFIXES or pfx in DEBANDED_PREFIXES:
            continue                                  # processing outputs, handled below
        if ext == "csv" and pfx in RAW_CSV:
            kind = "timestamps"
        elif ext == "avi" and pfx in RAW_VIDEO:
            kind = RAW_VIDEO[pfx]
        elif ext == "avi":
            unknown_prefixes[pfx].append(rel)          # hard error, collected for reporting
            continue
        else:
            continue                                   # non-raw csv (DLC etc.)

        mouse = resolve(rel, drive, key)
        if mouse is None:
            u = unattributed[f"{drive}:{'/'.join(rel.split('/')[:2])}"]
            u[0] += 1
            u[1] += size
            continue
        rec = raw[(mouse, date)]
        rec["kinds"].add(kind)
        rec["ts"].add(key)
        rec["locs"].add(f"{drive}:{os.path.dirname(rel)}")
        if not is_scratch(rel):
            rec["primary"] = True
        if DISPUTED_MARKER in rel.lower():
            rec["disputed"].add(key)
        if kind != "timestamps":
            rec["video_bytes"] += size

    if unknown_prefixes:
        print("\nERROR: unknown <prefix><TIMESTAMP>.avi prefixes found.", file=sys.stderr)
        print("Add each to RAW_VIDEO / TRIMMED_PREFIXES / DEBANDED_PREFIXES, or this "
              "script would silently skip real data:", file=sys.stderr)
        for pfx, paths in sorted(unknown_prefixes.items()):
            print(f"  {pfx!r}  ({len(paths)} files)  e.g. {paths[0]}", file=sys.stderr)
        return 2
    print("Prefix check: all timestamped .avi prefixes are known")

    # ---------------- processing state, from session directories ----------------
    proc = defaultdict(lambda: {"tasks": set(), "variants": set(), "caiman": False,
                                "dlc": False, "dlc160": False, "trimmed": False,
                                "trials": False, "rounds": set(), "disputed": set()})
    # DLC snapshot state is per RECORDING, not per day: an aging-cohort day holds a
    # Linear and a TMaze session, and typically only one of them has the 160000
    # snapshot. Aggregating to the day would hide every session that needs re-running.
    #
    # The 160000 rule is LINEAR-ONLY (CLAUDE.md: "Linear track snapshot versions").
    # The linear project has two snapshots on disk - 160000 (preferred) and 100000
    # (stale) - while the T-Maze project only ever had 1100000. Flagging a T-Maze
    # session for lacking 160000 is meaningless: no such snapshot exists for it.
    dlc_by_key = defaultdict(lambda: {"any": False, "has160": False, "linear": False})
    annotations = defaultdict(set)
    for drive, rel, size in files:
        base = os.path.basename(rel)
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})", rel)
        if not m:
            continue
        date = m.group(1) + m.group(2) + m.group(3)
        key = f"{date}T{m.group(4)}{m.group(5)}{m.group(6)}"
        mouse = resolve(rel, drive, key)
        if mouse is None:
            continue
        for seg in rel.split("/"):
            am = ANNOTATION_RE.match(seg)
            if am:
                annotations[(mouse, date)].add(am.group(1).strip())
        if is_scratch(rel):
            continue
        p = proc[(mouse, date)]
        segs = rel.split("/")
        if base == "caiman_results.hdf5":
            p["caiman"] = True
            p["variants"].add(segs[-2])
            for seg in segs:
                if seg.lower().startswith("tmaze"):
                    p["tasks"].add("TMaze")
                elif seg.lower().startswith("linear"):
                    p["tasks"].add("Linear")
            rm = re.search(r"Round\s?([0-9.]+?)(?:/|$)", rel)
            if rm:
                p["rounds"].add("Round" + rm.group(1).rstrip("."))
        if DISPUTED_MARKER in rel.lower():
            p["disputed"].add(key)
        if base.endswith("_filtered.csv"):
            p["dlc"] = True
            dlc_by_key[key]["any"] = True
            if re.search(r"DLC_resnet50_linear", base):
                dlc_by_key[key]["linear"] = True
            if "160000" in base:
                p["dlc160"] = True
                dlc_by_key[key]["has160"] = True
        if STAMPED.match(base) and STAMPED.match(base).group(1) in TRIMMED_PREFIXES:
            p["trimmed"] = True
        if "binned" in base and base.endswith(".csv"):
            p["trials"] = True

    # ---------------- assign rounds ----------------
    def aging_round(mouse, date):
        for rnd in AGING_EXPECTED.get(mouse, []):
            if date in round_days(mouse, rnd):
                return rnd
        return "Unscheduled"

    def young_rounds(mouse, days):
        """Segment a young mouse's recording days into blocks and label them."""
        if not days:
            return {}
        blocks, cur = [], [days[0]]
        for d in days[1:]:
            gap = (dt.datetime.strptime(d, "%Y%m%d").date()
                   - dt.datetime.strptime(cur[-1], "%Y%m%d").date()).days
            if gap > 7:
                blocks.append(cur)
                cur = [d]
            else:
                cur.append(d)
        blocks.append(cur)
        out = {}
        for i, bl in enumerate(blocks):
            # Cover the block's whole calendar span, so a missing day inherits the
            # round of the block it falls inside instead of landing in "Unscheduled".
            span = dayrange(bl[0], bl[-1])
            if len(bl) < 3:
                label = "Extra"          # e.g. Mouse847's lone 2026-01-16 aborted stubs
            elif i == 0:
                for j, d in enumerate(span):
                    out[d] = "Round1" if j < 20 else "Round1.5"
                continue
            else:
                label = f"Round{i + 1}"
            for d in span:
                out[d] = label
        return out

    all_mice = sorted({m for m, _ in raw} | {m for m, _ in proc})
    young_round_map = {}
    for mouse in all_mice:
        if mouse in COHORT_YOUNG:
            days = sorted({d for (mm, d) in raw if mm == mouse and raw[(mm, d)]["primary"]})
            young_round_map[mouse] = young_rounds(mouse, days)

    # ---------------- build rows ----------------
    out_rows = defaultdict(list)
    missing_total = []
    for mouse in all_mice:
        if mouse not in COHORT_AGING + COHORT_YOUNG + OUT_OF_COHORT:
            continue
        recorded = sorted({d for (mm, d) in raw if mm == mouse})
        # expected days per plan (aging only)
        expected = set()
        if mouse in AGING_EXPECTED:
            for rnd in AGING_EXPECTED[mouse]:
                expected |= set(round_days(mouse, rnd))
        # young: expected = the calendar span of each block
        elif mouse in young_round_map:
            rmap = young_round_map[mouse]
            for label in set(rmap.values()):
                bl = sorted(d for d, l in rmap.items() if l == label)
                if len(bl) >= 3:
                    expected |= set(dayrange(bl[0], bl[-1]))

        for date in sorted(set(recorded) | expected):
            rec = raw.get((mouse, date))
            p = proc.get((mouse, date), {})
            rnd = (aging_round(mouse, date) if mouse in AGING_EXPECTED
                   else young_round_map.get(mouse, {}).get(date, "Unscheduled"))
            if p.get("rounds") and len(p["rounds"]) == 1:
                rnd = next(iter(p["rounds"]))          # trust an explicit folder label
            tasks = sorted(p.get("tasks", ()))
            flags = []

            if rec is None:
                flags.append("MISSING_RAW")
                missing_total.append((mouse, rnd, date))
            else:
                kinds = rec["kinds"]
                if rec["video_bytes"] == 0:
                    flags.append("NO_VIDEO")
                elif "miniscope" not in kinds:
                    flags.append("NO_MINISCOPE")
                elif not {"arena", "home"} & kinds or "timestamps" not in kinds:
                    flags.append("INCOMPLETE_SESSION")
                if len(rec["ts"]) > 1:
                    flags.append("MULTI_FILE_DAY")
                if not rec["primary"]:
                    flags.append("ORPHAN")
                if rec["disputed"]:
                    flags.append("DISPUTED_ATTRIBUTION")

            if rec is not None and rec["video_bytes"] > 0:
                if not p.get("caiman"):
                    if mouse == "Mouse944" and date in MOUSE944_BLOCKED:
                        flags.append("BLOCKED_ON_DENOISE")
                    elif "NO_MINISCOPE" not in flags and "NO_VIDEO" not in flags:
                        flags.append("UNPROCESSED")
                if len(tasks) < 2 and rnd in ("Round1.5", "Round3", "Round2"):
                    flags.append("TASK_UNLABELLED")
            # Per-recording, not per-day: list exactly which sessions need re-running.
            dlc_bad = sorted(k for k in (rec["ts"] if rec else ())
                             if dlc_by_key[k]["linear"] and not dlc_by_key[k]["has160"])
            if dlc_bad:
                flags.append("DLC_WRONG_SNAPSHOT")
            if p.get("disputed"):
                flags.append("DISPUTED_ATTRIBUTION")
            notes_on_disk = sorted(annotations.get((mouse, date), ()))
            if notes_on_disk:
                flags.append("OWNER_FLAGGED")
                # The owner marked this data unusable or misattributed, yet it was
                # processed anyway - so it may be feeding the analysis tables.
                bad = ("unusable", "do not use", "does not belong", "needs to be removed",
                       "problematic", "too short", "no miniscope", "6 hz")
                if p.get("caiman") and any(b in a.lower() for a in notes_on_disk for b in bad):
                    flags.append("PROCESSED_DESPITE_OWNER_FLAG")
            if (mouse, date) in ID_MIXUP_DATES:
                flags.append("ID_MIXUP")

            note = NOTEBOOK_NOTES.get((mouse, date), "")
            if not note and (mouse, date) in ID_MIXUP_DATES:
                note = ID_MIXUP_DATES[(mouse, date)]
            if not note and "BLOCKED_ON_DENOISE" in flags:
                note = MOUSE944_DENOISE

            out_rows[mouse].append({
                "mouse": mouse, "round": rnd, "task": "|".join(tasks),
                "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                "n_timestamps": len(rec["ts"]) if rec else 0,
                "timestamps": "|".join(sorted(rec["ts"])) if rec else "",
                "expected_per_plan": 1 if date in expected else 0,
                "raw_miniscope": int(bool(rec and "miniscope" in rec["kinds"])),
                "raw_behavior": int(bool(rec and {"arena", "home"} & rec["kinds"])),
                "raw_timestamps": int(bool(rec and "timestamps" in rec["kinds"])),
                "raw_gb": round(rec["video_bytes"] / 1e9, 3) if rec else 0.0,
                "dlc": int(bool(p.get("dlc"))), "dlc_160000": int(bool(p.get("dlc160"))),
                "dlc_sessions_missing_160000": "|".join(dlc_bad),
                "trimmed": int(bool(p.get("trimmed"))), "caiman": int(bool(p.get("caiman"))),
                "caiman_variants": "|".join(sorted(p.get("variants", ()))),
                "trial_csv": int(bool(p.get("trials"))),
                "copies_on_mounted_drives": len(rec["locs"]) if rec else 0,
                "locations": " ;; ".join(sorted(rec["locs"])) if rec else "",
                "owner_annotation": " ;; ".join(notes_on_disk),
                "GAP": "|".join(flags), "notebook_note": note,
            })

    # ---------------- write ----------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(out_rows[all_mice[0]][0].keys()) if out_rows else []
    written = []
    for mouse, rows in sorted(out_rows.items()):
        path = args.out_dir / f"headcount_{mouse}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["date"]))
        written.append((mouse, len(rows), path))

    summary = args.out_dir / "headcount_summary.csv"
    with open(summary, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mouse", "round", "expected_days", "raw_days", "caiman_days",
                    "missing_days", "missing_dates", "flagged_days"])
        for mouse, rows in sorted(out_rows.items()):
            by_round = defaultdict(list)
            for r in rows:
                by_round[r["round"]].append(r)
            for rnd, rs in sorted(by_round.items()):
                miss = [r["date"] for r in rs if "MISSING_RAW" in r["GAP"]]
                w.writerow([mouse, rnd, sum(r["expected_per_plan"] for r in rs),
                            sum(1 for r in rs if r["n_timestamps"] > 0),
                            sum(r["caiman"] for r in rs), len(miss), "|".join(miss),
                            sum(1 for r in rs if r["GAP"])])

    # ---------------- report + assertions ----------------
    print(f"\nWrote {len(written)} per-mouse CSVs to {args.out_dir}")
    for mouse, n, path in written:
        flagged = sum(1 for r in out_rows[mouse] if r["GAP"])
        print(f"  {mouse:<10} {n:>3} day-rows, {flagged:>3} flagged  {path.name}")
    print(f"  summary -> {summary.name}")

    n_raw_ts = len({t for rec in raw.values() for t in rec["ts"]})
    n_proc_days = sum(1 for rows in out_rows.values() for r in rows if r["caiman"])
    flag_counts = defaultdict(int)
    for rows in out_rows.values():
        for r in rows:
            for f in filter(None, r["GAP"].split("|")):
                flag_counts[f] += 1
    if unattributed:
        tot = sum(v[1] for v in unattributed.values())
        print(f"\nUNATTRIBUTED - timestamped data in folders naming no mouse, and whose "
              f"timestamp appears nowhere under a mouse ({tot/1e9:.1f} GB):")
        for loc, (n, b) in sorted(unattributed.items(), key=lambda x: -x[1][1]):
            print(f"  {b/1e9:>8.1f} GB  {n:>5} files  {loc}")
        print("  -> these go to NEEDS_REVIEW/ in the consolidation copy, not to a mouse.")
    print(f"\nunique raw timestamps: {n_raw_ts}")
    print(f"day-rows with CaImAn : {n_proc_days}")
    print("flag counts:")
    for f, c in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {f:<22} {c}")
    print(f"\nMISSING_RAW days ({len(missing_total)}):")
    for m, rnd, d in missing_total:
        print(f"  {m:<10} {rnd:<10} {d}")

    ok = True
    if n_raw_ts != 2000:
        print(f"\nWARNING: expected 2000 unique raw timestamps, got {n_raw_ts}", file=sys.stderr)
        ok = False
    if len(missing_total) != 11:
        print(f"\nWARNING: expected 11 MISSING_RAW days, got {len(missing_total)}", file=sys.stderr)
        ok = False
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
