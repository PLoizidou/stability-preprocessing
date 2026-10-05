#!/usr/bin/env python3
"""Find every known filing inconsistency that the consolidation copy must correct.

Read-only. Emits `docs/consolidation_corrections.csv`, one row per correction, which
`build_copy_manifest.py` consumes so the fix is applied during the copy rather than
being re-derived by hand.

Categories:
  WRONG_MOUSE       recording filed under the wrong mouse (owner-ruled)
  ALIAS_FOLDER      folder names a mouse by an alias or bare number
  WRONG_ROUND       session filed under a round it does not belong to
  UNORGANIZED       raw sitting loose instead of in the session lifecycle
  ORPHAN_LOCATION   data living only in a temp/scratch folder
  ANNOTATION        owner judgement encoded in a folder name (would be lost on rename)
  FILENAME_TYPO     misspelled prefix
  SIZE_MISMATCH     same filename, different size between copies - one is corrupt
  UNRESOLVED        ownership the owner could not determine; needs a human, do not guess

Usage:
    python scripts/data_audit/find_inconsistencies.py [--cache-dir DIR] [--out FILE]
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_headcount import (  # noqa: E402  - single source of truth for these
    DRIVES, ID_RULINGS, MOUSE_ALIASES, ANNOTATION_RE, STAMPED, RAW_VIDEO,
    resolve_mouse, load_files,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent.parent / "docs" / "consolidation_corrections.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    files = load_files(args.cache_dir)
    if files is None:
        return 1
    rows = []

    def add(kind, path, current, corrected, evidence, action):
        rows.append(dict(kind=kind, path=path, current=current, corrected=corrected,
                         evidence=evidence, action=action))

    # ---- 1. WRONG_MOUSE: owner-ruled per-timestamp re-attributions ----------
    ruled = defaultdict(list)
    for drive, rel, size in files:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})", rel)
        if not m:
            continue
        key = "".join(m.groups()[:3]) + "T" + "".join(m.groups()[3:])
        raw_mouse = resolve_mouse(rel, drive)
        if raw_mouse is None:
            continue
        aliased = MOUSE_ALIASES.get(raw_mouse, raw_mouse)
        target = ID_RULINGS.get((aliased, key))
        if target:
            ruled[(key, aliased, target)].append(f"{drive}:{os.path.dirname(rel)}")
        elif raw_mouse in MOUSE_ALIASES:
            ruled[(key, raw_mouse, aliased)].append(f"{drive}:{os.path.dirname(rel)}")
    EVID = {
        "Mouse847": "Owner ruling; Mouse846 is Mouse847 / 2025-11-25 recordings from 18:05 are 847's "
                    "(943's own session that day runs 17:31-17:55).",
        "Mouse1636": "Owner ruling, corroborated by lab notebook (06-27 'Mistaken for 1639', "
                     "06-29 'Recordings wrongly under 163', 06-28 folders annotated "
                     "'DOES NOT BELONG TO THIS MOUSE').",
    }
    for (key, cur, tgt), locs in sorted(ruled.items()):
        add("WRONG_MOUSE", " ;; ".join(sorted(set(locs))), cur, tgt,
            EVID.get(tgt, "Owner ruling."),
            f"File this recording ({key}) under {tgt} on the new disk.")

    # ---- 2. ALIAS_FOLDER / 3. ANNOTATION / 4. TYPO -------------------------
    alias_seen, annot_seen = set(), {}
    for drive, rel, size in files:
        top = rel.split("/")[0]
        if drive == "Seagate3" and re.match(r"^94[56]$", top) and top not in alias_seen:
            alias_seen.add(top)
            add("ALIAS_FOLDER", f"{drive}:{top}/", top, f"Mouse{top}",
                "Bare numeric folder at drive root; same recordings also appear under "
                f"YOUNG_MICE/Mouse{top}.", f"Map to Mouse{top}/ on the new disk.")
        for frag, corrected in (("MiniscopeTest_AKA_Mouse847", "Mouse847"),
                                ("ALEKS_TEMP_OLD_MOUSE944_FILES", "Mouse944"),
                                ("Mouse945 - Round2", "Mouse945/Round2")):
            if frag in rel and frag not in alias_seen:
                alias_seen.add(frag)
                add("ALIAS_FOLDER", f"{drive}:{frag}", frag, corrected,
                    "Folder names the mouse by an alias/temp name rather than the canonical ID.",
                    f"Map to {corrected} on the new disk.")
        for seg in rel.split("/"):
            am = ANNOTATION_RE.match(seg)
            if am and seg not in annot_seen:
                annot_seen[seg] = (drive, am.group(1).strip())
        b = os.path.basename(rel)
        sm = STAMPED.match(b)
        if sm and sm.group(1) == "Liner":
            add("FILENAME_TYPO", f"{drive}:{rel}", "Liner<TS>.avi", "Linear<TS>.avi",
                "Misspelled prefix; only occurrence in the dataset.",
                "Rename to Linear<TS>.avi on copy (test-folder file, nothing downstream reads it).")

    for seg, (drive, note) in sorted(annot_seen.items()):
        ts = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", seg).group(1)
        add("ANNOTATION", f"{drive}: .../{seg}", seg, ts,
            f'Owner annotation in folder name: "{note}"',
            "Normalize folder to <TIMESTAMP>/ BUT write the annotation into "
            "<session>/session_notes.json first - it exists nowhere else.")

    # ---- 5. WRONG_ROUND ----------------------------------------------------
    add("WRONG_ROUND", "Seagate2:Mouse1639/Baseline/2024-05-21T17_25_56",
        "Mouse1639/Baseline", "Mouse1639/Round1 (or pre-Round1 test)",
        "Baseline is defined as the pre-sacrifice recording (2024-12-13/16) and applies only to "
        "Mouse1637; a 2024-05-21 recording predates Round1's 05-21 start by hours.",
        "Re-file out of Baseline/.")

    # ---- 6. UNORGANIZED ----------------------------------------------------
    for mouse, rnd, where in [
        ("Mouse1637", "Round2", "SeagatePortableDrive:AGING_MICE/Mouse1637/SecondRound"),
        ("Mouse163", "Round2", "SeagatePortableDrive:AGING_MICE/Mouse163/SecondRound"),
        ("Mouse1636", "Round2", "SeagatePortableDrive:AGING_MICE/Mouse1636/SecondRound"),
        ("Mouse1639", "Round2", "SeagatePortableDrive:AGING_MICE/Mouse1639/SecondRound"),
        ("Mouse1637", "Baseline", "SeagatePortableDrive:AGING_MICE/Mouse1637/BaselineRecordings"),
    ]:
        add("UNORGANIZED", where, "loose files in a flat round folder",
            f"{mouse}/RAW_DATA/<TIMESTAMP>/ + {mouse}/{rnd}/<Task>/<TIMESTAMP>/",
            "Raw sits loose with no per-session folder and no arena label; Round2 is unprocessed "
            "for all four aging mice.",
            "Organize into the CLAUDE.md lifecycle; arena needs classification (see TASK_UNLABELLED).")
    add("UNORGANIZED", "SeagatePortableDrive:AGING_MICE/<Mouse>/{FirstRound,SecondRound,ThirdRound}",
        "FirstRound / SecondRound / ThirdRound", "Round1 / Round2 / Round3",
        "Aging-cohort raw uses prose round names; the rest of the pipeline uses Round<N>.",
        "Rename to Round<N> on copy.")

    # ---- 7. ORPHAN_LOCATION ------------------------------------------------
    add("ORPHAN_LOCATION", "SeagateClean:ALEKS_TEMP_OLD_MOUSE944_FILES{,_2}",
        "temp folder only", "Mouse944/RAW_DATA/",
        "20 Mouse944 timestamps exist ONLY here - no primary copy anywhere.",
        "Copy into Mouse944's tree; without this they are the only copy and would be stranded.")

    # ---- 8. SIZE_MISMATCH --------------------------------------------------
    bysize = defaultdict(set)
    for drive, rel, size in files:
        b = os.path.basename(rel)
        if STAMPED.match(b) or b == "caiman_results.hdf5":
            bysize[(b, os.path.dirname(rel).split("/")[-1] if b == "caiman_results.hdf5" else "")].add(size)
    for (b, _), sizes in sorted(bysize.items()):
        if len(sizes) > 1 and STAMPED.match(b) and STAMPED.match(b).group(1) in RAW_VIDEO:
            add("SIZE_MISMATCH", b, f"sizes differ: {sorted(sizes)}", "pick the authoritative copy",
                "Same filename, different byte size between drives - one copy is truncated or "
                "re-encoded, so they are NOT interchangeable.",
                "Compare and choose before copying; do not let dedup pick arbitrarily.")

    # ---- 9. UNRESOLVED -----------------------------------------------------
    add("UNRESOLVED", "SeagateClean:Mouse163/Round1.5/2024-06-29T17_56_53 - looking for mouse owner",
        "Mouse1636 (by folder)", "UNKNOWN",
        "Owner's own annotation says ownership was never determined.",
        "Copy to NEEDS_REVIEW/ - do not silently assign to a mouse.")
    add("UNRESOLVED", "Mouse163 2024-06-28", "expected per notebook", "LOST",
        "Notebook records a 3:45 PM session; the only two recordings that existed under 163 that "
        "day belong to Mouse1636. Exhaustive search of all 5 drives found no unattributed copy.",
        "Record as lost in session_notes.json; nothing to copy.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "path", "current", "corrected",
                                           "evidence", "action"])
        w.writeheader()
        w.writerows(rows)

    counts = defaultdict(int)
    for r in rows:
        counts[r["kind"]] += 1
    print(f"{len(rows)} corrections -> {args.out}")
    for k, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<18} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
