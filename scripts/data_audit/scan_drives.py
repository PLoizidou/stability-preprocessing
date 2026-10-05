#!/usr/bin/env python3
"""Cache a full file listing of every mounted data drive.

Read-only. Writes one TSV per drive into the cache dir; `build_headcount.py`
consumes these instead of re-walking the drives (a full walk takes ~2 min and
competes for I/O with running CaImAn/megadata jobs).

Usage:
    python scripts/data_audit/scan_drives.py [--cache-dir DIR] [--force]
"""
import argparse
import subprocess
import sys
from pathlib import Path

DRIVES = ["Seagate2", "Seagate3", "SeagateClean", "SeagatePortableDrive", "Seagate5"]
# Seagate5 appeared after the initial audit (2026-09-17). Its ARCHIVE/MiniscopeTest
# tree is 235 GB spanning 2025-11-09..2025-12-18 - i.e. the whole Mouse847/Mouse943
# recording period, not a single pre-experiment test day. Most of it duplicates the
# other drives, but 78 timestamps (8.1 GB) appear nowhere else: 69 are sub-200 MB
# stubs and 9 are complete Home+Linear+Miniscope+timestamps sessions of unknown
# mouse. Those 9 are the reason this drive is scanned rather than ignored - see the
# UNATTRIBUTED report from build_headcount.py.
MEDIA = Path("/media/toor")
# Windows/NTFS bookkeeping dirs that hold no experimental data.
PRUNE = ["$RECYCLE.BIN", "System Volume Information", ".Trash-1000"]

DEFAULT_CACHE = Path(__file__).resolve().parent / "cache"


def scan(drive: str, out: Path) -> int:
    """Walk one drive, writing 'type<TAB>size<TAB>path' rows. Returns row count."""
    root = MEDIA / drive
    if not root.is_dir():
        print(f"  {drive}: NOT MOUNTED - skipped", file=sys.stderr)
        return 0
    prune = []
    for name in PRUNE:
        prune += ["-name", name, "-o"]
    cmd = ["find", str(root), "-mindepth", "1",
           "(", *prune[:-1], ")", "-prune", "-o",
           "-printf", "%y\t%s\t%p\n"]
    with open(out, "w") as fh:
        # find exits non-zero on unreadable dirs; partial output is still usable.
        subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL, check=False)
    n = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
    print(f"  {drive}: {n:,} entries -> {out.name}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true",
                    help="rescan even if a cache file already exists")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning drives -> {args.cache_dir}")
    total = 0
    for drive in DRIVES:
        out = args.cache_dir / f"ls_{drive}.tsv"
        if out.exists() and not args.force:
            n = sum(1 for _ in open(out, encoding="utf-8", errors="replace"))
            print(f"  {drive}: cached ({n:,} entries) - use --force to rescan")
            total += n
            continue
        total += scan(drive, out)
    print(f"Total: {total:,} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
