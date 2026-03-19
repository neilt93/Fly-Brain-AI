"""
Download BANC / Frankenbrain connectome data from Harvard Dataverse.

No authentication needed — direct download.

Files:
    banc_data.sqlite              — 160K neurons, full BANC connectome
    frankenbrain_v1.1_data.sqlite — FAFB brain + MANC VNC, bridged via neck_bridge

Schema:
    meta            — neuron annotations (cell_type, modality, super_class)
    edgelist_simple — connectivity (pre, post, count)
    neck_bridge     — BANC↔MANC ID mapping (Frankenbrain only)

Usage:
    python scripts/download_banc.py                  # download both
    python scripts/download_banc.py --banc-only       # just BANC
    python scripts/download_banc.py --frankenbrain-only
    python scripts/download_banc.py --inspect         # inspect existing DB
"""

import sys
import sqlite3
import argparse
from pathlib import Path
from time import time

ROOT = Path(__file__).resolve().parent.parent
BANC_DIR = ROOT / "data" / "banc"

# Harvard Dataverse URLs — update these with actual DOI links once confirmed
# The user indicated these are direct-download SQLite files on Harvard Dataverse.
# TODO: Replace with actual download URLs from Harvard Dataverse.
BANC_URL = None  # e.g., "https://dataverse.harvard.edu/api/access/datafile/XXXXX"
FRANKENBRAIN_URL = None


def _ensure_dirs():
    BANC_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, label: str):
    """Download a file with progress reporting."""
    if url is None:
        print(f"  {label}: URL not configured yet.")
        print(f"    Find the file on Harvard Dataverse and update the URL in this script,")
        print(f"    or manually download and place at: {dest}")
        return False

    try:
        import urllib.request
        print(f"  Downloading {label}...")
        t0 = time()

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = downloaded / total_size * 100
                mb = downloaded / 1024 / 1024
                print(f"\r    {mb:.0f} MB ({pct:.0f}%)", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print(f"\n    Done in {time()-t0:.1f}s ({dest.stat().st_size/1024/1024:.0f} MB)")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def inspect_db(db_path: Path):
    """Inspect a SQLite database and print schema + stats."""
    if not db_path.exists():
        print(f"  Not found: {db_path}")
        return

    print(f"\n=== {db_path.name} ({db_path.stat().st_size/1024/1024:.0f} MB) ===")
    con = sqlite3.connect(str(db_path))

    # List tables
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"  Tables: {[t[0] for t in tables]}")

    for (table_name,) in tables:
        # Row count
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        # Column info
        cols = con.execute(f"PRAGMA table_info({table_name})").fetchall()
        col_names = [c[1] for c in cols]

        print(f"\n  {table_name}: {count:,} rows")
        print(f"    Columns: {col_names}")

        # Sample first row
        sample = con.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchone()
        if sample:
            print(f"    Sample: {dict(zip(col_names, sample))}")

        # For meta table, show super_class breakdown
        if table_name == "meta" and "super_class" in col_names:
            breakdown = con.execute(
                f"SELECT super_class, COUNT(*) as n FROM {table_name} "
                f"GROUP BY super_class ORDER BY n DESC LIMIT 10"
            ).fetchall()
            print(f"    super_class: {dict(breakdown)}")

        # For meta table, show modality breakdown
        if table_name == "meta" and "modality" in col_names:
            breakdown = con.execute(
                f"SELECT modality, COUNT(*) as n FROM {table_name} "
                f"WHERE modality IS NOT NULL AND modality != '' "
                f"GROUP BY modality ORDER BY n DESC LIMIT 10"
            ).fetchall()
            if breakdown:
                print(f"    modality: {dict(breakdown)}")

    con.close()


def main():
    parser = argparse.ArgumentParser(description="Download BANC/Frankenbrain data")
    parser.add_argument("--banc-only", action="store_true",
                        help="Download only banc_data.sqlite")
    parser.add_argument("--frankenbrain-only", action="store_true",
                        help="Download only frankenbrain_v1.1_data.sqlite")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect existing database files")
    args = parser.parse_args()

    _ensure_dirs()

    if args.inspect:
        inspect_db(BANC_DIR / "banc_data.sqlite")
        inspect_db(BANC_DIR / "frankenbrain_v1.1_data.sqlite")
        return

    print("BANC / Frankenbrain download")
    print(f"  Destination: {BANC_DIR}")
    print()

    if not args.frankenbrain_only:
        dest = BANC_DIR / "banc_data.sqlite"
        if dest.exists():
            print(f"  banc_data.sqlite already exists ({dest.stat().st_size/1024/1024:.0f} MB)")
        else:
            download_file(BANC_URL, dest, "banc_data.sqlite")

    if not args.banc_only:
        dest = BANC_DIR / "frankenbrain_v1.1_data.sqlite"
        if dest.exists():
            print(f"  frankenbrain already exists ({dest.stat().st_size/1024/1024:.0f} MB)")
        else:
            download_file(FRANKENBRAIN_URL, dest, "frankenbrain_v1.1_data.sqlite")

    # Instructions for manual download
    if BANC_URL is None or FRANKENBRAIN_URL is None:
        print()
        print("=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print()
        print("Download these SQLite files from Harvard Dataverse:")
        print()
        print("  1. banc_data.sqlite")
        print("     -> 160K neurons, full BANC connectome")
        print()
        print("  2. frankenbrain_v1.1_data.sqlite")
        print("     -> FAFB brain + MANC VNC, bridged via neck_bridge table")
        print("     -> This is the most useful one for our pipeline")
        print()
        print(f"  Place both files in: {BANC_DIR}")
        print()
        print("  Then verify with: python scripts/download_banc.py --inspect")

    # Verify
    print()
    print("=== Verification ===")
    for name in ["banc_data.sqlite", "frankenbrain_v1.1_data.sqlite"]:
        p = BANC_DIR / name
        if p.exists():
            inspect_db(p)
        else:
            print(f"  {name}: NOT FOUND")


if __name__ == "__main__":
    main()
