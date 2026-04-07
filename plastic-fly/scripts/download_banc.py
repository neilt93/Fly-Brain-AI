"""
Download BANC connectome data from Harvard Dataverse.

No authentication needed — direct download via Dataverse API.

Dataset: doi:10.7910/DVN/8TFGGB (CC-BY 4.0 license)
    BANC = Brain And Nerve Cord — first complete brain+VNC connectome of an
    adult female Drosophila melanogaster (~160K neurons, 214M synapses).

Files:
    banc_626_data.sqlite — v626 connectivity and metadata (~684 MB)
    banc_meta_821.tab    — v821 neuron annotations with cell types (~36 MB)

Schema (banc_626_data.sqlite):
    meta            — neuron annotations (root_id, flow, super_class, cell_class,
                      cell_type, region, side, nerve, neurotransmitter_predicted)
    edgelist_simple — connectivity (pre_pt_root_id, post_pt_root_id, n)

Meta columns (33 total, key ones):
    root_id, root_626, supervoxel_id, nucleus_id, position, nucleus_position,
    region, side, proofread, roughly_proofread, flow, super_class, cell_class,
    cell_sub_class, hemilineage, cell_function, cell_function_detailed,
    neurotransmitter_verified, neuropeptide_verified, neurotransmitter_predicted,
    peripheral_target_type, body_part_sensory, body_part_effector, nerve,
    fafb_match, hemibrain_match, manc_match, fanc_match, sexually_dimorphic

Classification hierarchy:
    flow:        afferent | efferent | intrinsic
    super_class: ascending | descending | motor | sensory | visual_projection |
                 visual_centrifugal | endocrine | optic_lobe_intrinsic |
                 central_brain_intrinsic
    cell_class:  ~106 categories (e.g., leg_motor_neuron,
                 antennal_lobe_projection_neuron, olfactory_receptor_neuron)
    cell_type:   individual neuron names (e.g., DNge110, Ti flexor MN)

VNC neuron identification:
    region='ventral_nerve_cord'
    Leg MNs: super_class='motor' AND cell_class contains 'leg_motor'
    DNs: super_class='descending' OR flow='efferent'

Usage:
    python scripts/download_banc.py                # download BANC sqlite + meta
    python scripts/download_banc.py --inspect       # inspect existing DB
    python scripts/download_banc.py --meta-only     # just the v821 meta table
"""

import sys
import sqlite3
import argparse
from pathlib import Path
from time import time

ROOT = Path(__file__).resolve().parent.parent
BANC_DIR = ROOT / "data" / "banc"

# Harvard Dataverse: doi:10.7910/DVN/8TFGGB
# File download via Dataverse Data Access API:
#   https://dataverse.harvard.edu/api/access/datafile/{file_id}
#
# File IDs discovered via dataset metadata API (2026-04-03):
#   banc_626_data.sqlite: file_id = 11842995 (sql/ directory, ~684 MB)
#   banc_meta_821.tab:    file_id = 13457250 (root directory, ~36 MB)
DATAVERSE_BASE = "https://dataverse.harvard.edu/api/access/datafile"
BANC_SQLITE_FILE_ID = 11842995
BANC_META_821_FILE_ID = 13457250

BANC_SQLITE_URL = f"{DATAVERSE_BASE}/{BANC_SQLITE_FILE_ID}"
BANC_META_821_URL = f"{DATAVERSE_BASE}/{BANC_META_821_FILE_ID}"


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
    parser = argparse.ArgumentParser(description="Download BANC connectome data")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect existing database files")
    parser.add_argument("--meta-only", action="store_true",
                        help="Download only the v821 meta table (36 MB)")
    args = parser.parse_args()

    _ensure_dirs()

    if args.inspect:
        inspect_db(BANC_DIR / "banc_626_data.sqlite")
        meta_tab = BANC_DIR / "banc_meta_821.tab"
        if meta_tab.exists():
            print(f"\n=== banc_meta_821.tab ({meta_tab.stat().st_size/1024/1024:.0f} MB) ===")
            print(f"  (Tab-separated metadata file, {sum(1 for _ in open(meta_tab)):,} lines)")
        return

    print("BANC connectome download")
    print(f"  Dataset: doi:10.7910/DVN/8TFGGB (CC-BY 4.0)")
    print(f"  Destination: {BANC_DIR}")
    print()

    # Download SQLite (main database, ~684 MB)
    if not args.meta_only:
        dest = BANC_DIR / "banc_626_data.sqlite"
        if dest.exists():
            print(f"  banc_626_data.sqlite already exists ({dest.stat().st_size/1024/1024:.0f} MB)")
        else:
            ok = download_file(BANC_SQLITE_URL, dest, "banc_626_data.sqlite (~684 MB)")
            if not ok:
                print()
                print("  If automatic download fails, download manually:")
                print(f"    URL: {BANC_SQLITE_URL}")
                print(f"    Save to: {dest}")
                print()
                print("  Or browse the dataset at:")
                print("    https://dataverse.harvard.edu/dataset.xhtml"
                      "?persistentId=doi:10.7910/DVN/8TFGGB")

    # Download v821 meta table (~36 MB)
    dest = BANC_DIR / "banc_meta_821.tab"
    if dest.exists():
        print(f"  banc_meta_821.tab already exists ({dest.stat().st_size/1024/1024:.0f} MB)")
    else:
        ok = download_file(BANC_META_821_URL, dest, "banc_meta_821.tab (~36 MB)")
        if not ok:
            print()
            print("  If automatic download fails, download manually:")
            print(f"    URL: {BANC_META_821_URL}")
            print(f"    Save to: {dest}")

    # Verify
    print()
    print("=== Verification ===")
    db = BANC_DIR / "banc_626_data.sqlite"
    if db.exists():
        inspect_db(db)
    else:
        print(f"  banc_626_data.sqlite: NOT FOUND")

    meta = BANC_DIR / "banc_meta_821.tab"
    if meta.exists():
        print(f"\n  banc_meta_821.tab: {meta.stat().st_size/1024/1024:.0f} MB (OK)")
    else:
        print(f"  banc_meta_821.tab: NOT FOUND")


if __name__ == "__main__":
    main()
