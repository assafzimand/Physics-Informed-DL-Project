#!/usr/bin/env python3
"""
Fix Analysis HDF5 Files to Match Training Dataset Schema.

This utility script ensures that specified HDF5 analysis files contain a
`source_coords` dataset, which is the standard key used by the current `WaveDataset`
class.

It inspects each file and if a `coordinates` dataset exists but `source_coords`
does not, it creates `source_coords` as a hard link to `coordinates`. This
avoids data duplication while maintaining schema compatibility.

A `.bak` backup of the original file is created before any modifications are
made.

Usage:
    python scripts/utils/fix_analysis_h5.py
"""

import shutil
import h5py
from pathlib import Path
import sys

# Ensure project root is on sys.path for src.* imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.common.paths import get_data_dir


def ensure_source_coords(path: Path) -> None:
    """
    Ensures an HDF5 file at a given path has a `source_coords` dataset.

    If `source_coords` is missing but `coordinates` exists, it creates a hard
    link. Creates a backup file before modifying.

    Args:
        path: The path to the HDF5 file.
    """
    if not path.exists():
        print(f"SKIP (Not Found): {path}")
        return

    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"  -> Backup created: {backup_path}")

    try:
        with h5py.File(path, "r+") as f:
            print(f"Processing: {path.name}")
            keys = list(f.keys())
            print(f"  - Keys before: {keys}")

            if "source_coords" in f:
                print("  - OK: 'source_coords' already exists.")
                return

            if "coordinates" in f:
                # Create a hard link, which is efficient as it doesn't copy data
                f["source_coords"] = f["coordinates"]
                print("  - MODIFIED: Linked 'source_coords' -> 'coordinates'.")
            else:
                print("  - WARN: No 'coordinates' dataset found to link from.")

            print(f"  - Keys after: {list(f.keys())}")
    except Exception as e:
        print(f"  - ERROR: Failed to process file {path}: {e}")


def main() -> None:
    """Main function to run the fix on a predefined list of files."""
    files_to_fix = [
        "wave_dataset_analysis_20samples.h5",
        "wave_dataset_t250_analysis_20samples.h5",
    ]

    data_dir = get_data_dir()
    for filename in files_to_fix:
        ensure_source_coords(data_dir / filename)


if __name__ == "__main__":
    main()
