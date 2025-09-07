#!/usr/bin/env python3
"""
Fix analysis HDF5 files to match training dataset schema by ensuring
the 'source_coords' dataset exists (linking to existing 'coordinates').

Creates a .bak backup before modifying files.
"""

import os
import shutil
import h5py


def ensure_source_coords(path: str) -> None:
    if not os.path.exists(path):
        print(f"Skip (not found): {path}")
        return

    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")

    with h5py.File(path, "r+") as f:
        keys = list(f.keys())
        print(f"Before: {keys}")
        if "source_coords" in f:
            print(f"Already has 'source_coords': {path}")
            return
        if "coordinates" in f:
            # Create a hard link so no data is copied
            f["source_coords"] = f["coordinates"]
            print(f"Linked 'source_coords' -> 'coordinates' in: {path}")
        else:
            print(f"No 'coordinates' dataset found in: {path}")

        print(f"After: {list(f.keys())}")


def main() -> None:
    files = [
        "data/wave_dataset_analysis_20samples.h5",
        "data/wave_dataset_t250_analysis_20samples.h5",
    ]
    for p in files:
        ensure_source_coords(p)


if __name__ == "__main__":
    main()


