#!/usr/bin/env python3
"""Flatten nested contour data directories into a single flat directory."""

import argparse
import shutil
from pathlib import Path


METADATA_FILES = {"processed.json", "categories_done.json"}


def main():
    parser = argparse.ArgumentParser(description="Flatten contour data into one directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory with nested JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Flat output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for json_file in sorted(input_dir.rglob("*.json")):
        if json_file.name in METADATA_FILES:
            continue
        dest = output_dir / json_file.name
        if dest.exists():
            print(f"WARNING: duplicate name {json_file.name}, skipping {json_file}")
            continue
        shutil.copy2(json_file, dest)
        print(f"Copied {json_file} -> {dest}")
        count += 1

    print(f"\nDone. Copied {count} files to {output_dir}")


if __name__ == "__main__":
    main()
