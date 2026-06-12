#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def sync_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"source directory does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync root postprocess outputs from one directory to another.")
    parser.add_argument("--src", required=True, help="Source directory with corrected root outputs.")
    parser.add_argument("--dst", required=True, help="Destination directory to receive the outputs.")
    args = parser.parse_args()

    sync_dir(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
