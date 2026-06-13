#!/usr/bin/env python3
"""Sanity-check a DanceTrack download for MeMOTR training.

Verifies the on-disk layout that `data/dancetrack.py` and TrackEval expect:
    DATA_ROOT/DanceTrack/
        train/<seq>/img1/*.jpg + gt/gt.txt + seqinfo.ini
        val/<seq>/img1/*.jpg   + gt/gt.txt + seqinfo.ini
        test/<seq>/img1/*.jpg  + seqinfo.ini             (no gt — annotations private)
        train_seqmap.txt
        val_seqmap.txt
        test_seqmap.txt

Reports per-split: sequence count, first/last frame count, missing files,
malformed gt lines. Exits non-zero on any hard error.

Usage:
    python tools/check_dancetrack.py --data-root ./dancetrack
    python tools/check_dancetrack.py --data-root ./dancetrack --splits train val
    python tools/check_dancetrack.py --data-root ./dancetrack --strict   # also fail on warnings
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import os
import glob

# Sequences per split for the official DanceTrack release.
EXPECTED_COUNTS = {"train": 40, "val": 25, "test": 35}

# Splits that ship with public gt/.
SPLITS_WITH_GT = {"train", "val"}


def check_seq(seq_dir: Path, need_gt: bool) -> Tuple[List[str], List[str], int]:
    """Return (errors, warnings, n_frames) for one sequence."""
    errors: List[str] = []
    warnings: List[str] = []

    img1 = seq_dir / "img1"
    if not img1.is_dir():
        errors.append(f"missing img1/  ({seq_dir})")
        n_frames = 0
    else:
        frames = sorted(img1.glob("*.jpg"))
        n_frames = len(frames)
        if n_frames == 0:
            errors.append(f"img1/ has no .jpg files  ({seq_dir})")

    seqinfo = seq_dir / "seqinfo.ini"
    if not seqinfo.is_file():
        warnings.append(f"missing seqinfo.ini  ({seq_dir})")

    gt = seq_dir / "gt" / "gt.txt"
    if need_gt:
        if not gt.is_file():
            errors.append(f"missing gt/gt.txt  ({seq_dir})")
        else:
            # Spot-check first 5 lines have the 10-field MOT shape.
            try:
                with open(gt, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 5:
                            break
                        parts = line.strip().split(",")
                        if len(parts) < 9:
                            errors.append(f"gt line {i+1} has {len(parts)} fields (<9)  ({gt})")
                            break
                        try:
                            int(parts[0]); int(parts[1])
                            float(parts[2]); float(parts[3]); float(parts[4]); float(parts[5])
                        except ValueError:
                            errors.append(f"gt line {i+1} has non-numeric fields  ({gt})")
                            break
            except OSError as e:
                errors.append(f"cannot read gt.txt: {e}")
    else:
        if gt.is_file():
            warnings.append(f"unexpected gt/gt.txt on split without public annotations  ({seq_dir})")

    return errors, warnings, n_frames


def check_split(split_root: Path, split: str) -> Tuple[int, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not split_root.is_dir():
        errors.append(f"split dir missing: {split_root}")
        return 0, errors, warnings

    seqs = sorted(p for p in split_root.iterdir() if p.is_dir() and p.name.startswith("dancetrack"))
    if not seqs:
        errors.append(f"no dancetrack* sequences under {split_root}")
        return 0, errors, warnings

    expected = EXPECTED_COUNTS.get(split)
    if expected is not None and len(seqs) != expected:
        warnings.append(f"{split}: expected {expected} sequences, found {len(seqs)}")

    frame_counts: List[int] = []
    need_gt = split in SPLITS_WITH_GT
    for seq in seqs:
        e, w, n = check_seq(seq, need_gt=need_gt)
        errors.extend(e)
        warnings.extend(w)
        frame_counts.append(n)

    if frame_counts:
        print(f"  [{split}] {len(seqs)} sequences, "
              f"frames min/median/max = {min(frame_counts)}/"
              f"{sorted(frame_counts)[len(frame_counts)//2]}/{max(frame_counts)}")

    return len(seqs), errors, warnings


def check_seqmap(root: Path, split: str, split_dir: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    seqmap = root / f"{split}_seqmap.txt"
    if not seqmap.is_file():
        warnings.append(f"missing {seqmap.name} (TrackEval needs this — re-run download script "
                        f"with --skip-download to regenerate)")
        return errors, warnings

    on_disk = {p.name for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("dancetrack")}
    listed: List[str] = []
    with open(seqmap, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header.lower() != "name":
            warnings.append(f"{seqmap.name}: first line should be 'name', got '{header}'")
        for line in f:
            s = line.strip()
            if s:
                listed.append(s)

    missing_on_disk = sorted(set(listed) - on_disk)
    missing_in_map = sorted(on_disk - set(listed))
    if missing_on_disk:
        errors.append(f"{seqmap.name}: lists {len(missing_on_disk)} seq not on disk: {missing_on_disk[:3]}...")
    if missing_in_map:
        warnings.append(f"{seqmap.name}: {len(missing_in_map)} seq on disk not listed: {missing_in_map[:3]}...")

    return errors, warnings


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path,
                   help="Parent dir (contains DanceTrack/).")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on warnings as well as errors.")
    args = p.parse_args()

    root = args.data_root / "DanceTrack"
    if not root.is_dir():
        print(f"FAIL: {root} does not exist. Pass --data-root <parent of DanceTrack/>.", file=sys.stderr)
        return 2

    print(f"checking {root}")
    all_errors: List[str] = []
    all_warnings: List[str] = []
    for split in args.splits:
        n, e, w = check_split(root / split, split)
        all_errors.extend(e); all_warnings.extend(w)
        if n > 0:
            se, sw = check_seqmap(root, split, root / split)
            all_errors.extend(se); all_warnings.extend(sw)

    print()
    if all_warnings:
        print(f"{len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"  WARN  {w}")
    if all_errors:
        print(f"{len(all_errors)} error(s):")
        for e in all_errors:
            print(f"  ERR   {e}")
        return 1

    print("OK" if not all_warnings else "OK with warnings")
    return 1 if (args.strict and all_warnings) else 0

if __name__ == "__main__":
    sys.exit(main())
    
