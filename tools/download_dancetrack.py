#!/usr/bin/env python3
"""Download DanceTrack from Hugging Face into the layout MeMOTR expects.

Source: https://huggingface.co/datasets/noahcao/dancetrack
(the official mirror by the dataset authors, MOT-formatted annotations).

The repo stores plain zip archives at its root:
    train1.zip + train2.zip   (40 sequences total)
    val.zip                   (25 sequences)
    test1.zip + test2.zip     (35 sequences)

This script downloads only the archives for the requested splits, unzips
them, and moves each `dancetrack000*` folder into the expected location.

Resulting tree (under --data-root):
    DATA_ROOT/DanceTrack/
        train/                  (40 sequences, public annotations)
        val/                    (25 sequences, public annotations)
        test/                   (35 sequences, no public annotations)
        train_seqmap.txt
        val_seqmap.txt
        test_seqmap.txt

Usage:
    pip install huggingface_hub
    python tools/download_dancetrack.py --data-root /path/to/DATA_ROOT
    # optional: --splits train val test          (default: all three)
    #           --repo-id noahcao/dancetrack     (override HF dataset)
    #           --skip-download                  (regenerate seqmaps only)
    #           --hf-token TOKEN                 (for gated/private mirrors)

Notes:
    - The HF download is resumable; re-running picks up where it stopped.
    - `huggingface-cli login` is also honored.
    - If the HF layout changes, the script searches one level deep for
      `dancetrack*` folders and moves them into place.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List


DEFAULT_REPO_ID = "noahcao/dancetrack"

# noahcao/dancetrack stores plain zip archives at the repo root, split across
# multiple files for size. Each archive expands to a flat set of
# `dancetrack000*/` sequence folders.
SPLIT_ARCHIVES: Dict[str, List[str]] = {
    "train": ["train1.zip", "train2.zip"],
    "val":   ["val.zip"],
    "test":  ["test1.zip", "test2.zip"],
}


def ensure_hf_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is required. Install with: pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)


def hf_download_file(repo_id: str, filename: str, cache_dir: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download
    print(f"  hf_hub_download {repo_id}:{filename}")
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(cache_dir),
        token=token,
    )
    return Path(local)


def unzip_into(zip_path: Path, dest_dir: Path) -> None:
    print(f"  unzip {zip_path.name} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


def collect_dancetrack_dirs(root: Path) -> List[Path]:
    """Return every `dancetrack*` directory under `root` (recursive)."""
    found: List[Path] = []
    for p in root.rglob("dancetrack*"):
        if p.is_dir():
            # Skip nested duplicates: if a parent is already a dancetrack* dir, ignore.
            if any(parent.name.startswith("dancetrack") for parent in p.parents if parent != root):
                continue
            found.append(p)
    return found


def materialize_split(staging: Path, dst: Path) -> int:
    """Move each `dancetrack*` folder under `staging` into `dst`."""
    dst.mkdir(parents=True, exist_ok=True)
    seqs = collect_dancetrack_dirs(staging)
    moved = 0
    for seq in seqs:
        target = dst / seq.name
        if target.exists():
            continue
        shutil.move(str(seq), str(target))
        moved += 1
    return moved


def write_seqmap(split_dir: Path, out_file: Path) -> int:
    seqs = sorted(p.name for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("dancetrack"))
    if not seqs:
        raise RuntimeError(f"Cannot build seqmap: no sequences in {split_dir}")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("name\n")
        for s in seqs:
            f.write(f"{s}\n")
    print(f"  wrote {out_file} ({len(seqs)} sequences)")
    return len(seqs)


def have_split(split_dir: Path) -> bool:
    return split_dir.exists() and any(
        p.is_dir() and p.name.startswith("dancetrack") for p in split_dir.iterdir()
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path,
                   help="Parent dir; DanceTrack/ will be created inside.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"HF dataset id (default: {DEFAULT_REPO_ID})")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="HF cache dir (default: $HF_HOME or ~/.cache/huggingface).")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HF auth token (also reads $HF_TOKEN).")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip HF download; only regenerate seqmap files.")
    p.add_argument("--keep-zips", action="store_true",
                   help="Keep downloaded zip files in the HF cache after extraction.")
    args = p.parse_args()

    root = args.data_root / "DanceTrack"
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        ensure_hf_hub()
        cache_dir = args.cache_dir or (root / "_hf_cache")
        staging_root = root / "_staging"

        for split in args.splits:
            dst = root / split
            if have_split(dst):
                print(f"[{split}] already materialized at {dst}, skipping.")
                continue
            archives = SPLIT_ARCHIVES.get(split, [])
            if not archives:
                print(f"[{split}] no archive mapping known; skipping.", file=sys.stderr)
                continue

            split_staging = staging_root / split
            split_staging.mkdir(parents=True, exist_ok=True)

            print(f"[{split}] downloading {len(archives)} archive(s) from {args.repo_id}")
            zips: List[Path] = []
            for arc in archives:
                zips.append(hf_download_file(args.repo_id, arc, cache_dir, args.hf_token))

            for z in zips:
                unzip_into(z, split_staging)

            n = materialize_split(split_staging, dst)
            print(f"[{split}] {n} sequences moved into {dst}")

            # Clean staging; the HF cache holds the zips themselves.
            shutil.rmtree(split_staging, ignore_errors=True)
            if not args.keep_zips:
                for z in zips:
                    try:
                        z.unlink()
                    except OSError:
                        pass

        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)

    # Regenerate seqmaps for whatever is present.
    for split in args.splits:
        split_dir = root / split
        if not have_split(split_dir):
            print(f"[{split}] not present at {split_dir}; skipping seqmap.")
            continue
        write_seqmap(split_dir, root / f"{split}_seqmap.txt")

    print("\nDone. Point training/eval at:")
    print(f"  --data-root {args.data_root}")


if __name__ == "__main__":
    main()
