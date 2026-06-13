#!/usr/bin/env python3
"""
gating/scripts/fetch_pretrained.py
Usage:
    python gating/scripts/fetch_pretrained.py --gdrive_id <FILE_ID>
"""

import argparse, os, sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdrive_id", required=True,
                        help="Google Drive file ID from MeMOTR README")
    parser.add_argument("--out_dir",
                        default="/workspace/persistent/checkpoints_pretrained")
    parser.add_argument("--filename", default="memotr_dancetrack.pth")
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        os.system(f"{sys.executable} -m pip install gdown -q")
        import gdown

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.filename)

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Already exists: {out_path} ({size_mb:.1f} MB), skipping.")
        return

    print(f"Downloading MeMOTR DanceTrack checkpoint → {out_path}")
    gdown.download(f"https://drive.google.com/uc?id={args.gdrive_id}",
                   out_path, quiet=False)

    size_mb = os.path.getsize(out_path) / 1e6
    if size_mb < 50:
        print(f"ERROR: File is only {size_mb:.1f} MB — download likely failed.")
        sys.exit(1)

    print(f"Done: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()