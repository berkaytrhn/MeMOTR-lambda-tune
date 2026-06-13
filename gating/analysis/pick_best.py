#!/usr/bin/env python3
"""Read per-checkpoint TrackEval output and print the best epoch by HOTA.

Only reads the pedestrian_summary.txt files that EVAL_MODE=continue produced
under <eval_dir>/<split>/checkpoint_*_tracker/. Does not run TrackEval.

Usage:
    python gating/analysis/pick_best.py <eval_dir> [--split val]
"""
import argparse
import glob
import os
import re


def read_summary(path: str) -> dict:
    with open(path) as f:
        names = f.readline().strip().split(" ")
        values = f.readline().strip().split(" ")
    return {n: float(v) for n, v in zip(names, values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--split", default="val")
    ap.add_argument("--metric", default="HOTA")
    args = ap.parse_args()

    pattern = os.path.join(args.eval_dir, args.split, "checkpoint_*_tracker", "pedestrian_summary.txt")
    rows = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"checkpoint_(\d+)_tracker", path)
        epoch = int(m.group(1)) if m else -1
        metrics = read_summary(path)
        rows.append((epoch, metrics))

    if not rows:
        print(f"No summaries found under {pattern}")
        return

    rows.sort(key=lambda r: r[0])
    cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
    print(f"{'epoch':>5}  " + "  ".join(f"{c:>7}" for c in cols))
    for epoch, metrics in rows:
        print(f"{epoch:>5}  " + "  ".join(f"{metrics.get(c, float('nan')):>7.3f}" for c in cols))

    best = max(rows, key=lambda r: r[1].get(args.metric, float("-inf")))
    print(f"\nBest by {args.metric}: epoch {best[0]}")
    print("  " + "  ".join(f"{c}={best[1].get(c, float('nan')):.3f}" for c in cols))


if __name__ == "__main__":
    main()
