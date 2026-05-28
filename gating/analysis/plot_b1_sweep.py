#!/usr/bin/env python3
"""B1 analysis: collect the lambda-sweep metrics, print a table, write a CSV,
and plot HOTA & AssA vs lambda on a log-x axis.

Reads the pedestrian_summary.txt that each per-lambda eval dir produced under
<base_out>/lambda_<tag>/<split>/checkpoint_0_tracker/. Pure reader: does not run
TrackEval or the model.

Usage:
    python gating/analysis/plot_b1_sweep.py outputs/b1 [--split val]
"""
import argparse
import csv
import glob
import os
import re

COLS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]


def read_summary(path: str) -> dict:
    with open(path) as f:
        names = f.readline().strip().split(" ")
        values = f.readline().strip().split(" ")
    return {n: float(v) for n, v in zip(names, values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_out", help="sweep base dir, e.g. outputs/b1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--csv", default=None, help="output CSV path (default <base_out>/b1_sweep.csv)")
    ap.add_argument("--plot", default=None, help="output PNG path (default <base_out>/b1_sweep.png)")
    args = ap.parse_args()

    pattern = os.path.join(args.base_out, "lambda_*", args.split, "checkpoint_0_tracker", "pedestrian_summary.txt")
    rows = []
    for path in glob.glob(pattern):
        m = re.search(r"lambda_([0-9eE.+-]+)", path)
        if not m:
            continue
        lam = float(m.group(1))
        rows.append((lam, read_summary(path)))

    if not rows:
        print(f"No summaries found under {pattern}\nRun the sweep first: python gating/scripts/run_b1_sweep.py")
        return

    rows.sort(key=lambda r: r[0])

    print(f"{'lambda':>8}  " + "  ".join(f"{c:>7}" for c in COLS))
    for lam, m in rows:
        print(f"{lam:>8g}  " + "  ".join(f"{m.get(c, float('nan')):>7.3f}" for c in COLS))

    best = max(rows, key=lambda r: r[1].get("HOTA", float("-inf")))
    print(f"\nBest by HOTA: lambda={best[0]:g}  " + "  ".join(f"{c}={best[1].get(c, float('nan')):.3f}" for c in COLS))

    csv_path = args.csv or os.path.join(args.base_out, "b1_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda"] + COLS)
        for lam, m in rows:
            w.writerow([lam] + [m.get(c, "") for c in COLS])
    print(f"\nWrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (CSV written).")
        return

    lams = [r[0] for r in rows]
    hota = [r[1].get("HOTA", float("nan")) for r in rows]
    assa = [r[1].get("AssA", float("nan")) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lams, hota, "o-", label="HOTA")
    ax.plot(lams, assa, "s-", label="AssA")
    ax.set_xscale("log")
    ax.set_xlabel(r"long-memory $\lambda$")
    ax.set_ylabel("score")
    ax.set_title("B1: fixed-$\\lambda$ sweep (DanceTrack val)")
    ax.axvline(0.01, color="gray", ls="--", lw=1, label="released $\\lambda$=0.01")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    plot_path = args.plot or os.path.join(args.base_out, "b1_sweep.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
