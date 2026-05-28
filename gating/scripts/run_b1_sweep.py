#!/usr/bin/env python3
"""B1 — fixed-lambda inference sweep driver (no training, no code beyond base).

For each lambda value it builds a self-contained eval dir:

    <base_out>/lambda_<tag>/
        train/config.yaml      # copy of the released train config, LONG_MEMORY_LAMBDA patched
        checkpoint_0.pth        # symlink to the released checkpoint
        val/...                 # TrackEval output written here by eval

then runs `main.py --mode eval` (EVAL_MODE=specific) on it. submit() rebuilds the
model from that dir's train/config.yaml, so the patched lambda is what runs.

The sweep is the empirical answer to challenge C1 and the curve B3a is judged
against. lambda=0.01 must reproduce B0.

Idempotent: a lambda whose pedestrian_summary.txt already exists is skipped, so
the sweep can be resumed after an interruption.

Usage (from repo root):
    python gating/scripts/run_b1_sweep.py
    python gating/scripts/run_b1_sweep.py --lambdas 0.001 0.01 0.05 0.1 0.5
    python gating/scripts/run_b1_sweep.py --dry-run
"""
import argparse
import os
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LAMBDAS = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1]


def lambda_tag(lam: float) -> str:
    # filesystem-safe, stable, sorts sensibly: 0.001 -> "0.001"
    return f"{lam:g}"


def build_lambda_dir(base_out: str, lam: float, released_train_config: str, checkpoint: str) -> str:
    eval_dir = os.path.join(base_out, f"lambda_{lambda_tag(lam)}")
    train_dir = os.path.join(eval_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    with open(released_train_config) as f:
        cfg = yaml.safe_load(f)
    cfg["LONG_MEMORY_LAMBDA"] = float(lam)
    with open(os.path.join(train_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    ckpt_link = os.path.join(eval_dir, "checkpoint_0.pth")
    if os.path.islink(ckpt_link) or os.path.exists(ckpt_link):
        os.remove(ckpt_link)
    os.symlink(os.path.abspath(checkpoint), ckpt_link)
    return eval_dir


def already_done(eval_dir: str, split: str) -> bool:
    summary = os.path.join(eval_dir, split, "checkpoint_0_tracker", "pedestrian_summary.txt")
    return os.path.exists(summary)


def run_one(eval_dir: str, config_path: str, data_root: str, split: str, dry_run: bool):
    cmd = [
        sys.executable, "main.py",
        "--mode", "eval",
        "--config-path", config_path,
        "--eval-dir", eval_dir,
        "--eval-mode", "specific",
        "--eval-model", "checkpoint_0.pth",
        "--eval-data-split", split,
        "--data-root", data_root,
    ]
    print("  $ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="*", default=DEFAULT_LAMBDAS)
    ap.add_argument("--base-out", default=os.path.join(REPO_ROOT, "outputs", "b1"))
    ap.add_argument("--released-train-config",
                    default=os.path.join(REPO_ROOT, "outputs", "memotr_dancetrack", "train", "config.yaml"))
    ap.add_argument("--checkpoint",
                    default=os.path.join(REPO_ROOT, "checkpoints_pretrained", "memotr_dancetrack.pth"))
    ap.add_argument("--config-path", default=os.path.join(REPO_ROOT, "configs", "B1.yaml"))
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "dataset"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--force", action="store_true", help="re-run lambdas even if a summary already exists")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, do nothing")
    args = ap.parse_args()

    for p in (args.released_train_config, args.checkpoint, args.config_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: required path missing: {p}")

    os.makedirs(args.base_out, exist_ok=True)
    print(f"B1 sweep over lambda = {[lambda_tag(l) for l in args.lambdas]}")
    print(f"  released ckpt : {args.checkpoint}")
    print(f"  base out dir  : {args.base_out}\n")

    for lam in args.lambdas:
        eval_dir = build_lambda_dir(args.base_out, lam, args.released_train_config, args.checkpoint)
        print(f"[lambda={lambda_tag(lam)}] -> {eval_dir}")
        if already_done(eval_dir, args.split) and not args.force:
            print("  already evaluated, skipping (use --force to redo).\n")
            continue
        run_one(eval_dir, args.config_path, args.data_root, args.split, args.dry_run)
        print()

    print("Sweep dispatched. Summarize/plot with:")
    print(f"  python gating/analysis/plot_b1_sweep.py {args.base_out}")


if __name__ == "__main__":
    main()
