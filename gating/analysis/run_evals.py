#!/usr/bin/env python3
"""Run full-val eval on one or more checkpoints, consecutively.

Thin wrapper around:
    python main.py --mode eval --eval-mode specific --eval-model <ckpt> \
        --eval-dir <dir> --eval-data-split <split> --data-root <root> \
        --eval-threads <n> --config-path <cfg>

For each checkpoint it runs inference (submit) + TrackEval, writing
<eval-dir>/<split>/<ckpt>_tracker/pedestrian_summary.txt. Runs are sequential.

Usage (from repo root):
    python gating/analysis/run_evals.py checkpoint_3.pth
    python gating/analysis/run_evals.py checkpoint_3.pth checkpoint_4.pth
    python gating/analysis/run_evals.py 3 4                       # bare epoch ints ok
    python gating/analysis/run_evals.py checkpoint_4.pth --eval-dir outputs/b3b/seed_123
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_EVAL_DIR = os.path.join("outputs", "b3b", "seed_42")
DEFAULT_CONFIG = os.path.join("outputs", "b3b", "seed_42", "config_seed42.yaml")
DEFAULT_DATA_ROOT = "./dataset"
DEFAULT_SPLIT = "val"
DEFAULT_THREADS = 1


def normalize_ckpt(arg: str) -> str:
    """Accept 'checkpoint_3.pth', a full path, or a bare epoch int -> 'checkpoint_N.pth'.

    --eval-model must be a filename relative to --eval-dir, so we strip any path.
    """
    name = os.path.basename(arg)
    if name.isdigit():
        name = f"checkpoint_{name}.pth"
    if not name.endswith(".pth"):
        name += ".pth"
    return name


def main():
    ap = argparse.ArgumentParser(description="Run consecutive full-val evals on checkpoints.")
    ap.add_argument("checkpoints", nargs="+",
                    help="One or more checkpoints (filename, path, or bare epoch int).")
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    ap.add_argument("--config-path", default=DEFAULT_CONFIG)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--eval-data-split", default=DEFAULT_SPLIT)
    ap.add_argument("--eval-threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a checkpoint that already has a pedestrian_summary.txt.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpts = [normalize_ckpt(c) for c in args.checkpoints]
    print(f"[run_evals] {len(ckpts)} checkpoint(s): {', '.join(ckpts)}")

    for ckpt in ckpts:
        stem = ckpt[:-len(".pth")] if ckpt.endswith(".pth") else ckpt
        summary = os.path.join(REPO_ROOT, args.eval_dir, args.eval_data_split,
                               f"{stem}_tracker", "pedestrian_summary.txt")
        if args.skip_existing and os.path.exists(summary):
            print(f"[run_evals] {ckpt}: summary exists, skipping.")
            continue

        # Clear any stray tracker dir left by an aborted run (submit writes here,
        # then eval_engine moves it to <stem>_tracker).
        stray = os.path.join(REPO_ROOT, args.eval_dir, args.eval_data_split, "tracker")
        if os.path.isdir(stray) and not args.dry_run:
            shutil.rmtree(stray)

        cmd = [sys.executable, "main.py", "--mode", "eval", "--eval-mode", "specific",
               "--eval-model", ckpt,
               "--eval-dir", args.eval_dir,
               "--eval-data-split", args.eval_data_split,
               "--data-root", args.data_root,
               "--eval-threads", str(args.eval_threads),
               "--config-path", args.config_path]
        print(f"[run_evals] {ckpt}:\n  $ " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print("[run_evals] done. Pick the best epoch with:")
    print(f"  python gating/analysis/pick_best.py {args.eval_dir} --split {args.eval_data_split}")


if __name__ == "__main__":
    main()
