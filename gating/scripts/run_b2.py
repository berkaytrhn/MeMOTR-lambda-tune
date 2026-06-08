#!/usr/bin/env python3
"""B2 — short-term symmetric gate: train + per-checkpoint full-val driver.

For each seed it:
  1. writes a per-seed config (configs/B2.yaml with SEED + OUTPUTS_DIR patched)
     into  outputs/b2/seed_<S>/config_seed<S>.yaml,
  2. trains EPOCHS epochs  (python main.py --mode train ...). Training saves
     checkpoint_{0..EPOCHS-1}.pth and its own train/config.yaml into the seed dir,
  3. runs full-val on ALL checkpoints  (python main.py --mode eval, EVAL_MODE=continue).
     submit() rebuilds the (symmetric-gate) model from <seed_dir>/train/config.yaml,
     so no extra wiring is needed; TrackEval writes pedestrian_summary.txt per ckpt.

B2 trains for 10 epochs over a single seed (123) per the user's instruction, keeping
the ORIGINAL released DanceTrack training hyperparameters (LR 2e-4, MultiStep
scheduler, LR-drop, weight decay, grad-clip) — only EPOCHS (10) and the seed count
change vs. the paper, plus the symmetric short-term gate.

Pick the best epoch and plot the w_t / w_{t-1} weight distributions with:
    python gating/analysis/plot_b2.py outputs/b2

ALWAYS run the gradient check first (one-time, any seed):
    python gating/scripts/grad_check_b2.py

Idempotent: a seed whose final checkpoint already exists skips training (unless
--force); eval/continue itself skips checkpoints already summarized.

Usage (from repo root):
    python gating/scripts/run_b2.py                       # seed 123
    python gating/scripts/run_b2.py --seeds 123
    python gating/scripts/run_b2.py --skip-grad-check
    python gating/scripts/run_b2.py --dry-run
"""
import argparse
import os
import subprocess
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SEEDS = [123]


def write_seed_config(base_config: str, seed: int, seed_dir: str):
    os.makedirs(seed_dir, exist_ok=True)
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    cfg["SEED"] = int(seed)
    cfg["OUTPUTS_DIR"] = seed_dir + ("" if seed_dir.endswith("/") else "/")
    out = os.path.join(seed_dir, f"config_seed{seed}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out, cfg


def run(cmd, dry_run: bool):
    print("  $ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--base-out", default=os.path.join(REPO_ROOT, "outputs", "b2"))
    ap.add_argument("--config-path", default=os.path.join(REPO_ROOT, "configs", "B2.yaml"))
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "dataset"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--skip-grad-check", action="store_true")
    ap.add_argument("--skip-eval", action="store_true", help="train only, no per-checkpoint val")
    ap.add_argument("--force", action="store_true", help="retrain even if final checkpoint exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in (args.config_path, args.data_root):
        if not os.path.exists(p):
            sys.exit(f"ERROR: required path missing: {p}")

    # Mandatory grad check ONCE before any training.
    if not args.skip_grad_check:
        print("[B2] running mandatory gradient check before training ...")
        run([sys.executable, os.path.join("gating", "scripts", "grad_check_b2.py"),
             "--config-path", args.config_path, "--data-root", args.data_root], args.dry_run)
        print()

    os.makedirs(args.base_out, exist_ok=True)
    for seed in args.seeds:
        seed_dir = os.path.join(args.base_out, f"seed_{seed}")
        cfg_path, cfg = write_seed_config(args.config_path, seed, seed_dir)
        epochs = int(cfg["EPOCHS"])
        final_ckpt = os.path.join(seed_dir, f"checkpoint_{epochs - 1}.pth")
        print(f"[seed={seed}] dir={seed_dir}  epochs={epochs}  config={cfg_path}")

        # 1) train
        if os.path.exists(final_ckpt) and not args.force:
            print(f"  final checkpoint exists ({final_ckpt}); skipping training (use --force).")
        else:
            run([sys.executable, "main.py", "--mode", "train",
                 "--config-path", cfg_path,
                 "--data-root", args.data_root], args.dry_run)

        # 2) full-val on all checkpoints
        if not args.skip_eval:
            run([sys.executable, "main.py", "--mode", "eval",
                 "--config-path", cfg_path,
                 "--eval-dir", seed_dir,
                 "--eval-mode", "continue",
                 "--eval-data-split", args.split,
                 "--data-root", args.data_root], args.dry_run)
        print()

    print("Done. Inspect results with:")
    print(f"  python gating/analysis/plot_b2.py {args.base_out}")
    for seed in args.seeds:
        print(f"  python gating/analysis/pick_best.py {os.path.join(args.base_out, f'seed_{seed}')} --split {args.split}")


if __name__ == "__main__":
    main()
