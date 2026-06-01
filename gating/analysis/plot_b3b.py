#!/usr/bin/env python3
"""B3b analysis: per-channel learned-lambda distribution + HOTA-vs-epoch, across seeds.

B3b's gate is a vector theta in R^256, so "lambda" is a 256-dim per-channel
vector rather than B3a's single scalar. Reads, per seed dir <base>/seed_<S>/ :
  * checkpoint_{0..N-1}.pth   -> extracts the vector gate theta, computes
                                 lambda = clamp(sigmoid(theta), MIN, MAX) per
                                 channel, giving the lambda DISTRIBUTION's
                                 trajectory (mean and 10-90 percentile band)
                                 over epochs.
  * val/checkpoint_*_tracker/pedestrian_summary.txt  (if EVAL_MODE=continue ran)
                              -> HOTA / AssA per epoch, to pick the best epoch.

Outputs (written under <base>/):
  * b3b_lambda_trajectory.png   per-seed mean per-channel lambda vs epoch, with a
                                10-90 percentile band, plus released lambda=0.01 ref.
  * b3b_lambda_hist.png         histogram of the final per-channel lambda (last
                                checkpoint) per seed -- shows channel spread.
  * b3b_hota_vs_epoch.png       HOTA (and AssA) per seed vs epoch.
  * prints final lambda summary (mean/min/max/std over channels) per seed and the
    best-by-HOTA epoch per seed.
  * if outputs/b1 exists, prints B1's best fixed-lambda HOTA for the H1 comparison.

Usage:
    python gating/analysis/plot_b3b.py outputs/b3b
"""
import argparse
import glob
import os
import re

import torch

try:
    import yaml
except Exception:
    yaml = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_theta_vec(state_dict) -> torch.Tensor:
    for k, v in state_dict.items():
        if k.endswith("lt_gate.theta"):
            return v.reshape(-1).float()
    raise KeyError("no '*lt_gate.theta' key in checkpoint — is this a B3b (vector) run?")


def lam_from_theta(theta: torch.Tensor, clip=(1e-3, 0.5)) -> torch.Tensor:
    return torch.sigmoid(theta).clamp(clip[0], clip[1])


def read_clip(seed_dir: str):
    clip = (1e-3, 0.5)
    if yaml is None:
        return clip
    cands = glob.glob(os.path.join(seed_dir, "config_seed*.yaml")) + \
        [os.path.join(seed_dir, "train", "config.yaml")]
    for c in cands:
        if os.path.exists(c):
            with open(c) as f:
                cfg = yaml.safe_load(f)
            return (float(cfg.get("LAMBDA_CLIP_MIN", 1e-3)), float(cfg.get("LAMBDA_CLIP_MAX", 0.5)))
    return clip


def lambda_trajectory(seed_dir: str):
    """Return epochs and a list of per-channel lambda tensors (one per epoch)."""
    clip = read_clip(seed_dir)
    epochs, lam_vecs = [], []
    for path in sorted(glob.glob(os.path.join(seed_dir, "checkpoint_*.pth")),
                       key=lambda p: int(re.search(r"checkpoint_(\d+)\.pth", p).group(1))):
        epoch = int(re.search(r"checkpoint_(\d+)\.pth", path).group(1))
        state = torch.load(path, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        epochs.append(epoch)
        lam_vecs.append(lam_from_theta(find_theta_vec(sd), clip))
    return epochs, lam_vecs


def read_summary(path: str) -> dict:
    with open(path) as f:
        names = f.readline().strip().split(" ")
        values = f.readline().strip().split(" ")
    return {n: float(v) for n, v in zip(names, values)}


def hota_trajectory(seed_dir: str, split: str):
    epochs, hota, assa = [], [], []
    pattern = os.path.join(seed_dir, split, "checkpoint_*_tracker", "pedestrian_summary.txt")
    for path in sorted(glob.glob(pattern),
                       key=lambda p: int(re.search(r"checkpoint_(\d+)_tracker", p).group(1))):
        epoch = int(re.search(r"checkpoint_(\d+)_tracker", path).group(1))
        m = read_summary(path)
        epochs.append(epoch)
        hota.append(m.get("HOTA", float("nan")))
        assa.append(m.get("AssA", float("nan")))
    return epochs, hota, assa


def b1_best_hota(repo_root: str, split: str):
    best = None
    pattern = os.path.join(repo_root, "outputs", "b1", "lambda_*", split,
                           "checkpoint_0_tracker", "pedestrian_summary.txt")
    for path in glob.glob(pattern):
        tag = re.search(r"lambda_([^/]+)", path).group(1)
        h = read_summary(path).get("HOTA", float("-inf"))
        if best is None or h > best[1]:
            best = (tag, h)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="b3b base dir, e.g. outputs/b3b")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.base, "seed_*")))
    if not seed_dirs:
        print(f"no seed_* dirs under {args.base}")
        return

    fig_l, ax_l = plt.subplots(figsize=(6, 4))
    fig_h, ax_h = plt.subplots(figsize=(6, 4))
    fig_d, ax_d = plt.subplots(figsize=(6, 4))
    print(f"{'seed':>6} {'final_lam_mean':>15} {'final_lam_min':>14} "
          f"{'final_lam_max':>14} {'final_lam_std':>14} {'best_epoch':>11} {'best_HOTA':>10}")
    for sd in seed_dirs:
        seed = os.path.basename(sd).replace("seed_", "")
        ep, lam_vecs = lambda_trajectory(sd)
        if lam_vecs:
            means = [v.mean().item() for v in lam_vecs]
            lo = [v.quantile(0.10).item() for v in lam_vecs]
            hi = [v.quantile(0.90).item() for v in lam_vecs]
            line, = ax_l.plot(ep, means, marker="o", label=f"seed {seed} (mean)")
            ax_l.fill_between(ep, lo, hi, alpha=0.20, color=line.get_color())
            final = lam_vecs[-1]
            ax_d.hist(final.numpy(), bins=40, alpha=0.5, label=f"seed {seed}")
            fmean, fmin, fmax, fstd = (final.mean().item(), final.min().item(),
                                       final.max().item(), final.std().item())
        else:
            fmean = fmin = fmax = fstd = float("nan")

        hep, hota, assa = hota_trajectory(sd, args.split)
        best_epoch, best_hota = (None, float("nan"))
        if hota:
            bi = max(range(len(hota)), key=lambda i: hota[i])
            best_epoch, best_hota = hep[bi], hota[bi]
            ax_h.plot(hep, hota, marker="o", label=f"seed {seed} HOTA")
        print(f"{seed:>6} {fmean:>15.5f} {fmin:>14.5f} {fmax:>14.5f} {fstd:>14.5f} "
              f"{str(best_epoch):>11} {best_hota:>10.3f}")

    ax_l.axhline(0.01, color="k", ls="--", lw=1, label="released λ=0.01")
    ax_l.set_xlabel("epoch (checkpoint index)")
    ax_l.set_ylabel("per-channel λ = clamp(σ(θ))")
    ax_l.set_title("B3b: per-channel λ trajectory (mean, 10-90% band)")
    ax_l.legend()
    fig_l.tight_layout()
    out_l = os.path.join(args.base, "b3b_lambda_trajectory.png")
    fig_l.savefig(out_l, dpi=150)

    ax_d.axvline(0.01, color="k", ls="--", lw=1, label="released λ=0.01")
    ax_d.set_xlabel("final per-channel λ")
    ax_d.set_ylabel("channel count")
    ax_d.set_title("B3b: final per-channel λ distribution")
    ax_d.legend()
    fig_d.tight_layout()
    out_d = os.path.join(args.base, "b3b_lambda_hist.png")
    fig_d.savefig(out_d, dpi=150)

    ax_h.set_xlabel("epoch (checkpoint index)")
    ax_h.set_ylabel("HOTA (full val)")
    ax_h.set_title("B3b: full-val HOTA vs epoch")
    ax_h.legend()
    fig_h.tight_layout()
    out_h = os.path.join(args.base, "b3b_hota_vs_epoch.png")
    fig_h.savefig(out_h, dpi=150)

    print(f"\nwrote {out_l}\nwrote {out_d}\nwrote {out_h}")

    repo_root = os.path.abspath(os.path.join(args.base, "..", ".."))
    b1 = b1_best_hota(repo_root, args.split)
    if b1:
        print(f"\nH1 comparison — B1 best fixed λ: λ={b1[0]}  HOTA={b1[1]:.3f}  "
              f"(compare against B3b best-epoch HOTA above)")


if __name__ == "__main__":
    main()
