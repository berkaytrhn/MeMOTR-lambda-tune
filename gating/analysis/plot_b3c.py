#!/usr/bin/env python3
"""B3c analysis: learned signal->lambda response + HOTA-vs-epoch, across seeds.

B3c's lambda is a per-track function lambda = clamp(sigmoid(MLP(u_t))) of the
signal vector u_t = (c_t, s_t, delta_t, H_t), not a single scalar. So instead of
a scalar trajectory we visualize the LEARNED FUNCTION: how lambda responds to the
two wired signals (confidence c_t, cosine s_t), with the unwired delta_t/H_t held
at 0.

Reads, per seed dir <base>/seed_<S>/ :
  * checkpoint_{0..N-1}.pth   -> reconstructs the SignalGate MLP (net.0/net.2)
                                 and evaluates lambda on a (c_t, s_t) grid.
  * val/checkpoint_*_tracker/pedestrian_summary.txt  (if EVAL_MODE=continue ran)
                              -> HOTA / AssA per epoch, to pick the best epoch.

Outputs (written under <base>/):
  * b3c_lambda_response.png   lambda heatmap over (c_t, s_t) for each seed's best
                              checkpoint, with the released fixed lambda=0.01 noted.
  * b3c_hota_vs_epoch.png     HOTA (and AssA) per seed vs epoch.
  * prints best-by-HOTA epoch per seed, and B3a/B3b best HOTA if present (H3).

Usage:
    python gating/analysis/plot_b3c.py outputs/b3c
"""
import argparse
import glob
import math
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
import numpy as np  # noqa: E402


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


def load_gate_mlp(state_dict):
    """Pull the SignalGate MLP tensors (4->16->1) out of a checkpoint."""
    w = {}
    for k, v in state_dict.items():
        if ".lt_gate.net." in k:
            short = k.split(".lt_gate.net.")[-1]  # e.g. "0.weight"
            w[short] = v.float()
    if not w:
        raise KeyError("no '*lt_gate.net.*' keys — is this a B3c (signal) run?")
    return w


def lambda_grid(w, clip, n=60):
    """lambda over a (c_t in [0,1]) x (s_t in [-1,1]) grid; delta_t=H_t=0."""
    cs = np.linspace(0.0, 1.0, n)
    ss = np.linspace(-1.0, 1.0, n)
    C, S = np.meshgrid(cs, ss)
    u = np.stack([C.ravel(), S.ravel(), np.zeros(C.size), np.zeros(C.size)], axis=-1)
    u = torch.from_numpy(u).float()
    h = torch.relu(u @ w["0.weight"].t() + w["0.bias"])
    out = h @ w["2.weight"].t() + w["2.bias"]
    lam = torch.sigmoid(out).clamp(clip[0], clip[1]).reshape(n, n).numpy()
    return cs, ss, lam


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


def ckpt_for_epoch(seed_dir: str, epoch):
    if epoch is None:
        cks = sorted(glob.glob(os.path.join(seed_dir, "checkpoint_*.pth")),
                     key=lambda p: int(re.search(r"checkpoint_(\d+)\.pth", p).group(1)))
        return cks[-1] if cks else None
    p = os.path.join(seed_dir, f"checkpoint_{epoch}.pth")
    return p if os.path.exists(p) else None


def best_hota_in(base: str, split: str):
    best = None
    for sd in sorted(glob.glob(os.path.join(base, "seed_*"))):
        _, hota, _ = hota_trajectory(sd, split)
        if hota:
            h = max(hota)
            if best is None or h > best:
                best = h
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="b3c base dir, e.g. outputs/b3c")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.base, "seed_*")))
    if not seed_dirs:
        print(f"no seed_* dirs under {args.base}")
        return

    fig_h, ax_h = plt.subplots(figsize=(6, 4))
    fig_l, axes_l = plt.subplots(1, len(seed_dirs), figsize=(5 * len(seed_dirs), 4), squeeze=False)
    print(f"{'seed':>6} {'best_epoch':>11} {'best_HOTA':>10} {'lambda@(c=.9,s=.9)':>20}")
    for col, sd in enumerate(seed_dirs):
        seed = os.path.basename(sd).replace("seed_", "")
        clip = read_clip(sd)

        hep, hota, assa = hota_trajectory(sd, args.split)
        best_epoch, best_hota = (None, float("nan"))
        if hota:
            bi = max(range(len(hota)), key=lambda i: hota[i])
            best_epoch, best_hota = hep[bi], hota[bi]
            ax_h.plot(hep, hota, marker="o", label=f"seed {seed} HOTA")
            ax_h.plot(hep, assa, marker="x", ls="--", label=f"seed {seed} AssA")

        ckpt = ckpt_for_epoch(sd, best_epoch)
        lam_hi = float("nan")
        ax = axes_l[0][col]
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            sdic = state["model"] if isinstance(state, dict) and "model" in state else state
            w = load_gate_mlp(sdic)
            cs, ss, lam = lambda_grid(w, clip)
            im = ax.pcolormesh(cs, ss, lam, shading="auto")
            fig_l.colorbar(im, ax=ax, label="learned λ")
            ax.set_xlabel("confidence c_t")
            ax.set_ylabel("cosine s_t")
            ax.set_title(f"seed {seed} (epoch {best_epoch}) — released λ=0.01")
            # nearest grid point to (c=0.9, s=0.9)
            ci = int(np.argmin(np.abs(cs - 0.9)))
            si = int(np.argmin(np.abs(ss - 0.9)))
            lam_hi = float(lam[si, ci])
        print(f"{seed:>6} {str(best_epoch):>11} {best_hota:>10.3f} {lam_hi:>20.5f}")

    ax_h.set_xlabel("epoch (checkpoint index)")
    ax_h.set_ylabel("full-val metric")
    ax_h.set_title("B3c: full-val HOTA / AssA vs epoch")
    ax_h.legend(fontsize=8)
    fig_h.tight_layout()
    out_h = os.path.join(args.base, "b3c_hota_vs_epoch.png")
    fig_h.savefig(out_h, dpi=150)

    fig_l.suptitle("B3c: learned λ response to (c_t, s_t), δ_t=H_t=0")
    fig_l.tight_layout()
    out_l = os.path.join(args.base, "b3c_lambda_response.png")
    fig_l.savefig(out_l, dpi=150)

    print(f"\nwrote {out_l}\nwrote {out_h}")

    repo_root = os.path.abspath(os.path.join(args.base, "..", ".."))
    for tag in ("b3a", "b3b"):
        b = best_hota_in(os.path.join(repo_root, "outputs", tag), args.split)
        if b is not None:
            print(f"H3 comparison — {tag.upper()} best-epoch HOTA = {b:.3f} "
                  f"(compare against B3c best-epoch HOTA above)")


if __name__ == "__main__":
    main()
