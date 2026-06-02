#!/usr/bin/env python3
"""B2 analysis: short-term symmetric-gate weight distributions + HOTA-vs-epoch.

B2's gate emits, per track per frame, a convex pair (w_t, w_{t-1}) over the
current and previous output embeddings (short_memory = w_t*O_t + w_{t-1}*O_{t-1}).
The released MeMOTR fusion is asymmetric (O_{t-1} ungated), so the key question
is whether the learned symmetric gate keeps a non-trivial w_{t-1} or COLLAPSES
back to w_t ~ 1 (which would mean it rediscovered the asymmetric behavior — a
real, reportable finding).

Reads, per seed dir <base>/seed_<S>/ :
  * <best HOTA checkpoint>.pth                  -> runs the model on a few training
                                                   clips, collecting (w_t, w_{t-1})
                                                   over all active tracks/frames.
  * val/checkpoint_*_tracker/pedestrian_summary.txt  (if EVAL_MODE=continue ran)
                                                -> HOTA / AssA per epoch, best epoch.

Outputs (written under <base>/):
  * b2_weight_hist.png      histogram of w_t (and w_{t-1}) at the best checkpoint
                            per seed, with a reference line at w_t = 1 (asymmetric
                            collapse) and the released asymmetric regime annotated.
  * b2_hota_vs_epoch.png    HOTA (and AssA) per seed vs epoch.
  * prints, per seed: best-by-HOTA epoch + full metric row; mean/std of w_t and
    w_{t-1}; the fraction of tracks with w_t > 0.9 (collapse indicator); a verdict.
  * if a B0 summary is found, prints B0's val HOTA for the comparison.

Usage:
    python gating/analysis/plot_b2.py outputs/b2
    python gating/analysis/plot_b2.py outputs/b2 --no-weights   # HOTA only (no GPU)
    python gating/analysis/plot_b2.py outputs/b2 --clips 8
"""
import argparse
import glob
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import yaml
except Exception:
    yaml = None


# --------------------------------------------------------------------------- #
# HOTA-vs-epoch (reads TrackEval summaries; no GPU needed)
# --------------------------------------------------------------------------- #
def read_summary(path: str) -> dict:
    with open(path) as f:
        names = f.readline().strip().split(" ")
        values = f.readline().strip().split(" ")
    return {n: float(v) for n, v in zip(names, values)}


def read_hota_rows(seed_dir: str, split: str = "val"):
    pattern = os.path.join(seed_dir, split, "checkpoint_*_tracker", "pedestrian_summary.txt")
    rows = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"checkpoint_(\d+)_tracker", path)
        epoch = int(m.group(1)) if m else -1
        rows.append((epoch, read_summary(path)))
    rows.sort(key=lambda r: r[0])
    return rows


def best_epoch(rows, metric="HOTA"):
    if not rows:
        return None
    return max(rows, key=lambda r: r[1].get(metric, float("-inf")))


# --------------------------------------------------------------------------- #
# Weight collection (runs the model on a few training clips)
# --------------------------------------------------------------------------- #
def collect_weights(seed_dir: str, ckpt_path: str, clips: int, frames: int):
    """Run the B2 model on `clips` training clips and return numpy arrays of the
    per-track softmax weights (w_t, w_prev) the symmetric gate emitted."""
    import numpy as np
    import torch
    from utils.utils import yaml_to_dict, set_seed
    from utils.nested_tensor import tensor_list_to_nested_tensor
    from models import build_model
    from models.utils import get_model
    from data import build_dataset, build_sampler, build_dataloader
    from models.criterion import build as build_criterion
    from structures.track_instances import TrackInstances

    cfg_path = os.path.join(seed_dir, "train", "config.yaml")
    if not os.path.exists(cfg_path):
        cands = glob.glob(os.path.join(seed_dir, "config_seed*.yaml"))
        if not cands:
            raise FileNotFoundError(f"no train/config.yaml or config_seed*.yaml under {seed_dir}")
        cfg_path = cands[0]
    config = yaml_to_dict(cfg_path)
    config["DATA_ROOT"] = os.path.join(REPO_ROOT, "dataset")
    config["PRETRAINED_MODEL"] = None  # we load the trained checkpoint below
    # Deterministic, multi-track clips so the gate is exercised broadly.
    config["SAMPLE_LENGTHS"] = [int(frames)]
    config["SAMPLE_STEPS"] = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("AVAILABLE_GPUS", "0"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config.get("SEED", 42)))

    model = build_model(config=config).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    get_model(model).load_state_dict(state["model"] if "model" in state else state)
    model.train()  # training select_active_tracks => realistic multi-track clips

    st_gate = get_model(model).query_updater.st_gate
    assert st_gate is not None, f"{ckpt_path} has no st_gate — not a B2 checkpoint?"

    w_t_all, w_prev_all = [], []

    def hook(_m, inputs, _o):
        o_t, o_prev = inputs[0], inputs[1]
        l_t = st_gate.cur_mlp(o_t)
        l_prev = st_gate.prev_mlp(o_prev)
        weights = torch.softmax(torch.cat((l_t, l_prev), dim=-1), dim=-1)
        w_t_all.append(weights[:, 0].detach().cpu())
        w_prev_all.append(weights[:, 1].detach().cpu())

    handle = st_gate.register_forward_hook(hook)

    dataset = build_dataset(config=config, split="train")
    sampler = build_sampler(dataset=dataset, shuffle=True)
    dataloader = build_dataloader(dataset=dataset, sampler=sampler, batch_size=1, num_workers=0)
    criterion = build_criterion(config=config)
    criterion.set_device(device)

    it = iter(dataloader)
    with torch.no_grad():
        for _ in range(int(clips)):
            try:
                batch = next(it)
            except StopIteration:
                break
            tracks = TrackInstances.init_tracks(
                batch=batch, hidden_dim=get_model(model).hidden_dim,
                num_classes=get_model(model).num_classes, device=device, use_dab=config["USE_DAB"])
            criterion.init_a_clip(batch=batch, hidden_dim=get_model(model).hidden_dim,
                                  num_classes=get_model(model).num_classes, device=device)
            n_frames = len(batch["imgs"][0])
            for frame_idx in range(n_frames):
                frame = [fs[frame_idx] for fs in batch["imgs"]]
                frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
                res = model(frame=frame, tracks=tracks)
                previous_tracks, new_tracks, unmatched_dets = criterion.process_single_frame(
                    model_outputs=res, tracked_instances=tracks, frame_idx=frame_idx)
                if frame_idx < n_frames - 1:
                    tracks = get_model(model).postprocess_single_frame(
                        previous_tracks, new_tracks, unmatched_dets)

    handle.remove()
    if not w_t_all:
        return np.array([]), np.array([])
    return torch.cat(w_t_all).numpy(), torch.cat(w_prev_all).numpy()


# --------------------------------------------------------------------------- #
def find_b0_hota():
    """Best-effort: look for a B0 val summary to print alongside B2."""
    for pat in ("outputs/b0", "outputs/memotr_dancetrack"):
        for path in sorted(glob.glob(os.path.join(REPO_ROOT, pat, "val", "checkpoint_*_tracker",
                                                  "pedestrian_summary.txt"))):
            try:
                return read_summary(path).get("HOTA")
            except Exception:
                pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="B2 base output dir (contains seed_<S>/)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clips", type=int, default=6, help="training clips to sample for weight stats")
    ap.add_argument("--frames", type=int, default=5, help="clip length used for weight collection")
    ap.add_argument("--no-weights", action="store_true", help="skip the model forward (HOTA only)")
    args = ap.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.base, "seed_*")))
    if not seed_dirs:
        sys.exit(f"no seed_* dirs under {args.base}")

    b0_hota = find_b0_hota()

    # ---- HOTA-vs-epoch -----------------------------------------------------
    fig_h, ax_h = plt.subplots(figsize=(7, 5))
    any_hota = False
    print("=" * 72)
    for sd in seed_dirs:
        seed = os.path.basename(sd).replace("seed_", "")
        rows = read_hota_rows(sd, args.split)
        if not rows:
            print(f"[seed {seed}] no TrackEval summaries yet under {sd}/{args.split}/")
            continue
        any_hota = True
        epochs = [e for e, _ in rows]
        hota = [m.get("HOTA", float("nan")) for _, m in rows]
        assa = [m.get("AssA", float("nan")) for _, m in rows]
        ax_h.plot(epochs, hota, "-o", label=f"seed {seed} HOTA")
        ax_h.plot(epochs, assa, "--s", label=f"seed {seed} AssA", alpha=0.6)
        be = best_epoch(rows, "HOTA")
        cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
        print(f"[seed {seed}] per-epoch:")
        print(f"  {'epoch':>5}  " + "  ".join(f"{c:>7}" for c in cols))
        for e, m in rows:
            mark = "  <- best" if e == be[0] else ""
            print(f"  {e:>5}  " + "  ".join(f"{m.get(c, float('nan')):>7.3f}" for c in cols) + mark)
        print(f"  BEST epoch {be[0]}: " + ", ".join(f"{c}={be[1].get(c, float('nan')):.3f}" for c in cols))
    if b0_hota is not None:
        ax_h.axhline(b0_hota, color="k", ls=":", label=f"B0 HOTA={b0_hota:.2f}")
        print(f"\n[B0] val HOTA = {b0_hota:.3f} (anchor for the B2 comparison)")
    if any_hota:
        ax_h.set_xlabel("epoch"); ax_h.set_ylabel("score")
        ax_h.set_title("B2 short-term symmetric gate — HOTA/AssA vs epoch")
        ax_h.legend(fontsize=8); ax_h.grid(True, alpha=0.3)
        out = os.path.join(args.base, "b2_hota_vs_epoch.png")
        fig_h.tight_layout(); fig_h.savefig(out, dpi=130); print(f"\nwrote {out}")
    plt.close(fig_h)

    # ---- weight distributions ---------------------------------------------
    if args.no_weights:
        print("\n[--no-weights] skipped the model forward; weight histograms not produced.")
        return

    fig_w, ax_w = plt.subplots(figsize=(7, 5))
    produced = False
    print("\n" + "=" * 72)
    for sd in seed_dirs:
        seed = os.path.basename(sd).replace("seed_", "")
        rows = read_hota_rows(sd, args.split)
        if rows:
            be = best_epoch(rows, "HOTA")[0]
            ckpt = os.path.join(sd, f"checkpoint_{be}.pth")
        else:
            ckpts = sorted(glob.glob(os.path.join(sd, "checkpoint_*.pth")),
                           key=lambda p: int(re.search(r"checkpoint_(\d+)", p).group(1)))
            if not ckpts:
                print(f"[seed {seed}] no checkpoints; skipping weight stats.")
                continue
            be, ckpt = int(re.search(r"checkpoint_(\d+)", ckpts[-1]).group(1)), ckpts[-1]
        if not os.path.exists(ckpt):
            print(f"[seed {seed}] checkpoint {ckpt} missing; skipping weight stats.")
            continue
        try:
            w_t, w_prev = collect_weights(sd, ckpt, args.clips, args.frames)
        except Exception as e:
            print(f"[seed {seed}] weight collection failed: {e}")
            continue
        if w_t.size == 0:
            print(f"[seed {seed}] no tracks exercised the gate; no weights collected.")
            continue
        produced = True
        frac_collapse = float((w_t > 0.9).mean())
        verdict = ("COLLAPSED toward asymmetric (w_t~1)" if frac_collapse > 0.9
                   else "non-trivial symmetric weighting")
        print(f"[seed {seed}] best ckpt=checkpoint_{be}.pth  n={w_t.size} track-frames")
        print(f"  w_t   : mean={w_t.mean():.4f}  std={w_t.std():.4f}  "
              f"median={float(__import__('numpy').median(w_t)):.4f}")
        print(f"  w_t-1 : mean={w_prev.mean():.4f}  std={w_prev.std():.4f}")
        print(f"  fraction w_t > 0.9 = {frac_collapse:.3f}  ->  {verdict}")
        ax_w.hist(w_t, bins=40, range=(0, 1), alpha=0.5, label=f"seed {seed} w_t")
        ax_w.hist(w_prev, bins=40, range=(0, 1), alpha=0.3, label=f"seed {seed} w_t-1", histtype="step")

    if produced:
        ax_w.axvline(1.0, color="r", ls=":", label="w_t=1 (asymmetric collapse)")
        ax_w.set_xlabel("softmax weight"); ax_w.set_ylabel("count (track-frames)")
        ax_w.set_title("B2 symmetric gate — distribution of w_t and w_{t-1}")
        ax_w.legend(fontsize=8); ax_w.grid(True, alpha=0.3)
        out = os.path.join(args.base, "b2_weight_hist.png")
        fig_w.tight_layout(); fig_w.savefig(out, dpi=130); print(f"\nwrote {out}")
    plt.close(fig_w)


if __name__ == "__main__":
    main()
