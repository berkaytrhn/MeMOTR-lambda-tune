"""F9 / T4: log short-term gate weights (w_t, w_{t-1}) during a val pass.

The B2/B4 symmetric gate is ``gating.shortterm_gate.SymmetricSoftmaxGate``;
its softmax over (current, previous) is the headline mechanism. If training
drives w_t ~= 1 the gate has *collapsed back to asymmetric* - that is the key
sanity check.

Approach: monkey-patch ``SymmetricSoftmaxGate.forward`` to record the softmax
weights on every call, then drive the project's standard submit-mode pass
over the val split (single GPU, single seq is enough for the histogram).
Results are dumped to ``<out>/_short_term_weights.npz`` and rendered as
``F9_short_term_weights.png``.

Args mirror ``scripts/dancetrack_test/run_test.sh`` so a B4 directory laid
out as ``<submit-dir>/train/config.yaml`` + ``<submit-dir>/<ckpt>.pth`` works
unchanged. Run with --submit-data-split val for the F9 figure.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

WEIGHTS: list[np.ndarray] = []


def install_hook() -> int:
    """Wrap SymmetricSoftmaxGate.forward to also record (w_t, w_{t-1}). Returns 1 if patched."""
    from gating.shortterm_gate import SymmetricSoftmaxGate

    orig = SymmetricSoftmaxGate.forward

    def patched(self, output_embed: torch.Tensor, last_output_embed: torch.Tensor) -> torch.Tensor:
        l_t = self.cur_mlp(output_embed)
        l_prev = self.prev_mlp(last_output_embed)
        weights = torch.softmax(torch.cat((l_t, l_prev), dim=-1), dim=-1)
        WEIGHTS.append(weights.detach().float().cpu().numpy().reshape(-1, 2))
        w_t = weights[:, 0:1]; w_prev = weights[:, 1:2]
        return w_t * output_embed + w_prev * last_output_embed

    SymmetricSoftmaxGate.forward = patched
    return 1


def build_config(args) -> dict:
    """Replicate the merged-config main.py builds for --mode submit."""
    from utils.utils import yaml_to_dict
    cfg = yaml_to_dict(args.config_path)
    cfg["MODE"] = "submit"
    cfg["USE_DISTRIBUTED"] = False
    cfg["AVAILABLE_GPUS"] = args.available_gpus
    cfg["DATA_ROOT"] = args.data_root
    cfg["SUBMIT_DIR"] = args.submit_dir
    cfg["SUBMIT_MODEL"] = args.submit_model
    cfg["SUBMIT_DATA_SPLIT"] = args.submit_data_split
    # Submit-mode thresholds: prefer values from the runtime cfg; supply documented
    # MeMOTR defaults if missing so the script also works with a training cfg.
    cfg.setdefault("DET_SCORE_THRESH",    0.5)
    cfg.setdefault("TRACK_SCORE_THRESH",  0.5)
    cfg.setdefault("RESULT_SCORE_THRESH", 0.7)
    cfg.setdefault("USE_MOTION",          False)
    cfg.setdefault("MOTION_MIN_LENGTH",   0)
    cfg.setdefault("MOTION_MAX_LENGTH",   0)
    cfg.setdefault("MOTION_LAMBDA",       0.0)
    cfg.setdefault("MISS_TOLERANCE",      30)
    return cfg


def restrict_seqs(cfg: dict, max_seqs: int | None):
    """Optionally truncate the dataset listing so a smoke pass over 1-3 sequences
    is enough to fill the histogram. Wraps os.listdir in submit_engine's scope."""
    if not max_seqs:
        return
    import submit_engine, os
    orig_listdir = os.listdir

    def limited_listdir(p):
        names = sorted(orig_listdir(p))
        # only truncate the val/test seq directory; pass other listdirs through.
        if any(seg in str(p) for seg in (cfg["SUBMIT_DATA_SPLIT"], "DanceTrack")):
            return names[:max_seqs]
        return names

    submit_engine.os.listdir = limited_listdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit-dir",        required=True,
                    help="Same as run_test.sh SUBMIT_DIR (contains train/config.yaml + .pth).")
    ap.add_argument("--submit-model",      required=True,
                    help="Checkpoint filename inside --submit-dir (e.g. checkpoint_4.pth).")
    ap.add_argument("--config-path",       default="configs/train_dancetrack.yaml")
    ap.add_argument("--data-root",         default="./dataset")
    ap.add_argument("--submit-data-split", default="val")
    ap.add_argument("--available-gpus",    default="0")
    ap.add_argument("--max-seqs",          type=int, default=3,
                    help="Limit number of val sequences (0 = all).")
    ap.add_argument("--out", type=Path,
                    default=Path("/workspace/persistent/results/_figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = install_hook()
    print(f"Patched {n} SymmetricSoftmaxGate.forward")

    cfg = build_config(args)
    restrict_seqs(cfg, args.max_seqs)

    from submit_engine import submit
    submit(config=cfg)

    if not WEIGHTS:
        print("No gate calls recorded - is this an asymmetric (B0/B3*) checkpoint?")
        return

    W = np.concatenate(WEIGHTS, axis=0)
    npz = args.out / "_short_term_weights.npz"
    np.savez(npz, w=W)
    print(f"Captured {len(W):,} gate calls -> {npz}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.hist(W[:, 0], bins=40, color="C0", alpha=0.7); ax1.set_title(r"$w_t$ (current)")
    ax2.hist(W[:, 1], bins=40, color="C3", alpha=0.7); ax2.set_title(r"$w_{t-1}$ (previous)")
    for ax in (ax1, ax2):
        ax.axvline(0.5, color="grey", ls="--", lw=0.8); ax.set_xlim(0, 1); ax.grid(alpha=0.3)
    fig.suptitle(f"F9: short-term gate weights  "
                 f"(mean $w_t$={W[:,0].mean():.3f}, %$w_t$>0.9 = {(W[:,0] > 0.9).mean()*100:.1f}%)")
    fig.tight_layout(); fig.savefig(args.out / "F9_short_term_weights.png", dpi=150)
    print(f"Wrote {args.out/'F9_short_term_weights.png'}")


if __name__ == "__main__":
    main()
