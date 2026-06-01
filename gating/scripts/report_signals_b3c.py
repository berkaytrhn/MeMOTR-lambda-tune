#!/usr/bin/env python3
"""B3c signal-distribution report (IMPLEMENTATION_PLAN B3c requirement).

Before relying on B3c results, the plan says to extract the gate's input
signals on a slice of train, report their distributions to the human, and
validate their ranges. This script runs the (frozen, pretrained) B3c model over
a handful of train clips, captures every signal vector u_t the SignalGate would
consume, and prints per-signal stats + range checks. It also dumps the mean/std
the plan recommends using for normalization, and saves a histogram figure.

Only c_t (confidence) and s_t (cosine sim) are wired; delta_t and H_t are zero
placeholders (the extractor raises if asked to fabricate them), so they report
as constant 0 here — that is expected and flagged. Wiring those two needs human
go-ahead (see configs/B3c.yaml and gating/signals.py).

Usage (from repo root, GPU visible):
    python gating/scripts/report_signals_b3c.py
    python gating/scripts/report_signals_b3c.py --clips 30
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils.utils import yaml_to_dict, set_seed                       # noqa: E402
from utils.nested_tensor import tensor_list_to_nested_tensor          # noqa: E402
from models import build_model                                        # noqa: E402
from models.utils import get_model, load_pretrained_model             # noqa: E402
from data import build_dataset, build_sampler, build_dataloader       # noqa: E402
from models.criterion import build as build_criterion                 # noqa: E402
from structures.track_instances import TrackInstances                 # noqa: E402
from gating import signals as gating_signals                          # noqa: E402

SIGNAL_NAMES = gating_signals.SIGNAL_NAMES
# Plausible ranges from the plan (B3c spec) for a sanity check.
EXPECTED = {
    "c_t": (0.0, 1.0),       # confidence is a probability
    "s_t": (-1.0, 1.0),      # cosine similarity (plan expects mostly 0.5-1.0)
    "delta_t": (0.0, 0.0),   # not wired -> constant 0
    "H_t": (0.0, 0.0),       # not wired -> constant 0
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", default=os.path.join(REPO_ROOT, "configs", "B3c.yaml"))
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "dataset"))
    ap.add_argument("--clips", type=int, default=20, help="number of train clips to scan")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "b3c"))
    args = ap.parse_args()

    config = yaml_to_dict(args.config_path)
    config["DATA_ROOT"] = args.data_root
    config["SAMPLE_LENGTHS"] = [5]
    config["SAMPLE_STEPS"] = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("AVAILABLE_GPUS", "0"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["SEED"])
    assert config["LONG_TERM_GATE"] == "signal", "B3c needs LONG_TERM_GATE: signal"

    model = build_model(config=config).to(device)
    if config.get("PRETRAINED_MODEL"):
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False)
    model.train()  # training forward path exercises the gate (update_tracks_embedding)

    # Capture every u_t the gate consumes by wrapping build_signals.
    captured = []
    orig_build = gating_signals.build_signals

    def capture(*a, **kw):
        u = orig_build(*a, **kw)
        captured.append(u.detach().float().cpu())
        return u

    gating_signals.build_signals = capture

    dataset = build_dataset(config=config, split="train")
    sampler = build_sampler(dataset=dataset, shuffle=True)
    dataloader = build_dataloader(dataset=dataset, sampler=sampler, batch_size=1, num_workers=0)
    criterion = build_criterion(config=config)
    criterion.set_device(device)

    it = iter(dataloader)
    with torch.no_grad():
        for clip_idx in range(args.clips):
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

    gating_signals.build_signals = orig_build

    if not captured:
        sys.exit("No signal vectors captured — no active tracks in the scanned clips?")

    U = torch.cat(captured, dim=0)  # (total_tracks, 4)
    print(f"\nB3c signal report — {U.shape[0]} (track,frame) samples over up to {args.clips} clips\n")
    print(f"{'signal':>8} {'mean':>9} {'std':>9} {'min':>9} {'p5':>9} {'p50':>9} {'p95':>9} "
          f"{'max':>9}  range_ok")
    means, stds = [], []
    for j, name in enumerate(SIGNAL_NAMES):
        col = U[:, j]
        q = torch.quantile(col, torch.tensor([0.05, 0.5, 0.95]))
        lo, hi = EXPECTED[name]
        ok = bool((col.min() >= lo - 1e-4) and (col.max() <= hi + 1e-4))
        means.append(float(col.mean()))
        stds.append(float(col.std()))
        flag = "OK" if ok else "OUT-OF-RANGE"
        if name in ("delta_t", "H_t"):
            flag = "0 (NOT WIRED)"
        print(f"{name:>8} {col.mean():>9.4f} {col.std():>9.4f} {col.min():>9.4f} "
              f"{q[0]:>9.4f} {q[1]:>9.4f} {q[2]:>9.4f} {col.max():>9.4f}  {flag}")

    print("\nNormalization stats (for gating.signals.build_signals mean/std, if used):")
    print(f"  mean = {[round(m, 5) for m in means]}")
    print(f"  std  = {[round(s, 5) for s in stds]}")
    print("\nNOTE: delta_t and H_t are constant 0 (not wired). The SignalGate starts at "
          "lambda=0.01 regardless (final weight zeroed, bias -4.595), so the unwired "
          "signals do not bias the init. Wiring them needs human go-ahead (plan B3c).")

    # Histogram of the two real signals.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, j, name in zip(axes, (0, 1), ("c_t", "s_t")):
            ax.hist(U[:, j].numpy(), bins=40)
            ax.set_title(f"B3c signal {name}")
            ax.set_xlabel(name)
            ax.set_ylabel("count")
        fig.tight_layout()
        out_png = os.path.join(args.out, "b3c_signal_distributions.png")
        fig.savefig(out_png, dpi=150)
        print(f"\nwrote {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(skipped histogram: {e})")


if __name__ == "__main__":
    main()
