#!/usr/bin/env python3
"""B3a MANDATORY pre-train gradient check (IMPLEMENTATION_PLAN fact #3).

The released code detaches long_memory before it is read, so a learnable
long-term lambda gets ZERO gradient ("grad trap"). B3a sets
DETACH_LONG_MEMORY=False to relax that detach. This script proves the fix
worked BEFORE you spend GPU-hours training: it builds the B3a model, runs ONE
clip forward + backward on a single training batch, and asserts that
    model.query_updater.lt_gate.theta.grad is not None and != 0
and that exactly one parameter is trainable (the scalar theta).

If theta.grad is None/zero, DO NOT TRAIN — DETACH_LONG_MEMORY did not take
effect (or LONG_TERM_GATE/FREEZE flags are wrong). Fix the config first.

Usage (from repo root, GPU visible):
    python gating/scripts/grad_check_b3a.py
    python gating/scripts/grad_check_b3a.py --config-path configs/B3a.yaml
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
from gating.freeze import freeze_all_but_gates                        # noqa: E402


def find_theta(model):
    qu = get_model(model).query_updater
    assert getattr(qu, "lt_gate", None) is not None, \
        "query_updater.lt_gate is None — LONG_TERM_GATE is not 'scalar'."
    assert hasattr(qu.lt_gate, "theta"), "lt_gate has no .theta — not a ScalarGate."
    return qu.lt_gate.theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", default=os.path.join(REPO_ROOT, "configs", "B3a.yaml"))
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "dataset"))
    ap.add_argument("--frames", type=int, default=5,
                    help="force the smoke clip to this length. The learnable lambda only "
                         "reaches the loss when a clip is >=3 frames long: lambda updates "
                         "the long-term memory at frame t, that memory builds the query "
                         "feature at frame t+1, which feeds the decoder/loss at frame t+2. "
                         "A length-2 clip therefore gives ZERO gradient even with the "
                         "detach fix. Default 5 makes the check meaningful and deterministic.")
    args = ap.parse_args()

    config = yaml_to_dict(args.config_path)
    config["DATA_ROOT"] = args.data_root
    # Force a long-enough clip so the check actually exercises lambda's gradient
    # path (see --frames help). Stage stays 0 so SAMPLE_LENGTHS[0] is used.
    config["SAMPLE_LENGTHS"] = [int(args.frames)]
    config["SAMPLE_STEPS"] = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("AVAILABLE_GPUS", "0"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["SEED"])

    # --- sanity on the flags themselves -------------------------------------
    assert config["LONG_TERM_GATE"] == "scalar", "B3a needs LONG_TERM_GATE: scalar"
    assert config["DETACH_LONG_MEMORY"] is False, \
        "B3a needs DETACH_LONG_MEMORY: False or the gate gets zero gradient (grad trap)."
    assert config["FREEZE_ALL_BUT_GATES"] is True, "B3a needs FREEZE_ALL_BUT_GATES: True"

    # --- build + load + freeze, exactly like train() ------------------------
    model = build_model(config=config).to(device)
    if config.get("PRETRAINED_MODEL"):
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False)
    trainable = freeze_all_but_gates(model)
    print(f"[grad-check] trainable tensors ({len(trainable)}): {trainable}")
    assert len(trainable) == 1, f"B3a expects exactly 1 trainable tensor, got {len(trainable)}: {trainable}"
    assert trainable[0].endswith("lt_gate.theta"), f"unexpected trainable param name: {trainable[0]}"

    theta = find_theta(model)
    print(f"[grad-check] init theta={theta.item():.4f}  ->  lambda={torch.sigmoid(theta).item():.5f} "
          f"(released fixed lambda = {config['LONG_MEMORY_LAMBDA']})")

    # --- one clip forward + backward ----------------------------------------
    dataset = build_dataset(config=config, split="train")
    sampler = build_sampler(dataset=dataset, shuffle=True)
    dataloader = build_dataloader(dataset=dataset, sampler=sampler, batch_size=1, num_workers=0)
    criterion = build_criterion(config=config)
    criterion.set_device(device)

    model.train()
    model.zero_grad(set_to_none=True)
    batch = next(iter(dataloader))
    use_dab = config["USE_DAB"]

    tracks = TrackInstances.init_tracks(
        batch=batch, hidden_dim=get_model(model).hidden_dim,
        num_classes=get_model(model).num_classes, device=device, use_dab=use_dab)
    criterion.init_a_clip(batch=batch, hidden_dim=get_model(model).hidden_dim,
                          num_classes=get_model(model).num_classes, device=device)

    n_frames = len(batch["imgs"][0])
    print(f"[grad-check] clip length = {n_frames} frames")
    for frame_idx in range(n_frames):
        frame = [fs[frame_idx] for fs in batch["imgs"]]
        for f in frame:
            f.requires_grad_(False)
        frame = tensor_list_to_nested_tensor(tensor_list=frame).to(device)
        res = model(frame=frame, tracks=tracks)
        previous_tracks, new_tracks, unmatched_dets = criterion.process_single_frame(
            model_outputs=res, tracked_instances=tracks, frame_idx=frame_idx)
        if frame_idx < n_frames - 1:
            tracks = get_model(model).postprocess_single_frame(
                previous_tracks, new_tracks, unmatched_dets)

    loss_dict, _ = criterion.get_mean_by_n_gts()
    loss = criterion.get_sum_loss_dict(loss_dict=loss_dict)
    print(f"[grad-check] loss = {loss.item():.4f}")
    loss.backward()

    g = theta.grad
    print(f"[grad-check] theta.grad = {None if g is None else g.item():.3e}")

    ok = (g is not None) and torch.isfinite(g).all() and (g.abs().item() > 0.0)
    if not ok:
        print("\n[FAIL] theta received NO usable gradient. This is the grad trap.\n"
              "       Verify DETACH_LONG_MEMORY=False is actually in effect "
              "(configs/B3a.yaml and query_updater.update_tracks_embedding).\n"
              "       DO NOT TRAIN until this passes.")
        sys.exit(1)

    print("\n[PASS] lambda is learnable: theta has a non-zero, finite gradient. "
          "Safe to launch B3a training.")


if __name__ == "__main__":
    main()
