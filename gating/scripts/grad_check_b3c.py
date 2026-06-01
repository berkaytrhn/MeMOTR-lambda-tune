#!/usr/bin/env python3
"""B3c MANDATORY pre-train gradient check (IMPLEMENTATION_PLAN fact #3).

The released code detaches long_memory before it is read, so a learnable
long-term lambda gets ZERO gradient ("grad trap"). B3c sets
DETACH_LONG_MEMORY=False to relax that detach. This script proves the fix
worked BEFORE you spend GPU-hours training: it builds the B3c model, runs ONE
clip forward + backward on a single training batch, and asserts that the
SignalGate MLP (query_updater.lt_gate) receives a finite, non-zero gradient,
and that ONLY the gate is trainable (~97 params over 4 tensors).

A note on the SignalGate init: the final linear weight is zeroed and its bias
is -4.595 (so lambda starts at the released 0.01 for every track). At init the
two LOWER-layer tensors (net.0) therefore have ZERO gradient — their effect on
the output is multiplied by the zeroed final weight. This is expected and not a
grad trap: the FINAL layer (net.2.bias and net.2.weight) does receive gradient,
which moves the final weight off zero so the lower layers train from step 1.
The check asserts the gate as a whole gets usable gradient AND specifically that
the final-layer bias grad is non-zero (this is the bit that proves the gradient
chain from loss back through the EMA lambda is intact).

If the gate's gradient is None/zero, DO NOT TRAIN — DETACH_LONG_MEMORY did not
take effect (or LONG_TERM_GATE/FREEZE flags are wrong). Fix the config first.

Usage (from repo root, GPU visible):
    python gating/scripts/grad_check_b3c.py
    python gating/scripts/grad_check_b3c.py --config-path configs/B3c.yaml
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


def get_gate(model):
    qu = get_model(model).query_updater
    assert getattr(qu, "lt_gate", None) is not None, \
        "query_updater.lt_gate is None — LONG_TERM_GATE is not 'signal'."
    assert hasattr(qu.lt_gate, "net"), "lt_gate has no .net — not a SignalGate."
    return qu.lt_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", default=os.path.join(REPO_ROOT, "configs", "B3c.yaml"))
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
    assert config["LONG_TERM_GATE"] == "signal", "B3c needs LONG_TERM_GATE: signal"
    assert config["DETACH_LONG_MEMORY"] is False, \
        "B3c needs DETACH_LONG_MEMORY: False or the gate gets zero gradient (grad trap)."
    assert config["FREEZE_ALL_BUT_GATES"] is True, "B3c needs FREEZE_ALL_BUT_GATES: True"

    # --- build + load + freeze, exactly like train() ------------------------
    model = build_model(config=config).to(device)
    if config.get("PRETRAINED_MODEL"):
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False)
    trainable = freeze_all_but_gates(model)
    n_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    print(f"[grad-check] trainable tensors ({len(trainable)}): {trainable}")
    print(f"[grad-check] total trainable params: {n_params}")
    assert len(trainable) == 4, \
        f"B3c (SignalGate MLP 4->16->1) expects 4 trainable tensors, got {len(trainable)}: {trainable}"
    assert all(".lt_gate." in n for n in trainable), \
        f"unexpected non-gate trainable params: {trainable}"
    assert n_params == 97, f"B3c expects 97 trainable params (4*16+16 + 16*1+1), got {n_params}"

    gate = get_gate(model)
    out_bias = gate.net[-1].bias
    print(f"[grad-check] init final bias={out_bias.item():.4f}  ->  lambda(zero signal)="
          f"{torch.sigmoid(out_bias).item():.5f} (released fixed lambda = {config['LONG_MEMORY_LAMBDA']})")

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
    assert loss.requires_grad, \
        "loss has no grad_fn — the gate never reached the loss (clip too short or grad trap)."
    loss.backward()

    # Aggregate gate gradient + the all-important final-bias gradient.
    grad_norm = 0.0
    for n, p in gate.named_parameters():
        g = p.grad
        finite = (g is not None) and torch.isfinite(g).all()
        gn = 0.0 if g is None else float(g.norm().item())
        grad_norm += gn
        print(f"[grad-check]   {n:>12}  grad_norm={gn:.3e}  finite={bool(finite)}")
    out_bias_grad = out_bias.grad

    print(f"[grad-check] total gate grad norm = {grad_norm:.3e}")
    print(f"[grad-check] final bias grad = "
          f"{None if out_bias_grad is None else f'{out_bias_grad.item():.3e}'}")

    ok = (out_bias_grad is not None) and torch.isfinite(out_bias_grad).all() \
        and (out_bias_grad.abs().item() > 0.0) and (grad_norm > 0.0)
    if not ok:
        print("\n[FAIL] the signal gate received NO usable gradient. This is the grad trap.\n"
              "       Verify DETACH_LONG_MEMORY=False is actually in effect "
              "(configs/B3c.yaml and query_updater.update_tracks_embedding).\n"
              "       DO NOT TRAIN until this passes.")
        sys.exit(1)

    print("\n[PASS] the signal gate is learnable: its final bias has a non-zero, finite "
          "gradient. Safe to launch B3c training.\n"
          "       (Reminder: net.0 grads are ~0 at init by design — the zeroed final "
          "weight gates them; they start training once the final weight moves off zero.)")


if __name__ == "__main__":
    main()
