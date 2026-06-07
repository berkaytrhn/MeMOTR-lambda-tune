#!/usr/bin/env python3
"""B4 MANDATORY pre-train gradient check (IMPLEMENTATION_PLAN fact #3).

B4 turns on BOTH gates at once:
  * long-term  self.lt_gate = SignalGate          (LONG_TERM_GATE: signal)
  * short-term self.st_gate = SymmetricSoftmaxGate (SHORT_TERM_GATE: symmetric)
Only these train (FREEZE_ALL_BUT_GATES). This script proves both are learnable
BEFORE spending GPU-hours: it builds the B4 model, runs ONE clip forward +
backward on a single training batch, and asserts that BOTH gates receive a
finite, non-zero gradient, and that ONLY the gates are trainable
(12 tensors = 4 lt_gate + 8 st_gate = 33,123 params).

The long-term gate is the one exposed to the grad trap: the released code
detaches long_memory before it is read, so a learnable lambda gets ZERO
gradient unless DETACH_LONG_MEMORY=False AND the clip is >=3 frames. The
short-term gate is not detached and trains on any clip length, but we check it
too so a mis-wired symmetric gate is caught here rather than after training.

SignalGate init note: its final linear weight is zeroed and bias is -4.595, so
lambda starts at the released 0.01. At init the lower layer (net.0) therefore
has ~0 gradient (gated by the zeroed final weight); the FINAL layer (net.2)
must have non-zero gradient — that is the bit proving the chain from loss back
through the EMA lambda is intact. The check asserts on the final-bias grad.

If either gate's gradient is None/zero, DO NOT TRAIN — fix the config first.

Usage (from repo root, GPU visible):
    python gating/scripts/grad_check_b4.py
    python gating/scripts/grad_check_b4.py --config-path configs/B4.yaml
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

N_LT_TENSORS = 4         # SignalGate MLP 4->16->1: net.0.{w,b}, net.2.{w,b}
N_ST_TENSORS = 8         # SymmetricSoftmaxGate: cur_mlp + prev_mlp, 4 tensors each
N_LT_PARAMS = 97         # 4*16+16 + 16*1+1
N_ST_PARAMS = 33026      # 2 * (256*64+64 + 64*1+1)


def get_gates(model):
    qu = get_model(model).query_updater
    lt = getattr(qu, "lt_gate", None)
    st = getattr(qu, "st_gate", None)
    assert lt is not None, "query_updater.lt_gate is None — LONG_TERM_GATE is not 'signal'."
    assert hasattr(lt, "net"), "lt_gate has no .net — not a SignalGate."
    assert st is not None, "query_updater.st_gate is None — SHORT_TERM_GATE is not 'symmetric'."
    assert hasattr(st, "cur_mlp") and hasattr(st, "prev_mlp"), \
        "st_gate is not a SymmetricSoftmaxGate (missing cur_mlp/prev_mlp)."
    return lt, st


def grad_report(name, module):
    """Print per-tensor grad norms for a module; return total grad norm."""
    total = 0.0
    for n, p in module.named_parameters():
        g = p.grad
        gn = 0.0 if g is None else float(g.norm().item())
        finite = (g is not None) and torch.isfinite(g).all()
        total += gn
        print(f"[grad-check]   {name}.{n:<22} grad_norm={gn:.3e}  finite={bool(finite)}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", default=os.path.join(REPO_ROOT, "configs", "B4.yaml"))
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "dataset"))
    ap.add_argument("--frames", type=int, default=5,
                    help="force the smoke clip to this length. The long-term lambda only "
                         "reaches the loss when a clip is >=3 frames long. Default 5 makes "
                         "the check meaningful and deterministic.")
    args = ap.parse_args()

    config = yaml_to_dict(args.config_path)
    config["DATA_ROOT"] = args.data_root
    config["SAMPLE_LENGTHS"] = [int(args.frames)]
    config["SAMPLE_STEPS"] = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("AVAILABLE_GPUS", "0"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["SEED"])

    # --- sanity on the flags themselves -------------------------------------
    assert config["LONG_TERM_GATE"] == "signal", "B4 needs LONG_TERM_GATE: signal"
    assert config["SHORT_TERM_GATE"] == "symmetric", "B4 needs SHORT_TERM_GATE: symmetric"
    assert config["DETACH_LONG_MEMORY"] is False, \
        "B4 needs DETACH_LONG_MEMORY: False or the long-term gate gets zero gradient (grad trap)."
    assert config["FREEZE_ALL_BUT_GATES"] is True, "B4 needs FREEZE_ALL_BUT_GATES: True"

    # --- build + load + freeze, exactly like train() ------------------------
    model = build_model(config=config).to(device)
    if config.get("PRETRAINED_MODEL"):
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False)
    trainable = freeze_all_but_gates(model)
    n_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    lt_names = [n for n in trainable if ".lt_gate." in n]
    st_names = [n for n in trainable if ".st_gate." in n]
    print(f"[grad-check] trainable tensors ({len(trainable)}): "
          f"{len(lt_names)} lt_gate + {len(st_names)} st_gate")
    print(f"[grad-check] total trainable params: {n_params}")
    assert all((".lt_gate." in n) or (".st_gate." in n) for n in trainable), \
        f"unexpected non-gate trainable params: {trainable}"
    assert len(lt_names) == N_LT_TENSORS, \
        f"expected {N_LT_TENSORS} lt_gate tensors (SignalGate), got {len(lt_names)}: {lt_names}"
    assert len(st_names) == N_ST_TENSORS, \
        f"expected {N_ST_TENSORS} st_gate tensors (SymmetricSoftmaxGate), got {len(st_names)}: {st_names}"
    assert n_params == N_LT_PARAMS + N_ST_PARAMS, \
        f"B4 expects {N_LT_PARAMS + N_ST_PARAMS} trainable params, got {n_params}"

    lt_gate, st_gate = get_gates(model)
    out_bias = lt_gate.net[-1].bias
    print(f"[grad-check] lt_gate init final bias={out_bias.item():.4f} -> lambda(zero signal)="
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
        "loss has no grad_fn — the gates never reached the loss (clip too short or grad trap)."
    loss.backward()

    print("[grad-check] --- long-term gate (SignalGate) ---")
    lt_norm = grad_report("lt_gate", lt_gate)
    out_bias_grad = out_bias.grad
    print("[grad-check] --- short-term gate (SymmetricSoftmaxGate) ---")
    st_norm = grad_report("st_gate", st_gate)

    print(f"[grad-check] total lt_gate grad norm = {lt_norm:.3e}")
    print(f"[grad-check] lt_gate final-bias grad = "
          f"{None if out_bias_grad is None else f'{out_bias_grad.item():.3e}'}")
    print(f"[grad-check] total st_gate grad norm = {st_norm:.3e}")

    lt_ok = (out_bias_grad is not None) and torch.isfinite(out_bias_grad).all() \
        and (out_bias_grad.abs().item() > 0.0) and (lt_norm > 0.0)
    st_ok = st_norm > 0.0
    if not lt_ok:
        print("\n[FAIL] the LONG-TERM signal gate received NO usable gradient (grad trap).\n"
              "       Verify DETACH_LONG_MEMORY=False is in effect and the clip is >=3 frames.\n"
              "       DO NOT TRAIN until this passes.")
        sys.exit(1)
    if not st_ok:
        print("\n[FAIL] the SHORT-TERM symmetric gate received NO gradient.\n"
              "       Verify SHORT_TERM_GATE=symmetric is wired in query_updater.\n"
              "       DO NOT TRAIN until this passes.")
        sys.exit(1)

    print("\n[PASS] BOTH gates are learnable: the long-term gate's final bias and the "
          "short-term gate both have non-zero, finite gradients. Safe to launch B4 training.\n"
          "       (Reminder: lt_gate net.0 grads are ~0 at init by design — the zeroed "
          "final weight gates them; they start training once the final weight moves off zero.)")


if __name__ == "__main__":
    main()
