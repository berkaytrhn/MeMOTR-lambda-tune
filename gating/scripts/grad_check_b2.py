#!/usr/bin/env python3
"""B2 MANDATORY pre-train gradient check + init-gap measurement.

B2 replaces MeMOTR's asymmetric short-term fusion with a SYMMETRIC softmax gate
(gating/shortterm_gate.py: SymmetricSoftmaxGate). Unlike the long-term gates
(B3a/b/c), B2 is NOT affected by the "grad trap" (fact #3): its short_memory is
never detached, and it reaches the loss through only a 2-frame chain (st_gate at
frame t -> query_feat -> tracks.query_embed -> frozen decoder at frame t+1 ->
loss). A length-2 clip is therefore already enough to give the gate gradient.

This script proves the gate is learnable BEFORE spending GPU-hours: it builds the
B2 model, freezes everything but st_gate, and:
  1. asserts EXACTLY 8 trainable tensors (the two 2-layer MLPs of st_gate),
     33026 params total (~33k), all named "*.st_gate.*";
  2. measures and logs the INIT GAP between the freshly-initialized symmetric
     gate's short_memory and the released asymmetric fusion's short_memory on a
     real (O_t, O_{t-1}) batch (plan §2.4: the prev-branch bias is set low so the
     initial softmax approximates the released behavior — we quantify the residual);
  3. logs the initial softmax weights (mean w_t, w_{t-1}); if w_t ~ 1 the gate
     starts close to the asymmetric "current dominates" regime;
  4. runs ONE clip forward + backward and asserts every st_gate tensor receives a
     finite, non-zero gradient.

If any st_gate tensor has None/zero grad, DO NOT TRAIN — check the flags
(SHORT_TERM_GATE: symmetric, FREEZE_ALL_BUT_GATES: True) first.

Usage (from repo root, GPU visible):
    python gating/scripts/grad_check_b2.py
    python gating/scripts/grad_check_b2.py --config-path configs/B2.yaml
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

EXPECTED_PARAMS = 33026  # SymmetricSoftmaxGate(dim=256, hidden=64): 2 x (256->64->1)


def find_st_gate(model):
    qu = get_model(model).query_updater
    assert getattr(qu, "st_gate", None) is not None, \
        "query_updater.st_gate is None — SHORT_TERM_GATE is not 'symmetric'."
    assert hasattr(qu.st_gate, "cur_mlp") and hasattr(qu.st_gate, "prev_mlp"), \
        "st_gate is not a SymmetricSoftmaxGate (missing cur_mlp/prev_mlp)."
    return qu.st_gate


def measure_init_gap(qu, output_embed, last_output_embed):
    """Compare the symmetric gate's short_memory to the released asymmetric
    fusion's short_memory on the same (O_t, O_{t-1}). Returns a dict of stats."""
    with torch.no_grad():
        # Released asymmetric path (the pretrained confidence_weight_net /
        # short_memory_fusion are still present on the frozen model).
        confidence_weight = qu.confidence_weight_net(output_embed)
        sm_asym = qu.short_memory_fusion(
            torch.cat((confidence_weight * output_embed, last_output_embed), dim=-1))
        # New symmetric gate path + its internal weights.
        l_t = qu.st_gate.cur_mlp(output_embed)
        l_prev = qu.st_gate.prev_mlp(last_output_embed)
        weights = torch.softmax(torch.cat((l_t, l_prev), dim=-1), dim=-1)
        w_t, w_prev = weights[:, 0], weights[:, 1]
        sm_sym = qu.st_gate(output_embed, last_output_embed)
        rel = (sm_sym - sm_asym).norm(dim=-1) / (sm_asym.norm(dim=-1) + 1e-8)
    return {
        "n": output_embed.shape[0],
        "w_t_mean": w_t.mean().item(), "w_t_min": w_t.min().item(), "w_t_max": w_t.max().item(),
        "w_prev_mean": w_prev.mean().item(),
        "rel_gap_mean": rel.mean().item(), "rel_gap_max": rel.max().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", default=os.path.join(REPO_ROOT, "configs", "B2.yaml"))
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "dataset"))
    ap.add_argument("--frames", type=int, default=3,
                    help="force the smoke clip to this length. B2's gate reaches the "
                         "loss through a 2-frame chain, so >=2 is enough; default 3 is "
                         "a deterministic, slightly stronger check.")
    args = ap.parse_args()

    config = yaml_to_dict(args.config_path)
    config["DATA_ROOT"] = args.data_root
    # Deterministic, long-enough clip (stage stays 0 -> SAMPLE_LENGTHS[0] is used).
    config["SAMPLE_LENGTHS"] = [int(args.frames)]
    config["SAMPLE_STEPS"] = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("AVAILABLE_GPUS", "0"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["SEED"])

    # --- sanity on the flags themselves -------------------------------------
    assert config["SHORT_TERM_GATE"] == "symmetric", "B2 needs SHORT_TERM_GATE: symmetric"
    assert config["LONG_TERM_GATE"] == "fixed", \
        "B2 isolates the short-term gate: LONG_TERM_GATE must be 'fixed'."
    assert config["FREEZE_ALL_BUT_GATES"] is True, "B2 needs FREEZE_ALL_BUT_GATES: True"

    # --- build + load + freeze, exactly like train() ------------------------
    model = build_model(config=config).to(device)
    if config.get("PRETRAINED_MODEL"):
        model = load_pretrained_model(model, config["PRETRAINED_MODEL"], show_details=False)
    trainable = freeze_all_but_gates(model)
    print(f"[grad-check] trainable tensors ({len(trainable)}):")
    for n in trainable:
        print(f"             {n}")
    assert len(trainable) == 8, f"B2 expects exactly 8 trainable tensors, got {len(trainable)}"
    assert all(".st_gate." in n for n in trainable), \
        f"unexpected non-st_gate trainable param(s): {[n for n in trainable if '.st_gate.' not in n]}"

    st_gate = find_st_gate(model)
    n_params = sum(p.numel() for p in st_gate.parameters())
    print(f"[grad-check] st_gate params = {n_params} (expected {EXPECTED_PARAMS})")
    assert n_params == EXPECTED_PARAMS, \
        f"B2 expects {EXPECTED_PARAMS} st_gate params, got {n_params}"

    # --- one clip forward + backward ----------------------------------------
    dataset = build_dataset(config=config, split="train")
    sampler = build_sampler(dataset=dataset, shuffle=True)
    dataloader = build_dataloader(dataset=dataset, sampler=sampler, batch_size=1, num_workers=0)
    criterion = build_criterion(config=config)
    criterion.set_device(device)

    # Capture the first (O_t, O_{t-1}) pair seen by st_gate to measure the init gap.
    captured = {}

    def hook(_module, inputs, _output):
        if "in" not in captured:
            captured["in"] = (inputs[0].detach(), inputs[1].detach())

    handle = st_gate.register_forward_hook(hook)

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

    handle.remove()

    # --- init-gap report (plan §2.4) ----------------------------------------
    if "in" in captured:
        o_t, o_prev = captured["in"]
        stats = measure_init_gap(get_model(model).query_updater, o_t, o_prev)
        print(f"[grad-check] init softmax weights on {stats['n']} tracks: "
              f"w_t mean={stats['w_t_mean']:.4f} (min={stats['w_t_min']:.4f} "
              f"max={stats['w_t_max']:.4f}), w_prev mean={stats['w_prev_mean']:.4f}")
        print(f"[grad-check] INIT GAP vs released asymmetric fusion: "
              f"mean rel. L2 = {stats['rel_gap_mean']:.4f}, max = {stats['rel_gap_max']:.4f} "
              f"(the symmetric gate must learn to close this during training)")
    else:
        print("[grad-check] WARNING: st_gate was never called (no active tracks this clip); "
              "init-gap not measured.")

    loss_dict, _ = criterion.get_mean_by_n_gts()
    loss = criterion.get_sum_loss_dict(loss_dict=loss_dict)
    print(f"[grad-check] loss = {loss.item():.4f}")
    assert loss.requires_grad, \
        "loss has no grad_fn — st_gate did not reach the loss (clip too short / no tracks)."
    loss.backward()

    # --- gradient assertions over every st_gate tensor ----------------------
    bad = []
    total_norm = 0.0
    for n, p in st_gate.named_parameters():
        g = p.grad
        ok = (g is not None) and torch.isfinite(g).all() and (g.abs().sum().item() > 0.0)
        gn = (g.norm().item() if g is not None else float("nan"))
        total_norm += (gn ** 2 if g is not None else 0.0)
        print(f"[grad-check] st_gate.{n}: grad_ok={ok} ||grad||={gn:.3e}")
        if not ok:
            bad.append(n)
    print(f"[grad-check] total ||grad|| over st_gate = {total_norm ** 0.5:.3e}")

    if bad:
        print(f"\n[FAIL] these st_gate tensors got NO usable gradient: {bad}\n"
              "       Verify SHORT_TERM_GATE=symmetric and FREEZE_ALL_BUT_GATES=True "
              "are in effect (configs/B2.yaml + query_updater.update_tracks_embedding).\n"
              "       DO NOT TRAIN until this passes.")
        sys.exit(1)

    print("\n[PASS] the symmetric short-term gate is learnable: all 8 st_gate "
          "tensors have finite, non-zero gradients. Safe to launch B2 training.")


if __name__ == "__main__":
    main()
