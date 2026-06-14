"""Extract the learned lambda parameters from B3a / B3b / B3c checkpoints.

Walks <RESULTS>/{b3a,b3b,b3c}/seed_<S>/checkpoint_<K>.pth, locates the gate's
parameter tensor (``*.theta`` for ScalarGate/VectorGate, the MLP for SignalGate),
runs the same sigmoid -> clamp(1e-3, 0.5) as ``gating/longterm_gates.py``, and
writes:

  _tables/b3a_lambda_trajectory.csv   (seed, ck, lambda)
  _tables/b3b_lambda_per_channel.csv  (seed, ck, channel_index, lambda)
  _tables/b3c_signal_gate_state.json  (per-seed final MLP weights, for plotting F4)

Run after _parse_results.py.
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path
import torch

INIT_LOGIT = -4.595
CK_RE = re.compile(r"checkpoint_(\d+)\.pth$")
SEED_RE = re.compile(r"seed_(\d+)$")


def clamp_lambda(theta: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(theta).clamp(1e-3, 0.5)


def find_gate_theta(state_dict: dict) -> torch.Tensor | None:
    for k, v in state_dict.items():
        if k.endswith("long_term_gate.theta") or k.endswith("gate.theta") or k.endswith(".theta"):
            if torch.is_tensor(v) and (v.ndim == 0 or v.ndim == 1):
                return v.detach().float().cpu()
    return None


def find_signal_mlp(state_dict: dict) -> dict | None:
    weights = {k: v.detach().float().cpu() for k, v in state_dict.items()
               if (".net." in k) and ("long_term_gate" in k or "gate" in k)}
    return weights or None


def iter_checkpoints(root: Path, exp: str):
    for p in (root / exp).rglob("checkpoint_*.pth"):
        seed = next((int(SEED_RE.match(par.name).group(1)) for par in p.parents
                     if SEED_RE.match(par.name)), None)
        m = CK_RE.search(p.name)
        if seed is None or not m:
            continue
        yield seed, int(m.group(1)), p


def load(p: Path) -> dict | None:
    try:
        sd = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[warn] {p}: {e}", file=sys.stderr)
        return None
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]
    return sd if isinstance(sd, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("/workspace/persistent/results"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/persistent/results/_tables"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # B3a: scalar.
    rows = []
    for seed, ck, p in sorted(iter_checkpoints(args.results_root, "b3a")):
        sd = load(p)
        if sd is None:
            continue
        theta = find_gate_theta(sd)
        if theta is None:
            print(f"[skip b3a] no theta in {p}")
            continue
        lam = clamp_lambda(theta).item()
        rows.append([seed, ck, lam])
    with (args.out / "b3a_lambda_trajectory.csv").open("w") as f:
        w = csv.writer(f); w.writerow(["seed", "ck", "lambda"]); w.writerows(rows)
    print(f"b3a: wrote {len(rows)} rows")

    # B3b: per-channel vector.
    rows = []
    for seed, ck, p in sorted(iter_checkpoints(args.results_root, "b3b")):
        sd = load(p)
        if sd is None:
            continue
        theta = find_gate_theta(sd)
        if theta is None or theta.ndim != 1:
            continue
        lam = clamp_lambda(theta).tolist()
        for i, v in enumerate(lam):
            rows.append([seed, ck, i, v])
    with (args.out / "b3b_lambda_per_channel.csv").open("w") as f:
        w = csv.writer(f); w.writerow(["seed", "ck", "channel", "lambda"]); w.writerows(rows)
    print(f"b3b: wrote {len(rows)} rows (seed, ck, channel)")

    # B3c: stash the MLP weights of the final checkpoint per seed. The response
    # surface (F4) is rendered by _make_figures.py by running the MLP on a grid.
    b3c = {}
    for seed, ck, p in sorted(iter_checkpoints(args.results_root, "b3c")):
        sd = load(p)
        if sd is None:
            continue
        mlp = find_signal_mlp(sd)
        if mlp is None:
            continue
        # keep just the last ck per seed
        b3c[str(seed)] = {"ck": ck,
                          "weights": {k: v.tolist() for k, v in mlp.items()}}
    (args.out / "b3c_signal_gate_state.json").write_text(json.dumps(b3c, indent=2))
    print(f"b3c: stashed MLP for seeds {list(b3c)}")


if __name__ == "__main__":
    main()
