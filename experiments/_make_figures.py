"""Generate the report figures F1-F8 from the parsed tables and extracted lambdas.

Inputs (produced by the other experiments/_*.py scripts):
  _tables/_all_results.json
  _tables/_best_per_seed.json
  _tables/_mean_ck.json
  _tables/T2_lambda_sweep.csv
  _tables/b3a_lambda_trajectory.csv
  _tables/b3b_lambda_per_channel.csv
  _tables/b3c_signal_gate_state.json
  _tables/per_sequence.csv

Outputs (in --out):
  F1_lambda_sensitivity.png
  F2_b3a_lambda_trajectory.png
  F3_b3b_lambda_histogram.png
  F4_b3c_response_surface.png
  F5_noise_floor_panel.png
  F6_delta_vs_b0_bars.png
  F7_deta_assa_scatter.png
  F8_per_sequence_delta.png

Each figure is plain matplotlib so it can be regenerated headlessly.
F0 (method diagram) is conceptual - drawn separately in TikZ / draw.io.
F9 (B2 gate-collapse) requires a GPU forward pass - see _b2_gate_collapse.py.
F10 (qualitative clips) requires DanceTrack val frames - see _render_qualitative.py.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


COLORS = {"b0": "#444", "b1": "#888", "b2": "#1f77b4",
          "b3a": "#2ca02c", "b3b": "#d62728", "b3c": "#9467bd", "b4": "#ff7f0e"}


def fig1_lambda_sensitivity(tables: Path, out: Path):
    with (tables / "T2_lambda_sweep.csv").open() as f:
        r = list(csv.DictReader(f))
    lams = [float(x["lambda"]) for x in r]
    hota = [float(x["HOTA"]) for x in r]
    assa = [float(x["AssA"]) for x in r]
    # learned lambda from B3a (mean across seeds, last ck)
    traj = list(csv.DictReader((tables / "b3a_lambda_trajectory.csv").open()))
    last = {}
    for row in traj:
        last[row["seed"]] = float(row["lambda"])
    learned = float(np.mean(list(last.values()))) if last else None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lams, hota, "o-", color="C0", label="HOTA")
    ax.plot(lams, assa, "s--", color="C3", label="AssA")
    ax.set_xscale("log")
    ax.set_xlabel(r"fixed $\lambda$ (log scale)"); ax.set_ylabel("metric (%)")
    ax.axvline(0.01, color="grey", lw=0.8, ls=":", label=r"released $\lambda$=0.01")
    if learned is not None:
        ax.axvline(learned, color="green", lw=1.2, ls="-.",
                   label=rf"B3a learned $\lambda \approx${learned:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title(r"F1: HOTA/AssA vs fixed $\lambda$")
    fig.tight_layout(); fig.savefig(out / "F1_lambda_sensitivity.png", dpi=150); plt.close(fig)


def fig2_b3a_trajectory(tables: Path, out: Path):
    rows = list(csv.DictReader((tables / "b3a_lambda_trajectory.csv").open()))
    by_seed: dict = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append((int(r["ck"]), float(r["lambda"])))
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed, pts in by_seed.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"seed {seed}")
    ax.axhline(0.01, color="grey", ls=":", label=r"released $\lambda$=0.01")
    ax.set_xlabel("checkpoint (epoch)"); ax.set_ylabel(r"learned scalar $\lambda$")
    ax.set_title(r"F2: B3a scalar $\lambda$ trajectory")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "F2_b3a_lambda_trajectory.png", dpi=150); plt.close(fig)


def fig3_b3b_histogram(tables: Path, out: Path):
    rows = list(csv.DictReader((tables / "b3b_lambda_per_channel.csv").open()))
    if not rows:
        return
    last_per_seed = {}
    for r in rows:
        key = r["seed"]
        last_per_seed.setdefault(key, {"ck": -1, "vals": []})
        ck = int(r["ck"])
        if ck > last_per_seed[key]["ck"]:
            last_per_seed[key] = {"ck": ck, "vals": [float(r["lambda"])]}
        elif ck == last_per_seed[key]["ck"]:
            last_per_seed[key]["vals"].append(float(r["lambda"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 0.2, 60)
    for seed, d in last_per_seed.items():
        ax.hist(d["vals"], bins=bins, alpha=0.5, label=f"seed {seed} (ck{d['ck']})")
    ax.axvline(0.01, color="grey", ls=":", label=r"released $\lambda$=0.01")
    ax.set_xlabel(r"per-channel learned $\lambda$"); ax.set_ylabel("count (of 256)")
    ax.set_title("F3: B3b per-channel lambda histogram")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "F3_b3b_lambda_histogram.png", dpi=150); plt.close(fig)


def _signal_mlp_forward(weights: dict, grid: np.ndarray) -> np.ndarray:
    """Replicate gating.longterm_gates.SignalGate forward pass on a grid of (in_dim,) inputs."""
    state = {k.split(".net.")[-1]: torch.tensor(v) for k, v in weights.items()}
    w1, b1 = state["0.weight"], state["0.bias"]
    w2, b2 = state["2.weight"], state["2.bias"]
    x = torch.tensor(grid, dtype=torch.float32)
    h = torch.relu(x @ w1.T + b1)
    out = h @ w2.T + b2
    lam = torch.sigmoid(out).clamp(1e-3, 0.5)
    return lam.squeeze(-1).numpy()


def fig4_b3c_response(tables: Path, out: Path):
    state = json.loads((tables / "b3c_signal_gate_state.json").read_text())
    if not state:
        return
    # signals in order: (c, delta, H, s). Sweep c and s, hold delta=0, H=0 (mean-normalized).
    cs = np.linspace(-2, 2, 25)
    ss = np.linspace(-2, 2, 25)
    fig, axes = plt.subplots(1, len(state), figsize=(4.5 * len(state), 4), squeeze=False)
    for ax, (seed, d) in zip(axes[0], state.items()):
        grid = np.array([[c, 0.0, 0.0, s] for c in cs for s in ss])
        Z = _signal_mlp_forward(d["weights"], grid).reshape(len(cs), len(ss))
        im = ax.imshow(Z, origin="lower", aspect="auto",
                       extent=[ss.min(), ss.max(), cs.min(), cs.max()], cmap="viridis")
        ax.set_xlabel("similarity s (z-scored)")
        ax.set_ylabel("confidence c (z-scored)")
        ax.set_title(f"B3c seed {seed} (ck{d['ck']})")
        fig.colorbar(im, ax=ax, label=r"learned $\lambda$")
    fig.suptitle(r"F4: B3c $\lambda$ response surface (c, s)")
    fig.tight_layout(); fig.savefig(out / "F4_b3c_response_surface.png", dpi=150); plt.close(fig)


def fig5_noise_floor(tables: Path, out: Path):
    rows = json.loads((tables / "_all_results.json").read_text())
    b0 = next((r for r in rows if r["exp"] == "b0"), None)
    fig, ax = plt.subplots(figsize=(7, 4))
    exps = ["b3a", "b3b", "b3c", "b2", "b4"]
    for i, e in enumerate(exps):
        ys = [r["HOTA"] for r in rows if r["exp"] == e]
        xs = [i + np.random.uniform(-0.18, 0.18) for _ in ys]
        ax.scatter(xs, ys, c=COLORS[e], alpha=0.7, s=22, label=e)
    if b0:
        ax.axhline(b0["HOTA"], color="black", ls="--", lw=1, label=f"B0 = {b0['HOTA']:.2f}")
        ax.axhspan(b0["HOTA"] - 0.23, b0["HOTA"] + 0.23, color="grey", alpha=0.15,
                   label=r"$\pm$0.23 noise floor")
    ax.set_xticks(range(len(exps))); ax.set_xticklabels(exps)
    ax.set_ylabel("val HOTA (%)"); ax.set_title("F5: per-checkpoint HOTA vs B0 noise floor")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "F5_noise_floor_panel.png", dpi=150); plt.close(fig)


def fig6_delta_bars(tables: Path, out: Path):
    best = json.loads((tables / "_best_per_seed.json").read_text())
    meanck = json.loads((tables / "_mean_ck.json").read_text())
    b0_best_h = best["b0"]["HOTA"]["mean"]; b0_meanck_h = meanck["b0"]["HOTA_mean"]
    b0_best_a = best["b0"]["AssA"]["mean"]; b0_meanck_a = meanck["b0"]["AssA_mean"]
    exps = [e for e in ["b3a", "b3b", "b3c", "b2", "b4"] if e in best]
    dh_best = [best[e]["HOTA"]["mean"] - b0_best_h for e in exps]
    dh_mean = [meanck[e]["HOTA_mean"] - b0_meanck_h for e in exps]
    da_best = [best[e]["AssA"]["mean"] - b0_best_a for e in exps]
    da_mean = [meanck[e]["AssA_mean"] - b0_meanck_a for e in exps]
    x = np.arange(len(exps)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(x - w/2, dh_best, w, label="best-ck", color="C0")
    ax1.bar(x + w/2, dh_mean, w, label="mean-ck", color="C1")
    ax1.axhline(0, color="black", lw=0.8); ax1.axhspan(-0.23, 0.23, color="grey", alpha=0.15)
    ax1.set_xticks(x); ax1.set_xticklabels(exps); ax1.set_ylabel(r"$\Delta$HOTA vs B0")
    ax1.set_title("HOTA"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.bar(x - w/2, da_best, w, label="best-ck", color="C0")
    ax2.bar(x + w/2, da_mean, w, label="mean-ck", color="C1")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(exps); ax2.set_ylabel(r"$\Delta$AssA vs B0")
    ax2.set_title("AssA"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle("F6: best-ck vs mean-ck deltas - the selection-bias effect")
    fig.tight_layout(); fig.savefig(out / "F6_delta_vs_b0_bars.png", dpi=150); plt.close(fig)


def fig7_deta_assa_scatter(tables: Path, out: Path):
    rows = json.loads((tables / "_all_results.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 5))
    for e in ["b0", "b3a", "b3b", "b3c", "b2", "b4"]:
        rs = [r for r in rows if r["exp"] == e]
        if not rs: continue
        ax.scatter([r["DetA"] for r in rs], [r["AssA"] for r in rs],
                   c=COLORS[e], label=e, alpha=0.75, s=40)
    ax.set_xlabel("DetA (%)"); ax.set_ylabel("AssA (%)")
    ax.set_title("F7: variance lives in AssA, not DetA")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "F7_deta_assa_scatter.png", dpi=150); plt.close(fig)


def fig8_per_sequence(tables: Path, out: Path):
    path = tables / "per_sequence.csv"
    if not path.exists(): return
    rows = list(csv.DictReader(path.open()))
    # AUC values are in [0,1]; multiply by 100 to read as %.
    deltas = {"b4": [], "b3c": []}
    seqs = []
    for r in rows:
        seqs.append(r["seq"])
        for m in deltas:
            v = r.get(f"d{m}_vs_b0")
            deltas[m].append(float(v) * 100 if v else 0.0)
    order = np.argsort(deltas["b4"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(seqs)), [deltas["b4"][i] for i in order], color="C1", label="B4 - B0")
    ax.plot(np.arange(len(seqs)), [deltas["b3c"][i] for i in order], "o-", color="C4", ms=4, label="B3c - B0")
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(np.arange(len(seqs)))
    ax.set_xticklabels([seqs[i].replace("dancetrack", "") for i in order], rotation=60, fontsize=7)
    ax.set_ylabel(r"$\Delta$HOTA (pp)"); ax.set_title("F8: per-sequence delta sorted by B4")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out / "F8_per_sequence_delta.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=Path("/workspace/persistent/results/_tables"))
    ap.add_argument("--out",    type=Path, default=Path("/workspace/persistent/results/_figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    fig1_lambda_sensitivity(args.tables, args.out)
    fig2_b3a_trajectory(args.tables, args.out)
    fig3_b3b_histogram(args.tables, args.out)
    fig4_b3c_response(args.tables, args.out)
    fig5_noise_floor(args.tables, args.out)
    fig6_delta_bars(args.tables, args.out)
    fig7_deta_assa_scatter(args.tables, args.out)
    fig8_per_sequence(args.tables, args.out)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
