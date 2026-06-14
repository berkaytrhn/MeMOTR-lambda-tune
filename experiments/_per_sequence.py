"""Per-sequence aggregator for T7 / F8.

Reads every ``pedestrian_detailed.csv`` under --results-root for the chosen
(exp, seed, best-checkpoint) and builds a (method x seq) matrix of HOTA, AssA,
IDSW. Then computes Delta(method - B0) per sequence.

Best checkpoint per (exp, seed) is taken from ``_best_per_seed.json`` written
by ``_parse_results.py`` (run that first).
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from statistics import mean

SEQ_COLS = {
    "HOTA": "HOTA___AUC",     # TrackEval pedestrian_detailed has columns named e.g. HOTA___AUC
    "AssA": "AssA___AUC",
    "IDSW": "IDSW",
    "MOTA": "MOTA",
    "IDF1": "IDF1",
}


def find_col(header: list[str], wanted: str) -> str | None:
    """pedestrian_detailed.csv has many <metric>___<threshold> cols; AUC is the headline."""
    if wanted in header:
        return wanted
    for h in header:
        if h.startswith(wanted.split("___")[0] + "___AUC"):
            return h
    for h in header:
        if h.split("___")[0] == wanted.split("___")[0]:
            return h
    return None


def load_detailed(csv_path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {k: find_col(header, v) for k, v in SEQ_COLS.items()}
        seq_idx = header.index("seq") if "seq" in header else 0
        for row in reader:
            seq = row[seq_idx]
            if seq.upper() == "COMBINED":
                continue
            d = {}
            for k, col in cols.items():
                if col is None:
                    continue
                try:
                    d[k] = float(row[header.index(col)])
                except (ValueError, IndexError):
                    pass
            out[seq] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("/workspace/persistent/results"))
    ap.add_argument("--tables",       type=Path, default=Path("/workspace/persistent/results/_tables"))
    ap.add_argument("--out",          type=Path, default=Path("/workspace/persistent/results/_tables/per_sequence.csv"))
    args = ap.parse_args()

    best = json.loads((args.tables / "_best_per_seed.json").read_text())

    # Resolve each (exp, seed) -> pedestrian_detailed.csv (best ck), then average over seeds.
    matrix: dict[str, dict[str, list[float]]] = {}  # matrix[exp][seq] = list of HOTA across seeds

    def ck_dir(exp: str, seed: int | None, ck: int) -> Path | None:
        if exp == "b0":
            return args.results_root / "b0/outputs/memotr_dancetrack/val" / f"checkpoint_{ck}_tracker"
        if exp == "b1":
            return None
        return args.results_root / exp / f"seed_{seed}/val" / f"checkpoint_{ck}_tracker"

    for exp, entry in best.items():
        if exp in ("b0", "b1"):
            seeds_to_iter = [{"seed": None, "ck": 0}] if exp == "b0" else []
        else:
            seeds_to_iter = entry.get("seeds", [])
        for s in seeds_to_iter:
            d = ck_dir(exp, s.get("seed"), s["ck"])
            if d is None:
                continue
            csv_path = d / "pedestrian_detailed.csv"
            if not csv_path.exists():
                print(f"[skip] missing {csv_path}")
                continue
            for seq, metrics in load_detailed(csv_path).items():
                matrix.setdefault(exp, {}).setdefault(seq, []).append(metrics.get("HOTA", 0.0))

    # Average over seeds, then write Delta vs B0 per sequence.
    avg: dict[str, dict[str, float]] = {exp: {s: mean(v) for s, v in d.items()} for exp, d in matrix.items()}
    seqs = sorted({s for d in avg.values() for s in d})
    methods = [e for e in ["b0", "b3a", "b3b", "b3c", "b2", "b4"] if e in avg]

    with args.out.open("w") as f:
        w = csv.writer(f)
        w.writerow(["seq"] + methods + [f"d{m}_vs_b0" for m in methods if m != "b0"])
        for s in seqs:
            row = [s] + [f"{avg[m].get(s, ''):.4f}" if avg[m].get(s) is not None else "" for m in methods]
            for m in methods:
                if m == "b0":
                    continue
                if s in avg.get("b0", {}) and s in avg.get(m, {}):
                    row.append(f"{avg[m][s] - avg['b0'][s]:+.4f}")
                else:
                    row.append("")
            w.writerow(row)

    # Summary: mean Delta, #improved sequences per method.
    print("Per-sequence summary (avg HOTA over best-ckpt seeds):")
    b0 = avg.get("b0", {})
    for m in methods:
        if m == "b0":
            continue
        deltas = [avg[m][s] - b0[s] for s in seqs if s in avg[m] and s in b0]
        if deltas:
            n_pos = sum(1 for d in deltas if d > 0)
            print(f"  {m}: mean dHOTA = {mean(deltas):+.3f} on {len(deltas)} seqs, "
                  f"improved on {n_pos}/{len(deltas)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
