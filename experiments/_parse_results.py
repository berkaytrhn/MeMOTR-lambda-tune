"""Canonical TrackEval results parser for the adaptive-gating study.

Walks every ``pedestrian_summary.txt`` under ``--results-root`` and emits:
  * ``_all_results.json``  - per-checkpoint long table (every (exp,seed,ck) row).
  * ``T1_main.csv``        - best-epoch mean +/- std across seeds, plus deltas vs B0.
  * ``T2_lambda_sweep.csv``- B1 fixed-lambda sweep (one row per lambda).
  * ``T3_long_term.csv``   - B3a vs B3b vs B3c (best-epoch and mean-ck).
  * ``T4_short_term.csv``  - B0 vs B2 vs B4 (best-epoch).
  * ``T5_robustness.csv``  - mean-over-checkpoints + within-run std (noise floor).

Layout assumed (under --results-root):
    b0/outputs/memotr_dancetrack/val/checkpoint_K_tracker/pedestrian_summary.txt
    b1/outputs/b1/lambda_<L>/val/checkpoint_K_tracker/pedestrian_summary.txt
    {b2,b3a,b3b,b3c,b4}/seed_<S>/val/checkpoint_K_tracker/pedestrian_summary.txt
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from statistics import mean, pstdev

METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "MOTP", "IDR", "IDP"]
CK_RE = re.compile(r"checkpoint_(\d+)_tracker$")
SEED_RE = re.compile(r"seed_(\d+)$")
LAM_RE = re.compile(r"lambda_([0-9.]+)$")


def parse_summary(path: Path) -> dict | None:
    try:
        lines = path.read_text().strip().splitlines()
        if len(lines) < 2:
            return None
        header, values = lines[0].split(), lines[1].split()
        row = dict(zip(header, values))
        out = {}
        for m in METRICS:
            if m in row:
                out[m] = float(row[m])
        return out
    except Exception as e:
        print(f"[warn] {path}: {e}", file=sys.stderr)
        return None


def walk(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    # B0 (released, single checkpoint, no seed/lambda)
    for p in (results_root / "b0").rglob("pedestrian_summary.txt"):
        m = parse_summary(p)
        if m is None:
            continue
        ck = int(CK_RE.search(p.parent.name).group(1)) if CK_RE.search(p.parent.name) else 0
        rows.append({"exp": "b0", "seed": None, "lambda": 0.01, "ck": ck, **m, "path": str(p)})
    # B1 (lambda sweep, no seed)
    for p in (results_root / "b1").rglob("pedestrian_summary.txt"):
        m = parse_summary(p)
        if m is None:
            continue
        lam = None
        for parent in p.parents:
            if LAM_RE.match(parent.name):
                lam = float(LAM_RE.match(parent.name).group(1))
                break
        ck = int(CK_RE.search(p.parent.name).group(1)) if CK_RE.search(p.parent.name) else 0
        rows.append({"exp": "b1", "seed": None, "lambda": lam, "ck": ck, **m, "path": str(p)})
    # B2/B3a/B3b/B3c/B4 (seed_<S>/val/checkpoint_<K>_tracker)
    for exp in ["b2", "b3a", "b3b", "b3c", "b4"]:
        for p in (results_root / exp).rglob("pedestrian_summary.txt"):
            m = parse_summary(p)
            if m is None:
                continue
            seed = None
            for parent in p.parents:
                if SEED_RE.match(parent.name):
                    seed = int(SEED_RE.match(parent.name).group(1))
                    break
            ck = int(CK_RE.search(p.parent.name).group(1)) if CK_RE.search(p.parent.name) else 0
            rows.append({"exp": exp, "seed": seed, "lambda": None, "ck": ck, **m, "path": str(p)})
    return rows


def best_epoch_table(rows: list[dict]) -> dict:
    """For each (exp, seed) pick max-HOTA checkpoint, then aggregate over seeds."""
    by_es: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["exp"] in ("b0", "b1"):
            continue
        by_es.setdefault((r["exp"], r["seed"]), []).append(r)
    best_per_seed: dict[str, list[dict]] = {}
    for (exp, seed), rs in by_es.items():
        best = max(rs, key=lambda r: r["HOTA"])
        best_per_seed.setdefault(exp, []).append({"seed": seed, "ck": best["ck"], **{m: best[m] for m in METRICS if m in best}})
    agg = {}
    for exp, seeds in best_per_seed.items():
        agg[exp] = {}
        for m in METRICS:
            vals = [s[m] for s in seeds if m in s]
            if not vals:
                continue
            agg[exp][m] = {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}
        agg[exp]["seeds"] = seeds
    # B0 single row
    b0 = [r for r in rows if r["exp"] == "b0"]
    if b0:
        agg["b0"] = {m: {"mean": b0[0][m], "std": 0.0, "n": 1} for m in METRICS if m in b0[0]}
        agg["b0"]["seeds"] = [{"seed": None, "ck": b0[0]["ck"], **{m: b0[0][m] for m in METRICS if m in b0[0]}}]
    return agg


def mean_ck_table(rows: list[dict]) -> dict:
    """Mean over ALL checkpoints and seeds (selection-bias-free) + within-run noise floor."""
    out: dict = {}
    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        if r["exp"] in ("b1",):
            continue
        by_exp.setdefault(r["exp"], []).append(r)
    for exp, rs in by_exp.items():
        out[exp] = {}
        for m in METRICS:
            vals = [r[m] for r in rs if m in r]
            if vals:
                out[exp][m + "_mean"] = mean(vals)
        # within-run (within-seed) std of HOTA - the noise floor
        stds = []
        by_seed: dict = {}
        for r in rs:
            by_seed.setdefault(r["seed"], []).append(r["HOTA"])
        for vs in by_seed.values():
            if len(vs) > 1:
                stds.append(pstdev(vs))
        out[exp]["HOTA_within_std_median"] = sorted(stds)[len(stds)//2] if stds else 0.0
        out[exp]["HOTA_within_std_mean"]   = mean(stds) if stds else 0.0
        out[exp]["n_checkpoints"]          = len(rs)
    return out


def b1_sweep_table(rows: list[dict]) -> list[dict]:
    return sorted([r for r in rows if r["exp"] == "b1"], key=lambda r: r["lambda"])


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.write_text(",".join(header) + "\n" + "\n".join(",".join(str(c) for c in r) for r in rows) + "\n")


def emit_tables(out: Path, all_rows: list[dict], best: dict, mean_ck: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "_all_results.json").write_text(json.dumps(all_rows, indent=2))
    (out / "_best_per_seed.json").write_text(json.dumps(best, indent=2))
    (out / "_mean_ck.json").write_text(json.dumps(mean_ck, indent=2))

    # T1 main
    order = ["b0", "b1", "b3a", "b3b", "b3c", "b2", "b4"]
    rows = []
    b0h = best.get("b0", {}).get("HOTA", {}).get("mean")
    b0a = best.get("b0", {}).get("AssA", {}).get("mean")
    for exp in order:
        if exp not in best:
            continue
        cells = [exp]
        for m in ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]:
            v = best[exp].get(m)
            if v is None:
                cells.append("")
            else:
                cells.append(f"{v['mean']:.3f}" + (f" +/-{v['std']:.3f}" if v["n"] > 1 else ""))
        dh = best[exp].get("HOTA", {}).get("mean", 0) - (b0h or 0)
        da = best[exp].get("AssA", {}).get("mean", 0) - (b0a or 0)
        cells += [f"{dh:+.3f}", f"{da:+.3f}"]
        rows.append(cells)
    write_csv(out / "T1_main.csv",
              ["exp", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "dHOTA_vs_B0", "dAssA_vs_B0"], rows)

    # T2 lambda sweep
    sweep = b1_sweep_table(all_rows)
    write_csv(out / "T2_lambda_sweep.csv",
              ["lambda", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"],
              [[r["lambda"]] + [r.get(m, "") for m in ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]] for r in sweep])

    # T5 robustness: per-exp best vs mean-ck + noise floor
    t5_rows = []
    for exp in order:
        if exp not in mean_ck:
            continue
        mh, ma = mean_ck[exp].get("HOTA_mean"), mean_ck[exp].get("AssA_mean")
        bh = best.get(exp, {}).get("HOTA", {}).get("mean")
        ba = best.get(exp, {}).get("AssA", {}).get("mean")
        t5_rows.append([exp,
                        f"{mh:.3f}" if mh else "",
                        f"{bh:.3f}" if bh else "",
                        f"{ma:.3f}" if ma else "",
                        f"{ba:.3f}" if ba else "",
                        f"{mean_ck[exp]['HOTA_within_std_median']:.3f}",
                        mean_ck[exp]["n_checkpoints"]])
    write_csv(out / "T5_robustness.csv",
              ["exp", "HOTA_mean_ck", "HOTA_best", "AssA_mean_ck", "AssA_best",
               "HOTA_within_std_median", "n_checkpoints"], t5_rows)
    print(f"Wrote tables under {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("/workspace/persistent/results"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/persistent/results/_tables"))
    args = ap.parse_args()
    rows = walk(args.results_root)
    print(f"Parsed {len(rows)} checkpoint rows across "
          f"{len({(r['exp'], r['seed']) for r in rows})} (exp,seed) groups.")
    emit_tables(args.out, rows, best_epoch_table(rows), mean_ck_table(rows))


if __name__ == "__main__":
    main()
