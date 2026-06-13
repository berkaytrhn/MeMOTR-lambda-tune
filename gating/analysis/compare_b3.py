#!/usr/bin/env python3
"""Compare B3 variants (B3a/B3b/B3c) across seeds and recommend which to seed B4.

Reads only the per-checkpoint TrackEval summaries that EVAL_MODE=continue wrote
under <variant>/seed_<S>/<split>/checkpoint_*_tracker/pedestrian_summary.txt.
Does NOT run TrackEval, load checkpoints, or train anything.

ROBUSTNESS DESIGN (damp seed / epoch randomness as much as n allows):
  * Paired across seeds: only the seed set COMMON to every variant is used, so
    all variants are scored on identical seeds (no variant gets a lucky extra seed).
  * No max-over-epochs by default: that maximizes over noise and unfairly favors
    variants with more epochs. Default policy averages the LAST K checkpoints
    (--epoch-policy lastk, --window K). 'last' and 'best' are available for reference.
  * Variance is shown: per epoch we report mean, std, and the WORST seed (min),
    not just the mean.
  * Ranking can be worst-case aware (--rank mean|mean_minus_std|worst).
  * Paired unanimity: does the winner beat the runner-up on EVERY shared seed?
  * Separability verdict: if the winner's lead < pooled cross-seed std, the result
    is flagged WITHIN NOISE (n seeds is too few to separate them).

  Primary metric = HOTA; tie-break = AssA (the metric gating targets).

Caveat the script cannot fix: with only 2 seeds you cannot *eliminate* seed
randomness, only stop amplifying it. A 3rd seed is the real remedy.

Usage:
    python gating/analysis/compare_b3.py
    python gating/analysis/compare_b3.py outputs/b3a outputs/b3b outputs/b3c
    python gating/analysis/compare_b3.py --epoch-policy lastk --window 3 --rank mean
    python gating/analysis/compare_b3.py --epoch-policy best        # old optimistic mode
"""
import argparse
import glob
import os
import re
import statistics
from collections import defaultdict


def read_summary(path: str) -> dict:
    """Parse a TrackEval pedestrian_summary.txt: header row then values row."""
    with open(path) as f:
        names = f.readline().strip().split(" ")
        values = f.readline().strip().split(" ")
    return {n: float(v) for n, v in zip(names, values)}


def collect(base: str, split: str):
    """variant -> {seed: {epoch: metrics}} for one variant base dir."""
    out = defaultdict(dict)
    for sd in sorted(glob.glob(os.path.join(base, "seed_*"))):
        seed = os.path.basename(sd).replace("seed_", "")
        pattern = os.path.join(sd, split, "checkpoint_*_tracker", "pedestrian_summary.txt")
        for path in sorted(glob.glob(pattern),
                           key=lambda p: int(re.search(r"checkpoint_(\d+)_tracker", p).group(1))):
            epoch = int(re.search(r"checkpoint_(\d+)_tracker", path).group(1))
            out[seed][epoch] = read_summary(path)
    return out


def find_ckpt_pth(base: str, seed: str, epoch: int):
    for cand in (
        os.path.join(base, f"seed_{seed}", f"checkpoint_{epoch}.pth"),
        os.path.join(base, f"seed_{seed}", "train", f"checkpoint_{epoch}.pth"),
    ):
        if os.path.exists(cand):
            return cand
    return os.path.join(base, f"seed_{seed}", f"checkpoint_{epoch}.pth (NOT FOUND)")


def std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def per_seed_score(seed_epochs: dict, metric: str, policy: str, window: int):
    """Score one seed under the chosen epoch policy. Returns (score, epochs_used)."""
    epochs = sorted(seed_epochs)
    if not epochs:
        return float("nan"), []
    if policy == "last":
        use = [epochs[-1]]
    elif policy == "best":
        use = [max(epochs, key=lambda e: seed_epochs[e][metric])]
    else:  # lastk
        use = epochs[-window:]
    vals = [seed_epochs[e][metric] for e in use]
    return sum(vals) / len(vals), use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*",
                    default=["outputs/b3a", "outputs/b3b", "outputs/b3c"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--primary", default="HOTA")
    ap.add_argument("--tiebreak", default="AssA")
    ap.add_argument("--epoch-policy", choices=["lastk", "last", "best"], default="lastk",
                    help="how each seed's score is reduced over epochs (default lastk)")
    ap.add_argument("--window", type=int, default=3, help="K for --epoch-policy lastk")
    ap.add_argument("--rank", choices=["mean", "mean_minus_std", "worst"], default="mean",
                    help="how variants are ranked across seeds (default mean)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    P, T = args.primary, args.tiebreak
    names = [os.path.basename(b.rstrip("/")) for b in args.variants]
    raw = {}  # name -> {seed: {epoch: metrics}}
    for base, name in zip(args.variants, names):
        d = collect(base, args.split)
        if d:
            raw[name] = d
        else:
            print(f"[warn] no {args.split} summaries under {base}")
    if not raw:
        print("Nothing to compare.")
        return

    # ---- paired seeds: intersection across variants ----
    seed_sets = {n: set(d) for n, d in raw.items()}
    common = set.intersection(*seed_sets.values())
    union = set.union(*seed_sets.values())
    if not common:
        print("[error] no seed common to all variants; cannot compare fairly.")
        return
    if common != union:
        dropped = sorted(union - common)
        print(f"[warn] using only seeds common to all variants: {sorted(common)}. "
              f"Dropped (not present everywhere): {dropped}")
    common = sorted(common)

    # ---- per-variant per-seed scores under the epoch policy ----
    print(f"\nepoch policy = {args.epoch_policy}"
          + (f" (K={args.window})" if args.epoch_policy == "lastk" else "")
          + f" | rank = {args.rank} | seeds = {common}")
    scores = {}  # name -> {seed: (primary, tiebreak)}
    for name in raw:
        scores[name] = {}
        print(f"\n=== {name} ===")
        print(f"{'seed':>6}  {'epochs_used':>16}  {P:>8}  {T:>8}")
        for s in common:
            se = raw[name][s]
            ps, used = per_seed_score(se, P, args.epoch_policy, args.window)
            ts, _ = per_seed_score(se, T, args.epoch_policy, args.window)
            scores[name][s] = (ps, ts)
            print(f"{s:>6}  {str(used):>16}  {ps:>8.3f}  {ts:>8.3f}")
        pv = [scores[name][s][0] for s in common]
        tv = [scores[name][s][1] for s in common]
        print(f"  {P}: mean={sum(pv)/len(pv):.3f}  std={std(pv):.3f}  "
              f"worst={min(pv):.3f}   |   {T}: mean={sum(tv)/len(tv):.3f}")

    # ---- rank variants ----
    def agg(name):
        pv = [scores[name][s][0] for s in common]
        tv = [scores[name][s][1] for s in common]
        mean_p, sd_p = sum(pv) / len(pv), std(pv)
        key = {"mean": mean_p, "mean_minus_std": mean_p - sd_p, "worst": min(pv)}[args.rank]
        return key, sum(tv) / len(tv), mean_p, sd_p, min(pv)

    ranked = sorted(raw, key=lambda n: (agg(n)[0], agg(n)[1]), reverse=True)
    print("\n---------------- ranking ----------------")
    print(f"{'rank':>4}  {'variant':>8}  {'rank_key':>10}  {'mean':>8}  {'std':>7}  {'worst':>8}")
    for i, n in enumerate(ranked, 1):
        key, _t, mp, sd, wt = agg(n)
        print(f"{i:>4}  {n:>8}  {key:>10.3f}  {mp:>8.3f}  {sd:>7.3f}  {wt:>8.3f}")

    win, run = ranked[0], (ranked[1] if len(ranked) > 1 else None)

    # ---- paired unanimity: winner beats runner-up on every shared seed? ----
    if run:
        per_seed = {s: scores[win][s][0] - scores[run][s][0] for s in common}
        unanimous = all(d > 0 for d in per_seed.values())
        print(f"\nPaired vs runner-up ({run}): per-seed {P} deltas "
              + ", ".join(f"{s}:{d:+.3f}" for s, d in per_seed.items()))
        print(f"  winner wins on ALL seeds: {unanimous}")

        # ---- separability: lead vs pooled cross-seed noise ----
        gap = agg(win)[2] - agg(run)[2]                       # mean lead
        pooled = (std([scores[win][s][0] for s in common])
                  + std([scores[run][s][0] for s in common])) / 2
        verdict = "SEPARABLE" if gap > pooled else "WITHIN NOISE"
        print(f"  mean lead = {gap:.3f}, pooled cross-seed std = {pooled:.3f}  -> {verdict}")
        if verdict == "WITHIN NOISE" or not unanimous:
            print(f"  [caution] {win} vs {run} not robustly separated at n={len(common)} "
                  f"seeds — add a seed before treating this as a real win.")

    # ---- recommendation + concrete B4 .pth ----
    wbase = next(b for b, n in zip(args.variants, names) if n == win)
    # within the winner, pick the seed whose policy-score is the median/representative;
    # for the concrete file we take the best-scoring shared seed (and report all).
    best_seed = max(common, key=lambda s: (scores[win][s][0], scores[win][s][1]))
    # the actual epoch handed over depends on policy: for lastk/last use final epoch,
    # for best use that seed's argmax epoch.
    epochs = sorted(raw[win][best_seed])
    if args.epoch_policy == "best":
        pick_epoch = max(epochs, key=lambda e: raw[win][best_seed][e][P])
    else:
        pick_epoch = epochs[-1]
    print("\n================ RECOMMENDATION ================")
    print(f"Winning variant : {win}  (rank metric={args.rank})")
    print(f"Carry-forward seed for B4 init: seed {best_seed}, epoch {pick_epoch}")
    print(f"B4 init (--pretrained-model): {find_ckpt_pth(wbase, best_seed, pick_epoch)}")
    print("Note: the variant choice is the robust result; the single .pth above is one")
    print("concrete instance of it. Confirm the winner clears B0/base HOTA before B4.")

    if args.csv:
        cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
        with open(args.csv, "w") as f:
            f.write("variant,seed,epoch," + ",".join(cols) + "\n")
            for name in raw:
                for s in sorted(raw[name]):
                    for e in sorted(raw[name][s]):
                        m = raw[name][s][e]
                        f.write(f"{name},{s},{e}," +
                                ",".join(f"{m.get(c, float('nan')):.3f}" for c in cols) + "\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
