# B3 Comparison — selecting the best B3 variant to seed B4

This doc explains how `gating/analysis/compare_b3.py` compares the B3 variants
(B3a scalar / B3b vector / B3c signal long-term λ gate) across seeds and
recommends which one to carry forward into B4.

It is **read-only**: it parses the TrackEval summaries that `EVAL_MODE=continue`
already wrote. It never runs TrackEval, loads a `.pth`, or trains anything.

---

## What it reads

For every variant it globs:

```
outputs/<variant>/seed_<S>/val/checkpoint_*_tracker/pedestrian_summary.txt
```

Each `pedestrian_summary.txt` is a 2-line TrackEval file (header row, then
values row). The script extracts **HOTA** (primary metric) and **AssA**
(tie-break) per checkpoint, indexed by `(variant, seed, epoch)`.

---

## How the comparison works (robustness design)

The goal is to damp seed/epoch randomness as much as a small number of seeds
allows. The script does five things to avoid fooling itself:

1. **Paired seeds.** Only the seed set common to *every* variant is used, so all
   variants are scored on identical seeds (no variant gets a lucky extra seed).
2. **No max-over-epochs by default.** Picking the single best epoch maximizes
   over noise and unfairly rewards variants with more epochs. Instead each
   seed's score = the mean of its **last K checkpoints** (`--epoch-policy lastk`,
   `--window K`). `last` and `best` are available for reference.
3. **Variance is shown.** Per variant it reports HOTA **mean, std, and worst
   seed (min)** — not just the mean.
4. **Paired unanimity.** It checks whether the winner beats the runner-up on
   *every* shared seed, not just on average.
5. **Separability verdict.** If the winner's mean lead is smaller than the
   pooled cross-seed std, it prints `WITHIN NOISE` — i.e. the variants are not
   actually distinguishable at this seed count.

**Decision metric:** HOTA primary, AssA tie-break. Ranking can be made
worst-case aware with `--rank mean_minus_std` or `--rank worst`.

> Caveat the script cannot fix: with only 2 seeds you cannot *eliminate* seed
> randomness, only stop amplifying it. A 3rd seed is the real remedy.

---

## Step-by-step: running the comparison

### 1. Prerequisites
Make sure each variant/seed has been evaluated so the summaries exist:

```
ls outputs/b3a/seed_42/val/checkpoint_0_tracker/pedestrian_summary.txt
```

If summaries are missing, run the released eval flow (`EVAL_MODE=continue`)
first — `compare_b3.py` does not generate them.

Use the repo venv (the system Python lacks the CUDA deform-attn op, but the
comparison itself is pure-Python so either works):

```
cd ~/MeMOTR-lambda-tune
```

### 2. Run with the robust defaults

```
./venv/bin/python gating/analysis/compare_b3.py \
    --epoch-policy lastk --window 3 --rank mean
```

(With no positional args it defaults to `outputs/b3a outputs/b3b outputs/b3c`
and `--split val`.)

### 3. Read the output, top to bottom
* **Per-variant tables** — each seed's `epochs_used`, HOTA, AssA, then the
  variant's HOTA `mean / std / worst`.
* **Ranking table** — variants sorted by the rank key.
* **Paired vs runner-up** — per-seed deltas, "wins on ALL seeds", and the
  `SEPARABLE` / `WITHIN NOISE` verdict.
* **RECOMMENDATION** — winning variant, the carry-forward seed/epoch, and the
  concrete `.pth` path for B4's `--pretrained-model`.

### 4. Stress-test the winner
Re-run ranked by the worst seed:

```
./venv/bin/python gating/analysis/compare_b3.py \
    --epoch-policy lastk --window 3 --rank worst
```

If the winner is the same under both `--rank mean` and `--rank worst`, that's a
strong signal. If it flips, the variants are not robustly separated — treat the
result as a tie and add a seed before committing.

### 5. (Optional) dump the full table

```
./venv/bin/python gating/analysis/compare_b3.py --csv b3_compare.csv
```

Writes every `variant,seed,epoch,HOTA,DetA,AssA,MOTA,IDF1,IDSW` row.

---

## Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `variants` (positional) | `outputs/b3a outputs/b3b outputs/b3c` | variant base dirs |
| `--split` | `val` | which eval split to read |
| `--primary` | `HOTA` | primary decision metric |
| `--tiebreak` | `AssA` | tie-break metric |
| `--epoch-policy` | `lastk` | `lastk` (mean of last K) / `last` / `best` |
| `--window` | `3` | K for `lastk` |
| `--rank` | `mean` | `mean` / `mean_minus_std` / `worst` |
| `--csv` | _none_ | dump full per-checkpoint table |

---

## Example result (interpreting it)

```
================ RECOMMENDATION ================
Winning variant : b3c  (rank metric=mean)
Carry-forward seed for B4 init: seed 123, epoch 9
B4 init (--pretrained-model): outputs/b3c/seed_123/checkpoint_9.pth
```

The **variant choice (b3c)** is the robust result. The single `.pth` is one
concrete instance of it — necessarily a point estimate, so it is the least
robust number in the report. Before seeding B4:

* confirm the winner clears the **B0/base HOTA** (the script does not check this);
* if the verdict was `WITHIN NOISE`, add a seed rather than treating the win as real.

> Note on B4 init: for a clean ablation, B4 should be initialized from the
> **same released pretrained checkpoint** as the other arms, not warm-started
> from this B3c checkpoint — warm-starting confounds the comparison (different
> init + extra compute + val-selection leakage). Use the recommended `.pth` only
> if you are deliberately running a separately-labeled curriculum/staged arm.
