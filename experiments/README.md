# Reporting scripts — tables & figures for the final report

All scripts assume the experiment outputs live under
`/workspace/persistent/results/{b0,b1,b2,b3a,b3b,b3c,b4}/...`
(set `--results-root` to override). Run them from the repo root with the
same Python env you use for training.

## Pipeline

```bash
# 1. CPU-only: parse TrackEval summaries -> _all_results.json + T1, T2, T5 CSVs
python experiments/_parse_results.py

# 2. CPU-only: per-sequence (method x seq) HOTA matrix + Delta vs B0  -> T7/F8 data
python experiments/_per_sequence.py

# 3. CPU-only (needs torch): extract learned lambda from B3a/B3b/B3c .pth files
python experiments/_extract_lambda.py

# 4. CPU-only: render figures F1..F8
python experiments/_make_figures.py

# 5. GPU: short-term gate-collapse stats for B2/B4  -> F9 / T4
python experiments/_b2_gate_collapse.py \
  --checkpoint /workspace/persistent/results/b4/seed_42/checkpoint_4.pth \
  --config configs/b4.yaml

# 6. CPU + cv2: qualitative side-by-side videos B0 vs B4  -> F10
python experiments/_render_qualitative.py \
  --b0-tracker /workspace/persistent/results/b0/outputs/memotr_dancetrack/val/checkpoint_0_tracker \
  --b4-tracker /workspace/persistent/results/b4/seed_42/val/checkpoint_4_tracker \
  --data-root  /path/to/DanceTrack
```

Steps 1–4 are CPU-only and reproduce every number cited in
`reporting_plan.md` from the raw TrackEval outputs. Steps 5–6 are the
compute-gated stretch items (§7 of the plan).

## What each script writes

| Script | Outputs (under `_tables/` or `_figures/`) | Feeds |
|---|---|---|
| `_parse_results.py` | `_all_results.json`, `_best_per_seed.json`, `_mean_ck.json`, `T1_main.csv`, `T2_lambda_sweep.csv`, `T5_robustness.csv` | T1, T2, T5 |
| `_per_sequence.py`  | `per_sequence.csv` (method × seq, Δ vs B0) | T7, F8 |
| `_extract_lambda.py`| `b3a_lambda_trajectory.csv`, `b3b_lambda_per_channel.csv`, `b3c_signal_gate_state.json` | F2, F3, F4 |
| `_make_figures.py`  | `F1`–`F8`.png | report §5 |
| `_b2_gate_collapse.py` | `_short_term_weights.npz`, `F9_short_term_weights.png` | F9, T4 |
| `_render_qualitative.py` | `F10_qualitative/<seq>_b0_vs_b4.mp4` | F10 |

## Figure recipes (how to read & caption each one)

**F0 — Method diagram.** Not auto-generated. Draw in TikZ or draw.io: two
memory pathways (long-term EMA write, short-term 2-frame fusion), highlight
the two gate insertion points, show fixed-λ → learned-λ on the long path and
asym → symmetric on the short path. Caption: "MeMOTR with the two gate
points replaced by the variants studied here (B3*, B2/B4)."

**F1 — λ sensitivity.** From `T2_lambda_sweep.csv`. HOTA *and* AssA vs
log-λ, with the B3a learned-λ marker. Caption: "Shallow, non-monotonic
landscape; the released λ=0.01 already sits at the val optimum and the
learned scalar (B3a, dashed) rediscovers a value 2–3× larger but inside the
same flat plateau."

**F2 — B3a trajectory.** Two seeds overlaid (now that both checkpoints are
extracted), reference line at 0.01. Caption: "Scalar λ rises from the
released 0.01 to ~0.025–0.04 within 5 epochs on both seeds."

**F3 — B3b histogram.** Per-channel λ at the last checkpoint, both seeds.
Caption: "Per-channel capacity is exercised — the 256 λ spread heavily —
yet aggregate metrics drop vs B3a (H2 refuted)."

**F4 — B3c response surface.** Heatmap of λ over (c, s), with δ=H=0
(mean-normalised). Caption: "The signal gate admits more memory when the
detection is confident AND agrees with memory, less when noisy/divergent —
the SNR-adaptive bottleneck the IB motivation predicts."

**F5 — Noise floor panel.** Every checkpoint's HOTA jittered around its
method, B0 line + ±0.23 σ band. Caption: "Every per-checkpoint HOTA falls
inside ±0.23 of B0 — the within-run selection-noise envelope. The
best-epoch table thus credits selection bias as much as method."

**F6 — Δ-bars best vs mean.** Best-ck vs mean-over-ck deltas vs B0, HOTA and
AssA side-by-side, with the ±0.23 noise band on HOTA. Caption: "Selection
bias accounts for the entire aggregate HOTA gap; on the mean only B4 holds
AssA parity, and only barely."

**F7 — DetA–AssA scatter.** All 76 checkpoints, coloured by method.
Caption: "All variance lives on the AssA axis (range 2.7 pp) while DetA is
fixed in a 0.8-pp band — the intervention moves association, not detection."

**F8 — Per-sequence delta.** ΔHOTA(B4−B0) sorted across the 25 val seqs,
with B3c overlaid. Caption: "Sequence-level gains are roughly balanced
around zero; a handful of occlusion-heavy clips drive most of the positive
mass."

**F9 — Short-term gate weights.** Histogram of (w_t, w_{t-1}) collected on
a val pass. Test: did the symmetric gate collapse to w_t≈1? If mean
w_t ≈ 0.5 → gate is genuinely symmetric (the headline). If mean w_t > 0.9
→ collapse, and the B4/B2 effects are still asymmetric in disguise — also
reportable, as a negative-result twist on H4.

**F10 — Qualitative clips.** mp4 side-by-side. Pick 3 high-occlusion
sequences (0014, 0026, 0034). Caption: "B4 retains identity through the
synchronized turn at frame X where B0 reassigns track IDs."

## T8 (test-set) — not auto-generated

The DanceTrack test row (B4 seed-42/ck4) is the Codabench scorer output
from 2026-06-13. Hard-coded values for the report:

```
HOTA 68.077  DetA 80.582  AssA 57.679  MOTA 89.895  IDF1 70.999  IDSW 1142
```

A B0 same-pipeline anchor is still pending; the third row in T8 stays as
the published-release numbers with the caveat noted in `reporting_plan.md`
§9.

## Notes

- `pedestrian_detailed.csv` reports HOTA as `HOTA___AUC` in [0,1] — the
  per-sequence script multiplies by 100 in F8 to match table units (%).
- F2/F3 read whatever `.pth` files are present under `b3a/`, `b3b/`; we
  intentionally do NOT use the seed-template Hydra configs to pick ckpts.
- If `find_gate_theta` returns `None` in `_extract_lambda.py`, the
  state-dict key for the gate parameter differs from `*.theta` — open the
  checkpoint and inspect keys, then adjust the matcher.
- All scripts are idempotent: re-running overwrites their outputs.
