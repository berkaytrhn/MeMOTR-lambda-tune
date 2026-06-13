# DanceTrack test-set evaluation & codabench submission

The DanceTrack **test** split has no public ground truth, so metrics (HOTA / MOTA /
IDF1) cannot be computed locally — you run the tracker to produce per-sequence
result files, then upload them to the [codabench.org](https://www.codabench.org)
DanceTrack test server, which scores them for you.

This is a **two-step** workflow:

| Step | Script | What it does |
| ---- | ------ | ------------ |
| 1. Run | [`run_test.sh`](run_test.sh) | Runs `main.py --mode submit` over the test split → writes `<SUBMIT_DIR>/test/tracker/<seq>.txt` |
| 2. Package | [`make_submission.py`](make_submission.py) | Validates the result files and zips them into `tracker.zip` in the codabench layout |

## Prerequisites

- A trained checkpoint placed at `<SUBMIT_DIR>/<SUBMIT_MODEL>` (e.g.
  `./outputs/memotr_dancetrack/memotr_dancetrack.pth`).
- The matching `<SUBMIT_DIR>/train/config.yaml` (the model architecture config,
  produced during training).
- The dataset laid out as `<DATA_ROOT>/DanceTrack/test/<seq>/img1/*.jpg`
  (use `tools/download_dancetrack.py` if you don't have it yet).

## Step 1 — run inference on the test set

```bash
DATA_ROOT=./dataset \
SUBMIT_DIR=./outputs/memotr_dancetrack \
SUBMIT_MODEL=memotr_dancetrack.pth \
bash scripts/dancetrack_test/run_test.sh
```

All knobs are environment variables (defaults in the script): `DATA_ROOT`,
`SUBMIT_DIR`, `SUBMIT_MODEL`, `CONFIG_PATH`, `SUBMIT_DATA_SPLIT`,
`AVAILABLE_GPUS`, `NPROC`. On a single GPU leave `NPROC=1`; for multi-GPU
distributed inference set e.g. `NPROC=8 AVAILABLE_GPUS=0,1,2,3,4,5,6,7`.

Output: `<SUBMIT_DIR>/test/tracker/<seq>.txt`, one MOT-format file per sequence:

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,1,-1,-1,-1
```

## Step 2 — package for codabench

```bash
python scripts/dancetrack_test/make_submission.py \
    --submit-dir ./outputs/memotr_dancetrack \
    --split test \
    --data-root ./dataset      # optional: verifies every test seq has a result
```

This writes `<SUBMIT_DIR>/test/tracker.zip` with the **required** layout:

```
tracker.zip
└── tracker/
    ├── dancetrack0003.txt
    ├── dancetrack0009.txt
    └── ...
```

The script forces the internal folder name to `tracker` (a hard codabench
requirement), checks every line has the 10 expected fields, and — when
`--data-root` is given — refuses to build an incomplete submission unless you
pass `--allow-missing`.

Upload the resulting `tracker.zip` to the DanceTrack test competition on
codabench.org.

## Tuning

Tracking thresholds used at inference time come from `CONFIG_PATH`
(default `configs/train_dancetrack.yaml`): `DET_SCORE_THRESH`,
`TRACK_SCORE_THRESH`, `RESULT_SCORE_THRESH`, `MISS_TOLERANCE`. Adjust there if
you want to trade precision/recall before generating a submission.
