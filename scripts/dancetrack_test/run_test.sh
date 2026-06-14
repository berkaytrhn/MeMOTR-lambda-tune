#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Step 1/2 — Run MeMOTR inference on the DanceTrack TEST split.
#
# This drives `main.py --mode submit`, which loads the trained checkpoint, runs
# the tracker over every test sequence, and writes one MOT-format result file
# per sequence to:
#
#     <SUBMIT_DIR>/<SUBMIT_DATA_SPLIT>/tracker/<seq_name>.txt
#
# Each line is written as:
#     <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,1,-1,-1,-1
# which is exactly the format required by the DanceTrack codabench.org server.
#
# After this finishes, run step 2 to package the results:
#     python scripts/dancetrack_test/make_submission.py --submit-dir <SUBMIT_DIR>
#
# The DanceTrack test set has NO public ground truth, so HOTA/MOTA cannot be
# computed locally — you must submit the zip to the evaluation server.
# ------------------------------------------------------------------------------
set -euo pipefail

# ---- Configuration (override via environment variables or edit below) --------
# Root that contains the dataset, i.e. <DATA_ROOT>/DanceTrack/test/<seq>/img1/*.jpg
DATA_ROOT="${DATA_ROOT:-./dataset}"
# Output dir of the trained model; MUST contain train/config.yaml and the checkpoint.
SUBMIT_DIR="${SUBMIT_DIR:-./outputs/memotr_dancetrack_b4}"
# Checkpoint filename, relative to SUBMIT_DIR (e.g. memotr_dancetrack.pth).
SUBMIT_MODEL="${SUBMIT_MODEL:-memotr_dancetrack.pth}"
# Runtime/threshold config (provides DET/TRACK/RESULT score thresholds, etc.).
CONFIG_PATH="${CONFIG_PATH:-./configs/train_dancetrack.yaml}"
# Data split to run. Keep "test" for codabench submission ("val" for local eval).
SUBMIT_DATA_SPLIT="${SUBMIT_DATA_SPLIT:-test}"
# GPUs. Single RTX 5090 box -> "0". For multi-GPU distributed, see NPROC below.
AVAILABLE_GPUS="${AVAILABLE_GPUS:-0}"
# Number of processes for distributed inference. 1 = simple single-GPU run.
NPROC="${NPROC:-1}"
# ------------------------------------------------------------------------------

# Resolve repo root (this script lives in scripts/dancetrack_test/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- Sanity checks -----------------------------------------------------------
TEST_DIR="${DATA_ROOT}/DanceTrack/${SUBMIT_DATA_SPLIT}"
if [[ ! -d "${TEST_DIR}" ]]; then
  echo "ERROR: test data dir not found: ${TEST_DIR}" >&2
  echo "       Expected layout: <DATA_ROOT>/DanceTrack/${SUBMIT_DATA_SPLIT}/<seq>/img1/*.jpg" >&2
  exit 1
fi
if [[ ! -f "${SUBMIT_DIR}/train/config.yaml" ]]; then
  echo "ERROR: ${SUBMIT_DIR}/train/config.yaml not found (model architecture config)." >&2
  exit 1
fi
if [[ ! -f "${SUBMIT_DIR}/${SUBMIT_MODEL}" ]]; then
  echo "ERROR: checkpoint not found: ${SUBMIT_DIR}/${SUBMIT_MODEL}" >&2
  echo "       Set SUBMIT_MODEL to one of the checkpoints in ${SUBMIT_DIR}:" >&2
  ( cd "${SUBMIT_DIR}" && ls -1 *.pth 2>/dev/null | sed 's/^/         - /' >&2 ) \
    || echo "         (no .pth files found there)" >&2
  echo "       e.g.  SUBMIT_MODEL=checkpoint_4.pth $0" >&2
  exit 1
fi

echo "=============================================================="
echo " MeMOTR DanceTrack ${SUBMIT_DATA_SPLIT}-set inference"
echo "   DATA_ROOT    : ${DATA_ROOT}"
echo "   SUBMIT_DIR   : ${SUBMIT_DIR}"
echo "   SUBMIT_MODEL : ${SUBMIT_MODEL}"
echo "   CONFIG_PATH  : ${CONFIG_PATH}"
echo "   SPLIT        : ${SUBMIT_DATA_SPLIT}"
echo "   GPUS         : ${AVAILABLE_GPUS} (nproc=${NPROC})"
echo "   -> results   : ${SUBMIT_DIR}/${SUBMIT_DATA_SPLIT}/tracker/"
echo "=============================================================="

COMMON_ARGS=(
  --mode submit
  --config-path "${CONFIG_PATH}"
  --submit-dir "${SUBMIT_DIR}"
  --submit-model "${SUBMIT_MODEL}"
  --submit-data-split "${SUBMIT_DATA_SPLIT}"
  --data-root "${DATA_ROOT}"
  --available-gpus "${AVAILABLE_GPUS}"
)

if [[ "${NPROC}" -gt 1 ]]; then
  # Multi-GPU: sequences are sharded across processes.
  python -m torch.distributed.run --nproc_per_node="${NPROC}" main.py \
    "${COMMON_ARGS[@]}" --use-distributed
else
  # Single-GPU, non-distributed.
  python main.py "${COMMON_ARGS[@]}"
fi

echo
echo "Done. Per-sequence results written to: ${SUBMIT_DIR}/${SUBMIT_DATA_SPLIT}/tracker/"
echo "Next: package for codabench with"
echo "  python scripts/dancetrack_test/make_submission.py --submit-dir \"${SUBMIT_DIR}\" --split \"${SUBMIT_DATA_SPLIT}\" --data-root \"${DATA_ROOT}\""
