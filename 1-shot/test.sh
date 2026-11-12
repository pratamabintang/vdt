#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Few-shot landslide segmentation (1-shot, RGB + DTM) - test launcher.
#
# Every knob lives here. Override any value from the shell, e.g.:
#     TEST_ROOT=/content/.../5.TESTING LOAD=/content/.../logs/<run> bash test.sh
#     AMP=0 bash test.sh
#
# Load must point at a training log dir that contains best_model.pt
# (where Logger.save_model_miou wrote it), not at saved_model/.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# ------------------------------- data paths --------------------------------
# May point at either <dir>/{image,dtm,label} or a base dir with a test/ subfolder.
TEST_ROOT="${TEST_ROOT:-/content/drive/MyDrive/project/5.TESTING}"
# One blacklist covering broken tiles in train, val and test.
BLACKLIST="${BLACKLIST:-${BLACK_LIST_PATH:-}}"

# ------------------------------- model -------------------------------------
MODEL="${MODEL:-IFCNet}"
BACKBONE="${BACKBONE:-resnet50}"
SIZE="${SIZE:-512}"
DROP="${DROP:-0.2}"

# ------------------------------- run control -------------------------------
# Directory produced by training that holds best_model.pt.
LOAD="${LOAD:-}"                 # e.g. logs/_MMDD_HHMMSS_IFCNet.log
NWORKER="${NWORKER:-0}"
LOGPATH="${LOGPATH:-}"           # custom log folder name (empty -> timestamped)

# ------------------------------- AMP ---------------------------------------
AMP="${AMP:-1}"                  # 1 = enable mixed precision, 0 = full fp32
AMP_DTYPE="${AMP_DTYPE:-bf16}"   # bf16 | fp16

# ------------------------------- build flags -------------------------------
AMP_FLAG=""
if [ "$AMP" = "1" ]; then AMP_FLAG="--amp --amp-dtype $AMP_DTYPE"; fi

LOGPATH_FLAG=""
if [ -n "$LOGPATH" ]; then LOGPATH_FLAG="--logpath $LOGPATH"; fi

if [ -z "$LOAD" ]; then
  echo "[test.sh] ERROR: LOAD is empty. Set LOAD=<training log dir with best_model.pt>" >&2
  exit 1
fi

echo "[test.sh] test=$TEST_ROOT"
echo "[test.sh] blacklist=$BLACKLIST"
echo "[test.sh] load=$LOAD"
echo "[test.sh] model=$MODEL backbone=$BACKBONE size=$SIZE"
echo "[test.sh] amp=$AMP dtype=$AMP_DTYPE"

if [ -z "$BLACKLIST" ]; then
  echo "[test.sh] WARNING: BLACKLIST empty (set BLACKLIST or BLACK_LIST_PATH); broken tiles will NOT be skipped." >&2
fi

python test.py \
  --testdata "$TEST_ROOT" \
  --blacklist "$BLACKLIST" \
  --load "$LOAD" \
  --model "$MODEL" \
  --backbone "$BACKBONE" \
  --size "$SIZE" \
  --nworker "$NWORKER" \
  --drop "$DROP" \
  $AMP_FLAG \
  $LOGPATH_FLAG
