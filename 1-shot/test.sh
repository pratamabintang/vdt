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

# ------------------------------- SAM2 encoder ------------------------------
# Only used when BACKBONE=sam2_hiera_{t,s,b+,l}. Must match the config used in
# training; SAM2_CKPT is optional here (best_model.pt already holds the weights).
SAM2_CONFIG="${SAM2_CONFIG:-}"   # e.g. sam2_hiera_l.yaml
# Checkpoint path: SAM2_CKPT wins; else HIERA_PATH; else HIERA_LARGE_PATH; else default.
# Not strictly needed at test time (best_model.pt already holds the weights).
HIERA_PATH="${HIERA_PATH:-${HIERA_LARGE_PATH:-/content/drive/MyDrive/project/checkpoints/sam2_hiera_large.pt}}"
SAM2_CKPT="${SAM2_CKPT:-$HIERA_PATH}"   # e.g. /content/.../sam2_hiera_large.pt (SAM2 v2.0)
SAM2_ADAPTERS="${SAM2_ADAPTERS:-1}"

# At test time the trunk weights come from best_model.pt, so a missing checkpoint
# is not fatal: drop the flag rather than fail on the FileNotFoundError check.
if [ -n "$SAM2_CKPT" ] && [ ! -f "$SAM2_CKPT" ]; then
  echo "[test.sh] note: SAM2_CKPT not found ($SAM2_CKPT); loading weights from best_model.pt only." >&2
  SAM2_CKPT=""
fi

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

SAM2_FLAGS=""
if [ -n "$SAM2_CONFIG" ]; then SAM2_FLAGS="$SAM2_FLAGS --sam2-config $SAM2_CONFIG"; fi
if [ -n "$SAM2_CKPT" ]; then SAM2_FLAGS="$SAM2_FLAGS --sam2-ckpt $SAM2_CKPT"; fi
SAM2_FLAGS="$SAM2_FLAGS --sam2-adapters $SAM2_ADAPTERS"

if [ -z "$LOAD" ]; then
  echo "[test.sh] ERROR: LOAD is empty. Set LOAD=<training log dir with best_model.pt>" >&2
  exit 1
fi

echo "[test.sh] test=$TEST_ROOT"
echo "[test.sh] blacklist=$BLACKLIST"
echo "[test.sh] load=$LOAD"
echo "[test.sh] model=$MODEL backbone=$BACKBONE size=$SIZE"
echo "[test.sh] amp=$AMP dtype=$AMP_DTYPE"
if [ "$SAM2_ADAPTERS" != "0" ] || [ -n "$SAM2_CONFIG" ] || [ -n "$SAM2_CKPT" ]; then
  echo "[test.sh] sam2 config=${SAM2_CONFIG:-<backbone>.yaml} ckpt=${SAM2_CKPT:-none} adapters=$SAM2_ADAPTERS"
fi

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
  $SAM2_FLAGS \
  $LOGPATH_FLAG
