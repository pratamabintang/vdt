#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Few-shot landslide segmentation (1-shot, RGB + DTM) - training launcher.
#
# Every knob lives here. Override any value from the shell, e.g.:
#     TRAIN_ROOT=/content/... VAL_ROOT=/content/... BLACKLIST=/content/... bash train.sh
#     EPOCH=50 BSZ=2 AMP=0 bash train.sh
#
# Run from anywhere; the script cd's into 1-shot/ first.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# ------------------------------- data paths --------------------------------
# Each split lives in a different location. Point each var at either
#   <dir>/{image,dtm,label}          (the split folder itself), or
#   <dir>/{train,val}/{image,...}    (a base dir containing the split subfolder)
TRAIN_ROOT="${TRAIN_ROOT:-/content/drive/MyDrive/project/3.TRAINING}"
VAL_ROOT="${VAL_ROOT:-/content/drive/MyDrive/project/4.VALIDATION}"
# One blacklist covering broken tiles in train, val and test.
# Default falls back to $BLACK_LIST_PATH; at least one must be set.
BLACKLIST="${BLACKLIST:-${BLACK_LIST_PATH:-}}"

# ------------------------------- model -------------------------------------
MODEL="${MODEL:-IFCNet}"
BACKBONE="${BACKBONE:-resnet50}"
SIZE="${SIZE:-512}"
DROP="${DROP:-0.2}"

# ------------------------------- SAM2 encoder ------------------------------
# Only used when BACKBONE=sam2_hiera_{t,s,b+,l}. Leave SAM2_CONFIG empty to use
# "<backbone>.yaml" (resolves to the vendored SAM2 config automatically).
SAM2_CONFIG="${SAM2_CONFIG:-}"   # e.g. sam2_hiera_l.yaml
# Checkpoint path: SAM2_CKPT wins; else HIERA_PATH; else HIERA_LARGE_PATH; else default.
HIERA_PATH="${HIERA_PATH:-${HIERA_LARGE_PATH:-/content/drive/MyDrive/project/checkpoints/sam2_hiera_large.pt}}"
SAM2_CKPT="${SAM2_CKPT:-$HIERA_PATH}"   # e.g. /content/.../sam2_hiera_large.pt (SAM2 v2.0)
SAM2_ADAPTERS="${SAM2_ADAPTERS:-1}"  # 1 = train wide adapters, 0 = freeze trunk fully

# ------------------------------- optim -------------------------------------
BSZ="${BSZ:-4}"
EPISODES="${EPISODES:-0}"        # query episodes per epoch (0 = all positive tiles)
LR="${LR:-1e-4}"
EPOCH="${EPOCH:-200}"
NWORKER="${NWORKER:-2}"
AUXLOSS="${AUXLOSS:-0.4}"        # weight of coarse-scale losses in deep supervision

# ------------------------------- AMP ---------------------------------------
# L4 / Ada: keep AMP=1 AMP_DTYPE=bf16 (no GradScaler needed).
AMP="${AMP:-1}"                  # 1 = enable mixed precision, 0 = full fp32
AMP_DTYPE="${AMP_DTYPE:-bf16}"   # bf16 | fp16

# ------------------------------- run control -------------------------------
RESUME="${RESUME:-}"             # path to a previous log dir (contains best_model.pt)
LOGPATH="${LOGPATH:-}"           # custom log folder name (empty -> timestamped)

# ------------------------------- build flags -------------------------------
AMP_FLAG=""
if [ "$AMP" = "1" ]; then AMP_FLAG="--amp --amp-dtype $AMP_DTYPE"; fi

RESUME_FLAG=""
if [ -n "$RESUME" ]; then RESUME_FLAG="--resume $RESUME"; fi

LOGPATH_FLAG=""
if [ -n "$LOGPATH" ]; then LOGPATH_FLAG="--logpath $LOGPATH"; fi

SAM2_FLAGS=""
if [ -n "$SAM2_CONFIG" ]; then SAM2_FLAGS="$SAM2_FLAGS --sam2-config $SAM2_CONFIG"; fi
if [ -n "$SAM2_CKPT" ]; then SAM2_FLAGS="$SAM2_FLAGS --sam2-ckpt $SAM2_CKPT"; fi
SAM2_FLAGS="$SAM2_FLAGS --sam2-adapters $SAM2_ADAPTERS"

echo "[train.sh] train=$TRAIN_ROOT"
echo "[train.sh] val  =$VAL_ROOT"
echo "[train.sh] blacklist=$BLACKLIST"
echo "[train.sh] model=$MODEL backbone=$BACKBONE size=$SIZE bsz=$BSZ episodes=$EPISODES lr=$LR epoch=$EPOCH"
echo "[train.sh] amp=$AMP dtype=$AMP_DTYPE auxloss=$AUXLOSS"
if [ "$SAM2_ADAPTERS" != "0" ] || [ -n "$SAM2_CONFIG" ] || [ -n "$SAM2_CKPT" ]; then
  echo "[train.sh] sam2 config=${SAM2_CONFIG:-<backbone>.yaml} ckpt=${SAM2_CKPT:-none} adapters=$SAM2_ADAPTERS"
fi

if [ -z "$BLACKLIST" ]; then
  echo "[train.sh] WARNING: BLACKLIST empty (set BLACKLIST or BLACK_LIST_PATH); broken tiles will NOT be skipped." >&2
fi

python train.py \
  --traindata "$TRAIN_ROOT" \
  --valdata "$VAL_ROOT" \
  --blacklist "$BLACKLIST" \
  --model "$MODEL" \
  --backbone "$BACKBONE" \
  --size "$SIZE" \
  --bsz "$BSZ" \
  --episodes "$EPISODES" \
  --lr "$LR" \
  --epoch "$EPOCH" \
  --nworker "$NWORKER" \
  --drop "$DROP" \
  --auxloss "$AUXLOSS" \
  $AMP_FLAG \
  $SAM2_FLAGS \
  $RESUME_FLAG \
  $LOGPATH_FLAG
