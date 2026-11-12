#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Episode-budget sweep: train the SAME config with EPISODES = 0, 10, 100 to
# see how the per-epoch number of query episodes affects training.
#
# Run on Colab (needs the real data + GPU):
#     EPOCH=20 bash 1-shot/sweep_episodes.sh
#     BLACK_LIST_PATH=/content/.../black_list.txt EPOCH=50 bash 1-shot/sweep_episodes.sh
#
# Each run writes its own log dir: logs/ep0_<model>.log, logs/ep10_..., logs/ep100_...
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# Episodes to sweep (space-separated). 0 = one full pass over positive tiles.
EPISODES_LIST="${EPISODES_LIST:-0 10 100}"
# Epochs per run (keep small for a quick comparison).
EPOCH="${EPOCH:-20}"
# Batch size used for every run.
BSZ="${BSZ:-16}"
LR="${LR:-3e-4}"
LOSS="${LOSS:-tversky}"

echo "[sweep] episodes=$EPISODES_LIST  epoch=$EPOCH  bsz=$BSZ  lr=$LR  loss=$LOSS  backbone=${BACKBONE:-resnet50}"

# An episode budget below the batch size yields 0 batches -> the run would train
# nothing (and silently report a random model). Fail fast instead.
for E in $EPISODES_LIST; do
  if [ "$E" != "0" ] && [ "$E" -lt "$BSZ" ]; then
    echo "[sweep] ERROR: EPISODES=$E < BSZ=$BSZ -> 0 batches/epoch (no training)." >&2
    echo "[sweep]        Lower BSZ (e.g. BSZ=$E) or raise the episode budget." >&2
    exit 1
  fi
done

for E in $EPISODES_LIST; do
  echo ""
  echo "=============================================================="
  echo "===== RUN: EPISODES=$E  (log = logs/ep${E}_*_IFCNet.log) ====="
  echo "=============================================================="
  EPISODES="$E" \
  EPOCH="$EPOCH" \
  BSZ="$BSZ" \
  LR="$LR" \
  LOSS="$LOSS" \
  LOGPATH="ep${E}" \
  bash train.sh
done

echo ""
echo "[sweep] done. Compare the '*** Training/Validation [@Epoch ..]' lines in:"
echo "[sweep]   logs/ep0_IFCNet.log/log.txt"
echo "[sweep]   logs/ep10_IFCNet.log/log.txt"
echo "[sweep]   logs/ep100_IFCNet.log/log.txt"
