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
BSZ="${BSZ:-4}"

echo "[sweep] episodes=$EPISODES_LIST  epoch=$EPOCH  bsz=$BSZ"

for E in $EPISODES_LIST; do
  echo ""
  echo "=============================================================="
  echo "===== RUN: EPISODES=$E  (log = logs/ep${E}_*_IFCNet.log) ====="
  echo "=============================================================="
  EPISODES="$E" \
  EPOCH="$EPOCH" \
  BSZ="$BSZ" \
  LOGPATH="ep${E}" \
  bash train.sh
done

echo ""
echo "[sweep] done. Compare the '*** Training/Validation [@Epoch ..]' lines in:"
echo "[sweep]   logs/ep0_IFCNet.log/log.txt"
echo "[sweep]   logs/ep10_IFCNet.log/log.txt"
echo "[sweep]   logs/ep100_IFCNet.log/log.txt"
