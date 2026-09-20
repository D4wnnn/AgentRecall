#!/usr/bin/env bash
set -euo pipefail
cd /private/lc/others/LayerRecall
source exp/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EVAL_DATA_ROOT=/private/lc/others/LayerRecall/exp/00-smoke/prompt.txt
export EVAL_OUTPUT_DIR=/private/lc/others/LayerRecall/exp/12-dynamic-state-chpm/analysis/smoke_outputs
export LR_LOG_PATH=/private/lc/others/LayerRecall/exp/12-dynamic-state-chpm/logs/dynamic_smoke_router.jsonl
mkdir -p "$EVAL_OUTPUT_DIR" "$(dirname "$LR_LOG_PATH")"
exec /private/lc/envs/layerrecall/bin/python inference.py \
  --config_path exp/12-dynamic-state-chpm/configs/dynamic_smoke.yaml
