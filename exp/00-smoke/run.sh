#!/usr/bin/env bash
set -euo pipefail
cd /private/lc/others/LayerRecall
source exp/env.sh
export EVAL_DATA_ROOT=/private/lc/others/LayerRecall/exp/00-smoke/prompt.txt
export EVAL_OUTPUT_DIR=/private/lc/others/LayerRecall/exp/00-smoke/outputs
export LR_LOG_PATH=/private/lc/others/LayerRecall/exp/00-smoke/logs/router.jsonl
mkdir -p "$EVAL_OUTPUT_DIR" "$(dirname "$LR_LOG_PATH")"
exec /private/lc/envs/layerrecall/bin/python inference.py \
  --config_path exp/00-smoke/config.yaml

