#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 VARIANT GPU_DIT GPU_VAE PROMPT_ROOT [EXPERIMENT_ROOT]" >&2
  exit 2
fi
variant=$1
gpu_dit=$2
gpu_vae=$3
prompt_root=$4
experiment_root=${5:-/private/lc/others/LayerRecall/exp/01-routing-baselines}
case "$variant" in
  local_only|fixed10|all30) ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac
cd /private/lc/others/LayerRecall
source exp/env.sh
export EVAL_DATA_ROOT="$prompt_root"
export EVAL_OUTPUT_DIR="$experiment_root/outputs/$variant"
export LR_LOG_PATH="$experiment_root/logs/$variant.jsonl"
mkdir -p "$EVAL_OUTPUT_DIR" "$(dirname "$LR_LOG_PATH")"
export CUDA_VISIBLE_DEVICES="$gpu_dit,$gpu_vae"
exec /private/lc/envs/layerrecall/bin/python inference.py \
  --config_path "/private/lc/others/LayerRecall/exp/01-routing-baselines/configs/$variant.yaml"
