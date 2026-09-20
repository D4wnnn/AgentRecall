#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 VARIANT GPU_DIT GPU_VAE" >&2
  exit 2
fi
variant=$1
gpu_dit=$2
gpu_vae=$3
case "$variant" in
  agent_early|agent_late|agent_mixed) ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac
repo=/private/lc/others/LayerRecall
root=$repo/exp/07-agentic-within-event
cd "$repo"
source exp/env.sh
export EVAL_DATA_ROOT="$repo/exp/06-agentic-retrieval/prompts"
export EVAL_OUTPUT_DIR="$root/outputs/$variant"
export LR_LOG_PATH="$root/logs/$variant.jsonl"
mkdir -p "$EVAL_OUTPUT_DIR" "$root/logs"
export CUDA_VISIBLE_DEVICES="$gpu_dit,$gpu_vae"
exec /private/lc/envs/layerrecall/bin/python inference.py \
  --config_path "$root/configs/$variant.yaml"
