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
  cosine) config=agent_early.yaml; policy=cosine ;;
  agent_early) config=agent_early.yaml; policy=agent_plan ;;
  agent_late) config=agent_late.yaml; policy=agent_plan ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac
repo=/private/lc/others/LayerRecall
root=$repo/exp/08-agentic-seed1-replication
cd "$repo"
source exp/env.sh
export EVAL_DATA_ROOT="$repo/exp/06-agentic-retrieval/prompts"
export EVAL_OUTPUT_DIR="$root/outputs/$variant"
export LR_LOG_PATH="$root/logs/$variant.jsonl"
export LR_POLICY="$policy"
export LR_SEED=1
mkdir -p "$EVAL_OUTPUT_DIR" "$root/logs"
export CUDA_VISIBLE_DEVICES="$gpu_dit,$gpu_vae"
exec /private/lc/envs/layerrecall/bin/python inference.py \
  --config_path "$root/configs/$config"
