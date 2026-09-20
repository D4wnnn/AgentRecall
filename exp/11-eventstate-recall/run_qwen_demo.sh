#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU_DIT GPU_VAE CASE_INDEX ANALYSIS_NAME OUTPUT_VARIANT" >&2
  exit 2
fi

gpu_dit=$1
gpu_vae=$2
case_index=$3
analysis_name=$4
output_variant=$5
repo=/private/lc/others/LayerRecall
root=$repo/exp/11-eventstate-recall
source "$repo/exp/env.sh"

case_name=$(printf 'case_%04d' "$case_index")
prompt_root="$root/prompts/$case_name/caption"
mkdir -p "$prompt_root"
mkdir -p "$prompt_root/$case_name"
cp -a "$repo/exp/10-demo-long-transition/prompts/caption/$case_name/." "$prompt_root/$case_name/"

export EVAL_DATA_ROOT="$root/prompts/$case_name"
export EVAL_OUTPUT_DIR="$root/outputs/$output_variant/$case_name"
export LR_LOG_PATH="$root/logs/${output_variant}_${case_name}.jsonl"
export LR_AGENT_PLAN_PATH="$root/analysis/$analysis_name/layer_recall_plan.json"
mkdir -p "$EVAL_OUTPUT_DIR" "$root/logs"
export CUDA_VISIBLE_DEVICES="$gpu_dit,$gpu_vae"
cd "$repo"
exec /private/lc/envs/layerrecall/bin/python inference.py \
  --config_path "$root/config_qwen_demo.yaml"
