#!/usr/bin/env bash
set -euo pipefail

repo=/private/lc/others/LayerRecall
root=$repo/exp/11-eventstate-recall
source "$repo/exp/env.sh"
prompt_root="$root/prompts/update_clothing/caption/update_clothing"
mkdir -p "$prompt_root"
cp -a "$repo/exp/03-recall-update-conflict/prompts/caption/update_clothing/." "$prompt_root/"
export EVAL_DATA_ROOT="$root/prompts/update_clothing"
export EVAL_OUTPUT_DIR="$root/outputs/qwen_agent_update"
export LR_LOG_PATH="$root/logs/qwen_agent_update.jsonl"
export LR_AGENT_PLAN_PATH="$root/analysis/update_clothing_baseline/layer_recall_plan.json"
mkdir -p "$EVAL_OUTPUT_DIR" "$root/logs"
export CUDA_VISIBLE_DEVICES=0,1
cd "$repo"
exec /private/lc/envs/layerrecall/bin/python inference.py \
  --config_path "$root/config_qwen_update.yaml"
