#!/usr/bin/env bash
set -euo pipefail
cd /private/lc/others/LayerRecall
experiment_root=/private/lc/others/LayerRecall/exp/03-recall-update-conflict
prompt_root="$experiment_root/prompts"
mkdir -p "$experiment_root/logs"
nohup bash exp/01-routing-baselines/run_variant.sh local_only 0 1 "$prompt_root" "$experiment_root" \
  > "$experiment_root/logs/local_only.console.log" 2>&1 &
echo "local_only pid=$! GPUs=0,1"
nohup bash exp/01-routing-baselines/run_variant.sh fixed10 2 3 "$prompt_root" "$experiment_root" \
  > "$experiment_root/logs/fixed10.console.log" 2>&1 &
echo "fixed10 pid=$! GPUs=2,3"
nohup bash exp/01-routing-baselines/run_variant.sh all30 4 5 "$prompt_root" "$experiment_root" \
  > "$experiment_root/logs/all30.console.log" 2>&1 &
echo "all30 pid=$! GPUs=4,5"

