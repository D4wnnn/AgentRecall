#!/usr/bin/env bash
set -euo pipefail
cd /private/lc/others/LayerRecall
source exp/env.sh
export PATH=/private/lc/envs/layerrecall/bin:$PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CONFIG=/private/lc/others/LayerRecall/exp/12-dynamic-state-chpm/configs/dynamic_train_smoke.yaml
export RUN_NAME=dynamic_router_grad_smoke
export LOGDIR=/private/lc/others/LayerRecall/exp/12-dynamic-state-chpm/checkpoints/dynamic_router_grad_smoke
export NPROC_PER_NODE=4
export RESUME_MODE=none
export ENABLE_WANDB=0
exec ./scripts/train_chpm.sh
