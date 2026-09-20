#!/usr/bin/env bash
set -euo pipefail
cd /private/lc/others/LayerRecall
source exp/env.sh
export PATH=/private/lc/envs/layerrecall/bin:$PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CONFIG=/private/lc/others/LayerRecall/exp/14-dynamic-router-prior05/configs/dynamic_train.yaml
export RUN_NAME=dynamic_router_prior05
export LOGDIR=/private/lc/others/LayerRecall/exp/14-dynamic-router-prior05/checkpoints/dynamic_router_prior05
export NPROC_PER_NODE=4
export RESUME_MODE=none
export ENABLE_WANDB=0
exec ./scripts/train_chpm.sh
