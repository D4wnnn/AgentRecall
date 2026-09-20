# Temporal layer-demand analysis

Use the detailed router log produced by `01-routing-baselines/fixed10` and run:

```bash
/private/lc/envs/layerrecall/bin/python exp/analyze_router_log.py \
  exp/01-routing-baselines/logs/fixed10.jsonl \
  --output-dir exp/02-temporal-layer-demand/analysis/fixed10
```

The first pass measures selection turnover, uncertainty, and temporal distance by layer and
chunk. This does not yet prove that dynamically enabling a layer improves prediction; it is
a screening diagnostic. The next decisive experiment is a same-checkpoint layer-mask oracle
over phase-specific masks at equal average active-layer budget.

