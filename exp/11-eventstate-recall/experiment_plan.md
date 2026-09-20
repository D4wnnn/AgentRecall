# Fixed-Budget Evaluation Plan

## Main methods

1. LongLive-2.0 local/sink baseline.
2. Released LayerRecall.
3. EventState-Recall.

All use matched prompts, seeds, resolution, two visible historical chunks, the same ten
insertion layers, and an 80-frame or exactly matched byte-size physical cache.

## Main data

- Released LayerRecall 100-prompt bank: subject, color, position, and scene axes.
- Official MemoBench as external generalization; do not confuse its 360 V-D-R clips with the
  paper's 100 T2V prompts.
- Existing selected demos are qualitative only and never enter the main quantitative mean.

## Core ablations

- prompt-only Oracle event mask versus observed-video event mask;
- hard exact-chunk agent versus hard event mask versus soft event prior;
- no event graph, no state version, no diversity, and no validity gate;
- FIFO/LRU versus semantic KEEP/MERGE/EVICT at equal bytes;
- always invoke Qwen versus uncertainty-triggered Qwen;
- Qwen3-VL 2B/4B/8B scaling and a cross-architecture observer on a subset.

## Metrics

- identity and return-crop DINOv2/SigLIP similarity;
- object return detection and crop similarity;
- new-state correctness and old-state leakage, reported separately;
- layout/camera consistency and VBench-Long quality dimensions;
- event Hit@1/2, MRR, obsolete retrieval rate, residency coverage;
- state action F1, ECE/Brier calibration, abstention coverage/accuracy;
- latency, peak VRAM, cache bytes, and VLM calls per 100 chunks.

Local Qwen VLM scores must be named `Qwen-MemoScore` or `local MemoBench-style`; they are not
the paper's Gemini-based official score. The observer and evaluator should use different
model families where possible.

## Staged execution

1. 30 stratified prompts × 3 methods × 2 seeds.
2. If the causal gate passes, 100 prompts × 3 methods × one matched seed.
3. Add two more seeds on 30 hard prompts.
4. Run ablations on the fixed 30-prompt subset.

## Predeclared gates

- Stop agent work if an Oracle event mask improves the primary return metric by less than
  0.02 over LayerRecall.
- If online agent recovers less than 50% of the Oracle gap, improve the observer before the
  generator integration.
- If Hit@2 exceeds 80% but generation does not improve, work on within-event selection and
  injection, not a larger VLM.
- If semantic gain reduces motion smoothness by more than 1% or dynamic degree by more than
  5%, strengthen abstention/validity gating.
- Final claim requires paired-bootstrap 95% CI above zero on the memory macro average,
  improvement on at least three of five memory dimensions, VBench-Long non-inferiority
  within -0.01, and preferably less than 10% end-to-end overhead.
