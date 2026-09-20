# LayerRecall feasibility experiments

This directory is intentionally organized by hypothesis. Every experiment must keep its
configuration, prompt manifest, raw logs, generated videos, and analysis outputs together.

## Environment and released assets

- Conda environment: `/private/lc/envs/layerrecall`
- Wan2.2 root: `/private/lc/download/Wan2.2-TI2V-5B`
- LongLive-2.0 checkpoint: `/private/lc/download/LongLive-2.0-5B/model_bf16.pt`
- LayerRecall checkpoint: `/private/lc/download/LayerRecall/layer_recall_chpm_v3_step200.pt`
- Released evaluation prompts: repository `examples/prompts/layerrecall_100cases`

The paper's 1,600 CHPM training prompts are not released. Do not label the included
100-case prompt bank as training data.

The first L20 screening pass uses latent spatial size `24 x 40` (approximately
`384 x 640` decoded) because the paper's `44 x 80` H100 configuration exceeds a
46GB L20 during the 80-frame K/V-cache allocation. Confirmed ideas must later be
revalidated at paper resolution on a larger-memory GPU or with sequence/model sharding.

## Experiment map

| Directory | Question | First decision criterion |
|---|---|---|
| `00-smoke` | Does the official checkpoint run correctly on L20? | A video/latent output and a non-empty router log |
| `01-routing-baselines` | Are selected layers better than local-only and all-layer memory? | Same prompt/seed/budget comparison |
| `02-temporal-layer-demand` | Does the useful layer pattern change within a video? | Chunk-by-layer routing heatmap and phase-conditioned variation |
| `03-recall-update-conflict` | Does old memory help recall but hurt legitimate updates? | Recall and update cases move in opposite directions |
| `04-multi-prototype-diagnostic` | Does mean pooling dilute small or multiple entities? | Retrieval confidence drops as distractors/entities increase |
| `05-cache-horizon-stress` | Is the apparent long-range recall mostly due to the fixed sink? | Performance under early, middle, and cache-evicted target placement |
| `06-agentic-retrieval` | Does an Oracle-like semantic plan causally improve chunk retrieval? | Intended-event hit rate and equal-budget generation quality |
| `07-agentic-within-event` | Which key states within an event should be recalled? | Early vs late vs mixed two-chunk plans |
| `08-agentic-seed1-replication` | Does the within-event finding survive a second seed? | Identity/attribute retention and temporal diagnostics |
| `09-demo-reappearance` | Which paper-style three-shot cases give visible demo differences? | Six cases × baseline/LayerRecall/agent-event |
| `10-demo-long-transition` | Can longer camera-away shots produce clean target absence? | Four effective 8/8/8 cases × three methods |
| `demo_ready` | Presentation-ready synchronized comparisons | Four labeled side-by-side MP4 files |

## Evidence gates

1. Do not implement a trainable dynamic layer router until `02` shows meaningful
   within-video layer variation or an oracle gap over the fixed ten-layer policy.
2. Promote validity/conflict routing if `03` shows that stronger memory improves recall
   while reverting an explicitly updated state.
3. Promote multi-prototype summaries only if `04` shows a repeatable loss of retrieval
   confidence or wrong-chunk selection in multi-entity cases.
4. Always compare at equal visible-token and average active-layer budgets.

## Layout convention

Each numbered directory contains `configs/`, `prompts/`, `logs/`, `outputs/`, and
`analysis/` as needed. Generated files are ignored by Git through the repository's
existing output/checkpoint rules; large model assets remain outside this repository.
