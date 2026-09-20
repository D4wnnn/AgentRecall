# EventState-Recall: Current Experimental Status

Updated: 2026-09-05. All numbers below are measured outputs, not projected paper claims.

## Implemented pipeline

1. Qwen3-VL-4B observes labeled generated-frame montages after each completed event and
   produces auditable per-chunk observations plus an event summary.
2. Before the next event is generated, the same local model emits one typed action:
   `RECALL`, `KEEP_CURRENT`, `UPDATE`, or `IGNORE`.
3. `RECALL` selects an event, while LayerRecall independently selects full K/V chunks at
   each profiled DiT layer. An MMR-like penalty discourages two redundant slots.
4. An explicit empty `IGNORE` plan now abstains from long-range K/V instead of silently
   falling back to cosine retrieval. This prevents avoidable memory leakage during absence.
5. Every raw model response, validation attempt, structural normalization, event graph,
   compiled K/V plan, and router selection is logged.

This is currently an audited two-pass prototype: Qwen observes an already generated
baseline trace, then its plan drives a matched rerender. It is not yet an asynchronous
observer inside the single generation process.

## Four-case reappearance diagnostic

The cases are library archivist, railway traveler, rooftop artist, and seaside clockmaker.
All have an 8/8/8-chunk establish–absence–return structure. The oracle return event is
Shot 1.

| Agent diagnostic | Result |
|---|---:|
| Return action accuracy | 4/4 |
| Return event Hit@1 | 4/4 |
| Absence correctly treated as temporary | 4/4 |
| Historical chunk observation coverage | 96.9% |
| Native strict-JSON rate across raw attempts | 94.4% |
| Mean full observer time per 24-chunk video | 82.0 s |
| Mean replan-only time per video | 10.7 s |

The 96.9% coverage reflects the first rooftop trace created before exact chunk-ID checking
was added. New traces fail validation unless every requested chunk ID appears exactly once.
Container-type normalizations are logged and never change visual claims.

Router logs contain 320 sampled return decisions per case (10 layers × 8 chunks × 4
denoising steps). In all four cases:

- Shot 2 memory-use rate: 0%;
- Shot 3 memory-use rate: 100%;
- Shot 3 selected-event accuracy: 100%;
- two distinct K/V slots were selected per active decision.

## DINOv2 global-frame screening

These scores use mean full-frame DINOv2-large embeddings. They are useful for rapid
screening but confound subject, background, and camera; they are not object-crop scores and
must not be called official MemoBench.

| Method | Return cosine ↑ | Absence cosine ↓ | Return−absence gap ↑ | Adjacent smoothness ↑ |
|---|---:|---:|---:|---:|
| LongLive local baseline | 0.357 | 0.461 | -0.105 | 0.789 |
| LayerRecall | 0.474 | 0.467 | 0.008 | 0.734 |
| Oracle event mask | 0.711 | 0.493 | 0.218 | 0.727 |
| EventState-Recall (Qwen) | **0.698** | 0.540 | **0.158** | 0.700 |

On return cosine, Qwen closes about 94.5% of the oracle-mask gap over LayerRecall. The gain
is not uniform: it is strong on rooftop and clockmaker, positive on library, and negative on
traveler. The lower adjacent score and higher absence similarity are explicit warnings that
semantic recall can overconstrain motion or leave locally propagated objects even when
long-range memory is disabled.

For rooftop, the validity gate gives a controlled within-method comparison:

| Variant | Return cosine ↑ | Absence cosine ↓ | Gap ↑ |
|---|---:|---:|---:|
| Qwen plan without explicit abstention | 0.949 | 0.693 | 0.256 |
| Qwen plan with `IGNORE → no memory` | 0.913 | **0.552** | **0.361** |

The gate trades a small amount of raw recurrence for substantially cleaner separation
between absence and return, matching the intended behavior.

## What is and is not ready to claim

Supported now:

- semantic event selection is a real bottleneck: the oracle mask substantially beats flat
  LayerRecall on these selected cases;
- a local Qwen agent can recover the correct historical event in 4/4 reappearance cases;
- semantic event routing composes with layer-specific neural K/V selection;
- explicit abstention changes actual attention behavior and improves the rooftop
  absence/return separation.

Not supported yet:

- statistical superiority on the released 100-prompt bank or official MemoBench;
- fixed 80-frame/identical-byte memory superiority;
- robust state-update handling, until the red-to-blue and moved-object tests finish;
- less than 10% end-to-end overhead; the current synchronous 4B observer is too expensive;
- a complete semantic KEEP/MERGE/EVICT physical-cache implementation.

## State-update probe

For `red coat → explicit change to blue coat → same woman returns in blue`, Qwen emits
`UPDATE` for Shot 2 and correctly routes Shot 3 to Shot 2 (chunks 16–31), not to the obsolete
red-coat Shot 1. The 192-frame rerender visibly returns with a blue outer coat rather than
fully reverting to red. This validates the event-selection logic on one example.

It is not yet evidence of method superiority: a deliberately simple central-crop color
occupancy screen gives Shot-3 blue share 0.779 for baseline, 0.771 for LayerRecall, and 0.458
for EventState-Recall, partly because the EventState output preserves a large red inner shirt.
This screen is not segmented clothing accuracy, but the negative result means the state-update
claim must remain diagnostic until object masks or VLM/human labels are added across more
cases.

The next decision gate is therefore state validity and fixed-budget residency, not a larger
VLM. If the agent identifies the right updated event but generation still restores the old
state, injection/validity control is the bottleneck. If the correct event is no longer
resident at 80 frames, semantic anchor retention is the bottleneck.
