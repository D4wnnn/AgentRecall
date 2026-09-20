# Initial screening results

Date: 2026-09-03. Hardware: 7 x NVIDIA L20 46GB. These are screening results,
not paper-level claims. All generated comparisons use the same seed and a reduced latent
spatial size of `24 x 40` so that the 5B model and 80-frame cache fit an L20.

## 00: Runtime validation

- Official Wan2.2, LongLive-2.0, and LayerRecall checkpoints load successfully.
- A 32-latent-frame run completed and produced 600 router events.
- The original inference path allocated positive and negative CFG caches even when
  `guidance_scale=1.0`. The local patch skips the unused negative cache in that case.
- Sixteen core LayerRecall tests passed before the patch; nine relevant tests passed after it.

## 01: Local versus fixed-10 versus all-30 routing

One released reappearance prompt was generated under all three policies. Low-cost temporal
diagnostics were:

| Policy | Mean frame change | p95 abrupt residual | 2-12Hz power ratio |
|---|---:|---:|---:|
| Local only | 0.02347 | 0.01646 | 0.76134 |
| Fixed 10 layers | 0.01436 | **0.01309** | 0.49683 |
| All 30 layers | 0.01433 | 0.01775 | **0.48795** |

Interpretation: fixed-10 has the lowest abrupt-change tail. All-layer routing has a similarly
low frequency ratio but more abrupt spikes. Mean frame change also drops under both memory
variants, so lower values must not automatically be interpreted as better; some of the change
may be motion suppression.

Across the three synthetic recall/update prompts, fixed-10 had lower p95 abrupt residual than
both alternatives in every case. Mean p95 abrupt residual was `0.01549` for fixed-10,
`0.01823` for local-only, and `0.02527` for all-30. This is a promising but very small-sample
signal that selective insertion matters.

## 02: Router confidence and temporal layer demand

The 48-chunk fixed-10 run produced 7,200 detailed router events. For active retrieval phases:

- normalized candidate entropy is approximately `0.896-0.902` across shots and layers;
- mean top-1 margins are typically only `0.0002-0.0024`;
- selected chunks change frequently within a layer (roughly `0.21-0.52` turnover for the
  fixed memory layers in this sample).

Interpretation: retrieval decisions are dynamic, but the score distribution is nearly flat.
This supports investigating uncertainty-aware routing, adaptive temperature, or richer memory
summaries. It does **not** yet prove dynamic layer activation is useful, because the log measures
selection behavior rather than the causal utility of enabling each layer. A phase-conditioned
layer-mask oracle remains the required evidence gate.

## 03: Recall versus legitimate state update

Qualitative frame inspection found:

- Recall case: all three variants restore a dark-haired person in red, but identity drifts in
  every variant; one seed is insufficient to rank them.
- Red-to-blue coat update: all variants returned in blue/purple rather than reverting to red.
  This single case does not support an obsolete-memory conflict claim.
- Vase relocation: all variants violated the intended table/shelf relation and introduced a
  camera-like object. Prompt-following failure confounds memory-validity analysis.

Conclusion: validity routing remains conceptually attractive, but the present three examples
do not establish it. The next dataset must make updates visually unambiguous and separately
score instruction following, identity recall, old-state reversion, and final-state correctness.

## Current innovation ranking

1. **Uncertainty/calibration before multi-prototype memory.** Near-uniform retrieval scores are
   the clearest measured weakness. Test adaptive temperature, confidence rejection, and
   query-key projection calibration before implementing semantic slots.
2. **Dynamic layer routing, conditional on an oracle gap.** Restrict early experiments to a
   profiled layer superset; routing across all 30 untrained insertion sites is unstable.
3. **Validity routing after a stronger benchmark.** Current synthetic evidence is inconclusive.
4. **Event memory and CHPM efficiency later.** They are higher-scope projects and should not
   precede the causal diagnostics above.

## 06--08: Agentic retrieval screening

A controlled Oracle-like planner was implemented to test semantic event selection before
building an LLM/VLM agent. At equal two-chunk and fixed-ten-layer budgets, it raised intended
first-event hit rate from 33.75% for cosine to 100%. Recent and random retrieval were worse
than cosine, confirming that history choice has causal impact.

The more important finding is that hard agent replacement is unsafe. At seed 0, early
first-event chunks `[2,1]` and an early/late mixture `[7,2]` retained the burgundy suitcase,
whereas late chunks `[7,6]` caused end-of-video color drift. At seed 1, forcing `[7,6]` caused
a clear identity change to a different dark-haired woman; early chunks retained identity,
while cosine remained at least as temporally stable and naturally used a varied mixture led
by chunk 2.

Decision: stop the hard chunk-ID agent baseline. Promote a hierarchical hybrid: semantic
event constraint/veto at the high level, diversity-aware key-state selection by the learned
router within the event, and uncertainty/validity gating before fixed-layer K/V injection.
Full details are in `06-agentic-retrieval/RESULTS.md`.

## 09--10: Demo-oriented paper-style cases

Thirty valid videos were generated over ten prompts, with matched baseline, LayerRecall,
and hybrid agent-event variants. The short-transition suite yielded two visually useful
cases; the longer 8/8/8 camera-away construction yielded three additional cases with a
genuine target-absent middle shot. Four strongest examples were exported as synchronized
three-column videos under `exp/demo_ready`.

The strongest result is the rooftop artist. Baseline never returns to the subject, ordinary
LayerRecall restores an easel but changes the artwork and loses the artist, while agent-event
returns to the correct first-shot event and restores the platinum-haired artist, violet beret,
orange jacket, and red sailboat painting. Flower-market, library, and clockmaker cases show
similar identity recovery or within-shot self-correction. All sampled return events in the
agent runs selected their two K/V chunks exclusively from Shot 1.

These are deliberately selected qualitative demos. Other generated prompts were retained,
including non-discriminative cases where the camera failed to leave the subject, so the demo
set should not be presented as a quantitative success rate.

## Important confounds

- Reduced spatial resolution differs from the paper.
- One seed and four prompts are not statistically meaningful.
- Pixel temporal diagnostics cannot measure semantic identity.
- The released 1,600-prompt CHPM training set is unavailable.
- The paper schedule (`24/12/12`) plus an 80-frame cache means non-sink first-shot chunks are
  evicted before shot 3; the permanently visible first sink chunk may carry much of the target.
  Experiment `05-cache-horizon-stress` should therefore be prioritized.
