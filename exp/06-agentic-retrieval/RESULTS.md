# Results: controlled agentic retrieval

Date: 2026-09-04. Hardware: NVIDIA L20 46GB. These are reduced-resolution screening
experiments at latent size 24 x 40, four denoising steps, and one prompt with two seeds.

## Experimental control

The video has three prompt phases of 8 / 4 / 8 chunks. The distinctive traveler appears
in phase one, is requested to leave in phase two, and returns in phase three. All retrieval
variants use the released checkpoint, the same fixed ten memory-sensitive layers, and two
visible history chunks per active layer. Only history ranking changes.

Experiment 06 used a 112-frame physical cache and compared cosine, recent, deterministic
random, and a controlled agent plan preferring first-event chunks 7 and 6. The plan is an
Oracle-like diagnostic, not an LLM/VLM agent.

## Retrieval intervention check

Across 80 active return-shot events sampled at the last denoising step:

| Policy | Any target hit | Both targets hit | Both in post-policy candidate pool |
|---|---:|---:|---:|
| Cosine | 33.75% | 2.50% | 52.50% |
| Recent | 0.00% | 0.00% | 37.50% |
| Random | 17.50% | 0.00% | 23.75% |
| Agent plan | **100.00%** | **87.50%** | 87.50% |

The intervention therefore changed the actual visible K/V chunks as intended. Inspection of
the pre-pool filter counters and cache schedule shows that the difference between 87.5% and
100% is physical eviction of chunk 6 in the final return chunk, not a planning failure.

## Seed-0 generation result

| Policy | Mean frame change | p95 abrupt residual | High-frequency ratio |
|---|---:|---:|---:|
| Cosine | 0.01264 | 0.00882 | **0.54405** |
| Recent | 0.02464 | 0.01567 | 0.68273 |
| Random | 0.01624 | 0.01162 | 0.62220 |
| Agent plan (late) | **0.01213** | **0.00732** | 0.62793 |

Recent retrieval is clearly harmful. The late-chunk agent plan improves the abrupt-change
tail over cosine, but the suitcase drifts from burgundy toward dark blue/black late in the
return shot. Thus forcing the correct event is not sufficient; the location within the event
matters.

## Within-event selection

Experiment 07 increased the physical cache to 160 frames, which fit on one L20 at about
36.7GB peak diffusion-card allocation. It compared early `[2,1]`, late `[7,6]`, and mixed
`[7,2]` plans at seed 0.

| Plan | p95 abrupt residual | Visual finding |
|---|---:|---|
| Early `[2,1]` | **0.00705** | Burgundy suitcase retained through the end |
| Late `[7,6]` | 0.00816 | Suitcase drifts to dark blue/black at the end |
| Mixed `[7,2]` | 0.00731 | Burgundy suitcase retained through the end |

This is evidence that an event should not be represented by an arbitrary single temporal
neighborhood. Complementary key states can be safer than the two most recent event chunks.

## Seed-1 stopping-gate replication

At seed 1 and full 160-frame residence, cosine, early, and late used identical layer and
two-chunk budgets.

| Policy | p95 abrupt residual | High-frequency ratio | Visual finding |
|---|---:|---:|---|
| Cosine | **0.01042** | 0.68769 | Identity and attributes retained |
| Agent early `[2,1]` | 0.01191 | 0.67613 | Identity and attributes retained |
| Agent late `[7,6]` | 0.01217 | **0.66662** | Final identity changes to a different dark-haired woman |

Cosine used chunk 2 most frequently and included chunk 7 in some layers, rather than forcing
one pair across every layer. Agent-early forced `[2,1]` in all 80 sampled layer events and
agent-late forced `[7,6]` in all 80. The late plan's identity failure replicates the conclusion
that hard global agent replacement is unsafe, while cosine already approximates a useful
early/late mixture.

## Decision

Stop expanding the hard agent-plan baseline. The supported direction is a hierarchical
hybrid router:

1. an agent or semantic event model constrains or vetoes candidate events;
2. the learned low-level router selects diverse key states within the allowed event;
3. a confidence/validity gate rejects uncertain or obsolete memory;
4. full K/V is injected only in the profiled ten layers initially.

The next implementation should test event masking plus diversity-aware two-slot selection,
not an LLM directly selecting exact chunk IDs for every layer. The current prompt also failed
to produce a truly empty middle shot in every policy, so future semantic evaluation must use
stronger transition cases and more seeds before paper-level claims.
