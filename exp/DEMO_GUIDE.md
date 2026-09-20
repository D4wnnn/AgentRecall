# LayerRecall / Agentic Retrieval Demo Guide

All comparisons use the same prompt, seed, Wan/LongLive-2.0 backbone, four denoising
steps, and reduced L20 screening resolution. The three columns are synchronized.

## Recommended presentation order

### 1. Rooftop artist — strongest clean return failure

File: `demo_01_rooftop_artist.mp4`

- Shot 1 establishes the platinum-blond artist, violet beret, orange jacket, and red
  sailboat painting.
- Shot 2 is a genuine empty skyline; the target leaves the local context.
- Baseline never recovers the artist. Ordinary LayerRecall recovers an easel but changes
  the artwork and still loses the person. Agent-event returns to the correct historical
  event and restores the same artist and red painting.

### 2. Flower-market woman — visible self-correction

File: `demo_02_flower_market.mp4`

- Shot 1 establishes white hair, red glasses, yellow raincoat, violet umbrella, and the
  turquoise lily cart.
- Shot 2 moves onto the flower stall and the woman leaves view.
- Baseline returns a different woman in pink. Ordinary LayerRecall fails to stably restore
  the person. Agent-event first produces a distractor but then visibly self-corrects to the
  original woman and attributes. This resembles the paper's memory-guided self-correction.

### 3. Library archivist — identity/layout recovery

File: `demo_03_library_archivist.mp4`

- Shot 2 moves behind the bookshelf and settles on the empty aisle.
- Ordinary LayerRecall returns to the desk but loses the subject for much of Shot 3.
- Agent-event restores the same black bob, square glasses, gray cardigan, books, and globe.

### 4. Seaside clockmaker — wrong identity versus self-correction

File: `demo_04_seaside_clockmaker.mp4`

- Shot 2 holds on the sea window with no clockmaker.
- Ordinary LayerRecall returns as a visibly different younger worker.
- Agent-event briefly mismatches, then returns to the original long white beard, navy cap,
  and burgundy vest. The clock itself remains imperfect, so present this as an identity
  recovery example rather than complete object recovery.

## Accurate claim

The agent used here is a controlled event-level planner: it marks Shot 1 as the relevant
event, then the trained LayerRecall cosine router independently chooses two chunks within
that event at each profiled layer. It is not yet an LLM/VLM agent. The result supports a
hierarchical agent-plus-neural-router design, not hard agent selection of exact K/V chunks.

These are selected qualitative demonstrations, not statistically sufficient benchmark
results. The prompt bank also contains failed/non-discriminative cases and should be retained
as evidence against cherry-picking claims beyond the demo purpose.
