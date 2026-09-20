# EventState-Recall Demo Guide

Each video is synchronized as Baseline / LayerRecall / EventState-Recall. Shot 1 establishes
the target; Shot 2 removes it from view and the agent abstains from long-range memory; Shot 3
returns and the agent recalls Shot 1 while LayerRecall chooses per-layer K/V chunks.

Recommended presentation order:

1. `rooftop_three_way.mp4`: clearest recovery of the same artist, violet beret, and red
   painting. The gated variant also reduces Shot 2 memory leakage.
2. `clockmaker_three_way.mp4`: tests elderly identity, clothing, clock, and workbench.
3. `library_three_way.mp4`: tests identity plus desk/globe/atlas layout.
4. `traveler_three_way.mp4`: retain as a failure/limitation example; the global DINO return
   score does not improve over LayerRecall.
5. `update_clothing_three_way.mp4`: mechanism demo for version validity. The agent recalls
   the updated blue-coat event rather than the original red-coat event. Present it as proof
   that the controller chooses the right state version, not as a quantitative win: the
   current coarse color metric favors baseline/LayerRecall.

Accurate spoken claim:

> A local Qwen3-VL agent reasons over event summaries and decides when to recall or abstain.
> It selects the historical event, not exact tensors; the trained LayerRecall router still
> chooses different full K/V chunks per DiT layer. On four selected reappearance cases, the
> agent chooses the correct event in all four and substantially improves the mean DINO return
> score, though one case regresses and fixed-budget benchmark evaluation remains pending.
