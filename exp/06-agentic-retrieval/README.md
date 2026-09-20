# 06: Agent-planned retrieval causality test

This experiment asks a deliberately narrow question: if a semantic planner knows which
resident historical chunks contain the entity that is about to reappear, does selecting
those chunks improve the generated return shot at the same memory budget?

All variants use the released LayerRecall checkpoint, the fixed ten profiled layers, two
selected history chunks per active layer, seed 0, and latent size 24 x 40. The only changed
variable is the history ranking policy: cosine, recent, deterministic random, or an
Oracle-like agent plan. The plan is not yet an LLM/VLM agent; it is a controlled upper-bound
test for whether semantic event planning has causal value.

The prompt schedule is 8 / 4 / 8 chunks. During the return shot (chunks 12--19), the plan
prefers chunks 7 and 6 from the end of the first shot. A 112-frame physical cache keeps
chunk 7 available throughout the return and chunk 6 until the final return chunk, while the
ordinary local window remains 32 frames. The analysis reports this availability boundary.

Promotion gate: continue to a real agent or learned planner only if the planned policy
improves return-shot identity/attribute agreement over cosine and deterministic random
without a clear temporal-stability regression.
