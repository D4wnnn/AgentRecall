# Exp 12 — Dynamic Layer–State CHPM

## Goal

LayerRecall retrieves remote chunks with a current-conditioned query, but the
set of ten memory-reading DiT layers is fixed for every scene and denoising
state.  This experiment tests whether **where to recall** can be learned under
the same attention-token and active-layer budget as the released method.

This is the first implementation stage of the larger EventState method:

1. a semantic agent selects a valid event/state version (implemented in
   `exp/11-eventstate-recall`);
2. this shared router selects the DiT layers that should consume the event's
   K/V;
3. a later State-Contrastive CHPM objective will prefer valid memory over stale
   or absent memory.

The agent deliberately does **not** choose DiT layers.  Layer choice depends on
hidden diffusion state and is trained by the generation objective.

## Implemented dynamic router

For a current chunk summary `s_t` and learned layer embedding `e_l`, one shared
MLP predicts

```text
a_t,l = prior_l + MLP([LayerNorm(s_t), e_l]).
```

An exact Top-r gate is used in the forward pass.  A sigmoid straight-through
surrogate supplies gradients during CHPM training.  With `r=10`, exactly ten
layers replace local K/V slots with retrieved remote K/V.  An inactive layer
uses the corresponding recent local K/V slots; it does not receive zero K/V,
and the visible token count is unchanged.

The initial prior selects LayerRecall's released layers
`[4,9,10,12,13,15,16,17,18,26]`, so an untrained dynamic router starts from the
published behavior instead of a random architecture.

## Status (2026-09-05)

- dynamic configuration, shared router, layer embeddings, exact-budget
  straight-through gate, local/remote K/V interpolation, audit logging, and
  released-checkpoint compatibility are implemented;
- 64-frame inference smoke passed without NaN/OOM and selected exactly the ten
  prior layers at every denoising call;
- the generated latent is in `analysis/smoke_outputs` and router events are in
  `logs/dynamic_smoke_router.jsonl`;
- relevant regression suite: 49 tests plus 2 subtests passed;
- four-GPU standard-resolution CHPM gradient smoke is running in tmux
  `lr_dynamic_train_smoke`.

No quality improvement is claimed yet.  The initial smoke only verifies the
mechanism and backward-compatible starting point.

## Commands

```bash
tmux attach -t lr_dynamic_train_smoke
tail -f exp/12-dynamic-state-chpm/logs/dynamic_train_smoke_console.log
```

Inference smoke:

```bash
./exp/12-dynamic-state-chpm/run_dynamic_smoke.sh
```

One-step four-GPU gradient smoke:

```bash
./exp/12-dynamic-state-chpm/run_dynamic_train_smoke.sh
```

## Next gates

1. Require nonzero finite gradients on router output, layer embeddings, and
   prior logits.
2. Overfit 10–20 iterations and confirm layer choices can depart from the
   prior while maintaining exactly ten active layers.
3. Compare fixed-10, dynamic-10, and dynamic-5 at matched seeds and cache
   budgets.
4. Only after the router is validated, add valid/stale/no-memory
   State-Contrastive CHPM.  Keeping these stages separate makes failures and
   gains attributable.

