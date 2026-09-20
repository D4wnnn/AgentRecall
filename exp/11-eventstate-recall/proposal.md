# EventState-Recall: Agentic, Versioned Neural Memory for Long Video Generation

## Problem formulation

Long-horizon video generation requires more than retrieving visually similar history. The
generator must determine (i) which past event explains a reappearing entity, (ii) which state
version is still valid after changes, and (iii) which layer-specific neural evidence should
enter attention. Flat similarity retrieval conflates semantic validity with feature affinity:
an obsolete red coat can remain visually retrievable after the character changes into blue,
and two adjacent chunks can redundantly occupy both visible memory slots.

We formulate the task as **versioned event-state retrieval**. At generation step `t`, the
system retrieves an event `e`, a valid entity-state version `v`, and layer-specific neural
records `i_l`, then decides whether the evidence should be injected:

$$
(e^*, v^*) = \pi_{agent}(G_t, p_t, u_t), \qquad
i_l^* = \operatorname{TopK}_{i\in\mathcal C(e^*,v^*)}
[s^l_{t,i}-\beta d(i,S_l)],
$$

where `G_t` is the online event-state graph, `p_t` is the current prompt, `u_t` contains
retrieval uncertainty and boundary signals, and `d` penalizes redundant selections.

## Contribution 1: Event-State Dual Memory

The memory has two synchronized address spaces:

- a semantic graph containing events, entities, relations, state versions, valid intervals,
  confidence, and provenance;
- the original LayerRecall banks containing per-layer pre-RoPE summaries and pointers to
  complete RoPE-applied K/V payloads.

Every semantic observation stores its `chunk_id`, and each event/state node owns a set of
chunk IDs. Thus an agent can reason over human-interpretable state without compressing away
the full K/V evidence eventually consumed by attention.

The slow observer is a local Qwen3-VL model. A cheap controller emits per-chunk signals from
prompt changes, router entropy/margin, shot change, and visual embeddings. Qwen is invoked
only when these signals indicate CREATE_EVENT, EXTEND_EVENT, UPDATE_STATE,
INVALIDATE_STATE, or LINK_REAPPEARANCE. One VLM call can inspect a labeled montage covering
multiple chunks and return both chunk observations and an event summary.

## Contribution 2: Hierarchical Semantic-to-KV Routing

The agent does not hard-code the same chunk pair for all DiT layers. It produces:

- an event/state candidate mask or soft prior;
- target entity and attributes;
- a typed action: `RECALL`, `KEEP_CURRENT`, `UPDATE`, or `IGNORE`;
- a calibrated confidence.

Inside the allowed event, the frozen LayerRecall router retains its per-layer current-state
cosine scores. The second memory slot includes a diversity penalty so that the two selected
chunks cover complementary canonical and recent-valid evidence. This combines semantic
coarse routing with layer-specific neural retrieval.

## Contribution 3: Budgeted Validity-Aware Lifecycle Control

The semantic graph maintains explicit state versions and valid intervals. A state update
invalidates only the superseded attribute version, rather than erasing the entity identity.
The agent emits typed operations rather than a generic visual-difference scalar. These
operations control:

- `KEEP`: retain a canonical or most-recent valid event anchor;
- `MERGE`: merge redundant adjacent chunks into one semantic event while retaining chosen
  K/V anchors;
- `EVICT`: remove low-utility or obsolete physical records;
- `INVALIDATE`: prevent an obsolete state version from being injected;
- `ABSTAIN`: use only local attention when semantic or neural confidence is low.

All main comparisons use the same 80-frame physical-cache byte budget, two visible history
chunks, and the profiled ten insertion layers. The previous 160/192-frame demonstrations are
retrieval-isolation studies, not evidence for bounded-memory superiority.

## Claimed scope

The defensible novelty is the closed loop: online versioned event/state reasoning is aligned
directly to layer-indexed full K/V; an agent constrains semantic validity while a neural
router resolves layer-specific evidence; uncertainty-triggered typed operations manage the
same bounded physical memory. We should not claim the first agentic video generator, first
semantic K/V memory, first forgetting mechanism, or first hierarchical memory.
