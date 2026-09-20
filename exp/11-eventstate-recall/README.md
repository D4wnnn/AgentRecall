# 11: EventState-Recall

This directory develops the paper-oriented extension of LayerRecall. The method is organized
as one online memory loop rather than three unrelated modules:

1. **Event-State Dual Memory** aligns a semantic, versioned event graph with the original
   layer-indexed full-fidelity K/V records.
2. **Hierarchical Semantic-to-KV Routing** lets an agent constrain valid events and states,
   while the current-conditioned neural router makes layer-specific, diversity-aware chunk
   selections inside those events.
3. **Budgeted Validity-Aware Lifecycle Control** performs typed state operations and
   KEEP/MERGE/EVICT under the same physical-cache byte budget as LayerRecall. Expensive VLM
   deliberation is triggered only at boundaries, conflicts, or low-confidence retrieval.

The current implementation milestone includes the local Qwen3-VL observer, validated JSON
trace, versioned event graph, event-to-chunk addressing, diversity-aware within-event routing,
external plan loading, and typed `IGNORE` abstention that disables remote K/V for a planned
chunk range. The older `agent_event` policy is an Oracle event-mask diagnostic; the new
`agent_event_diverse` runs Qwen-produced plans. Full semantic physical-cache lifecycle control
and learned distillation remain proposed milestones and must not be reported as completed.

## Local model

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Path: `/private/lc/download/Qwen3-VL-4B-Instruct`
- Environment: `/private/lc/envs/event_agent`
- Intended device: physical GPU 6

The observer groups several chunk thumbnails into one event-level VLM call and still returns
one structured observation per chunk. This realizes a fast/slow design rather than invoking
the VLM synchronously after every 8 latent frames.

## Reproducibility status

- `proposal.md`: complete method and contribution framing.
- `related_work.md`: verified closest work and differentiation.
- `experiment_plan.md`: fixed-budget evaluation and stopping criteria.
- `schemas/event_observation.schema.json`: structured observer contract.
- `observe_video.py`: Qwen3-VL event/chunk observer.
- `event_memory.py`: versioned event-state graph and K/V address mapping.
- `replan_trace.py`: causal re-planning from an audited trace without repeating observation.
- `evaluate_agent_traces.py`: event/action/JSON reliability audit.
- `evaluate_dino_return.py`: explicitly labeled global DINOv2 screening metric.
- `config_qwen_demo.yaml`, `run_qwen_demo.sh`: end-to-end Qwen-plan rerender.
- `RESULTS.md`, `DEMO_GUIDE.md`: measured findings, limitations, and presentation assets.
- `tests/`: deterministic unit tests for state versioning and event masks.
