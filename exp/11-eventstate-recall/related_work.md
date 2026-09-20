# Closest Related Work and Positioning

Links were checked on 2026-09-05. Recheck immediately before submission.

| Work | Main mechanism | Difference from EventState-Recall |
|---|---|---|
| [LayerRecall](https://arxiv.org/abs/2608.28460) | Current-conditioned per-layer chunk K/V retrieval and profiled insertion layers | No semantic event/entity graph, state versions, or obsolete-state operation |
| [LongLive-RAG](https://arxiv.org/abs/2606.02553) | Query-based retrieval of historical latents | Flat history; no event-to-layer-K/V alignment or version validity |
| [MemFlow](https://arxiv.org/abs/2512.14699) | Text-conditioned historical frame retrieval and token activation | No agent state reasoning or explicit superseded state |
| [Context-as-Memory](https://arxiv.org/abs/2506.03141) | Camera-overlap frame retrieval | Geometry-oriented rather than entity/event state validity |
| [SlotMemory](https://arxiv.org/abs/2605.31033) | Object-centric semantic addresses linked to high-fidelity K/V | Closest dual-address memory; lacks versioned events and layer-specific agent-constrained recall |
| [SlotMem](https://arxiv.org/abs/2607.15772) | Character-semantic probe, role slots, conservative writer | Character-focused compressed slots rather than general event/state full-K/V routing |
| [Echo-Forcing](https://arxiv.org/abs/2605.16003) | Hierarchical temporal memory and difference-aware decay | Visual difference decay rather than typed semantic state-version operations |
| [WorldTrace](https://arxiv.org/abs/2608.07408) | Addressable virtual RoPE positions and scene landmarks | Solves positional addressability rather than semantic validity |
| [OmniMem](https://arxiv.org/abs/2605.30519) | Efficient sparse full-range K/V access | Focuses access scale, not event/state reasoning |
| [Ring Forcing](https://arxiv.org/abs/2608.26794) | Training-time far-history retrieval and sparse RoPE | No explicit online agent or state-version graph |

Event-centric agent memories are strongest in video understanding rather than generation.
[VideoAgent](https://arxiv.org/abs/2403.10517),
[Homer](https://arxiv.org/abs/2607.02588),
[PyraVid](https://arxiv.org/abs/2605.17065), and
[EM2Mem](https://arxiv.org/abs/2609.00551) organize long videos around events, entities,
and multimodal evidence for question answering. Agentic video-generation systems such as
[VideoDirectorGPT](https://arxiv.org/abs/2309.15091),
[StoryAgent](https://arxiv.org/abs/2411.04925), and
[GenMAC](https://arxiv.org/abs/2412.04440) mainly plan or inspect shots outside the generator.
Our target is different: the local multimodal agent controls the generator's internal,
layer-indexed K/V memory during autoregressive generation.

## Safe positioning sentence

Existing work has separately studied event-centric multimodal memory for understanding,
object/character-addressable neural memory for generation, and difference-aware forgetting.
EventState-Recall unifies these ideas by aligning an online state-version graph to full
layer-indexed K/V records and composing semantic agent constraints, layer-specific neural
retrieval, and typed bounded-memory control.
