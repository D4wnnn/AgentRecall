# AgentRecall

AgentRecall is an inference-time semantic memory controller built on top of
LayerRecall for long-horizon video generation. The main path in this repository
does **not** train a new DiT, router, or VLM. A frozen local Qwen3-VL model
maintains a structured event/state view of previous chunks, and LayerRecall
still stores and injects the original full-fidelity K/V cache.

## Main idea: semantic event-state retrieval

LayerRecall provides the low-level visual memory mechanism: historical chunks
are represented by K/V records and a retrieval policy selects chunks to expose
to selected attention layers. The problem is that a purely hidden-state-based
retriever can confuse semantically similar events, prefer a recent but wrong
chunk, or restore an obsolete state after an explicit update.

AgentRecall adds a high-level semantic controller:

```text
video chunks / captions
        |
        v
Qwen3-VL observer
        |
        v
versioned event + entity-state memory
        |
        v
RECALL / UPDATE / IGNORE decision
        |
        v
preferred historical chunk IDs
        |
        v
LayerRecall candidate restriction / reranking
        |
        v
original full-fidelity K/V injection
```

The agent is responsible for **which event/state is valid**. LayerRecall is
responsible for **recovering visual details from the corresponding K/V cache**.
The agent does not choose DiT layers and does not replace the K/V tensors with
text summaries.

### Event and state versioning

Each event points to one or more historical chunk IDs. Each entity can have
multiple state versions with validity intervals:

```text
woman_A: v1 = red coat, valid through chunk 4
woman_A: v2 = black coat, valid from chunk 5
```

When a subject reappears, the agent emits `RECALL` and points to the valid
event/version. When the prompt explicitly changes its state, the agent emits
`UPDATE`; the old state is not blindly restored. `IGNORE` is an explicit
abstention that prevents an unrelated historical event from being injected.

This is an inference-time controller. The current validated prototype uses a
two-stage workflow: observe/plan first, then run generation with the resulting
plan. It is not an asynchronous agent call after every denoising step.

## Core code paths

### Agent and event memory

- `exp/11-eventstate-recall/observe_video.py`
  - Loads local Qwen3-VL.
  - Creates chunk thumbnails/montages and requests structured JSON.
  - Validates and repairs JSON against the event observation schema.
  - Produces typed decisions and compiles a LayerRecall plan.
- `exp/11-eventstate-recall/event_memory.py`
  - Implements `EventNode`, `StateVersion`, and `EventStateMemory`.
  - Maps semantic events and state versions to historical chunk IDs.
- `exp/11-eventstate-recall/replan_trace.py`
  - Re-runs planning from an audited trace without repeating visual
    observation.
- `exp/11-eventstate-recall/schemas/event_observation.schema.json`
  - JSON contract used to validate model output.

### LayerRecall integration

- `utils/layer_recall.py`
  - Parses `layer_recall_agent_plan_path` and inline plans.
  - Implements the `agent_plan`, `agent_event`, and
    `agent_event_diverse` retrieval policies.
  - Restricts or reranks the candidate historical chunks.
- `wan_5b/modules/causal_model.py`
  - Passes the current chunk and agent-preferred chunk IDs through the
    retrieval path before attention.
  - Logs the selected chunks and agent preference for auditing.
- `inference.py`
  - Runs the ordinary LayerRecall generation pipeline with the external plan.
- `exp/11-eventstate-recall/config_qwen_demo.yaml`
  - Enables `agent_event_diverse` and loads the plan through
    `LR_AGENT_PLAN_PATH`.
- `exp/11-eventstate-recall/run_qwen_demo.sh`
  - Example end-to-end launcher for a Qwen-planned case.

The plan format is intentionally small:

```json
{
  "layer_recall_agent_plan": [
    {
      "current_chunks": [2, 2],
      "preferred_chunks": [0],
      "action": "RECALL"
    }
  ]
}
```

## Reproducing the agent path

The exact paths in the original development machine are machine-specific. Set
the model and data roots in a local environment file, then adapt the launcher
to the target server. The required model assets are:

- Wan2.2-TI2V-5B;
- the released LayerRecall/LongLive checkpoints;
- Qwen3-VL-4B-Instruct for the observer.

The important environment variable for the generation stage is:

```text
LR_AGENT_PLAN_PATH=/path/to/layer_recall_plan.json
```

First produce the plan with `observe_video.py` (or `replan_trace.py`), inspect
the JSON trace, and then run `run_qwen_demo.sh`/`inference.py`. Keep the agent
trace and retrieval log together: they are needed to distinguish a semantic
planning error from a visual K/V injection error.

## Experiment index

The `exp/` directory records the development history. Generated videos,
checkpoints, machine logs, and local environment files are intentionally
ignored by Git.

| Directory | Purpose | Status / interpretation |
|---|---|---|
| `00-smoke` | Minimal model and environment smoke test. | Basic pipeline sanity check. |
| `01-routing-baselines` | Local-only, fixed-10-layer, and all-layer routing baselines. | Establishes the original LayerRecall comparison. |
| `02-temporal-layer-demand` | Diagnostic analysis of temporal/layer memory demand. | Motivates selective rather than indiscriminate memory use. |
| `03-recall-update-conflict` | Reappearance versus explicit clothing/object update cases. | Tests whether old memory should be recalled or superseded. |
| `04-multi-prototype-diagnostic` | Multiple memory prototypes/summary diagnostic. | Explores semantic ambiguity in one-vector retrieval. |
| `05-cache-horizon-stress` | Longer temporal gaps and cache-horizon stress tests. | Measures degradation as the relevant chunk gets farther away. |
| `06-agentic-retrieval` | First agentic retrieval prototype and retrieval analysis. | Early proof-of-concept; inspect `RESULTS.md`. |
| `07-agentic-within-event` | Agent selection with within-event alternatives. | Tests early/late/mixed agent policies. |
| `08-agentic-seed1-replication` | Seed replication of the agentic experiments. | Checks whether the observed effect is seed-specific. |
| `09-demo-reappearance` | Curated reappearance demos. | Presentation-oriented qualitative cases. |
| `10-demo-long-transition` | Longer transitions before a subject returns. | Stress-tests temporal separation and distractors. |
| `11-eventstate-recall` | Main no-training AgentRecall prototype. | Qwen event/state planning plus LayerRecall K/V retrieval. |
| `12-dynamic-state-chpm` | Smoke tests for a trainable dynamic layer router. | Verifies topology, gradients, and checkpoint compatibility. |
| `13-dynamic-router-train` | 500-step dynamic-router training with prior scale 4.0. | Numerically healthy, but Top-10 stayed at the original fixed layers. |
| `14-dynamic-router-prior05` | Same training with prior scale 0.5. | Still selected the same fixed 10 layers; not evidence for dynamic layer selection. |

## What has and has not been demonstrated

The strongest current result is the no-training semantic retrieval path in
`exp/11-eventstate-recall`. In the small audited cases, the Qwen planner
correctly selected the intended action/event and approached an oracle event
plan on the DINO-based return-consistency screening metric. These are
retrieval-isolation results, not a claim of a complete large-scale benchmark
win.

The later dynamic-layer CHPM experiments demonstrate that the training path is
numerically valid, but both long runs collapsed to the original LayerRecall
Top-10 layer set. They should be treated as engineering/ablation records, not
as evidence that learned dynamic layer selection has succeeded. A future
trainable version needs explicit exploration (for example soft gates,
temperature annealing, or Gumbel exploration) before another long run.

## Repository hygiene

Model weights, checkpoints, logs, generated videos, and machine-specific
`exp/env.sh` files are excluded by `.gitignore`. Keep large outputs outside the
source repository and commit only reproducible code, configurations, schemas,
small traces/metrics, and documentation.
