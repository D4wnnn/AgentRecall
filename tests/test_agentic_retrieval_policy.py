import torch

from utils.layer_recall import (
    HistoryChunkRecord,
    LayerRecallConfig,
    apply_retrieval_policy,
)


def records():
    return [
        HistoryChunkRecord(i, i * 8, 8, i * 10, (i + 1) * 10, torch.zeros(4))
        for i in (13, 14, 15, 23, 31)
    ]


def test_agent_plan_prioritizes_resident_preferred_chunks():
    scores = torch.tensor([0.4, 0.5, 0.6, 0.1, 0.9])
    routed, preferred = apply_retrieval_policy(
        scores,
        records(),
        policy="agent_plan",
        current_chunk_index=32,
        layer_index=4,
        agent_plan=((32, 39, (23, 15)),),
    )
    assert preferred == [23, 15]
    assert torch.topk(routed, 2).indices.tolist() == [3, 2]


def test_agent_plan_falls_back_to_cosine_outside_planned_phase():
    scores = torch.tensor([0.4, 0.5, 0.6, 0.1, 0.9])
    routed, preferred = apply_retrieval_policy(
        scores,
        records(),
        policy="agent_plan",
        current_chunk_index=20,
        layer_index=4,
        agent_plan=((32, 39, (23, 15)),),
    )
    assert preferred == []
    torch.testing.assert_close(routed, scores)


def test_agent_event_constrains_event_but_preserves_cosine_order():
    scores = torch.tensor([0.4, 0.5, 0.6, 0.1, 0.9])
    routed, preferred = apply_retrieval_policy(
        scores,
        records(),
        policy="agent_event",
        current_chunk_index=32,
        layer_index=4,
        agent_plan=((32, 39, (13, 14, 15)),),
    )
    assert preferred == [13, 14, 15]
    assert torch.topk(routed, 3).indices.tolist() == [2, 1, 0]


def test_agent_event_diverse_keeps_top1_and_prefers_complementary_second_slot():
    recs = [
        HistoryChunkRecord(1, 8, 8, 10, 20, torch.tensor([1.0, 0.0])),
        HistoryChunkRecord(2, 16, 8, 20, 30, torch.tensor([0.99, 0.01])),
        HistoryChunkRecord(3, 24, 8, 30, 40, torch.tensor([0.0, 1.0])),
    ]
    routed, _ = apply_retrieval_policy(
        torch.tensor([0.9, 0.89, 0.7]),
        recs,
        policy="agent_event_diverse",
        current_chunk_index=12,
        layer_index=4,
        agent_plan=((12, 19, (1, 2, 3)),),
        diversity_lambda=0.5,
    )
    assert torch.topk(routed, 2).indices.tolist() == [0, 2]


def test_agent_event_explicit_empty_plan_abstains_from_memory():
    scores = torch.tensor([0.4, 0.5, 0.6, 0.1, 0.9])
    routed, preferred = apply_retrieval_policy(
        scores,
        records(),
        policy="agent_event",
        current_chunk_index=12,
        layer_index=4,
        agent_plan=((8, 15, ()),),
    )
    assert routed.numel() == 0
    assert preferred == []


def test_config_loads_agent_plan_from_json(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        '{"layer_recall_agent_plan": [{"current_chunks": [12, 19], '
        '"preferred_chunks": [1, 2, 3]}]}',
        encoding="utf-8",
    )
    config = LayerRecallConfig.from_repo_config({"layer_recall": {
        "layer_recall_retrieval_policy": "agent_event_diverse",
        "layer_recall_agent_plan_path": str(plan),
        "layer_recall_diversity_lambda": 0.5,
    }})
    assert config.layer_recall_agent_plan == ((12, 19, (1, 2, 3)),)
    assert config.layer_recall_diversity_lambda == 0.5


def test_recent_oldest_and_random_are_deterministic():
    scores = torch.zeros(5)
    recent, _ = apply_retrieval_policy(
        scores, records(), policy="recent", current_chunk_index=32, layer_index=4
    )
    oldest, _ = apply_retrieval_policy(
        scores, records(), policy="oldest", current_chunk_index=32, layer_index=4
    )
    random_a, _ = apply_retrieval_policy(
        scores, records(), policy="random", current_chunk_index=32, layer_index=4
    )
    random_b, _ = apply_retrieval_policy(
        scores, records(), policy="random", current_chunk_index=32, layer_index=4
    )
    assert torch.argmax(recent).item() == 4
    assert torch.argmax(oldest).item() == 0
    torch.testing.assert_close(random_a, random_b)


def test_config_parses_agent_plan():
    config = LayerRecallConfig.from_repo_config({
        "layer_recall": {
            "layer_recall_enabled": True,
            "layer_recall_retrieval_policy": "agent_plan",
            "layer_recall_agent_plan": [
                {"current_chunks": [32, 39], "preferred_chunks": [23, 15]}
            ],
        }
    })
    assert config.layer_recall_retrieval_policy == "agent_plan"
    assert config.layer_recall_agent_plan == ((32, 39, (23, 15)),)
