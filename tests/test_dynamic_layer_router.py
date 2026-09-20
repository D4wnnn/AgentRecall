import pytest
import torch

from utils.layer_recall import (
    LayerRecallConfig,
    budgeted_straight_through_layer_gate,
)


def test_dynamic_layer_config_defaults_to_backward_compatible_disabled():
    config = LayerRecallConfig()
    assert config.layer_recall_dynamic_layer_enabled is False
    assert config.layer_recall_dynamic_layer_budget == 10


def test_dynamic_layer_config_rejects_invalid_budget():
    with pytest.raises(ValueError, match="dynamic_layer_budget"):
        LayerRecallConfig(
            layer_recall_num_layers=4,
            layer_recall_dynamic_layer_enabled=True,
            layer_recall_dynamic_layer_budget=5,
            memory_sensitive_layers=(0,),
        )


def test_budgeted_gate_has_exact_forward_budget_and_surrogate_gradients():
    logits = torch.tensor([-2.0, 0.5, 3.0, 1.0], requires_grad=True)
    gate, probabilities = budgeted_straight_through_layer_gate(
        logits, budget=2, temperature=0.7, training=True
    )
    torch.testing.assert_close(gate.detach(), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert int(gate.detach().sum().item()) == 2
    (gate * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 4
    assert torch.all((probabilities > 0) & (probabilities < 1))


def test_budgeted_gate_eval_is_binary_without_straight_through_graph():
    logits = torch.tensor([0.1, 0.3, 0.2], requires_grad=True)
    gate, _ = budgeted_straight_through_layer_gate(
        logits, budget=1, training=False
    )
    torch.testing.assert_close(gate, torch.tensor([0.0, 1.0, 0.0]))
    assert gate.requires_grad is False
