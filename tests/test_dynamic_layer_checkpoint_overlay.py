import ast
from pathlib import Path

import pytest
import torch


def load_preflight_functions():
    source_path = Path(__file__).parents[1] / "inference.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    keep = {
        "_is_dynamic_layer_router_key",
        "_preflight_layer_recall_state_dict",
    }
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in keep
    ]
    namespace = {"torch": torch}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["_preflight_layer_recall_state_dict"]


def test_released_checkpoint_can_initialize_only_new_dynamic_router_parameters():
    preflight = load_preflight_functions()
    model = {
        "model.layer_recall_base_query": torch.zeros(4),
        "model.layer_recall_dynamic_router.0.weight": torch.zeros(2, 6),
        "model.layer_recall_dynamic_layer_prior_logits": torch.zeros(3),
    }
    released = {"model.layer_recall_base_query": torch.zeros(4)}
    preflight(model, released, allow_missing_dynamic_layer_router=True)


def test_old_layer_recall_parameter_may_not_be_silently_missing():
    preflight = load_preflight_functions()
    model = {
        "model.layer_recall_base_query": torch.zeros(4),
        "model.layer_recall_current_alpha": torch.zeros(1),
        "model.layer_recall_dynamic_layer_prior_logits": torch.zeros(3),
    }
    released = {"model.layer_recall_base_query": torch.zeros(4)}
    with pytest.raises(ValueError, match="current_alpha"):
        preflight(model, released, allow_missing_dynamic_layer_router=True)

