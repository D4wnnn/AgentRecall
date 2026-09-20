#!/usr/bin/env python3
"""Audit local VLM traces for routing correctness and output reliability."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def strict_json(raw: str) -> bool:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return False
        text = text[start:end + 1]
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def audit(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data["events"]
    decisions = {item["event_id"]: item["decision"] for item in data["pre_generation_decisions"]}
    final = decisions[events[-1]["event_id"]]
    expected_chunks = set(range(events[-1]["chunk_ids"][0]))
    observed_chunks = {
        int(item["chunk_id"])
        for event in events
        for item in event.get("chunk_observations", [])
    }
    attempts = [attempt for item in data.get("raw_outputs", []) for attempt in item.get("attempts", [])]
    repairs = [repair for item in data.get("raw_outputs", []) for repair in item.get("structural_repairs", [])]
    target_ids = final.get("target_event_ids", [])
    hit1 = bool(target_ids) and target_ids[0] == "shot_1"
    return {
        "trace": str(path),
        "final_action": final.get("action"),
        "final_target_event_ids": target_ids,
        "return_event_hit_at_1": float(hit1),
        "return_action_correct": float(final.get("action") == "RECALL"),
        "absence_does_not_invalidate_origin": float(
            "shot_1" not in decisions.get("shot_2", {}).get("invalid_event_ids", [])
        ),
        "historical_chunk_observation_coverage": (
            len(observed_chunks.intersection(expected_chunks)) / max(1, len(expected_chunks))
        ),
        "raw_attempts": len(attempts),
        "strict_json_rate": sum(strict_json(item) for item in attempts) / max(1, len(attempts)),
        "structural_repair_count": len(repairs),
        "observer_elapsed_seconds": data.get("elapsed_seconds"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [audit(path) for path in args.trace]
    metric_names = [
        "return_event_hit_at_1", "return_action_correct",
        "absence_does_not_invalidate_origin", "historical_chunk_observation_coverage",
        "strict_json_rate", "structural_repair_count", "observer_elapsed_seconds",
    ]
    aggregate = {
        name: sum(float(row[name]) for row in rows if row[name] is not None)
        / max(1, sum(row[name] is not None for row in rows))
        for name in metric_names
    }
    payload = {
        "scope": "three-shot reappearance diagnostics with shot_1 as the oracle return event",
        "num_cases": len(rows), "per_case": rows, "mean": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
