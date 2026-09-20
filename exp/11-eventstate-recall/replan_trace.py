#!/usr/bin/env python3
"""Re-run only pre-generation decisions from an audited visual event trace."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from observe_video import QwenEventObserver, compile_layer_recall_plan, read_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("/private/lc/download/Qwen3-VL-4B-Instruct"))
    args = parser.parse_args()

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    case_dir = Path(payload["case_dir"])
    if not case_dir.exists():
        case_dir = Path(__file__).resolve().parents[2] / case_dir
    durations, captions = read_case(case_dir)
    events = payload["events"]
    observer = QwenEventObserver(args.model)
    decisions = []
    started = time.perf_counter()
    for index, event in enumerate(events):
        if index == 0:
            decision = {
                "action": "KEEP_CURRENT", "target_event_ids": [], "invalid_event_ids": [],
                "target_entities": [], "confidence": 1.0,
                "rationale": "No historical event exists.",
            }
            attempts, repairs = [], []
        else:
            decision, attempts, repairs = observer.plan_return(
                events[:index], event["event_id"], captions[index]
            )
        decisions.append({"event_id": event["event_id"], "decision": decision})
        payload.setdefault("raw_outputs", []).append({
            "event_id": event["event_id"], "stage": "replan",
            "attempts": attempts, "structural_repairs": repairs,
        })

    final = decisions[-1]["decision"]
    plan = compile_layer_recall_plan(events, decisions)
    payload["pre_generation_decisions"] = decisions
    payload["decision"] = final
    payload["layer_recall_plan"] = plan
    payload["replan_elapsed_seconds"] = time.perf_counter() - started
    args.trace.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plan_path = args.trace.parent / "layer_recall_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"decision": final, "plan": plan}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
