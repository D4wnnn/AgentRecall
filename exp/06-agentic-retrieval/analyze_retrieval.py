import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
rows = []
for variant in ("cosine", "recent", "random", "agent_plan"):
    path = ROOT / "logs" / f"{variant}.jsonl"
    if not path.exists():
        continue
    events = [json.loads(line) for line in path.open()]
    active = [
        event for event in events
        if event.get("YX_chunk_index", -1) >= 12
        and event.get("layer_recall_gate_active")
        and event.get("YX_denoising_step_index") == 3
    ]
    planned = {6, 7}
    hits = [bool(planned.intersection(event.get("YX_selected_chunk_ids", []))) for event in active]
    both = [planned.issubset(set(event.get("YX_selected_chunk_ids", []))) for event in active]
    resident = [planned.issubset(set(event.get("YX_candidate_chunk_ids", []))) for event in active]
    rows.append({
        "variant": variant,
        "return_events": len(active),
        "preferred_any_hit_rate": sum(hits) / len(hits) if hits else 0.0,
        "preferred_both_hit_rate": sum(both) / len(both) if both else 0.0,
        "preferred_both_candidate_pool_rate": sum(resident) / len(resident) if resident else 0.0,
    })

out = ROOT / "analysis" / "retrieval_metrics.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
