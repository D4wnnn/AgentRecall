#!/usr/bin/env python3
"""Summarize LayerRecall JSONL logs by layer and generation chunk."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else 0.0


def number(value, default=0.0):
    return default if value is None else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    total = 0
    with args.log.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            layer = int(item.get("YX_layer_index", -1))
            chunk = int(item.get("YX_chunk_index", -1))
            if layer < 0 or chunk < 0:
                continue
            groups[(chunk, layer)].append(item)
            total += 1

    fields = [
        "chunk", "layer", "events", "gate_rate", "layout_rate",
        "candidate_count", "entropy", "normalized_entropy", "top1_weight",
        "top1_margin", "score_margin", "temporal_distance", "top1_chunk",
    ]
    rows = []
    for (chunk, layer), items in sorted(groups.items()):
        top_chunks = [int(number(x.get("YX_top1_chunk_id"), -1)) for x in items]
        valid_top_chunks = [x for x in top_chunks if x >= 0]
        hist = defaultdict(int)
        for value in valid_top_chunks:
            hist[value] += 1
        modal_chunk = max(hist, key=hist.get) if hist else -1
        candidate_count = mean([float(x.get("YX_candidate_count", 0) or 0) for x in items])
        entropy = mean([float(number(x.get("YX_selection_entropy"))) for x in items])
        rows.append({
            "chunk": chunk,
            "layer": layer,
            "events": len(items),
            "gate_rate": mean([float(bool(x.get("layer_recall_gate_active"))) for x in items]),
            "layout_rate": mean([float(bool(x.get("layer_recall_layout_applied"))) for x in items]),
            "candidate_count": candidate_count,
            "entropy": entropy,
            "normalized_entropy": entropy / math.log(candidate_count) if candidate_count > 1 else 0.0,
            "top1_weight": mean([float(number(x.get("YX_top1_weight"))) for x in items]),
            "top1_margin": mean([float(number(x.get("YX_top1_margin"))) for x in items]),
            "score_margin": mean([float(number(x.get("YX_score_margin"))) for x in items]),
            "temporal_distance": mean([float(number(x.get("YX_top1_temporal_distance"))) for x in items]),
            "top1_chunk": modal_chunk,
        })

    csv_path = args.output_dir / "chunk_layer_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_layer = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)
    layer_rows = []
    for layer, items in sorted(by_layer.items()):
        selected = [r["top1_chunk"] for r in items if r["top1_chunk"] >= 0]
        turnover = mean([float(a != b) for a, b in zip(selected, selected[1:])])
        layer_rows.append({
            "layer": layer,
            "chunks": len(items),
            "gate_rate": mean([r["gate_rate"] for r in items]),
            "entropy": mean([r["entropy"] for r in items]),
            "top1_margin": mean([r["top1_margin"] for r in items]),
            "temporal_distance": mean([r["temporal_distance"] for r in items]),
            "selection_turnover": turnover,
        })

    layer_path = args.output_dir / "layer_summary.csv"
    with layer_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]) if layer_rows else ["layer"])
        writer.writeheader()
        writer.writerows(layer_rows)

    def phase(chunk):
        if chunk < 24:
            return "shot1_establish"
        if chunk < 36:
            return "shot2_absent"
        return "shot3_return"

    phase_groups = defaultdict(list)
    for row in rows:
        if row["candidate_count"] > 0:
            phase_groups[(phase(row["chunk"]), row["layer"])].append(row)
    phase_rows = []
    for (name, layer), items in sorted(phase_groups.items()):
        phase_rows.append({
            "phase": name,
            "layer": layer,
            "chunks": len(items),
            "candidate_count": mean([r["candidate_count"] for r in items]),
            "normalized_entropy": mean([r["normalized_entropy"] for r in items]),
            "top1_margin": mean([r["top1_margin"] for r in items]),
            "temporal_distance": mean([r["temporal_distance"] for r in items]),
            "sink_top1_rate": mean([float(r["top1_chunk"] == 0) for r in items]),
        })
    phase_path = args.output_dir / "phase_layer_summary.csv"
    with phase_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(phase_rows[0]) if phase_rows else ["phase", "layer"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(phase_rows)

    summary = {
        "input": str(args.log),
        "events": total,
        "chunk_layer_groups": len(rows),
        "layers": sorted(by_layer),
        "chunks": sorted({row["chunk"] for row in rows}),
        "outputs": [str(csv_path), str(layer_path), str(phase_path)],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
