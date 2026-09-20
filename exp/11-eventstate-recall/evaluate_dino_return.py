#!/usr/bin/env python3
"""DINOv2 screening metric for establish/absence/return demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def decode(path: Path) -> list[Image.Image]:
    container = av.open(str(path))
    frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    container.close()
    return frames


def sample_shot(frames: list[Image.Image], start: int, end: int, total: int, count: int):
    lo = start * len(frames) / total
    hi = end * len(frames) / total
    indices = torch.linspace(lo, max(lo, hi - 1), count).round().long().tolist()
    return [frames[min(max(index, 0), len(frames) - 1)] for index in indices]


def embed(model, processor, images, device):
    batch = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        features = model(**batch).last_hidden_state[:, 0].float()
    return F.normalize(features, dim=-1).cpu()


def score_video(path, durations, model, processor, device, samples):
    frames = decode(path)
    offsets = [0]
    for duration in durations:
        offsets.append(offsets[-1] + duration)
    shots = [sample_shot(frames, offsets[i], offsets[i + 1], offsets[-1], samples)
             for i in range(3)]
    embeddings = [embed(model, processor, images, device) for images in shots]
    means = [F.normalize(item.mean(0), dim=0) for item in embeddings]
    return_pairwise = embeddings[2] @ embeddings[0].T
    absent_pairwise = embeddings[1] @ embeddings[0].T
    adjacent = sum(
        float((shot[1:] * shot[:-1]).sum(-1).mean()) for shot in embeddings
    ) / len(embeddings)
    return {
        "video": str(path),
        "dino_global_return_cosine": float(means[2] @ means[0]),
        "dino_global_absent_cosine": float(means[1] @ means[0]),
        "dino_global_return_gap": float(means[2] @ means[0] - means[1] @ means[0]),
        "dino_global_return_nn": float(return_pairwise.max(dim=1).values.mean()),
        "dino_global_absent_nn": float(absent_pairwise.max(dim=1).values.mean()),
        "dino_global_adjacent_smoothness": adjacent,
        "decoded_frames": len(frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/private/lc/download/models/dinov2-large"))
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-shot", type=int, default=6)
    args = parser.parse_args()
    if len(args.video) != len(args.label):
        raise ValueError("--video and --label counts must match")
    durations = [int(x) for x in (args.case_dir / "shot_durations.txt").read_text().split()]
    if len(durations) != 3:
        raise ValueError("screening metric requires exactly three shots")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(str(args.model))
    model = AutoModel.from_pretrained(str(args.model), torch_dtype=torch.float16).to(device).eval()
    results = []
    for label, video in zip(args.label, args.video):
        item = score_video(video, durations, model, processor, device, args.samples_per_shot)
        item["label"] = label
        results.append(item)
    payload = {
        "metric_scope": "global-frame DINOv2 screening; not object-crop or official MemoBench",
        "model": str(args.model), "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
