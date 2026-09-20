#!/usr/bin/env python3
"""Simple auditable red/blue occupancy screen for the clothing-update case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np


def score(path: Path, samples: int = 12) -> dict:
    container = av.open(str(path))
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    boundaries = np.linspace(0, len(frames), 4).round().astype(int)
    result = {"video": str(path)}
    for shot in (0, 2):
        indices = np.linspace(boundaries[shot], boundaries[shot + 1] - 1, samples).round().astype(int)
        red_count = blue_count = total = 0
        for index in indices:
            image = frames[int(index)]
            h, w = image.shape[:2]
            crop = image[int(.1*h):int(.95*h), int(.2*w):int(.8*w)].astype(np.float32)
            red, green, blue = crop[..., 0], crop[..., 1], crop[..., 2]
            red_mask = (red > 70) & (red > 1.25 * blue) & (red > 1.10 * green)
            blue_mask = (blue > 60) & (blue > 1.20 * red) & (blue > 1.05 * green)
            red_count += int(red_mask.sum())
            blue_count += int(blue_mask.sum())
            total += crop.shape[0] * crop.shape[1]
        result[f"shot_{shot + 1}_red_fraction"] = red_count / total
        result[f"shot_{shot + 1}_blue_fraction"] = blue_count / total
        result[f"shot_{shot + 1}_blue_share"] = blue_count / max(1, red_count + blue_count)
    result["state_update_margin"] = (
        result["shot_3_blue_share"] - result["shot_1_blue_share"]
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.video) != len(args.label):
        raise ValueError("--video and --label counts must match")
    rows = []
    for label, video in zip(args.label, args.video):
        row = score(video)
        row["label"] = label
        rows.append(row)
    payload = {
        "scope": "central-crop RGB occupancy screen; not segmented clothing accuracy",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
