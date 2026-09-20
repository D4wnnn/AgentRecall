#!/usr/bin/env python3
"""Extract matched inspection frames from experiment videos."""

import argparse
from pathlib import Path

import av


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        paths = sorted((args.root / variant).glob("*.mp4"))
        for path in paths:
            sample = path.stem.split("-")[1]
            container = av.open(str(path))
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            total = stream.frames
            targets = {0, int(total * 0.38), int(total * 0.62), int(total * 0.82), total - 1}
            for index, frame in enumerate(container.decode(stream)):
                if index in targets:
                    name = f"{variant}_sample{sample}_{index:04d}.png"
                    frame.to_image().save(args.output_dir / name)
            print(variant, sample, "fps", fps, "frames", total, "duration", total / fps)


if __name__ == "__main__":
    main()
