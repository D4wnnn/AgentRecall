#!/usr/bin/env python3
"""Create synchronized Baseline / LayerRecall / EventState-Recall demo video."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont


LABELS = ("Baseline (local only)", "LayerRecall", "EventState-Recall (Qwen agent)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shot1-end", type=int, default=255)
    parser.add_argument("--shot2-end", type=int, default=510)
    args = parser.parse_args()
    if len(args.video) != 3:
        raise ValueError("exactly three --video arguments are required")
    containers = [av.open(str(path)) for path in args.video]
    streams = [container.streams.video[0] for container in containers]
    decoders = [container.decode(stream) for container, stream in zip(containers, streams)]
    fps = float(streams[0].average_rate)
    first = [next(decoder).to_image().convert("RGB") for decoder in decoders]
    width, height = first[0].size
    header = 44
    args.output.parent.mkdir(parents=True, exist_ok=True)
    destination = av.open(str(args.output), "w")
    output_stream = destination.add_stream("libx264", rate=round(fps))
    output_stream.width, output_stream.height = width * 3, height + header
    output_stream.pix_fmt = "yuv420p"
    output_stream.options = {"crf": "20", "preset": "medium"}
    font = ImageFont.load_default()

    def emit(images, index):
        canvas = Image.new("RGB", (width * 3, height + header), "black")
        draw = ImageDraw.Draw(canvas)
        for column, (image, label) in enumerate(zip(images, LABELS)):
            canvas.paste(image, (column * width, header))
            draw.text((column * width + 10, 8), label, fill="white", font=font)
        if index < args.shot1_end:
            phase = "SHOT 1: establish"
        elif index < args.shot2_end:
            phase = "SHOT 2: target absent / memory abstains"
        else:
            phase = "SHOT 3: same target returns / memory recalls"
        draw.text((width * 3 // 2 - 120, 26), phase, fill=(255, 220, 80), font=font)
        for packet in output_stream.encode(av.VideoFrame.from_image(canvas)):
            destination.mux(packet)

    emit(first, 0)
    index = 1
    while True:
        try:
            images = [next(decoder).to_image().convert("RGB") for decoder in decoders]
        except StopIteration:
            break
        emit(images, index)
        index += 1
    for packet in output_stream.encode():
        destination.mux(packet)
    destination.close()
    for container in containers:
        container.close()
    print(args.output)


if __name__ == "__main__":
    main()
