#!/usr/bin/env python3
import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont


VARIANTS = ("baseline", "layerrecall", "agent_event")
LABELS = ("Baseline (local only)", "LayerRecall", "Agent + LayerRecall")


def video_for(root: Path, variant: str, case_index: int) -> Path:
    files = sorted((root / "outputs" / variant).glob("*.mp4"))
    return files[case_index]


def make_demo(root: Path, case_index: int, shot_boundaries, output: Path):
    containers = [av.open(str(video_for(root, v, case_index))) for v in VARIANTS]
    streams = [container.streams.video[0] for container in containers]
    decoders = [container.decode(stream) for container, stream in zip(containers, streams)]
    fps = float(streams[0].average_rate)

    first = [next(decoder).to_image().convert("RGB") for decoder in decoders]
    width, height = first[0].size
    header = 42
    output.parent.mkdir(parents=True, exist_ok=True)
    dst = av.open(str(output), "w")
    out_stream = dst.add_stream("libx264", rate=round(fps))
    out_stream.width = width * 3
    out_stream.height = height + header
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "20", "preset": "medium"}

    font = ImageFont.load_default()

    def emit(images, frame_index):
        canvas = Image.new("RGB", (width * 3, height + header), "black")
        draw = ImageDraw.Draw(canvas)
        for col, (image, label) in enumerate(zip(images, LABELS)):
            canvas.paste(image, (col * width, header))
            draw.text((col * width + 12, 12), label, fill="white", font=font)
        if frame_index < shot_boundaries[0]:
            phase = "SHOT 1  establish identity and attributes"
        elif frame_index < shot_boundaries[1]:
            phase = "SHOT 2  subject outside the view"
        else:
            phase = "SHOT 3  same subject returns"
        draw.text((width * 3 // 2 - 110, 12), phase, fill=(255, 220, 80), font=font)
        frame = av.VideoFrame.from_image(canvas)
        for packet in out_stream.encode(frame):
            dst.mux(packet)

    emit(first, 0)
    index = 1
    while True:
        try:
            images = [next(decoder).to_image().convert("RGB") for decoder in decoders]
        except StopIteration:
            break
        emit(images, index)
        index += 1
    for packet in out_stream.encode():
        dst.mux(packet)
    dst.close()
    for container in containers:
        container.close()
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("case_index", type=int)
    parser.add_argument("shot1_end", type=int)
    parser.add_argument("shot2_end", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_demo(args.root, args.case_index, (args.shot1_end, args.shot2_end), args.output)


if __name__ == "__main__":
    main()
