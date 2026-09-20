#!/usr/bin/env python3
"""Compute low-cost temporal diagnostics for matched videos."""

import argparse
import csv
from pathlib import Path

import av
import numpy as np


def metric(path):
    container = av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    energies = []
    previous = None
    for frame in container.decode(stream):
        image = frame.to_ndarray(format="gray").astype(np.float32) / 255.0
        image = image[::4, ::4]
        if previous is not None:
            energies.append(float(np.mean(np.abs(image - previous))))
        previous = image
    energy = np.asarray(energies, dtype=np.float64)
    window = max(3, round(2 * fps))
    smooth = np.convolve(energy, np.ones(window) / window, mode="same")
    abrupt = np.maximum(energy - smooth, 0)
    centered = energy - energy.mean()
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    freq = np.fft.rfftfreq(len(centered), d=1 / fps)
    denom = power[(freq >= 0.2) & (freq <= 12)].sum()
    high = power[(freq >= 2) & (freq <= 12)].sum()
    return {
        "video": str(path),
        "frames": len(energy) + 1,
        "fps": fps,
        "mean_change": energy.mean(),
        "p95_change": np.quantile(energy, 0.95),
        "mean_abrupt": abrupt.mean(),
        "p95_abrupt": np.quantile(abrupt, 0.95),
        "high_frequency_ratio": high / denom if denom else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [metric(path) for path in args.videos]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

