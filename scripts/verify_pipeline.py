#!/usr/bin/env python3
"""Headless smoke test for the lane pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.pipeline import LanePipeline


def main() -> None:
    image_path = ROOT / "assets" / "test_road.png"
    if not image_path.exists():
        raise SystemExit(f"Missing test image: {image_path}")

    frame = cv2.imread(str(image_path))
    pipeline = LanePipeline()
    annotated, lines, edges = pipeline.process_frame(frame)

    out = ROOT / "assets" / "test_result.png"
    cv2.imwrite(str(out), annotated)

    assert edges.ndim == 2
    assert annotated.shape == frame.shape
    assert lines.is_valid(), "Expected both lane lines on synthetic road"

    print(f"OK: lanes detected, wrote {out}")


if __name__ == "__main__":
    main()
