#!/usr/bin/env python3
"""Generate a synthetic road frame for offline testing."""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "assets" / "test_road.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)

    # Road surface
    cv2.fillPoly(
        frame,
        [np.array([[80, h], [260, 200], [380, 200], [560, h]], np.int32)],
        (70, 70, 70),
    )

    # Left lane (white)
    cv2.line(frame, (200, h), (300, 200), (255, 255, 255), 12)
    # Right lane (yellow)
    cv2.line(frame, (440, h), (340, 200), (0, 220, 255), 12)

    cv2.imwrite(str(out), frame)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
