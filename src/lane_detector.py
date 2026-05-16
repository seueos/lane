"""Lane line detection from edge maps via Hough transform."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .lane_geometry import validate_near_detection


@dataclass
class LaneLines:
    """Left and right lane lines as (slope, intercept) in image coordinates."""

    left: tuple[float, float] | None = None
    right: tuple[float, float] | None = None

    def is_valid(self) -> bool:
        return self.left is not None and self.right is not None


def _line_params(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float] | None:
    if x2 == x1:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _weighted_average_line(
    segments: list[tuple[float, float, float]],
    positive_slope: bool,
    min_slope: float = 0.35,
    max_slope: float = 4.0,
) -> tuple[float, float] | None:
    if not segments:
        return None

    filtered: list[tuple[float, float, float]] = []
    for slope, intercept, weight in segments:
        if (slope > 0) != positive_slope:
            continue
        if abs(slope) < min_slope or abs(slope) > max_slope:
            continue
        filtered.append((slope, intercept, weight))

    if not filtered:
        return None

    total_w = sum(w for _, _, w in filtered)
    if total_w < 1e-6:
        return None

    m = sum(s * w for s, _, w in filtered) / total_w
    b = sum(i * w for _, i, w in filtered) / total_w
    return float(m), float(b)


def detect_lane_lines(
    edges: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 20,
    min_line_length: int = 50,
    max_line_gap: int = 15,
    y_detect_min_ratio: float = 0.52,
    min_slope: float = 0.35,
    max_slope: float = 4.0,
    y_near_ratio: float = 0.78,
    min_separation_px: float = 60.0,
) -> LaneLines:
    """Detect left/right lane lines from an edge image (near-field weighted)."""
    height, width = edges.shape[:2]
    y_detect_min = int(height * y_detect_min_ratio)

    lines = cv2.HoughLinesP(
        edges,
        rho,
        theta,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    left_segments: list[tuple[float, float, float]] = []
    right_segments: list[tuple[float, float, float]] = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (y1 + y2) / 2 < y_detect_min:
                continue

            params = _line_params(x1, y1, x2, y2)
            if params is None:
                continue

            slope, intercept = params
            weight = ((y1 + y2) / 2) / height
            weight *= weight

            if slope < 0:
                left_segments.append((slope, intercept, weight))
            else:
                right_segments.append((slope, intercept, weight))

    left = _weighted_average_line(
        left_segments, positive_slope=False, min_slope=min_slope, max_slope=max_slope
    )
    right = _weighted_average_line(
        right_segments, positive_slope=True, min_slope=min_slope, max_slope=max_slope
    )

    candidate = LaneLines(left=left, right=right)
    if not validate_near_detection(
        candidate.left,
        candidate.right,
        width,
        height,
        y_near_ratio=y_near_ratio,
        min_separation_px=min_separation_px,
        min_slope=min_slope,
        max_slope=max_slope,
    ):
        return LaneLines()

    return candidate


def extrapolate_line(
    line: tuple[float, float],
    y_start: int,
    y_end: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    slope, intercept = line
    if abs(slope) < 1e-6:
        return None

    x_start = int((y_start - intercept) / slope)
    x_end = int((y_end - intercept) / slope)
    x_start = max(0, min(width - 1, x_start))
    x_end = max(0, min(width - 1, x_end))
    return x_start, y_start, x_end, y_end


def lane_polygon(
    lines: LaneLines,
    image_shape: tuple[int, ...],
    y_top_ratio: float = 0.72,
) -> np.ndarray | None:
    """Build a filled polygon between left and right lane lines."""
    if not lines.is_valid():
        return None

    height, width = image_shape[:2]
    y_bottom = height
    y_top = int(height * y_top_ratio)

    assert lines.left is not None and lines.right is not None
    left_seg = extrapolate_line(lines.left, y_top, y_bottom, width)
    right_seg = extrapolate_line(lines.right, y_top, y_bottom, width)
    if left_seg is None or right_seg is None:
        return None

    lx1, _, lx2, _ = left_seg
    rx1, _, rx2, _ = right_seg
    return np.array(
        [
            [lx1, y_top],
            [lx2, y_bottom],
            [rx2, y_bottom],
            [rx1, y_top],
        ],
        dtype=np.int32,
    )
