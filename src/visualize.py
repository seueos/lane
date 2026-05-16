"""Overlay lane detection results on frames."""

from __future__ import annotations

import cv2
import numpy as np

from .lane_detector import LaneLines, extrapolate_line, lane_polygon


def draw_lane_overlay(
    frame: np.ndarray,
    lines: LaneLines,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
    line_thickness: int = 8,
    y_top_ratio: float = 0.72,
) -> np.ndarray:
    output = frame.copy()
    height, width = frame.shape[:2]
    y_bottom = height
    y_top = int(height * y_top_ratio)

    polygon = lane_polygon(lines, frame.shape, y_top_ratio=y_top_ratio)
    if polygon is not None:
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], color)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    for line, lane_color in ((lines.left, (255, 0, 0)), (lines.right, (0, 0, 255))):
        if line is None:
            continue
        segment = extrapolate_line(line, y_top, y_bottom, width)
        if segment is not None:
            cv2.line(output, segment[:2], segment[2:], lane_color, line_thickness)

    return output


def draw_edges_preview(edges: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
