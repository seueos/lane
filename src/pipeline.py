"""End-to-end lane detection and tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edge_detector import detect_edges
from .lane_detector import LaneLines, detect_lane_lines
from .lane_geometry import find_adaptive_y_top_ratio
from .lane_tracker import LaneTracker
from .visualize import draw_edges_preview, draw_lane_overlay


@dataclass
class LanePipelineConfig:
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel: int = 5
    hough_threshold: int = 20
    min_line_length: int = 50
    max_line_gap: int = 15
    smoothing: float = 0.93
    horizon_smoothing: float = 0.88
    use_tracking: bool = True
    roi_top_min_ratio: float = 0.58
    y_top_min_ratio: float = 0.52
    y_top_max_ratio: float = 0.74
    y_near_ratio: float = 0.78
    y_detect_min_ratio: float = 0.48
    min_separation_px: float = 60.0


class LanePipeline:
    def __init__(self, config: LanePipelineConfig | None = None) -> None:
        self.config = config or LanePipelineConfig()
        self.tracker = LaneTracker(
            smoothing=self.config.smoothing,
            horizon_smoothing=self.config.horizon_smoothing,
            y_top_min_ratio=self.config.y_top_min_ratio,
            y_top_max_ratio=self.config.y_top_max_ratio,
            y_near_ratio=self.config.y_near_ratio,
            min_separation_px=self.config.min_separation_px,
        )

    def reset(self) -> None:
        self.tracker.reset()

    def _roi_top_ratio(self, draw_y_top: float) -> float:
        """Edge ROI extends slightly beyond the draw horizon for far-lane cues."""
        return min(self.config.roi_top_min_ratio, draw_y_top - 0.04)

    def _draw_y_top(
        self,
        lines: LaneLines,
        width: int,
        height: int,
    ) -> float:
        if not lines.is_valid():
            return self.config.y_top_max_ratio

        assert lines.left is not None and lines.right is not None
        if self.config.use_tracking:
            return self.tracker.horizon_for_draw(lines, width, height)

        return find_adaptive_y_top_ratio(
            lines.left,
            lines.right,
            width,
            height,
            y_top_min_ratio=self.config.y_top_min_ratio,
            y_top_max_ratio=self.config.y_top_max_ratio,
            min_separation_px=self.config.min_separation_px,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        show_edges: bool = False,
    ) -> tuple[np.ndarray, LaneLines, np.ndarray]:
        """
        Process one BGR frame.

        Returns (annotated_frame, lane_lines, edge_map).
        """
        height, width = frame.shape[:2]
        draw_y_top = self.tracker.y_top_ratio if self.config.use_tracking else self.config.y_top_max_ratio

        edges = detect_edges(
            frame,
            low_threshold=self.config.canny_low,
            high_threshold=self.config.canny_high,
            blur_kernel=self.config.blur_kernel,
            roi_top_ratio=self._roi_top_ratio(draw_y_top),
        )

        detected = detect_lane_lines(
            edges,
            threshold=self.config.hough_threshold,
            min_line_length=self.config.min_line_length,
            max_line_gap=self.config.max_line_gap,
            y_detect_min_ratio=self.config.y_detect_min_ratio,
            y_near_ratio=self.config.y_near_ratio,
            min_separation_px=self.config.min_separation_px,
        )

        if self.config.use_tracking:
            lines = self.tracker.update(detected, width, height)
        else:
            lines = detected

        draw_y_top = self._draw_y_top(lines, width, height)

        annotated = draw_lane_overlay(
            frame,
            lines,
            y_top_ratio=draw_y_top,
        )
        if show_edges:
            edge_bgr = draw_edges_preview(edges)
            annotated = np.hstack([annotated, edge_bgr])

        return annotated, lines, edges
