"""Resolution scaling helpers for Jetson real-time inference."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .lane_tracker import LaneTracker
from .pipeline import LanePipelineConfig


def resize_frame(
    frame: np.ndarray,
    process_width: int,
) -> tuple[np.ndarray, float]:
    """Downscale frame for processing. Returns (frame, scale) where scale is new_w / orig_w."""
    if process_width <= 0:
        return frame, 1.0

    height, width = frame.shape[:2]
    if width <= process_width:
        return frame, 1.0

    scale = process_width / width
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame,
        (process_width, new_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def upscale_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize annotated output back to the original capture resolution."""
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame

    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def scale_config(config: LanePipelineConfig, scale: float) -> LanePipelineConfig:
    """Scale pixel-based detection parameters for a smaller working resolution."""
    if scale >= 0.999:
        return config

    return replace(
        config,
        min_line_length=max(1, int(round(config.min_line_length * scale))),
        max_line_gap=max(1, int(round(config.max_line_gap * scale))),
        min_separation_px=config.min_separation_px * scale,
    )


def apply_tracker_scale(tracker: LaneTracker, scale: float) -> None:
    """Scale tracker pixel thresholds to match the working resolution."""
    if scale >= 0.999:
        return

    tracker.min_separation_px *= scale
    tracker.max_intercept_delta *= scale
