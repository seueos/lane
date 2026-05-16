"""Canny edge detection with ROI masking."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import preprocess_frame


def region_of_interest(
    image: np.ndarray,
    vertices: np.ndarray | None = None,
    roi_top_ratio: float = 0.72,
) -> np.ndarray:
    """Mask image to a trapezoidal driving corridor (near-field only)."""
    height, width = image.shape[:2]
    if vertices is None:
        y_top = int(height * roi_top_ratio)
        vertices = np.array(
            [
                [
                    (int(width * 0.08), height),
                    (int(width * 0.42), y_top),
                    (int(width * 0.58), y_top),
                    (int(width * 0.92), height),
                ]
            ],
            dtype=np.int32,
        )

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def detect_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_kernel: int = 5,
    apply_roi: bool = True,
    roi_top_ratio: float = 0.72,
) -> np.ndarray:
    """
    Run preprocessing + Canny edge detection.

    Returns a single-channel edge map (uint8, 0 or 255).
    """
    preprocessed = preprocess_frame(image, blur_kernel)
    edges = cv2.Canny(preprocessed, low_threshold, high_threshold)

    if apply_roi:
        edges = region_of_interest(edges, roi_top_ratio=roi_top_ratio)

    return edges
