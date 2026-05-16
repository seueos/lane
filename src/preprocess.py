"""Image preprocessing for lane detection."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def preprocess_frame(
    image: np.ndarray,
    blur_kernel: int = 5,
) -> np.ndarray:
    gray = to_grayscale(image)
    return apply_gaussian_blur(gray, blur_kernel)
