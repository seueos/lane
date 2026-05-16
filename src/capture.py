"""Video capture helpers for desktop and NVIDIA Jetson."""

from __future__ import annotations

import os
from pathlib import Path

import cv2


def is_jetson() -> bool:
    """True when running on a Jetson device (L4T / Tegra)."""
    if Path("/etc/nv_tegra_release").is_file():
        return True

    model_path = Path("/proc/device-tree/model")
    if model_path.is_file():
        try:
            model = model_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            model = ""
        model_lower = model.lower()
        if "jetson" in model_lower or "tegra" in model_lower:
            return True

    return os.environ.get("JETSON_SIM") == "1"


def csi_gstreamer_pipeline(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """GStreamer pipeline for Jetson CSI camera (nvarguscamerasrc)."""
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){width}"
        f", height=(int){height}, format=(string)NV12, framerate=(fraction){fps}"
        f"/1 ! nvvidconv flip-method={flip_method}"
        f" ! video/x-raw, width=(int){width}, height=(int){height}"
        ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR"
        " ! appsink drop=true max-buffers=1"
    )


def open_capture(
    source: str,
    *,
    use_csi: bool = False,
    camera_width: int = 1280,
    camera_height: int = 720,
    camera_fps: int = 30,
    flip_method: int = 0,
) -> cv2.VideoCapture:
    """
    Open a video file, USB camera, or Jetson CSI stream.

    ``source`` may be a path, numeric camera index, or ``csi`` / ``jetson``.
    """
    lowered = source.strip().lower()
    if use_csi or lowered in {"csi", "jetson", "nvargus"}:
        if not is_jetson():
            raise RuntimeError(
                "CSI capture requires a Jetson with nvarguscamerasrc. "
                "On a desktop, use a video file or USB camera index (e.g. 0)."
            )

        pipeline = csi_gstreamer_pipeline(
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            flip_method=flip_method,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        raise RuntimeError(f"Failed to open CSI camera via GStreamer:\n{pipeline}")

    if lowered.isdigit():
        return cv2.VideoCapture(int(lowered))

    return cv2.VideoCapture(source)
