#!/usr/bin/env python3
"""Verify lane_tracking runtime matches expectations (desktop sim or Jetson)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ok = True
    print(f"Python: {sys.version.split()[0]} ({platform.machine()})")

    try:
        import numpy as np

        print(f"numpy: {np.__version__}")
    except ImportError:
        print("numpy: MISSING")
        ok = False

    try:
        import cv2

        print(f"opencv: {cv2.__version__}")
        gst = [b for b in cv2.getBuildInformation().split("\n") if "GStreamer" in b]
        if gst:
            print(f"  {gst[0].strip()}")
    except ImportError:
        print("opencv: MISSING")
        ok = False

    try:
        import onnxruntime as ort

        print(f"onnxruntime: {ort.__version__}")
        print(f"  providers: {ort.get_available_providers()}")
    except ImportError:
        print("onnxruntime: MISSING (optional until you add ONNX models)")
    except Exception as exc:
        print(f"onnxruntime: error — {exc}")
        ok = False

    from src.capture import is_jetson

    jetson = is_jetson()
    sim = __import__("os").environ.get("JETSON_SIM") == "1"
    print(f"is_jetson(): {jetson}" + (" (JETSON_SIM)" if sim and jetson else ""))

    if not ok:
        return 1
    print("Environment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
