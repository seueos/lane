#!/usr/bin/env python3
"""Lane tracking demo using edge detection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from src.capture import is_jetson, open_capture
from src.jetson_config import apply_tracker_scale, resize_frame, scale_config, upscale_frame
from src.pipeline import LanePipeline, LanePipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge-based lane detection and tracking",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="0",
        help="Video path, image path, or camera index (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save annotated video to this path",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Show edge map next to the result",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable temporal smoothing",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save result image (for image input) instead of opening a window",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open GUI windows (use with --output or --save)",
    )
    parser.add_argument(
        "--csi",
        action="store_true",
        help="Use Jetson CSI camera via GStreamer (input can be 'csi' or 0)",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=0,
        help="Resize frames to this width before detection (0 = full resolution)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="CSI/USB capture width when using GStreamer",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="CSI/USB capture height when using GStreamer",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Target capture framerate for GStreamer pipelines",
    )
    parser.add_argument(
        "--flip",
        type=int,
        default=0,
        help="Jetson CSI flip-method (0-7) for nvvidconv",
    )
    parser.add_argument(
        "--print-fps",
        action="store_true",
        help="Print processing FPS to stderr every second",
    )
    parser.add_argument(
        "--backend",
        choices=("classical", "onnx"),
        default="classical",
        help="Inference backend: OpenCV pipeline or ONNX Runtime model",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="artifacts/onnx/lane_detector.onnx",
        help="ONNX model path when --backend onnx (default: pinned artifact)",
    )
    return parser.parse_args()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_image(
    pipeline: LanePipeline,
    path: Path,
    show_edges: bool,
    save_path: str | None,
    no_display: bool,
) -> int:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Failed to read image: {path}", file=sys.stderr)
        return 1

    annotated, lines, _ = pipeline.process_frame(frame, show_edges=show_edges)

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved: {save_path}")
        print(f"Lanes detected: {lines.is_valid()}")

    if not no_display and not save_path:
        window = "Lane Tracking"
        cv2.imshow(window, annotated)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif not save_path and no_display:
        print("Use --save with --no-display for headless image runs.", file=sys.stderr)
        return 1

    return 0


def run_video(
    pipeline: LanePipeline,
    source: str,
    output: str | None,
    show_edges: bool,
    no_display: bool,
    *,
    use_csi: bool = False,
    process_width: int = 0,
    camera_width: int = 1280,
    camera_height: int = 720,
    camera_fps: int = 30,
    flip_method: int = 0,
    print_fps: bool = False,
) -> int:
    cap = open_capture(
        source,
        use_csi=use_csi,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        flip_method=flip_method,
    )
    if not cap.isOpened():
        print(f"Failed to open input: {source}", file=sys.stderr)
        return 1

    writer = None
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or camera_width
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or camera_height
    if output:
        width = out_width
        height = out_height
        if show_edges:
            width *= 2
        fps = cap.get(cv2.CAP_PROP_FPS) or float(camera_fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    window = "Lane Tracking"
    frame_count = 0
    fps_start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        work, scale = resize_frame(frame, process_width)
        annotated, _, _ = pipeline.process_frame(work, show_edges=show_edges)
        if scale < 0.999:
            annotated = upscale_frame(annotated, orig_w, orig_h)

        if writer is not None:
            writer.write(annotated)

        if not no_display:
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        if print_fps:
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                print(f"FPS: {frame_count / elapsed:.1f}", file=sys.stderr)
                frame_count = 0
                fps_start = time.perf_counter()

    cap.release()
    if writer is not None:
        writer.release()
    if not no_display:
        cv2.destroyAllWindows()
    return 0


def _is_live_source(source: str, use_csi: bool) -> bool:
    lowered = source.strip().lower()
    if use_csi or lowered in {"csi", "jetson", "nvargus"}:
        return True
    return source.isdigit()


def main() -> int:
    args = parse_args()
    source = args.input
    use_csi = args.csi or source.strip().lower() in {"csi", "jetson", "nvargus"}
    if use_csi and source in {"0", "csi", "jetson"}:
        source = "csi"

    process_width = args.process_width
    if process_width <= 0 and is_jetson() and _is_live_source(source, use_csi):
        process_width = 640

    ref_w = args.camera_width if use_csi else 1280
    scale = process_width / ref_w if process_width > 0 else 1.0

    if args.backend == "onnx":
        from pathlib import Path as _Path

        from src.onnx_lane.pipeline import OnnxLanePipeline, OnnxLanePipelineConfig

        model_path = _Path(args.onnx_model)
        if not model_path.is_file():
            print(
                f"ONNX model not found: {model_path}\n"
                "  bash scripts/export_onnx.sh --src /path/to/lane_detector.onnx --name lane_detector",
                file=sys.stderr,
            )
            return 1
        onnx_cfg = OnnxLanePipelineConfig(
            model_path=str(model_path),
            use_tracking=not args.no_track,
        )
        pipeline = OnnxLanePipeline(onnx_cfg)
        if scale < 0.999:
            apply_tracker_scale(pipeline.tracker, scale)
    else:
        config = LanePipelineConfig(
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            use_tracking=not args.no_track,
        )
        if scale < 0.999:
            config = scale_config(config, scale)
            pipeline = LanePipeline(config)
            apply_tracker_scale(pipeline.tracker, scale)
        else:
            pipeline = LanePipeline(config)

    path = Path(source)

    if path.exists() and _is_image(path):
        return run_image(
            pipeline,
            path,
            args.show_edges,
            args.save or args.output,
            args.no_display,
        )

    return run_video(
        pipeline,
        source,
        args.output,
        args.show_edges,
        args.no_display,
        use_csi=use_csi,
        process_width=process_width,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        flip_method=args.flip,
        print_fps=args.print_fps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
