#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Copy an ONNX file into artifacts/onnx with metadata (sha256, size).",
    )
    p.add_argument("--src", required=True, help="Source ONNX path to copy.")
    p.add_argument(
        "--name",
        default="lane_detector",
        help="Artifact base name (default: lane_detector). Creates <name>.onnx and <name>.json.",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path("artifacts") / "onnx"),
        help="Output directory (default: artifacts/onnx).",
    )
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: src does not exist: {src}", file=sys.stderr)
        return 2
    if src.suffix.lower() != ".onnx":
        print(f"ERROR: src must be a .onnx file: {src}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_onnx = out_dir / f"{args.name}.onnx"
    dst_meta = out_dir / f"{args.name}.json"

    shutil.copy2(src, dst_onnx)

    stat = dst_onnx.stat()
    meta = {
        "name": args.name,
        "src": str(src),
        "dst": str(dst_onnx),
        "sha256": _sha256(dst_onnx),
        "size_bytes": stat.st_size,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
        },
        "cwd": os.getcwd(),
    }
    dst_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote: {dst_onnx}")
    print(f"Wrote: {dst_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
