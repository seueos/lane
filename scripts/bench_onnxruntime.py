#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _make_feed(sess: ort.InferenceSession, batch: int) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        shape = tuple((batch if (d is None and i == 0) else (1 if d is None else int(d))) for i, d in enumerate(inp.shape))
        if any((isinstance(d, str) for d in inp.shape)):
            shape = tuple((batch if i == 0 else 1) for i, _ in enumerate(inp.shape))

        if not inp.type.startswith("tensor(") or not inp.type.endswith(")"):
            raise ValueError(f"Unsupported input type for benchmark: {inp.type} ({inp.name})")
        t = inp.type[len("tensor(") : -1]
        dtype_map = {
            "float": np.float32,
            "float16": np.float16,
            "double": np.float64,
            "int8": np.int8,
            "uint8": np.uint8,
            "int16": np.int16,
            "uint16": np.uint16,
            "int32": np.int32,
            "uint32": np.uint32,
            "int64": np.int64,
            "uint64": np.uint64,
            "bool": np.bool_,
        }
        if t not in dtype_map:
            raise ValueError(f"Unsupported tensor dtype for benchmark: {inp.type} ({inp.name})")
        dtype = dtype_map[t]

        if dtype == np.bool_:
            feed[inp.name] = (np.random.rand(*shape) > 0.5).astype(np.bool_)
        elif np.issubdtype(dtype, np.integer):
            feed[inp.name] = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
        else:
            feed[inp.name] = np.random.rand(*shape).astype(dtype)
    return feed


def main() -> int:
    p = argparse.ArgumentParser(description="Simple ONNX Runtime latency benchmark (single-process).")
    p.add_argument("--onnx", required=True, help="Path to ONNX model.")
    p.add_argument("--batch", type=int, default=1, help="Default batch for dynamic first dim (default: 1).")
    p.add_argument("--warmup", type=int, default=10, help="Warmup runs (default: 10).")
    p.add_argument("--iters", type=int, default=50, help="Measured runs (default: 50).")
    p.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ORT providers preference (default: CPUExecutionProvider).",
    )
    args = p.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    feed = _make_feed(sess, batch=args.batch)

    for _ in range(args.warmup):
        sess.run(None, feed)

    times_ms: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    p50 = statistics.median(times_ms)
    p90 = statistics.quantiles(times_ms, n=10)[8]
    p99 = statistics.quantiles(times_ms, n=100)[98]
    mean = statistics.mean(times_ms)

    print(f"providers={sess.get_providers()}")
    print(f"warmup={args.warmup} iters={args.iters} batch={args.batch}")
    print(f"latency_ms: mean={mean:.3f} p50={p50:.3f} p90={p90:.3f} p99={p99:.3f} min={min(times_ms):.3f} max={max(times_ms):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
