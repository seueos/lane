#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main() -> int:
    p = argparse.ArgumentParser(description="Validate ONNX by running a single ORT inference with random inputs.")
    p.add_argument("--onnx", required=True, help="Path to ONNX model.")
    p.add_argument("--batch", type=int, default=1, help="Default batch for dynamic first dim (default: 1).")
    p.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ORT providers preference (default: CPUExecutionProvider).",
    )
    args = p.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.exists():
        print(f"ERROR: ONNX not found: {onnx_path}", file=sys.stderr)
        return 2

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    feed: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        shape = tuple((args.batch if (d is None and i == 0) else (1 if d is None else int(d))) for i, d in enumerate(inp.shape))
        if any((isinstance(d, str) for d in inp.shape)):
            shape = tuple((args.batch if i == 0 else 1) for i, _ in enumerate(inp.shape))

        if inp.type == "tensor(string)":
            raise ValueError(f"String tensor input not supported for random validation: {inp.name}")

        if inp.type.startswith("tensor(") and inp.type.endswith(")"):
            ort_type = inp.type[len("tensor(") : -1]
            ort_map = {
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
            if ort_type not in ort_map:
                raise ValueError(f"Unsupported ORT input type for random validation: {inp.type} ({inp.name})")
            dtype = ort_map[ort_type]
        else:
            raise ValueError(f"Unsupported ORT input type for random validation: {inp.type} ({inp.name})")

        if dtype == np.bool_:
            feed[inp.name] = (np.random.rand(*shape) > 0.5).astype(np.bool_)
        elif np.issubdtype(dtype, np.integer):
            feed[inp.name] = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
        else:
            feed[inp.name] = np.random.rand(*shape).astype(dtype)

    outputs = sess.run(None, feed)

    print("OK: onnx.checker passed")
    print(f"OK: ORT ran 1 inference with providers={sess.get_providers()}")
    print(f"Inputs: {[i.name for i in sess.get_inputs()]}")
    print(f"Outputs: {[o.name for o in sess.get_outputs()]}")
    print(f"Output dtypes/shapes: {[(np.asarray(x).dtype.name, tuple(np.asarray(x).shape)) for x in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
