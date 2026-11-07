#!/usr/bin/env python3
"""Writes `python_written.safetensors`, read by `tests/python_interop.rs`.

Run with Python `safetensors` and `numpy` installed (any recent versions; the committed file was
written with safetensors 0.8.0 and numpy 2.5.4):

    python generate_python_fixture.py

Fixed values, no randomness; the test hard-codes the expected values. The order of the
`__metadata__` entries in the header can differ between runs (Python dict order), the content can not.
`f16_as_f32` holds numpy's own f16 -> f32 conversion of `f16`, the reference for Polypous's
`f16_to_f32`.
"""

import pathlib

import numpy as np
from safetensors.numpy import save_file

f16_bits = np.array(
    [0x0000, 0x8000, 0x3C00, 0xC000, 0x7BFF, 0x0001, 0x03FF, 0x0400, 0x7C00, 0xFC00, 0x3555, 0x7E00],
    dtype=np.uint16,
)
f16 = f16_bits.view(np.float16)

tensors = {
    "f32": np.array([[0.5, -1.25, 3.0], [1e-7, -0.0, 65504.0]], dtype=np.float32),
    "f16": f16,
    "f16_as_f32": f16.astype(np.float32),
    "i64": np.array([-(2**63), -1, 0, 1, 2**63 - 1], dtype=np.int64),
    "i32": np.array([[-(2**31)], [7], [2**31 - 1]], dtype=np.int32),
    "bool": np.array([True, False, True], dtype=np.bool_),
    "u8": np.arange(4, dtype=np.uint8),
    "f64": np.array([1.5], dtype=np.float64),
    "scalar": np.array(2.5, dtype=np.float32),
    "empty": np.zeros((0, 3), dtype=np.float32),
}

out = pathlib.Path(__file__).with_name("python_written.safetensors")
save_file(tensors, str(out), metadata={"format": "np", "source": "generate_python_fixture.py"})
print(f"wrote {out} ({out.stat().st_size} bytes)")
