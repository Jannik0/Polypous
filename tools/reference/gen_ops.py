#!/usr/bin/env python3
"""Generates the op reference fixtures in fixtures/small/ops/ from PyTorch.

    python tools/reference/gen_ops.py            # regenerate fixtures/small/ops/
    python tools/reference/gen_ops.py --check    # fail if the committed fixtures differ

Layout and sidecar schema: docs/testing-strategy.md §3. Every case is declared in `CASES` below,
with its own fixed seed. Input data comes from NumPy's PCG64 (`Generator.random` and
`Generator.integers` are integer arithmetic, so they give the same values on every CPU and OS).
Outputs are computed with PyTorch on the CPU; matmul, pow and the transcendental ops are computed in
f64 and rounded once to the dtype under test, so the bytes don't depend on the BLAS or vectorized
kernel the CPU picks (see `ref_matmul` and `OpSpec.f64`). PyTorch's own result in the dtype under
test is checked against the tolerance.

Needs the pinned versions from requirements.txt; `--check` refuses to compare otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import platform
import re
import sys
import tempfile
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixturelib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO, "fixtures", "small", "ops")
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
GENERATOR = "tools/reference/gen_ops.py"
# The Python version the committed fixtures are generated with (docs/dev-environment.md). The
# tensors don't depend on it, but it is recorded in every sidecar.
PYTHON = "3.14"

try:
    import numpy as np
    import safetensors
    import safetensors.torch
    import torch
except ImportError as e:  # pragma: no cover - depends on the environment
    np = safetensors = torch = None  # type: ignore[assignment]
    IMPORT_ERROR: ImportError | None = e
else:
    IMPORT_ERROR = None

# ---------------------------------------------------------------------------------------------
# Tolerances (docs/testing-strategy.md §2), written into every sidecar.

EXACT = {"exact": True, "atol": 0.0, "rtol": 0.0}
F32_ELEMENTWISE = {"exact": False, "atol": 1e-5, "rtol": 1e-4}
F32_MATMUL = {"exact": False, "atol": 1e-5, "rtol": 1e-3}
# f16/bf16: §2 gives rtol 1e-2 and no atol (also for outputs near zero, which every backend meets by
# computing f16/bf16 transcendentals in f32).
LOW_PRECISION = {"exact": False, "atol": 0.0, "rtol": 1e-2}


def elementwise_tol(dtype: str) -> dict[str, Any]:
    if dtype in ("i64", "i32", "bool"):
        return EXACT
    return F32_ELEMENTWISE if dtype == "f32" else LOW_PRECISION


def matmul_tol(dtype: str) -> dict[str, Any]:
    return F32_MATMUL if dtype == "f32" else LOW_PRECISION


# ---------------------------------------------------------------------------------------------
# Input data.


def torch_dtype(name: str) -> Any:
    return {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "i64": torch.int64,
        "i32": torch.int32,
        "bool": torch.bool,
    }[name]


def uniform(rng: Any, shape: list[int], dtype: str, low: float = -2.0, high: float = 2.0) -> Any:
    """Uniform values in [low, high), drawn in f64 and rounded once to `dtype`."""
    values = rng.random(size=shape) * (high - low) + low
    return torch.from_numpy(values).to(torch_dtype(dtype))


def integers(rng: Any, shape: list[int], dtype: str, low: int, high: int) -> Any:
    values = rng.integers(low, high, size=shape, dtype=np.int64)
    return torch.from_numpy(values).to(torch_dtype(dtype))


def from_list(values: list[Any], dtype: str, shape: list[int] | None = None) -> Any:
    """A tensor from Python values, rounded once from f64 (floats) to `dtype`."""
    if dtype == "bool":
        t = torch.tensor(values, dtype=torch.bool)
    elif dtype in ("i64", "i32"):
        t = torch.tensor(values, dtype=torch.int64).to(torch_dtype(dtype))
    else:
        t = torch.tensor(values, dtype=torch.float64).to(torch_dtype(dtype))
    return t.reshape(shape) if shape is not None else t


# ---------------------------------------------------------------------------------------------
# Cases.


@dataclasses.dataclass(frozen=True)
class Case:
    op: str  # IR op name, docs/specs/graph-ir.md §3
    name: str
    description: str
    seed: int
    attributes: dict[str, Any]
    # Builds the named inputs, in order, from a seeded NumPy generator.
    inputs: Callable[[Any], list[tuple[str, Any]]]
    # Computes the reference outputs from the inputs, in order.
    reference: Callable[[list[Any]], list[tuple[str, Any]]]
    reference_text: str  # the PyTorch expression, for the sidecar
    tolerance: dict[str, Any]
    notes: str = ""
    # For references computed in f64 and rounded once: PyTorch's own result in the dtype under
    # test, which must meet the tolerance (checked by the generator, see `check_native`).
    native: Callable[[list[Any]], Any] | None = None


def ref_matmul(xs: list[Any]) -> list[tuple[str, Any]]:
    # f64, rounded once to the dtype under test: bit-reproducible on every CPU. A backend's f32
    # (or f16/bf16) accumulation differs by rounding only, which the tolerance covers.
    a, b = xs
    return [("out", torch.matmul(a.double(), b.double()).to(a.dtype))]


def check_native(case: Case, xs: list[Any], outs: list[tuple[str, Any]]) -> None:
    """PyTorch's own result in the dtype under test must meet the case's tolerance (NaN where the
    reference has NaN, the same inf where it has one)."""
    if case.native is None:
        return
    native = case.native(xs).double()
    expected = outs[0][1].double()
    tol = case.tolerance
    if not torch.allclose(native, expected, atol=tol["atol"], rtol=tol["rtol"], equal_nan=True):
        worst = (native - expected).abs().nan_to_num(0.0).max().item()
        raise SystemExit(f"{case.op}/{case.name}: native torch misses the tolerance (max abs err {worst})")


MATMUL_NOTES = (
    "out is torch.matmul in f64, rounded once to the dtype under test, so the bytes don't depend on "
    "the CPU's BLAS kernel; torch.matmul in the dtype itself meets the tolerance (checked by the generator)."
)


def matmul_case(name: str, seed: int, dtype: str, lhs: list[int], rhs: list[int], description: str) -> Case:
    return Case(
        op="core.matmul",
        name=name,
        description=description,
        seed=seed,
        attributes={},
        inputs=lambda rng: [("a", uniform(rng, lhs, dtype)), ("b", uniform(rng, rhs, dtype))],
        reference=ref_matmul,
        reference_text="torch.matmul(a.double(), b.double()).to(a.dtype)",
        tolerance=matmul_tol(dtype),
        notes=MATMUL_NOTES,
        native=lambda xs: torch.matmul(xs[0], xs[1]),
    )


def reshape_case(name: str, seed: int, dtype: str, shape: list[int], new_shape: list[int]) -> Case:
    def inputs(rng: Any) -> list[tuple[str, Any]]:
        if dtype == "bool":
            return [("x", torch.from_numpy(rng.integers(0, 2, size=shape, dtype=np.int64)).to(torch.bool))]
        if dtype in ("i64", "i32"):
            return [("x", integers(rng, shape, dtype, -1000, 1000))]
        return [("x", uniform(rng, shape, dtype))]

    return Case(
        op="core.reshape",
        name=name,
        description=f"{dtype} {shape} -> {new_shape}, row-major order kept.",
        seed=seed,
        attributes={"shape": new_shape},
        inputs=inputs,
        reference=lambda xs: [("out", torch.reshape(xs[0], new_shape))],
        reference_text="torch.reshape(x, shape)",
        tolerance=EXACT,
    )


# Cast inputs. Values are rounded once to the source dtype; per target, values whose result is not
# pinned down by PyTorch's documented semantics are left out (see `cast_values`).
CAST_FLOAT_VALUES = [
    0.0, -0.0, 1.0, -1.0, 0.5, -2.75, 1.0 / 3.0, -0.1,
    1.00390625,  # 1 + 2^-8: bf16 tie, rounds to even (1.0)
    1.01171875,  # 1 + 3·2^-8: bf16 tie, rounds to even (1.015625)
    1.00048828125,  # 1 + 2^-11: f16 tie, rounds to even (1.0)
    2049.0,  # f16 rounds to 2048
    65504.0,  # largest f16
    65520.0,  # rounds to inf in f16
    1e5, -1e5,  # inf in f16
    6e-8,  # f16 subnormal
    1e-30,  # underflows to 0 in f16, normal in bf16
    3.0e38,  # finite in f32 and bf16
    float("inf"), float("-inf"), float("nan"),
    123456.789, -7.5,
]
CAST_FLOAT_TO_INT_VALUES = [
    0.0, -0.0, 0.4, -0.4, 0.99, -0.99, 1.0, -1.5, 2.5, -2.5, 7.9, -7.9,
    100.25, -100.75, 1000.0, -1024.5, 65504.0, -65504.0, 123456.7, -2147483648.0, 2147483520.0,
]
CAST_INT_VALUES = [
    0, 1, -1, 2, -7, 127, -128, 255, 256, 257, 2049, -2049, 65504, -65504,
    16777217, -16777217, 2147483647, -2147483648,
]
CAST_I64_ONLY_VALUES = [2**40 + 3, -(2**53) - 1]
CAST_BOOL_VALUES = [True, False, True, True, False, False]
DTYPE_NAMES = ["f32", "f16", "bf16", "i64", "i32", "bool"]
FLOATS = ("f32", "f16", "bf16")
INTS = ("i64", "i32")


def cast_values(src: str, dst: str) -> tuple[Any, str]:
    """The input tensor for a cast case, and the sidecar note on what was left out and why."""
    if src == "bool":
        return from_list(CAST_BOOL_VALUES, "bool", [2, 3]), ""
    if src in FLOATS and dst in INTS:
        x = from_list(CAST_FLOAT_TO_INT_VALUES, src)
        # Only values that are finite and in i32 range after rounding to the source dtype.
        x64 = x.double()
        x = x[(x64 >= -(2.0**31)) & (x64 < 2.0**31)]
        return x, (
            "Float to int truncates toward zero. Inputs are finite, not NaN and inside the i32 range "
            "after rounding to the source dtype; out-of-range and NaN conversions are undefined in PyTorch."
        )
    if src in FLOATS:
        note = "Includes ±0.0, ±inf, NaN, subnormals, f16/bf16 rounding ties and overflow to inf."
        if dst == "bool":
            note += " Non-zero is true, NaN is true, -0.0 is false."
        if dst in FLOATS and dst != src:
            note += " NaN payloads are not part of the semantics: any NaN matches any NaN."
        return from_list(CAST_FLOAT_VALUES, src), note
    # Integer source.
    values = CAST_INT_VALUES + (CAST_I64_ONLY_VALUES if src == "i64" and dst in ("i64", "f32", "bool") else [])
    note = "Int to float rounds to nearest, ties to even."
    if dst in ("f16", "bf16"):
        # PyTorch converts through f32 first; keep |v| < 2^24 so that step is exact and there is
        # one rounding. f16 also drops values above its range (overflow to inf).
        values = [v for v in values if abs(v) < 2**24 and (dst != "f16" or abs(v) <= 65504)]
        note += " Values are below 2^24 (no double rounding through f32)"
        note += " and inside the f16 range." if dst == "f16" else "."
    if src == "i64" and dst == "i32":
        note = "Values are inside the i32 range; narrowing out-of-range integers is not tested."
    if dst == "bool":
        note = "Non-zero is true."
    if dst == src:
        note = "Cast to the same dtype is the identity."
    return from_list(values, src), note


def cast_case(src: str, dst: str, seed: int) -> Case:
    x, note = cast_values(src, dst)
    return Case(
        op="core.cast",
        name=f"{src}_to_{dst}",
        description=f"Tensor.to({dst}) from {src}.",
        seed=seed,
        attributes={"dtype": dst},
        inputs=lambda rng: [("x", x)],
        reference=lambda xs: [("out", xs[0].to(torch_dtype(dst)))],
        reference_text=f"x.to({dst})",
        tolerance=EXACT,
        notes=note,
    )


# ---------------------------------------------------------------------------------------------
# Elementwise, comparison and logic ops (docs/specs/graph-ir.md §3.2). Each op is an `OpSpec`;
# `ew_case` turns an op, a dtype and input shapes (or explicit values) into a Case.

INF, NAN = float("inf"), float("nan")
I32_MIN, I32_MAX = -(2**31), 2**31 - 1
I64_MIN, I64_MAX = -(2**63), 2**63 - 1
# Ranges of random integer inputs: small enough that add/sub/mul don't overflow (the wrap cases
# cover overflow on purpose).
INT_RANGE = {"i32": (-1000, 1000), "i64": (-(2**30), 2**30)}


@dataclasses.dataclass(frozen=True)
class OpSpec:
    op: str
    inputs: tuple[str, ...]
    # The PyTorch call on tensors (and the case's attributes, bound by the caller).
    fn: Callable[..., Any]
    text: str  # reference text, with the input names
    # Float results computed in f64 and rounded once (transcendentals and pow), so the bytes don't
    # depend on the CPU's vectorized kernels. PyTorch's own result in the dtype under test is
    # checked against the tolerance.
    f64: bool = False
    # Default range of random float inputs, per input.
    ranges: tuple[tuple[float, float], ...] = ()
    notes: str = ""


F64_NOTES = (
    "out is computed in f64 from the inputs and rounded once to the dtype under test, so the bytes "
    "don't depend on the CPU's vectorized kernels; torch in the dtype itself meets the tolerance "
    "(checked by the generator)."
)
NAN_NOTES = "Expected NaN must be NaN (any NaN matches any NaN) and expected ±inf must be the same inf."
WRAP_NOTES = "Integer overflow wraps (two's complement), as in PyTorch (graph-ir.md §3.1.8)."
TIE_NOTES = "Equal values pick either operand; inputs avoid -0.0/0.0 ties, where the choice would show."


def _binary(op: str, fn: Callable[..., Any], text: str, **kw: Any) -> OpSpec:
    return OpSpec(op=op, inputs=("a", "b"), fn=fn, text=text, **kw)


def _unary(op: str, fn: Callable[..., Any], text: str, **kw: Any) -> OpSpec:
    return OpSpec(op=op, inputs=("x",), fn=fn, text=text, **kw)


ADD = _binary("core.add", torch.add if torch else None, "torch.add(a, b)")
SUB = _binary("core.sub", torch.sub if torch else None, "torch.sub(a, b)")
MUL = _binary("core.mul", torch.mul if torch else None, "torch.mul(a, b)")
DIV = _binary("core.div", torch.div if torch else None, "torch.div(a, b)", ranges=((-2.0, 2.0), (0.5, 2.5)))
POW = _binary(
    "core.pow", torch.pow if torch else None, "torch.pow(a, b)", f64=True, ranges=((0.1, 4.0), (-3.0, 3.0))
)
MAXIMUM = _binary("core.maximum", torch.maximum if torch else None, "torch.maximum(a, b)", notes=TIE_NOTES)
MINIMUM = _binary("core.minimum", torch.minimum if torch else None, "torch.minimum(a, b)", notes=TIE_NOTES)
NEG = _unary("core.neg", torch.neg if torch else None, "torch.neg(x)")
ABS = _unary("core.abs", torch.abs if torch else None, "torch.abs(x)")
EXP = _unary("core.exp", torch.exp if torch else None, "torch.exp(x)", f64=True, ranges=((-10.0, 10.0),))
LOG = _unary("core.log", torch.log if torch else None, "torch.log(x)", f64=True, ranges=((0.01, 100.0),))
SQRT = _unary("core.sqrt", torch.sqrt if torch else None, "torch.sqrt(x)", f64=True, ranges=((0.0, 100.0),))
RSQRT = _unary("core.rsqrt", torch.rsqrt if torch else None, "torch.rsqrt(x)", f64=True, ranges=((0.01, 100.0),))
SIN = _unary("core.sin", torch.sin if torch else None, "torch.sin(x)", f64=True, ranges=((-10.0, 10.0),))
COS = _unary("core.cos", torch.cos if torch else None, "torch.cos(x)", f64=True, ranges=((-10.0, 10.0),))
TANH = _unary("core.tanh", torch.tanh if torch else None, "torch.tanh(x)", f64=True, ranges=((-5.0, 5.0),))
EQ = _binary("core.eq", torch.eq if torch else None, "torch.eq(a, b)")
NE = _binary("core.ne", torch.ne if torch else None, "torch.ne(a, b)")
LT = _binary("core.lt", torch.lt if torch else None, "torch.lt(a, b)")
LE = _binary("core.le", torch.le if torch else None, "torch.le(a, b)")
GT = _binary("core.gt", torch.gt if torch else None, "torch.gt(a, b)")
GE = _binary("core.ge", torch.ge if torch else None, "torch.ge(a, b)")
AND = _binary("core.and", torch.logical_and if torch else None, "torch.logical_and(a, b)")
OR = _binary("core.or", torch.logical_or if torch else None, "torch.logical_or(a, b)")
NOT = _unary("core.not", torch.logical_not if torch else None, "torch.logical_not(x)")
WHERE = OpSpec(
    op="core.where", inputs=("cond", "a", "b"), fn=torch.where if torch else None, text="torch.where(cond, a, b)"
)


def random_input(rng: Any, shape: list[int], dtype: str, rng_range: tuple[float, float] = (-2.0, 2.0)) -> Any:
    """Random values of `dtype`: uniform floats in `rng_range`, integers in `INT_RANGE`, fair bools."""
    if dtype == "bool":
        return torch.from_numpy(rng.integers(0, 2, size=shape, dtype=np.int64)).to(torch.bool)
    if dtype in INTS:
        return integers(rng, shape, dtype, *INT_RANGE[dtype])
    return uniform(rng, shape, dtype, *rng_range)


def ties_input(rng: Any, shape: list[int], dtype: str) -> Any:
    """Small integers (as `dtype`), so that comparisons see many equal pairs."""
    if dtype == "bool":
        return random_input(rng, shape, dtype)
    return integers(rng, shape, "i64", -3, 4).to(torch_dtype(dtype))


def ew_case(
    spec: OpSpec,
    name: str,
    seed: int,
    dtype: str,
    description: str,
    shapes: list[list[int]] | None = None,
    values: list[list[Any]] | None = None,
    make: Callable[[Any, list[int], str], Any] | None = None,
    attributes: dict[str, Any] | None = None,
    fn: Callable[..., Any] | None = None,
    text: str | None = None,
    notes: str = "",
) -> Case:
    """A case of an elementwise op. Inputs are `values` (lists, with `shapes` if given) or random
    data of `shapes` (`make`, default `random_input` with the op's ranges). `fn`/`text` override the
    op's call (for attributes)."""
    fn = fn or spec.fn
    text = text or spec.text
    dtypes = ["bool" if n == "cond" else dtype for n in spec.inputs]

    def inputs(rng: Any) -> list[tuple[str, Any]]:
        out = []
        for i, (n, dt) in enumerate(zip(spec.inputs, dtypes)):
            shape = shapes[i] if shapes is not None else None
            if values is not None:
                out.append((n, from_list(values[i], dt, shape)))
            elif make is not None and dt == dtype:
                out.append((n, make(rng, shape, dt)))
            else:
                r = spec.ranges[i] if i < len(spec.ranges) else (-2.0, 2.0)
                out.append((n, random_input(rng, shape, dt, r)))
        return out

    f64 = spec.f64 and dtype in FLOATS

    def reference(xs: list[Any]) -> list[tuple[str, Any]]:
        if f64:
            return [("out", fn(*[x.double() for x in xs]).to(torch_dtype(dtype)))]
        return [("out", fn(*xs))]

    probe = fn(*[torch.zeros(1, dtype=torch_dtype(d)) for d in dtypes])
    out_dtype = DTYPE_OF_TORCH()[probe.dtype]
    all_notes = " ".join(x for x in (spec.notes, notes, F64_NOTES if f64 else "") if x)
    return Case(
        op=spec.op,
        name=name,
        description=description,
        seed=seed,
        attributes=attributes or {},
        inputs=inputs,
        reference=reference,
        reference_text=f"{text} in f64, rounded once to {dtype}" if f64 else text,
        tolerance=elementwise_tol(out_dtype),
        notes=all_notes,
        native=(lambda xs: fn(*xs)) if f64 else None,
    )


def DTYPE_OF_TORCH() -> dict[Any, str]:  # noqa: N802 - mirrors DTYPE_OF
    return {torch_dtype(n): n for n in DTYPE_NAMES}


def wrap_values(dtype: str) -> tuple[list[int], list[int]]:
    lo, hi = (I32_MIN, I32_MAX) if dtype == "i32" else (I64_MIN, I64_MAX)
    a = [hi, lo, hi, lo, -1, 1, 0, hi // 2 + 7, lo // 3, 46341 if dtype == "i32" else 3037000500, hi, lo]
    b = [1, -1, hi, lo, lo, hi, lo, hi // 2 + 9, -5, 46341 if dtype == "i32" else 3037000500, -1, 1]
    return a, b


def binary_arith_cases(spec: OpSpec, seed: int, special: tuple[list[float], list[float]] | None) -> list[Case]:
    """Shared case list of add, sub and mul: every dtype, broadcasting, large values, wrap-around."""
    cases = [
        ew_case(spec, "same_shape_f32", seed + 1, "f32", "Same shape [3, 4].", [[3, 4], [3, 4]]),
        ew_case(spec, "broadcast_row_f32", seed + 2, "f32", "[2, 3, 4] and [1, 1, 4].", [[2, 3, 4], [1, 1, 4]]),
        ew_case(spec, "broadcast_both_f32", seed + 3, "f32", "[2, 1, 4] and [1, 3, 1]: both sides broadcast to [2, 3, 4].", [[2, 1, 4], [1, 3, 1]]),
        ew_case(spec, "broadcast_rank_f32", seed + 4, "f32", "[2, 3, 4] and [4]: different ranks, size-1 dims are prepended.", [[2, 3, 4], [4]]),
        ew_case(spec, "broadcast_scalar_f32", seed + 5, "f32", "[1] and [2, 3]: a scalar on the left.", [[1], [2, 3]]),
        ew_case(spec, "large_f32", seed + 6, "f32", "Large magnitudes (up to 1e18) in [3, 4].", [[3, 4], [3, 4]], make=lambda rng, s, d: uniform(rng, s, d, -1e18, 1e18)),
        ew_case(spec, "broadcast_f16", seed + 7, "f16", "[2, 3, 4] and [2, 1, 4] in f16.", [[2, 3, 4], [2, 1, 4]]),
        ew_case(spec, "broadcast_bf16", seed + 8, "bf16", "[2, 3, 4] and [2, 1, 4] in bf16.", [[2, 3, 4], [2, 1, 4]]),
        ew_case(spec, "broadcast_i64", seed + 9, "i64", "[3, 4] and [1, 4] in i64.", [[3, 4], [1, 4]]),
        ew_case(spec, "broadcast_rank_i32", seed + 10, "i32", "[2, 3, 4] and [3, 1] in i32: different ranks.", [[2, 3, 4], [3, 1]]),
    ]
    for i, dt in enumerate(INTS):
        a, b = wrap_values(dt)
        cases.append(ew_case(spec, f"wrap_{dt}", seed + 11 + i, dt, f"Overflow at the {dt} limits wraps around, [12].", values=[a, b], notes=WRAP_NOTES))
    a, b = wrap_values("i64")
    cases.append(
        ew_case(spec, "wrap_broadcast_i64", seed + 13, "i64", "i64 overflow with broadcasting: [12, 1] and [3].",
                [[12, 1], [3]], values=[a, [I64_MAX, I64_MIN, -1]], notes=WRAP_NOTES)
    )
    if special is not None:
        cases.append(
            ew_case(spec, "special_values_f32", seed + 14, "f32", "Signed zeros, infinities, NaN and overflow, [3, 4].",
                    [[3, 4], [3, 4]], values=list(special),
                    notes=NAN_NOTES + " The tolerance can't see the sign of zero; the bytes keep it.")
        )
    return cases


UNARY_SPECIAL = {
    "core.exp": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 88.0, 89.0, -87.0, -104.0, 1e-30],
    "core.log": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 1e-30, 1e30, 2.718281828459045, 0.5, 3e38],
    "core.sqrt": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 1e-30, 1e30, 2.0, 0.25, 3e38],
    "core.rsqrt": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 1e-30, 1e30, 2.0, 0.25, 3e38],
    "core.sin": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 3.14159265, 100.0, -100.0, 1e-30, 1.5707964],
    "core.cos": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 3.14159265, 100.0, -100.0, 1e-30, 1.5707964],
    "core.tanh": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 20.0, -20.0, 1e-30, 0.5, 9.0],
    "core.neg": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 3e38, -3e38, 1e-45, 0.5, -2.5],
    "core.abs": [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 3e38, -3e38, -1e-45, 0.5, -2.5],
}


def unary_float_cases(spec: OpSpec, seed: int) -> list[Case]:
    return [
        ew_case(spec, "random_f32", seed + 1, "f32", "Random [3, 4].", [[3, 4]]),
        ew_case(spec, "random_rank4_f32", seed + 2, "f32", "Random [2, 1, 3, 2]: a rank-4 input with a size-1 dim.", [[2, 1, 3, 2]]),
        ew_case(spec, "random_f16", seed + 3, "f16", "Random [3, 4] in f16.", [[3, 4]]),
        ew_case(spec, "random_bf16", seed + 4, "bf16", "Random [3, 4] in bf16.", [[3, 4]]),
        ew_case(spec, "special_values_f32", seed + 5, "f32", "Signed zeros, infinities, NaN, tiny and large values, [12].",
                values=[UNARY_SPECIAL[spec.op]], notes=NAN_NOTES),
    ]


def int_unary_cases(spec: OpSpec, seed: int) -> list[Case]:
    cases = []
    for i, dt in enumerate(INTS):
        lo, hi = (I32_MIN, I32_MAX) if dt == "i32" else (I64_MIN, I64_MAX)
        cases.append(ew_case(spec, f"random_{dt}", seed + 1 + 2 * i, dt, f"Random [3, 4] in {dt}.", [[3, 4]]))
        cases.append(ew_case(spec, f"limits_{dt}", seed + 2 + 2 * i, dt, f"The {dt} limits: the minimum stays unchanged (wraps).",
                             values=[[lo, lo + 1, hi, -1, 0, 1, -7, 7]], notes=WRAP_NOTES))
    return cases


def minmax_cases(spec: OpSpec, seed: int) -> list[Case]:
    nan_a = [NAN, 1.0, NAN, -INF, INF, 2.0, -1.0, 3.0, NAN, 0.5, -INF, 7.0]
    nan_b = [1.0, NAN, NAN, 5.0, -INF, 2.0, -2.0, 3.0, -INF, -0.5, INF, NAN]
    cases = [
        ew_case(spec, "broadcast_f32", seed + 1, "f32", "[2, 3, 4] and [3, 1].", [[2, 3, 4], [3, 1]]),
        ew_case(spec, "ties_f32", seed + 2, "f32", "Small integers in f32 [4, 4]: many equal pairs.", [[4, 4], [4, 4]], make=ties_input),
        ew_case(spec, "broadcast_f16", seed + 3, "f16", "[2, 3, 4] and [1, 4] in f16.", [[2, 3, 4], [1, 4]]),
        ew_case(spec, "broadcast_bf16", seed + 4, "bf16", "[2, 3, 4] and [1, 4] in bf16.", [[2, 3, 4], [1, 4]]),
        ew_case(spec, "broadcast_i64", seed + 5, "i64", "[3, 4] and [4] in i64.", [[3, 4], [4]]),
        ew_case(spec, "broadcast_i32", seed + 6, "i32", "[3, 4] and [3, 1] in i32.", [[3, 4], [3, 1]]),
        ew_case(spec, "limits_i64", seed + 7, "i64", "The i64 limits.", values=[[I64_MIN, I64_MAX, 0, -1], [I64_MAX, I64_MIN, I64_MIN, I64_MAX]]),
        ew_case(spec, "nan_f32", seed + 8, "f32", "NaN in either or both operands propagates; infinities, [12].",
                values=[nan_a, nan_b], notes=NAN_NOTES),
        ew_case(spec, "nan_bf16", seed + 9, "bf16", "NaN propagation in bf16, [12].", values=[nan_a, nan_b], notes=NAN_NOTES),
    ]
    return cases


def clamp_case(name: str, seed: int, dtype: str, description: str, lo: Any, hi: Any, shape: list[int] | None = None,
               values: list[Any] | None = None, notes: str = "") -> Case:
    attrs = {k: v for k, v in (("min", lo), ("max", hi)) if v is not None}

    def as_number(v: Any) -> Any:
        return float(v) if isinstance(v, str) else v

    def fn(x: Any) -> Any:
        return torch.clamp(x, None if lo is None else as_number(lo), None if hi is None else as_number(hi))

    return ew_case(
        CLAMP, name, seed, dtype, description, [shape] if shape else None,
        values=[values] if values is not None else None,
        make=lambda rng, s, d: random_input(rng, s, d, (-3.0, 3.0)) if d in FLOATS else integers(rng, s, d, -50, 50),
        attributes=attrs, fn=fn, text="torch.clamp(x, min, max)",
        notes=" ".join(x for x in (notes, "Bounds are exact in the dtype under test.") if x),
    )


CLAMP = _unary("core.clamp", None, "torch.clamp(x, min, max)")


def clamp_cases(seed: int) -> list[Case]:
    nan_x = [NAN, -2.0, -1.0, 0.0, 1.0, 2.0, INF, -INF, NAN, 0.25, -0.75, 3.0]
    return [
        clamp_case("both_f32", seed + 1, "f32", "min -1.5, max 0.75 on [3, 4].", -1.5, 0.75, [3, 4]),
        clamp_case("min_only_f32", seed + 2, "f32", "min only (0.0) on [2, 3, 2].", 0.0, None, [2, 3, 2]),
        clamp_case("max_only_f32", seed + 3, "f32", "max only (-0.5) on [12].", None, -0.5, [12]),
        clamp_case("equal_bounds_f32", seed + 4, "f32", "min = max = 0.5 on [6].", 0.5, 0.5, [6]),
        clamp_case("both_f16", seed + 5, "f16", "min -1.5, max 0.75 in f16.", -1.5, 0.75, [3, 4]),
        clamp_case("both_bf16", seed + 6, "bf16", "min -1.5, max 0.75 in bf16.", -1.5, 0.75, [3, 4]),
        clamp_case("both_i64", seed + 7, "i64", "min -10, max 20 in i64.", -10, 20, [3, 4]),
        clamp_case("min_only_i32", seed + 8, "i32", "min only (-5) in i32.", -5, None, [3, 4]),
        clamp_case("max_only_i32", seed + 9, "i32", "max only (7) in i32.", None, 7, [3, 4]),
        clamp_case("limits_i64", seed + 10, "i64", "Bounds at the i64 limits.", I64_MIN, I64_MAX - 1, values=[I64_MIN, I64_MAX, 0, -1]),
        clamp_case("nan_f32", seed + 11, "f32", "NaN elements stay NaN; infinities are clamped.", -1.0, 1.0, values=nan_x, notes=NAN_NOTES),
        clamp_case("nan_bound_f32", seed + 12, "f32", "A NaN bound makes every element NaN (PyTorch).", "nan", 1.0, values=nan_x, notes=NAN_NOTES),
    ]


def compare_cases(spec: OpSpec, seed: int) -> list[Case]:
    nan_a = [NAN, 1.0, NAN, INF, -INF, INF, 0.0, -0.0, 2.0, -INF, NAN, 1e-45]
    nan_b = [1.0, NAN, NAN, INF, INF, -INF, -0.0, 0.0, 2.0, -INF, INF, 0.0]
    a64, b64 = [I64_MIN, I64_MAX, 2**53 + 1, -(2**40), 7, 0], [I64_MAX, I64_MAX, 2**53, -(2**40), -7, 0]
    cases = [
        ew_case(spec, "ties_f32", seed + 1, "f32", "Small integers in f32, [2, 3, 4] and [3, 4]: many equal pairs.", [[2, 3, 4], [3, 4]], make=ties_input),
        ew_case(spec, "broadcast_f32", seed + 2, "f32", "Random [2, 1, 4] and [3, 1].", [[2, 1, 4], [3, 1]]),
        ew_case(spec, "ties_f16", seed + 3, "f16", "Small integers in f16, [3, 4] and [4].", [[3, 4], [4]], make=ties_input),
        ew_case(spec, "ties_bf16", seed + 4, "bf16", "Small integers in bf16, [3, 4] and [3, 1].", [[3, 4], [3, 1]], make=ties_input),
        ew_case(spec, "ties_i32", seed + 5, "i32", "Small integers in i32, [3, 4] and [1, 4].", [[3, 4], [1, 4]], make=ties_input),
        ew_case(spec, "limits_i64", seed + 6, "i64", "Large i64 values (beyond 2^53) and the limits, [6].", values=[a64, b64]),
        ew_case(spec, "nan_f32", seed + 7, "f32", "NaN, infinities and signed zeros, [12].", values=[nan_a, nan_b],
                notes="NaN compares false, except in ne (true); -0.0 equals 0.0."),
    ]
    if spec.op in ("core.eq", "core.ne"):
        cases.append(ew_case(spec, "broadcast_bool", seed + 8, "bool", "bool [2, 3, 4] and [3, 1].", [[2, 3, 4], [3, 1]]))
    return cases


def logic_cases(spec: OpSpec, seed: int) -> list[Case]:
    if spec.op == "core.not":
        return [
            ew_case(spec, "random_bool", seed + 1, "bool", "Random [3, 4].", [[3, 4]]),
            ew_case(spec, "rank5_bool", seed + 2, "bool", "Random [2, 1, 2, 3, 2].", [[2, 1, 2, 3, 2]]),
        ]
    return [
        ew_case(spec, "same_shape_bool", seed + 1, "bool", "Random [3, 4].", [[3, 4], [3, 4]]),
        ew_case(spec, "broadcast_bool", seed + 2, "bool", "[2, 1, 4] and [3, 1].", [[2, 1, 4], [3, 1]]),
        ew_case(spec, "truth_table_bool", seed + 3, "bool", "Every input pair, [4].",
                values=[[False, False, True, True], [False, True, False, True]]),
    ]


def where_cases(seed: int) -> list[Case]:
    def case(name: str, s: int, dt: str, shapes: list[list[int]], description: str) -> Case:
        return ew_case(WHERE, name, s, dt, description, shapes)

    return [
        case("same_shape_f32", seed + 1, "f32", [[3, 4], [3, 4], [3, 4]], "cond, a and b all [3, 4]."),
        case("broadcast3_f32", seed + 2, "f32", [[2, 1, 4], [3, 1], [1]], "cond [2, 1, 4], a [3, 1], b [1]: three-way broadcast to [2, 3, 4]."),
        case("broadcast_cond_f32", seed + 3, "f32", [[4], [2, 3, 4], [2, 3, 4]], "cond [4] against a, b [2, 3, 4]."),
        case("broadcast_f16", seed + 4, "f16", [[3, 1], [1, 4], [3, 4]], "f16, cond [3, 1], a [1, 4], b [3, 4]."),
        case("broadcast_bf16", seed + 5, "bf16", [[3, 4], [4], [3, 1]], "bf16, cond [3, 4], a [4], b [3, 1]."),
        case("broadcast_i64", seed + 6, "i64", [[2, 3], [2, 3], [1]], "i64, cond [2, 3], a [2, 3], b [1]."),
        case("broadcast_i32", seed + 7, "i32", [[1, 3], [2, 1], [2, 3]], "i32, cond [1, 3], a [2, 1], b [2, 3]."),
        case("broadcast_bool", seed + 8, "bool", [[3, 4], [1, 4], [3, 1]], "bool a and b, cond [3, 4], a [1, 4], b [3, 1]."),
        ew_case(WHERE, "nan_f32", seed + 9, "f32", "NaN and infinities are selected like any value, [6].",
                values=[[True, False, True, False, True, False], [NAN, 1.0, INF, 2.0, -INF, 3.0], [0.0, NAN, 1.0, -INF, 2.0, INF]],
                notes=NAN_NOTES),
    ]


ADD_SPECIAL = (
    [0.0, -0.0, -0.0, INF, -INF, NAN, 1.0, 1e38, 3e38, 1.5, -2.25, 0.1],
    [-0.0, 0.0, -0.0, -INF, -INF, 1.0, NAN, 1e38, 3e38, -1.5, 2.25, 0.2],
)
SUB_SPECIAL = (
    [0.0, -0.0, -0.0, INF, INF, NAN, 1.0, 3e38, -3e38, 1.5, -2.25, 0.3],
    [0.0, 0.0, -0.0, INF, -INF, 1.0, NAN, -3e38, 3e38, 1.5, 2.25, 0.1],
)
MUL_SPECIAL = (
    [0.0, -0.0, -0.0, INF, -INF, NAN, 1.0, 1e38, -1e-30, 0.0, -2.25, 3e38],
    [INF, 1.0, -0.0, -INF, 2.0, 0.0, NAN, 10.0, 1e-30, -1.0, 4.0, -3e38],
)
DIV_SPECIAL = (
    [1.0, -1.0, 1.0, 0.0, INF, 1.0, -1.0, NAN, 3e38, -0.0, 6.0, INF],
    [0.0, 0.0, -0.0, 0.0, INF, INF, INF, 1.0, 1e-10, 2.0, -3.0, -2.0],
)
POW_SPECIAL = (
    [1.0, 0.0, 0.0, -2.0, -2.0, NAN, -8.0, 2.0, INF, -INF, 4.0, -3.0],
    [NAN, 0.0, -1.0, 3.0, 0.5, 0.0, 1.0 / 3.0, 1000.0, -1.0, 3.0, 0.5, 2.0],
)
# Negative finite bases: even and odd integer exponents (real results) and non-integer exponents
# (NaN), each sign of exponent.
POW_NEGATIVE_MIXED = (
    [-2.0, -2.0, -2.0, -0.5, -3.0, -1.5, -4.0, -1.0, -2.5, -8.0, -0.25, -1.0],
    [2.0, 3.0, 0.5, -2.0, -1.0, 2.5, 0.25, -0.5, 4.0, 1.0 / 3.0, -3.0, 1000.0],
)

# Zero and infinite bases and exponents (C99 Annex F pow, which torch.pow follows).
POW_SPECIAL_LIMITS = (
    [0.0, -0.0, -0.0, -0.0, 0.0, INF, -INF, -INF, -INF, -1.0, 0.5, 0.5, 2.0, 2.0, -1.0, 1.0],
    [2.0, 3.0, -1.0, -2.0, 0.5, 0.0, -1.0, 2.0, -3.0, INF, INF, -INF, INF, -INF, -INF, INF],
)


def elementwise_cases() -> list[Case]:
    cases = binary_arith_cases(ADD, 100, ADD_SPECIAL)
    cases += binary_arith_cases(SUB, 500, SUB_SPECIAL)
    cases += binary_arith_cases(MUL, 520, MUL_SPECIAL)
    cases += [
        ew_case(DIV, "broadcast_f32", 541, "f32", "[2, 3, 4] and [4].", [[2, 3, 4], [4]]),
        ew_case(DIV, "same_shape_f32", 542, "f32", "Same shape [3, 4], signed denominators.", [[3, 4], [3, 4]],
                make=lambda rng, s, d: uniform(rng, s, d, -4.0, 4.0)),
        ew_case(DIV, "broadcast_f16", 543, "f16", "[3, 4] and [3, 1] in f16.", [[3, 4], [3, 1]]),
        ew_case(DIV, "broadcast_bf16", 544, "bf16", "[3, 4] and [1, 4] in bf16.", [[3, 4], [1, 4]]),
        ew_case(DIV, "special_values_f32", 545, "f32", "Division by ±0, infinities, NaN and overflow, [12].",
                values=list(DIV_SPECIAL), notes=NAN_NOTES),
        ew_case(POW, "broadcast_f32", 551, "f32", "Positive bases [2, 3, 4], exponents [4].", [[2, 3, 4], [4]]),
        ew_case(
            POW, "negative_base_f32", 552, "f32", "Negative bases with integer exponents give real results, [12].",
            values=[[-2.0, -2.0, -2.0, -0.5, -3.0, -1.0, -1.0, -1.5, -4.0, -10.0, -0.25, -7.0],
                    [2.0, 3.0, -1.0, -2.0, 0.0, 1001.0, 1000.0, 4.0, 5.0, 3.0, -3.0, 1.0]]),
        ew_case(POW, "negative_base_mixed_f32", 557, "f32",
                "Negative bases with even, odd and non-integer exponents, [12]: non-integer exponents give NaN.",
                values=list(POW_NEGATIVE_MIXED), notes=NAN_NOTES),
        ew_case(POW, "negative_base_mixed_f16", 558, "f16",
                "Negative bases with even, odd and non-integer exponents in f16, [12].",
                values=list(POW_NEGATIVE_MIXED), notes=NAN_NOTES),
        ew_case(POW, "broadcast_f16", 553, "f16", "Positive bases [3, 4], exponents [3, 1] in f16.", [[3, 4], [3, 1]]),
        ew_case(POW, "broadcast_bf16", 554, "bf16", "Positive bases [3, 4], exponents [4] in bf16.", [[3, 4], [4]]),
        ew_case(POW, "special_values_f32", 555, "f32", "0^0, 1^NaN, NaN^0, 0^-1, negative bases with fractional exponents, overflow, [12].",
                values=list(POW_SPECIAL), notes=NAN_NOTES + " pow(1, NaN) = pow(NaN, 0) = 1, as in PyTorch and C."),
        ew_case(POW, "zero_inf_limits_f32", 556, "f32", "Signed zero and infinite bases, infinite exponents, [16].",
                values=list(POW_SPECIAL_LIMITS),
                notes=NAN_NOTES + " The sign of a zero or infinite result follows the base for odd integer exponents."),
    ]
    cases += minmax_cases(MAXIMUM, 560)
    cases += minmax_cases(MINIMUM, 580)
    for spec, seed in ((NEG, 600), (ABS, 620)):
        cases += unary_float_cases(spec, seed)
        cases += int_unary_cases(spec, seed + 10)
    for spec, seed in ((EXP, 640), (LOG, 650), (SQRT, 660), (RSQRT, 670), (SIN, 680), (COS, 690), (TANH, 700)):
        cases += unary_float_cases(spec, seed)
    cases += clamp_cases(710)
    cases += where_cases(730)
    for spec, seed in ((EQ, 750), (NE, 760), (LT, 770), (LE, 780), (GT, 790), (GE, 800)):
        cases += compare_cases(spec, seed)
    for spec, seed in ((AND, 810), (OR, 820), (NOT, 830)):
        cases += logic_cases(spec, seed)
    return cases


def build_cases() -> list[Case]:
    cases = elementwise_cases() + [
        matmul_case("2d_f32", 201, "f32", [3, 5], [5, 4], "[3, 5] @ [5, 4]."),
        matmul_case("2d_k64_f32", 202, "f32", [8, 64], [64, 16], "[8, 64] @ [64, 16]: a longer reduction."),
        matmul_case("batched_3d_f32", 203, "f32", [2, 3, 5], [2, 5, 4], "[2, 3, 5] @ [2, 5, 4]."),
        matmul_case("broadcast_batch_4d_f32", 204, "f32", [2, 1, 3, 5], [1, 3, 5, 4], "[2, 1, 3, 5] @ [1, 3, 5, 4] -> [2, 3, 3, 4]: broadcast batch dims."),
        matmul_case("2d_f16", 205, "f16", [3, 5], [5, 4], "[3, 5] @ [5, 4] in f16."),
        matmul_case("batched_3d_f16", 206, "f16", [2, 3, 5], [2, 5, 4], "[2, 3, 5] @ [2, 5, 4] in f16."),
        matmul_case("2d_bf16", 207, "bf16", [3, 5], [5, 4], "[3, 5] @ [5, 4] in bf16."),
        matmul_case("broadcast_batch_4d_bf16", 208, "bf16", [2, 1, 3, 5], [1, 3, 5, 4], "[2, 1, 3, 5] @ [1, 3, 5, 4] in bf16."),
        reshape_case("3d_to_2d_f32", 301, "f32", [2, 3, 4], [4, 6]),
        reshape_case("1d_to_6d_f32", 302, "f32", [24], [2, 1, 3, 1, 2, 2]),
        reshape_case("2d_to_3d_bf16", 303, "bf16", [4, 6], [2, 2, 6]),
        reshape_case("1d_to_2d_i64", 304, "i64", [12], [3, 4]),
        reshape_case("2d_to_3d_bool", 305, "bool", [2, 6], [3, 2, 2]),
    ]
    seed = 400
    for src in DTYPE_NAMES:
        for dst in DTYPE_NAMES:
            seed += 1
            cases.append(cast_case(src, dst, seed))
    return cases


# ---------------------------------------------------------------------------------------------
# Writing.

DTYPE_OF = {}  # filled lazily: torch dtype -> Polypous dtype name


def dtype_name(t: Any) -> str:
    if not DTYPE_OF:
        DTYPE_OF.update({torch_dtype(n): n for n in DTYPE_NAMES})
    return DTYPE_OF[t.dtype]


def pinned_versions() -> dict[str, str]:
    pins = {}
    with open(REQUIREMENTS, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line.strip())
            if m:
                pins[m.group(1).lower()] = m.group(2)
    return pins


def installed_versions() -> dict[str, str]:
    return {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "numpy": np.__version__,
    }


def version_problems() -> list[str]:
    have = installed_versions()
    problems = []
    if have["python"] != PYTHON:
        problems.append(f"Python {have['python']} is running, the fixtures are generated with Python {PYTHON}")
    return problems + [f"{name} {have.get(name)} is installed, requirements.txt pins {want}" for name, want in pinned_versions().items() if have.get(name) != want]


def write_case(case: Case, out_dir: str, versions: dict[str, str]) -> None:
    rng = np.random.Generator(np.random.PCG64(case.seed))
    inputs = [(n, t.contiguous().clone()) for n, t in case.inputs(rng)]
    outputs = [(n, t.contiguous().clone()) for n, t in case.reference([t for _, t in inputs])]
    check_native(case, [t for _, t in inputs], outputs)
    names = [n for n, _ in inputs + outputs]
    if len(set(names)) != len(names):
        raise SystemExit(f"{case.op}/{case.name}: tensor names are not unique: {names}")
    # No metadata: nothing machine- or time-dependent ends up in the file.
    blob = safetensors.torch.save(dict(inputs + outputs), metadata=None)
    sidecar = {
        "schema_version": fixturelib.SCHEMA_VERSION,
        "op": case.op,
        "case": case.name,
        "description": case.description,
        "attributes": case.attributes,
        "inputs": [{"name": n, "dtype": dtype_name(t), "shape": list(t.shape)} for n, t in inputs],
        "outputs": [{"name": n, "dtype": dtype_name(t), "shape": list(t.shape)} for n, t in outputs],
        "tolerance": case.tolerance,
        "reference": case.reference_text,
        "notes": case.notes,
        "seed": case.seed,
        "generator": GENERATOR,
        "versions": versions,
    }
    op_dir = os.path.join(out_dir, case.op)
    os.makedirs(op_dir, exist_ok=True)
    with open(os.path.join(op_dir, case.name + ".safetensors"), "wb") as f:
        f.write(blob)
    with open(os.path.join(op_dir, case.name + ".json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(fixturelib.canonical_json(sidecar))


def remove_stale(out_dir: str) -> None:
    """Deletes earlier fixture files (only `.json` and `.safetensors`), so removed cases go away."""
    for rel in fixturelib.list_files(out_dir):
        if rel.endswith((".json", ".safetensors")):
            os.remove(os.path.join(out_dir, rel))
    for dirpath, _, _ in sorted(os.walk(out_dir), reverse=True):
        if dirpath != out_dir and not os.listdir(dirpath):
            os.rmdir(dirpath)


def generate(out_dir: str) -> int:
    cases = build_cases()
    keys = [(c.op, c.name) for c in cases]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate case names")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    versions = installed_versions()
    for case in cases:
        write_case(case, out_dir, versions)
    return len(cases)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: fixtures/small/ops)")
    parser.add_argument("--check", action="store_true", help="regenerate into a temporary directory and compare")
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="generate with versions other than the pinned ones (never for committed fixtures)",
    )
    args = parser.parse_args(argv)
    if IMPORT_ERROR is not None:
        print(f"error: {IMPORT_ERROR}. Install tools/reference/requirements.txt (docs/dev-environment.md).", file=sys.stderr)
        return 2
    mismatch = version_problems()
    if mismatch and not args.allow_version_mismatch:
        for line in mismatch:
            print(f"error: {line}", file=sys.stderr)
        return 2
    if not args.check:
        remove_stale(args.out)
        n = generate(args.out)
        print(f"wrote {n} cases to {args.out}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        n = generate(tmp)
        problems = fixturelib.compare_trees(args.out, tmp)
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s); run python {GENERATOR} and commit the result", file=sys.stderr)
        return 1
    print(f"ok: {n} cases match {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
