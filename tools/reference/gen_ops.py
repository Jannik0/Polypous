#!/usr/bin/env python3
"""Generates the op reference fixtures in fixtures/small/ops/ from PyTorch.

    python tools/reference/gen_ops.py            # regenerate fixtures/small/ops/
    python tools/reference/gen_ops.py --check    # fail if the committed fixtures differ

Layout and sidecar schema: docs/testing-strategy.md §3. Every case is declared in `CASES` below,
with its own fixed seed. Input data comes from NumPy's PCG64 (`Generator.random` and
`Generator.integers` are integer arithmetic, so they give the same values on every CPU and OS).
Outputs are computed with PyTorch on the CPU; matmul is computed in f64 and rounded once to the
dtype under test, so the bytes don't depend on the BLAS kernel the CPU picks (see `ref_matmul`).

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
# f16/bf16: §2 gives rtol 1e-2 and no atol. The current cases have no outputs near zero.
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


def ref_add(xs: list[Any]) -> list[tuple[str, Any]]:
    return [("out", torch.add(xs[0], xs[1]))]


def ref_matmul(xs: list[Any]) -> list[tuple[str, Any]]:
    # f64, rounded once to the dtype under test: bit-reproducible on every CPU. A backend's f32
    # (or f16/bf16) accumulation differs by rounding only, which the tolerance covers.
    a, b = xs
    return [("out", torch.matmul(a.double(), b.double()).to(a.dtype))]


def check_matmul_native(case: Case, xs: list[Any], outs: list[tuple[str, Any]]) -> None:
    """PyTorch's own matmul in the dtype under test must meet the case's tolerance."""
    native = torch.matmul(xs[0], xs[1]).double()
    expected = outs[0][1].double()
    tol = case.tolerance
    if not torch.allclose(native, expected, atol=tol["atol"], rtol=tol["rtol"]):
        worst = (native - expected).abs().max().item()
        raise SystemExit(f"{case.op}/{case.name}: torch.matmul misses the tolerance (max abs err {worst})")


MATMUL_NOTES = (
    "out is torch.matmul in f64, rounded once to the dtype under test, so the bytes don't depend on "
    "the CPU's BLAS kernel; torch.matmul in the dtype itself meets the tolerance (checked by the generator)."
)


def add_case(name: str, seed: int, dtype: str, lhs: list[int], rhs: list[int], description: str) -> Case:
    def inputs(rng: Any) -> list[tuple[str, Any]]:
        if dtype == "i64":
            return [("lhs", integers(rng, lhs, dtype, -(2**40), 2**40)), ("rhs", integers(rng, rhs, dtype, -(2**40), 2**40))]
        if dtype == "i32":
            return [("lhs", integers(rng, lhs, dtype, -1000, 1000)), ("rhs", integers(rng, rhs, dtype, -1000, 1000))]
        return [("lhs", uniform(rng, lhs, dtype)), ("rhs", uniform(rng, rhs, dtype))]

    return Case(
        op="core.add",
        name=name,
        description=description,
        seed=seed,
        attributes={},
        inputs=inputs,
        reference=ref_add,
        reference_text="torch.add(lhs, rhs)",
        tolerance=elementwise_tol(dtype),
        notes="Integer inputs are small enough that the sum doesn't overflow." if dtype in ("i64", "i32") else "",
    )


def add_special_case() -> Case:
    inf, nan = float("inf"), float("nan")
    lhs = [0.0, -0.0, -0.0, inf, -inf, nan, 1.0, 1e38, 3e38, 1.5, -2.25, 0.1]
    rhs = [-0.0, 0.0, -0.0, -inf, -inf, 1.0, nan, 1e38, 3e38, -1.5, 2.25, 0.2]
    return Case(
        op="core.add",
        name="special_values_f32",
        description="Signed zeros, infinities, NaN and overflow to inf, same shape [3, 4].",
        seed=0,
        attributes={},
        inputs=lambda rng: [("lhs", from_list(lhs, "f32", [3, 4])), ("rhs", from_list(rhs, "f32", [3, 4]))],
        reference=ref_add,
        reference_text="torch.add(lhs, rhs)",
        tolerance=F32_ELEMENTWISE,
        notes=(
            "Expected NaN must be NaN and expected ±inf must be the same inf; -0.0 + -0.0 is -0.0, "
            "-0.0 + 0.0 is 0.0 (the tolerance can't see the sign of zero, the bytes keep it)."
        ),
    )


def matmul_case(name: str, seed: int, dtype: str, lhs: list[int], rhs: list[int], description: str) -> Case:
    return Case(
        op="core.matmul",
        name=name,
        description=description,
        seed=seed,
        attributes={},
        inputs=lambda rng: [("lhs", uniform(rng, lhs, dtype)), ("rhs", uniform(rng, rhs, dtype))],
        reference=ref_matmul,
        reference_text="torch.matmul(lhs.double(), rhs.double()).to(lhs.dtype)",
        tolerance=matmul_tol(dtype),
        notes=MATMUL_NOTES,
    )


def reshape_case(name: str, seed: int, dtype: str, shape: list[int], new_shape: list[int]) -> Case:
    def inputs(rng: Any) -> list[tuple[str, Any]]:
        if dtype == "bool":
            return [("input", torch.from_numpy(rng.integers(0, 2, size=shape, dtype=np.int64)).to(torch.bool))]
        if dtype in ("i64", "i32"):
            return [("input", integers(rng, shape, dtype, -1000, 1000))]
        return [("input", uniform(rng, shape, dtype))]

    return Case(
        op="core.reshape",
        name=name,
        description=f"{dtype} {shape} -> {new_shape}, row-major order kept.",
        seed=seed,
        attributes={"shape": new_shape},
        inputs=inputs,
        reference=lambda xs: [("out", torch.reshape(xs[0], new_shape))],
        reference_text="torch.reshape(input, shape)",
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
        inputs=lambda rng: [("input", x)],
        reference=lambda xs: [("out", xs[0].to(torch_dtype(dst)))],
        reference_text=f"input.to({dst})",
        tolerance=EXACT,
        notes=note,
    )


def build_cases() -> list[Case]:
    cases = [
        add_case("same_shape_f32", 101, "f32", [3, 4], [3, 4], "Same shape [3, 4]."),
        add_case("broadcast_row_f32", 102, "f32", [2, 3, 4], [1, 1, 4], "[2, 3, 4] + [1, 1, 4]."),
        add_case("broadcast_both_f32", 103, "f32", [2, 1, 4], [1, 3, 1], "[2, 1, 4] + [1, 3, 1]: both sides broadcast to [2, 3, 4]."),
        add_case("broadcast_rank_f32", 104, "f32", [2, 3, 4], [4], "[2, 3, 4] + [4]: different ranks, PyTorch prepends size-1 dims."),
        add_case("empty_dim_f32", 105, "f32", [0, 4], [1, 4], "[0, 4] + [1, 4]: an empty dimension."),
        add_case("broadcast_f16", 106, "f16", [2, 3, 4], [2, 1, 4], "[2, 3, 4] + [2, 1, 4] in f16."),
        add_case("broadcast_bf16", 107, "bf16", [2, 3, 4], [2, 1, 4], "[2, 3, 4] + [2, 1, 4] in bf16."),
        add_case("broadcast_i64", 108, "i64", [3, 4], [1, 4], "[3, 4] + [1, 4] in i64."),
        add_case("broadcast_i32", 109, "i32", [3, 4], [3, 1], "[3, 4] + [3, 1] in i32."),
        add_special_case(),
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
    if case.op == "core.matmul":
        check_matmul_native(case, [t for _, t in inputs], outputs)
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
