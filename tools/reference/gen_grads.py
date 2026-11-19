#!/usr/bin/env python3
"""Generates the gradient fixtures in fixtures/small/grads/ from PyTorch autograd.

    python tools/reference/gen_grads.py            # regenerate fixtures/small/grads/
    python tools/reference/gen_grads.py --check    # fail if the committed fixtures differ

Layout and sidecar schema (op sidecar schema version 2): docs/testing-strategy.md §3.6. Gradient
conventions: docs/specs/graph-ir.md §3.1.11. Every case is declared in `build_cases()` with its own
fixed seed. Inputs, params, dropout masks and upstream gradients come from NumPy's PCG64, in f32.

For each case the generator
- runs the op's PyTorch call (graph-ir.md §3, the PyTorch column) and `torch.autograd.grad` in f64
  on the f32 tensors, and rounds the outputs, the running statistics and the gradients once to f32:
  those are the expected values;
- runs the same in f32 and refuses to write the case unless PyTorch's own f32 result meets
  `tolerance` (outputs, running statistics) and `gradient_tolerance` (gradients);
- for the tie, kink and special-value cases, checks the f64 gradients against the formula of
  graph-ir.md §3.1.11's table (`spec=`), and stops if PyTorch disagrees with the spec;
- refuses non-finite inputs, params, masks and upstream gradients (their gradients are
  unspecified), and subnormal values in any tensor.

The forward fixtures (gen_ops.py, fixtures/small/ops/) are not touched; this script only borrows
gen_ops.py's input helpers and version checks. Needs the pinned versions from requirements.txt.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixturelib  # noqa: E402
import gen_ops  # noqa: E402  (input helpers, views, versions; its own output is unaffected)
from gen_ops import IMPORT_ERROR, F, np, safetensors, torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO, "fixtures", "small", "grads")
GENERATOR = "tools/reference/gen_grads.py"

# ---------------------------------------------------------------------------------------------
# Tolerances (docs/testing-strategy.md §2), written into every sidecar.

EXACT = gen_ops.EXACT
ELEMENTWISE = gen_ops.F32_ELEMENTWISE  # atol 1e-5, rtol 1e-4
REDUCTION = gen_ops.F32_MATMUL  # atol 1e-5, rtol 1e-3
# §2's gradient row, for every op (testing-strategy.md §2 "Gradient tolerance").
GRADIENT = {"exact": False, "atol": 1e-5, "rtol": 1e-3}
# Far negative inputs of softplus, mish, silu and sigmoid: tiny but normal results and gradients,
# which atol 1e-5 would accept as 0. These cases use atol 0 (rtol unchanged), a stricter tolerance,
# as the forward tail cases do (gen_ops.TAIL_TOL). PyTorch's f32 meets it (checked).
TAIL = gen_ops.TAIL_TOL
TAIL_GRADIENT = {"exact": False, "atol": 0.0, "rtol": 1e-3}

EXACT_OPS = {
    "core.reshape", "core.flatten", "core.permute", "core.transpose", "core.squeeze", "core.unsqueeze",
    "core.expand", "core.concat", "core.split", "core.slice", "core.pad", "core.gather", "core.index_select",
    "core.cast", "core.max", "core.min", "core.top_k", "nn.embedding", "nn.max_pool2d",
}
REDUCTION_OPS = {
    "core.sum", "core.mean", "core.cumsum", "core.matmul", "core.softmax", "core.log_softmax", "nn.linear",
    "nn.conv1d", "nn.conv2d", "nn.batch_norm", "nn.layer_norm", "nn.group_norm", "nn.rms_norm",
}


def op_tolerance(op: str) -> dict[str, Any]:
    """The forward tolerance of an op's f32 outputs: the class its forward fixtures use (§2)."""
    if op in EXACT_OPS:
        return EXACT
    return REDUCTION if op in REDUCTION_OPS else ELEMENTWISE


# ---------------------------------------------------------------------------------------------
# Tensor builders. Each takes the case's NumPy generator and the tensors built so far.

Builder = Callable[[Any, dict[str, Any]], Any]


def rnd(shape: list[int], low: float = -2.0, high: float = 2.0) -> Builder:
    """Uniform in [low, high), drawn in f64 and rounded once to f32."""
    return lambda rng, _: gen_ops.uniform(rng, shape, "f32", low, high)


def signed(shape: list[int], low: float, high: float) -> Builder:
    """Magnitude uniform in [low, high), random sign: values away from zero."""

    def build(rng: Any, _: dict[str, Any]) -> Any:
        mag = rng.random(size=shape) * (high - low) + low
        sign = np.where(rng.integers(0, 2, size=shape) == 0, -1.0, 1.0)
        return torch.from_numpy(mag * sign).float()

    return build


def vals(values: list[float], shape: list[int] | None = None) -> Builder:
    """Exact values (they are all exactly representable in f32 where a convention depends on them)."""
    return lambda rng, _: gen_ops.from_list(values, "f32", shape)


def choice(shape: list[int], values: list[float]) -> Builder:
    """Random picks from a few exact values: many ties."""
    return lambda rng, _: gen_ops.from_list([values[i] for i in rng.integers(0, len(values), size=shape).ravel()],
                                            "f32", shape)


def ints(values: list[int], shape: list[int] | None = None, dtype: str = "i64") -> Builder:
    return lambda rng, _: gen_ops.from_list(values, dtype, shape)


def bools(shape: list[int]) -> Builder:
    return lambda rng, _: torch.from_numpy(rng.integers(0, 2, size=shape, dtype=np.int64)).to(torch.bool)


def view(layout: str, shape: list[int]) -> Builder:
    """A non-contiguous PyTorch view (gen_ops.LAYOUTS), uniform in [-2, 2). The Rust harness feeds the
    same kind of view (crates/polypous-backend/tests/reference/common.rs, `feed_layout`)."""
    return lambda rng, _: gen_ops.layout_input(rng, shape, "f32", layout)


def copy_of(name: str) -> Builder:
    """The same values as an earlier tensor (one value used twice)."""
    return lambda rng, built: built[name].clone()


def keep_mask(shape: list[int], p: float) -> Builder:
    """A dropout keep mask (1 kept, 0 dropped), each element kept with probability 1 - p."""
    return lambda rng, _: torch.from_numpy((rng.random(size=shape) >= p).astype(np.float64)).float()


def upstream_values(rng: Any, shape: list[int]) -> Any:
    """The default upstream gradient: magnitude uniform in [0.25, 1), random sign. Never zero (a
    gradient `g / 0`-type convention such as `sqrt'(0)` then has a defined sign) and never tiny (no
    subnormal products); scale ≤ 1 as testing-strategy.md §2 assumes."""
    return signed(shape, 0.25, 1.0)(rng, {})


# ---------------------------------------------------------------------------------------------
# Cases.


@dataclasses.dataclass(frozen=True)
class GradCase:
    op: str  # IR op name, docs/specs/graph-ir.md §3
    name: str  # [a-z0-9_], ends in _f32
    description: str
    seed: int
    # The op's inputs in order, then its present params in catalog order, then `mask` (dropout in
    # training mode). Names as in graph-ir.md §3.
    tensors: tuple[tuple[str, Builder], ...]
    # The forward call on a dict of the tensors: one tensor (port `out`) or a dict port -> tensor.
    fn: Callable[[dict[str, Any]], Any]
    call: str  # the PyTorch expression of the forward call, for `reference`
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    params: tuple[str, ...] | None = None  # as gen_ops.Case.params
    mode: str = "inference"
    tolerance: dict[str, Any] | None = None  # default: op_tolerance(op)
    gradient_tolerance: dict[str, Any] = dataclasses.field(default_factory=lambda: GRADIENT)
    notes: str = ""
    # Explicit upstream gradients per port (default: `upstream_values`).
    upstream: dict[str, Builder] | None = None
    # graph-ir.md §3.1.11's formula for this case: from the f64 tensors and upstream gradients
    # (dicts by name and by port) to the expected f64 gradients (by gradient name) and, for
    # batch_norm in training mode, running statistics. Checked against PyTorch's f64 result.
    spec: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None


RUNNING_STATS = ("running_mean", "running_var")
NOTES = (
    "Expected values are PyTorch's (autograd) in f64 on the f32 tensors, rounded once to f32; PyTorch's own f32 "
    "result meets tolerance and gradient_tolerance (checked by the generator)."
)
SAME_VALUE_NOTES = (
    "{b} holds the same values as {a}: a test may feed one value to both, whose gradient is then "
    "grad_{a} + grad_{b} (gradients add up over every use, graph-ir.md §3.1.11)."
)
SPEC_NOTES = "The gradients follow graph-ir.md §3.1.11's convention here (checked by the generator): "


def port_output_name(port: str) -> str:
    """The output tensor name of a port (§3.2): `out`, `out0`… for `core.split`, `values`."""
    return "out" + port if port.isdigit() else port


def case(op: str, name: str, seed: int, description: str, tensors: list[tuple[str, Builder]],
         fn: Callable[[dict[str, Any]], Any], call: str, **kw: Any) -> GradCase:
    return GradCase(op=op, name=name, description=description, seed=seed, tensors=tuple(tensors), fn=fn,
                    call=call, **kw)


def with_spec_note(kw: dict[str, Any], text: str) -> dict[str, Any]:
    kw = dict(kw)
    kw["notes"] = " ".join(x for x in (kw.get("notes", ""), SPEC_NOTES + text) if x)
    return kw


# --- Spec formulas (graph-ir.md §3.1.11), in f64 ------------------------------------------------


def sum_to(t: Any, shape: list[int]) -> Any:
    """Sums a broadcast gradient back to an input's shape (§3.1.3)."""
    while t.dim() > len(shape):
        t = t.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and t.shape[i] != 1:
            t = t.sum(i, keepdim=True)
    return t


def spec_extremum(dims: list[int] | None, largest: bool) -> Callable[..., dict[str, Any]]:
    """`core.max`/`core.min`: the k elements of a slice equal to the result each get g / k."""

    def f(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
        x = d["x"]
        all_dims = list(range(x.dim())) if dims is None else [a % x.dim() for a in dims]
        r = (torch.amax if largest else torch.amin)(x, dim=all_dims, keepdim=True)
        mask = (x == r).double()
        k = mask.sum(dim=all_dims, keepdim=True)
        return {"grad_x": mask * up["out"].reshape(r.shape) / k}

    return f


def spec_maximum(largest: bool) -> Callable[..., dict[str, Any]]:
    """`core.maximum`/`core.minimum`: g to the larger (smaller) input, g / 2 to each at a = b."""

    def f(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
        a, b, g = d["a"], d["b"], up["out"]
        wins_a = a > b if largest else a < b
        wins_b = b > a if largest else b < a
        tie = a == b
        ga = torch.where(wins_a, g, torch.where(tie, g / 2, torch.zeros_like(g)))
        gb = torch.where(wins_b, g, torch.where(tie, g / 2, torch.zeros_like(g)))
        return {"grad_a": sum_to(ga, list(a.shape)), "grad_b": sum_to(gb, list(b.shape))}

    return f


def spec_clamp(lo: float | None, hi: float | None) -> Callable[..., dict[str, Any]]:
    """`core.clamp`: g only where min < x < max (one bound: x > min or x < max), 0 at the bounds."""

    def f(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
        x = d["x"]
        inside = torch.ones_like(x, dtype=torch.bool)
        if lo is not None:
            inside &= x > lo
        if hi is not None:
            inside &= x < hi
        return {"grad_x": torch.where(inside, up["out"], torch.zeros_like(x))}

    return f


def spec_where(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
    """`core.where`: g to a where cond is true, to b elsewhere, 0 to the other."""
    c, g = d["cond"], up["out"]
    z = torch.zeros_like(g)
    return {"grad_a": sum_to(torch.where(c, g, z), list(d["a"].shape)),
            "grad_b": sum_to(torch.where(c, z, g), list(d["b"].shape))}


def spec_unary(fn: Callable[[Any, Any], Any]) -> Callable[..., dict[str, Any]]:
    return lambda d, up: {"grad_x": fn(d["x"], up["out"])}


def spec_abs(x: Any, g: Any) -> Any:
    """`core.abs`: g · sign(x), 0 at x = 0 and -0."""
    return torch.where(x > 0, g, torch.where(x < 0, -g, torch.zeros_like(g)))


def spec_relu(x: Any, g: Any) -> Any:
    """`nn.relu`: g where x > 0, 0 at x = 0 and -0."""
    return torch.where(x > 0, g, torch.zeros_like(g))


def spec_leaky_relu(slope: float) -> Callable[[Any, Any], Any]:
    """`nn.leaky_relu`: g where x > 0, negative_slope · g elsewhere (also at 0 and -0)."""
    return lambda x, g: torch.where(x > 0, g, slope * g)


def spec_softplus(beta: float, threshold: float) -> Callable[[Any, Any], Any]:
    """`nn.softplus`: g · σ(βx), so g / 2 at βx = 0; g above the threshold (βx > threshold)."""
    return lambda x, g: torch.where(beta * x > threshold, g, g * torch.sigmoid(beta * x))


def spec_sqrt(x: Any, g: Any) -> Any:
    """`core.sqrt`: g / (2 · sqrt(x)), so +inf at x = 0 for g > 0."""
    return g / (2 * torch.sqrt(x))


def spec_pow(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
    """`core.pow`: ∂/∂a = 0 where b = 0, ∂/∂b = 0 where a = 0 and b ≥ 0, and elsewhere
    ∂/∂a = g · b · a^(b-1), ∂/∂b = g · a^b · ln(a)."""
    a, b, g = d["a"], d["b"], up["out"]
    a_full, b_full = torch.broadcast_tensors(a, b)
    da = torch.where(b_full == 0, torch.zeros_like(g), g * b_full * torch.pow(a_full, b_full - 1))
    db = torch.where((a_full == 0) & (b_full >= 0), torch.zeros_like(g), g * torch.pow(a_full, b_full) * torch.log(a_full))
    return {"grad_a": sum_to(da, list(a.shape)), "grad_b": sum_to(db, list(b.shape))}


def spec_scatter(index_name: str, dim: int, mode: str) -> Callable[..., dict[str, Any]]:
    """`core.gather`, `core.index_select`, `nn.embedding`: repeated indices add up."""

    def f(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
        g, idx = up["out"].numpy(), d[index_name].numpy().astype(np.int64)
        if mode == "embedding":
            w = d["weight"]
            out = np.zeros(tuple(w.shape))
            np.add.at(out, idx.reshape(-1), g.reshape(-1, w.shape[1]))
            return {"grad_weight": torch.from_numpy(out)}
        x = d["x"]
        out = np.zeros(tuple(x.shape))
        dd = dim % x.dim()
        for pos in np.ndindex(*g.shape):
            src = list(pos)
            src[dd] = idx[pos] if mode == "gather" else idx[pos[dd]]
            out[tuple(src)] += g[pos]
        return {"grad_x": torch.from_numpy(out)}

    return f


def spec_top_k(k: int, dim: int, largest: bool) -> Callable[..., dict[str, Any]]:
    """`core.top_k`: g to the elements at the returned indices; among equal values those are the
    lowest indices (graph-ir.md §3.1.8), found here by a stable argsort."""

    def f(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
        x = d["x"].numpy()
        order = np.argsort(-x if largest else x, axis=dim, kind="stable")
        idx = np.take(order, np.arange(k), axis=dim)
        out = np.zeros(x.shape)
        np.put_along_axis(out, idx, up["values"].numpy(), axis=dim)
        return {"grad_x": torch.from_numpy(out)}

    return f


def spec_max_pool2d(kernel: list[int], stride: list[int], padding: list[int], dilation: list[int]) -> Callable[..., dict[str, Any]]:
    """`nn.max_pool2d`: all of g to the first maximum of each window in row-major order (kh, then kw);
    overlapping windows add up. Refuses a window entirely in the padding (graph-ir.md §3.1.11: no
    fixture, PyTorch 2.15.0's backward corrupts its heap there)."""

    def f(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
        x, g = d["x"].numpy(), up["out"].numpy()
        h, w = x.shape[2:]
        out = np.zeros(x.shape)
        for n, c, i, j in np.ndindex(*g.shape):
            best = None
            for a in range(kernel[0]):
                for b in range(kernel[1]):
                    r = i * stride[0] - padding[0] + a * dilation[0]
                    q = j * stride[1] - padding[1] + b * dilation[1]
                    if 0 <= r < h and 0 <= q < w and (best is None or x[n, c, r, q] > x[n, c, best[0], best[1]]):
                        best = (r, q)
            if best is None:
                raise SystemExit(f"max_pool2d: window ({i}, {j}) lies entirely in the padding; no fixture for that")
            out[n, c, best[0], best[1]] += g[n, c, i, j]
        return {"grad_x": torch.from_numpy(out)}

    return f


def spec_dropout(p: float) -> Callable[..., dict[str, Any]]:
    """`nn.dropout` in training mode: the gradient is g · m with the same mask, m = mask / (1 - p)."""
    return lambda d, up: {"grad_x": up["out"] * (d["mask"] / (1 - p))}


def spec_running_stats(momentum: float) -> Callable[..., dict[str, Any]]:
    """`nn.batch_norm` in training mode (graph-ir.md §3.3): running_mean' = (1 - m) · running_mean +
    m · mean_c, running_var' = (1 - m) · running_var + m · var_c · n / (n - 1)."""

    def f(d: dict[str, Any], up: dict[str, Any]) -> dict[str, Any]:
        x = d["x"]
        dims = [0] + list(range(2, x.dim()))
        n = x.numel() // x.shape[1]
        mean = x.mean(dim=dims)
        var = ((x - mean.reshape([1, -1] + [1] * (x.dim() - 2))) ** 2).mean(dim=dims)
        return {"new_running_mean": (1 - momentum) * d["running_mean"] + momentum * mean,
                "new_running_var": (1 - momentum) * d["running_var"] + momentum * var * n / (n - 1)}

    return f


# --- core elementwise ------------------------------------------------------------------------


def binary_cases(op: str, fn: Callable[[Any, Any], Any], call: str, seed: int,
                 b_build: Callable[[list[int]], Builder] | None = None,
                 a_build: Callable[[list[int]], Builder] | None = None) -> list[GradCase]:
    """The standard cases of a broadcasting binary op: same shape, broadcasting in every form (the
    gradient of a broadcast input is summed over the broadcast dims), and one value used twice."""
    a_of = a_build or (lambda s: rnd(s))
    b_of = b_build or (lambda s: rnd(s))
    shapes = [
        ("same_shape_f32", [3, 4], [3, 4], "same shapes."),
        ("broadcast_row_f32", [3, 4], [4], "b broadcast along dim 0: grad_b sums over it."),
        ("broadcast_both_f32", [3, 1], [1, 4], "both broadcast: grad_a sums over dim 1, grad_b over dim 0."),
        ("broadcast_rank_f32", [2, 3, 4], [3, 1], "b of lower rank: grad_b sums over the new dim 0 and dim 2."),
        ("broadcast_scalar_f32", [3, 4], [1], "b a scalar [1]: grad_b sums over every element."),
    ]
    cases = [case(op, n, seed + i, d, [("a", a_of(sa)), ("b", b_of(sb))], lambda t: fn(t["a"], t["b"]), call)
             for i, (n, sa, sb, d) in enumerate(shapes)]
    cases.append(case(op, "same_value_f32", seed + 5, "a and b are the same value.",
                      [("a", a_of([3, 4])), ("b", copy_of("a"))], lambda t: fn(t["a"], t["b"]), call,
                      notes=SAME_VALUE_NOTES.format(a="a", b="b")))
    return cases


def unary(op: str, name: str, seed: int, description: str, x: Builder, fn: Callable[[Any], Any], call: str,
          **kw: Any) -> GradCase:
    return case(op, name, seed, description, [("x", x)], lambda t: fn(t["x"]), call, **kw)


def elementwise_cases() -> list[GradCase]:
    cases: list[GradCase] = []
    cases += binary_cases("core.add", torch.add, "torch.add(a, b)", 10)
    cases += binary_cases("core.sub", torch.sub, "torch.sub(a, b)", 20)
    cases += binary_cases("core.mul", torch.mul, "torch.mul(a, b)", 30)
    away = lambda s: signed(s, 0.5, 2.0)  # noqa: E731 - divisors away from zero
    cases += binary_cases("core.div", torch.div, "torch.div(a, b)", 40, b_build=away, a_build=away)
    cases.append(case(
        "core.div", "zero_divisor_f32", 47, "b = 0 and -0 at finite a (also a = 0): ±inf and NaN gradients.",
        [("a", vals([1.0, -1.0, 0.0, 2.0, -3.0, 0.0])), ("b", vals([0.0, 0.0, 0.0, -0.0, -0.0, -0.0]))],
        lambda t: torch.div(t["a"], t["b"]), "torch.div(a, b)",
        notes="Finite inputs: PyTorch's gradient is the reference also where it is ±inf or NaN (graph-ir.md §3.1.11)."))
    # pow: a > 0 for the ordinary cases (a^b is NaN for a < 0 and non-integer b, see negative_base_f32).
    positive = lambda s: rnd(s, 0.5, 2.0)  # noqa: E731
    cases += [
        case("core.pow", "same_shape_f32", 51, "a in [0.5, 2), b in [-2, 2).",
             [("a", positive([3, 4])), ("b", rnd([3, 4]))], lambda t: torch.pow(t["a"], t["b"]), "torch.pow(a, b)",
             spec=spec_pow),
        case("core.pow", "broadcast_row_f32", 52, "b broadcast along dim 0: grad_b sums over it.",
             [("a", positive([3, 4])), ("b", rnd([4]))], lambda t: torch.pow(t["a"], t["b"]), "torch.pow(a, b)",
             spec=spec_pow),
        case("core.pow", "broadcast_scalar_a_f32", 53, "a a scalar [1]: grad_a sums over every element.",
             [("a", positive([1])), ("b", rnd([3, 4]))], lambda t: torch.pow(t["a"], t["b"]), "torch.pow(a, b)",
             spec=spec_pow),
        case("core.pow", "same_value_f32", 54, "a and b are the same value (x^x).",
             [("a", positive([3, 4])), ("b", copy_of("a"))], lambda t: torch.pow(t["a"], t["b"]), "torch.pow(a, b)",
             spec=spec_pow, notes=SAME_VALUE_NOTES.format(a="a", b="b")),
        case("core.pow", "b_zero_f32", 55, "b = 0 (and -0): ∂/∂a = 0, also at a = 0; ∂/∂b = g · ln(a) (NaN for a < 0, "
             "0 at a = 0).",
             [("a", vals([2.0, -1.5, 0.0, -0.0, 0.5, 3.0, 1.0, -0.25])),
              ("b", vals([0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0]))],
             lambda t: torch.pow(t["a"], t["b"]), "torch.pow(a, b)", spec=spec_pow,
             notes=SPEC_NOTES + "b = 0 gives ∂/∂a = 0; a = 0 with b ≥ 0 gives ∂/∂b = 0; elsewhere the formulas."),
        case("core.pow", "a_zero_f32", 56, "a = 0 (and -0) with b ≥ 0: ∂/∂b = 0; ∂/∂a = g · b · 0^(b-1): +inf for "
             "0 < b < 1, g at b = 1, 0 above.",
             [("a", vals([0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0])),
              ("b", vals([0.5, 0.25, 1.0, 2.0, 3.5, 0.5, 1.0, 2.0]))],
             lambda t: torch.pow(t["a"], t["b"]), "torch.pow(a, b)", spec=spec_pow,
             notes=SPEC_NOTES + "a = 0 with b ≥ 0 gives ∂/∂b = 0; ∂/∂a is the formula, ±inf for 0 < b < 1."),
        case("core.pow", "a_zero_negative_b_f32", 57, "a = 0 (and -0) with b < 0: ±inf gradients.",
             [("a", vals([0.0, -0.0, 0.0, -0.0, 0.0, -0.0])), ("b", vals([-1.0, -1.0, -0.5, -2.0, -3.0, -0.5]))],
             lambda t: torch.pow(t["a"], t["b"]), "torch.pow(a, b)", spec=spec_pow,
             notes=SPEC_NOTES + "elsewhere the formulas: ±inf at a = 0 with b < 0."),
        case("core.pow", "negative_base_f32", 58, "a < 0: ∂/∂b = NaN (ln of a negative number); ∂/∂a finite for "
             "integer b, NaN otherwise.",
             [("a", vals([-2.0, -0.5, -3.0, -1.0, -1.5, -2.0])), ("b", vals([2.0, 3.0, 0.5, -1.0, 1.0, -0.5]))],
             lambda t: torch.pow(t["a"], t["b"]), "torch.pow(a, b)", spec=spec_pow,
             notes=SPEC_NOTES + "elsewhere the formulas: NaN for a < 0."),
    ]
    for op, fn, largest, seed in (("core.maximum", torch.maximum, True, 60), ("core.minimum", torch.minimum, False, 70)):
        call = f"torch.{fn.__name__}(a, b)"
        spec = spec_maximum(largest)
        tie_note = SPEC_NOTES + "at a = b, g / 2 to a and g / 2 to b."
        cases += [
            case(op, "same_shape_f32", seed + 1, "random values, no ties.", [("a", rnd([3, 4])), ("b", rnd([3, 4]))],
                 lambda t, fn=fn: fn(t["a"], t["b"]), call, spec=spec),
            case(op, "broadcast_row_f32", seed + 2, "b broadcast along dim 0: grad_b sums over it.",
                 [("a", rnd([3, 4])), ("b", rnd([4]))], lambda t, fn=fn: fn(t["a"], t["b"]), call, spec=spec),
            case(op, "ties_f32", seed + 3, "a = b at about a quarter of the elements, also 0 against -0.",
                 [("a", vals([1.0, 2.0, -1.0, 0.0, 0.5, -0.5, 3.0, 0.0, -2.0, 1.5, 0.25, -0.0])),
                  ("b", vals([1.0, -2.0, -1.0, -0.0, 0.75, -0.5, 2.0, 0.0, -2.5, 1.5, 0.5, 0.0]))],
                 lambda t, fn=fn: fn(t["a"], t["b"]), call, spec=spec, notes=tie_note),
            case(op, "ties_broadcast_f32", seed + 4, "ties with b broadcast: grad_b sums g, g / 2 and 0.",
                 [("a", choice([4, 3], [-1.0, 0.5, 2.0])), ("b", vals([0.5, 2.0, -1.0]))],
                 lambda t, fn=fn: fn(t["a"], t["b"]), call, spec=spec, notes=tie_note),
            case(op, "same_value_f32", seed + 5, "a and b are the same value: a tie everywhere.",
                 [("a", rnd([3, 4])), ("b", copy_of("a"))], lambda t, fn=fn: fn(t["a"], t["b"]), call, spec=spec,
                 notes=tie_note + " " + SAME_VALUE_NOTES.format(a="a", b="b")),
        ]

    plain = [
        ("core.neg", torch.neg, (-2.0, 2.0), 80),
        ("core.abs", torch.abs, (-2.0, 2.0), 90),
        ("core.exp", torch.exp, (-3.0, 3.0), 100),
        ("core.log", torch.log, (0.1, 10.0), 110),
        ("core.sqrt", torch.sqrt, (0.1, 4.0), 120),
        ("core.rsqrt", torch.rsqrt, (0.1, 4.0), 130),
        ("core.sin", torch.sin, (-4.0, 4.0), 140),
        ("core.cos", torch.cos, (-4.0, 4.0), 150),
        ("core.tanh", torch.tanh, (-3.0, 3.0), 160),
    ]
    for op, fn, (lo, hi), seed in plain:
        call = f"torch.{fn.__name__}(x)"
        cases.append(unary(op, "random_f32", seed + 1, f"uniform in [{lo}, {hi}).", rnd([3, 4, 5], lo, hi), fn, call))
    cases += [
        unary("core.abs", "zero_f32", 92, "x = 0 and -0 among other values: gradient 0 there.",
              vals([0.0, -0.0, 1.5, -2.0, 0.0, -0.25, -0.0, 0.5]), torch.abs, "torch.abs(x)",
              spec=spec_unary(spec_abs), notes=SPEC_NOTES + "abs'(0) = abs'(-0) = 0."),
        unary("core.log", "zero_f32", 112, "x = 0 (log'(0) = +inf · g) and x < 0 (forward NaN, gradient g / x).",
              vals([0.0, 1.0, 0.0, 2.0, -1.0, -0.5]), torch.log, "torch.log(x)",
              notes="Finite inputs: PyTorch's gradient is the reference also where it is ±inf (graph-ir.md §3.1.11). "
              "x = -0 is left out: graph-ir.md doesn't say which sign its infinite gradient has."),
        unary("core.sqrt", "zero_f32", 122, "x = 0: gradient g / (2 · sqrt(x)) = ±inf (sign of g); x < 0 gives NaN.",
              vals([0.0, 1.0, 0.0, 4.0, 0.0, 0.25, -1.0, 2.0]), torch.sqrt, "torch.sqrt(x)",
              spec=spec_unary(spec_sqrt),
              notes=SPEC_NOTES + "sqrt'(0) = g / (2 · sqrt(0)), +inf for g > 0. x = -0 is left out: there PyTorch "
              "gives -inf for g > 0 (the formula with sqrt(-0) = -0), which graph-ir.md's \"+inf for g > 0\" doesn't cover."),
        unary("core.rsqrt", "zero_f32", 132, "x = 0: rsqrt'(0) = -inf · g.",
              vals([0.0, 1.0, 0.0, 4.0, 0.25, 2.0]), torch.rsqrt, "torch.rsqrt(x)",
              notes="Finite inputs: PyTorch's gradient is the reference also where it is ±inf (graph-ir.md §3.1.11). "
              "x = -0 is left out: graph-ir.md doesn't say which sign its infinite gradient has."),
        unary("core.tanh", "saturated_f32", 162, "|x| up to 10: the gradient 1 - tanh² nearly vanishes.",
              vals([-10.0, -6.0, -4.0, 4.0, 6.0, 10.0]), torch.tanh, "torch.tanh(x)"),
        unary("core.exp", "permuted_f32", 103, "a permuted view.", view("permuted", [2, 3, 4]), torch.exp, "torch.exp(x)",
              notes=gen_ops.LAYOUT_NOTES["permuted"]),
    ]

    # clamp: bounds exact in f32, so f64 and f32 agree on which elements sit on a bound.
    def clamp(name: str, seed: int, description: str, x: Builder, lo: Any, hi: Any, convention: bool) -> GradCase:
        attrs = {k: v for k, v in (("min", lo), ("max", hi)) if v is not None}
        kw: dict[str, Any] = {"attributes": attrs, "spec": spec_clamp(lo, hi)}
        if convention:
            kw = with_spec_note(kw, "0 at x = min and x = max (also min = max); g only where min < x < max.")
        return unary("core.clamp", name, seed, description, x, lambda x: torch.clamp(x, lo, hi), "torch.clamp(x, min, max)", **kw)

    at_bounds = [-0.5, 0.75, -1.0, 0.0, -0.5, 1.0, 0.75, 0.25, -0.25, 0.5, 2.0, -0.5]
    cases += [
        clamp("both_f32", 171, "min -1, max 1, x uniform in [-2, 2).", rnd([3, 4]), -1.0, 1.0, False),
        clamp("at_bounds_f32", 172, "min -0.5, max 0.75; x on both bounds, inside and outside.",
              vals(at_bounds, [3, 4]), -0.5, 0.75, True),
        clamp("min_eq_max_f32", 173, "min = max = 0.5: gradient 0 everywhere, also at x = 0.5.",
              vals([0.5, 0.25, 0.5, 1.0, -1.0, 0.5]), 0.5, 0.5, True),
        clamp("min_only_f32", 174, "min 0 only; x = 0 and -0 sit on the bound.",
              vals([0.0, -0.0, 1.0, -1.0, 0.5, 0.0, 2.0, -0.25]), 0.0, None, True),
        clamp("max_only_f32", 175, "max 1 only; x = 1 sits on the bound.",
              vals([1.0, 0.5, 1.0, 2.0, -1.0, 1.0, 0.99951171875, 1.00048828125]), None, 1.0, True),
        clamp("int_bounds_f32", 176, "integer scalar bounds min -1, max 2; x on both.",
              vals([-1.0, 2.0, 0.0, -2.0, 3.0, 1.5, -1.0, 2.0]), -1, 2, True),
    ]

    where_call = "torch.where(cond, a, b)"
    wfn = lambda t: torch.where(t["cond"], t["a"], t["b"])  # noqa: E731
    wnote = SPEC_NOTES + "g to a where cond is true, to b elsewhere, 0 to the other; summed over broadcast dims."
    cases += [
        case("core.where", "same_shape_f32", 181, "cond, a, b of the same shape.",
             [("cond", bools([3, 4])), ("a", rnd([3, 4])), ("b", rnd([3, 4]))], wfn, where_call, spec=spec_where,
             notes=wnote),
        case("core.where", "cond_broadcast_f32", 182, "cond [4] broadcast along dim 0.",
             [("cond", bools([4])), ("a", rnd([3, 4])), ("b", rnd([3, 4]))], wfn, where_call, spec=spec_where,
             notes=wnote),
        case("core.where", "all_broadcast_f32", 183, "cond [3, 1], a [1, 4], b [1]: all three broadcast.",
             [("cond", bools([3, 1])), ("a", rnd([1, 4])), ("b", rnd([1]))], wfn, where_call, spec=spec_where,
             notes=wnote),
        case("core.where", "same_value_f32", 184, "a and b are the same value.",
             [("cond", bools([3, 4])), ("a", rnd([3, 4])), ("b", copy_of("a"))], wfn, where_call, spec=spec_where,
             notes=wnote + " " + SAME_VALUE_NOTES.format(a="a", b="b")),
    ]
    return cases


# --- core shape and indexing -------------------------------------------------------------------


def x_case(op: str, name: str, seed: int, description: str, x: Builder, fn: Callable[[Any], Any], call: str,
           attributes: dict[str, Any], **kw: Any) -> GradCase:
    layout = gen_ops.layout_of(name)
    if layout is not None:
        kw["notes"] = " ".join(n for n in (gen_ops.LAYOUT_NOTES[layout], kw.get("notes", "")) if n)
    return unary(op, name, seed, description, x, fn, call, attributes=attributes, **kw)


def shape_cases() -> list[GradCase]:
    cases: list[GradCase] = []
    reshape = "torch.reshape(x, shape)"
    cases += [
        x_case("core.reshape", "basic_f32", 201, "[2, 3, 4] -> [6, 4].", rnd([2, 3, 4]),
               lambda x: torch.reshape(x, [6, 4]), reshape, {"shape": [6, 4]}),
        x_case("core.reshape", "minus_one_f32", 202, "[2, 3, 4] -> [-1, 2].", rnd([2, 3, 4]),
               lambda x: torch.reshape(x, [-1, 2]), reshape, {"shape": [-1, 2]}),
        x_case("core.reshape", "permuted_f32", 203, "a permuted view [2, 3, 4] -> [4, 6].", view("permuted", [2, 3, 4]),
               lambda x: torch.reshape(x, [4, 6]), reshape, {"shape": [4, 6]}),
    ]
    flatten = "torch.flatten(x, start_dim, end_dim)"
    cases += [
        x_case("core.flatten", "basic_f32", 211, "[2, 3, 4] from dim 1.", rnd([2, 3, 4]),
               lambda x: torch.flatten(x, 1, -1), flatten, {"start_dim": 1}),
        x_case("core.flatten", "middle_f32", 212, "[2, 3, 4, 2], dims 1 to 2.", rnd([2, 3, 4, 2]),
               lambda x: torch.flatten(x, 1, 2), flatten, {"start_dim": 1, "end_dim": 2}),
        x_case("core.flatten", "permuted_f32", 213, "a permuted view [2, 3, 4], all dims.", view("permuted", [2, 3, 4]),
               lambda x: torch.flatten(x, 0, -1), flatten, {}),
    ]
    permute = "torch.permute(x, dims)"
    cases += [
        x_case("core.permute", "basic_f32", 221, "[2, 3, 4] with dims [2, 0, 1].", rnd([2, 3, 4]),
               lambda x: torch.permute(x, [2, 0, 1]), permute, {"dims": [2, 0, 1]}),
        x_case("core.permute", "transposed_f32", 222, "a transposed view [3, 4, 2] with dims [1, 2, 0].",
               view("transposed", [3, 4, 2]), lambda x: torch.permute(x, [1, 2, 0]), permute, {"dims": [1, 2, 0]}),
    ]
    transpose = "torch.transpose(x, dim0, dim1)"
    cases += [
        x_case("core.transpose", "basic_f32", 231, "[3, 4], dims 0 and 1.", rnd([3, 4]),
               lambda x: torch.transpose(x, 0, 1), transpose, {"dim0": 0, "dim1": 1}),
        x_case("core.transpose", "permuted_f32", 232, "a permuted view [2, 3, 4], dims 0 and -1.",
               view("permuted", [2, 3, 4]), lambda x: torch.transpose(x, 0, -1), transpose, {"dim0": 0, "dim1": -1}),
    ]
    squeeze = "torch.squeeze(x, dim)"
    cases += [
        x_case("core.squeeze", "dim_f32", 241, "[3, 1, 4], dim [1].", rnd([3, 1, 4]),
               lambda x: torch.squeeze(x, [1]), squeeze, {"dim": [1]}),
        x_case("core.squeeze", "all_f32", 242, "[1, 3, 1], every 1 removed.", rnd([1, 3, 1]),
               lambda x: torch.squeeze(x), "torch.squeeze(x)", {}),
        x_case("core.squeeze", "permuted_f32", 243, "a permuted view [2, 1, 4], dim [-2].", view("permuted", [2, 1, 4]),
               lambda x: torch.squeeze(x, [-2]), squeeze, {"dim": [-2]}),
    ]
    unsqueeze = "torch.unsqueeze(x, dim)"
    cases += [
        x_case("core.unsqueeze", "basic_f32", 251, "[3, 4], dim -1.", rnd([3, 4]),
               lambda x: torch.unsqueeze(x, -1), unsqueeze, {"dim": -1}),
        x_case("core.unsqueeze", "transposed_f32", 252, "a transposed view [3, 4], dim 1.", view("transposed", [3, 4]),
               lambda x: torch.unsqueeze(x, 1), unsqueeze, {"dim": 1}),
    ]
    expand = "x.expand(shape)"
    cases += [
        x_case("core.expand", "basic_f32", 261, "[3, 1] -> [3, 4]: the gradient sums over the expanded dim.",
               rnd([3, 1]), lambda x: x.expand([3, 4]), expand, {"shape": [3, 4]}),
        x_case("core.expand", "new_dims_f32", 262, "[1, 4] -> [2, 3, 4] with -1: sums over a new and an expanded dim.",
               rnd([1, 4]), lambda x: x.expand([2, 3, -1]), expand, {"shape": [2, 3, -1]}),
        x_case("core.expand", "expanded_f32", 263, "an expanded view [3, 4] -> [2, 3, 4].", view("expanded", [3, 4]),
               lambda x: x.expand([2, 3, 4]), expand, {"shape": [2, 3, 4]}),
    ]
    cat = "torch.cat(xs, dim)"
    cases += [
        case("core.concat", "dim0_f32", 271, "[2, 4] and [3, 4] along dim 0.", [("xs0", rnd([2, 4])), ("xs1", rnd([3, 4]))],
             lambda t: torch.cat([t["xs0"], t["xs1"]], 0), cat, attributes={"dim": 0}),
        case("core.concat", "three_last_dim_f32", 272, "three inputs along dim -1.",
             [("xs0", rnd([2, 3, 1])), ("xs1", rnd([2, 3, 2])), ("xs2", rnd([2, 3, 3]))],
             lambda t: torch.cat([t["xs0"], t["xs1"], t["xs2"]], -1), cat, attributes={"dim": -1}),
        case("core.concat", "same_value_f32", 273, "xs0 and xs1 are the same value.",
             [("xs0", rnd([2, 3])), ("xs1", copy_of("xs0"))], lambda t: torch.cat([t["xs0"], t["xs1"]], 1), cat,
             attributes={"dim": 1}, notes=SAME_VALUE_NOTES.format(a="xs0", b="xs1")),
        case("core.concat", "expanded_f32", 274, "an expanded view [3, 4] and [2, 4] along dim 0.",
             [("xs0", view("expanded", [3, 4])), ("xs1", rnd([2, 4]))], lambda t: torch.cat([t["xs0"], t["xs1"]], 0), cat,
             attributes={"dim": 0}, notes=gen_ops.LAYOUT_NOTES["expanded"].replace("x is", "xs0 is")),
    ]

    def split(sizes: list[int], dim: int) -> Callable[[Any], Any]:
        return lambda x: {str(j): t for j, t in enumerate(torch.split(x, sizes, dim))}

    cases += [
        x_case("core.split", "even_f32", 281, "[6, 2] into [2, 2, 2] along dim 0: an upstream gradient per port.",
               rnd([6, 2]), split([2, 2, 2], 0), "torch.split(x, sizes, dim)", {"dim": 0, "sizes": [2, 2, 2]}),
        x_case("core.split", "uneven_f32", 282, "[2, 7] into [3, 1, 3] along dim -1.", rnd([2, 7]),
               split([3, 1, 3], -1), "torch.split(x, sizes, dim)", {"dim": -1, "sizes": [3, 1, 3]}),
        x_case("core.split", "transposed_f32", 283, "a transposed view [4, 3] into [1, 3] along dim 0.",
               view("transposed", [4, 3]), split([1, 3], 0), "torch.split(x, sizes, dim)", {"dim": 0, "sizes": [1, 3]}),
    ]
    sl = "x[(slice(None),) * dim + (slice(start, end, step),)]"

    def slicer(dim: int, start: int, end: int | None, step: int) -> Callable[[Any], Any]:
        return lambda x: x[(slice(None),) * dim + (slice(start, end, step),)]

    cases += [
        x_case("core.slice", "basic_f32", 291, "[3, 6], dim 1, 1 to 4: the gradient is 0 outside the slice.",
               rnd([3, 6]), slicer(1, 1, 4, 1), sl, {"dim": 1, "start": 1, "end": 4}),
        x_case("core.slice", "step2_f32", 292, "[7, 2], dim 0, step 2.", rnd([7, 2]), slicer(0, 0, None, 2), sl,
               {"dim": 0, "step": 2}),
        x_case("core.slice", "negative_bounds_f32", 293, "[2, 8], dim 1, -6 to -1, step 3.", rnd([2, 8]),
               slicer(1, -6, -1, 3), sl, {"dim": 1, "start": -6, "end": -1, "step": 3}),
        x_case("core.slice", "transposed_f32", 294, "a transposed view [5, 4], dim 0, 1 to 4.",
               view("transposed", [5, 4]), slicer(0, 1, 4, 1), sl, {"dim": 0, "start": 1, "end": 4}),
    ]
    pad = "F.pad(x, pad, mode, value)"
    cases += [
        x_case("core.pad", "last_dim_f32", 301, "[3, 4], pad [1, 2] (last dim): the gradient drops the padding.",
               rnd([3, 4]), lambda x: F.pad(x, [1, 2], "constant", 0.0), pad, {"pad": [1, 2]}),
        x_case("core.pad", "two_dims_value_f32", 302, "[2, 3, 2], pad [1, 1, 2, 0] with value 0.5.", rnd([2, 3, 2]),
               lambda x: F.pad(x, [1, 1, 2, 0], "constant", 0.5), pad, {"pad": [1, 1, 2, 0], "value": 0.5}),
        x_case("core.pad", "permuted_f32", 303, "a permuted view [2, 3, 4], pad [0, 1, 1, 0].", view("permuted", [2, 3, 4]),
               lambda x: F.pad(x, [0, 1, 1, 0], "constant", 0.0), pad, {"pad": [0, 1, 1, 0]}),
    ]
    gather = "torch.gather(x, dim, index)"
    gnote = SPEC_NOTES + "repeated indices receive the sum of their gradients."
    cases += [
        case("core.gather", "repeated_dim0_f32", 311, "x [3, 4], index [4, 4] along dim 0 with repeated indices.",
             [("x", rnd([3, 4])), ("index", ints([0, 1, 2, 0, 2, 2, 0, 1, 0, 0, 1, 1, 2, 2, 2, 0], [4, 4]))],
             lambda t: torch.gather(t["x"], 0, t["index"]), gather, attributes={"dim": 0},
             spec=spec_scatter("index", 0, "gather"), notes=gnote),
        case("core.gather", "repeated_last_dim_i32_f32", 312, "x [2, 5], i32 index [2, 6] along dim -1.",
             [("x", rnd([2, 5])), ("index", ints([4, 4, 0, 1, 4, 2, 3, 3, 3, 0, 0, 1], [2, 6], "i32"))],
             lambda t: torch.gather(t["x"], -1, t["index"].long()), gather, attributes={"dim": -1},
             spec=spec_scatter("index", -1, "gather"), notes=gnote),
        case("core.gather", "permuted_f32", 313, "a permuted view x [2, 3, 4], index [2, 3, 2] along dim 2.",
             [("x", view("permuted", [2, 3, 4])), ("index", ints([3, 3, 0, 1, 2, 2, 1, 1, 0, 0, 3, 2], [2, 3, 2]))],
             lambda t: torch.gather(t["x"], 2, t["index"]), gather, attributes={"dim": 2},
             spec=spec_scatter("index", 2, "gather"), notes=gen_ops.LAYOUT_NOTES["permuted"] + " " + gnote),
    ]
    isel = "torch.index_select(x, dim, index)"
    cases += [
        case("core.index_select", "repeated_dim0_f32", 321, "x [4, 3], index [0, 2, 0, 0, 3] along dim 0.",
             [("x", rnd([4, 3])), ("index", ints([0, 2, 0, 0, 3]))],
             lambda t: torch.index_select(t["x"], 0, t["index"]), isel, attributes={"dim": 0},
             spec=spec_scatter("index", 0, "index_select"), notes=gnote),
        case("core.index_select", "repeated_dim1_i32_f32", 322, "x [2, 4, 3], i32 index [3, 3, 1] along dim 1.",
             [("x", rnd([2, 4, 3])), ("index", ints([3, 3, 1], None, "i32"))],
             lambda t: torch.index_select(t["x"], 1, t["index"].long()), isel, attributes={"dim": 1},
             spec=spec_scatter("index", 1, "index_select"), notes=gnote),
        case("core.index_select", "transposed_f32", 323, "a transposed view x [3, 5], index [4, 0, 4] along dim -1.",
             [("x", view("transposed", [3, 5])), ("index", ints([4, 0, 4]))],
             lambda t: torch.index_select(t["x"], -1, t["index"]), isel, attributes={"dim": -1},
             spec=spec_scatter("index", -1, "index_select"), notes=gen_ops.LAYOUT_NOTES["transposed"] + " " + gnote),
    ]
    cases.append(x_case("core.cast", "f32_to_f32", 331, "[3, 4] cast to its own dtype: the identity.", rnd([3, 4]),
                        lambda x: x.to(x.dtype), "x.to(dtype)", {"dtype": "f32"},
                        notes="Casts that change the dtype involve f16/bf16 (or integers, which get no gradient); M2's "
                        "gradient fixtures are f32 only (graph-ir.md §3.1.11)."))
    return cases


# --- core reductions and others ----------------------------------------------------------------


def reduction_cases() -> list[GradCase]:
    cases: list[GradCase] = []
    for op, fn, seed in (("core.sum", torch.sum, 400), ("core.mean", torch.mean, 410)):
        name = fn.__name__
        call = f"torch.{name}(x, dim, keepdim)"
        cases += [
            x_case(op, "all_f32", seed + 1, "[3, 4], every dim: out is [1].", rnd([3, 4]),
                   lambda x, fn=fn: fn(x).reshape(1), f"torch.{name}(x)", {}),
            x_case(op, "dim1_f32", seed + 2, "[3, 4, 2], dim [1].", rnd([3, 4, 2]),
                   lambda x, fn=fn: fn(x, [1]), call, {"dim": [1]}),
            x_case(op, "dims_keepdim_f32", seed + 3, "[2, 3, 4], dims [0, 2], keepdim.", rnd([2, 3, 4]),
                   lambda x, fn=fn: fn(x, [0, 2], True), call, {"dim": [0, 2], "keepdim": True}),
            x_case(op, "permuted_f32", seed + 4, "a permuted view [2, 3, 4], dim [-1].", view("permuted", [2, 3, 4]),
                   lambda x, fn=fn: fn(x, [-1]), call, {"dim": [-1]}),
        ]
    for op, fn, largest, seed in (("core.max", torch.amax, True, 420), ("core.min", torch.amin, False, 440)):
        name = fn.__name__
        call = f"torch.{name}(x, dim, keepdim)"
        s = 1.0 if largest else -1.0  # min cases use the negated values: the same ties at the minimum

        def v(values: list[float], shape: list[int], s: float = s) -> Builder:
            return vals([s * x for x in values], shape)

        note = SPEC_NOTES + "the k elements of a reduced slice equal to the result each get g / k."
        cases += [
            x_case(op, "random_f32", seed + 1, "[3, 4, 5], dim [-1], no ties.", rnd([3, 4, 5]),
                   lambda x, fn=fn: fn(x, [-1]), call, {"dim": [-1]}, spec=spec_extremum([-1], largest)),
            x_case(op, "ties_all_f32", seed + 2, "[3, 4], every dim: the extremum occurs 4 times (g / 4 each).",
                   v([1.0, 3.0, 0.5, 3.0, -1.0, 3.0, 2.0, 0.0, 3.0, -2.0, 1.0, 1.0], [3, 4]),
                   lambda x, fn=fn: fn(x).reshape(1), f"torch.{name}(x)", {}, spec=spec_extremum(None, largest),
                   notes=note),
            x_case(op, "ties_dim_f32", seed + 3, "[3, 4], dim [1]: rows with 2, 4 and 1 equal extrema.",
                   v([1.0, 3.0, 3.0, 0.0, 2.0, 2.0, 2.0, 2.0, -1.0, 0.5, -1.0, 0.25], [3, 4]),
                   lambda x, fn=fn: fn(x, [1]), call, {"dim": [1]}, spec=spec_extremum([1], largest), notes=note),
            x_case(op, "ties_dims_keepdim_f32", seed + 4, "[2, 3, 2], dims [0, 2], keepdim: ties across both dims.",
                   v([1.0, 2.0, 0.5, 0.5, 3.0, -1.0, 2.0, 1.0, 0.5, 0.0, 3.0, 3.0], [2, 3, 2]),
                   lambda x, fn=fn: fn(x, [0, 2], True), call, {"dim": [0, 2], "keepdim": True},
                   spec=spec_extremum([0, 2], largest), notes=note),
            x_case(op, "zero_ties_f32", seed + 5, "0 and -0 tie as the extremum (g / 2 each).",
                   v([0.0, -0.0, -1.0, -2.0, -0.5], [5]),
                   lambda x, fn=fn: fn(x).reshape(1), f"torch.{name}(x)", {}, spec=spec_extremum(None, largest),
                   notes=note),
            x_case(op, "permuted_f32", seed + 6, "a permuted view [2, 3, 4], dim [0].", view("permuted", [2, 3, 4]),
                   lambda x, fn=fn: fn(x, [0]), call, {"dim": [0]}, spec=spec_extremum([0], largest)),
        ]
    mm = "torch.matmul(a, b)"
    mfn = lambda t: torch.matmul(t["a"], t["b"])  # noqa: E731
    cases += [
        case("core.matmul", "basic_f32", 461, "[3, 4] @ [4, 5].", [("a", rnd([3, 4])), ("b", rnd([4, 5]))], mfn, mm),
        case("core.matmul", "batch_f32", 462, "[2, 3, 4] @ [2, 4, 5].", [("a", rnd([2, 3, 4])), ("b", rnd([2, 4, 5]))],
             mfn, mm),
        case("core.matmul", "broadcast_rhs_f32", 463, "[2, 3, 4] @ [4, 5]: grad_b sums over the batch.",
             [("a", rnd([2, 3, 4])), ("b", rnd([4, 5]))], mfn, mm),
        case("core.matmul", "broadcast_batch_f32", 464, "[2, 1, 3, 4] @ [3, 4, 2]: both batch dims broadcast.",
             [("a", rnd([2, 1, 3, 4])), ("b", rnd([3, 4, 2]))], mfn, mm),
        case("core.matmul", "same_value_f32", 465, "a and b are the same [4, 4] value (x @ x).",
             [("a", rnd([4, 4])), ("b", copy_of("a"))], mfn, mm, notes=SAME_VALUE_NOTES.format(a="a", b="b")),
        case("core.matmul", "transposed_f32", 466, "a transposed view a [3, 4] @ b [4, 2].",
             [("a", view("transposed", [3, 4])), ("b", rnd([4, 2]))], mfn, mm,
             notes=gen_ops.LAYOUT_NOTES["transposed"].replace("x is", "a is")),
    ]
    for op, fn, seed in (("core.softmax", torch.softmax, 470), ("core.log_softmax", torch.log_softmax, 480)):
        call = f"torch.{fn.__name__}(x, dim)"
        cases += [
            x_case(op, "last_dim_f32", seed + 1, "[3, 5], dim -1.", rnd([3, 5], -4.0, 4.0),
                   lambda x, fn=fn: fn(x, -1), call, {"dim": -1}),
            x_case(op, "dim0_f32", seed + 2, "[4, 2, 3], dim 0.", rnd([4, 2, 3], -4.0, 4.0),
                   lambda x, fn=fn: fn(x, 0), call, {"dim": 0}),
            x_case(op, "transposed_f32", seed + 3, "a transposed view [3, 4], dim 1.", view("transposed", [3, 4]),
                   lambda x, fn=fn: fn(x, 1), call, {"dim": 1}),
        ]
    cumsum = "torch.cumsum(x, dim)"
    cases += [
        x_case("core.cumsum", "last_dim_f32", 491, "[3, 5], dim -1.", rnd([3, 5]), lambda x: torch.cumsum(x, -1),
               cumsum, {"dim": -1}),
        x_case("core.cumsum", "dim0_f32", 492, "[4, 2, 3], dim 0.", rnd([4, 2, 3]), lambda x: torch.cumsum(x, 0),
               cumsum, {"dim": 0}),
        x_case("core.cumsum", "permuted_f32", 493, "a permuted view [2, 3, 4], dim 1.", view("permuted", [2, 3, 4]),
               lambda x: torch.cumsum(x, 1), cumsum, {"dim": 1}),
    ]

    def top_k(k: int, dim: int, largest: bool) -> Callable[[Any], Any]:
        def fn(x: Any) -> Any:
            values, indices = torch.sort(x, dim=dim, descending=largest, stable=True)
            return {"values": values.narrow(dim, 0, k), "indices": indices.narrow(dim, 0, k)}

        return fn

    topk = "torch.sort(x, dim, descending=largest, stable=True) cut to the first k"
    tnote = SPEC_NOTES + ("the gradient of values goes to the elements at the returned indices, which among equal "
                          "values are the lowest (graph-ir.md §3.1.8). torch.topk leaves the order of ties open, so "
                          "the forward reference is the stable sort it otherwise equals (graph-ir.md §3.2).")
    cases += [
        x_case("core.top_k", "largest_f32", 501, "[3, 6], k 2, dim -1.", rnd([3, 6]), top_k(2, -1, True), topk,
               {"k": 2}, spec=spec_top_k(2, -1, True), notes=tnote),
        x_case("core.top_k", "smallest_f32", 502, "[3, 6], k 3, largest false.", rnd([3, 6]), top_k(3, -1, False),
               topk, {"k": 3, "largest": False}, spec=spec_top_k(3, -1, False), notes=tnote),
        x_case("core.top_k", "ties_f32", 503, "[2, 6], k 3, dim -1: ties at the cut go to the lowest indices.",
               vals([1.0, 3.0, 3.0, 3.0, 0.5, 3.0, 2.0, 1.0, 2.0, 2.0, 1.0, 0.0], [2, 6]), top_k(3, -1, True), topk,
               {"k": 3}, spec=spec_top_k(3, -1, True), notes=tnote),
        x_case("core.top_k", "ties_smallest_f32", 504, "[2, 5], k 2, largest false, ties.",
               vals([1.0, 0.0, 1.0, 0.0, 0.0, 2.0, -1.0, 2.0, -1.0, 3.0], [2, 5]), top_k(2, -1, False), topk,
               {"k": 2, "largest": False}, spec=spec_top_k(2, -1, False), notes=tnote),
        x_case("core.top_k", "dim0_f32", 505, "[5, 3], k 2, dim 0.", rnd([5, 3]), top_k(2, 0, True), topk,
               {"k": 2, "dim": 0}, spec=spec_top_k(2, 0, True), notes=tnote),
        x_case("core.top_k", "permuted_f32", 506, "a permuted view [2, 3, 4], k 2, dim -1.", view("permuted", [2, 3, 4]),
               top_k(2, -1, True), topk, {"k": 2}, spec=spec_top_k(2, -1, True), notes=tnote),
    ]
    return cases


# --- nn layers ---------------------------------------------------------------------------------


def nn_x(name: str, shape: list[int], low: float = -2.0, high: float = 2.0) -> Builder:
    """`x` of an nn case, as the view the case name asks for (default range only for views)."""
    layout = gen_ops.layout_of(name)
    return rnd(shape, low, high) if layout is None else view(layout, shape)


def nn_notes(name: str, notes: str = "") -> str:
    layout = gen_ops.layout_of(name)
    return " ".join(n for n in ((gen_ops.LAYOUT_NOTES[layout] if layout else ""), notes) if n)


def linear(name: str, seed: int, x_shape: list[int], out: int, bias: bool, description: str) -> GradCase:
    inp = x_shape[-1]
    tensors = [("x", nn_x(name, x_shape)), ("weight", rnd([out, inp], -1.0, 1.0))]
    tensors += [("bias", rnd([out], -1.0, 1.0))] if bias else []
    return case("nn.linear", name, seed, description, tensors, lambda t: F.linear(t["x"], t["weight"], t.get("bias")),
                "F.linear(x, weight, bias)", params=("weight", "bias") if bias else ("weight",), notes=nn_notes(name))


def conv(op: str, name: str, seed: int, x_shape: list[int], c_out: int, kernel: list[int], description: str,
         bias: bool = True, **attrs: Any) -> GradCase:
    groups = attrs.get("groups", 1)
    w_shape = [c_out, x_shape[1] // groups] + kernel
    fn = F.conv1d if op == "nn.conv1d" else F.conv2d
    nd = len(kernel)
    defaults = {"stride": 1 if nd == 1 else [1, 1], "padding": 0 if nd == 1 else [0, 0],
                "dilation": 1 if nd == 1 else [1, 1], "groups": 1}
    a = defaults | attrs
    tensors = [("x", nn_x(name, x_shape)), ("weight", rnd(w_shape, -1.0, 1.0))]
    tensors += [("bias", rnd([c_out], -1.0, 1.0))] if bias else []
    return case(op, name, seed, description, tensors,
                lambda t: fn(t["x"], t["weight"], t.get("bias"), a["stride"], a["padding"], a["dilation"], a["groups"]),
                f"F.{fn.__name__}(x, weight, bias, stride, padding, dilation, groups)", attributes=attrs,
                params=("weight", "bias") if bias else ("weight",), notes=nn_notes(name))


def layer_cases() -> list[GradCase]:
    cases = [
        linear("2d_f32", 601, [3, 6], 4, True, "x [3, 6], weight [4, 6], bias."),
        linear("no_bias_f32", 602, [3, 6], 4, False, "no bias."),
        linear("rank1_f32", 603, [5], 3, True, "x of rank 1 [5]."),
        linear("rank3_f32", 604, [2, 3, 4], 5, True, "x [2, 3, 4]: weight and bias gradients sum over two dims."),
        linear("permuted_f32", 605, [2, 3, 4], 3, True, "a permuted view x [2, 3, 4]."),
    ]
    enote = SPEC_NOTES + "repeated ids receive the sum of their gradients."
    emb = "F.embedding(ids, weight)"
    efn = lambda t: F.embedding(t["ids"], t["weight"])  # noqa: E731
    cases += [
        case("nn.embedding", "repeated_i64_f32", 611, "ids [2, 3] with repeats, weight [5, 4].",
             [("ids", ints([1, 3, 1, 0, 3, 3], [2, 3])), ("weight", rnd([5, 4]))], efn, emb, params=("weight",),
             spec=spec_scatter("ids", 0, "embedding"), notes=enote),
        case("nn.embedding", "repeated_i32_f32", 612, "i32 ids [7] with repeats, weight [4, 3]; row 2 unused.",
             [("ids", ints([0, 0, 3, 1, 0, 3, 3], None, "i32")), ("weight", rnd([4, 3]))],
             lambda t: F.embedding(t["ids"].long(), t["weight"]), emb, params=("weight",),
             spec=spec_scatter("ids", 0, "embedding"), notes=enote),
        case("nn.embedding", "rank3_i64_f32", 613, "ids [2, 2, 3], weight [6, 2].",
             [("ids", ints([5, 0, 5, 2, 2, 1, 0, 0, 4, 5, 3, 5], [2, 2, 3])), ("weight", rnd([6, 2]))], efn, emb,
             params=("weight",), spec=spec_scatter("ids", 0, "embedding"), notes=enote),
    ]
    c1, c2 = "nn.conv1d", "nn.conv2d"
    cases += [
        conv(c1, "basic_f32", 621, [2, 3, 8], 4, [3], "defaults."),
        conv(c1, "stride_padding_f32", 622, [2, 2, 9], 3, [3], "stride 2, padding 1.", stride=2, padding=1),
        conv(c1, "dilation_f32", 623, [1, 2, 10], 2, [3], "dilation 2.", dilation=2),
        conv(c1, "groups2_f32", 624, [2, 4, 7], 6, [2], "groups 2.", groups=2),
        conv(c1, "depthwise_f32", 625, [1, 3, 6], 3, [3], "groups = C_in = C_out.", groups=3, padding=1),
        conv(c1, "no_bias_f32", 626, [2, 2, 6], 3, [2], "no bias.", bias=False),
        conv(c1, "all_attrs_f32", 627, [2, 4, 11], 2, [3], "stride 2, padding 2, dilation 2, groups 2.", stride=2,
             padding=2, dilation=2, groups=2),
        conv(c1, "permuted_f32", 628, [2, 3, 6], 2, [3], "a permuted view x."),
        conv(c2, "basic_f32", 631, [2, 2, 5, 5], 3, [3, 3], "defaults."),
        conv(c2, "stride_padding_f32", 632, [1, 2, 6, 6], 2, [3, 3], "stride [2, 2], padding [1, 1].", stride=[2, 2],
             padding=[1, 1]),
        conv(c2, "dilation_f32", 633, [1, 2, 7, 7], 2, [3, 3], "dilation [2, 2].", dilation=[2, 2]),
        conv(c2, "groups2_f32", 634, [1, 4, 5, 5], 4, [2, 2], "groups 2.", groups=2),
        conv(c2, "depthwise_f32", 635, [2, 3, 4, 4], 3, [3, 3], "groups = C_in = C_out, padding [1, 1].", groups=3,
             padding=[1, 1]),
        conv(c2, "no_bias_f32", 636, [1, 2, 4, 4], 2, [2, 2], "no bias.", bias=False),
        conv(c2, "nonsquare_f32", 637, [1, 2, 5, 6], 3, [2, 3], "kernel [2, 3] on [5, 6].", stride=[1, 2]),
        conv(c2, "all_attrs_f32", 638, [2, 4, 7, 6], 2, [3, 2], "stride [2, 1], padding [1, 2], dilation [2, 1], "
             "groups 2.", stride=[2, 1], padding=[1, 2], dilation=[2, 1], groups=2),
        conv(c2, "permuted_f32", 639, [1, 2, 4, 5], 2, [3, 3], "a permuted view x.", padding=[1, 1]),
    ]
    return cases


def pool_cases() -> list[GradCase]:
    cases: list[GradCase] = []

    def maxpool(name: str, seed: int, description: str, x: Builder, k: list[int], stride: list[int] | None = None,
                padding: list[int] | None = None, dilation: list[int] | None = None, convention: bool = False) -> GradCase:
        attrs: dict[str, Any] = {"kernel_size": k}
        for key, v in (("stride", stride), ("padding", padding), ("dilation", dilation)):
            if v is not None:
                attrs[key] = v
        s, p, dl = stride or k, padding or [0, 0], dilation or [1, 1]
        kw: dict[str, Any] = {"attributes": attrs, "spec": spec_max_pool2d(k, s, p, dl), "notes": nn_notes(name)}
        if convention:
            kw = with_spec_note(kw, "all of g to the first maximum of a window in row-major order (kh, then kw); "
                                "overlapping windows add up.")
        return unary("nn.max_pool2d", name, seed, description, x, lambda x: F.max_pool2d(x, k, s, p, dl),
                     "F.max_pool2d(x, kernel_size, stride, padding, dilation)", **kw)

    ties = [1.0, 2.0, 2.0, 0.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 3.0, 3.0, -1.0, 3.0, 3.0, 0.0]
    cases += [
        maxpool("k2_f32", 701, "[2, 2, 4, 4], kernel [2, 2].", rnd([2, 2, 4, 4]), [2, 2]),
        maxpool("ties_f32", 702, "[1, 1, 4, 4] with equal maxima inside windows, kernel [2, 2].",
                vals(ties, [1, 1, 4, 4]), [2, 2], convention=True),
        maxpool("overlap_ties_f32", 703, "[1, 1, 3, 3] all equal, kernel [2, 2], stride [1, 1]: overlapping windows.",
                vals([1.0] * 9, [1, 1, 3, 3]), [2, 2], [1, 1], convention=True),
        maxpool("overlap_random_ties_f32", 704, "[1, 2, 4, 5] of few values, kernel [3, 3], stride [1, 2].",
                choice([1, 2, 4, 5], [0.0, 1.0, 2.0]), [3, 3], [1, 2], convention=True),
        maxpool("k3_s1_p1_f32", 705, "[1, 2, 4, 4] negative values, kernel [3, 3], stride [1, 1], padding [1, 1].",
                rnd([1, 2, 4, 4], -3.0, -1.0), [3, 3], [1, 1], [1, 1]),
        maxpool("padded_ties_f32", 706, "[1, 1, 3, 3] all equal, kernel [2, 2], stride [1, 1], padding [1, 1].",
                vals([-1.0] * 9, [1, 1, 3, 3]), [2, 2], [1, 1], [1, 1], convention=True),
        maxpool("dilation_f32", 707, "[1, 2, 6, 6], kernel [2, 2], stride [1, 1], dilation [2, 2].",
                rnd([1, 2, 6, 6]), [2, 2], [1, 1], None, [2, 2]),
        maxpool("dilation_padding_ties_f32", 708, "[1, 1, 5, 5] of few values, kernel [3, 2], stride [2, 1], "
                "padding [1, 1], dilation [2, 2].", choice([1, 1, 5, 5], [0.0, 1.0]), [3, 2], [2, 1], [1, 1], [2, 2],
                convention=True),
        maxpool("nonsquare_f32", 709, "[2, 1, 5, 6], kernel [2, 3], stride [3, 2].", rnd([2, 1, 5, 6]), [2, 3], [3, 2]),
        maxpool("permuted_f32", 710, "a permuted view [2, 2, 4, 4], kernel [2, 2].", nn_x("permuted_f32", [2, 2, 4, 4]),
                [2, 2]),
    ]

    def avgpool(name: str, seed: int, description: str, shape: list[int], k: list[int], stride: list[int] | None = None,
                padding: list[int] | None = None, include: bool | None = None) -> GradCase:
        attrs: dict[str, Any] = {"kernel_size": k}
        for key, v in (("stride", stride), ("padding", padding), ("count_include_pad", include)):
            if v is not None:
                attrs[key] = v
        s, p, c = stride or k, padding or [0, 0], True if include is None else include
        return unary("nn.avg_pool2d", name, seed, description, nn_x(name, shape),
                     lambda x: F.avg_pool2d(x, k, s, p, count_include_pad=c),
                     "F.avg_pool2d(x, kernel_size, stride, padding, count_include_pad=count_include_pad)",
                     attributes=attrs, notes=nn_notes(name))

    cases += [
        avgpool("k2_f32", 721, "[2, 2, 4, 4], kernel [2, 2].", [2, 2, 4, 4], [2, 2]),
        avgpool("overlap_f32", 722, "[1, 2, 5, 5], kernel [3, 3], stride [1, 1]: overlapping windows.", [1, 2, 5, 5],
                [3, 3], [1, 1]),
        avgpool("padding_include_f32", 723, "[1, 2, 5, 5], kernel [3, 3], stride [2, 2], padding [1, 1], padding "
                "counted.", [1, 2, 5, 5], [3, 3], [2, 2], [1, 1]),
        avgpool("padding_exclude_f32", 724, "the same with count_include_pad false.", [1, 2, 5, 5], [3, 3], [2, 2],
                [1, 1], False),
        avgpool("nonsquare_f32", 725, "[2, 1, 5, 6], kernel [2, 3], stride [1, 2], padding [1, 1].", [2, 1, 5, 6],
                [2, 3], [1, 2], [1, 1]),
        avgpool("permuted_f32", 726, "a permuted view [2, 2, 4, 4], kernel [2, 2].", [2, 2, 4, 4], [2, 2]),
    ]

    def adaptive(name: str, seed: int, description: str, shape: list[int], size: list[int]) -> GradCase:
        return unary("nn.adaptive_avg_pool2d", name, seed, description, nn_x(name, shape),
                     lambda x: F.adaptive_avg_pool2d(x, size), "F.adaptive_avg_pool2d(x, output_size)",
                     attributes={"output_size": size}, notes=nn_notes(name))

    cases += [
        adaptive("divisible_f32", 731, "[2, 2, 4, 4] -> [2, 2].", [2, 2, 4, 4], [2, 2]),
        adaptive("overlap_f32", 732, "[1, 2, 5, 7] -> [3, 4]: cells overlap.", [1, 2, 5, 7], [3, 4]),
        adaptive("global_f32", 733, "[2, 3, 3, 4] -> [1, 1].", [2, 3, 3, 4], [1, 1]),
        adaptive("permuted_f32", 734, "a permuted view [1, 2, 4, 6] -> [2, 3].", [1, 2, 4, 6], [2, 3]),
    ]

    def dropout(name: str, seed: int, description: str, x: Builder, p: float, training: bool, mask: Builder | None = None,
                notes: str = "") -> GradCase:
        attrs = {} if p == 0.5 else {"p": p}
        if not training:
            return unary("nn.dropout", name, seed, description, x, lambda x: F.dropout(x, p, training=False),
                         "F.dropout(x, p, training=False)", attributes=attrs,
                         notes="Inference mode: the identity for every p (graph-ir.md §3.3).")
        return case("nn.dropout", name, seed, description, [("x", x), ("mask", mask or keep_mask([3, 4], p))],
                    lambda t: t["x"] * (t["mask"] / (1 - p)), "x * (mask / (1 - p))", attributes=attrs,
                    mode="training", spec=spec_dropout(p),
                    notes=" ".join(n for n in (notes, "Training mode with the given keep mask instead of a generated one "
                                               "(graph-ir.md §3.3): out = x · m and the gradient g · m, with m = mask / (1 - p).") if n))

    cases += [
        dropout("inference_f32", 741, "[3, 4], p 0.5 (default), inference mode.", rnd([3, 4]), 0.5, False),
        dropout("train_p05_f32", 742, "[3, 4], p 0.5 (default), training mode.", rnd([3, 4]), 0.5, True),
        dropout("train_p0_f32", 743, "[3, 4], p 0, training mode: the identity.", rnd([3, 4]), 0.0, True,
                vals([1.0] * 12, [3, 4]), notes="p = 0 draws nothing and keeps every element, so the mask is all ones."),
        dropout("train_p03_f32", 744, "[3, 4], p 0.3, training mode: a scale that is not a power of two.", rnd([3, 4]),
                0.3, True),
        dropout("permuted_train_f32", 745, "a permuted view [2, 3, 2], p 0.5, training mode.",
                view("permuted", [2, 3, 2]), 0.5, True, keep_mask([2, 3, 2], 0.5),
                notes=gen_ops.LAYOUT_NOTES["permuted"]),
    ]
    return cases


def norm_cases() -> list[GradCase]:
    cases: list[GradCase] = []

    def norm(op: str, name: str, seed: int, description: str, shape: list[int], params: tuple[str, ...],
             attrs: dict[str, Any] | None = None, mode: str = "inference", x: Builder | None = None) -> GradCase:
        attrs = attrs or {}
        eps = attrs.get("eps", 1e-5)
        pshape = [shape[1]] if op in ("nn.batch_norm", "nn.group_norm") else list(attrs["normalized_shape"])
        builders = {"running_mean": rnd(pshape, -0.5, 0.5), "running_var": rnd(pshape, 0.5, 2.0),
                    "weight": rnd(pshape, 0.5, 1.5), "bias": rnd(pshape, -1.0, 1.0)}
        tensors = [("x", x or nn_x(name, shape))] + [(p, builders[p]) for p in params]
        training = mode == "training"
        momentum = attrs.get("momentum", 0.1)

        def fn(t: dict[str, Any]) -> Any:
            xx, w, b = t["x"], t.get("weight"), t.get("bias")
            if op == "nn.batch_norm":
                return F.batch_norm(xx, t["running_mean"], t["running_var"], w, b, training=training, momentum=momentum,
                                    eps=eps)
            if op == "nn.layer_norm":
                return F.layer_norm(xx, pshape, w, b, eps)
            if op == "nn.group_norm":
                return F.group_norm(xx, attrs["num_groups"], w, b, eps)
            return F.rms_norm(xx, pshape, w, eps)

        call = {
            "nn.batch_norm": f"F.batch_norm(x, running_mean, running_var, weight, bias, training={training}, "
                             "momentum=momentum, eps=eps)",
            "nn.layer_norm": "F.layer_norm(x, normalized_shape, weight, bias, eps)",
            "nn.group_norm": "F.group_norm(x, num_groups, weight, bias, eps)",
            "nn.rms_norm": "F.rms_norm(x, normalized_shape, weight, eps)",
        }[op]
        notes = nn_notes(name)
        spec = None
        if op == "nn.batch_norm":
            if training:
                notes = " ".join(n for n in (notes, "Training mode: normalizes with the batch statistics (biased "
                                             "variance), the gradient flows through them; new_running_mean and "
                                             "new_running_var are the updated running statistics (unbiased variance, "
                                             "graph-ir.md §3.3), checked against that formula by the generator. The "
                                             "running statistics get no gradient.") if n)
                spec = spec_running_stats(momentum)
            else:
                notes = " ".join(n for n in (notes, "Inference mode: normalizes with the running statistics, which "
                                             "get no gradient.") if n)
        return case(op, name, seed, description, tensors, fn, call, attributes=attrs, params=params, mode=mode,
                    notes=notes, spec=spec)

    bn, all_bn = "nn.batch_norm", ("running_mean", "running_var", "weight", "bias")
    stats = ("running_mean", "running_var")
    cases += [
        norm(bn, "rank4_f32", 801, "[2, 3, 3, 3], every param, inference mode.", [2, 3, 3, 3], all_bn),
        norm(bn, "rank2_f32", 802, "[4, 3], every param, inference mode.", [4, 3], all_bn),
        norm(bn, "no_weight_no_bias_f32", 803, "[2, 3, 4], running statistics only, inference mode.", [2, 3, 4], stats),
        norm(bn, "permuted_f32", 804, "a permuted view [2, 3, 2, 2], inference mode.", [2, 3, 2, 2], all_bn),
        norm(bn, "train_rank4_f32", 811, "[2, 3, 3, 3], every param, training mode, momentum 0.1 (default).",
             [2, 3, 3, 3], all_bn, mode="training"),
        norm(bn, "train_rank2_f32", 812, "[5, 3], every param, training mode, momentum 0.3.", [5, 3], all_bn,
             {"momentum": 0.3}, "training"),
        norm(bn, "train_no_weight_no_bias_f32", 813, "[2, 3, 2, 2], no weight and bias, training mode.", [2, 3, 2, 2],
             stats, mode="training"),
        norm(bn, "train_no_bias_rank2_f32", 814, "[4, 2], weight without bias, training mode, momentum 0.25.", [4, 2],
             ("running_mean", "running_var", "weight"), {"momentum": 0.25}, "training"),
        norm(bn, "train_no_weight_f32", 815, "[3, 2, 4], bias without weight, training mode.", [3, 2, 4],
             ("running_mean", "running_var", "bias"), mode="training"),
        norm(bn, "train_momentum0_f32", 816, "[3, 2, 2, 2], training mode, momentum 0: running statistics kept.",
             [3, 2, 2, 2], all_bn, {"momentum": 0.0}, "training"),
        norm(bn, "train_momentum1_f32", 817, "[3, 2, 2, 2], training mode, momentum 1: replaced by the batch's.",
             [3, 2, 2, 2], all_bn, {"momentum": 1.0}, "training"),
        norm(bn, "train_n2_eps_f32", 818, "[2, 3]: n = 2, the fewest values per channel, eps 1e-3, training mode.",
             [2, 3], all_bn, {"eps": 1e-3}, "training"),
        norm(bn, "permuted_train_f32", 819, "a permuted view [2, 3, 2, 2], training mode.", [2, 3, 2, 2], all_bn,
             mode="training"),
    ]
    ln = "nn.layer_norm"
    cases += [
        norm(ln, "last_dim_f32", 821, "[3, 5], normalized_shape [5], weight and bias.", [3, 5], ("weight", "bias"),
             {"normalized_shape": [5]}),
        norm(ln, "two_dims_f32", 822, "[2, 3, 4], normalized_shape [3, 4].", [2, 3, 4], ("weight", "bias"),
             {"normalized_shape": [3, 4]}),
        norm(ln, "no_params_f32", 823, "[4, 6], no weight and bias.", [4, 6], (), {"normalized_shape": [6]}),
        norm(ln, "no_bias_eps_f32", 824, "[3, 4], weight only, eps 1e-3.", [3, 4], ("weight",),
             {"normalized_shape": [4], "eps": 1e-3}),
        norm(ln, "permuted_f32", 825, "a permuted view [2, 3, 4], normalized_shape [4].", [2, 3, 4], ("weight", "bias"),
             {"normalized_shape": [4]}),
    ]
    gn = "nn.group_norm"
    cases += [
        norm(gn, "groups2_f32", 831, "[2, 4, 3, 3], 2 groups.", [2, 4, 3, 3], ("weight", "bias"), {"num_groups": 2}),
        norm(gn, "one_group_f32", 832, "[2, 3, 5], 1 group.", [2, 3, 5], ("weight", "bias"), {"num_groups": 1}),
        norm(gn, "instance_no_params_f32", 833, "[2, 3, 4], 3 groups (one per channel), no params.", [2, 3, 4], (),
             {"num_groups": 3}),
        norm(gn, "permuted_f32", 834, "a permuted view [2, 4, 3], 2 groups.", [2, 4, 3], ("weight", "bias"),
             {"num_groups": 2}),
    ]
    rn = "nn.rms_norm"
    cases += [
        norm(rn, "last_dim_f32", 841, "[3, 5], normalized_shape [5], weight.", [3, 5], ("weight",),
             {"normalized_shape": [5], "eps": 1e-6}),
        norm(rn, "two_dims_f32", 842, "[2, 3, 4], normalized_shape [3, 4].", [2, 3, 4], ("weight",),
             {"normalized_shape": [3, 4], "eps": 1e-5}),
        norm(rn, "no_weight_f32", 843, "[4, 6], no weight.", [4, 6], (), {"normalized_shape": [6], "eps": 1e-6}),
        norm(rn, "permuted_f32", 844, "a permuted view [2, 3, 4], normalized_shape [4].", [2, 3, 4], ("weight",),
             {"normalized_shape": [4], "eps": 1e-6}),
    ]
    return cases


def activation_cases() -> list[GradCase]:
    cases: list[GradCase] = []

    def act(op: str, name: str, seed: int, description: str, x: Builder, attrs: dict[str, Any] | None = None,
            **kw: Any) -> GradCase:
        fn, text = gen_ops.ACTIVATIONS[op]
        a = attrs or {}
        kw["notes"] = nn_notes(name, kw.get("notes", ""))
        return unary(op, name, seed, description, x, lambda x: fn(x, **a), text, attributes=a, **kw)

    for k, op in enumerate(gen_ops.ACTIVATIONS):
        base = 900 + 20 * k
        cases += [
            act(op, "random_f32", base + 1, "[3, 4, 5] uniform in [-4, 4).", rnd([3, 4, 5], -4.0, 4.0)),
            act(op, "permuted_f32", base + 2, "a permuted view [2, 3, 4].", view("permuted", [2, 3, 4])),
        ]
    zeros = [0.0, -0.0, 1.5, -1.5, 0.0, -0.0, 0.25, -0.25]
    tail_note = ("atol 0 in both tolerances (stricter than the defaults, which would accept 0 here): outputs and "
                 "gradients are tiny but normal. Values around -3 (softplus) and -2.25 (mish) sit on both sides of "
                 "where Polypous switches from a series to the plain formula.")
    tails = gen_ops.TAIL_VALUES
    cases += [
        act("nn.relu", "zero_f32", 903, "x = 0 and -0: gradient 0 there.", vals(zeros), spec=spec_unary(spec_relu),
            notes=SPEC_NOTES + "relu'(0) = relu'(-0) = 0."),
        act("nn.leaky_relu", "zero_f32", 923, "x = 0 and -0, default negative_slope 0.01.", vals(zeros),
            spec=spec_unary(spec_leaky_relu(0.01)), notes=SPEC_NOTES + "negative_slope · g at x = 0 and -0."),
        act("nn.leaky_relu", "zero_slope_0_25_f32", 924, "x = 0 and -0, negative_slope 0.25.", vals(zeros),
            {"negative_slope": 0.25}, spec=spec_unary(spec_leaky_relu(0.25)),
            notes=SPEC_NOTES + "negative_slope · g at x = 0 and -0."),
        act("nn.gelu", "tails_f32", 945, "|x| from 3 to 10: the gradient vanishes below and tends to g above.",
            vals([-10.0, -8.0, -6.0, -5.0, -4.0, -3.5, -3.0, 3.0, 4.0, 6.0, 10.0]),
            notes="PyTorch's f32 gradient loses relative accuracy below x = -5 (1 + erf cancels), so these values use "
            "the default tolerances."),
        act("nn.gelu_tanh", "tails_f32", 965, "|x| from 3 to 10: the gradient vanishes below and tends to g above.",
            vals([-10.0, -8.0, -6.0, -5.0, -4.0, -3.0, 3.0, 4.0, 6.0, 10.0]),
            notes="PyTorch's f32 gradient loses relative accuracy below x = -4 (1 + tanh cancels), so these values use "
            "the default tolerances."),
        act("nn.silu", "tail_f32", 985, "far negative x.", vals(tails), tolerance=TAIL, gradient_tolerance=TAIL_GRADIENT,
            notes=tail_note.split(" Values")[0]),
        act("nn.sigmoid", "tail_f32", 1003, "far negative x: tiny but normal outputs and gradients.",
            vals([-85.0, -80.0, -50.0, -30.0, -20.0, -10.0, -3.0, 0.0]), tolerance=TAIL,
            gradient_tolerance=TAIL_GRADIENT, notes=tail_note.split(" Values")[0]),
        act("nn.softplus", "zero_f32", 1023, "βx = 0 at x = 0 and -0: g / 2.", vals(zeros),
            spec=spec_unary(spec_softplus(1.0, 20.0)), notes=SPEC_NOTES + "g / 2 (= σ(0)) at βx = 0."),
        act("nn.softplus", "zero_beta2_f32", 1024, "βx = 0 with beta 2.", vals(zeros), {"beta": 2.0},
            spec=spec_unary(spec_softplus(2.0, 20.0)), notes=SPEC_NOTES + "g / 2 (= σ(0)) at βx = 0."),
        act("nn.softplus", "threshold_f32", 1025, "beta 2, threshold 5: βx = 5 (formula), 5.5 and above (g).",
            vals([2.5, 2.75, 3.0, 10.0, 2.25, -2.5]), {"beta": 2.0, "threshold": 5.0},
            spec=spec_unary(spec_softplus(2.0, 5.0)),
            notes=SPEC_NOTES + "g above the threshold (βx > threshold); at βx = threshold the formula g · σ(βx)."),
        act("nn.softplus", "default_threshold_f32", 1026, "beta 1, threshold 20: x = 19.5, 20, 20.5, 30.",
            vals([19.5, 20.0, 20.5, 30.0]), spec=spec_unary(spec_softplus(1.0, 20.0)),
            notes=SPEC_NOTES + "g above the threshold (βx > threshold)."),
        act("nn.softplus", "tail_f32", 1027, "far negative x: the gradient is σ(x), tiny but normal.", vals(tails),
            tolerance=TAIL, gradient_tolerance=TAIL_GRADIENT, notes=tail_note),
        act("nn.mish", "tail_f32", 1043, "far negative x: tiny but normal outputs and gradients.", vals(tails),
            tolerance=TAIL, gradient_tolerance=TAIL_GRADIENT, notes=tail_note),
    ]
    return cases


def build_cases() -> list[GradCase]:
    return (elementwise_cases() + shape_cases() + reduction_cases() + layer_cases() + pool_cases() + norm_cases()
            + activation_cases())


# ---------------------------------------------------------------------------------------------
# Computing and writing.


def roles(c: GradCase) -> tuple[list[str], list[str], list[str]]:
    """The case's tensor names split into inputs, params and mask (in that order in `tensors`)."""
    names = [n for n, _ in c.tensors]
    params = list(c.params or ())
    mask = ["mask"] if "mask" in names else []
    inputs = [n for n in names if n not in params and n not in mask]
    if names != inputs + params + mask:
        raise SystemExit(f"{c.op}/{c.name}: tensors must be the inputs, then the params in order, then the mask")
    return inputs, params, mask


def differentiable(c: GradCase, built: dict[str, Any]) -> list[str]:
    """Every float input in order, then every param but the running statistics (graph-ir.md §3.1.11)."""
    inputs, params, _ = roles(c)
    return [n for n in inputs if built[n].is_floating_point()] + [p for p in params if p not in RUNNING_STATS]


def run(c: GradCase, built: dict[str, Any], upstream: dict[str, Any], dtype: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The forward call and `torch.autograd.grad` in `dtype`: outputs by port, gradients by name, and
    the running statistics after the call (batch_norm in training mode)."""
    diff = differentiable(c, built)
    d: dict[str, Any] = {}
    for n, t in built.items():
        if t.is_floating_point():
            v = t.detach().to(dtype)
            # Fresh storage (batch_norm updates its running statistics in place), except for a view,
            # which the f32 run feeds as it is.
            v = v.clone() if v.is_contiguous() else v
            d[n] = v.requires_grad_() if n in diff else v
        else:
            d[n] = t
    res = c.fn(d)
    outs = res if isinstance(res, dict) else {"out": res}
    ports = [p for p, t in outs.items() if t.is_floating_point()]
    grads = torch.autograd.grad([outs[p] for p in ports], [d[n] for n in diff], [upstream[p].to(dtype) for p in ports],
                                allow_unused=True)
    if any(g is None for g in grads):
        raise SystemExit(f"{c.op}/{c.name}: an input gets no gradient")
    running = {}
    if c.op == "nn.batch_norm" and c.mode == "training":
        running = {"new_running_mean": d["running_mean"].detach(), "new_running_var": d["running_var"].detach()}
    return {p: t.detach() for p, t in outs.items()}, {f"grad_{n}": g for n, g in zip(diff, grads)}, running


def meets(actual: Any, expected: Any, tol: dict[str, Any]) -> bool:
    """§3.2's comparison: exact is bit for bit (any NaN matches any NaN); otherwise atol + rtol · |expected|,
    NaN where expected has NaN, the same inf where it has one."""
    if not actual.is_floating_point():
        return torch.equal(actual, expected)
    if tol["exact"]:
        a, e = actual.float(), expected.float()
        both_nan = a.isnan() & e.isnan()
        return bool(((a.view(torch.int32) == e.view(torch.int32)) | both_nan).all())
    return torch.allclose(actual.double(), expected.double(), atol=tol["atol"], rtol=tol["rtol"], equal_nan=True)


def agrees(actual: Any, expected: Any) -> bool:
    """PyTorch's f64 result against a spec formula in f64: the same NaNs and infinities, finite values
    within rounding."""
    return torch.allclose(actual, expected.to(actual.dtype), atol=1e-300, rtol=1e-12, equal_nan=True)


def worst(actual: Any, expected: Any) -> float:
    diff = (actual.double() - expected.double()).abs()
    return float(diff.nan_to_num(nan=float("inf")).max().item()) if diff.numel() else 0.0


def compute(c: GradCase) -> dict[str, Any]:
    """Builds the case's tensors, its expected values and the checks of the module docstring."""
    rng = np.random.Generator(np.random.PCG64(c.seed))
    built: dict[str, Any] = {}
    for n, b in c.tensors:
        built[n] = b(rng, built)
    for n, t in built.items():
        if t.is_floating_point() and t.dtype != torch.float32:
            raise SystemExit(f"{c.op}/{c.name}: {n} is {t.dtype}; gradient fixtures are f32")
        if t.is_floating_point() and not torch.isfinite(t).all():
            raise SystemExit(f"{c.op}/{c.name}: {n} is not finite; gradients at non-finite inputs are unspecified")
    with torch.no_grad():
        probe = c.fn({n: (t.double() if t.is_floating_point() else t) for n, t in built.items()})
    probe = probe if isinstance(probe, dict) else {"out": probe}
    upstream = {}
    for port, t in probe.items():
        if t.is_floating_point():
            up = (c.upstream or {}).get(port)
            upstream[port] = up(rng, built) if up else upstream_values(rng, list(t.shape))
    outs64, grads64, run64 = run(c, built, upstream, torch.float64)
    outs32, grads32, run32 = run(c, built, upstream, torch.float32)
    tol = c.tolerance or op_tolerance(c.op)
    expected_outs = {p: (t.float() if t.is_floating_point() else t) for p, t in outs64.items()}
    expected_grads = {n: g.float() for n, g in grads64.items()}
    expected_run = {n: t.float() for n, t in run64.items()}
    where = f"{c.op}/{c.name}"
    for p in expected_outs:
        if not meets(outs32[p], expected_outs[p], tol):
            raise SystemExit(f"{where}: PyTorch's f32 output {p} misses the tolerance (max abs err {worst(outs32[p], expected_outs[p])})")
    for n in expected_run:
        if not meets(run32[n], expected_run[n], tol):
            raise SystemExit(f"{where}: PyTorch's f32 {n} misses the tolerance (max abs err {worst(run32[n], expected_run[n])})")
    for n in expected_grads:
        if not meets(grads32[n], expected_grads[n], c.gradient_tolerance):
            raise SystemExit(f"{where}: PyTorch's f32 {n} misses the gradient tolerance "
                             f"(max abs err {worst(grads32[n], expected_grads[n])})")
    if c.spec is not None:
        d64 = {n: (t.double() if t.is_floating_point() else t) for n, t in built.items()}
        want = c.spec(d64, {p: u.double() for p, u in upstream.items()})
        got = grads64 | run64
        for n, w in want.items():
            if not agrees(got[n], w):
                raise SystemExit(f"{where}: PyTorch 2.15.0 disagrees with graph-ir.md §3.1.11 on {n}:\n"
                                 f"  inputs   {[(k, v.tolist()) for k, v in built.items()]}\n"
                                 f"  upstream {[(k, v.tolist()) for k, v in upstream.items()]}\n"
                                 f"  PyTorch  {got[n].tolist()}\n  spec     {w.tolist()}")
    return {"built": built, "upstream": upstream, "outs": expected_outs, "running": expected_run, "grads": expected_grads,
            "tolerance": tol}


def entry(name: str, t: Any, **extra: Any) -> dict[str, Any]:
    e: dict[str, Any] = {"name": name}
    e |= extra
    e |= {"dtype": gen_ops.dtype_name(t), "shape": list(t.shape)}
    return e


def reference_text(c: GradCase, diff: list[str], ports: list[str]) -> str:
    wrt = "(" + ", ".join(diff) + ("," if len(diff) == 1 else "") + ")"
    ups = [f"grad_{p}" for p in ports]
    up_text = ups[0] if len(ups) == 1 else "(" + ", ".join(ups) + ")"
    text = f"torch.autograd.grad({c.call}, {wrt}, {up_text}) in f64, rounded once to f32"
    if c.op == "nn.batch_norm" and c.mode == "training":
        text += "; new_running_mean and new_running_var: running_mean and running_var after the call"
    return text


def write_case(c: GradCase, out_dir: str, versions: dict[str, str]) -> int:
    """Writes one case; returns its size in bytes."""
    r = compute(c)
    built, upstream = r["built"], r["upstream"]
    inputs, params, mask = roles(c)
    diff = differentiable(c, built)
    ports = list(upstream)
    tensors: list[tuple[str, Any]] = [(n, built[n]) for n in inputs + params + mask]
    tensors += [(f"grad_{p}", upstream[p]) for p in ports]
    tensors += [(port_output_name(p), t) for p, t in r["outs"].items()]
    tensors += list(r["running"].items()) + list(r["grads"].items())
    for n, t in tensors:
        if t.is_floating_point():
            d = t.double()
            if ((d != 0) & (d.abs() < torch.finfo(torch.float32).tiny)).any():
                raise SystemExit(f"{c.op}/{c.name}: tensor {n} has subnormal values (graph-ir.md §3.1.8)")
    names = [n for n, _ in tensors]
    if len(set(names)) != len(names):
        raise SystemExit(f"{c.op}/{c.name}: tensor names are not unique: {names}")
    if not c.name.endswith("_f32"):
        raise SystemExit(f"{c.op}/{c.name}: gradient case names end in _f32")
    blob = safetensors.torch.save({n: t.contiguous().clone() for n, t in tensors}, metadata=None)
    sidecar: dict[str, Any] = {
        "schema_version": fixturelib.GRAD_SCHEMA_VERSION,
        "op": c.op,
        "case": c.name,
        "description": c.description,
        "mode": c.mode,
        "attributes": c.attributes,
        "inputs": [entry(n, built[n]) for n in inputs],
    }
    if c.params is not None:
        sidecar["params"] = [entry(n, built[n]) for n in params]
    if mask:
        sidecar["mask"] = entry("mask", built["mask"])
    sidecar["upstream"] = [entry(f"grad_{p}", upstream[p], port=p) for p in ports]
    sidecar["outputs"] = [entry(port_output_name(p), t) for p, t in r["outs"].items()]
    if r["running"]:
        sidecar["running_stats"] = [entry(n, t, param=n.removeprefix("new_")) for n, t in r["running"].items()]
    sidecar |= {
        "gradients": [entry(f"grad_{n}", r["grads"][f"grad_{n}"], of=n) for n in diff],
        "tolerance": r["tolerance"],
        "gradient_tolerance": c.gradient_tolerance,
        "reference": reference_text(c, diff, ports),
        "notes": " ".join(n for n in (c.notes, NOTES) if n),
        "seed": c.seed,
        "generator": GENERATOR,
        "versions": versions,
    }
    op_dir = os.path.join(out_dir, c.op)
    os.makedirs(op_dir, exist_ok=True)
    text = fixturelib.canonical_json(sidecar)
    with open(os.path.join(op_dir, c.name + ".safetensors"), "wb") as f:
        f.write(blob)
    with open(os.path.join(op_dir, c.name + ".json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return len(blob) + len(text.encode("utf-8"))


def generate(out_dir: str) -> int:
    cases = build_cases()
    keys = [(c.op, c.name) for c in cases]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate case names")
    seeds = [c.seed for c in cases]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("duplicate seeds")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    versions = gen_ops.installed_versions()
    for c in cases:
        write_case(c, out_dir, versions)
    return len(cases)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: fixtures/small/grads)")
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
    mismatch = gen_ops.version_problems()
    if mismatch and not args.allow_version_mismatch:
        for line in mismatch:
            print(f"error: {line}", file=sys.stderr)
        return 2
    if not args.check:
        gen_ops.remove_stale(args.out)
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
