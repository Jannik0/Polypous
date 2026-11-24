#!/usr/bin/env python3
"""Generates the training component fixtures (fixtures/small/train/<component>/): the optimizers,
gradient clipping, the learning-rate schedules and the losses of docs/specs/training.md §3.1–§3.3.

    python tools/reference/gen_train.py            # regenerate fixtures/small/train/
    python tools/reference/gen_train.py --check    # fail if the committed fixtures differ
    python tools/reference/gen_train.py --report   # print the measured errors behind the tolerances

Schema: docs/testing-strategy.md §3.7 (training component fixtures, schema version 1). Tolerances:
§2 ("Optimizer tolerance", gradient clipping, "Schedule tolerance", loss values and gradients).

- **Optimizers** (`optimizer.sgd`, `optimizer.adam`, `optimizer.adamw`): 3 steps of
  `torch.optim.SGD`/`Adam`/`AdamW` with `foreach=False` (the single-tensor algorithm) and a
  constant `lr`, from a param and 3 gradients drawn from NumPy's PCG64 (scale ≤ 5, `lr ≤ 0.1`).
  The expected values are PyTorch's own f32 results: the param and every state tensor after each
  step (training.md §5.2's state names). Before writing a case, the generator runs **training.md
  §3.1's op sequence without FMA** (NumPy, f32: one rounding per op, scalars computed in f64 and
  rounded once) and checks that it meets the case's tolerance against PyTorch, so a fixture never
  asks for more than the spec's algorithm can give.
- **Gradient clipping** (`gradient_clip`): `torch.nn.utils.clip_grad_norm_` (norm type 2,
  `foreach=False`) on several gradients, in f64 on the f32 gradients and rounded once to f32
  (testing-strategy.md §3.3), including non-finite totals. The generator checks that PyTorch's own
  f32 result and §3.1.4's formula in NumPy f32 both meet the tolerance.
- **Schedules** (`schedule.<kind>`): `lr(k)` for `k = 0 … T−1` from PyTorch's schedulers stepped
  once per optimizer step (training.md §3.2's reference column: `StepLR`, `LambdaLR`,
  `CosineAnnealingLR`'s closed form). The generator checks that §3.2's formulas, evaluated in
  Python floats as written, meet the schedule tolerance (exact, or `1e-14 · lr` for `cosine`).

- **Losses** (`loss.mse`, `loss.cross_entropy`, `loss.binary_cross_entropy`; the `with_logits`
  cases are in `loss.binary_cross_entropy` with `"with_logits": true`): `F.mse_loss`,
  `F.cross_entropy` and `F.binary_cross_entropy(_with_logits)` (reduction `mean`) and their
  gradient with respect to the prediction (upstream gradient 1, `torch.autograd.grad`), in f64 on
  the f32 inputs and rounded once to f32 (testing-strategy.md §3.3, §3.7). Class indices are
  stored as `I64` (§3.7); the Rust test runs every `cross_entropy` case with them as `i64` and
  converted to `i32`. The generator checks that PyTorch's own f32 result meets the tolerance, and
  that training.md §3.3's formulas (the loss, and for `binary_cross_entropy` the gradient
  `(x − y) / max(x · (1 − x), 1e-12) / N`), evaluated in NumPy f64, agree with PyTorch's f64
  result, so the fixtures and the spec can't drift apart.

Needs the pinned versions from requirements.txt; `--check` refuses to compare otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import tempfile
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixturelib  # noqa: E402
import gen_ops  # noqa: E402  (version pins)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO, "fixtures", "small", "train")
GENERATOR = "tools/reference/gen_train.py"

try:
    import numpy as np
    import safetensors.torch
    import torch
except ImportError as e:  # pragma: no cover - depends on the environment
    np = safetensors = torch = None  # type: ignore[assignment]
    IMPORT_ERROR: ImportError | None = e
else:
    IMPORT_ERROR = None

# Tolerances (docs/testing-strategy.md §2).
OPTIMIZER_TOLERANCE = {"exact": False, "atol": 1e-6, "rtol": 1e-6}
CLIP_TOLERANCE = {"exact": False, "atol": 1e-5, "rtol": 1e-3}
# Loss values and f32 gradients (testing-strategy.md §2): one tolerance for both outputs.
LOSS_TOLERANCE = {"exact": False, "atol": 1e-5, "rtol": 1e-3}
EXACT = {"exact": True, "atol": 0, "rtol": 0}
COSINE_ATOL_PER_LR = 1e-14

# The state names of each optimizer, in output order (training.md §5.2).
STATE_NAMES = {"sgd": ("momentum_buffer",), "adam": ("exp_avg", "exp_avg_sq"), "adamw": ("exp_avg", "exp_avg_sq")}
STEPS = 3
# Scale bounds of testing-strategy.md §2 ("Optimizer tolerance").
MAX_SCALE = 5.0
MAX_LR = 0.1

# The defaults of training.md §2.8 (PyTorch's), to fill in left-out config fields.
DEFAULTS = {
    "sgd": {"lr": 1e-3, "momentum": 0.0, "dampening": 0.0, "nesterov": False, "weight_decay": 0.0},
    "adam": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
    "adamw": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
}


def save_tensors(path: str, tensors: dict[str, Any]) -> None:
    # No metadata: nothing machine- or time-dependent ends up in the file.
    with open(path, "wb") as f:
        f.write(safetensors.torch.save(tensors, metadata=None))


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(fixturelib.canonical_json(obj))


def check_no_subnormals(where: str, tensors: dict[str, Any]) -> None:
    for name, t in tensors.items():
        d = t.double()
        finite = torch.isfinite(d)
        if ((d != 0) & finite & (d.abs() < torch.finfo(torch.float32).tiny)).any():
            raise SystemExit(f"{where}: tensor {name} has subnormal values (graph-ir.md §3.1.8)")


def entry(name: str, t: Any) -> dict[str, Any]:
    if t.dtype == torch.int64:
        return {"name": name, "dtype": "i64", "shape": list(t.shape)}
    if t.dtype != torch.float32:
        raise SystemExit(f"{name}: training component tensors are f32 (class indices i64)")
    return {"name": name, "dtype": "f32", "shape": list(t.shape)}


@dataclasses.dataclass
class Measured:
    abs: float
    ulp: int
    ratio: float


def ulp_distance(a: Any, b: Any) -> int:
    """The largest distance in f32 ulps between finite elements of two f32 arrays."""
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    key = lambda i: np.where(i < 0, -(i & 0x7FFFFFFF), i)  # noqa: E731
    finite = np.isfinite(a) & np.isfinite(b)
    d = np.abs(key(ia) - key(ib))[finite]
    return int(d.max()) if d.size else 0


def measure(actual: Any, expected: Any, tol: dict[str, Any]) -> Measured:
    """The worst error of `actual` against `expected` (NumPy f32 arrays) under testing-strategy.md
    §3.2's rule: NaN exactly where expected, an expected inf must be the same inf, every other
    element within `atol + rtol · |expected|`. A broken rule gives `ratio = inf`."""
    a, e = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if a.shape != e.shape:
        return Measured(math.inf, 0, math.inf)
    nan_a, nan_e = np.isnan(a), np.isnan(e)
    inf_e = np.isinf(e)
    if (nan_a != nan_e).any() or (inf_e & (a != e)).any():
        return Measured(math.inf, 0, math.inf)
    finite = ~nan_e & ~inf_e
    if not finite.any():
        return Measured(0.0, 0, 0.0)
    a64, e64 = a[finite].astype(np.float64), e[finite].astype(np.float64)
    err = np.abs(a64 - e64)
    allowed = tol["atol"] + tol["rtol"] * np.abs(e64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(allowed > 0, err / allowed, np.where(err > 0, math.inf, 0.0))
    return Measured(float(err.max()), ulp_distance(a[finite], e[finite]), float(ratio.max()))


# ---------------------------------------------------------------------------------------------
# Optimizers (training.md §3.1; testing-strategy.md §3.7).


@dataclasses.dataclass(frozen=True)
class OptimizerCase:
    kind: str
    name: str
    seed: int
    shape: tuple[int, ...]
    # The optimizer object of the sidecar's `config` as written (left-out fields take defaults).
    config: dict[str, Any]
    description: str


OPTIMIZER_CASES = [
    OptimizerCase("sgd", "defaults", 1, (4, 8), {"kind": "sgd"}, "SGD with every default (lr 1e-3, no momentum, no weight decay)"),
    OptimizerCase("sgd", "weight_decay", 2, (16,), {"kind": "sgd", "lr": 0.1, "weight_decay": 0.01},
                  "SGD without momentum, with weight decay"),
    OptimizerCase("sgd", "momentum", 3, (3, 2, 5), {"kind": "sgd", "lr": 0.1, "momentum": 0.9},
                  "SGD with momentum 0.9 and no dampening: the buffer is a copy of the gradient at step 1"),
    OptimizerCase("sgd", "momentum_dampening", 4, (4, 8),
                  {"kind": "sgd", "lr": 0.05, "momentum": 0.9, "dampening": 0.1, "weight_decay": 0.001},
                  "SGD with momentum, dampening and weight decay"),
    OptimizerCase("sgd", "nesterov", 5, (32,),
                  {"kind": "sgd", "lr": 0.1, "momentum": 0.9, "nesterov": True, "weight_decay": 0.01},
                  "SGD with Nesterov momentum and weight decay"),
    OptimizerCase("adam", "defaults", 11, (4, 8), {"kind": "adam"}, "Adam with every default (lr 1e-3, betas (0.9, 0.999), eps 1e-8)"),
    OptimizerCase("adam", "weight_decay", 12, (16,), {"kind": "adam", "lr": 0.01, "weight_decay": 0.01},
                  "Adam with L2 weight decay added to the gradient"),
    OptimizerCase("adam", "betas_eps", 13, (3, 2, 5),
                  {"kind": "adam", "lr": 0.05, "betas": [0.8, 0.99], "eps": 1e-6, "weight_decay": 0.0},
                  "Adam with non-default betas and eps"),
    OptimizerCase("adam", "small_beta1", 14, (32,),
                  {"kind": "adam", "lr": 0.1, "betas": [0.3, 0.9], "eps": 1e-4, "weight_decay": 0.1},
                  "Adam with beta1 0.3, so the lerp weight 1 - beta1 is >= 0.5 (ATen's second lerp formula)"),
    OptimizerCase("adamw", "defaults", 21, (4, 8), {"kind": "adamw"},
                  "AdamW with every default (lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight_decay 0.01)"),
    OptimizerCase("adamw", "weight_decay", 22, (16,), {"kind": "adamw", "lr": 0.01, "weight_decay": 0.1},
                  "AdamW with a large decoupled weight decay"),
    OptimizerCase("adamw", "betas_eps", 23, (3, 2, 5),
                  {"kind": "adamw", "lr": 0.1, "betas": [0.85, 0.995], "eps": 1e-7, "weight_decay": 0.05},
                  "AdamW with non-default betas and eps"),
    OptimizerCase("adamw", "zero_beta1", 24, (32,),
                  {"kind": "adamw", "lr": 0.02, "betas": [0.0, 0.5], "eps": 1e-8, "weight_decay": 0.0},
                  "AdamW with beta1 0 (the lerp weight is 1, so exp_avg is the gradient), beta2 0.5 and no weight decay"),
]


def full_config(c: OptimizerCase) -> dict[str, Any]:
    return DEFAULTS[c.kind] | {k: v for k, v in c.config.items() if k != "kind"}


def torch_optimizer(kind: str, cfg: dict[str, Any], params: list[Any]) -> tuple[Any, str]:
    """The PyTorch optimizer and its call as text (the sidecar's `reference`)."""
    if kind == "sgd":
        args = dict(lr=cfg["lr"], momentum=cfg["momentum"], dampening=cfg["dampening"],
                    weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"], foreach=False)
        cls = torch.optim.SGD
    else:
        args = dict(lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"],
                    weight_decay=cfg["weight_decay"], foreach=False)
        cls = torch.optim.Adam if kind == "adam" else torch.optim.AdamW
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return cls(params, **args), f"torch.optim.{cls.__name__}([p], {text})"


def emulate_optimizer(kind: str, cfg: dict[str, Any], p: Any, grads: list[Any]) -> list[dict[str, Any]]:
    """training.md §3.1's op sequence without FMA, in NumPy f32: every op rounds once, every scalar
    is computed in Python floats (f64) and rounded once to f32 (`F(...)`). Returns, per step, the
    param and the state by name."""
    F = np.float32
    out: list[dict[str, Any]] = []
    lr = cfg["lr"]
    state: dict[str, Any] = {}
    for s, g in enumerate(grads, start=1):
        g = g.copy()
        if kind == "sgd":
            wd, mom, damp = cfg["weight_decay"], cfg["momentum"], cfg["dampening"]
            if wd != 0:
                g = g + F(wd) * p
            if mom != 0:
                buf = state.get("momentum_buffer")
                buf = g.copy() if buf is None else buf * F(mom) + F(1 - damp) * g
                state["momentum_buffer"] = buf
                g = g + F(mom) * buf if cfg["nesterov"] else buf
            p = p + F(-lr) * g
        else:
            (b1, b2), eps, wd = cfg["betas"], cfg["eps"], cfg["weight_decay"]
            if wd != 0:
                if kind == "adamw":
                    p = p * F(1 - lr * wd)
                else:
                    g = g + F(wd) * p
            m = state.get("exp_avg", np.zeros_like(p))
            v = state.get("exp_avg_sq", np.zeros_like(p))
            w = F(1 - b1)
            m = m + w * (g - m) if abs(w) < 0.5 else g - (g - m) * F(1 - float(w))
            v = v * F(b2) + F(1 - b2) * (g * g)
            bias_correction1 = 1 - b1 ** float(s)
            bias_correction2 = 1 - b2 ** float(s)
            step_size = lr / bias_correction1
            denom = np.sqrt(v) / F(bias_correction2 ** 0.5) + F(eps)
            p = p + F(-step_size) * (m / denom)
            state = {"exp_avg": m, "exp_avg_sq": v}
        out.append({"param": p.copy()} | {k: t.copy() for k, t in state.items()})
    return out


def draw(rng: Any, shape: tuple[int, ...], scale: float) -> Any:
    v = np.clip(rng.standard_normal(size=shape) * scale, -MAX_SCALE, MAX_SCALE)
    return v.astype(np.float32)


def write_optimizer_case(c: OptimizerCase, out_dir: str, versions: dict[str, str], report: list[str] | None) -> None:
    cfg = full_config(c)
    if not cfg["lr"] <= MAX_LR:
        raise SystemExit(f"optimizer.{c.kind}/{c.name}: lr above {MAX_LR} needs its own tolerance (testing-strategy.md §2)")
    rng = np.random.Generator(np.random.PCG64(c.seed))
    p0 = draw(rng, c.shape, 2.0)
    grads = [draw(rng, c.shape, 1.0) for _ in range(STEPS)]
    param = torch.from_numpy(p0.copy()).requires_grad_(False)
    opt, reference = torch_optimizer(c.kind, cfg, [param])
    expected: list[dict[str, Any]] = []
    for g in grads:
        param.grad = torch.from_numpy(g.copy())
        opt.step()
        st = opt.state[param]
        record = {"param": param.detach().clone()}
        for name in STATE_NAMES[c.kind]:
            if st.get(name) is not None:
                record[name] = st[name].detach().clone()
        expected.append(record)
    emulated = emulate_optimizer(c.kind, cfg, p0.copy(), grads)
    where = f"optimizer.{c.kind}/{c.name}"
    worst = Measured(0.0, 0, 0.0)
    for s, (want, got) in enumerate(zip(expected, emulated), start=1):
        if set(want) != set(got):
            raise SystemExit(f"{where}: step {s}: state {sorted(want)} in PyTorch, {sorted(got)} in the emulation")
        for name, t in want.items():
            m = measure(got[name], t.numpy(), OPTIMIZER_TOLERANCE)
            if m.ratio > 1.0:
                raise SystemExit(
                    f"{where}: training.md §3.1's op sequence misses the optimizer tolerance on {name} after "
                    f"step {s} (abs {m.abs:.3e}, {m.ulp} ulp, worst/allowed {m.ratio:.3f})"
                )
            if m.ratio >= worst.ratio:
                worst = Measured(max(worst.abs, m.abs), max(worst.ulp, m.ulp), m.ratio)
            else:
                worst = Measured(max(worst.abs, m.abs), max(worst.ulp, m.ulp), worst.ratio)
    if report is not None:
        report.append(
            f"{where}: §3.1 without FMA vs torch f32, worst over params and state: abs {worst.abs:.2e}, "
            f"{worst.ulp} ulp, worst/allowed {worst.ratio:.3f}"
        )
    tensors: dict[str, Any] = {"param": torch.from_numpy(p0)}
    tensors |= {f"grad_{s}": torch.from_numpy(g) for s, g in enumerate(grads, start=1)}
    inputs = [entry(n, t) for n, t in tensors.items()]
    outputs = []
    for s, record in enumerate(expected, start=1):
        for name in ("param",) + STATE_NAMES[c.kind]:
            if name in record:
                key = f"{name}_{s}"
                tensors[key] = record[name].contiguous()
                outputs.append(entry(key, tensors[key]))
    check_no_subnormals(where, tensors)
    component = f"optimizer.{c.kind}"
    d = os.path.join(out_dir, component)
    os.makedirs(d, exist_ok=True)
    sidecar = {
        "schema_version": 1,
        "component": component,
        "case": c.name,
        "description": f"{c.description}. Shape {list(c.shape)}, {STEPS} steps with a constant lr.",
        "config": c.config,
        "inputs": inputs,
        "outputs": outputs,
        "tolerance": OPTIMIZER_TOLERANCE,
        "reference": reference,
        "notes": "Step s gives the gradient grad_s to the param; param_s and the state after it (training.md §5.2's "
        "names) are PyTorch's own f32 results (testing-strategy.md §2, \"Optimizer tolerance\"). Config fields left out "
        "take training.md §2.8's defaults. The global step counter k is s - 1, and lr(k) is the config's lr.",
        "seed": c.seed,
        "generator": GENERATOR,
        "versions": versions,
    }
    save_tensors(os.path.join(d, c.name + ".safetensors"), tensors)
    write_json(os.path.join(d, c.name + ".json"), sidecar)


# ---------------------------------------------------------------------------------------------
# Gradient clipping (training.md §3.1.4).


@dataclasses.dataclass(frozen=True)
class ClipCase:
    name: str
    seed: int
    shapes: tuple[tuple[int, ...], ...]
    max_norm: float
    description: str
    # Flat index of an element to replace in grad_0 (non-finite totals), or None.
    special: float | None = None


CLIP_CASES = [
    ClipCase("clipped", 31, ((4, 5), (7,), (2, 3, 2)), 1.0, "Three gradients with a total norm far above max_norm"),
    ClipCase("not_clipped", 32, ((4, 5), (7,)), 100.0, "A total norm below max_norm: c = 1, the gradients are multiplied by 1"),
    ClipCase("single", 33, ((64,),), 0.5, "One gradient"),
    ClipCase("inf", 34, ((4, 5), (7,)), 1.0, "A gradient with +inf: the total is inf, c = 0, finite elements become 0 and the inf NaN", math.inf),
    ClipCase("nan", 35, ((4, 5), (7,)), 1.0, "A gradient with NaN: the total and every clipped element are NaN", math.nan),
]


def torch_clip(grads: list[Any], max_norm: float, dtype: Any) -> tuple[Any, list[Any]]:
    ps = []
    for g in grads:
        p = torch.zeros(g.shape, dtype=dtype, requires_grad=True)
        p.grad = g.to(dtype).clone()
        ps.append(p)
    total = torch.nn.utils.clip_grad_norm_(ps, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=False)
    return total.reshape(1), [p.grad for p in ps]


def emulate_clip(grads: list[Any], max_norm: float) -> tuple[Any, list[Any]]:
    """training.md §3.1.4's formula in NumPy f32."""
    F = np.float32
    with np.errstate(invalid="ignore", over="ignore"):
        norms = np.array([np.sqrt(np.sum(g * g, dtype=np.float32)) for g in grads], dtype=np.float32)
        total = np.sqrt(np.sum(norms * norms, dtype=np.float32))
        c = np.minimum(F(max_norm) / (total + F(1e-6)), F(1))
        return np.array([total], dtype=np.float32), [g * c for g in grads]


def write_clip_case(c: ClipCase, out_dir: str, versions: dict[str, str], report: list[str] | None) -> None:
    rng = np.random.Generator(np.random.PCG64(c.seed))
    grads = [draw(rng, shape, 1.0) for shape in c.shapes]
    if c.special is not None:
        grads[0].reshape(-1)[3] = np.float32(c.special)
    tg = [torch.from_numpy(g) for g in grads]
    total64, clipped64 = torch_clip(tg, c.max_norm, torch.float64)
    expected = {"total_norm": total64.float()} | {f"clipped_{i}": g.float() for i, g in enumerate(clipped64)}
    total32, clipped32 = torch_clip(tg, c.max_norm, torch.float32)
    native = {"total_norm": total32} | {f"clipped_{i}": g for i, g in enumerate(clipped32)}
    etotal, eclipped = emulate_clip(grads, c.max_norm)
    emulated = {"total_norm": etotal} | {f"clipped_{i}": g for i, g in enumerate(eclipped)}
    where = f"gradient_clip/{c.name}"
    for label, got in (("PyTorch f32", native), ("§3.1.4 in NumPy f32", emulated)):
        worst = 0.0
        for key, want in expected.items():
            g = got[key].numpy() if hasattr(got[key], "numpy") else got[key]
            m = measure(g, want.numpy(), CLIP_TOLERANCE)
            if m.ratio > 1.0:
                raise SystemExit(f"{where}: {label} misses the clipping tolerance on {key} (worst/allowed {m.ratio:.3f})")
            worst = max(worst, m.ratio)
        if report is not None:
            report.append(f"{where}: {label} vs f64, worst/allowed {worst:.2e} (total norm {float(total64[0]):.6g})")
    tensors = {f"grad_{i}": t for i, t in enumerate(tg)} | {k: v.contiguous() for k, v in expected.items()}
    check_no_subnormals(where, tensors)
    d = os.path.join(out_dir, "gradient_clip")
    os.makedirs(d, exist_ok=True)
    sidecar = {
        "schema_version": 1,
        "component": "gradient_clip",
        "case": c.name,
        "description": f"{c.description}; max_norm {c.max_norm}, shapes {[list(s) for s in c.shapes]}.",
        "config": {"max_norm": c.max_norm},
        "inputs": [entry(f"grad_{i}", t) for i, t in enumerate(tg)],
        "outputs": [entry(k, v) for k, v in expected.items()],
        "tolerance": CLIP_TOLERANCE,
        "reference": f"torch.nn.utils.clip_grad_norm_(params, max_norm={c.max_norm!r}, norm_type=2.0, "
        "error_if_nonfinite=False, foreach=False)",
        "notes": "grad_i is the gradient of param i; total_norm is the returned total norm and clipped_i the gradient "
        "after clipping. Computed in f64 on the f32 gradients and rounded once to f32 (testing-strategy.md §3.3); "
        "PyTorch's own f32 result and training.md §3.1.4's formula in f32 meet the tolerance. NaN must be NaN "
        "exactly where expected (testing-strategy.md §3.2).",
        "seed": c.seed,
        "generator": GENERATOR,
        "versions": versions,
    }
    save_tensors(os.path.join(d, c.name + ".safetensors"), tensors)
    write_json(os.path.join(d, c.name + ".json"), sidecar)


# ---------------------------------------------------------------------------------------------
# Schedules (training.md §3.2).


@dataclasses.dataclass(frozen=True)
class ScheduleCase:
    name: str
    lr: float
    total_steps: int
    # The schedule object with every field (training.md §2.8).
    schedule: dict[str, Any]
    description: str


SCHEDULE_CASES = [
    ScheduleCase("constant", 0.01, 5, {"kind": "constant"}, "No scheduler"),
    ScheduleCase("every3", 0.1, 10, {"kind": "step", "every": 3, "gamma": 0.1},
                 "Boundaries at k = 3, 6, 9; the value after three multiplications differs from one pow"),
    ScheduleCase("every1", 0.05, 12, {"kind": "step", "every": 1, "gamma": 0.7}, "A multiplication at every step"),
    ScheduleCase("gamma0", 0.1, 8, {"kind": "step", "every": 4, "gamma": 0.0}, "gamma = 0: the learning rate is 0 from step 4 on"),
    ScheduleCase("warmup4", 0.1, 10, {"kind": "linear_warmup", "warmup_steps": 4}, "Warmup over 4 steps, then lr"),
    ScheduleCase("warmup3", 0.01, 3, {"kind": "linear_warmup", "warmup_steps": 3}, "W = T: every step is a warmup step"),
    ScheduleCase("warmup_ge_total", 0.1, 4, {"kind": "linear_warmup", "warmup_steps": 7}, "W > T: lr is never reached"),
    ScheduleCase("warmup5", 0.1, 20, {"kind": "cosine", "warmup_steps": 5, "min_lr": 1e-4},
                 "Warmup over 5 steps, then a cosine to min_lr"),
    ScheduleCase("no_warmup", 0.003, 15, {"kind": "cosine", "warmup_steps": 0, "min_lr": 0.0},
                 "W = 0: a cosine from lr down to just above 0"),
    ScheduleCase("warmup_ge_total", 0.1, 6, {"kind": "cosine", "warmup_steps": 10, "min_lr": 0.01},
                 "W >= T: every step is a warmup step (T - W is never used)"),
    ScheduleCase("warmup1_min_eq_lr", 0.01, 5, {"kind": "cosine", "warmup_steps": 1, "min_lr": 0.01},
                 "min_lr = lr: constant after the one warmup step"),
]


def torch_schedule_values(c: ScheduleCase) -> tuple[list[float], str, float]:
    """`lr(k)` for k = 0 … T−1 from PyTorch, its reference text, and (cosine) the largest difference
    of PyTorch's chainable `SequentialLR([LinearLR, CosineAnnealingLR])` from them, in units of lr."""
    from torch.optim import lr_scheduler as S

    lr0, total, sch = c.lr, c.total_steps, c.schedule
    kind = sch["kind"]

    def run(make: Callable[[Any], Any] | None) -> list[float]:
        p = torch.zeros(1, requires_grad=True)
        opt = torch.optim.SGD([p], lr=lr0)
        sched = make(opt) if make else None
        values = []
        for _ in range(total):
            values.append(opt.param_groups[0]["lr"])
            opt.step()
            if sched is not None:
                sched.step()
        return values

    spread = 0.0
    if kind == "constant":
        return run(None), "no scheduler: torch.optim.SGD([p], lr) alone", spread
    if kind == "step":
        ref = f"StepLR(optimizer, step_size={sch['every']}, gamma={sch['gamma']!r}), stepped once per optimizer step"
        return run(lambda o: S.StepLR(o, step_size=sch["every"], gamma=sch["gamma"])), ref, spread
    w = sch["warmup_steps"]
    warm = lambda o: S.LambdaLR(o, lambda k: min(1, (k + 1) / w))  # noqa: E731
    if kind == "linear_warmup":
        ref = f"LambdaLR(optimizer, lambda k: min(1, (k + 1) / {w})), stepped once per optimizer step"
        return run(warm), ref, spread
    # cosine: LambdaLR's warmup for k < W, CosineAnnealingLR's closed form at last_epoch = k − W after.
    min_lr = sch["min_lr"]
    values = run(warm) if w > 0 else []
    values = values[: min(w, total)]
    if total > w:
        p = torch.zeros(1, requires_grad=True)
        cos = S.CosineAnnealingLR(torch.optim.SGD([p], lr=lr0), T_max=total - w, eta_min=min_lr)
        for k in range(w, total):
            cos.last_epoch = k - w
            values.append(cos._get_closed_form_lr()[0])
        # The chainable forms agree within f64 rounding (training.md §3.2); measured for --report.
        if w > 0:
            chained = run(lambda o: S.SequentialLR(o, [
                S.LinearLR(o, start_factor=1 / w, end_factor=1.0, total_iters=w - 1) if w > 1 else S.LambdaLR(o, lambda k: 1.0),
                S.CosineAnnealingLR(o, T_max=total - w, eta_min=min_lr)], milestones=[w]))
        else:
            chained = run(lambda o: S.CosineAnnealingLR(o, T_max=total, eta_min=min_lr))
        spread = max(abs(a - b) for a, b in zip(chained, values)) / lr0
    ref = (f"k < {w}: LambdaLR(optimizer, lambda k: min(1, (k + 1) / {w})); k >= {w}: CosineAnnealingLR(optimizer, "
           f"T_max={total - w}, eta_min={min_lr!r})._get_closed_form_lr() at last_epoch = k - {w}")
    return values, ref, spread


def spec_lr(c: ScheduleCase, k: int) -> float:
    """training.md §3.2's formulas, in Python floats (f64), evaluated left to right as written."""
    lr0, total, sch = c.lr, c.total_steps, c.schedule
    kind = sch["kind"]
    if kind == "constant":
        return lr0
    if kind == "step":
        v = lr0
        for _ in range(k // sch["every"]):
            v = v * sch["gamma"]
        return v
    w = sch["warmup_steps"]
    if k < w:
        return lr0 * ((k + 1) / w)
    if kind == "linear_warmup":
        return lr0
    min_lr = sch["min_lr"]
    return min_lr + (lr0 - min_lr) * (1 + math.cos(math.pi * (k - w) / (total - w))) / 2


def write_schedule_case(c: ScheduleCase, out_dir: str, versions: dict[str, str], report: list[str] | None) -> None:
    kind = c.schedule["kind"]
    component = f"schedule.{kind}"
    where = f"{component}/{c.name}"
    values, reference, spread = torch_schedule_values(c)
    if len(values) != c.total_steps:
        raise SystemExit(f"{where}: {len(values)} values for T = {c.total_steps}")
    tol = {"exact": False, "atol": COSINE_ATOL_PER_LR * c.lr, "rtol": 0} if kind == "cosine" else EXACT
    worst = 0.0
    for k, want in enumerate(values):
        got = spec_lr(c, k)
        err = abs(got - want)
        if (tol["exact"] and got != want) or err > tol["atol"]:
            raise SystemExit(f"{where}: training.md §3.2 gives {got!r} at k = {k}, PyTorch {want!r}")
        worst = max(worst, err)
    if report is not None:
        extra = f"; chainable SequentialLR/CosineAnnealingLR differ by up to {spread:.2e} · lr" if kind == "cosine" else ""
        report.append(f"{where}: §3.2 in f64 vs PyTorch, worst |error| {worst:.2e}{extra}")
    d = os.path.join(out_dir, component)
    os.makedirs(d, exist_ok=True)
    sidecar = {
        "schema_version": 1,
        "component": component,
        "case": c.name,
        "description": f"{c.description}. lr {c.lr!r}, T = {c.total_steps}.",
        "config": {"lr": c.lr, "total_steps": c.total_steps, "schedule": c.schedule},
        "inputs": [],
        "outputs": [],
        "values": values,
        "tolerance": tol,
        "reference": reference,
        "notes": "values[k] is the learning rate of optimizer step k (training.md §3.2): the optimizer's lr after k "
        "scheduler steps, the scheduler stepped once after every optimizer.step(). `lr` is the optimizer's lr and "
        "`total_steps` is T.",
        "seed": 0,
        "generator": GENERATOR,
        "versions": versions,
    }
    write_json(os.path.join(d, c.name + ".json"), sidecar)


# ---------------------------------------------------------------------------------------------
# Losses (training.md §3.3; testing-strategy.md §3.7).


@dataclasses.dataclass(frozen=True)
class LossCase:
    kind: str
    name: str
    seed: int
    # The loss object of the sidecar's `config` as written, without `prediction`/`target`
    # (left-out fields take training.md §2.6's defaults).
    config: dict[str, Any]
    # Draws the f32 prediction and the target (f32, or int64 class indices) from the case's PCG64.
    make: Callable[[Any], tuple[Any, Any]]
    description: str


def _normal(rng: Any, shape: tuple[int, ...], scale: float) -> Any:
    return (rng.standard_normal(size=shape) * scale).astype(np.float32)


def _classes(c: int, b: int, scale: float) -> Callable[[Any], tuple[Any, Any]]:
    return lambda rng: (_normal(rng, (b, c), scale), rng.integers(0, c, size=b).astype(np.int64))


def _large_logits(rng: Any) -> tuple[Any, Any]:
    # Differences of 0, 500, 1000 or 2000: every probability is 0, 1/2 or 1 in f64 (no subnormal
    # gradients), and a naive exp overflows.
    x = np.array([[1000, 0, -1000], [1000, 0, -1000], [-1000, 1000, 0], [500, 500, -500],
                  [-1000, -1000, -1000], [0, 1000, 1000]], dtype=np.float32)
    return x, np.array([2, 0, 1, 1, 2, 0], dtype=np.int64)


def _probabilities(shape: tuple[int, ...], hard: bool) -> Callable[[Any], tuple[Any, Any]]:
    def make(rng: Any) -> tuple[Any, Any]:
        x = rng.uniform(0.02, 0.98, size=shape).astype(np.float32)
        y = (rng.integers(0, 2, size=shape) if hard else rng.uniform(0, 1, size=shape)).astype(np.float32)
        return x, y
    return make


def _probability_edges(rng: Any) -> tuple[Any, Any]:
    # Exactly 0 and 1 (the −100 clamp), x · (1 − x) below 1e-12 (the gradient's bound), and next
    # to 1 (1 − x exact in f32).
    x = np.array([0, 1, 0, 1, 0, 1, 1e-13, 1e-13, 1 - 2**-24, 0.5, 1e-30, 0.25], dtype=np.float32)
    y = np.array([0, 1, 1, 0, 0.5, 0.25, 0, 0.3, 1, 0.5, 1, 0], dtype=np.float32)
    return x.reshape(3, 4), y.reshape(3, 4)


def _logits(shape: tuple[int, ...], scale: float) -> Callable[[Any], tuple[Any, Any]]:
    return lambda rng: (_normal(rng, shape, scale), rng.uniform(0, 1, size=shape).astype(np.float32))


def _large_binary_logits(rng: Any) -> tuple[Any, Any]:
    # ±200 with hard targets (losses 200 and 0); 0 (softplus' kink); ±20 (softplus' threshold);
    # ±30 and 100 with soft targets. No case has σ(x) − y in f32's subnormal range.
    x = np.array([200, -200, 200, -200, 0, 20, -20, 30, -30, 100, -0.5, 3], dtype=np.float32)
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0.5, 0.5, 0.25, 0, 1], dtype=np.float32)
    return x.reshape(2, 6), y.reshape(2, 6)


LOSS_CASES = [
    LossCase("mse", "random", 41, {"kind": "mse"}, lambda rng: (_normal(rng, (8, 5), 2.0), _normal(rng, (8, 5), 2.0)),
             "Prediction and target from N(0, 2²), [8, 5]"),
    LossCase("mse", "rank3", 42, {"kind": "mse"}, lambda rng: (_normal(rng, (4, 3, 2), 1.0), _normal(rng, (4, 3, 2), 1.0)),
             "A rank-3 prediction, [4, 3, 2]: the mean is over every element"),
    LossCase("mse", "batch1", 43, {"kind": "mse"}, lambda rng: (_normal(rng, (1,), 1.0), _normal(rng, (1,), 1.0)),
             "A batch of 1, shape [1]"),
    LossCase("cross_entropy", "random", 51, {"kind": "cross_entropy"}, _classes(10, 16, 3.0),
             "Logits from N(0, 3²), [16, 10], no label smoothing"),
    LossCase("cross_entropy", "smoothing_0_1", 52, {"kind": "cross_entropy", "label_smoothing": 0.1}, _classes(10, 16, 3.0),
             "Label smoothing 0.1, [16, 10]"),
    LossCase("cross_entropy", "smoothing_1", 53, {"kind": "cross_entropy", "label_smoothing": 1.0}, _classes(7, 8, 3.0),
             "Label smoothing 1: the target class has the same weight as every other, [8, 7]"),
    LossCase("cross_entropy", "large_logits", 54, {"kind": "cross_entropy"}, _large_logits,
             "Logits up to ±1000, [6, 3] (the stable log-sum-exp; [1000, 0, −1000] with class 2 gives 2000)"),
    LossCase("cross_entropy", "large_logits_smoothing", 55, {"kind": "cross_entropy", "label_smoothing": 0.1}, _large_logits,
             "Logits up to ±1000 with label smoothing 0.1, [6, 3]"),
    LossCase("cross_entropy", "one_class", 56, {"kind": "cross_entropy", "label_smoothing": 0.3}, _classes(1, 4, 3.0),
             "C = 1 with label smoothing 0.3: the loss and its gradient are 0, [4, 1]"),
    LossCase("cross_entropy", "batch1", 57, {"kind": "cross_entropy"}, _classes(5, 1, 2.0), "A batch of 1, [1, 5]"),
    LossCase("binary_cross_entropy", "random", 61, {"kind": "binary_cross_entropy"}, _probabilities((8, 4), hard=False),
             "Probabilities in [0.02, 0.98] and soft targets in [0, 1), [8, 4]"),
    LossCase("binary_cross_entropy", "hard_targets", 62, {"kind": "binary_cross_entropy"}, _probabilities((16,), hard=True),
             "Probabilities in [0.02, 0.98] and targets 0 or 1, [16]"),
    LossCase("binary_cross_entropy", "edges", 63, {"kind": "binary_cross_entropy"}, _probability_edges,
             "Probabilities of exactly 0 and 1 (the −100 clamp), 1e-13 (x · (1 − x) below 1e-12) and 1 − 2^-24, [3, 4]"),
    LossCase("binary_cross_entropy", "batch1", 64, {"kind": "binary_cross_entropy"}, _probabilities((1,), hard=False),
             "A batch of 1, shape [1]"),
    LossCase("binary_cross_entropy", "logits_random", 65, {"kind": "binary_cross_entropy", "with_logits": True},
             _logits((8, 4), 3.0), "Logits from N(0, 3²) and soft targets, [8, 4]"),
    LossCase("binary_cross_entropy", "logits_large", 66, {"kind": "binary_cross_entropy", "with_logits": True},
             _large_binary_logits, "Logits of ±200 (losses 200 and 0), 0, ±20, ±30 and 100, [2, 6]"),
    LossCase("binary_cross_entropy", "logits_batch1", 67, {"kind": "binary_cross_entropy", "with_logits": True},
             _logits((1, 1), 2.0), "A batch of 1, shape [1, 1]"),
]


def torch_loss(c: LossCase, x: Any, y: Any) -> tuple[Any, str]:
    """The reference call on tensors of any float dtype, and its text (the sidecar's `reference`)."""
    F = torch.nn.functional
    if c.kind == "mse":
        return F.mse_loss(x, y), "F.mse_loss(prediction, target)"
    if c.kind == "cross_entropy":
        eps = c.config.get("label_smoothing", 0.0)
        return F.cross_entropy(x, y, label_smoothing=eps), f"F.cross_entropy(prediction, target, label_smoothing={eps!r})"
    if c.config.get("with_logits", False):
        return F.binary_cross_entropy_with_logits(x, y), "F.binary_cross_entropy_with_logits(prediction, target)"
    return F.binary_cross_entropy(x, y), "F.binary_cross_entropy(prediction, target)"


def spec_loss(c: LossCase, x: Any, y: Any) -> tuple[float, Any | None]:
    """training.md §3.3's formula in NumPy f64 on the f32 inputs, and for `binary_cross_entropy` the
    spec's gradient formula (None for the others, whose gradient is the formula's derivative)."""
    x = x.astype(np.float64)
    if c.kind == "mse":
        return float(np.mean((x - y.astype(np.float64)) ** 2)), None
    if c.kind == "cross_entropy":
        eps = c.config.get("label_smoothing", 0.0)
        m = x.max(axis=1, keepdims=True)
        lse = (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))[:, 0]
        n, classes = x.shape
        picked = x[np.arange(n), y]
        per = (1 - eps) * (lse - picked) + (eps / classes) * (lse[:, None] - x).sum(axis=1)
        return float(per.mean()), None
    y = y.astype(np.float64)
    if c.config.get("with_logits", False):
        log_sigmoid = np.minimum(x, 0) - np.log1p(np.exp(-np.abs(x)))
        return float(np.mean((1 - y) * x - log_sigmoid)), None
    with np.errstate(divide="ignore"):
        per = -(y * np.maximum(np.log(x), -100) + (1 - y) * np.maximum(np.log(1 - x), -100))
    grad = (x - y) / np.maximum(x * (1 - x), 1e-12) / x.size
    return float(per.mean()), grad


def write_loss_case(c: LossCase, out_dir: str, versions: dict[str, str], report: list[str] | None) -> None:
    component = f"loss.{c.kind}"
    where = f"{component}/{c.name}"
    rng = np.random.Generator(np.random.PCG64(c.seed))
    x, y = c.make(rng)
    if x.dtype != np.float32 or (y.dtype != np.int64 if c.kind == "cross_entropy" else y.dtype != np.float32):
        raise SystemExit(f"{where}: the prediction is f32, the target f32 or int64 class indices")
    target = torch.from_numpy(y)

    def run(dtype: Any) -> tuple[Any, Any, str]:
        p = torch.from_numpy(x).to(dtype).requires_grad_(True)
        t = target if c.kind == "cross_entropy" else target.to(dtype)
        loss, text = torch_loss(c, p, t)
        (grad,) = torch.autograd.grad(loss, p)
        return loss.detach().reshape(1), grad, text

    loss64, grad64, call = run(torch.float64)
    expected = {"loss": loss64.float(), "grad_prediction": grad64.float()}
    loss32, grad32, _ = run(torch.float32)
    native = {"loss": loss32, "grad_prediction": grad32}
    worst = 0.0
    for key, want in expected.items():
        m = measure(native[key].numpy(), want.numpy(), LOSS_TOLERANCE)
        if m.ratio > 1.0:
            raise SystemExit(f"{where}: PyTorch f32 misses the loss tolerance on {key} (worst/allowed {m.ratio:.3f})")
        worst = max(worst, m.ratio)
    # The spec's formulas against PyTorch's f64 result.
    value, grad = spec_loss(c, x, y)
    spec = {"loss": np.array([value]), "grad_prediction": grad}
    spec_worst = 0.0
    for key, got in spec.items():
        if got is None:
            continue
        want = (loss64 if key == "loss" else grad64).numpy()
        err = np.abs(got - want)
        allowed = LOSS_TOLERANCE["atol"] + LOSS_TOLERANCE["rtol"] * np.abs(want)
        if not (err <= allowed).all():
            raise SystemExit(f"{where}: training.md §3.3's formula for {key} differs from PyTorch's f64 result "
                             f"(worst |error| {float(err.max()):.3e})")
        rel = err / np.maximum(np.abs(want), np.finfo(np.float64).tiny)
        spec_worst = max(spec_worst, float(np.where(err > 0, rel, 0).max()))
    if report is not None:
        report.append(f"{where}: PyTorch f32 vs f64, worst/allowed {worst:.2e}; §3.3 formula in f64 vs PyTorch f64, "
                      f"worst relative difference {spec_worst:.1e} (loss {float(loss64[0]):.6g})")
    tensors = {"prediction": torch.from_numpy(x), "target": target} | {k: v.contiguous() for k, v in expected.items()}
    check_no_subnormals(where, {k: v for k, v in tensors.items() if v.is_floating_point()})
    d = os.path.join(out_dir, component)
    os.makedirs(d, exist_ok=True)
    indices = (" The target holds class indices, stored as I64 (testing-strategy.md §3.7); the Rust test also runs "
               "the case with them converted to i32, which training.md §3.3 reads as i64.") if c.kind == "cross_entropy" else ""
    sidecar = {
        "schema_version": 1,
        "component": component,
        "case": c.name,
        "description": f"{c.description}.",
        "config": c.config,
        "inputs": [entry("prediction", tensors["prediction"]), entry("target", target)],
        "outputs": [entry(k, v) for k, v in expected.items()],
        "tolerance": LOSS_TOLERANCE,
        "reference": f"torch.autograd.grad({call}, prediction)",
        "notes": "loss is the batch mean (reduction \"mean\"), shape [1]; grad_prediction its gradient with respect to "
        "the prediction, upstream gradient 1. Both computed in f64 on the f32 inputs and rounded once to f32 "
        "(testing-strategy.md §3.3, §3.7); PyTorch's own f32 result meets the tolerance, and training.md §3.3's "
        "formulas in f64 agree with PyTorch's f64 result. The config is the loss object without prediction and "
        "target; left-out fields take training.md §2.6's defaults." + indices,
        "seed": c.seed,
        "generator": GENERATOR,
        "versions": versions,
    }
    save_tensors(os.path.join(d, c.name + ".safetensors"), tensors)
    write_json(os.path.join(d, c.name + ".json"), sidecar)


# ---------------------------------------------------------------------------------------------


def case_keys() -> list[tuple[str, str]]:
    return (
        [(f"optimizer.{c.kind}", c.name) for c in OPTIMIZER_CASES]
        + [("gradient_clip", c.name) for c in CLIP_CASES]
        + [(f"schedule.{c.schedule['kind']}", c.name) for c in SCHEDULE_CASES]
        + [(f"loss.{c.kind}", c.name) for c in LOSS_CASES]
    )


def generate(out_dir: str, report: list[str] | None = None) -> int:
    keys = case_keys()
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate case names")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    versions = gen_ops.installed_versions()
    for c in OPTIMIZER_CASES:
        write_optimizer_case(c, out_dir, versions, report)
    for c in CLIP_CASES:
        write_clip_case(c, out_dir, versions, report)
    for c in SCHEDULE_CASES:
        write_schedule_case(c, out_dir, versions, report)
    for c in LOSS_CASES:
        write_loss_case(c, out_dir, versions, report)
    return len(keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="default: fixtures/small/train")
    parser.add_argument("--check", action="store_true", help="regenerate into a temporary directory and compare")
    parser.add_argument("--report", action="store_true", help="also print the measured errors behind the tolerances")
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
    report: list[str] | None = [] if args.report else None
    if not args.check:
        gen_ops.remove_stale(args.out)
        n = generate(args.out, report)
        print(f"wrote {n} training component cases to {args.out}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            n = generate(tmp, report)
            problems = fixturelib.compare_trees(args.out, tmp)
        for line in problems:
            print(line, file=sys.stderr)
        if problems:
            print(f"{len(problems)} problem(s); run python {GENERATOR} and commit the result", file=sys.stderr)
            return 1
        print(f"ok: {n} training component cases match")
    if report is not None:
        print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
