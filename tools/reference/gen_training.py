#!/usr/bin/env python3
"""Generates the initial-value fixtures (fixtures/small/init/) and the training-mode graph fixtures
(fixtures/small/training/).

    python tools/reference/gen_training.py            # regenerate both trees
    python tools/reference/gen_training.py --check    # fail if the committed fixtures differ
    python tools/reference/gen_training.py --report   # print the measured errors behind the tolerances

Schemas: docs/testing-strategy.md §3.9 (initial values) and §3.10 (training-mode graphs).

- **Initial values** come from `polyrng.py`, a Python implementation of graph-ir.md §2.4.1 and
  §3.1.12 written from the spec (f64, rounded once to f32): every `init` kind on params of each fan
  role, the table case of §3.1.12 among them.
- **Training-mode graphs**: `linear → relu → linear`, and a graph with `conv2d`, `batch_norm`,
  `dropout` and a `repeat` whose body drops in each iteration. Weights, inputs and upstream gradients
  come from NumPy's PCG64 with fixed seeds. The reference is PyTorch in f64 on the f32 weights and
  inputs, rounded once to f32: the outputs, the new running statistics (`F.batch_norm(...,
  training=True)` updates them) and the gradient of every trainable param of the loss
  `Σ out · grad_out` (testing-strategy.md §3.6, "Comparing"). Dropout multiplies by the Polypous keep
  mask of the case's `seed`, `step` and node path (`polyrng.dropout_mask`), as `x * (mask / (1 - p))`
  (graph-ir.md §3.3). Before writing a case the generator checks that PyTorch's own f32 result meets
  the case's tolerances.

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
import polyrng  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_INIT_OUT = os.path.join(REPO, "fixtures", "small", "init")
DEFAULT_TRAINING_OUT = os.path.join(REPO, "fixtures", "small", "training")
GENERATOR = "tools/reference/gen_training.py"

try:
    import numpy as np
    import safetensors.torch
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover - depends on the environment
    np = safetensors = torch = F = None  # type: ignore[assignment]
    IMPORT_ERROR: ImportError | None = e
else:
    IMPORT_ERROR = None

# Tolerances (docs/testing-strategy.md §2): outputs and running statistics as f32 model outputs
# (the reductions/matmul class), gradients as f32 gradients.
OUTPUT_TOLERANCE = {"exact": False, "atol": 1e-5, "rtol": 1e-3}
GRADIENT_TOLERANCE = {"exact": False, "atol": 1e-5, "rtol": 1e-3}


def save_tensors(path: str, tensors: dict[str, Any]) -> None:
    # No metadata: nothing machine- or time-dependent ends up in the file.
    with open(path, "wb") as f:
        f.write(safetensors.torch.save(tensors, metadata=None))


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(fixturelib.canonical_json(obj))


def check_no_subnormals(where: str, tensors: dict[str, Any]) -> None:
    for name, t in tensors.items():
        if not t.is_floating_point():
            continue
        d = t.double()
        if ((d != 0) & (d.abs() < torch.finfo(t.dtype).tiny)).any():
            raise SystemExit(f"{where}: tensor {name} has subnormal values (graph-ir.md §3.1.8)")


def entry(name: str, t: Any, **extra: Any) -> dict[str, Any]:
    dtype = {torch.float32: "f32", torch.bool: "bool"}[t.dtype]
    return {"name": name} | extra | {"dtype": dtype, "shape": list(t.shape)}


# ---------------------------------------------------------------------------------------------
# Initial values (testing-strategy.md §3.9).


@dataclasses.dataclass(frozen=True)
class InitCase:
    case: str
    seed: int
    name: str
    shape: list[int]
    init: dict[str, Any]
    # The (op, op param) using the param, or None for an unused param.
    role: tuple[str, str] | None
    description: str


# Params of each fan role (graph-ir.md §2.4.1, "Fans"): name, shape, role.
FAN_ROLES = [
    ("linear_weight", "fc.weight", [16, 12], ("nn.linear", "weight")),
    ("embedding_weight", "emb.weight", [20, 8], ("nn.embedding", "weight")),
    ("conv1d_weight", "conv1.weight", [6, 2, 3], ("nn.conv1d", "weight")),
    ("conv2d_weight", "conv2.weight", [4, 3, 3, 3], ("nn.conv2d", "weight")),
]
# Roles without fans, for the kinds that need none, and an unused param.
OTHER_ROLES = [
    ("linear_weight", "fc.weight", [16, 12], ("nn.linear", "weight")),
    ("linear_bias", "fc.bias", [16], ("nn.linear", "bias")),
    ("batch_norm_running_var", "bn.running_var", [8], ("nn.batch_norm", "running_var")),
    ("layer_norm_weight", "ln.weight", [3, 4], ("nn.layer_norm", "weight")),
    ("unused", "spare", [5, 7], None),
]


def init_cases() -> list[InitCase]:
    cases: list[InitCase] = []
    seed = 100
    for kind, init in [
        ("kaiming_uniform", {"kind": "kaiming_uniform", "a": math.sqrt(5.0)}),
        ("kaiming_normal", {"kind": "kaiming_normal"}),
        ("xavier_uniform", {"kind": "xavier_uniform"}),
        ("xavier_normal", {"kind": "xavier_normal"}),
    ]:
        for case, name, shape, role in FAN_ROLES:
            seed += 1
            cases.append(InitCase(case, seed, name, shape, init, role, f"{kind} of a {role[0]} {role[1]} {shape}"))
    cases.append(
        InitCase(
            "fc1_weight_seed42",
            42,
            "fc1.weight",
            [128, 784],
            {"kind": "kaiming_uniform"},
            ("nn.linear", "weight"),
            "kaiming_uniform (a = 0) of fc1.weight [128, 784], seed 42: the known values of graph-ir.md §3.1.12",
        )
    )
    for kind, init in [
        ("zeros", {"kind": "zeros"}),
        ("ones", {"kind": "ones"}),
        ("constant", {"kind": "constant", "value": 0.1}),
        ("uniform", {"kind": "uniform", "low": -0.3, "high": 0.7}),
        ("normal", {"kind": "normal", "mean": 0.5, "std": 2.0}),
    ]:
        for case, name, shape, role in OTHER_ROLES:
            seed += 1
            where = f"a {role[0]} {role[1]}" if role else "an unused param"
            cases.append(InitCase(case, seed, name, shape, init, role, f"{kind} of {where} {shape}"))
    seed += 1
    cases += [
        InitCase("equal_bounds", seed, "u", [6], {"kind": "uniform", "low": 0.25, "high": 0.25}, None,
                 "uniform with low = high: a constant"),
        InitCase("zero_std", seed + 1, "n", [6], {"kind": "normal", "mean": -1.5, "std": 0.0}, None,
                 "normal with std = 0: a constant"),
        InitCase("overflow", seed + 2, "n", [64], {"kind": "normal", "mean": 0.0, "std": 3.0e38}, None,
                 "normal with std 3e38: values beyond the f32 range round to +-inf (graph-ir.md §2.4.1)"),
    ]
    return cases


def write_init_case(c: InitCase, out_dir: str, versions: dict[str, str]) -> None:
    kind = c.init["kind"]
    fans = polyrng.fans_of(c.role[0], c.role[1], c.shape) if c.role else None
    if kind in polyrng.FAN_KINDS and fans is None:
        raise SystemExit(f"{kind}/{c.case}: a fan-based kind needs a role with fans")
    values = polyrng.initial_values(c.seed, c.name, c.shape, c.init, fans)
    t = torch.tensor(values, dtype=torch.float64).to(torch.float32).reshape(c.shape)
    # Python's floats are the f32 values already: the conversion is exact.
    if [float(v) for v in t.flatten().tolist()] != values:
        raise SystemExit(f"{kind}/{c.case}: values changed when stored as f32")
    tensors = {"values": t.contiguous()}
    check_no_subnormals(f"init/{kind}/{c.case}", tensors)
    normal_based = kind in ("normal", "kaiming_normal", "xavier_normal")
    sidecar = {
        "schema_version": 1,
        "kind": kind,
        "case": c.case,
        "description": c.description,
        "seed": c.seed,
        "param": {
            "name": c.name,
            "shape": c.shape,
            "init": c.init,
            "role": {"op": c.role[0], "param": c.role[1]} if c.role else None,
            "fans": {"fan_in": fans[0], "fan_out": fans[1]} if fans else None,
        },
        "outputs": [entry("values", t)],
        "tolerance": {"ulp": 1 if normal_based and c.init.get("std") != 0.0 else 0},
        "reference": "tools/reference/polyrng.py initial_values: graph-ir.md §2.4.1 in f64, rounded once to f32",
        "notes": "ln and cos are the platform's, so normal-based values may differ by 1 ulp (graph-ir.md §3.1.12)"
        if normal_based
        else "bit-exact on every platform (graph-ir.md §3.1.12)",
        "generator": GENERATOR,
        "versions": versions,
    }
    d = os.path.join(out_dir, kind)
    os.makedirs(d, exist_ok=True)
    save_tensors(os.path.join(d, c.case + ".safetensors"), tensors)
    write_json(os.path.join(d, c.case + ".json"), sidecar)


# ---------------------------------------------------------------------------------------------
# Training-mode graphs (testing-strategy.md §3.10).


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: list[int]
    low: float
    high: float
    trainable: bool = True


@dataclasses.dataclass(frozen=True)
class TrainModel:
    name: str
    description: str
    graph: dict[str, Any]
    params: list[ParamSpec]
    input_shape: list[int]  # without the batch dim
    weight_seed: int
    # forward(params by name, x, masks by node path) -> (y, running stats by param name, in order)
    forward: Callable[[dict[str, Any], Any, dict[str, Any]], tuple[Any, list[tuple[str, Any]]]]
    # Dropout nodes: (node path, p), in the order they run.
    dropouts: list[tuple[str, float]]


@dataclasses.dataclass(frozen=True)
class TrainCase:
    model: str
    name: str
    batch: int
    seed: int  # input data and upstream gradient
    train_seed: int
    step: int


def linear_node(node_id: str, x: str, prefix: str = "", bias: bool = True) -> dict[str, Any]:
    params = {"weight": f"{prefix}{node_id}.weight"}
    if bias:
        params["bias"] = f"{prefix}{node_id}.bias"
    return {"id": node_id, "op": "nn.linear", "inputs": {"x": x}, "params": params}


def graph_doc(name: str, input_shape: list[int], params: list[ParamSpec], nodes: list[Any], out: str) -> dict[str, Any]:
    return {
        "format": "polypous.graph",
        "format_version": 1,
        "name": name,
        "opsets": {"core": 1, "nn": 1},
        "dims": {"batch": {}},
        "inputs": [{"name": "x", "dtype": "float", "shape": ["batch"] + input_shape}],
        "outputs": [{"name": "y", "value": out}],
        "params": [
            {"name": p.name, "dtype": "float", "shape": p.shape} | ({} if p.trainable else {"trainable": False})
            for p in params
        ],
        "nodes": nodes,
    }


def mlp_model() -> TrainModel:
    params = [
        ParamSpec("fc1.weight", [5, 6], -0.4, 0.4),
        ParamSpec("fc1.bias", [5], -0.2, 0.2, trainable=False),
        ParamSpec("fc2.weight", [3, 5], -0.45, 0.45),
        ParamSpec("fc2.bias", [3], -0.2, 0.2),
    ]
    nodes = [
        linear_node("fc1", "x"),
        {"id": "act", "op": "nn.relu", "inputs": {"x": "fc1"}},
        linear_node("fc2", "act"),
    ]

    def forward(p: dict[str, Any], x: Any, masks: dict[str, Any]) -> tuple[Any, list[tuple[str, Any]]]:
        h = F.relu(F.linear(x, p["fc1.weight"], p["fc1.bias"]))
        return F.linear(h, p["fc2.weight"], p["fc2.bias"]), []

    return TrainModel(
        "mlp",
        "linear 6->5 (bias trainable: false), relu, linear 5->3, in training mode.",
        graph_doc("mlp", [6], params, nodes, "fc2"),
        params,
        [6],
        301,
        forward,
        [],
    )


BN_MOMENTUM = 0.2
BN_EPS = 1e-5
P_DROP1 = 0.25
P_BLOCK = 0.5


def bn_dropout_model() -> TrainModel:
    params = [
        ParamSpec("conv.weight", [3, 2, 3, 3], -0.3, 0.3),
        ParamSpec("conv.bias", [3], -0.1, 0.1),
        ParamSpec("bn.running_mean", [3], -0.1, 0.1),
        ParamSpec("bn.running_var", [3], 0.5, 1.5),
        ParamSpec("bn.weight", [3], 0.5, 1.5),
        ParamSpec("bn.bias", [3], -0.2, 0.2),
        ParamSpec("fc.weight", [6, 48], -0.15, 0.15),
        ParamSpec("fc.bias", [6], -0.1, 0.1),
        ParamSpec("blocks.0.lin.weight", [6, 6], -0.4, 0.4),
        ParamSpec("blocks.0.lin.bias", [6], -0.1, 0.1),
        ParamSpec("blocks.1.lin.weight", [6, 6], -0.4, 0.4),
        ParamSpec("blocks.1.lin.bias", [6], -0.1, 0.1),
        ParamSpec("head.weight", [3, 6], -0.4, 0.4),
        ParamSpec("head.bias", [3], -0.1, 0.1),
    ]
    nodes = [
        {"id": "conv", "op": "nn.conv2d", "inputs": {"x": "x"},
         "params": {"weight": "conv.weight", "bias": "conv.bias"}, "attrs": {"padding": [1, 1]}},
        {"id": "bn", "op": "nn.batch_norm", "inputs": {"x": "conv"},
         "params": {"running_mean": "bn.running_mean", "running_var": "bn.running_var",
                    "weight": "bn.weight", "bias": "bn.bias"},
         "attrs": {"eps": BN_EPS, "momentum": BN_MOMENTUM}},
        {"id": "act", "op": "nn.relu", "inputs": {"x": "bn"}},
        {"id": "drop1", "op": "nn.dropout", "inputs": {"x": "act"}, "attrs": {"p": P_DROP1}},
        {"id": "flat", "op": "core.flatten", "inputs": {"x": "drop1"}, "attrs": {"start_dim": 1}},
        linear_node("fc", "flat"),
        {"id": "blocks", "op": "repeat", "count": 2, "carry": {"h": "fc"}, "body": {
            "params_prefix": "blocks.{i}.",
            "nodes": [
                linear_node("lin", "h"),
                {"id": "relu", "op": "nn.relu", "inputs": {"x": "lin"}},
                {"id": "drop", "op": "nn.dropout", "inputs": {"x": "relu"}, "attrs": {"p": P_BLOCK}},
            ],
            "carry_out": {"h": "drop"},
        }},
        linear_node("head", "blocks:h"),
    ]

    def forward(p: dict[str, Any], x: Any, masks: dict[str, Any]) -> tuple[Any, list[tuple[str, Any]]]:
        h = F.conv2d(x, p["conv.weight"], p["conv.bias"], padding=1)
        mean, var = p["bn.running_mean"].detach().clone(), p["bn.running_var"].detach().clone()
        h = F.batch_norm(h, mean, var, p["bn.weight"], p["bn.bias"], training=True, momentum=BN_MOMENTUM, eps=BN_EPS)
        h = F.relu(h)
        h = h * (masks["drop1"] / (1 - P_DROP1))
        h = F.linear(h.flatten(1), p["fc.weight"], p["fc.bias"])
        for i in range(2):
            h = F.relu(F.linear(h, p[f"blocks.{i}.lin.weight"], p[f"blocks.{i}.lin.bias"]))
            h = h * (masks[f"blocks[{i}]/drop"] / (1 - P_BLOCK))
        y = F.linear(h, p["head.weight"], p["head.bias"])
        return y, [("bn.running_mean", mean), ("bn.running_var", var)]

    return TrainModel(
        "bn_dropout",
        "conv2d 2->3 (3x3, padding 1), batch_norm (momentum 0.2), relu, dropout 0.25, flatten, linear 48->6, "
        "a repeat of 2 iterations (blocks.{i}.: linear 6->6, relu, dropout 0.5), linear 6->3, in training mode.",
        graph_doc("bn_dropout", [2, 4, 4], params, nodes, "head"),
        params,
        [2, 4, 4],
        302,
        forward,
        [("drop1", P_DROP1), ("blocks[0]/drop", P_BLOCK), ("blocks[1]/drop", P_BLOCK)],
    )


def train_models() -> dict[str, TrainModel]:
    return {m.name: m for m in (mlp_model(), bn_dropout_model())}


TRAIN_CASES = [
    TrainCase("mlp", "batch4_f32", 4, 401, 11, 0),
    TrainCase("bn_dropout", "batch4_step3_f32", 4, 402, 1234, 3),
    TrainCase("bn_dropout", "batch4_step4_f32", 4, 403, 1234, 4),
]


def uniform(rng: Any, shape: list[int], low: float, high: float) -> Any:
    return torch.from_numpy(rng.uniform(low, high, size=shape).astype(np.float32))


def model_weights(m: TrainModel) -> dict[str, Any]:
    rng = np.random.default_rng(m.weight_seed)
    return {p.name: uniform(rng, p.shape, p.low, p.high) for p in m.params}


@dataclasses.dataclass
class Measured:
    abs: float
    rel: float
    ratio: float


def measure(actual: Any, expected: Any, tol: dict[str, Any]) -> Measured:
    a, e = actual.double(), expected.double()
    err = (a - e).abs()
    allowed = tol["atol"] + tol["rtol"] * e.abs()
    return Measured(float(err.max()), float((err / e.abs().clamp_min(1e-30)).max()), float((err / allowed).max()))


def run_reference(m: TrainModel, weights: dict[str, Any], x: Any, g: Any, masks: dict[str, Any], dtype: Any):
    """(y, running stats, gradients by trainable param name) of the loss Σ y · g, in `dtype`."""
    trainable = [p.name for p in m.params if p.trainable and not p.name.endswith(("running_mean", "running_var"))]
    params = {}
    for name, w in weights.items():
        t = w.to(dtype)
        params[name] = t.requires_grad_(True) if name in trainable else t
    y, stats = m.forward(params, x.to(dtype), {k: v.to(dtype) for k, v in masks.items()})
    loss = (y * g.to(dtype)).sum()
    grads = torch.autograd.grad(loss, [params[n] for n in trainable])
    return y.detach(), [(n, s.detach()) for n, s in stats], dict(zip(trainable, grads))


def write_train_model(m: TrainModel, out_dir: str, versions: dict[str, str], report: list[str] | None) -> int:
    weights = model_weights(m)
    d = os.path.join(out_dir, m.name)
    os.makedirs(d, exist_ok=True)
    check_no_subnormals(f"{m.name}/model", weights)
    save_tensors(os.path.join(d, "model.safetensors"), {k: v.contiguous() for k, v in weights.items()})
    write_json(os.path.join(d, "model.graph.json"), m.graph)
    cases = [c for c in TRAIN_CASES if c.model == m.name]
    for c in cases:
        rng = np.random.default_rng(c.seed)
        x = uniform(rng, [c.batch] + m.input_shape, -1.0, 1.0)
        masks_bool: dict[str, Any] = {}
        shape_of = {"drop1": [c.batch, 3, 4, 4]}
        for path, p in m.dropouts:
            shape = shape_of.get(path, [c.batch, 6])
            keep = polyrng.dropout_mask(c.train_seed, c.step, path, shape, p)
            masks_bool[path] = torch.tensor(keep, dtype=torch.bool).reshape(shape)
        masks = {k: v.to(torch.float64) for k, v in masks_bool.items()}
        y64, _, _ = run_reference(m, weights, x, torch.zeros(1), masks, torch.float64)
        g = uniform(rng, list(y64.shape), -1.0, 1.0)
        y, stats, grads = run_reference(m, weights, x, g, masks, torch.float64)
        y32, stats32, grads32 = run_reference(m, weights, x, g, masks, torch.float32)
        expected = {"y": y.float()} | {n: s.float() for n, s in stats} | {n: t.float() for n, t in grads.items()}
        native = {"y": y32} | dict(stats32) | grads32
        for key, want in expected.items():
            tol = GRADIENT_TOLERANCE if key in grads else OUTPUT_TOLERANCE
            mm = measure(native[key], want, tol)
            if mm.ratio > 1.0:
                raise SystemExit(f"{m.name}/{c.name}: PyTorch f32 misses the tolerance on {key} (worst/allowed {mm.ratio:.3f})")
            if report is not None:
                report.append(
                    f"{m.name}/{c.name} {key}: max|expected| {float(want.abs().max()):.3e}; torch f32: "
                    f"abs {mm.abs:.2e} rel {mm.rel:.2e} worst/allowed {mm.ratio:.3f}"
                )
        tensors: dict[str, Any] = {"x": x, "y": expected["y"]}
        running = [entry(f"new/{n}", expected[n], param=n) for n, _ in stats]
        tensors |= {f"new/{n}": expected[n] for n, _ in stats}
        mask_entries = [entry(f"mask/{path}", masks_bool[path], node=path) for path, _ in m.dropouts]
        tensors |= {f"mask/{path}": masks_bool[path] for path, _ in m.dropouts}
        tensors["grad/y"] = g
        gradients = [entry(f"grad/{n}", expected[n], param=n) for n in grads]
        tensors |= {f"grad/{n}": expected[n] for n in grads}
        tensors = {k: v.contiguous() for k, v in tensors.items()}
        check_no_subnormals(f"{m.name}/{c.name}", tensors)
        sidecar = {
            "schema_version": 2,
            "model": m.name,
            "case": c.name,
            "description": f"{m.description} Batch {c.batch}, seed {c.train_seed}, step {c.step}.",
            "compute_dtype": "f32",
            "mode": "training",
            "training": {"seed": c.train_seed, "step": c.step},
            "inputs": [entry("x", x)],
            "outputs": [entry("y", expected["y"])],
            "running_stats": running,
            "masks": mask_entries,
            "upstream": [entry("grad/y", g, output="y")],
            "gradients": gradients,
            "tolerance": OUTPUT_TOLERANCE,
            "gradient_tolerance": GRADIENT_TOLERANCE,
            "reference": "the graph as PyTorch functional ops in f64 on the f32 weights and input, rounded once to f32; "
            "gradients: torch.autograd.grad((y * grad_y).sum(), trainable params)",
            "notes": "Dropout multiplies by x * (mask / (1 - p)) with the Polypous keep mask of `training.seed`, "
            "`training.step` and the node path (graph-ir.md §3.1.12, polyrng.dropout_mask), stored under `masks`. "
            "batch_norm runs F.batch_norm(..., training=True); `running_stats` are its updated running statistics. "
            "Gradients are listed for every trainable param (trainable: true and not a running statistic) in "
            "declaration order; the other params get none.",
            "seed": c.seed,
            "generator": GENERATOR,
            "versions": versions,
        }
        save_tensors(os.path.join(d, c.name + ".safetensors"), tensors)
        write_json(os.path.join(d, c.name + ".json"), sidecar)
    return len(cases)


# ---------------------------------------------------------------------------------------------


def generate(init_out: str, training_out: str, report: list[str] | None = None) -> tuple[int, int]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    versions = gen_ops.installed_versions()
    cases = init_cases()
    keys = [(c.init["kind"], c.case) for c in cases]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate init case names")
    for c in cases:
        write_init_case(c, init_out, versions)
    n_train = sum(write_train_model(m, training_out, versions, report) for m in train_models().values())
    return len(cases), n_train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init-out", default=DEFAULT_INIT_OUT, help="default: fixtures/small/init")
    parser.add_argument("--training-out", default=DEFAULT_TRAINING_OUT, help="default: fixtures/small/training")
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
        for out in (args.init_out, args.training_out):
            gen_ops.remove_stale(out)
        n_init, n_train = generate(args.init_out, args.training_out, report)
        print(f"wrote {n_init} init cases to {args.init_out} and {n_train} training cases to {args.training_out}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            init_tmp, train_tmp = os.path.join(tmp, "init"), os.path.join(tmp, "training")
            n_init, n_train = generate(init_tmp, train_tmp, report)
            problems = fixturelib.compare_trees(args.init_out, init_tmp)
            problems += fixturelib.compare_trees(args.training_out, train_tmp)
        for line in problems:
            print(line, file=sys.stderr)
        if problems:
            print(f"{len(problems)} problem(s); run python {GENERATOR} and commit the result", file=sys.stderr)
            return 1
        print(f"ok: {n_init} init cases and {n_train} training cases match")
    if report is not None:
        print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
