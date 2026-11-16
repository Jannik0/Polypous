#!/usr/bin/env python3
"""Generates the end-to-end model fixtures in fixtures/small/models/ from PyTorch.

    python tools/reference/gen_models.py            # regenerate fixtures/small/models/
    python tools/reference/gen_models.py --check    # fail if the committed fixtures differ
    python tools/reference/gen_models.py --report   # print the measured errors behind the tolerances

Layout, case sidecar schema and tolerances: docs/testing-strategy.md §2 ("Model end-to-end") and
§3.1/§3.5. Each model is a PyTorch `nn.Sequential` with named layers (`MODELS`). The generator writes,
per model:

- `model.graph.json`: the graph (graph-ir.md), written from the module structure. Node ids are the
  layer names and param names are the `state_dict` keys (`fc1.weight`), so the PyTorch weights load
  unchanged. The batch dim is the symbol `batch`.
- `model.safetensors`: the params, f32, in PyTorch layouts (`state_dict` minus BatchNorm's
  `num_batches_tracked`, a training counter that inference doesn't read).
- per case `<case>.safetensors` (inputs and expected outputs, in the compute dtype) and
  `<case>.json` (the sidecar).

Weights and inputs come from NumPy's PCG64 with fixed seeds (the same values on every CPU and OS)
and are loaded into the modules. The reference is the module in f64, run on the weights and inputs
as the runtime sees them (the f32 weights, or for f16/bf16 cases the weights rounded once to that
dtype, runtime.md §2 step 5, and the inputs in the compute dtype), rounded once to the compute
dtype. Before writing a case the generator checks that two other computations meet the case's
tolerance against that reference: PyTorch's own result in the compute dtype, and an emulation of
the runtime's numerics (graph-ir.md §3.1.8: every layer in f64, its output rounded to the compute
dtype, subnormal results flushed to zero as a device may do). `--report` prints these errors.

Needs the pinned versions from requirements.txt; `--check` refuses to compare otherwise.
"""

from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import os
import sys
import tempfile
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixturelib  # noqa: E402
import gen_ops  # noqa: E402  (version pins, dtype names)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO, "fixtures", "small", "models")
GENERATOR = "tools/reference/gen_models.py"

try:
    import numpy as np
    import safetensors.torch
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover - depends on the environment
    np = safetensors = torch = nn = None  # type: ignore[assignment]
    IMPORT_ERROR: ImportError | None = e
else:
    IMPORT_ERROR = None

# ---------------------------------------------------------------------------------------------
# Tolerances (docs/testing-strategy.md §2, "Model end-to-end"), written into every sidecar. Why
# each number is what it is, and the measured errors behind them, is in testing-strategy.md §2.

# f32: the reductions/matmul class of the ops the models are made of (atol 1e-5, rtol 1e-3).
MODEL_F32 = {"exact": False, "atol": 1e-5, "rtol": 1e-3}
# f16/bf16: the op class's rtol 1e-2, which covers the final rounding (at most one ulp, 2u, between
# a device's rounding and the reference's), plus atol = 4u (u = unit roundoff, 2^-11 for f16 and 2^-8
# for bf16), rounded up, for outputs of scale <= 1 like these. Unlike one op, a model rounds every
# intermediate to the compute dtype (graph-ir.md §3.1.8); an output that sums many such terms has an
# absolute error of order u times the size of the terms, however close to zero the sum itself is,
# so with atol 0 a correct device would fail those elements.
MODEL_F16 = {"exact": False, "atol": 2e-3, "rtol": 1e-2}
MODEL_BF16 = {"exact": False, "atol": 2e-2, "rtol": 1e-2}
TOLERANCES = {"f32": MODEL_F32, "f16": MODEL_F16, "bf16": MODEL_BF16}

# ---------------------------------------------------------------------------------------------
# Models.


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    # Builds the layers (named, in order). The graph has one node per layer.
    layers: Callable[[], list[tuple[str, Any]]]
    input_name: str
    input_shape: list[int]  # without the batch dim
    # Seeds: params (`weight_seed`), BatchNorm statistics (`calibration_seed`).
    weight_seed: int
    calibration_seed: int
    # Extra outputs computed from the last layer's output: (output name, node id, module).
    extra_outputs: Callable[[], list[tuple[str, str, Any]]] = lambda: []
    output_name: str = "logits"
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    model: str
    name: str
    batch: int
    dtype: str
    seed: int


def mlp_layers() -> list[tuple[str, Any]]:
    return [("fc1", nn.Linear(784, 128)), ("relu", nn.ReLU()), ("fc2", nn.Linear(128, 10))]


def cnn_layers() -> list[tuple[str, Any]]:
    return [
        ("conv1", nn.Conv2d(1, 8, 3, padding=1)),
        ("relu1", nn.ReLU()),
        ("pool1", nn.MaxPool2d(2)),
        # No bias before a BatchNorm, as usual in PyTorch models (also covers an absent optional param).
        ("conv2", nn.Conv2d(8, 16, 3, padding=1, bias=False)),
        ("bn", nn.BatchNorm2d(16)),
        ("relu2", nn.ReLU()),
        # 14 -> 3: bins of different sizes that overlap (start floor(i*14/3), end ceil((i+1)*14/3)).
        ("pool2", nn.AdaptiveAvgPool2d((3, 3))),
        ("flatten", nn.Flatten(1)),
        ("fc", nn.Linear(16 * 3 * 3, 10)),
    ]


MODELS = {
    "mlp": ModelSpec(
        name="mlp",
        description="MLP 784 -> 128 -> 10 with ReLU; logits and their log_softmax.",
        layers=mlp_layers,
        input_name="image",
        input_shape=[784],
        weight_seed=46001,
        calibration_seed=46002,
        extra_outputs=lambda: [("log_probs", "log_softmax", nn.LogSoftmax(dim=-1))],
    ),
    "cnn": ModelSpec(
        name="cnn",
        description=(
            "Small CNN on 1x28x28 images: conv2d 1->8 (3x3, padding 1), relu, max_pool2d 2x2, conv2d 8->16 "
            "(3x3, padding 1, no bias), batch_norm (inference), relu, adaptive_avg_pool2d 3x3, flatten, linear 144->10."
        ),
        layers=cnn_layers,
        input_name="image",
        input_shape=[1, 28, 28],
        weight_seed=46011,
        calibration_seed=46012,
        notes=(
            "bn's running_mean and running_var are the per-channel mean and (unbiased) variance of conv2's output "
            "on a calibration batch of 32 images (calibration_seed in the generator), times 1.1 and 0.9 "
            "respectively, so they are non-trivial and the normalized values stay O(1); bn.weight is in [0.5, 1.5), "
            "bn.bias in [-0.5, 0.5)."
        ),
    ),
}

CASES = [
    CaseSpec("mlp", "batch1_f32", 1, "f32", 46101),
    CaseSpec("mlp", "batch4_f32", 4, "f32", 46102),
    CaseSpec("mlp", "batch1_f16", 1, "f16", 46103),
    CaseSpec("mlp", "batch4_f16", 4, "f16", 46104),
    CaseSpec("mlp", "batch1_bf16", 1, "bf16", 46105),
    CaseSpec("mlp", "batch4_bf16", 4, "bf16", 46106),
    CaseSpec("cnn", "batch1_f32", 1, "f32", 46111),
    CaseSpec("cnn", "batch4_f32", 4, "f32", 46112),
    CaseSpec("cnn", "batch1_f16", 1, "f16", 46113),
    CaseSpec("cnn", "batch4_f16", 4, "f16", 46114),
    CaseSpec("cnn", "batch1_bf16", 1, "bf16", 46115),
    CaseSpec("cnn", "batch4_bf16", 4, "bf16", 46116),
]

# ---------------------------------------------------------------------------------------------
# Graph export: one node per layer, from the module's type and attributes.


def pair(v: Any) -> list[int]:
    return list(v) if isinstance(v, (tuple, list)) else [v, v]


def layer_node(node_id: str, module: Any, source: str) -> tuple[dict[str, Any], list[str]]:
    """The graph node for one layer and the names of the params it uses (catalog order)."""
    params: dict[str, str] = {}
    attrs: dict[str, Any] = {}
    if isinstance(module, nn.Linear):
        op = "nn.linear"
        params = {"weight": "weight"} | ({"bias": "bias"} if module.bias is not None else {})
    elif isinstance(module, nn.Conv2d):
        if module.padding_mode != "zeros" or isinstance(module.padding, str):
            raise SystemExit(f"{node_id}: only numeric zero padding is exported")
        op = "nn.conv2d"
        params = {"weight": "weight"} | ({"bias": "bias"} if module.bias is not None else {})
        attrs = {"stride": pair(module.stride), "padding": pair(module.padding),
                 "dilation": pair(module.dilation), "groups": module.groups}
    elif isinstance(module, nn.MaxPool2d):
        if module.ceil_mode or module.return_indices:
            raise SystemExit(f"{node_id}: ceil_mode and return_indices are not exported")
        op = "nn.max_pool2d"
        attrs = {"kernel_size": pair(module.kernel_size), "stride": pair(module.stride or module.kernel_size),
                 "padding": pair(module.padding), "dilation": pair(module.dilation)}
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        op = "nn.adaptive_avg_pool2d"
        attrs = {"output_size": pair(module.output_size)}
    elif isinstance(module, nn.BatchNorm2d):
        if not (module.affine and module.track_running_stats):
            raise SystemExit(f"{node_id}: only affine BatchNorm with running statistics is exported")
        op = "nn.batch_norm"
        params = {"running_mean": "running_mean", "running_var": "running_var", "weight": "weight", "bias": "bias"}
        attrs = {"eps": module.eps}
    elif isinstance(module, nn.ReLU):
        op = "nn.relu"
    elif isinstance(module, nn.Flatten):
        op = "core.flatten"
        attrs = {"start_dim": module.start_dim, "end_dim": module.end_dim}
    elif isinstance(module, nn.LogSoftmax):
        op = "core.log_softmax"
        attrs = {"dim": module.dim}
    else:
        raise SystemExit(f"{node_id}: no export for {type(module).__name__}")
    node: dict[str, Any] = {"id": node_id, "op": op, "inputs": {"x": source}}
    if params:
        node["params"] = {slot: f"{node_id}.{p}" for slot, p in params.items()}
    if attrs:
        node["attrs"] = attrs
    return node, list(node.get("params", {}).values())


def build_graph(spec: ModelSpec, layers: list[tuple[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    nodes, used = [], []
    source = spec.input_name
    for node_id, module in layers:
        node, names = layer_node(node_id, module, source)
        nodes.append(node)
        used += names
        source = node_id
    outputs = [{"name": spec.output_name, "value": source}]
    for out_name, node_id, module in spec.extra_outputs():
        node, names = layer_node(node_id, module, source)
        nodes.append(node)
        used += names
        outputs.append({"name": out_name, "value": node_id})
    if sorted(used) != sorted(state):
        raise SystemExit(f"{spec.name}: graph params {sorted(used)} differ from the state_dict {sorted(state)}")
    opsets = sorted({n["op"].split(".")[0] for n in nodes})
    return {
        "format": "polypous.graph",
        "format_version": 1,
        "opsets": {o: 1 for o in opsets},
        "name": spec.name,
        "dims": {"batch": {"min": 1}},
        "inputs": [{"name": spec.input_name, "dtype": "float", "shape": ["batch"] + spec.input_shape}],
        "outputs": outputs,
        "params": [{"name": n, "dtype": "float", "shape": list(state[n].shape)} for n in used],
        "nodes": nodes,
    }


# ---------------------------------------------------------------------------------------------
# Weights and inputs.

DTYPES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}


def torch_dtype(name: str) -> Any:
    return getattr(torch, DTYPES[name])


def no_f16_subnormals(t: Any) -> Any:
    """Zeroes values whose magnitude is below the smallest normal f16 (2^-14). The f16 cases round
    the f32 weights to f16 (runtime.md §2 step 5), and a device may flush f16 subnormals to zero
    (graph-ir.md §3.1.8), so the fixtures keep them out of every dtype's weights and inputs."""
    return t.masked_fill(t.abs() < 2.0**-14, 0.0)


def uniform(rng: Any, shape: list[int], low: float, high: float) -> Any:
    """Uniform f32 values in [low, high), drawn in f64 and rounded once, without f16 subnormals."""
    values = rng.random(size=shape) * (high - low) + low
    return no_f16_subnormals(torch.from_numpy(values).to(torch.float32))


def build_model(spec: ModelSpec) -> tuple[Any, list[tuple[str, Any]]]:
    """The f32 module with the generator's weights, and its named layers (with the extra outputs)."""
    torch.manual_seed(spec.weight_seed)  # module construction draws from it; every value is replaced below
    layers = spec.layers()
    model = nn.Sequential(collections.OrderedDict(layers)).eval()
    rng = np.random.Generator(np.random.PCG64(spec.weight_seed))
    with torch.no_grad():
        for name, module in layers:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # PyTorch's default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias.
                fan_in = module.weight[0].numel()
                bound = 1.0 / fan_in**0.5
                module.weight.copy_(uniform(rng, list(module.weight.shape), -bound, bound))
                if module.bias is not None:
                    module.bias.copy_(uniform(rng, list(module.bias.shape), -bound, bound))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.copy_(uniform(rng, list(module.weight.shape), 0.5, 1.5))
                module.bias.copy_(uniform(rng, list(module.bias.shape), -0.5, 0.5))
        calibrate_batch_norm(spec, model)
    return model, layers


def calibrate_batch_norm(spec: ModelSpec, model: Any) -> None:
    """Sets every BatchNorm's running statistics from a calibration batch (see the CNN's notes)."""
    rng = np.random.Generator(np.random.PCG64(spec.calibration_seed))
    x = uniform(rng, [32] + spec.input_shape, 0.0, 1.0).double()
    for name, module in copy.deepcopy(model).double().named_children():
        if isinstance(module, nn.BatchNorm2d):
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=True)
            # The f32 module's own BatchNorm (same name), not the f64 copy.
            target = getattr(model, name)
            target.running_mean.copy_(no_f16_subnormals((mean * 1.1).float()))
            target.running_var.copy_(no_f16_subnormals((var * 0.9).float()))
            module.running_mean.copy_(target.running_mean.double())
            module.running_var.copy_(target.running_var.double())
        x = module(x)


def state_tensors(model: Any) -> dict[str, Any]:
    """The params as written to model.safetensors: f32, PyTorch layouts, `state_dict` names."""
    state = model.state_dict()
    return {k: v.detach().to(torch.float32).contiguous().clone() for k, v in state.items()
            if not k.endswith("num_batches_tracked")}


def case_input(spec: ModelSpec, case: CaseSpec) -> Any:
    """Image-like inputs in [0, 1), drawn in f64 and rounded once to the compute dtype."""
    rng = np.random.Generator(np.random.PCG64(case.seed))
    values = torch.from_numpy(rng.random(size=[case.batch] + spec.input_shape))
    return no_f16_subnormals(values).to(torch_dtype(case.dtype))


# ---------------------------------------------------------------------------------------------
# References and the checks behind the tolerances.


def run_layers(model: Any, extra: list[tuple[str, str, Any]], x: Any,
               after: Callable[[Any], Any] = lambda t: t) -> list[Any]:
    """Runs the layers in order, applying `after` to every layer's output; returns the outputs."""
    for module in model:
        x = after(module(x))
    outs = [x]
    for _, _, module in extra:
        outs.append(after(module(x)))
    return outs


def flush_subnormals(t: Any, dtype: Any) -> Any:
    return t.masked_fill(t.abs() < torch.finfo(dtype).tiny, 0.0)


@dataclasses.dataclass
class Measured:
    """Errors of one computation against the reference: max abs error, max rel error over elements
    with |expected| >= 1e-3, and the worst element's error over its allowed error (<= 1 passes)."""

    abs: float
    rel: float
    ratio: float


def measure(actual: Any, expected: Any, tol: dict[str, Any]) -> Measured:
    a, e = actual.double(), expected.double()
    err = (a - e).abs()
    big = e.abs() >= 1e-3
    rel = (err[big] / e.abs()[big]).max().item() if big.any() else 0.0
    ratio = (err / (tol["atol"] + tol["rtol"] * e.abs())).max().item()
    return Measured(err.max().item(), rel, ratio)


def compute_case(spec: ModelSpec, model: Any, case: CaseSpec) -> tuple[Any, list[Any], dict[str, list[Measured]]]:
    """The case's input, its reference outputs (in the compute dtype) and the measured errors of
    PyTorch's own result and of the runtime emulation, per output."""
    dtype = torch_dtype(case.dtype)
    tol = TOLERANCES[case.dtype]
    extra = spec.extra_outputs()
    x = case_input(spec, case)
    # The weights as the runtime holds them: rounded once from f32 to the compute dtype.
    ref_model = copy.deepcopy(model).to(dtype).double()
    with torch.no_grad():
        exact = run_layers(ref_model, [(a, b, m.double()) for a, b, m in extra], x.double())
        expected = [t.to(dtype) for t in exact]
        emulated = run_layers(ref_model, extra, x.double(),
                              lambda t: flush_subnormals(t.to(dtype), dtype).double())
        native_model = copy.deepcopy(model).to(dtype)
        native = run_layers(native_model, [(a, b, m.to(dtype)) for a, b, m in extra], x)
    measured: dict[str, list[Measured]] = {}
    names = [spec.output_name] + [n for n, _, _ in extra]
    for i, name in enumerate(names):
        measured[name] = [measure(native[i], expected[i], tol), measure(emulated[i], expected[i], tol)]
    return x, expected, measured


def case_notes(spec: ModelSpec, case: CaseSpec) -> str:
    text = (
        "The reference is the PyTorch module in f64 on the weights and the input as the runtime holds them, "
        "rounded once to the compute dtype. PyTorch's own result in the compute dtype and an emulation of the "
        "runtime (each layer in f64, its output rounded to the compute dtype, subnormals flushed to zero) meet "
        "the tolerance (checked by the generator)."
    )
    if case.dtype != "f32":
        text += (
            f" {case.dtype} compute: the reference starts from the f32 weights of model.safetensors rounded once "
            f"to {case.dtype} (runtime.md §2 step 5) and the input stored here in {case.dtype}."
        )
    return " ".join(t for t in (text, spec.notes) if t)


# ---------------------------------------------------------------------------------------------
# Writing.


def save_tensors(path: str, tensors: dict[str, Any]) -> None:
    # No metadata: nothing machine- or time-dependent ends up in the file.
    with open(path, "wb") as f:
        f.write(safetensors.torch.save(tensors, metadata=None))


def check_no_subnormals(where: str, tensors: dict[str, Any]) -> None:
    for name, t in tensors.items():
        d = t.double()
        if ((d != 0) & (d.abs() < torch.finfo(t.dtype).tiny)).any():
            raise SystemExit(f"{where}: tensor {name} has subnormal values (graph-ir.md §3.1.8)")


def entry(name: str, t: Any) -> dict[str, Any]:
    return {"name": name, "dtype": gen_ops.dtype_name(t), "shape": list(t.shape)}


def write_model(spec: ModelSpec, out_dir: str, versions: dict[str, str],
                report: list[str] | None = None) -> int:
    model, layers = build_model(spec)
    state = state_tensors(model)
    graph = build_graph(spec, layers, state)
    model_dir = os.path.join(out_dir, spec.name)
    os.makedirs(model_dir, exist_ok=True)
    check_no_subnormals(f"{spec.name}/model", state)
    save_tensors(os.path.join(model_dir, "model.safetensors"), state)
    with open(os.path.join(model_dir, "model.graph.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(fixturelib.canonical_json(graph))
    cases = [c for c in CASES if c.model == spec.name]
    for case in cases:
        x, expected, measured = compute_case(spec, model, case)
        tol = TOLERANCES[case.dtype]
        for out, (native, emulated) in measured.items():
            for what, m in (("PyTorch in the compute dtype", native), ("the runtime emulation", emulated)):
                if m.ratio > 1.0:
                    raise SystemExit(
                        f"{spec.name}/{case.name}: {what} misses the tolerance on {out} "
                        f"(max abs err {m.abs:.3e}, max rel err {m.rel:.3e}, worst/allowed {m.ratio:.3f})"
                    )
            if report is not None:
                report.append(
                    f"{spec.name}/{case.name} {out}: max|expected| {expected_scale(expected, measured, out):.3e}; "
                    f"torch {case.dtype}: abs {native.abs:.2e} rel {native.rel:.2e} worst/allowed {native.ratio:.3f}; "
                    f"emulation: abs {emulated.abs:.2e} rel {emulated.rel:.2e} worst/allowed {emulated.ratio:.3f}"
                )
        outputs = dict(zip(measured, expected))
        tensors = {spec.input_name: x.contiguous()} | {k: v.contiguous() for k, v in outputs.items()}
        check_no_subnormals(f"{spec.name}/{case.name}", tensors)
        sidecar = {
            "schema_version": fixturelib.SCHEMA_VERSION,
            "model": spec.name,
            "case": case.name,
            "description": f"{spec.description} Batch {case.batch}, compute dtype {case.dtype}.",
            "compute_dtype": case.dtype,
            "inputs": [entry(spec.input_name, x)],
            "outputs": [entry(k, v) for k, v in outputs.items()],
            "tolerance": tol,
            "reference": "model.double()(x.double()) on the weights and input in the compute dtype, rounded once to it",
            "notes": case_notes(spec, case),
            "seed": case.seed,
            "generator": GENERATOR,
            "versions": versions,
        }
        save_tensors(os.path.join(model_dir, case.name + ".safetensors"), tensors)
        with open(os.path.join(model_dir, case.name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(fixturelib.canonical_json(sidecar))
    return len(cases)


def expected_scale(expected: list[Any], measured: dict[str, Any], out: str) -> float:
    return expected[list(measured).index(out)].double().abs().max().item()


def tf32_report(report: list[str]) -> None:
    """Informational: the f32 cases with `nn.linear` inputs rounded to TF32 (10-bit mantissa), as
    CUDA runs f32 matmul on Ampere and newer (docs/status.md §4). Not a check."""
    def tf32(t: Any) -> Any:
        bits = t.float().contiguous().view(torch.int32)
        rounded = ((bits + 0x1000) & ~0x1FFF).view(torch.float32)  # round to nearest on 13 dropped bits
        return rounded.double()

    for spec in MODELS.values():
        model, _ = build_model(spec)
        for case in [c for c in CASES if c.model == spec.name and c.dtype == "f32"]:
            _, expected, _ = compute_case(spec, model, case)
            x = case_input(spec, case).double()
            with torch.no_grad():
                for module in copy.deepcopy(model).double():
                    if isinstance(module, nn.Linear):
                        x = torch.nn.functional.linear(tf32(x), tf32(module.weight), module.bias)
                    else:
                        x = module(x)
                    x = x.float().double()
            r = measure(x, expected[0], MODEL_F32)
            report.append(f"{spec.name}/{case.name} logits with TF32 nn.linear (info only): abs {r.abs:.2e} "
                          f"rel {r.rel:.2e} worst/allowed {r.ratio:.3f}")


def generate(out_dir: str, report: list[str] | None = None) -> int:
    names = [(c.model, c.name) for c in CASES]
    if len(set(names)) != len(names) or len({c.seed for c in CASES}) != len(CASES):
        raise SystemExit("duplicate case names or seeds")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    versions = gen_ops.installed_versions()
    return sum(write_model(spec, out_dir, versions, report) for spec in MODELS.values())


def remove_stale(out_dir: str) -> None:
    """Deletes earlier fixture files (only `.json` and `.safetensors`), so removed cases go away."""
    gen_ops.remove_stale(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: fixtures/small/models)")
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
        remove_stale(args.out)
        n = generate(args.out, report)
        print(f"wrote {n} cases to {args.out}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            n = generate(tmp, report)
            problems = fixturelib.compare_trees(args.out, tmp)
        for line in problems:
            print(line, file=sys.stderr)
        if problems:
            print(f"{len(problems)} problem(s); run python {GENERATOR} and commit the result", file=sys.stderr)
            return 1
        print(f"ok: {n} cases match {args.out}")
    if report is not None:
        tf32_report(report)
        print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
