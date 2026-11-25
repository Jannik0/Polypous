#!/usr/bin/env python3
"""Generates the MNIST training fixtures in fixtures/small/mnist/ (docs/testing-strategy.md §3.8).

    python tools/reference/gen_mnist.py            # regenerate fixtures/small/mnist/
    python tools/reference/gen_mnist.py --check    # fail if the committed fixtures differ
    python tools/reference/gen_mnist.py --report   # also print PyTorch f32's errors against the reference

Files (all generated; never edit them by hand):

| File | What |
|---|---|
| `cnn.graph.json` | The MNIST CNN: `conv2d` → `relu` → `conv2d` → `batch_norm` → `relu` → `max_pool2d` → `dropout` → `flatten` → `linear` → `relu` → `dropout` → `linear`, every param with PyTorch's default `init` |
| `train.json` | Its training config: AdamW, cross-entropy, accuracy, the MNIST test set as the validation set, 5 epochs, seeded shuffling, checkpoints |
| `reference_cnn.graph.json` | The same graph with every `dropout` at `p = 0` (the identity, graph-ir.md §3.3) |
| `reference_train.json` | The same config with `shuffle: false`, a constant learning rate, 1 epoch, no validation set and no checkpoints |
| `reference.safetensors` | The initial weights of the reference run (every param, F32, no `__metadata__`) |
| `reference.json` | Its sidecar: the expected first 10 training losses and the inference-mode loss of the first 64 test images after step 10 |

**Data paths.** Both configs point at the converted MNIST files in the fixture cache, relative to
the config file (`../../cache/mnist/…`, training.md §2.1: the CLI resolves them against the
config's directory), so `polyp train` works from any directory once `fetch_fixtures.py` (or
`convert_mnist.py`) has filled the cache. `train.json`'s run directory is `../../cache/runs/mnist`,
inside the git-ignored cache.

**Initial weights** come from `polyrng.initial_values` (graph-ir.md §2.4.1) with seed `INIT_SEED`
and each param's `init` of the graph, so they are bit-identical on every platform (uniform-based
kinds and constants only; no platform `ln`/`cos`).

**Reference** (testing-strategy.md §3.8). The reference graph as PyTorch functional ops in f64 on
the f32 weights and data: `F.conv2d`, `F.batch_norm(..., training=True)` (which updates the f64
running statistics in place), `F.max_pool2d`, `F.linear`, `F.cross_entropy`, and
`torch.optim.AdamW(lr, betas, eps, weight_decay, foreach=False)`, stepped once per batch of 64
training samples in dataset order (`shuffle: false`). The losses of the first 10 steps, then the
inference-mode loss (`training=False`, the running statistics) of the first 64 test images, are
rounded once to f32 and written to `reference.json`. Rounding once to f32 keeps the bytes the same
on CPUs whose f64 BLAS kernels sum in another order (their f64 results differ far below an f32 ulp).
Before writing, the generator runs the same in PyTorch f32 and stops if that misses
`rtol = 1e-3` against the f64 reference (`--report` prints both errors).

**The MNIST cache.** The reference needs the converted MNIST files: by default
`fixtures/cache/mnist/mnist-train.safetensors` and `mnist-test.safetensors` (`--cache-dir`), each
checked against its `fixtures/manifest.json` entry (size and SHA-256) before it is read. If they are
missing, run `python tools/reference/fetch_fixtures.py mnist-train mnist-test` (or
`convert_mnist.py`).

Needs the pinned versions from requirements.txt; `--check` refuses to compare otherwise.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_fixtures  # noqa: E402
import fixturelib  # noqa: E402
import gen_ops  # noqa: E402  (version pins)
import polyrng  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO, "fixtures", "small", "mnist")
DEFAULT_CACHE = os.path.join(REPO, "fixtures", "cache")
DEFAULT_MANIFEST = os.path.join(REPO, "fixtures", "manifest.json")
GENERATOR = "tools/reference/gen_mnist.py"

try:
    import numpy as np
    import safetensors.torch
    import torch
    import torch.nn.functional as F  # noqa: N812 - PyTorch's usual name, as in the spec's `F.`
except ImportError as e:  # pragma: no cover - depends on the environment
    np = safetensors = torch = F = None  # type: ignore[assignment]
    IMPORT_ERROR: ImportError | None = e
else:
    IMPORT_ERROR = None

# ---------------------------------------------------------------------------------------------
# The model (testing-strategy.md §3.8's layer order). Sized so the reference's initial weights
# stay a committed fixture under 1 MiB (149,466 values, 598 KB) and the CNN still reaches 98 %.

C1, C2, HIDDEN, CLASSES = 8, 16, 64, 10
SIDE = (28 - 2 - 2) // 2  # two 3x3 convolutions without padding, then a 2x2 max pool: 12
FLAT = C2 * SIDE * SIDE  # 2304
P_DROP1, P_DROP2 = 0.25, 0.5
BN_EPS, BN_MOMENTUM = 1e-5, 0.1

# The configs (training.md §2).
SEED = 42
BATCH = 64
LR, BETAS, EPS, WEIGHT_DECAY = 1e-3, [0.9, 0.999], 1e-8, 0.01
EPOCHS = 5
TRAIN_DATA = "../../cache/mnist/mnist-train.safetensors"
TEST_DATA = "../../cache/mnist/mnist-test.safetensors"
RUN_DIR = "../../cache/runs/mnist"

# The reference run.
INIT_SEED = 84
REFERENCE_STEPS = 10
INFERENCE_SAMPLES = 64
TOLERANCE = {"rtol": 1e-3}
DATA_ENTRIES = {"train": "mnist-train", "test": "mnist-test"}

# PyTorch's default initialization of Conv2d, BatchNorm2d and Linear, as graph `init` objects.
KAIMING = {"kind": "kaiming_uniform", "a": math.sqrt(5.0)}


def bias_init(fan_in: int) -> dict[str, Any]:
    bound = 1.0 / math.sqrt(fan_in)
    return {"kind": "uniform", "low": -bound, "high": bound}


# (name, shape, init); declaration order.
PARAMS: list[tuple[str, list[int], dict[str, Any]]] = [
    ("conv1.weight", [C1, 1, 3, 3], KAIMING),
    ("conv1.bias", [C1], bias_init(1 * 3 * 3)),
    ("conv2.weight", [C2, C1, 3, 3], KAIMING),
    ("bn.running_mean", [C2], {"kind": "zeros"}),
    ("bn.running_var", [C2], {"kind": "ones"}),
    ("bn.weight", [C2], {"kind": "ones"}),
    ("bn.bias", [C2], {"kind": "zeros"}),
    ("fc1.weight", [HIDDEN, FLAT], KAIMING),
    ("fc1.bias", [HIDDEN], bias_init(FLAT)),
    ("fc2.weight", [CLASSES, HIDDEN], KAIMING),
    ("fc2.bias", [CLASSES], bias_init(HIDDEN)),
]
# The op and op param that use each weight with fans (graph-ir.md §2.4.1).
ROLES = {"conv1.weight": "nn.conv2d", "conv2.weight": "nn.conv2d", "fc1.weight": "nn.linear", "fc2.weight": "nn.linear"}


def graph(name: str, p1: float, p2: float) -> dict[str, Any]:
    def linear(node_id: str, x: str) -> dict[str, Any]:
        return {"id": node_id, "op": "nn.linear", "inputs": {"x": x},
                "params": {"weight": f"{node_id}.weight", "bias": f"{node_id}.bias"}}

    return {
        "format": "polypous.graph",
        "format_version": 1,
        "name": name,
        "opsets": {"core": 1, "nn": 1},
        "dims": {"batch": {"min": 1}},
        "inputs": [{"name": "image", "dtype": "float", "shape": ["batch", 1, 28, 28]}],
        "outputs": [{"name": "logits", "value": "fc2"}],
        "params": [{"name": n, "dtype": "float", "shape": s, "init": i} for n, s, i in PARAMS],
        "nodes": [
            {"id": "conv1", "op": "nn.conv2d", "inputs": {"x": "image"},
             "params": {"weight": "conv1.weight", "bias": "conv1.bias"}},
            {"id": "relu1", "op": "nn.relu", "inputs": {"x": "conv1"}},
            {"id": "conv2", "op": "nn.conv2d", "inputs": {"x": "relu1"}, "params": {"weight": "conv2.weight"}},
            {"id": "bn", "op": "nn.batch_norm", "inputs": {"x": "conv2"},
             "params": {"running_mean": "bn.running_mean", "running_var": "bn.running_var",
                        "weight": "bn.weight", "bias": "bn.bias"},
             "attrs": {"eps": BN_EPS, "momentum": BN_MOMENTUM}},
            {"id": "relu2", "op": "nn.relu", "inputs": {"x": "bn"}},
            {"id": "pool", "op": "nn.max_pool2d", "inputs": {"x": "relu2"}, "attrs": {"kernel_size": [2, 2]}},
            {"id": "drop1", "op": "nn.dropout", "inputs": {"x": "pool"}, "attrs": {"p": p1}},
            {"id": "flatten", "op": "core.flatten", "inputs": {"x": "drop1"}, "attrs": {"start_dim": 1}},
            linear("fc1", "flatten"),
            {"id": "relu3", "op": "nn.relu", "inputs": {"x": "fc1"}},
            {"id": "drop2", "op": "nn.dropout", "inputs": {"x": "relu3"}, "attrs": {"p": p2}},
            linear("fc2", "drop2"),
        ],
    }


def train_config() -> dict[str, Any]:
    return {
        "format": "polypous.training",
        "format_version": 1,
        "seed": SEED,
        "device": "cpu",
        "precision": "f32",
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "data": {
            "train": {"kind": "safetensors", "path": TRAIN_DATA},
            "validation": {"kind": "safetensors", "path": TEST_DATA},
            "shuffle": True,
        },
        "inputs": {"image": "image"},
        "loss": {"kind": "cross_entropy", "prediction": "logits", "target": "label"},
        "metrics": [{"kind": "accuracy", "prediction": "logits", "target": "label"}],
        "optimizer": {"kind": "adamw", "lr": LR, "betas": BETAS, "eps": EPS, "weight_decay": WEIGHT_DECAY},
        "schedule": {"kind": "constant"},
        "checkpoint": {"dir": RUN_DIR, "every_epochs": 1, "keep_last": 2, "keep_best": "val.accuracy"},
    }


def reference_config() -> dict[str, Any]:
    doc = copy.deepcopy(train_config())
    doc["epochs"] = 1
    doc["data"] = {"train": {"kind": "safetensors", "path": TRAIN_DATA}, "shuffle": False}
    doc["checkpoint"] = None
    return doc


# ---------------------------------------------------------------------------------------------
# Data


def load_dataset(cache_dir: str, manifest: str, role: str) -> dict[str, Any]:
    """The `image` and `label` tensors of a converted MNIST file, after checking its size and hash
    against the manifest entry."""
    entries = {e["name"]: e for e in fetch_fixtures.load_manifest(manifest)}
    entry = entries.get(DATA_ENTRIES[role])
    if entry is None:
        raise SystemExit(f"{manifest} has no entry {DATA_ENTRIES[role]!r}")
    path = os.path.join(cache_dir, *entry["path"].split("/"))
    hint = f"run python tools/reference/fetch_fixtures.py {' '.join(DATA_ENTRIES.values())} (or convert_mnist.py)"
    if not os.path.isfile(path):
        raise SystemExit(f"the MNIST {role} set {path} is missing: {hint}")
    if os.path.getsize(path) != entry["size"] or fetch_fixtures.sha256_of(path) != entry["sha256"]:
        raise SystemExit(f"{path} does not match its manifest entry {entry['name']!r}: {hint}")
    tensors = safetensors.torch.load_file(path)
    if set(tensors) != {"image", "label"}:
        raise SystemExit(f"{path}: unexpected tensors {sorted(tensors)}")
    return tensors


# ---------------------------------------------------------------------------------------------
# Reference


def initial_weights() -> dict[str, Any]:
    weights = {}
    for name, shape, init in PARAMS:
        fans = polyrng.fans_of(ROLES[name], "weight", shape) if name in ROLES else None
        values = polyrng.initial_values(INIT_SEED, name, shape, init, fans)
        t = torch.tensor(values, dtype=torch.float64).to(torch.float32).reshape(shape)
        # polyrng's values are f32 values already: the conversion is exact.
        if [float(v) for v in t.flatten().tolist()] != values:
            raise SystemExit(f"{name}: values changed when stored as f32")
        weights[name] = t.contiguous()
    return weights


def forward(p: dict[str, Any], x: Any, training: bool) -> Any:
    """The reference graph (every dropout at p = 0, the identity). In training mode batch_norm
    updates p["bn.running_mean"] and p["bn.running_var"] in place, as the trainer writes them back."""
    h = F.relu(F.conv2d(x, p["conv1.weight"], p["conv1.bias"]))
    h = F.conv2d(h, p["conv2.weight"])
    h = F.batch_norm(h, p["bn.running_mean"], p["bn.running_var"], p["bn.weight"], p["bn.bias"],
                     training=training, momentum=BN_MOMENTUM, eps=BN_EPS)
    h = F.max_pool2d(F.relu(h), 2)
    h = F.relu(F.linear(h.flatten(1), p["fc1.weight"], p["fc1.bias"]))
    return F.linear(h, p["fc2.weight"], p["fc2.bias"])


def run_reference(weights: dict[str, Any], train: dict[str, Any], test: dict[str, Any], dtype: Any) -> tuple[list[float], float]:
    """The first REFERENCE_STEPS training losses and the inference-mode loss of the first
    INFERENCE_SAMPLES test images after them, computed in `dtype`."""
    running = ("bn.running_mean", "bn.running_var")
    params = {n: w.to(dtype).clone().requires_grad_(n not in running) for n, w in weights.items()}
    trainable = [params[n] for n, _, _ in PARAMS if n not in running]
    opt = torch.optim.AdamW(trainable, lr=LR, betas=tuple(BETAS), eps=EPS, weight_decay=WEIGHT_DECAY, foreach=False)
    losses = []
    for k in range(REFERENCE_STEPS):
        x = train["image"][k * BATCH:(k + 1) * BATCH].to(dtype)
        y = train["label"][k * BATCH:(k + 1) * BATCH]
        loss = F.cross_entropy(forward(params, x, training=True), y)
        losses.append(float(loss.detach()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = test["image"][:INFERENCE_SAMPLES].to(dtype)
        y = test["label"][:INFERENCE_SAMPLES]
        inference = float(F.cross_entropy(forward(params, x, training=False), y))
    return losses, inference


def f32(v: float) -> float:
    return float(np.float32(v))


def rel_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


def generate(out_dir: str, cache_dir: str, manifest: str, report: list[str] | None = None) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    versions = gen_ops.installed_versions()
    train = load_dataset(cache_dir, manifest, "train")
    test = load_dataset(cache_dir, manifest, "test")
    weights = initial_weights()
    losses64, inference64 = run_reference(weights, train, test, torch.float64)
    losses32, inference32 = run_reference(weights, train, test, torch.float32)
    expected = [f32(v) for v in losses64]
    expected_inference = f32(inference64)
    worst = 0.0
    for k, (got, want) in enumerate(zip(losses32 + [inference32], expected + [expected_inference])):
        err = rel_error(got, want)
        worst = max(worst, err)
        if report is not None:
            what = f"loss of step {k}" if k < REFERENCE_STEPS else "inference loss after step 10"
            report.append(f"{what}: f64 {want:.9g}, torch f32 {got:.9g}, relative error {err:.2e}")
    if worst > TOLERANCE["rtol"]:
        raise SystemExit(f"PyTorch f32 misses rtol {TOLERANCE['rtol']} against the f64 reference (worst {worst:.2e})")
    if report is not None:
        report.append(f"torch f32 vs f64: worst relative error {worst:.2e} (tolerance rtol {TOLERANCE['rtol']}, "
                      f"{TOLERANCE['rtol'] / worst:.0f}x margin)")
    if not all(math.isfinite(v) for v in expected + [expected_inference]):
        raise SystemExit("the reference losses are not finite")
    for name, t in weights.items():
        d = t.double()
        if ((d != 0) & (d.abs() < torch.finfo(torch.float32).tiny)).any():
            raise SystemExit(f"initial weight {name} has subnormal values (graph-ir.md §3.1.8)")

    entries = {e["name"]: e for e in fetch_fixtures.load_manifest(manifest)}
    sidecar = {
        "schema_version": 1,
        "case": "mnist_reference",
        "description": (
            "The MNIST CNN with every dropout at p = 0, trained from the initial weights in "
            "reference.safetensors with reference_train.json: the losses of the first 10 steps "
            "(batch 64, shuffle false, constant lr), then the inference-mode loss of the first 64 "
            "test images (testing-strategy.md §3.8)."
        ),
        "graph": "reference_cnn.graph.json",
        "config": "reference_train.json",
        "weights": {
            "seed": INIT_SEED,
            "params": [{"name": n, "dtype": "f32", "shape": s} for n, s, _ in PARAMS],
        },
        "data": {role: {"name": name, "sha256": entries[name]["sha256"]} for role, name in DATA_ENTRIES.items()},
        "steps": REFERENCE_STEPS,
        "losses": expected,
        "inference": {"dataset": "test", "first": INFERENCE_SAMPLES, "loss": expected_inference},
        "tolerance": TOLERANCE,
        "reference": (
            "PyTorch in f64 on the f32 weights and data, rounded once to f32: F.conv2d, F.batch_norm "
            "(training=True updates the running statistics), F.max_pool2d, F.linear, F.cross_entropy, "
            "torch.optim.AdamW(foreach=False); the inference loss with training=False"
        ),
        "notes": (
            "Initial weights: polyrng.initial_values (graph-ir.md §2.4.1) with seed `weights.seed` and "
            "each param's init in the graph, so bit-identical on every platform. PyTorch's own f32 run "
            "meets the tolerance against the reference (gen_mnist.py --report)."
        ),
        "generator": GENERATOR,
        "versions": versions,
    }
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "cnn.graph.json"), graph("mnist_cnn", P_DROP1, P_DROP2))
    write_json(os.path.join(out_dir, "train.json"), train_config())
    write_json(os.path.join(out_dir, "reference_cnn.graph.json"), graph("mnist_cnn_reference", 0.0, 0.0))
    write_json(os.path.join(out_dir, "reference_train.json"), reference_config())
    with open(os.path.join(out_dir, "reference.safetensors"), "wb") as f:
        # No metadata: nothing machine- or time-dependent ends up in the file.
        f.write(safetensors.torch.save(weights, metadata=None))
    write_json(os.path.join(out_dir, "reference.json"), sidecar)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(fixturelib.canonical_json(obj))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="default: fixtures/small/mnist")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, help="where the MNIST files are (default: fixtures/cache)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="regenerate into a temporary directory and compare")
    parser.add_argument("--report", action="store_true", help="also print PyTorch f32's errors against the reference")
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
        generate(args.out, args.cache_dir, args.manifest, report)
        print(f"wrote the MNIST fixtures to {args.out}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp, args.cache_dir, args.manifest, report)
            problems = fixturelib.compare_trees(args.out, tmp)
        for line in problems:
            print(line, file=sys.stderr)
        if problems:
            print(f"{len(problems)} problem(s); run python {GENERATOR} and commit the result", file=sys.stderr)
            return 1
        print(f"ok: the MNIST fixtures in {args.out} match")
    if report is not None:
        print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
