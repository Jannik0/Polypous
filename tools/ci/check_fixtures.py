#!/usr/bin/env python3
"""Checks the committed reference fixtures in fixtures/small/ (docs/testing-strategy.md §3).

- Size budget: each fixture (a case's `.safetensors` and `.json` together) under 1 MiB, everything
  under fixtures/small/ under 50 MiB.
- Every sidecar under fixtures/small/ops/ follows the schema and matches its safetensors header
  (names, dtypes, shapes, data offsets), and every `.safetensors` file has a sidecar.
- Every gradient sidecar under fixtures/small/grads/<op>/ follows op sidecar schema version 2
  (§3.6: keys and their order, `mask` and `running_stats` only where allowed, upstream and gradient
  names, dtypes and shapes) and matches its safetensors header.
- Every model under fixtures/small/models/<model>/ (§3.1, §3.5): `model.safetensors` holds exactly
  the params of `model.graph.json` (F32, declared shapes), and every case sidecar follows the model
  schema, names the graph's inputs and outputs in order and matches its safetensors header.
- Every initial-value sidecar under fixtures/small/init/<kind>/ (§3.9): schema, fans of its role,
  tolerance, one `values` tensor with the param's shape.
- Every training-mode model under fixtures/small/training/<model>/ (§3.10): graph and weights as for
  models, and every case follows model case schema version 2 (running statistics, dropout masks,
  upstream gradients and the gradients of the trainable params) and matches its safetensors header.
- Every training component sidecar under fixtures/small/train/<component>/ (§3.7): schema, the
  component and its config's kind and fields, the tensor names of the component (the optimizers'
  state names of training.md §5.2), the losses' prediction and target (f32 of the same shape, or
  logits [B, C] with i64 class indices [B]), the tolerance of §2, and for schedules one value per
  step and no tensors.
- No float tensor anywhere holds a subnormal value.

Standard library only (no PyTorch), so it runs in CI without setup. Exit status 0 if clean, 1 otherwise.
The helpers live in tools/reference/fixturelib.py.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools", "reference"))

import fixturelib  # noqa: E402


def check(root: str) -> list[str]:
    problems = fixturelib.check_sizes(root)
    files = fixturelib.list_files(root)
    present = set(files)
    for area, validate in (("models", fixturelib.validate_model_dir), ("training", fixturelib.validate_training_dir)):
        dirs = sorted({rel.split("/")[1] for rel in files if rel.startswith(area + "/") and rel.count("/") >= 2})
        for model in dirs:
            problems += validate(os.path.join(root, area, model))
    for rel in files:
        stem, ext = os.path.splitext(rel)
        area = rel.split("/", 1)[0]
        if area in ("models", "training"):
            if rel.count("/") < 2:
                problems.append(f"{rel}: a file directly in {area}/ (expected {area}/<model>/)")
            continue
        if ext == ".json" and area == "train":
            if rel.count("/") != 2:
                problems.append(f"{rel}: not in train/<component>/")
            else:
                problems += fixturelib.validate_train_case(os.path.join(root, rel))
            continue
        if ext == ".json" and area == "init":
            if rel.count("/") != 2:
                problems.append(f"{rel}: not in init/<kind>/")
            else:
                problems += fixturelib.validate_init_case(os.path.join(root, rel))
            continue
        if ext == ".json" and rel.startswith("ops/"):
            problems += fixturelib.validate_case(os.path.join(root, rel))
        elif ext == ".json" and rel.startswith("grads/"):
            if rel.count("/") != 2:
                problems.append(f"{rel}: not in grads/<op>/")
            else:
                problems += fixturelib.validate_grad_case(os.path.join(root, rel))
        elif ext == ".safetensors" and stem + ".json" not in present:
            problems.append(f"{rel}: no JSON sidecar")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(REPO, "fixtures", "small"))
    args = parser.parse_args(argv)
    problems = check(args.root)
    for line in problems:
        print(f"::error::{line}" if os.environ.get("GITHUB_ACTIONS") else f"error: {line}")
    if problems:
        return 1
    sizes = fixturelib.fixture_sizes(args.root)
    print(f"ok: {len(sizes)} fixtures, {sum(sizes.values())} bytes, largest {max(sizes.values(), default=0)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
