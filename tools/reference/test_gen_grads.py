#!/usr/bin/env python3
"""Tests for gen_grads.py and the schema-2 helpers of fixturelib.py (docs/testing-strategy.md §3.6).

Run with: python -m unittest discover -s tools/reference -v

`GradSchemaTest` uses the standard library only and runs in CI. `GradGeneratorCheckTest` runs the real
generator and needs the pinned PyTorch environment (requirements.txt); it is skipped without it. The
sidecar validation itself is tested in tools/ci/test_check_fixtures.py.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import fixturelib  # noqa: E402

GRADS = os.path.join(REPO, "fixtures", "small", "grads")
GRAPH_IR = os.path.join(REPO, "docs", "specs", "graph-ir.md")


def differentiable_ops() -> list[str]:
    """The ops marked differentiable in graph-ir.md §3.2/§3.3: table rows whose "Diff." cell is not "no"."""
    with open(GRAPH_IR, encoding="utf-8") as f:
        lines = f.read().splitlines()
    ops = []
    columns: list[str] = []
    for line in lines:
        if line.startswith("| Op |"):
            columns = [c.strip() for c in line.strip("|").split("|")]
            continue
        m = re.match(r"^\| `((?:core|nn)\.[a-z0-9_]+)` \|", line)
        if m and "Diff." in columns:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[columns.index("Diff.")] != "no":
                ops.append(m.group(1))
    return ops


class GradSchemaTest(unittest.TestCase):
    def test_keys_by_op_and_mode(self) -> None:
        base = fixturelib.grad_sidecar_keys("nn.linear", "inference", True)
        self.assertEqual(base[:8], ("schema_version", "op", "case", "description", "mode", "attributes", "inputs", "params"))
        self.assertNotIn("mask", base)
        self.assertNotIn("running_stats", base)
        dropout = fixturelib.grad_sidecar_keys("nn.dropout", "training", False)
        self.assertEqual(dropout[dropout.index("inputs") + 1], "mask")
        bn = fixturelib.grad_sidecar_keys("nn.batch_norm", "training", True)
        self.assertEqual(bn[bn.index("outputs") + 1], "running_stats")
        self.assertNotIn("running_stats", fixturelib.grad_sidecar_keys("nn.batch_norm", "inference", True))
        self.assertNotIn("mask", fixturelib.grad_sidecar_keys("nn.dropout", "inference", False))
        self.assertEqual(base[-8:], ("gradients", "tolerance", "gradient_tolerance", "reference", "notes", "seed",
                                     "generator", "versions"))

    def test_output_ports(self) -> None:
        self.assertEqual(fixturelib.output_port_name("out"), "out")
        self.assertEqual(fixturelib.output_port_name("out2"), "2")
        self.assertEqual(fixturelib.output_port_name("values"), "values")

    def test_spec_table_parse(self) -> None:
        ops = differentiable_ops()
        self.assertIn("core.add", ops)
        self.assertIn("nn.batch_norm", ops)
        self.assertIn("core.cast", ops)  # "float → float only"
        self.assertNotIn("core.eq", ops)
        self.assertNotIn("core.argmax", ops)
        self.assertNotIn("core.zeros", ops)

    @unittest.skipUnless(os.path.isdir(GRADS), "committed gradient fixtures not present")
    def test_every_differentiable_op_has_cases(self) -> None:
        have = sorted(d for d in os.listdir(GRADS) if os.path.isdir(os.path.join(GRADS, d)))
        self.assertEqual(have, sorted(differentiable_ops()))


try:
    import gen_grads  # noqa: E402

    HAVE_TORCH = gen_grads.IMPORT_ERROR is None and not gen_grads.gen_ops.version_problems()
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "needs the pinned PyTorch environment from tools/reference/requirements.txt")
class GradGeneratorCheckTest(unittest.TestCase):
    """End to end: --check on a copy of the committed fixtures, then broken copies."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out = os.path.join(self.tmp, "grads")
        shutil.copytree(gen_grads.DEFAULT_OUT, self.out)

    def check(self) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = gen_grads.main(["--check", "--out", self.out])
        return code, err.getvalue()

    def test_committed_fixtures_pass(self) -> None:
        self.assertEqual(self.check(), (0, ""))

    def test_detects_changed(self) -> None:
        path = os.path.join(self.out, "nn.linear", "2d_f32.safetensors")
        with open(path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([last[0] ^ 1]))
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("changed: nn.linear/2d_f32.safetensors", err)

    def test_detects_missing_and_extra(self) -> None:
        os.remove(os.path.join(self.out, "core.max", "ties_all_f32.json"))
        with open(os.path.join(self.out, "core.max", "no_such_case.json"), "w", encoding="utf-8") as f:
            f.write("{}\n")
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("missing: core.max/ties_all_f32.json", err)
        self.assertIn("extra: core.max/no_such_case.json", err)

    def test_case_names_and_seeds(self) -> None:
        cases = gen_grads.build_cases()
        self.assertEqual(len({c.seed for c in cases}), len(cases))
        self.assertTrue(all(re.fullmatch(r"[a-z0-9_]+_f32", c.name) for c in cases))
        self.assertEqual(sorted({c.op for c in cases}), sorted(differentiable_ops()))

    def test_refuses_a_spec_disagreement(self) -> None:
        # A wrong convention (maximum's tie split 1 : 0 instead of g / 2 each) must stop the generator.
        c = next(c for c in gen_grads.build_cases() if (c.op, c.name) == ("core.maximum", "ties_f32"))

        def wrong(d, up):
            g = up["out"]
            return {"grad_a": gen_grads.torch.where(d["a"] >= d["b"], g, 0 * g),
                    "grad_b": gen_grads.torch.where(d["a"] < d["b"], g, 0 * g)}

        bad = gen_grads.dataclasses.replace(c, spec=wrong)
        with self.assertRaises(SystemExit) as ctx:
            gen_grads.compute(bad)
        self.assertIn("disagrees with graph-ir.md", str(ctx.exception))

    def test_refuses_non_finite_inputs(self) -> None:
        c = next(c for c in gen_grads.build_cases() if (c.op, c.name) == ("nn.relu", "zero_f32"))
        bad = gen_grads.dataclasses.replace(c, tensors=(("x", gen_grads.vals([1.0, float("nan")])),))
        with self.assertRaises(SystemExit) as ctx:
            gen_grads.compute(bad)
        self.assertIn("not finite", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
