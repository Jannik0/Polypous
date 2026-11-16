#!/usr/bin/env python3
"""Tests for gen_models.py: `--check` on the committed model fixtures and the graph export.

Run with: python -m unittest discover -s tools/reference -v

These run the real generator and need the pinned PyTorch environment (requirements.txt); they are
skipped without it. The standard-library checks of the model fixtures are in
tools/ci/test_check_fixtures.py.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fixturelib  # noqa: E402

try:
    import gen_models  # noqa: E402
    import safetensors.torch

    HAVE_TORCH = gen_models.IMPORT_ERROR is None and not gen_models.gen_ops.version_problems()
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "needs the pinned PyTorch environment from tools/reference/requirements.txt")
class GeneratorCheckTest(unittest.TestCase):
    """End to end: --check on a copy of the committed model fixtures, then broken copies."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out = os.path.join(self.tmp, "models")
        shutil.copytree(gen_models.DEFAULT_OUT, self.out)

    def check(self) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = gen_models.main(["--check", "--out", self.out])
        return code, err.getvalue()

    def test_committed_fixtures_pass(self) -> None:
        self.assertEqual(self.check(), (0, ""))

    def test_detects_a_changed_weight(self) -> None:
        path = os.path.join(self.out, "mlp", "model.safetensors")
        with open(path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([last[0] ^ 1]))
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("changed: mlp/model.safetensors", err)

    def test_detects_missing_and_extra(self) -> None:
        os.remove(os.path.join(self.out, "cnn", "batch4_bf16.json"))
        with open(os.path.join(self.out, "cnn", "batch9_f32.json"), "w") as f:
            f.write("{}\n")
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("missing: cnn/batch4_bf16.json", err)
        self.assertIn("extra: cnn/batch9_f32.json", err)

    def test_regenerating_removes_stale_cases(self) -> None:
        with open(os.path.join(self.out, "cnn", "batch9_f32.json"), "w") as f:
            f.write("{}\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_models.main(["--out", self.out]), 0)
        self.assertEqual(fixturelib.compare_trees(self.out, gen_models.DEFAULT_OUT), [])


@unittest.skipUnless(HAVE_TORCH, "needs the pinned PyTorch environment from tools/reference/requirements.txt")
class GraphExportTest(unittest.TestCase):
    def graph(self, name: str) -> dict:
        with open(os.path.join(gen_models.DEFAULT_OUT, name, "model.graph.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_params_are_the_state_dict_keys(self) -> None:
        for name, spec in gen_models.MODELS.items():
            model, _ = gen_models.build_model(spec)
            keys = [k for k in model.state_dict() if not k.endswith("num_batches_tracked")]
            # Same names; the graph lists each node's params in catalog order (batch_norm: running
            # statistics first), the state_dict in module order.
            self.assertEqual(sorted(p["name"] for p in self.graph(name)["params"]), sorted(keys))
            # PyTorch loads the weights unchanged (the counter is the only key left out).
            state = safetensors.torch.load_file(os.path.join(gen_models.DEFAULT_OUT, name, "model.safetensors"))
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.assertEqual(unexpected, [])
            self.assertTrue(all(k.endswith("num_batches_tracked") for k in missing), missing)

    def test_cnn_nodes_follow_the_modules(self) -> None:
        nodes = self.graph("cnn")["nodes"]
        self.assertEqual(
            [(n["id"], n["op"]) for n in nodes],
            [("conv1", "nn.conv2d"), ("relu1", "nn.relu"), ("pool1", "nn.max_pool2d"), ("conv2", "nn.conv2d"),
             ("bn", "nn.batch_norm"), ("relu2", "nn.relu"), ("pool2", "nn.adaptive_avg_pool2d"),
             ("flatten", "core.flatten"), ("fc", "nn.linear")],
        )
        self.assertEqual(nodes[3]["params"], {"weight": "conv2.weight"})  # no bias
        self.assertEqual(nodes[2]["attrs"]["stride"], [2, 2])
        self.assertEqual(nodes[6]["attrs"], {"output_size": [3, 3]})

    def test_batch_norm_statistics_are_not_defaults(self) -> None:
        model, _ = gen_models.build_model(gen_models.MODELS["cnn"])
        bn = model.bn
        self.assertFalse(bool((bn.running_mean == 0).all()))
        self.assertFalse(bool((bn.running_var == 1).all()))
        self.assertFalse(bool((bn.weight == 1).all()))
        self.assertFalse(bool((bn.bias == 0).all()))

    def test_unsupported_module_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            gen_models.layer_node("x", gen_models.nn.Tanh(), "image")


if __name__ == "__main__":
    unittest.main()
