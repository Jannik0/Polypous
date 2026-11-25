#!/usr/bin/env python3
"""Tests for gen_mnist.py: the committed MNIST graphs and configs against the generator's
definitions and testing-strategy.md §3.8 (standard library only), and `--check` on the whole tree
(needs the pinned PyTorch environment and the MNIST cache; skipped without them).

Run with: python -m unittest discover -s tools/reference -v
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fixturelib  # noqa: E402
import gen_mnist  # noqa: E402  (imports without PyTorch; generating needs it)

try:
    HAVE_TORCH = gen_mnist.IMPORT_ERROR is None and not gen_mnist.gen_ops.version_problems()
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False

# testing-strategy.md §3.8.
LAYERS = ["nn.conv2d", "nn.relu", "nn.conv2d", "nn.batch_norm", "nn.relu", "nn.max_pool2d", "nn.dropout",
          "core.flatten", "nn.linear", "nn.relu", "nn.dropout", "nn.linear"]


def committed(name: str) -> str:
    with open(os.path.join(gen_mnist.DEFAULT_OUT, name), encoding="utf-8") as f:
        return f.read()


class DefinitionTest(unittest.TestCase):
    def test_committed_json_is_the_generators(self) -> None:
        for name, doc in [
            ("cnn.graph.json", gen_mnist.graph("mnist_cnn", gen_mnist.P_DROP1, gen_mnist.P_DROP2)),
            ("reference_cnn.graph.json", gen_mnist.graph("mnist_cnn_reference", 0.0, 0.0)),
            ("train.json", gen_mnist.train_config()),
            ("reference_train.json", gen_mnist.reference_config()),
        ]:
            self.assertEqual(committed(name), fixturelib.canonical_json(doc), name)

    def test_layer_order_and_dropout(self) -> None:
        for p1, p2 in [(gen_mnist.P_DROP1, gen_mnist.P_DROP2), (0.0, 0.0)]:
            nodes = gen_mnist.graph("g", p1, p2)["nodes"]
            self.assertEqual([n["op"] for n in nodes], LAYERS)
            # Each node reads the one before it: a chain.
            for prev, node in zip(nodes, nodes[1:]):
                self.assertEqual(node["inputs"], {"x": prev["id"]})
            self.assertEqual([n["attrs"]["p"] for n in nodes if n["op"] == "nn.dropout"], [p1, p2])
        self.assertGreater(gen_mnist.P_DROP1, 0.0)
        self.assertGreater(gen_mnist.P_DROP2, 0.0)

    def test_every_param_has_an_init_and_the_size_fits_a_fixture(self) -> None:
        doc = gen_mnist.graph("g", 0.0, 0.0)
        self.assertTrue(all("init" in p for p in doc["params"]))
        n = sum(math.prod(p["shape"]) for p in doc["params"])
        self.assertEqual(n, 149466)
        size = os.path.getsize(os.path.join(gen_mnist.DEFAULT_OUT, "reference.safetensors"))
        self.assertLess(size, fixturelib.MAX_FIXTURE_BYTES)

    def test_reference_config_differs_only_as_specified(self) -> None:
        train, ref = gen_mnist.train_config(), gen_mnist.reference_config()
        self.assertEqual(train["optimizer"]["kind"], "adamw")
        self.assertEqual(train["loss"]["kind"], "cross_entropy")
        self.assertEqual([m["kind"] for m in train["metrics"]], ["accuracy"])
        self.assertEqual(train["epochs"], 5)
        self.assertTrue(train["data"]["shuffle"])
        self.assertEqual(train["data"]["validation"]["path"], gen_mnist.TEST_DATA)
        self.assertIsNotNone(train["checkpoint"])
        self.assertFalse(ref["data"]["shuffle"])
        self.assertEqual(ref["schedule"], {"kind": "constant"})
        same = {k: v for k, v in train.items() if k not in ("epochs", "data", "checkpoint")}
        self.assertEqual(same, {k: v for k, v in ref.items() if k not in ("epochs", "data", "checkpoint")})
        self.assertEqual(ref["data"]["train"], train["data"]["train"])
        # The run directory and the data stay inside the git-ignored fixture cache.
        for path in (train["checkpoint"]["dir"], gen_mnist.TRAIN_DATA, gen_mnist.TEST_DATA):
            resolved = os.path.normpath(os.path.join(gen_mnist.DEFAULT_OUT, path))
            self.assertTrue(resolved.startswith(os.path.normpath(gen_mnist.DEFAULT_CACHE) + os.sep), path)

    def test_sidecar(self) -> None:
        sidecar = json.loads(committed("reference.json"))
        self.assertEqual(sidecar["steps"], 10)
        self.assertEqual(len(sidecar["losses"]), 10)
        self.assertEqual(sidecar["tolerance"], {"rtol": 1e-3})
        self.assertEqual(sidecar["inference"]["first"], 64)
        self.assertEqual([p["name"] for p in sidecar["weights"]["params"]], [n for n, _, _ in gen_mnist.PARAMS])
        header = fixturelib.read_safetensors_header(os.path.join(gen_mnist.DEFAULT_OUT, "reference.safetensors"))
        self.assertEqual({n: (h["dtype"], h["shape"]) for n, h in header.items()},
                         {n: ("F32", s) for n, s, _ in gen_mnist.PARAMS})


@unittest.skipUnless(HAVE_TORCH, "needs the pinned PyTorch environment (tools/reference/requirements.txt)")
class CheckTest(unittest.TestCase):
    def test_committed_fixtures_match(self) -> None:
        for name in gen_mnist.DATA_ENTRIES.values():
            if not os.path.isfile(os.path.join(gen_mnist.DEFAULT_CACHE, "mnist", f"{name}.safetensors")):
                self.skipTest("the MNIST cache is empty (fetch_fixtures.py mnist-train mnist-test)")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gen_mnist.main(["--check"])
        self.assertEqual(code, 0, err.getvalue())


if __name__ == "__main__":
    unittest.main()
