#!/usr/bin/env python3
"""Tests for gen_training.py: case coverage (standard library only) and `--check` on the committed
initial-value and training-mode fixtures (needs the pinned PyTorch environment; skipped without it).

Run with: python -m unittest discover -s tools/reference -v
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_training  # noqa: E402  (imports without PyTorch; generating needs it)
import polyrng  # noqa: E402

try:
    HAVE_TORCH = gen_training.IMPORT_ERROR is None and not gen_training.gen_ops.version_problems()
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False


class CoverageTest(unittest.TestCase):
    """testing-strategy.md §3.8, "Initial values": every kind on params of each fan role."""

    def test_every_kind_on_each_fan_role(self) -> None:
        cases = gen_training.init_cases()
        roles = {kind: set() for kind in polyrng.FAN_KINDS}
        for c in cases:
            if c.init["kind"] in roles and c.role is not None:
                roles[c.init["kind"]].add(c.role)
        want = {r for *_, r in gen_training.FAN_ROLES}
        for kind, got in roles.items():
            self.assertEqual(got, want, kind)
        others = {c.init["kind"] for c in cases} - set(polyrng.FAN_KINDS)
        self.assertEqual(others, {"zeros", "ones", "constant", "uniform", "normal"})
        for kind in others:
            used = {c.role for c in cases if c.init["kind"] == kind}
            self.assertIn(None, used, f"{kind}: an unused param")
            self.assertIn(("nn.linear", "bias"), used, f"{kind}: a role without fans")

    def test_the_spec_table_case_is_included(self) -> None:
        table = [c for c in gen_training.init_cases() if c.case == "fc1_weight_seed42"]
        self.assertEqual(len(table), 1)
        c = table[0]
        self.assertEqual((c.seed, c.name, c.shape, c.init), (42, "fc1.weight", [128, 784], {"kind": "kaiming_uniform"}))

    def test_case_names_are_unique(self) -> None:
        keys = [(c.init["kind"], c.case) for c in gen_training.init_cases()]
        self.assertEqual(len(keys), len(set(keys)))
        train = [(c.model, c.name) for c in gen_training.TRAIN_CASES]
        self.assertEqual(len(train), len(set(train)))


@unittest.skipUnless(HAVE_TORCH, "needs the pinned PyTorch environment from tools/reference/requirements.txt")
class GeneratorCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.init = os.path.join(self.tmp, "init")
        self.training = os.path.join(self.tmp, "training")
        shutil.copytree(gen_training.DEFAULT_INIT_OUT, self.init)
        shutil.copytree(gen_training.DEFAULT_TRAINING_OUT, self.training)

    def check(self) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = gen_training.main(["--check", "--init-out", self.init, "--training-out", self.training])
        return code, err.getvalue()

    def test_committed_fixtures_match(self) -> None:
        self.assertEqual(self.check(), (0, ""))

    def test_a_changed_value_is_found(self) -> None:
        path = os.path.join(self.training, "mlp", "batch4_f32.safetensors")
        with open(path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([last[0] ^ 1]))
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("changed: mlp/batch4_f32.safetensors", err)

    def test_an_extra_file_is_found(self) -> None:
        shutil.copy(os.path.join(self.init, "zeros", "unused.json"), os.path.join(self.init, "zeros", "stray.json"))
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("extra: zeros/stray.json", err)


if __name__ == "__main__":
    unittest.main()
