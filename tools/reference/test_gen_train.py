#!/usr/bin/env python3
"""Tests for gen_train.py: case coverage (testing-strategy.md §3.7; standard library only), the
§3.1 emulation against hand-computed values (needs NumPy) and `--check` on the committed training
component fixtures (needs the pinned PyTorch environment; skipped without it).

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

import fixturelib  # noqa: E402
import gen_train  # noqa: E402  (imports without PyTorch; generating needs it)

try:
    HAVE_TORCH = gen_train.IMPORT_ERROR is None and not gen_train.gen_ops.version_problems()
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False
HAVE_NUMPY = gen_train.np is not None


class CoverageTest(unittest.TestCase):
    """testing-strategy.md §3.7: what the optimizer and schedule cases must cover."""

    def configs(self, kind: str) -> list[dict]:
        return [gen_train.full_config(c) for c in gen_train.OPTIMIZER_CASES if c.kind == kind]

    def test_optimizer_cases(self) -> None:
        sgd = self.configs("sgd")
        self.assertTrue(any(c["weight_decay"] != 0 for c in sgd), "SGD weight decay")
        self.assertTrue(any(c["momentum"] != 0 and c["dampening"] == 0 and not c["nesterov"] for c in sgd), "momentum")
        self.assertTrue(any(c["momentum"] != 0 and c["dampening"] != 0 for c in sgd), "momentum with dampening")
        self.assertTrue(any(c["nesterov"] for c in sgd), "Nesterov")
        self.assertTrue(any(c["momentum"] == 0 for c in sgd), "no momentum (no state)")
        for kind in ("adam", "adamw"):
            adam = self.configs(kind)
            default = gen_train.DEFAULTS[kind]
            self.assertTrue(any(c["betas"] == default["betas"] and c["eps"] == default["eps"] for c in adam), kind)
            self.assertTrue(any(c["betas"] != default["betas"] for c in adam), f"{kind}: non-default betas")
            self.assertTrue(any(c["eps"] != default["eps"] for c in adam), f"{kind}: non-default eps")
            self.assertTrue(any(c["weight_decay"] != 0 for c in adam), f"{kind}: weight decay")
            self.assertTrue(any(1 - c["betas"][0] >= 0.5 for c in adam), f"{kind}: the second lerp formula")
        self.assertTrue(any(c["weight_decay"] == 0 for c in self.configs("adamw")), "AdamW without decay")
        self.assertTrue(any(c.config == {"kind": k} for c in gen_train.OPTIMIZER_CASES for k in ("sgd", "adam", "adamw")))

    def test_optimizer_cases_stay_in_the_tolerance_bounds(self) -> None:
        # testing-strategy.md §2: lr ≤ 0.1 (params and gradients ≤ 5 are clipped by `draw`).
        for c in gen_train.OPTIMIZER_CASES:
            self.assertLessEqual(gen_train.full_config(c)["lr"], gen_train.MAX_LR, c.name)

    def test_schedule_cases(self) -> None:
        cases = gen_train.SCHEDULE_CASES
        kinds = {c.schedule["kind"] for c in cases}
        self.assertEqual(kinds, set(fixturelib.TRAIN_SCHEDULE_FIELDS))
        step = [c for c in cases if c.schedule["kind"] == "step"]
        self.assertTrue(any(c.total_steps > c.schedule["every"] for c in step), "a step boundary")
        self.assertTrue(any(c.schedule["gamma"] == 0 for c in step), "gamma = 0")
        for kind in ("linear_warmup", "cosine"):
            warm = [c for c in cases if c.schedule["kind"] == kind]
            self.assertTrue(any(0 < c.schedule["warmup_steps"] < c.total_steps for c in warm), f"{kind}: warmup")
            self.assertTrue(any(c.schedule["warmup_steps"] >= c.total_steps for c in warm), f"{kind}: W >= T")
        self.assertTrue(any(c.schedule["kind"] == "cosine" and c.schedule["warmup_steps"] == 0 for c in cases), "W = 0")

    def test_loss_cases(self) -> None:
        # testing-strategy.md §3.7: large logits, label smoothing 0, 0.1 and 1, BCE probabilities of
        # exactly 0 and 1, logits of ±200; plus C = 1 and a batch of 1 (i32 indices: the Rust test).
        cases = gen_train.LOSS_CASES
        self.assertEqual({c.kind for c in cases}, set(fixturelib.TRAIN_LOSS_FIELDS))
        ce = [c for c in cases if c.kind == "cross_entropy"]
        self.assertEqual({c.config.get("label_smoothing", 0.0) for c in ce} >= {0.0, 0.1, 1.0}, True)
        bce = [c for c in cases if c.kind == "binary_cross_entropy"]
        self.assertTrue(any(c.config.get("with_logits") for c in bce), "with_logits")
        self.assertTrue(any(not c.config.get("with_logits", False) for c in bce), "probabilities")
        for c in cases:
            self.assertLessEqual(set(c.config) - {"kind"}, set(fixturelib.TRAIN_LOSS_FIELDS[c.kind]), c.name)

    @unittest.skipUnless(HAVE_NUMPY, "needs NumPy")
    def test_loss_edge_values(self) -> None:
        import numpy as np

        def drawn(kind: str, pred) -> list:
            out = []
            for c in gen_train.LOSS_CASES:
                if c.kind == kind and pred(c):
                    out.append(c.make(np.random.Generator(np.random.PCG64(c.seed))))
            return out

        ce = drawn("cross_entropy", lambda c: True)
        self.assertTrue(any(np.abs(x).max() >= 1000 for x, _ in ce), "large logits")
        self.assertTrue(any(x.shape[1] == 1 for x, _ in ce), "C = 1")
        self.assertTrue(any(x.shape[0] == 1 for x, _ in ce), "a batch of 1")
        self.assertTrue(any(((x == 1000).any(axis=1) & (x == -1000).any(axis=1)).any() for x, _ in ce))
        probs = drawn("binary_cross_entropy", lambda c: not c.config.get("with_logits", False))
        self.assertTrue(any((x == 0).any() and (x == 1).any() for x, _ in probs), "probabilities 0 and 1")
        self.assertTrue(any(x.size == 1 for x, _ in probs), "a batch of 1")
        logits = drawn("binary_cross_entropy", lambda c: c.config.get("with_logits", False))
        self.assertTrue(any((x == 200).any() and (x == -200).any() for x, _ in logits), "logits of ±200")
        for x, y in probs + logits:
            self.assertTrue(((y >= 0) & (y <= 1)).all())
        for x, y in ce:
            self.assertTrue(((y >= 0) & (y < x.shape[1])).all())

    def test_case_names_are_unique_and_valid(self) -> None:
        keys = gen_train.case_keys()
        self.assertEqual(len(keys), len(set(keys)))
        for component, name in keys:
            self.assertIn(component, fixturelib.train_components())
            self.assertRegex(name, r"^[a-z0-9_]+$")

    def test_tolerances_are_the_documented_ones(self) -> None:
        self.assertEqual(gen_train.OPTIMIZER_TOLERANCE, fixturelib.TRAIN_OPTIMIZER_TOLERANCE)
        self.assertEqual(gen_train.CLIP_TOLERANCE, fixturelib.TRAIN_CLIP_TOLERANCE)
        self.assertEqual(gen_train.LOSS_TOLERANCE, fixturelib.TRAIN_LOSS_TOLERANCE)
        self.assertEqual(gen_train.COSINE_ATOL_PER_LR, fixturelib.TRAIN_COSINE_ATOL_PER_LR)


class ScheduleFormulaTest(unittest.TestCase):
    """training.md §3.2's formulas as `spec_lr` writes them (standard library only)."""

    def lr(self, kind: str, schedule: dict, lr: float, total: int) -> list[float]:
        c = gen_train.ScheduleCase("x", lr, total, {"kind": kind} | schedule, "")
        return [gen_train.spec_lr(c, k) for k in range(total)]

    def test_step_multiplies_once_per_interval(self) -> None:
        self.assertEqual(self.lr("step", {"every": 2, "gamma": 0.5}, 1.0, 5), [1.0, 1.0, 0.5, 0.5, 0.25])
        # Repeated multiplication, not one pow: 0.01 · 0.1 · 0.1 = 1e-4 exactly.
        self.assertEqual(self.lr("step", {"every": 1, "gamma": 0.1}, 0.01, 3)[2], 0.0001)

    def test_warmup(self) -> None:
        self.assertEqual(self.lr("linear_warmup", {"warmup_steps": 4}, 1.0, 6), [0.25, 0.5, 0.75, 1.0, 1.0, 1.0])

    def test_cosine(self) -> None:
        v = self.lr("cosine", {"warmup_steps": 2, "min_lr": 0.0}, 1.0, 6)
        self.assertEqual(v[:3], [0.5, 1.0, 1.0])
        self.assertGreater(v[-1], 0.0)
        self.assertEqual(self.lr("cosine", {"warmup_steps": 9, "min_lr": 0.0}, 1.0, 3), [1 / 9, 2 / 9, 3 / 9])


@unittest.skipUnless(HAVE_NUMPY, "needs NumPy")
class LossFormulaTest(unittest.TestCase):
    """training.md §3.3's formulas as `spec_loss` writes them, on hand-computed values."""

    def case(self, kind: str, **config) -> "gen_train.LossCase":
        return gen_train.LossCase(kind, "x", 0, {"kind": kind} | config, lambda rng: (None, None), "")

    def test_cross_entropy(self) -> None:
        import numpy as np

        x = np.array([[1000, 0, -1000]], dtype=np.float32)
        self.assertEqual(gen_train.spec_loss(self.case("cross_entropy"), x, np.array([2]))[0], 2000.0)
        v, _ = gen_train.spec_loss(self.case("cross_entropy", label_smoothing=0.1), x, np.array([2]))
        self.assertAlmostEqual(v, 0.9 * 2000 + 0.1 / 3 * 3000, places=9)
        v, _ = gen_train.spec_loss(self.case("cross_entropy", label_smoothing=0.5), np.array([[3.0]]), np.array([0]))
        self.assertEqual(v, 0.0)

    def test_binary_cross_entropy(self) -> None:
        import numpy as np

        x, y = np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0, 0.0])
        v, g = gen_train.spec_loss(self.case("binary_cross_entropy"), x, y)
        self.assertEqual(v, 50.0)
        self.assertEqual(list(g), [0.0, 0.0, -0.25e12, 0.25e12])
        v, _ = gen_train.spec_loss(self.case("binary_cross_entropy", with_logits=True), np.array([200.0, -200.0]),
                                   np.array([0.0, 0.0]))
        self.assertAlmostEqual(v, 100.0, places=12)


@unittest.skipUnless(HAVE_NUMPY, "needs NumPy")
class EmulationTest(unittest.TestCase):
    """The NumPy emulation of training.md §3.1 on hand-computable values."""

    def test_sgd_momentum(self) -> None:
        np = gen_train.np
        p = np.array([1.0], dtype=np.float32)
        g = [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]
        cfg = gen_train.DEFAULTS["sgd"] | {"lr": 0.5, "momentum": 0.5}
        out = gen_train.emulate_optimizer("sgd", cfg, p, g)
        # Step 1: buf = g = 1, p = 1 − 0.5 · 1 = 0.5. Step 2: buf = 1 · 0.5 + 2 = 2.5, p = 0.5 − 1.25.
        self.assertEqual([float(o["momentum_buffer"][0]) for o in out], [1.0, 2.5])
        self.assertEqual([float(o["param"][0]) for o in out], [0.5, -0.75])

    def test_adam_first_step_is_lr_times_sign(self) -> None:
        np = gen_train.np
        p = np.array([0.0, 0.0], dtype=np.float32)
        g = [np.array([3.0, -0.5], dtype=np.float32)]
        cfg = gen_train.DEFAULTS["adam"] | {"lr": 0.1, "eps": 0.0}
        out = gen_train.emulate_optimizer("adam", cfg, p, g)
        # m̂ = g and v̂ = g², so the first step is −lr · sign(g) (up to f32 rounding).
        np.testing.assert_allclose(out[0]["param"], [-0.1, 0.1], rtol=1e-6)

    def test_measure_rules(self) -> None:
        np = gen_train.np
        tol = {"atol": 0.0, "rtol": 0.0}
        nan, inf = float("nan"), float("inf")
        self.assertEqual(gen_train.measure(np.array([nan]), np.array([nan]), tol).ratio, 0.0)
        self.assertEqual(gen_train.measure(np.array([1.0]), np.array([nan]), tol).ratio, float("inf"))
        self.assertEqual(gen_train.measure(np.array([-inf]), np.array([inf]), tol).ratio, float("inf"))
        self.assertEqual(gen_train.measure(np.array([1.0]), np.array([2.0]), {"atol": 2.0, "rtol": 0.0}).ratio, 0.5)


@unittest.skipUnless(HAVE_TORCH, "needs the pinned PyTorch environment from tools/reference/requirements.txt")
class GeneratorCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out = os.path.join(self.tmp, "train")
        shutil.copytree(gen_train.DEFAULT_OUT, self.out)

    def check(self) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = gen_train.main(["--check", "--out", self.out])
        return code, err.getvalue()

    def test_committed_fixtures_match(self) -> None:
        self.assertEqual(self.check(), (0, ""))

    def test_a_changed_value_is_found(self) -> None:
        path = os.path.join(self.out, "optimizer.adam", "defaults.safetensors")
        with open(path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([last[0] ^ 1]))
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("changed: optimizer.adam/defaults.safetensors", err)

    def test_an_extra_file_is_found(self) -> None:
        src = os.path.join(self.out, "schedule.step", "every3.json")
        shutil.copy(src, os.path.join(self.out, "schedule.step", "stray.json"))
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("extra: schedule.step/stray.json", err)


if __name__ == "__main__":
    unittest.main()
