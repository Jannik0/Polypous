#!/usr/bin/env python3
"""Tests for check_fixtures.py (size budget and sidecar validation).

Run with: python3 -m unittest discover -s tools/ci -p 'test_*.py' -v
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_fixtures  # noqa: E402
import fixturelib  # noqa: E402  (put on sys.path by check_fixtures)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE = os.path.join(REPO, "fixtures", "small", "ops", "core.add", "same_shape_f32")
MODEL_SAMPLE = os.path.join(REPO, "fixtures", "small", "models", "cnn")
GRADS = os.path.join(REPO, "fixtures", "small", "grads")
INIT = os.path.join(REPO, "fixtures", "small", "init")
TRAINING_SAMPLE = os.path.join(REPO, "fixtures", "small", "training", "bn_dropout")
TRAIN = os.path.join(REPO, "fixtures", "small", "train")


def write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class SizeBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def test_small_tree_passes(self) -> None:
        write(os.path.join(self.root, "quant", "a.bin"), b"x" * 100)
        self.assertEqual(fixturelib.check_sizes(self.root, max_fixture=1000, max_total=10_000), [])

    def test_fixture_counts_all_files_of_a_case(self) -> None:
        # 600 + 600 bytes: each file is under the budget, the case together is not.
        write(os.path.join(self.root, "ops", "x", "c.safetensors"), b"x" * 600)
        write(os.path.join(self.root, "ops", "x", "c.json"), b"x" * 600)
        problems = fixturelib.check_sizes(self.root, max_fixture=1000, max_total=10_000)
        self.assertEqual(len(problems), 1)
        self.assertIn("ops/x/c: 1200 bytes", problems[0])

    def test_total_budget(self) -> None:
        for i in range(12):
            write(os.path.join(self.root, "ops", "x", f"c{i}.json"), b"x" * 900)
        problems = fixturelib.check_sizes(self.root, max_fixture=1000, max_total=10_000)
        self.assertEqual(len(problems), 1)
        self.assertIn("10800 bytes in total", problems[0])

    def test_defaults_are_the_documented_budget(self) -> None:
        self.assertEqual(fixturelib.MAX_FIXTURE_BYTES, 1024 * 1024)
        self.assertEqual(fixturelib.MAX_SMALL_TOTAL_BYTES, 50 * 1024 * 1024)


@unittest.skipUnless(os.path.exists(SAMPLE + ".json"), "committed fixtures not present")
class SidecarValidationTest(unittest.TestCase):
    """Copies one committed fixture and breaks it in different ways."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.dir = os.path.join(self.root, "ops", "core.add")
        os.makedirs(self.dir)
        for ext in (".json", ".safetensors"):
            shutil.copy(SAMPLE + ext, os.path.join(self.dir, "same_shape_f32" + ext))
        self.json = os.path.join(self.dir, "same_shape_f32.json")
        self.st = os.path.join(self.dir, "same_shape_f32.safetensors")

    def edit_sidecar(self, change) -> None:
        with open(self.json, encoding="utf-8") as f:
            sidecar = json.load(f)
        change(sidecar)
        with open(self.json, "w", encoding="utf-8", newline="\n") as f:
            f.write(fixturelib.canonical_json(sidecar))

    def test_copy_is_clean(self) -> None:
        self.assertEqual(check_fixtures.check(self.root), [])

    def test_wrong_shape_in_sidecar(self) -> None:
        self.edit_sidecar(lambda s: s["outputs"][0].update(shape=[4, 3]))
        self.assertTrue(any("'out' is f32 [4, 3]" in p for p in check_fixtures.check(self.root)))

    def test_wrong_dtype_in_sidecar(self) -> None:
        self.edit_sidecar(lambda s: s["inputs"][0].update(dtype="f16"))
        self.assertTrue(any("'a' is f16" in p for p in check_fixtures.check(self.root)))

    def test_undeclared_tensor(self) -> None:
        self.edit_sidecar(lambda s: s["inputs"].pop())
        self.assertTrue(any("undeclared tensors ['b']" in p for p in check_fixtures.check(self.root)))

    def test_subnormal_values_are_rejected(self) -> None:
        # The last element of the file's data: the smallest positive f32 subnormal.
        with open(self.st, "r+b") as f:
            f.seek(-4, os.SEEK_END)
            f.write(struct.pack("<I", 1))
        self.assertTrue(any("has subnormal values at flat indices [11]" in p for p in check_fixtures.check(self.root)))

    def test_subnormal_detection_per_float_dtype(self) -> None:
        # f16 0x0001 and bf16 0x0001 are subnormal; 0x0400 (f16) and 0x0080 (bf16) are the smallest
        # normals, 0x8000 is -0.0, and ints are never checked.
        tensors = {
            "h": ("F16", [0x0400, 0x8000, 0x0001]),
            "b": ("BF16", [0x0080, 0x0000, 0x0001]),
            "f": ("F32", [0x00800000, 0x80000000, 0x7FC00000]),
            "i": ("I32", [1, 2, 3]),
        }
        header, data = {}, b""
        for name, (dtype, bits) in tensors.items():
            code = "H" if dtype in ("F16", "BF16") else "I"
            chunk = struct.pack(f"<{len(bits)}{code}", *bits)
            header[name] = {"dtype": dtype, "shape": [len(bits)], "data_offsets": [len(data), len(data) + len(chunk)]}
            data += chunk
        raw = json.dumps(header).encode()
        path = os.path.join(self.root, "t.safetensors")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(raw)) + raw + data)
        problems = fixturelib.subnormal_problems(path, fixturelib.read_safetensors_header(path))
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(any("'h' has subnormal values at flat indices [2]" in p for p in problems))
        self.assertTrue(any("'b' has subnormal values at flat indices [2]" in p for p in problems))

    def params_after_inputs(self, sidecar: dict) -> None:
        """Moves input `b` into a `params` key right after `inputs` (the form for ops with params)."""
        items = list(sidecar.items())
        sidecar.clear()
        for key, value in items:
            if key == "inputs":
                sidecar["inputs"] = value[:1]
                sidecar["params"] = value[1:]
            else:
                sidecar[key] = value

    def test_params_key_after_inputs(self) -> None:
        self.edit_sidecar(self.params_after_inputs)
        self.assertEqual(check_fixtures.check(self.root), [])

    def test_params_are_checked_against_the_file(self) -> None:
        def change(sidecar: dict) -> None:
            self.params_after_inputs(sidecar)
            sidecar["params"][0]["shape"] = [2]

        self.edit_sidecar(change)
        self.assertTrue(any("params tensor 'b' is f32 [2]" in p for p in check_fixtures.check(self.root)))

    def test_params_key_in_the_wrong_place(self) -> None:
        def change(sidecar: dict) -> None:
            sidecar["inputs"] = sidecar["inputs"][:1]
            sidecar["params"] = []

        self.edit_sidecar(change)
        self.assertTrue(any("differ from the schema" in p for p in check_fixtures.check(self.root)))

    def test_case_name_must_match_file(self) -> None:
        self.edit_sidecar(lambda s: s.update(case="other"))
        self.assertTrue(any("does not match the file name" in p for p in check_fixtures.check(self.root)))

    def test_exact_tolerance_must_be_zero(self) -> None:
        self.edit_sidecar(lambda s: s.update(tolerance={"exact": True, "atol": 1e-5, "rtol": 0.0}))
        self.assertTrue(any("bad tolerance" in p for p in check_fixtures.check(self.root)))

    def test_non_canonical_json(self) -> None:
        with open(self.json, encoding="utf-8") as f:
            text = f.read()
        with open(self.json, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace("\n", "\n ", 1))
        self.assertTrue(any("canonical" in p for p in check_fixtures.check(self.root)))

    def test_truncated_safetensors(self) -> None:
        with open(self.st, "rb") as f:
            data = f.read()
        write(self.st, data[:-4])
        self.assertTrue(any("tensor data covers" in p for p in check_fixtures.check(self.root)))

    def test_huge_header_length_is_refused_before_reading(self) -> None:
        write(self.st, (2**62).to_bytes(8, "little") + b"{}")
        self.assertTrue(any("header length" in p for p in check_fixtures.check(self.root)))

    def test_safetensors_without_sidecar(self) -> None:
        os.remove(self.json)
        self.assertEqual(check_fixtures.check(self.root), ["ops/core.add/same_shape_f32.safetensors: no JSON sidecar"])


@unittest.skipUnless(os.path.isdir(MODEL_SAMPLE), "committed model fixtures not present")
class ModelValidationTest(unittest.TestCase):
    """Copies the committed CNN model (graph, weights, one case) and breaks it in different ways."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.dir = os.path.join(self.root, "models", "cnn")
        os.makedirs(self.dir)
        for name in ("model.graph.json", "model.safetensors", "batch1_f32.json", "batch1_f32.safetensors"):
            shutil.copy(os.path.join(MODEL_SAMPLE, name), os.path.join(self.dir, name))
        self.json = os.path.join(self.dir, "batch1_f32.json")
        self.graph = os.path.join(self.dir, "model.graph.json")
        self.weights = os.path.join(self.dir, "model.safetensors")

    def edit(self, path: str, change) -> None:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        change(doc)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(fixturelib.canonical_json(doc))

    def problems(self) -> list[str]:
        return check_fixtures.check(self.root)

    def assert_problem(self, text: str) -> None:
        problems = self.problems()
        self.assertTrue(any(text in p for p in problems), problems)

    def test_copy_is_clean(self) -> None:
        self.assertEqual(self.problems(), [])

    def test_graph_and_weights_are_one_fixture(self) -> None:
        sizes = fixturelib.fixture_sizes(self.root)
        self.assertEqual(sorted(sizes), ["models/cnn/batch1_f32", "models/cnn/model"])
        self.assertEqual(
            sizes["models/cnn/model"], os.path.getsize(self.graph) + os.path.getsize(self.weights)
        )

    def test_param_missing_from_the_weights(self) -> None:
        self.edit(self.graph, lambda g: g["params"].append({"name": "extra.weight", "dtype": "float", "shape": [2]}))
        self.assert_problem("param 'extra.weight' is missing")

    def test_weight_that_is_not_a_param(self) -> None:
        self.edit(self.graph, lambda g: g["params"].pop())
        self.assert_problem("tensor 'fc.bias' is not a param of the graph")

    def test_weight_with_another_shape(self) -> None:
        self.edit(self.graph, lambda g: g["params"][-1].update(shape=[11]))
        self.assert_problem("param 'fc.bias' is F32 [10], the graph declares F32 [11]")

    def test_case_inputs_must_match_the_graph(self) -> None:
        self.edit(self.graph, lambda g: g["inputs"][0].update(name="x"))
        self.assert_problem("inputs ['image'] differ from the graph's inputs ['x']")

    def test_case_outputs_must_match_the_graph(self) -> None:
        self.edit(self.json, lambda s: s["outputs"].append({"name": "probs", "dtype": "f32", "shape": [1, 10]}))
        self.assert_problem("outputs ['logits', 'probs'] differ from the graph's outputs ['logits']")

    def test_input_must_have_the_compute_dtype(self) -> None:
        self.edit(self.json, lambda s: s.update(compute_dtype="f16"))
        self.assert_problem("input 'image' is f32, the compute dtype is f16")

    def test_unknown_compute_dtype(self) -> None:
        self.edit(self.json, lambda s: s.update(compute_dtype="f64"))
        self.assert_problem("compute_dtype 'f64' is not one of")

    def test_sidecar_shape_against_the_file(self) -> None:
        self.edit(self.json, lambda s: s["outputs"][0].update(shape=[2, 10]))
        self.assert_problem("outputs tensor 'logits' is f32 [2, 10] in the sidecar, F32 [1, 10] in the file")

    def test_model_name_must_match_the_directory(self) -> None:
        self.edit(self.json, lambda s: s.update(model="mlp"))
        self.assert_problem("model 'mlp' does not match the directory name")

    def test_keys_follow_the_model_schema(self) -> None:
        self.edit(self.json, lambda s: s.pop("compute_dtype"))
        self.assert_problem("differ from the schema")

    def test_subnormal_weight(self) -> None:
        with open(self.weights, "r+b") as f:
            f.seek(-4, os.SEEK_END)
            f.write(struct.pack("<I", 1))
        self.assert_problem("model.safetensors: tensor 'fc.weight' has subnormal values")

    def test_missing_graph(self) -> None:
        os.remove(self.graph)
        self.assertEqual(self.problems(), [f"{self.dir}: no model.graph.json"])

    def test_case_without_tensors(self) -> None:
        os.remove(os.path.join(self.dir, "batch1_f32.safetensors"))
        self.assert_problem("batch1_f32.safetensors")

    def test_model_without_cases(self) -> None:
        for ext in (".json", ".safetensors"):
            os.remove(os.path.join(self.dir, "batch1_f32" + ext))
        self.assertEqual(self.problems(), [f"{self.dir}: no cases"])

    def test_unexpected_files(self) -> None:
        write(os.path.join(self.dir, "notes.txt"), b"x")
        write(os.path.join(self.root, "models", "stray.json"), b"{}")
        problems = self.problems()
        self.assertTrue(any("notes.txt: unexpected file in a model directory" in p for p in problems), problems)
        self.assertIn("models/stray.json: a file directly in models/ (expected models/<model>/)", problems)


@unittest.skipUnless(os.path.isdir(GRADS), "committed gradient fixtures not present")
class GradSidecarValidationTest(unittest.TestCase):
    """Copies committed gradient fixtures (schema version 2, testing-strategy.md §3.6) and breaks them."""

    CASES = {
        "linear": ("nn.linear", "2d_f32"),
        "dropout": ("nn.dropout", "train_p05_f32"),
        "bn": ("nn.batch_norm", "train_rank2_f32"),
        "split": ("core.split", "even_f32"),
        "gather": ("core.gather", "repeated_dim0_f32"),
    }

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        for op, name in self.CASES.values():
            os.makedirs(os.path.join(self.root, "grads", op), exist_ok=True)
            for ext in (".json", ".safetensors"):
                shutil.copy(os.path.join(GRADS, op, name + ext), os.path.join(self.root, "grads", op, name + ext))

    def path(self, key: str, ext: str = ".json") -> str:
        op, name = self.CASES[key]
        return os.path.join(self.root, "grads", op, name + ext)

    def edit(self, key: str, change) -> None:
        with open(self.path(key), encoding="utf-8") as f:
            sidecar = json.load(f)
        change(sidecar)
        with open(self.path(key), "w", encoding="utf-8", newline="\n") as f:
            f.write(fixturelib.canonical_json(sidecar))

    def assert_problem(self, text: str) -> None:
        problems = check_fixtures.check(self.root)
        self.assertTrue(any(text in p for p in problems), problems)

    def test_copy_is_clean(self) -> None:
        self.assertEqual(check_fixtures.check(self.root), [])

    def test_schema_version_must_be_2(self) -> None:
        self.edit("linear", lambda s: s.update(schema_version=1))
        self.assert_problem("schema_version 1 is not 2")

    def test_a_schema_1_sidecar_is_refused_under_grads(self) -> None:
        op_dir = os.path.join(self.root, "grads", "core.add")
        os.makedirs(op_dir)
        for ext in (".json", ".safetensors"):
            shutil.copy(SAMPLE + ext, os.path.join(op_dir, "same_shape_f32" + ext))
        self.assert_problem("differ from schema 2")

    def test_keys_in_order(self) -> None:
        def swap(s: dict) -> None:
            items = list(s.items())
            i = [k for k, _ in items].index("upstream")
            items[i], items[i + 1] = items[i + 1], items[i]
            s.clear()
            s.update(items)

        self.edit("linear", swap)
        self.assert_problem("differ from schema 2")

    def test_mask_only_for_dropout_in_training_mode(self) -> None:
        self.edit("dropout", lambda s: s.update(mode="inference"))
        self.assert_problem("differ from schema 2")

    def test_running_stats_only_for_batch_norm_in_training_mode(self) -> None:
        self.edit("bn", lambda s: s.update(mode="inference"))
        self.assert_problem("differ from schema 2")

    def test_training_mode_only_for_dropout_and_batch_norm(self) -> None:
        self.edit("linear", lambda s: s.update(mode="training"))
        self.assert_problem("have training-mode cases")

    def test_unknown_mode(self) -> None:
        self.edit("linear", lambda s: s.update(mode="eval"))
        self.assert_problem("mode 'eval' is not one of")

    def test_mask_must_have_x_shape(self) -> None:
        self.edit("dropout", lambda s: s["mask"].update(shape=[4, 3]))
        self.assert_problem("mask must be named mask and have x's shape")

    def test_upstream_for_every_float_port(self) -> None:
        self.edit("split", lambda s: s["upstream"].pop())
        self.assert_problem("upstream ports ['0', '1'] differ from the float output ports ['0', '1', '2']")

    def test_upstream_name_follows_the_port(self) -> None:
        self.edit("linear", lambda s: s["upstream"][0].update(name="grad_y"))
        self.assert_problem("must be grad_out")

    def test_no_gradient_for_running_stats(self) -> None:
        self.edit("bn", lambda s: s["gradients"].append(
            {"name": "grad_running_mean", "of": "running_mean", "dtype": "f32", "shape": [3]}))
        self.assert_problem("every param but the running statistics")

    def test_gradient_of_every_float_input_and_param(self) -> None:
        self.edit("linear", lambda s: s["gradients"].pop())
        self.assert_problem("gradients are of ['x', 'weight']")

    def test_integer_inputs_get_no_gradient(self) -> None:
        self.edit("gather", lambda s: s["gradients"].append(
            {"name": "grad_index", "of": "index", "dtype": "f32", "shape": [4, 4]}))
        self.assert_problem("gradients are of ['x', 'index']")

    def test_gradient_shape(self) -> None:
        self.edit("linear", lambda s: s["gradients"][0].update(shape=[6, 3]))
        self.assert_problem("must be grad_x with x's shape")

    def test_running_stats_names(self) -> None:
        self.edit("bn", lambda s: s["running_stats"].reverse())
        self.assert_problem("running_stats must be new_running_mean, new_running_var")

    def test_gradients_are_f32(self) -> None:
        self.edit("linear", lambda s: s["gradients"][0].update(dtype="f16"))
        self.assert_problem("gradients tensor 'grad_x' is f16, not f32")

    def test_case_name_ends_in_f32(self) -> None:
        for ext in (".json", ".safetensors"):
            os.rename(self.path("linear", ext), self.path("linear", ext).replace("2d_f32", "2d"))
        with open(self.path("linear").replace("2d_f32", "2d"), encoding="utf-8") as f:
            sidecar = json.load(f)
        sidecar["case"] = "2d"
        with open(self.path("linear").replace("2d_f32", "2d"), "w", encoding="utf-8", newline="\n") as f:
            f.write(fixturelib.canonical_json(sidecar))
        self.assert_problem("gradient case names end in _f32")

    def test_bad_gradient_tolerance(self) -> None:
        self.edit("linear", lambda s: s.update(gradient_tolerance={"exact": False, "atol": -1.0, "rtol": 0.0}))
        self.assert_problem("bad tolerance")

    def test_sidecar_against_the_file(self) -> None:
        self.edit("linear", lambda s: s["outputs"][0].update(shape=[3, 5]))
        self.assert_problem("outputs tensor 'out' is f32 [3, 5] in the sidecar, F32 [3, 4] in the file")

    def test_undeclared_tensor(self) -> None:
        self.edit("bn", lambda s: s.update(running_stats=s["running_stats"][:1]))
        self.assert_problem("undeclared tensors ['new_running_var']")

    def test_subnormal_gradient(self) -> None:
        header = fixturelib.read_safetensors_header(self.path("linear", ".safetensors"))
        start = header["grad_x"]["data_offsets"][0]
        with open(self.path("linear", ".safetensors"), "r+b") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            f.seek(8 + n + start)
            f.write(struct.pack("<I", 1))
        self.assert_problem("tensor 'grad_x' has subnormal values at flat indices [0]")

    def test_file_directly_in_grads(self) -> None:
        write(os.path.join(self.root, "grads", "stray.json"), b"{}")
        self.assert_problem("grads/stray.json: not in grads/<op>/")


class CommittedFixturesTest(unittest.TestCase):
    def test_committed_fixtures_are_clean(self) -> None:
        self.assertEqual(check_fixtures.check(os.path.join(REPO, "fixtures", "small")), [])


class InitValidationTest(unittest.TestCase):
    """Copies committed initial-value fixtures (testing-strategy.md §3.9) and breaks them."""

    CASES = {"kaiming": ("kaiming_normal", "conv2d_weight"), "uniform": ("uniform", "unused")}

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        for kind, name in self.CASES.values():
            os.makedirs(os.path.join(self.root, "init", kind), exist_ok=True)
            for ext in (".json", ".safetensors"):
                shutil.copy(os.path.join(INIT, kind, name + ext), os.path.join(self.root, "init", kind, name + ext))

    def edit(self, key: str, change) -> None:
        kind, name = self.CASES[key]
        path = os.path.join(self.root, "init", kind, name + ".json")
        with open(path, encoding="utf-8") as f:
            sidecar = json.load(f)
        change(sidecar)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(fixturelib.canonical_json(sidecar))

    def assert_problem(self, text: str) -> None:
        problems = check_fixtures.check(self.root)
        self.assertTrue(any(text in p for p in problems), problems)

    def test_copy_is_clean(self) -> None:
        self.assertEqual(check_fixtures.check(self.root), [])

    def test_keys(self) -> None:
        self.edit("uniform", lambda s: s.pop("notes"))
        self.assert_problem("differ from the schema")

    def test_kind_must_match_the_directory(self) -> None:
        self.edit("uniform", lambda s: s.update(kind="normal"))
        self.assert_problem("does not match the directory name")

    def test_fans_must_match_the_shape(self) -> None:
        self.edit("kaiming", lambda s: s["param"]["fans"].update(fan_in=1))
        self.assert_problem("differ from the fans of its shape")

    def test_fan_kind_needs_a_role_with_fans(self) -> None:
        self.edit("kaiming", lambda s: s["param"].update(role={"op": "nn.linear", "param": "bias"}, fans=None))
        self.assert_problem("needs a role with fans")

    def test_fans_without_a_role_with_fans(self) -> None:
        self.edit("uniform", lambda s: s["param"].update(fans={"fan_in": 1, "fan_out": 1}))
        self.assert_problem("must be null")

    def test_tolerance(self) -> None:
        self.edit("uniform", lambda s: s.update(tolerance={"ulp": 1}))
        self.assert_problem("tolerance must be")
        self.edit("uniform", lambda s: s.update(tolerance={"exact": True, "atol": 0, "rtol": 0}))
        self.assert_problem("tolerance must be")

    def test_values_shape(self) -> None:
        self.edit("uniform", lambda s: s["param"].update(shape=[7, 5]))
        self.assert_problem("with the param's shape")


class TrainingValidationTest(unittest.TestCase):
    """Copies the committed bn_dropout training model (testing-strategy.md §3.10) and breaks it."""

    CASE = "batch4_step3_f32"

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self.dir = os.path.join(self.root, "training", "bn_dropout")
        os.makedirs(self.dir)
        for name in ("model.graph.json", "model.safetensors", self.CASE + ".json", self.CASE + ".safetensors"):
            shutil.copy(os.path.join(TRAINING_SAMPLE, name), os.path.join(self.dir, name))
        self.json = os.path.join(self.dir, self.CASE + ".json")
        self.graph = os.path.join(self.dir, "model.graph.json")

    def edit(self, path: str, change) -> None:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        change(doc)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(fixturelib.canonical_json(doc))

    def assert_problem(self, text: str) -> None:
        problems = check_fixtures.check(self.root)
        self.assertTrue(any(text in p for p in problems), problems)

    def test_copy_is_clean(self) -> None:
        self.assertEqual(check_fixtures.check(self.root), [])

    def test_keys_follow_the_mode(self) -> None:
        self.edit(self.json, lambda s: s.update(mode="inference"))
        self.assert_problem("differ from schema 2 for mode 'inference'")

    def test_schema_version(self) -> None:
        self.edit(self.json, lambda s: s.update(schema_version=1))
        self.assert_problem("is not 2")

    def test_training_seed_and_step(self) -> None:
        self.edit(self.json, lambda s: s.update(training={"seed": -1, "step": 3}))
        self.assert_problem("both u64")

    def test_masks_name_every_dropout_path(self) -> None:
        self.edit(self.json, lambda s: s["masks"].pop())
        self.assert_problem("masks must be mask/<node path>")

    def test_running_stats_name_every_statistic(self) -> None:
        self.edit(self.json, lambda s: s["running_stats"].pop())
        self.assert_problem("running_stats must be")

    def test_gradients_only_of_trainable_params(self) -> None:
        def add(s: dict) -> None:
            s["gradients"].append({"name": "grad/bn.running_mean", "param": "bn.running_mean", "dtype": "f32", "shape": [3]})
        self.edit(self.json, add)
        self.assert_problem("gradients must be grad/<param> of trainable params")

    def test_a_param_marked_untrainable_has_no_gradient(self) -> None:
        self.edit(self.graph, lambda g: g["params"][0].update(trainable=False))
        self.assert_problem("gradients must be grad/<param> of trainable params")

    def test_upstream_names_the_output(self) -> None:
        self.edit(self.json, lambda s: s["upstream"][0].update(output="z"))
        self.assert_problem("upstream must be")

    def test_only_f32(self) -> None:
        self.edit(self.json, lambda s: s.update(compute_dtype="f16"))
        self.assert_problem("compute_dtype must be f32")

    def test_mask_must_be_bool(self) -> None:
        self.edit(self.json, lambda s: s["masks"][0].update(dtype="f32"))
        self.assert_problem("must be bool")

    def test_file_directly_in_training(self) -> None:
        write(os.path.join(self.root, "training", "stray.json"), b"{}")
        self.assert_problem("a file directly in training/")


class TrainValidationTest(unittest.TestCase):
    """Copies committed training component fixtures (testing-strategy.md §3.7) and breaks them."""

    CASES = {
        "sgd": ("optimizer.sgd", "momentum"),
        "plain": ("optimizer.sgd", "weight_decay"),
        "adam": ("optimizer.adam", "defaults"),
        "clip": ("gradient_clip", "clipped"),
        "cosine": ("schedule.cosine", "warmup5"),
        "step": ("schedule.step", "every3"),
    }

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        for component, name in self.CASES.values():
            os.makedirs(os.path.join(self.root, "train", component), exist_ok=True)
            for ext in (".json", ".safetensors"):
                src = os.path.join(TRAIN, component, name + ext)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(self.root, "train", component, name + ext))

    def path(self, key: str, ext: str = ".json") -> str:
        component, name = self.CASES[key]
        return os.path.join(self.root, "train", component, name + ext)

    def edit(self, key: str, change) -> None:
        with open(self.path(key), encoding="utf-8") as f:
            sidecar = json.load(f)
        change(sidecar)
        with open(self.path(key), "w", encoding="utf-8", newline="\n") as f:
            f.write(fixturelib.canonical_json(sidecar))

    def assert_problem(self, text: str) -> None:
        problems = check_fixtures.check(self.root)
        self.assertTrue(any(text in p for p in problems), problems)

    def test_copy_is_clean(self) -> None:
        self.assertEqual(check_fixtures.check(self.root), [])

    def test_schema_version(self) -> None:
        self.edit("adam", lambda s: s.update(schema_version=2))
        self.assert_problem("schema_version 2 is not 1")

    def test_keys_in_order(self) -> None:
        def swap(s: dict) -> None:
            items = list(s.items())
            items[4], items[5] = items[5], items[4]
            s.clear()
            s.update(items)

        self.edit("adam", swap)
        self.assert_problem("differ from the schema")

    def test_values_only_for_schedules(self) -> None:
        self.edit("adam", lambda s: s.update(values=[]))
        self.assert_problem("differ from the schema")

    def test_unknown_component(self) -> None:
        self.edit("adam", lambda s: s.update(component="optimizer.rmsprop"))
        self.assert_problem("component 'optimizer.rmsprop' is not one of")

    def test_component_must_match_the_directory(self) -> None:
        self.edit("adam", lambda s: s.update(component="optimizer.adamw", config={"kind": "adamw"}))
        self.assert_problem("does not match the directory name")

    def test_case_name_must_match_file(self) -> None:
        self.edit("clip", lambda s: s.update(case="other"))
        self.assert_problem("does not match the file name")

    def test_config_kind_and_fields(self) -> None:
        self.edit("adam", lambda s: s.update(config={"kind": "sgd"}))
        self.assert_problem("needs kind 'adam'")
        self.edit("adam", lambda s: s.update(config={"kind": "adam", "momentum": 0.9}))
        self.assert_problem("needs kind 'adam'")

    def test_optimizer_state_outputs(self) -> None:
        # momentum ≠ 0 has a momentum_buffer per step.
        self.edit("sgd", lambda s: s["config"].pop("momentum"))
        self.assert_problem("outputs ['param_1', 'param_2', 'param_3']")
        self.edit("plain", lambda s: s["config"].update(momentum=0.5))
        self.assert_problem("'momentum_buffer_1'")

    def test_optimizer_tolerance(self) -> None:
        self.edit("adam", lambda s: s.update(tolerance={"exact": False, "atol": 1e-5, "rtol": 1e-6}))
        self.assert_problem("tolerance must be")

    def test_optimizer_shapes(self) -> None:
        self.edit("adam", lambda s: s["outputs"][0].update(shape=[8, 4]))
        self.assert_problem("every tensor has the param's shape")

    def test_sidecar_against_the_file(self) -> None:
        def reshape(s: dict) -> None:
            for e in s["inputs"] + s["outputs"]:
                e["shape"] = [32]

        self.edit("adam", reshape)
        self.assert_problem("is f32 [32] in the sidecar, F32 [4, 8] in the file")

    def test_clip_names_and_shapes(self) -> None:
        self.edit("clip", lambda s: s["outputs"][0].update(name="total"))
        self.assert_problem("outputs ['total_norm'")
        self.edit("clip", lambda s: s["outputs"][0].update(name="total_norm", shape=[]))
        self.assert_problem("total_norm has shape [1]")

    def test_clip_config(self) -> None:
        self.edit("clip", lambda s: s.update(config={"max_norm": 0}))
        self.assert_problem("max_norm")

    def test_clip_tolerance(self) -> None:
        self.edit("clip", lambda s: s.update(tolerance={"exact": False, "atol": 1e-6, "rtol": 1e-6}))
        self.assert_problem("tolerance must be")

    def test_schedule_values_per_step(self) -> None:
        self.edit("step", lambda s: s.update(values=s["values"][:-1]))
        self.assert_problem("values must be total_steps (10) finite numbers")

    def test_schedule_tolerances(self) -> None:
        self.edit("step", lambda s: s.update(tolerance={"exact": False, "atol": 1e-15, "rtol": 0}))
        self.assert_problem("tolerance must be {'exact': True")
        self.edit("cosine", lambda s: s.update(tolerance={"exact": False, "atol": 1e-14, "rtol": 0}))
        self.assert_problem("tolerance must be {'exact': False, 'atol': 1e-15")

    def test_schedule_config(self) -> None:
        self.edit("cosine", lambda s: s["config"]["schedule"].update(gamma=0.1))
        self.assert_problem("schedule needs kind 'cosine'")
        self.edit("cosine", lambda s: s["config"].pop("total_steps"))
        self.assert_problem("must have exactly the keys lr, total_steps, schedule")

    def test_schedule_has_no_tensors(self) -> None:
        shutil.copy(self.path("clip", ".safetensors"), self.path("step", ".safetensors"))
        self.assert_problem("a schedule case has no safetensors file")
        self.edit("step", lambda s: s.update(inputs=[{"name": "x", "dtype": "f32", "shape": [1]}]))
        self.assert_problem("a schedule case has no inputs and no outputs")

    def test_subnormal_state(self) -> None:
        header = fixturelib.read_safetensors_header(self.path("adam", ".safetensors"))
        start = header["exp_avg_2"]["data_offsets"][0]
        with open(self.path("adam", ".safetensors"), "r+b") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            f.seek(8 + n + start)
            f.write(struct.pack("<I", 1))
        self.assert_problem("tensor 'exp_avg_2' has subnormal values at flat indices [0]")

    def test_safetensors_without_sidecar(self) -> None:
        os.remove(self.path("clip"))
        self.assert_problem("no JSON sidecar")

    def test_file_directly_in_train(self) -> None:
        write(os.path.join(self.root, "train", "stray.json"), b"{}")
        self.assert_problem("train/stray.json: not in train/<component>/")


if __name__ == "__main__":
    unittest.main()
