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


class CommittedFixturesTest(unittest.TestCase):
    def test_committed_fixtures_are_clean(self) -> None:
        self.assertEqual(check_fixtures.check(os.path.join(REPO, "fixtures", "small")), [])


if __name__ == "__main__":
    unittest.main()
