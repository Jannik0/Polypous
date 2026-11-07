#!/usr/bin/env python3
"""Tests for the `--check` comparison of gen_ops.py.

Run with: python -m unittest discover -s tools/reference -v

`CompareTreesTest` uses the standard library only and runs in CI. `GeneratorCheckTest` runs the real
generator and needs the pinned PyTorch environment (requirements.txt); it is skipped without it.
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


def write(root: str, rel: str, data: bytes) -> None:
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class CompareTreesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.committed = tempfile.mkdtemp()
        self.generated = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.committed)
        self.addCleanup(shutil.rmtree, self.generated)
        for root in (self.committed, self.generated):
            write(root, "core.add/a.json", b"{}\n")
            write(root, "core.add/a.safetensors", b"\x00\x01")

    def test_identical(self) -> None:
        self.assertEqual(fixturelib.compare_trees(self.committed, self.generated), [])

    def test_changed(self) -> None:
        write(self.committed, "core.add/a.safetensors", b"\x00\x02")
        self.assertEqual(
            fixturelib.compare_trees(self.committed, self.generated),
            ["changed: core.add/a.safetensors (differs from the generator's output)"],
        )

    def test_missing(self) -> None:
        write(self.generated, "core.cast/b.json", b"{}\n")
        self.assertEqual(
            fixturelib.compare_trees(self.committed, self.generated),
            ["missing: core.cast/b.json (generated, not committed)"],
        )

    def test_extra(self) -> None:
        write(self.committed, "core.add/old.json", b"{}\n")
        self.assertEqual(
            fixturelib.compare_trees(self.committed, self.generated),
            ["extra: core.add/old.json (committed, no case generates it)"],
        )

    def test_missing_committed_directory(self) -> None:
        problems = fixturelib.compare_trees(os.path.join(self.committed, "nope"), self.generated)
        self.assertEqual(len(problems), 2)
        self.assertTrue(all(p.startswith("missing: ") for p in problems))


try:
    import gen_ops  # noqa: E402

    HAVE_TORCH = gen_ops.IMPORT_ERROR is None and not gen_ops.version_problems()
except Exception:  # pragma: no cover - environment dependent
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "needs the pinned PyTorch environment from tools/reference/requirements.txt")
class GeneratorCheckTest(unittest.TestCase):
    """End to end: --check on a copy of the committed fixtures, then broken copies."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out = os.path.join(self.tmp, "ops")
        shutil.copytree(gen_ops.DEFAULT_OUT, self.out)

    def check(self) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = gen_ops.main(["--check", "--out", self.out])
        return code, err.getvalue()

    def test_committed_fixtures_pass(self) -> None:
        self.assertEqual(self.check(), (0, ""))

    def test_detects_changed(self) -> None:
        path = os.path.join(self.out, "core.add", "same_shape_f32.safetensors")
        with open(path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([last[0] ^ 1]))
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("changed: core.add/same_shape_f32.safetensors", err)

    def test_detects_missing(self) -> None:
        os.remove(os.path.join(self.out, "core.cast", "f32_to_bf16.json"))
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("missing: core.cast/f32_to_bf16.json", err)

    def test_detects_extra(self) -> None:
        write(self.out, "core.add/no_such_case.json", b"{}\n")
        code, err = self.check()
        self.assertEqual(code, 1)
        self.assertIn("extra: core.add/no_such_case.json", err)

    def test_regenerating_removes_stale_cases(self) -> None:
        write(self.out, "core.add/no_such_case.json", b"{}\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_ops.main(["--out", self.out]), 0)
        self.assertEqual(self.check(), (0, ""))
        self.assertEqual(fixturelib.compare_trees(self.out, gen_ops.DEFAULT_OUT), [])


if __name__ == "__main__":
    unittest.main()
