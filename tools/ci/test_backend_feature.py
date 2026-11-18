#!/usr/bin/env python3
"""Tests for backend_feature.py.

Run with: python3 -m unittest discover -s tools/ci -p 'test_*.py' -v
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backend_feature as bf  # noqa: E402

METADATA = {
    "packages": [
        {"name": "polypous-backend", "features": {"cpu": [], "vulkan": [], "default": ["cpu"]}},
        {"name": "polypous-cli", "features": {"cpu": [], "vulkan": ["polypous/vulkan"]}},
    ]
}


class BackendFeature(unittest.TestCase):
    def run_script(self, *argv: str, metadata: object = METADATA) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metadata.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = bf.main([*argv, "--metadata", path])
        return code, out.getvalue(), err.getvalue()

    def test_default_package_is_backend(self) -> None:
        code, out, err = self.run_script("vulkan")
        self.assertEqual(code, 0)
        self.assertEqual(out, "available=true\nargs=--features polypous-backend/vulkan\n")
        self.assertEqual(err, "")

    def test_other_package(self) -> None:
        code, out, _ = self.run_script("vulkan", "--package", "polypous-cli")
        self.assertEqual(code, 0)
        self.assertEqual(out, "available=true\nargs=--features polypous-cli/vulkan\n")

    def test_missing_feature_falls_back_with_warning(self) -> None:
        code, out, err = self.run_script("metal", "--package", "polypous-cli")
        self.assertEqual(code, 0)
        self.assertEqual(out, "available=false\nargs=\n")
        self.assertIn("::warning", err)
        self.assertIn("polypous-cli", err)

    def test_missing_package_is_an_error(self) -> None:
        code, out, err = self.run_script("vulkan", "--package", "polypous-nope")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("polypous-nope", err)

    def test_malformed_metadata_is_an_error(self) -> None:
        code, _, err = self.run_script("vulkan", metadata={"no_packages": []})
        self.assertEqual(code, 2)
        self.assertIn("cannot read cargo metadata", err)


if __name__ == "__main__":
    unittest.main()
