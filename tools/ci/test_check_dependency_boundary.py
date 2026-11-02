#!/usr/bin/env python3
"""Tests for check_dependency_boundary.py.

Run with: python3 -m unittest discover -s tools/ci -p 'test_*.py' -v

`fixtures/workspace-clean.json` is trimmed `cargo metadata --format-version 1 --all-features`
output of the 10-crate skeleton workspace (paths replaced by /workspace). Each test copies it and
injects one deliberate violation (or an allowed dependency) to check that the script reacts.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_dependency_boundary as cdb  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "workspace-clean.json")


def load_clean() -> dict:
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def pkg(meta: dict, name: str) -> dict:
    return next(p for p in meta["packages"] if p["name"] == name)


def node(meta: dict, name: str) -> dict:
    return next(n for n in meta["resolve"]["nodes"] if n["id"] == pkg(meta, name)["id"])


def add_registry_crate(meta: dict, name: str, deps: tuple[str, ...] = ()) -> str:
    """Add a crates.io package (with resolved deps on other packages) and return its id."""
    pkg_id = f"registry+https://github.com/rust-lang/crates.io-index#{name}@1.0.0"
    meta["packages"].append(
        {
            "name": name,
            "version": "1.0.0",
            "id": pkg_id,
            "dependencies": [
                {"name": d, "req": "^1", "kind": None, "rename": None, "optional": False, "target": None}
                for d in deps
            ],
        }
    )
    meta["resolve"]["nodes"].append(
        {
            "id": pkg_id,
            "deps": [
                {"name": d.replace("-", "_"), "pkg": pkg(meta, d)["id"], "dep_kinds": [{"kind": None, "target": None}]}
                for d in deps
            ],
        }
    )
    return pkg_id


def add_dep(
    meta: dict,
    from_name: str,
    to_name: str,
    kind: str | None = None,
    *,
    rename: str | None = None,
    optional: bool = False,
    target: str | None = None,
) -> None:
    """Declare and resolve a dependency `from_name -> to_name` (registry crate created if needed)."""
    if not any(p["name"] == to_name for p in meta["packages"]):
        add_registry_crate(meta, to_name)
    pkg(meta, from_name)["dependencies"].append(
        {"name": to_name, "req": "^1", "kind": kind, "rename": rename, "optional": optional, "target": target}
    )
    node(meta, from_name)["deps"].append(
        {
            "name": (rename or to_name).replace("-", "_"),
            "pkg": pkg(meta, to_name)["id"],
            "dep_kinds": [{"kind": kind, "target": target}],
        }
    )


class CheckTests(unittest.TestCase):
    def assert_violation(self, meta: dict, *fragments: str) -> None:
        violations = cdb.check(meta)
        self.assertTrue(violations, "expected a violation, got none")
        text = "\n".join(violations)
        for fragment in fragments:
            self.assertIn(fragment, text)

    def test_clean_workspace_passes(self):
        self.assertEqual(cdb.check(load_clean()), [])

    def test_burn_allowed_in_backend_and_quant(self):
        meta = load_clean()
        add_dep(meta, "polypous-backend", "burn")
        add_dep(meta, "polypous-backend", "cubecl")
        add_dep(meta, "polypous-quant", "cubecl")
        self.assertEqual(cdb.check(meta), [])

    def test_burn_in_runtime_fails(self):
        meta = load_clean()
        add_dep(meta, "polypous-runtime", "burn")
        self.assert_violation(meta, "polypous-runtime", "`burn`", "ADR-0007")

    def test_cubecl_dev_dependency_in_facade_fails(self):
        meta = load_clean()
        add_dep(meta, "polypous", "cubecl-runtime", kind="dev")
        self.assert_violation(meta, "polypous:", "`cubecl-runtime`", "dev")

    def test_renamed_optional_burn_dependency_fails(self):
        meta = load_clean()
        add_dep(meta, "polypous-train", "burn-tensor", rename="bt", optional=True)
        self.assert_violation(meta, "polypous-train", "`burn-tensor`", "renamed to `bt`", "optional")

    def test_burn_in_pure_crate_fails(self):
        meta = load_clean()
        add_dep(meta, "polypous-ir", "burn-ndarray")
        self.assert_violation(meta, "polypous-ir", "`burn-ndarray`")

    def test_gpu_dependencies_in_pure_crates_fail(self):
        for crate, dep in [
            ("polypous-ir", "wgpu"),
            ("polypous-formats", "cudarc"),
            ("polypous-polyp", "ash"),
            ("polypous-ir", "vulkano"),
            ("polypous-formats", "metal"),
        ]:
            with self.subTest(crate=crate, dep=dep):
                meta = load_clean()
                add_dep(meta, crate, dep, target="cfg(unix)")
                self.assert_violation(meta, crate, f"`{dep}`", "pure Rust", "target cfg(unix)")

    def test_gpu_dependency_outside_pure_crates_allowed(self):
        # wgpu itself is not Burn; only the pure crates are restricted from GPU crates.
        meta = load_clean()
        add_dep(meta, "polypous-backend", "wgpu")
        add_dep(meta, "polypous-runtime", "cudarc")
        self.assertEqual(cdb.check(meta), [])

    def test_transitive_gpu_dependency_through_registry_crate_fails(self):
        meta = load_clean()
        add_registry_crate(meta, "wgpu-core")
        add_registry_crate(meta, "innocent-looking", deps=("wgpu-core",))
        add_dep(meta, "polypous-polyp", "innocent-looking")
        self.assert_violation(
            meta, "polypous-polyp: transitively", "`wgpu-core`", "polypous-polyp -> innocent-looking -> wgpu-core"
        )

    def test_transitive_gpu_dependency_through_workspace_crate_fails(self):
        meta = load_clean()
        add_dep(meta, "polypous-backend", "burn")
        add_dep(meta, "polypous-formats", "polypous-backend")
        self.assert_violation(meta, "polypous-formats -> polypous-backend -> burn")
        # polypous-polyp depends on polypous-formats, so it is affected too.
        self.assert_violation(meta, "polypous-polyp -> polypous-formats -> polypous-backend -> burn")

    def test_dev_dependencies_of_other_crates_are_not_followed(self):
        meta = load_clean()
        add_registry_crate(meta, "wgpu")
        add_registry_crate(meta, "helper")
        add_dep(meta, "polypous-ir", "helper")
        # helper's dev-dependency on wgpu is not linked into polypous-ir.
        node(meta, "helper")["deps"].append(
            {"name": "wgpu", "pkg": pkg(meta, "wgpu")["id"], "dep_kinds": [{"kind": "dev", "target": None}]}
        )
        self.assertEqual(cdb.check(meta), [])

    def test_gpu_name_pattern(self):
        for name in ["burn", "burn-wgpu", "cubecl-cuda", "wgpu-hal", "cudarc", "ash", "ash-window", "vulkano", "metal"]:
            self.assertTrue(cdb.GPU_RE.match(name), name)
        for name in ["ashpd", "serde", "half", "bytemuck", "polypous-ir", "safetensors"]:
            self.assertFalse(cdb.GPU_RE.match(name), name)

    def test_no_deps_metadata_still_checks_direct_deps(self):
        meta = load_clean()
        add_dep(meta, "polypous-runtime", "burn")
        del meta["resolve"]
        self.assert_violation(meta, "polypous-runtime", "`burn`")

    def test_missing_expected_crate_is_an_error(self):
        meta = load_clean()
        ir_id = pkg(meta, "polypous-ir")["id"]
        pkg(meta, "polypous-ir")["name"] = "polypous-graph"
        self.assertIn(ir_id, meta["workspace_members"])
        with self.assertRaisesRegex(ValueError, "polypous-ir"):
            cdb.check(meta)


class CliTests(unittest.TestCase):
    def run_cli(self, meta: dict) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metadata.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cdb.main(["--metadata", path])
        return code, out.getvalue(), err.getvalue()

    def test_clean_exit_0(self):
        code, out, _ = self.run_cli(load_clean())
        self.assertEqual(code, 0)
        self.assertIn("passed", out)

    def test_violation_exit_1(self):
        meta = copy.deepcopy(load_clean())
        add_dep(meta, "polypous-runtime", "burn")
        code, _, err = self.run_cli(meta)
        self.assertEqual(code, 1)
        self.assertIn("FAILED", err)
        self.assertIn("polypous-runtime", err)

    def test_invalid_input_exit_2(self):
        code, _, err = self.run_cli({"packages": []})
        self.assertEqual(code, 2)
        self.assertIn("workspace_members", err)


if __name__ == "__main__":
    unittest.main()
