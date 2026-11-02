#!/usr/bin/env python3
"""Dependency-boundary check for the Polypous workspace.

Enforces two rules from the architecture (ADR-0007, CLAUDE.md "Hard rules"):

1. Only `polypous-backend` and `polypous-quant` may depend directly on `burn*` / `cubecl*` crates.
   Every dependency kind (normal, dev, build), every target and optional dependencies count.
2. The pure crates `polypous-ir`, `polypous-formats` and `polypous-polyp` must not have any
   GPU-related dependency (`burn*`, `cubecl*`, `wgpu*`, `cudarc*`, `ash`, `vulkan*`, `metal*`),
   neither directly nor transitively (for example through another workspace crate).

Input is `cargo metadata --format-version 1` output. By default the script runs
`cargo metadata --format-version 1 --all-features --locked` itself; `--metadata FILE` reads a saved
JSON document instead (used by the tests in `test_check_dependency_boundary.py`).

Exit status: 0 if the workspace is clean, 1 on a violation, 2 on a usage or input error.
Standard library only, so it runs on every CI runner without setup.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any

# Crates allowed to depend on Burn/CubeCL (ADR-0007).
BURN_ALLOWED = frozenset({"polypous-backend", "polypous-quant"})

# Crates that must stay free of any GPU dependency (CLAUDE.md "Hard rules").
PURE_CRATES = frozenset({"polypous-ir", "polypous-formats", "polypous-polyp"})

BURN_RE = re.compile(r"^(burn|cubecl)")
GPU_RE = re.compile(r"^(burn|cubecl|wgpu|cudarc|vulkan|metal)|^ash([-_]|$)")


def load_metadata(path: str | None) -> dict[str, Any]:
    """Load cargo metadata from `path`, or run `cargo metadata` when `path` is None."""
    if path is not None:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    cmd = ["cargo", "metadata", "--format-version", "1", "--all-features", "--locked"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _dep_desc(dep: dict[str, Any]) -> str:
    """Human-readable qualifiers of a declared dependency (kind, target, optional)."""
    parts = [dep.get("kind") or "normal"]
    if dep.get("target"):
        parts.append(f"target {dep['target']}")
    if dep.get("optional"):
        parts.append("optional")
    if dep.get("rename"):
        parts.append(f"renamed to `{dep['rename']}`")
    return ", ".join(parts)


def check(metadata: dict[str, Any]) -> list[str]:
    """Return a list of violation messages (empty if the workspace is clean)."""
    packages = {p["id"]: p for p in metadata.get("packages", [])}
    members = metadata.get("workspace_members")
    if not members:
        raise ValueError("metadata has no `workspace_members`")
    missing = [m for m in members if m not in packages]
    if missing:
        raise ValueError(f"workspace members missing from `packages`: {missing}")

    violations: list[str] = []
    member_names = set()

    # Rule 1 and the direct part of rule 2: declared dependencies of each workspace member.
    # `dep["name"]` is the real package name, even when the dependency is renamed.
    for member_id in members:
        pkg = packages[member_id]
        name = pkg["name"]
        member_names.add(name)
        for dep in pkg.get("dependencies", []):
            dep_name = dep["name"]
            if name not in BURN_ALLOWED and BURN_RE.match(dep_name):
                violations.append(
                    f"{name}: depends on `{dep_name}` ({_dep_desc(dep)}); only "
                    f"{' and '.join(sorted(BURN_ALLOWED))} may depend on burn*/cubecl* (ADR-0007)"
                )
            elif name in PURE_CRATES and GPU_RE.match(dep_name):
                violations.append(
                    f"{name}: depends on GPU-related crate `{dep_name}` ({_dep_desc(dep)}); "
                    f"{', '.join(sorted(PURE_CRATES))} must stay pure Rust"
                )

    absent = sorted((BURN_ALLOWED | PURE_CRATES) - member_names)
    if absent:
        # A renamed or removed crate would silently disable the check.
        raise ValueError(f"expected workspace crates not found: {absent}")

    # Transitive part of rule 2, using the resolved dependency graph when present
    # (`cargo metadata --no-deps` has no `resolve`; then only the direct check applies).
    resolve = metadata.get("resolve")
    if resolve:
        nodes = {n["id"]: n for n in resolve.get("nodes", [])}
        for member_id in members:
            name = packages[member_id]["name"]
            if name not in PURE_CRATES:
                continue
            violations.extend(_transitive_gpu_deps(member_id, name, nodes, packages))

    return violations


def _transitive_gpu_deps(
    root: str,
    root_name: str,
    nodes: dict[str, Any],
    packages: dict[str, Any],
) -> list[str]:
    """Find GPU crates reachable from `root` through resolved dependency edges.

    From the root every edge counts (its dev-dependencies are its own choice). Further down only
    normal and build edges count, because dev-dependencies of other crates are not linked into
    the root.
    """
    found: dict[str, list[str]] = {}
    # Breadth-first search, remembering the path for the error message.
    queue: list[tuple[str, list[str]]] = [(root, [root_name])]
    seen = {root}
    while queue:
        node_id, path = queue.pop(0)
        node = nodes.get(node_id)
        if node is None:
            continue
        for edge in node.get("deps", []):
            kinds = {k.get("kind") or "normal" for k in edge.get("dep_kinds", [])} or {"normal"}
            if node_id != root and kinds <= {"dev"}:
                continue
            dep_id = edge["pkg"]
            if dep_id in seen:
                continue
            seen.add(dep_id)
            dep_name = packages.get(dep_id, {}).get("name", edge.get("name", dep_id))
            dep_path = path + [dep_name]
            if GPU_RE.match(dep_name):
                found.setdefault(dep_name, dep_path)
                continue
            queue.append((dep_id, dep_path))
    # Direct GPU deps are already reported by the declared-dependency check.
    return [
        f"{root_name}: transitively depends on GPU-related crate `{dep}` via "
        f"{' -> '.join(path)}; {', '.join(sorted(PURE_CRATES))} must stay pure Rust"
        for dep, path in sorted(found.items())
        if len(path) > 2
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--metadata",
        metavar="FILE",
        help="read `cargo metadata --format-version 1` JSON from FILE instead of running cargo",
    )
    args = parser.parse_args(argv)

    try:
        metadata = load_metadata(args.metadata)
        violations = check(metadata)
    except subprocess.CalledProcessError as e:
        print(f"error: `cargo metadata` failed:\n{e.stderr}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"error: cannot check dependency boundary: {e!r}", file=sys.stderr)
        return 2

    if violations:
        print("Dependency-boundary check FAILED:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "See docs/adr/0007-burn-version-pinning.md and CLAUDE.md (Hard rules).",
            file=sys.stderr,
        )
        return 1

    n = len(metadata["workspace_members"])
    print(f"Dependency-boundary check passed ({n} workspace crates).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
