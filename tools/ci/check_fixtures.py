#!/usr/bin/env python3
"""Checks the committed reference fixtures in fixtures/small/ (docs/testing-strategy.md §3).

- Size budget: each fixture (a case's `.safetensors` and `.json` together) under 1 MiB, everything
  under fixtures/small/ under 50 MiB.
- Every sidecar under fixtures/small/ops/ follows the schema and matches its safetensors header
  (names, dtypes, shapes, data offsets), and every `.safetensors` file has a sidecar.

Standard library only (no PyTorch), so it runs in CI without setup. Exit status 0 if clean, 1 otherwise.
The helpers live in tools/reference/fixturelib.py.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools", "reference"))

import fixturelib  # noqa: E402


def check(root: str) -> list[str]:
    problems = fixturelib.check_sizes(root)
    files = fixturelib.list_files(root)
    present = set(files)
    for rel in files:
        stem, ext = os.path.splitext(rel)
        if ext == ".json" and rel.startswith("ops/"):
            problems += fixturelib.validate_case(os.path.join(root, rel))
        elif ext == ".safetensors" and stem + ".json" not in present:
            problems.append(f"{rel}: no JSON sidecar")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(REPO, "fixtures", "small"))
    args = parser.parse_args(argv)
    problems = check(args.root)
    for line in problems:
        print(f"::error::{line}" if os.environ.get("GITHUB_ACTIONS") else f"error: {line}")
    if problems:
        return 1
    sizes = fixturelib.fixture_sizes(args.root)
    print(f"ok: {len(sizes)} fixtures, {sum(sizes.values())} bytes, largest {max(sizes.values(), default=0)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
