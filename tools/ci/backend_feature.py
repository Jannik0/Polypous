#!/usr/bin/env python3
"""Tell a CI job whether a workspace package has a Cargo backend feature, and how to enable it.

Usage: backend_feature.py FEATURE [--package PACKAGE] [--metadata FILE]

PACKAGE defaults to `polypous-backend`. Prints two `key=value` lines meant for `$GITHUB_OUTPUT`:

    available=true|false
    args=--features PACKAGE/FEATURE    (empty when the feature does not exist)

The GPU test jobs pass `--package polypous-cli`: the CLI's backend features enable the facade's,
which enable the runtime's and the backend's, so one feature reaches every crate's GPU tests in a
workspace build (issue #65).

When the feature is missing it also prints a GitHub Actions `::warning::` annotation to stderr, so
a job that falls back to the default features is visible in the run summary instead of passing
silently. This lets the jobs for the `cpu`, `vulkan` and `metal` features (issue #2) exist before
the features do; once the features land, the jobs pick them up without a workflow change.

Exit status: 0 on success (whether or not the feature exists), 2 on an input error.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

DEFAULT_PACKAGE = "polypous-backend"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("feature")
    parser.add_argument(
        "--package",
        default=DEFAULT_PACKAGE,
        help=f"workspace package that should have the feature (default: {DEFAULT_PACKAGE})",
    )
    parser.add_argument("--metadata", metavar="FILE", help="read cargo metadata JSON from FILE")
    args = parser.parse_args(argv)

    try:
        if args.metadata:
            with open(args.metadata, encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            out = subprocess.run(
                ["cargo", "metadata", "--format-version", "1", "--no-deps"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            metadata = json.loads(out)
        pkg = next(p for p in metadata["packages"] if p["name"] == args.package)
    except subprocess.CalledProcessError as e:
        print(f"error: `cargo metadata` failed:\n{e.stderr}", file=sys.stderr)
        return 2
    except StopIteration:
        print(f"error: package {args.package} not found in cargo metadata", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"error: cannot read cargo metadata: {e!r}", file=sys.stderr)
        return 2

    if args.feature in pkg.get("features", {}):
        print("available=true")
        print(f"args=--features {args.package}/{args.feature}")
    else:
        print("available=false")
        print("args=")
        print(
            f"::warning title=Feature `{args.feature}` missing::{args.package} has no "
            f"`{args.feature}` feature; this job uses the default features instead.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
