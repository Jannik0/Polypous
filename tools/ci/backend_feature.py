#!/usr/bin/env python3
"""Tell a CI job whether `polypous-backend` has a Cargo feature, and how to enable it.

Usage: backend_feature.py FEATURE [--metadata FILE]

Prints two `key=value` lines meant for `$GITHUB_OUTPUT`:

    available=true|false
    args=--features polypous-backend/FEATURE    (empty when the feature does not exist)

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

PACKAGE = "polypous-backend"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("feature")
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
        pkg = next(p for p in metadata["packages"] if p["name"] == PACKAGE)
    except subprocess.CalledProcessError as e:
        print(f"error: `cargo metadata` failed:\n{e.stderr}", file=sys.stderr)
        return 2
    except StopIteration:
        print(f"error: package {PACKAGE} not found in cargo metadata", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"error: cannot read cargo metadata: {e!r}", file=sys.stderr)
        return 2

    if args.feature in pkg.get("features", {}):
        print("available=true")
        print(f"args=--features {PACKAGE}/{args.feature}")
    else:
        print("available=false")
        print("args=")
        print(
            f"::warning title=Feature `{args.feature}` missing::{PACKAGE} has no `{args.feature}` "
            "feature yet (added by issue #2); this job uses the default features instead.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
