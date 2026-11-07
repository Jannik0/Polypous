#!/usr/bin/env python3
"""Downloads large fixtures and test models listed in fixtures/manifest.json into fixtures/cache/.

    python tools/reference/fetch_fixtures.py                # every entry
    python tools/reference/fetch_fixtures.py NAME [NAME…]   # selected entries
    python tools/reference/fetch_fixtures.py --list         # show the entries

Each entry is downloaded to a temporary file next to its target, never past the size the manifest
gives (and never past --max-bytes), checked against its SHA-256, and only then renamed into place.
A file that is already in the cache with the right hash is not downloaded again. A wrong size or
hash is an error, and nothing is left behind.

Manifest schema: docs/testing-strategy.md §3. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from typing import Any, BinaryIO

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_MANIFEST = os.path.join(REPO, "fixtures", "manifest.json")
DEFAULT_CACHE = os.path.join(REPO, "fixtures", "cache")
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MAX_BYTES = 8 * 1024**3  # largest entry we accept at all
MAX_MANIFEST_BYTES = 1024 * 1024
CHUNK = 1024 * 1024
TIMEOUT_SECONDS = 60
# `file` is for tests and local mirrors; plain `http` is never accepted.
ALLOWED_SCHEMES = ("https", "file")
ENTRY_KEYS = {"name", "url", "sha256", "size", "license", "path"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FetchError(Exception):
    """The manifest is malformed, or a download failed or didn't match."""


def safe_relative_path(path: Any) -> str:
    """A '/'-separated path that stays inside the cache directory."""
    if not isinstance(path, str) or not path or "\\" in path or path.startswith("/"):
        raise FetchError(f"path {path!r} must be a relative '/'-separated path")
    parts = path.split("/")
    if any(p in ("", ".", "..") or ":" in p or not NAME_RE.match(p) for p in parts):
        raise FetchError(f"path {path!r} has an unsafe component")
    return path


def load_manifest(path: str) -> list[dict[str, Any]]:
    """Reads and validates the manifest. Every field is checked before anything is downloaded."""
    if os.path.getsize(path) > MAX_MANIFEST_BYTES:
        raise FetchError(f"{path}: larger than {MAX_MANIFEST_BYTES} bytes")
    with open(path, encoding="utf-8") as f:
        try:
            manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise FetchError(f"{path}: not valid JSON: {e}") from e
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise FetchError(f"{path}: expected an object with schema_version {MANIFEST_SCHEMA_VERSION}")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise FetchError(f"{path}: 'entries' must be a list")
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for e in entries:
        if not isinstance(e, dict) or set(e) != ENTRY_KEYS:
            raise FetchError(f"{path}: entry {e!r} must have exactly the keys {sorted(ENTRY_KEYS)}")
        if not isinstance(e["name"], str) or not NAME_RE.match(e["name"]):
            raise FetchError(f"{path}: bad entry name {e['name']!r}")
        if e["name"] in seen_names:
            raise FetchError(f"{path}: duplicate entry name {e['name']!r}")
        seen_names.add(e["name"])
        if not isinstance(e["url"], str) or urllib.parse.urlparse(e["url"]).scheme not in ALLOWED_SCHEMES:
            raise FetchError(f"{e['name']}: url must use one of {ALLOWED_SCHEMES}")
        if not isinstance(e["sha256"], str) or not SHA256_RE.match(e["sha256"]):
            raise FetchError(f"{e['name']}: sha256 must be 64 lowercase hex digits")
        if not isinstance(e["size"], int) or isinstance(e["size"], bool) or e["size"] < 0:
            raise FetchError(f"{e['name']}: size must be a non-negative integer")
        if not isinstance(e["license"], str) or not e["license"]:
            raise FetchError(f"{e['name']}: license must be a non-empty string")
        rel = safe_relative_path(e["path"])
        if rel in seen_paths:
            raise FetchError(f"{e['name']}: path {rel!r} is used twice")
        seen_paths.add(rel)
    return entries


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_bounded(src: BinaryIO, dst: BinaryIO, limit: int) -> tuple[int, str]:
    """Copies at most `limit` bytes; fails as soon as the source has more. Returns size and hash."""
    h = hashlib.sha256()
    total = 0
    while True:
        chunk = src.read(min(CHUNK, limit - total + 1))
        if not chunk:
            return total, h.hexdigest()
        total += len(chunk)
        if total > limit:
            raise FetchError(f"more than the expected {limit} bytes")
        h.update(chunk)
        dst.write(chunk)


def fetch_entry(entry: dict[str, Any], cache_dir: str, max_bytes: int) -> str:
    """Makes sure the entry is in the cache with the right hash. Returns 'cached' or 'downloaded'."""
    name, size, want = entry["name"], entry["size"], entry["sha256"]
    if size > max_bytes:
        raise FetchError(f"{name}: {size} bytes is more than the limit of {max_bytes}")
    target = os.path.join(cache_dir, *entry["path"].split("/"))
    if os.path.isfile(target) and os.path.getsize(target) == size and sha256_of(target) == want:
        return "cached"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".partial-", dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "wb") as out:
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "polypous-fetch-fixtures"})
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                # urlopen follows redirects; refuse one that leaves the allowed schemes.
                final = urllib.parse.urlparse(response.geturl()).scheme
                if final not in ALLOWED_SCHEMES:
                    raise FetchError(f"{name}: redirected to a {final!r} URL")
                length = response.headers.get("Content-Length")
                if length is not None and length.isdigit() and int(length) > size:
                    raise FetchError(f"{name}: server announces {length} bytes, manifest says {size}")
                try:
                    got, digest = copy_bounded(response, out, size)
                except FetchError as e:
                    raise FetchError(f"{name}: {e}") from e
        if got != size:
            raise FetchError(f"{name}: got {got} bytes, manifest says {size}")
        if digest != want:
            raise FetchError(f"{name}: SHA-256 {digest} does not match the manifest's {want}")
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="entries to fetch (default: all)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="refuse larger entries")
    parser.add_argument("--list", action="store_true", help="list the manifest's entries and exit")
    args = parser.parse_args(argv)
    try:
        entries = load_manifest(args.manifest)
        by_name = {e["name"]: e for e in entries}
        if args.list:
            for e in entries:
                print(f"{e['name']}\t{e['size']}\t{e['license']}\t{e['path']}")
            return 0
        unknown = [n for n in args.names if n not in by_name]
        if unknown:
            raise FetchError(f"not in the manifest: {', '.join(unknown)}")
        selected = [by_name[n] for n in args.names] if args.names else entries
        if not selected:
            print("the manifest has no entries")
            return 0
        failed = 0
        for e in selected:
            try:
                print(f"{e['name']}: {fetch_entry(e, args.cache_dir, args.max_bytes)}")
            except (FetchError, OSError) as err:
                failed += 1
                print(f"error: {err}", file=sys.stderr)
        return 1 if failed else 0
    except (FetchError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
