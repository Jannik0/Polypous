#!/usr/bin/env python3
"""Tests for fetch_fixtures.py with local `file://` URLs.

Run with: python -m unittest discover -s tools/reference -v   (standard library only)
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fetch_fixtures as ff  # noqa: E402

PAYLOAD = b"polypous fixture payload\n" * 100


class FetchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.cache = os.path.join(self.tmp, "cache")
        self.source = os.path.join(self.tmp, "source.bin")
        with open(self.source, "wb") as f:
            f.write(PAYLOAD)
        self.manifest = os.path.join(self.tmp, "manifest.json")

    def entry(self, **overrides) -> dict:
        e = {
            "name": "tiny-model",
            "url": pathlib.Path(self.source).as_uri(),
            "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
            "size": len(PAYLOAD),
            "license": "MIT",
            "path": "models/tiny.bin",
        }
        e.update(overrides)
        return e

    def run_main(self, entries: list, *args: str) -> tuple[int, str, str]:
        with open(self.manifest, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "entries": entries}, f)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ff.main(["--manifest", self.manifest, "--cache-dir", self.cache, *args])
        return code, out.getvalue(), err.getvalue()

    def target(self) -> str:
        return os.path.join(self.cache, "models", "tiny.bin")

    def leftovers(self) -> list[str]:
        found = []
        for dirpath, _, files in os.walk(self.cache):
            found += [os.path.join(dirpath, f) for f in files]
        return found

    def test_good_hash_downloads_then_uses_cache(self) -> None:
        code, out, _ = self.run_main([self.entry()])
        self.assertEqual(code, 0)
        self.assertIn("tiny-model: downloaded", out)
        with open(self.target(), "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)
        code, out, _ = self.run_main([self.entry()])
        self.assertEqual((code, out.strip()), (0, "tiny-model: cached"))

    def test_bad_hash_is_refused_and_nothing_left(self) -> None:
        code, _, err = self.run_main([self.entry(sha256="0" * 64)])
        self.assertEqual(code, 1)
        self.assertIn("does not match the manifest", err)
        self.assertEqual(self.leftovers(), [])

    def test_corrupt_cached_file_is_replaced(self) -> None:
        os.makedirs(os.path.dirname(self.target()))
        with open(self.target(), "wb") as f:
            f.write(b"x" * len(PAYLOAD))
        code, out, _ = self.run_main([self.entry()])
        self.assertEqual(code, 0)
        self.assertIn("downloaded", out)

    def test_oversize_source_is_refused(self) -> None:
        # The manifest says fewer bytes than the source has: stop at the limit, keep nothing.
        code, _, err = self.run_main([self.entry(size=len(PAYLOAD) - 1)])
        self.assertEqual(code, 1)
        self.assertIn("announces", err)
        self.assertEqual(self.leftovers(), [])

    def test_oversize_stream_without_length_is_refused(self) -> None:
        dst = io.BytesIO()
        with self.assertRaises(ff.FetchError):
            ff.copy_bounded(io.BytesIO(PAYLOAD), dst, len(PAYLOAD) - 1)
        self.assertLessEqual(len(dst.getvalue()), len(PAYLOAD) - 1)

    def test_short_source_is_refused(self) -> None:
        code, _, err = self.run_main([self.entry(size=len(PAYLOAD) + 1)])
        self.assertEqual(code, 1)
        self.assertIn(f"got {len(PAYLOAD)} bytes", err)
        self.assertEqual(self.leftovers(), [])

    def test_entry_over_max_bytes_is_refused_before_download(self) -> None:
        code, _, err = self.run_main([self.entry()], "--max-bytes", "10")
        self.assertEqual(code, 1)
        self.assertIn("more than the limit", err)
        self.assertFalse(os.path.exists(self.cache))

    def test_missing_entry(self) -> None:
        code, _, err = self.run_main([self.entry()], "no-such-entry")
        self.assertEqual(code, 1)
        self.assertIn("not in the manifest: no-such-entry", err)

    def test_missing_source_file(self) -> None:
        os.remove(self.source)
        code, _, err = self.run_main([self.entry()])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertEqual(self.leftovers(), [])

    def test_empty_manifest(self) -> None:
        self.assertEqual(self.run_main([])[0], 0)

    def test_committed_manifest_is_valid(self) -> None:
        self.assertEqual(ff.load_manifest(ff.DEFAULT_MANIFEST), [])

    def test_rejects_unsafe_entries(self) -> None:
        bad = [
            self.entry(path="../escape.bin"),
            self.entry(path="/abs.bin"),
            self.entry(path="a\\b.bin"),
            self.entry(path="C:/x.bin"),
            self.entry(url="http://example.com/x.bin"),
            self.entry(sha256="ABC"),
            self.entry(size=-1),
            self.entry(size=True),
            self.entry(license=""),
            self.entry(name="../x"),
            {k: v for k, v in self.entry().items() if k != "license"},
        ]
        for e in bad:
            with self.subTest(entry=e):
                code, _, err = self.run_main([e])
                self.assertEqual(code, 1)
                self.assertIn("error:", err)
        self.assertFalse(os.path.exists(self.cache))

    def test_rejects_duplicates(self) -> None:
        self.assertEqual(self.run_main([self.entry(), self.entry()])[0], 1)
        self.assertEqual(self.run_main([self.entry(), self.entry(name="other")])[0], 1)


if __name__ == "__main__":
    unittest.main()
