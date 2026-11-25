#!/usr/bin/env python3
"""Tests for convert_mnist.py on small synthetic IDX files: the IDX parser's checks (magic number,
dims, sizes, gzip), the pinned hashes and mirror fallback (standard library only), and the
conversion's output bytes (needs NumPy and safetensors; skipped without them).

Run with: python -m unittest discover -s tools/reference -v
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import convert_mnist as cm  # noqa: E402  (imports without NumPy; converting needs it)

HAVE_NUMPY = cm.IMPORT_ERROR is None


def idx(magic: int, dims: tuple[int, ...], body: bytes) -> bytes:
    """A gzip-compressed IDX file. `mtime=0` keeps the bytes fixed."""
    raw = struct.pack(">I", magic) + struct.pack(f">{len(dims)}I", *dims) + body
    return gzip.compress(raw, mtime=0)


def raw_file(name: str, data: bytes, magic: int, dims: tuple[int, ...]) -> cm.RawFile:
    return cm.RawFile(name, hashlib.sha256(data).hexdigest(), len(data), magic, dims)


# Three 2x3 "images" and their labels.
PIXELS = bytes([0, 1, 2, 127, 128, 255, 10, 20, 30, 40, 50, 60, 255, 254, 253, 0, 0, 0])
LABELS = bytes([7, 0, 9])
IMAGES_GZ = idx(0x00000803, (3, 2, 3), PIXELS)
LABELS_GZ = idx(0x00000801, (3,), LABELS)


def synthetic(images_gz: bytes = IMAGES_GZ, labels_gz: bytes = LABELS_GZ) -> dict[str, cm.RawFile]:
    files = {}
    for split in ("train", "test"):
        files[f"{split}_images"] = raw_file(f"{split}-images.gz", images_gz, 0x00000803, (3, 2, 3))
        files[f"{split}_labels"] = raw_file(f"{split}-labels.gz", labels_gz, 0x00000801, (3,))
    return files


class ParseTest(unittest.TestCase):
    """The IDX parser checks everything before it allocates (standard library only)."""

    def images(self, gz: bytes) -> cm.RawFile:
        return raw_file("images.gz", gz, 0x00000803, (3, 2, 3))

    def test_valid_file(self) -> None:
        self.assertEqual(cm.parse_idx(self.images(IMAGES_GZ), IMAGES_GZ), PIXELS)
        labels = raw_file("labels.gz", LABELS_GZ, 0x00000801, (3,))
        self.assertEqual(cm.parse_idx(labels, LABELS_GZ), LABELS)

    def assert_error(self, gz: bytes, text: str) -> None:
        with self.assertRaises(cm.ConvertError) as ctx:
            cm.parse_idx(self.images(gz), gz)
        self.assertIn(text, str(ctx.exception))

    def test_wrong_magic(self) -> None:
        self.assert_error(idx(0x00000801, (3, 2, 3), PIXELS), "magic number 0x00000801, expected 0x00000803")
        self.assert_error(idx(0x00000c03, (3, 2, 3), PIXELS), "magic number")  # another element type

    def test_wrong_dims(self) -> None:
        self.assert_error(idx(0x00000803, (3, 3, 2), PIXELS), "dims [3, 3, 2], expected [3, 2, 3]")
        # A huge count is rejected from the header, before anything that size is read.
        self.assert_error(idx(0x00000803, (0xFFFFFFFF, 2, 3), PIXELS), "dims [4294967295, 2, 3]")

    def test_short_and_long_data(self) -> None:
        self.assert_error(idx(0x00000803, (3, 2, 3), PIXELS[:-1]), "only 17 data bytes, expected 18")
        self.assert_error(idx(0x00000803, (3, 2, 3), PIXELS + b"\0" * 1000), "more data bytes, expected 18")

    def test_truncated_header(self) -> None:
        self.assert_error(gzip.compress(b"\0\0", mtime=0), "shorter than the IDX magic number")
        self.assert_error(gzip.compress(struct.pack(">II", 0x00000803, 3), mtime=0), "the header ends early")

    def test_not_gzip(self) -> None:
        self.assert_error(b"not a gzip stream at all", "not a valid gzip stream")
        self.assert_error(IMAGES_GZ[: len(IMAGES_GZ) // 2], "not a valid gzip stream")

    def test_labels_must_be_classes(self) -> None:
        labels = raw_file("labels.gz", LABELS_GZ, 0x00000801, (3,))
        cm.check_labels(labels, LABELS)
        with self.assertRaisesRegex(cm.ConvertError, "label 10 of sample 1 is not a class 0-9"):
            cm.check_labels(labels, bytes([1, 10, 2]))

    def test_pinned_mnist_files(self) -> None:
        self.assertEqual(sorted(cm.RAW), ["test_images", "test_labels", "train_images", "train_labels"])
        for raw in cm.RAW.values():
            self.assertRegex(raw.sha256, r"^[0-9a-f]{64}$")
            self.assertGreater(raw.size, 0)
        self.assertEqual(cm.RAW["train_images"].dims, (60000, 28, 28))
        self.assertEqual(cm.RAW["test_labels"].dims, (10000,))


class DownloadTest(unittest.TestCase):
    """Pinned hashes, bounded reads and the mirror fallback, with a fake fetcher."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.raw = raw_file("labels.gz", LABELS_GZ, 0x00000801, (3,))
        self.calls: list[str] = []

    def fetcher(self, serve: dict[str, bytes]):
        def fetch(url: str, limit: int) -> bytes:
            self.calls.append(url)
            if url not in serve:
                raise OSError("404")
            data = serve[url]
            if len(data) > limit:
                raise cm.ConvertError("more than the expected bytes")
            return data

        return fetch

    def test_wrong_hash_is_refused_then_the_next_mirror_is_used(self) -> None:
        tampered = bytearray(LABELS_GZ)
        tampered[-5] ^= 1
        fetch = self.fetcher({"https://a/labels.gz": bytes(tampered), "https://b/labels.gz": LABELS_GZ})
        data = cm.obtain_raw(self.raw, self.tmp, fetch, ("https://a/", "https://b/"))
        self.assertEqual(data, LABELS_GZ)
        self.assertEqual(self.calls, ["https://a/labels.gz", "https://b/labels.gz"])
        # Kept in the raw directory and used from there next time.
        self.calls.clear()
        self.assertEqual(cm.obtain_raw(self.raw, self.tmp, fetch, ("https://a/",)), LABELS_GZ)
        self.assertEqual(self.calls, [])

    def test_no_mirror_with_the_pinned_file_is_an_error(self) -> None:
        fetch = self.fetcher({"https://a/labels.gz": LABELS_GZ + b"x"})
        with self.assertRaises(cm.ConvertError) as ctx:
            cm.obtain_raw(self.raw, self.tmp, fetch, ("https://a/", "https://b/"))
        self.assertIn("no mirror served the pinned file", str(ctx.exception))
        self.assertIn("more than the expected bytes", str(ctx.exception))
        self.assertIn("404", str(ctx.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "labels.gz")))

    def test_a_damaged_local_copy_is_downloaded_again(self) -> None:
        with open(os.path.join(self.tmp, "labels.gz"), "wb") as f:
            f.write(b"\0" * len(LABELS_GZ))
        fetch = self.fetcher({"https://a/labels.gz": LABELS_GZ})
        self.assertEqual(cm.obtain_raw(self.raw, self.tmp, fetch, ("https://a/",)), LABELS_GZ)
        self.assertEqual(self.calls, ["https://a/labels.gz"])

    def test_check_raw(self) -> None:
        cm.check_raw(self.raw, LABELS_GZ)
        with self.assertRaisesRegex(cm.ConvertError, "bytes, expected"):
            cm.check_raw(self.raw, LABELS_GZ[:-1])
        wrong = cm.RawFile("labels.gz", "0" * 64, len(LABELS_GZ), 0x00000801, (3,))
        with self.assertRaisesRegex(cm.ConvertError, "does not match the pinned"):
            cm.check_raw(wrong, LABELS_GZ)


@unittest.skipUnless(HAVE_NUMPY, "needs NumPy and safetensors (tools/reference/requirements.txt)")
class ConvertTest(unittest.TestCase):
    """The output bytes of the conversion on the synthetic files."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.raw_dir = os.path.join(self.tmp, "raw")
        os.makedirs(self.raw_dir)
        files = synthetic()
        for key, data in [("train_images", IMAGES_GZ), ("train_labels", LABELS_GZ),
                          ("test_images", IMAGES_GZ), ("test_labels", LABELS_GZ)]:
            raw = files[key]
            with open(os.path.join(self.raw_dir, raw.name), "wb") as f:
                f.write(data)
        self.files = files

    def no_fetch(self, url: str, limit: int) -> bytes:
        raise AssertionError(f"no download expected: {url}")

    def convert(self, out: str) -> list[dict]:
        return cm.convert(out, self.raw_dir, self.no_fetch, ("https://unused/",), self.files)

    def test_layout_values_and_determinism(self) -> None:
        out = os.path.join(self.tmp, "out")
        entries = self.convert(out)
        self.assertEqual([e["name"] for e in entries], ["mnist-train", "mnist-test"])
        path = os.path.join(out, "mnist", "mnist-train.safetensors")
        with open(path, "rb") as f:
            data = f.read()
        self.assertEqual(entries[0]["size"], len(data))
        self.assertEqual(entries[0]["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(entries[0]["path"], "mnist/mnist-train.safetensors")
        self.assertTrue(entries[0]["url"].startswith("https://github.com/Jannik0/Polypous/releases/download/"))
        self.assertIn("CC BY-SA 3.0", entries[0]["license"])

        # The header: no __metadata__, label (I64) first, then image (F32).
        (n,) = struct.unpack("<Q", data[:8])
        header = json.loads(data[8 : 8 + n])
        self.assertEqual(
            header,
            {
                "label": {"dtype": "I64", "shape": [3], "data_offsets": [0, 24]},
                "image": {"dtype": "F32", "shape": [3, 1, 2, 3], "data_offsets": [24, 24 + 18 * 4]},
            },
        )
        body = data[8 + n :]
        self.assertEqual(list(struct.unpack("<3q", body[:24])), [7, 0, 9])
        pixels = struct.unpack("<18f", body[24:])
        # float32(byte) / float32(255), correctly rounded: compare with the exact quotient rounded once.
        for got, byte in zip(pixels, PIXELS):
            want = struct.unpack("<f", struct.pack("<f", byte / 255))[0]
            self.assertEqual(got, want, byte)
        self.assertEqual((pixels[0], pixels[5]), (0.0, 1.0))

        # A second conversion writes the same bytes.
        again = self.convert(os.path.join(self.tmp, "again"))
        self.assertEqual(again, entries)

    def test_check_against_a_manifest(self) -> None:
        entries = self.convert(os.path.join(self.tmp, "out"))
        manifest = os.path.join(self.tmp, "manifest.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "entries": entries}, f)
        self.assertEqual(cm.manifest_problems(manifest, entries), [])
        changed = [dict(entries[0], sha256="0" * 64), entries[1]]
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "entries": changed}, f)
        problems = cm.manifest_problems(manifest, entries)
        self.assertEqual(len(problems), 1)
        self.assertIn("mnist-train: manifest sha256", problems[0])
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "entries": []}, f)
        self.assertEqual(len(cm.manifest_problems(manifest, entries)), 2)

    def test_bad_labels_stop_the_conversion(self) -> None:
        bad = idx(0x00000801, (3,), bytes([1, 2, 200]))
        self.files = synthetic(labels_gz=bad)
        for split in ("train", "test"):
            with open(os.path.join(self.raw_dir, f"{split}-labels.gz"), "wb") as f:
                f.write(bad)
        out = os.path.join(self.tmp, "out")
        with self.assertRaisesRegex(cm.ConvertError, "label 200 of sample 2"):
            self.convert(out)
        self.assertFalse(os.path.exists(os.path.join(out, "mnist", "mnist-train.safetensors")))

    def test_main_check_reports_a_manifest_mismatch(self) -> None:
        manifest = os.path.join(self.tmp, "manifest.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "entries": []}, f)
        # main() converts the real MNIST files; only run it when they are already downloaded.
        raw_dir = os.path.join(cm.DEFAULT_CACHE, "mnist", "raw")
        if not all(os.path.isfile(os.path.join(raw_dir, r.name)) for r in cm.RAW.values()):
            self.skipTest("the MNIST .gz files are not in fixtures/cache/mnist/raw")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cm.main(["--check", "--manifest", manifest, "--raw-dir", raw_dir])
        self.assertEqual(code, 1, err.getvalue())
        self.assertIn("mnist-train: not in", err.getvalue())


if __name__ == "__main__":
    unittest.main()
