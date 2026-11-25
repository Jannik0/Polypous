#!/usr/bin/env python3
"""Converts MNIST once into the two safetensors datasets of the MNIST training test (issue #84).

    python tools/reference/convert_mnist.py            # download, convert into fixtures/cache/, print hashes
    python tools/reference/convert_mnist.py --check    # convert into a temp dir, compare with the manifest
    python tools/reference/convert_mnist.py --raw-dir DIR   # use the four .gz files in DIR, no download

The outputs are the `mnist-train` and `mnist-test` entries of `fixtures/manifest.json`
(docs/testing-strategy.md §3.4, §3.8), hosted as assets of the Polypous GitHub release
`fixtures-mnist-v1`. Nothing is committed: `fetch_fixtures.py` downloads them into the same paths
this script writes. Tests read them with the `safetensors` dataset kind (training.md §3.5.2).

**Data and license.** The MNIST database of handwritten digits, by Yann LeCun and Corinna Cortes
(with Christopher J. C. Burges), licensed under the Creative Commons Attribution-Share Alike 3.0
license (CC BY-SA 3.0). The converted files are a derivative of it under the same license. The
original site no longer serves the files, so they are downloaded from a public mirror (below); each
download is checked against a pinned SHA-256 before it is parsed, so the mirror doesn't matter.

**Output** (one file per split, written exactly as `fetch_fixtures.py` places the manifest entries):

| File | Tensors |
|---|---|
| `fixtures/cache/mnist/mnist-train.safetensors` | `image` `F32` `[60000, 1, 28, 28]`, `label` `I64` `[60000]` |
| `fixtures/cache/mnist/mnist-test.safetensors`  | `image` `F32` `[10000, 1, 28, 28]`, `label` `I64` `[10000]` |

- Samples stay in the IDX files' order. `image[n, 0, h, w]` is pixel `(h, w)` of sample `n`.
- **Pixel values:** `float32(byte) / float32(255)`, one IEEE `float32` division in NumPy, which is
  correctly rounded, so the bytes are the same on every platform. This is the `image_folder` formula
  `v / 255` of training.md §3.5.4 and `torchvision.transforms.ToTensor`. Values are in `[0, 1]`.
- **Labels** are the class indices `0`–`9` as `int64` (training.md §3.3: `cross_entropy` and
  multiclass `accuracy` take `i64` or `i32` class indices; `i64` is what PyTorch's `cross_entropy`
  takes, so the reference generator uses the tensor as it is).
- **Deterministic bytes:** written with `safetensors.numpy` (pinned in requirements.txt) without
  `__metadata__`, so the file holds nothing machine- or time-dependent; the tensor order in the
  header and the data is the library's fixed order (by dtype alignment, then name: `label`, then
  `image`), checked after writing.

**Untrusted input.** Each `.gz` download is read up to its pinned size only and checked against
its SHA-256 first. The IDX parser then checks the magic number, the number of dims and every dim
against the expected values before it allocates anything, decompresses at most the expected number
of bytes (plus one, to detect trailing data), and checks every label is in `0`–`9`.

Needs NumPy and safetensors in the pinned versions (gen_ops.py's version check); no PyTorch.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import urllib.request
import zlib
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_fixtures  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CACHE = os.path.join(REPO, "fixtures", "cache")
DEFAULT_MANIFEST = os.path.join(REPO, "fixtures", "manifest.json")
GENERATOR = "tools/reference/convert_mnist.py"

try:
    import numpy as np
    import safetensors.numpy
except ImportError as e:  # pragma: no cover - depends on the environment
    np = safetensors = None  # type: ignore[assignment]
    IMPORT_ERROR: ImportError | None = e
else:
    IMPORT_ERROR = None

# Public mirrors of the original IDX files, tried in order. The SHA-256 below pins the content.
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
TIMEOUT_SECONDS = 60
CHUNK = 1024 * 1024

LICENSE = (
    "CC BY-SA 3.0: the MNIST database by Yann LeCun and Corinna Cortes, "
    "converted to safetensors by tools/reference/convert_mnist.py"
)
RELEASE_URL = "https://github.com/Jannik0/Polypous/releases/download/fixtures-mnist-v1/"


class RawFile:
    """One of the four original files: name, pinned SHA-256 and size of the `.gz`."""

    def __init__(self, name: str, sha256: str, size: int, magic: int, dims: tuple[int, ...]):
        self.name, self.sha256, self.size, self.magic, self.dims = name, sha256, size, magic, dims


# The MD5s of these files are the well-known ones (torchvision's MNIST class lists the same).
RAW = {
    "train_images": RawFile(
        "train-images-idx3-ubyte.gz",
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        9912422,
        0x00000803,
        (60000, 28, 28),
    ),
    "train_labels": RawFile(
        "train-labels-idx1-ubyte.gz",
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        28881,
        0x00000801,
        (60000,),
    ),
    "test_images": RawFile(
        "t10k-images-idx3-ubyte.gz",
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        1648877,
        0x00000803,
        (10000, 28, 28),
    ),
    "test_labels": RawFile(
        "t10k-labels-idx1-ubyte.gz",
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        4542,
        0x00000801,
        (10000,),
    ),
}

# The outputs: manifest name, cache path, and the raw files they come from.
OUTPUTS = (
    ("mnist-train", "mnist/mnist-train.safetensors", "train_images", "train_labels"),
    ("mnist-test", "mnist/mnist-test.safetensors", "test_images", "test_labels"),
)
NUM_CLASSES = 10


class ConvertError(Exception):
    """A download, an IDX file or an output didn't match what is expected."""


# ---------------------------------------------------------------------------------------------
# Downloads


def check_raw(raw: RawFile, data: bytes) -> None:
    if len(data) != raw.size:
        raise ConvertError(f"{raw.name}: {len(data)} bytes, expected {raw.size}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != raw.sha256:
        raise ConvertError(f"{raw.name}: SHA-256 {digest} does not match the pinned {raw.sha256}")


def download(url: str, limit: int) -> bytes:
    """At most `limit` bytes of `url`; more is an error."""
    request = urllib.request.Request(url, headers={"User-Agent": "polypous-convert-mnist"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        if not response.geturl().startswith("https://"):
            raise ConvertError(f"{url}: redirected to a non-https URL")
        out = io.BytesIO()
        try:
            fetch_fixtures.copy_bounded(response, out, limit)
        except fetch_fixtures.FetchError as e:
            raise ConvertError(f"{url}: {e}") from e
        return out.getvalue()


Fetcher = Callable[[str, int], bytes]


def obtain_raw(raw: RawFile, raw_dir: str, fetch: Fetcher = download, mirrors: tuple[str, ...] = MIRRORS) -> bytes:
    """The checked bytes of a raw file: from `raw_dir` if it has the right hash there, else from
    the first mirror that serves the right bytes (then kept in `raw_dir`)."""
    path = os.path.join(raw_dir, raw.name)
    if os.path.isfile(path) and os.path.getsize(path) == raw.size:
        with open(path, "rb") as f:
            data = f.read(raw.size + 1)
        try:
            check_raw(raw, data)
            return data
        except ConvertError:
            pass  # a stale or damaged copy: download again
    errors = []
    for mirror in mirrors:
        try:
            data = fetch(mirror + raw.name, raw.size)
            check_raw(raw, data)
        except (ConvertError, OSError) as e:
            errors.append(f"{mirror}: {e}")
            continue
        os.makedirs(raw_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".partial-", dir=raw_dir)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return data
    raise ConvertError(f"{raw.name}: no mirror served the pinned file: " + "; ".join(errors))


# ---------------------------------------------------------------------------------------------
# IDX parsing


def parse_idx(raw: RawFile, gz: bytes) -> bytes:
    """The data bytes of a gzip-compressed IDX file (`uint8`, row-major, `raw.dims`), after
    checking its magic number and dims. Standard library only.

    The header is checked before the data is decompressed; at most the expected size plus one byte
    is decompressed, so a file with extra data is an error, not an allocation."""
    header_len = 4 + 4 * len(raw.dims)
    data_len = 1
    for d in raw.dims:
        data_len *= d
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(gz)) as f:
            head = f.read(header_len)
            if len(head) < 4:
                raise ConvertError(f"{raw.name}: shorter than the IDX magic number")
            (magic,) = struct.unpack(">I", head[:4])
            if magic != raw.magic:
                raise ConvertError(f"{raw.name}: magic number 0x{magic:08x}, expected 0x{raw.magic:08x}")
            if len(head) < header_len:
                raise ConvertError(f"{raw.name}: the header ends early")
            dims = struct.unpack(f">{len(raw.dims)}I", head[4:])
            if dims != raw.dims:
                raise ConvertError(f"{raw.name}: dims {list(dims)}, expected {list(raw.dims)}")
            body = f.read(data_len + 1)
    except (OSError, EOFError, zlib.error) as e:
        raise ConvertError(f"{raw.name}: not a valid gzip stream: {e}") from e
    if len(body) != data_len:
        what = "more" if len(body) > data_len else f"only {len(body)}"
        raise ConvertError(f"{raw.name}: {what} data bytes, expected {data_len}")
    return body


def check_labels(raw: RawFile, body: bytes) -> None:
    bad = next((i for i, v in enumerate(body) if v >= NUM_CLASSES), None)
    if bad is not None:
        raise ConvertError(f"{raw.name}: label {body[bad]} of sample {bad} is not a class 0-{NUM_CLASSES - 1}")


def to_tensors(images: Any, labels: Any) -> dict[str, Any]:
    """The dataset fields from the `uint8` arrays `images [N, H, W]` and `labels [N]`:
    `image` `float32 [N, 1, H, W]` = byte / 255 and `label` `int64 [N]`."""
    if images.shape[0] != labels.shape[0]:
        raise ConvertError(f"{images.shape[0]} images but {labels.shape[0]} labels")
    n, h, w = images.shape
    pixels = images.astype(np.float32) / np.float32(255.0)
    if pixels.dtype != np.float32:  # pragma: no cover - NumPy's promotion rules
        raise ConvertError(f"pixel dtype {pixels.dtype}")
    return {
        "image": np.ascontiguousarray(pixels.reshape(n, 1, h, w)),
        "label": np.ascontiguousarray(labels.astype(np.int64)),
    }


def serialize(tensors: dict[str, Any]) -> bytes:
    """The safetensors bytes, without `__metadata__`; the header is checked to have exactly the
    expected tensors in the expected order."""
    data = safetensors.numpy.save(tensors, metadata=None)
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n])
    if set(header) != set(tensors):
        raise ConvertError(f"unexpected safetensors header {list(header)}")
    order = sorted(header, key=lambda k: header[k]["data_offsets"][0])
    if order != ["label", "image"]:
        raise ConvertError(f"unexpected tensor order {order}")
    return data


# ---------------------------------------------------------------------------------------------


def convert(
    out_dir: str,
    raw_dir: str,
    fetch: Fetcher = download,
    mirrors: tuple[str, ...] = MIRRORS,
    raw_files: dict[str, RawFile] | None = None,
) -> list[dict[str, Any]]:
    """Writes both outputs under `out_dir` (as fetch_fixtures.py places them) and returns their
    manifest entries. `raw_files` replaces `RAW` (for tests)."""
    raw_files = RAW if raw_files is None else raw_files
    entries = []
    for name, rel, images_key, labels_key in OUTPUTS:
        arrays = []
        for key in (images_key, labels_key):
            raw = raw_files[key]
            body = parse_idx(raw, obtain_raw(raw, raw_dir, fetch, mirrors))
            if key == labels_key:
                check_labels(raw, body)
            arrays.append(np.frombuffer(body, dtype=np.uint8).reshape(raw.dims))
        data = serialize(to_tensors(*arrays))
        target = os.path.join(out_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".partial-", dir=os.path.dirname(target))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.chmod(tmp, 0o644)  # mkstemp's 0o600 would carry over to the cache file
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        entries.append(
            {
                "name": name,
                "url": RELEASE_URL + rel.rsplit("/", 1)[1],
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "license": LICENSE,
                "path": rel,
            }
        )
    return entries


def manifest_problems(manifest_path: str, entries: list[dict[str, Any]]) -> list[str]:
    """Differences between the converted files' entries and the manifest's."""
    try:
        listed = {e["name"]: e for e in fetch_fixtures.load_manifest(manifest_path)}
    except (fetch_fixtures.FetchError, OSError) as e:
        return [str(e)]
    problems = []
    for want in entries:
        have = listed.get(want["name"])
        if have is None:
            problems.append(f"{want['name']}: not in {manifest_path}")
            continue
        for key in ("url", "sha256", "size", "license", "path"):
            if have[key] != want[key]:
                problems.append(f"{want['name']}: manifest {key} {have[key]!r}, converted {want[key]!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, help="default: fixtures/cache")
    parser.add_argument("--raw-dir", default=None, help="the four .gz files (default: <cache-dir>/mnist/raw)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="convert into a temporary directory, compare with the manifest")
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="convert with NumPy/safetensors versions other than the pinned ones (never for the release)",
    )
    args = parser.parse_args(argv)
    if IMPORT_ERROR is not None:
        print(f"error: {IMPORT_ERROR}. Install tools/reference/requirements.txt (docs/dev-environment.md).", file=sys.stderr)
        return 2
    mismatch = version_problems()
    if mismatch and not args.allow_version_mismatch:
        for line in mismatch:
            print(f"error: {line}", file=sys.stderr)
        return 2
    raw_dir = args.raw_dir or os.path.join(args.cache_dir, "mnist", "raw")
    try:
        if args.check:
            with tempfile.TemporaryDirectory() as tmp:
                entries = convert(tmp, raw_dir)
        else:
            entries = convert(args.cache_dir, raw_dir)
    except (ConvertError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for e in entries:
        print(f"{e['name']}\t{e['sha256']}\t{e['size']}\t{e['path']}")
    if not args.check:
        print(f"wrote {len(entries)} files to {args.cache_dir}")
        return 0
    problems = manifest_problems(args.manifest, entries)
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s): the converted files differ from {args.manifest}", file=sys.stderr)
        return 1
    print(f"ok: {len(entries)} files match {args.manifest}")
    return 0


def version_problems() -> list[str]:
    """NumPy and safetensors must be the pinned versions (the converter doesn't need PyTorch)."""
    import gen_ops

    pinned = gen_ops.pinned_versions()
    installed = {"numpy": np.__version__, "safetensors": safetensors.__version__}
    return [
        f"{name} {installed[name]} is installed, requirements.txt pins {pinned.get(name)}"
        for name in ("numpy", "safetensors")
        if pinned.get(name) != installed[name]
    ]


if __name__ == "__main__":
    sys.exit(main())
