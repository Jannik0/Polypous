"""Shared helpers for reference fixtures: layout, sidecar schema, safetensors headers, tree checks.

Standard library only (Python 3.12+), so CI and the unit tests can use it without PyTorch.
The layout and the sidecar schema are specified in docs/testing-strategy.md §3.
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import Any

SCHEMA_VERSION = 1

# Polypous dtype name -> (safetensors dtype, element size in bytes).
DTYPES: dict[str, tuple[str, int]] = {
    "f32": ("F32", 4),
    "f16": ("F16", 2),
    "bf16": ("BF16", 2),
    "i64": ("I64", 8),
    "i32": ("I32", 4),
    "bool": ("BOOL", 1),
}

# Size budget (docs/testing-strategy.md §3). A fixture is one case: its `.safetensors` and `.json`.
MAX_FIXTURE_BYTES = 1024 * 1024
MAX_SMALL_TOTAL_BYTES = 50 * 1024 * 1024

# Bound on a safetensors header we are willing to read. Fixture headers are a few hundred bytes.
MAX_HEADER_BYTES = 1024 * 1024

SIDECAR_KEYS = (
    "schema_version",
    "op",
    "case",
    "description",
    "attributes",
    "inputs",
    "outputs",
    "tolerance",
    "reference",
    "notes",
    "seed",
    "generator",
    "versions",
)

# The keys of a model case sidecar (docs/testing-strategy.md §3.5), in order.
MODEL_SIDECAR_KEYS = (
    "schema_version",
    "model",
    "case",
    "description",
    "compute_dtype",
    "inputs",
    "outputs",
    "tolerance",
    "reference",
    "notes",
    "seed",
    "generator",
    "versions",
)

# The files of a model directory besides its cases (docs/testing-strategy.md §3.1).
MODEL_GRAPH = "model.graph.json"
MODEL_WEIGHTS = "model.safetensors"
COMPUTE_DTYPES = ("f32", "f16", "bf16")

# The keys of a sidecar for an op with params (docs/testing-strategy.md §3.2): `params` right after
# `inputs`. Which ops have params is the op catalog's business (crates/polypous-ir/tests/catalog.rs
# checks it); here both forms are valid.
SIDECAR_KEYS_WITH_PARAMS = SIDECAR_KEYS[:6] + ("params",) + SIDECAR_KEYS[6:]


# Float formats by safetensors dtype: (struct code of the bit pattern, mantissa bits, exponent bits).
# Used to find subnormal values, which fixtures must not contain (docs/specs/graph-ir.md §3.1.8:
# devices may flush them to zero, so results involving them are backend-defined).
FLOAT_BITS: dict[str, tuple[str, int, int]] = {
    "F32": ("I", 23, 8),
    "F16": ("H", 10, 5),
    "BF16": ("H", 7, 8),
}


class FixtureError(Exception):
    """A fixture file is malformed or does not match its sidecar."""


def canonical_json(obj: Any) -> str:
    """The exact text written for a sidecar: 2-space indent, ASCII only, LF, trailing newline."""
    return json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def read_safetensors_header(path: str) -> dict[str, Any]:
    """Reads and bounds-checks the JSON header of a safetensors file.

    Returns the header without `__metadata__`. Raises FixtureError on any inconsistency: header
    length, dtype, shape, data offsets that overlap, leave gaps or don't match the file size.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise FixtureError(f"{path}: shorter than the 8-byte header length")
        (n,) = struct.unpack("<Q", prefix)
        if n > MAX_HEADER_BYTES or n > size - 8:
            raise FixtureError(f"{path}: header length {n} is out of bounds")
        raw = f.read(n)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise FixtureError(f"{path}: header is not valid JSON: {e}") from e
    if not isinstance(header, dict):
        raise FixtureError(f"{path}: header is not a JSON object")
    header.pop("__metadata__", None)
    by_name = {st: (name, width) for name, (st, width) in DTYPES.items()}
    spans = []
    for name, info in header.items():
        if not isinstance(info, dict):
            raise FixtureError(f"{path}: tensor {name!r} entry is not an object")
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if dtype not in by_name:
            raise FixtureError(f"{path}: tensor {name!r} has unsupported dtype {dtype!r}")
        if not (isinstance(shape, list) and all(isinstance(d, int) and d >= 0 for d in shape)):
            raise FixtureError(f"{path}: tensor {name!r} has a bad shape {shape!r}")
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(o, int) for o in offsets)
            and 0 <= offsets[0] <= offsets[1]
        ):
            raise FixtureError(f"{path}: tensor {name!r} has bad data offsets {offsets!r}")
        expected = math.prod(shape) * by_name[dtype][1]
        if offsets[1] - offsets[0] != expected:
            raise FixtureError(f"{path}: tensor {name!r} needs {expected} bytes, has {offsets[1] - offsets[0]}")
        spans.append((offsets[0], offsets[1]))
    position = 0
    for start, end in sorted(spans):
        if start != position:
            raise FixtureError(f"{path}: tensor data has a gap or overlap at byte {start}")
        position = end
    if position != size - 8 - n:
        raise FixtureError(f"{path}: tensor data covers {position} bytes, file has {size - 8 - n}")
    return header


def validate_case(json_path: str) -> list[str]:
    """Checks a sidecar against the schema and against its safetensors file. Returns problems."""
    problems: list[str] = []
    try:
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        sidecar = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return [f"{json_path}: cannot read sidecar: {e}"]
    if not isinstance(sidecar, dict):
        return [f"{json_path}: sidecar is not a JSON object"]
    if tuple(sidecar.keys()) not in (SIDECAR_KEYS, SIDECAR_KEYS_WITH_PARAMS):
        problems.append(
            f"{json_path}: keys {list(sidecar.keys())} differ from the schema {list(SIDECAR_KEYS)} "
            "(with `params` after `inputs` for ops with params)"
        )
        return problems
    if sidecar["schema_version"] != SCHEMA_VERSION:
        problems.append(f"{json_path}: schema_version {sidecar['schema_version']!r} is not {SCHEMA_VERSION}")
    if text != canonical_json(sidecar):
        problems.append(f"{json_path}: not in canonical form (rerun the generator)")
    stem = os.path.splitext(os.path.basename(json_path))[0]
    if sidecar["case"] != stem:
        problems.append(f"{json_path}: case {sidecar['case']!r} does not match the file name")
    if sidecar["op"] != os.path.basename(os.path.dirname(json_path)):
        problems.append(f"{json_path}: op {sidecar['op']!r} does not match the directory name")
    problems += tolerance_problems(json_path, sidecar["tolerance"])
    st_path = os.path.splitext(json_path)[0] + ".safetensors"
    try:
        header = read_safetensors_header(st_path)
    except (OSError, FixtureError) as e:
        return problems + [str(e)]
    problems += entry_problems(json_path, st_path, sidecar, header, ("inputs", "params", "outputs"))
    return problems + subnormal_problems(st_path, header)


def tolerance_problems(json_path: str, tol: Any) -> list[str]:
    """A sidecar tolerance must be `{"exact", "atol", "rtol"}`, non-negative, and exact only with zeros."""
    if not (
        isinstance(tol, dict)
        and set(tol) == {"exact", "atol", "rtol"}
        and isinstance(tol["exact"], bool)
        and all(isinstance(tol[k], (int, float)) and tol[k] >= 0 for k in ("atol", "rtol"))
        and (not tol["exact"] or tol["atol"] == tol["rtol"] == 0)
    ):
        return [f"{json_path}: bad tolerance {tol!r}"]
    return []


def entry_problems(
    json_path: str, st_path: str, sidecar: dict[str, Any], header: dict[str, Any], roles: tuple[str, ...]
) -> list[str]:
    """The tensor lists of a sidecar (`roles` that are present) against the safetensors header: each
    entry `{"name", "dtype", "shape"}` is in the file with that dtype and shape, names are unique and
    the file has no other tensors. `outputs` must not be empty."""
    problems: list[str] = []
    declared: list[str] = []
    for role in roles:
        if role not in sidecar:
            continue
        entries = sidecar[role]
        if not isinstance(entries, list) or (role == "outputs" and not entries):
            problems.append(f"{json_path}: {role} must be a list (outputs non-empty)")
            continue
        for entry in entries:
            if not (isinstance(entry, dict) and set(entry) == {"name", "dtype", "shape"}):
                problems.append(f"{json_path}: bad {role} entry {entry!r}")
                continue
            name, dtype, shape = entry["name"], entry["dtype"], entry["shape"]
            declared.append(name)
            info = header.get(name)
            if info is None:
                problems.append(f"{json_path}: {role} tensor {name!r} is missing from {st_path}")
            elif dtype not in DTYPES or DTYPES[dtype][0] != info["dtype"] or shape != info["shape"]:
                problems.append(
                    f"{json_path}: {role} tensor {name!r} is {dtype} {shape} in the sidecar, "
                    f"{info['dtype']} {info['shape']} in the file"
                )
    if len(set(declared)) != len(declared):
        problems.append(f"{json_path}: tensor names are not unique")
    extra = sorted(set(header) - set(declared))
    if extra:
        problems.append(f"{json_path}: {st_path} has undeclared tensors {extra}")
    return problems


def read_sidecar(json_path: str) -> tuple[Any, str | None]:
    """The parsed JSON of a sidecar and its text, or `(None, problem)`."""
    try:
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return json.loads(text), text
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return None, f"{json_path}: cannot read: {e}"


def read_model_graph(model_dir: str) -> tuple[dict[str, Any] | None, list[str]]:
    """The graph document of a model directory, checked for what the case checks need: an object
    with `inputs`, `outputs` and `params` lists of named entries, params with literal shapes. Full
    validation is polypous-ir's (the Rust model tests load every graph)."""
    path = os.path.join(model_dir, MODEL_GRAPH)
    graph, text = read_sidecar(path)
    if graph is None:
        return None, [text or f"{path}: cannot read"]
    if not isinstance(graph, dict):
        return None, [f"{path}: not a JSON object"]
    for key in ("inputs", "outputs", "params"):
        entries = graph.get(key)
        if not (isinstance(entries, list) and all(isinstance(e, dict) and isinstance(e.get("name"), str) for e in entries)):
            return None, [f"{path}: {key} must be a list of objects with a name"]
    for p in graph["params"]:
        shape = p.get("shape")
        if not (isinstance(shape, list) and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)):
            return None, [f"{path}: param {p['name']!r} has no literal shape"]
    return graph, []


def validate_model_weights(model_dir: str, graph: dict[str, Any]) -> list[str]:
    """`model.safetensors` holds exactly the graph's params, as F32 with the declared shapes, and no
    subnormal values (runtime.md §2: strict name match)."""
    st_path = os.path.join(model_dir, MODEL_WEIGHTS)
    try:
        header = read_safetensors_header(st_path)
    except (OSError, FixtureError) as e:
        return [str(e)]
    problems = []
    declared = {p["name"]: p["shape"] for p in graph["params"]}
    if len(declared) != len(graph["params"]):
        problems.append(f"{st_path}: the graph declares a param twice")
    for name in sorted(set(declared) - set(header)):
        problems.append(f"{st_path}: param {name!r} is missing")
    for name in sorted(set(header) - set(declared)):
        problems.append(f"{st_path}: tensor {name!r} is not a param of the graph")
    for name in sorted(set(header) & set(declared)):
        info = header[name]
        if info["dtype"] != "F32" or info["shape"] != declared[name]:
            problems.append(
                f"{st_path}: param {name!r} is {info['dtype']} {info['shape']}, the graph declares F32 {declared[name]}"
            )
    return problems + subnormal_problems(st_path, header)


def validate_model_case(json_path: str, graph: dict[str, Any]) -> list[str]:
    """Checks a model case sidecar against the schema (docs/testing-strategy.md §3.5), the model's
    graph (input and output names in order; float inputs in the compute dtype) and its safetensors
    file. Returns problems."""
    sidecar, text = read_sidecar(json_path)
    if sidecar is None:
        return [text or f"{json_path}: cannot read"]
    if not isinstance(sidecar, dict):
        return [f"{json_path}: sidecar is not a JSON object"]
    if tuple(sidecar.keys()) != MODEL_SIDECAR_KEYS:
        return [f"{json_path}: keys {list(sidecar.keys())} differ from the schema {list(MODEL_SIDECAR_KEYS)}"]
    problems: list[str] = []
    if sidecar["schema_version"] != SCHEMA_VERSION:
        problems.append(f"{json_path}: schema_version {sidecar['schema_version']!r} is not {SCHEMA_VERSION}")
    if text != canonical_json(sidecar):
        problems.append(f"{json_path}: not in canonical form (rerun the generator)")
    stem = os.path.splitext(os.path.basename(json_path))[0]
    if sidecar["case"] != stem:
        problems.append(f"{json_path}: case {sidecar['case']!r} does not match the file name")
    if sidecar["model"] != os.path.basename(os.path.dirname(json_path)):
        problems.append(f"{json_path}: model {sidecar['model']!r} does not match the directory name")
    compute = sidecar["compute_dtype"]
    if compute not in COMPUTE_DTYPES:
        problems.append(f"{json_path}: compute_dtype {compute!r} is not one of {list(COMPUTE_DTYPES)}")
    problems += tolerance_problems(json_path, sidecar["tolerance"])
    for role, key in (("inputs", "inputs"), ("outputs", "outputs")):
        entries = sidecar[role]
        names = [e.get("name") for e in entries] if isinstance(entries, list) and all(isinstance(e, dict) for e in entries) else None
        want = [e["name"] for e in graph[key]]
        if names != want:
            problems.append(f"{json_path}: {role} {names} differ from the graph's {key} {want} (same names, same order)")
    for spec, entry in zip(graph["inputs"], sidecar["inputs"] if isinstance(sidecar["inputs"], list) else []):
        if spec.get("dtype") == "float" and isinstance(entry, dict) and entry.get("dtype") != compute:
            problems.append(f"{json_path}: input {spec['name']!r} is {entry.get('dtype')}, the compute dtype is {compute}")
    st_path = os.path.splitext(json_path)[0] + ".safetensors"
    try:
        header = read_safetensors_header(st_path)
    except (OSError, FixtureError) as e:
        return problems + [str(e)]
    problems += entry_problems(json_path, st_path, sidecar, header, ("inputs", "outputs"))
    return problems + subnormal_problems(st_path, header)


def validate_model_dir(model_dir: str) -> list[str]:
    """Checks one `models/<model>/` directory: its graph, its weights and every case (a `.json`
    other than the graph, with a `.safetensors` next to it). No other files are allowed."""
    if not os.path.isfile(os.path.join(model_dir, MODEL_GRAPH)):
        return [f"{model_dir}: no {MODEL_GRAPH}"]
    graph, problems = read_model_graph(model_dir)
    if graph is None:
        return problems
    problems += validate_model_weights(model_dir, graph)
    names = sorted(os.listdir(model_dir))
    cases = 0
    for name in names:
        path = os.path.join(model_dir, name)
        stem, ext = os.path.splitext(name)
        if name in (MODEL_GRAPH, MODEL_WEIGHTS):
            continue
        if os.path.isdir(path) or ext not in (".json", ".safetensors") or "." in stem:
            problems.append(f"{path}: unexpected file in a model directory")
        elif ext == ".json":
            cases += 1
            problems += validate_model_case(path, graph)
        elif stem + ".json" not in names:
            problems.append(f"{path}: no JSON sidecar")
    if cases == 0:
        problems.append(f"{model_dir}: no cases")
    return problems


def subnormal_problems(st_path: str, header: dict[str, Any]) -> list[str]:
    """Float tensors of a safetensors file that contain subnormal values (non-zero, exponent bits all
    zero). `header` comes from `read_safetensors_header`, which checked the offsets."""
    with open(st_path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(8 + n)
        data = f.read()
    problems = []
    for name, info in sorted(header.items()):
        if info["dtype"] not in FLOAT_BITS:
            continue
        code, mantissa, _ = FLOAT_BITS[info["dtype"]]
        start, end = info["data_offsets"]
        count = (end - start) // struct.calcsize(code)
        values = struct.unpack(f"<{count}{code}", data[start:end])
        exponent_mask = ((1 << (struct.calcsize(code) * 8 - 1)) - 1) & ~((1 << mantissa) - 1)
        bad = [i for i, v in enumerate(values) if v & exponent_mask == 0 and v & ((1 << mantissa) - 1)]
        if bad:
            problems.append(
                f"{st_path}: tensor {name!r} has subnormal values at flat indices {bad[:8]}; devices may "
                "flush them to zero (graph-ir.md §3.1.8), so fixtures use normal values"
            )
    return problems


def list_files(root: str) -> list[str]:
    """All files under `root`, as sorted '/'-separated relative paths. Empty if root is missing."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            found.append(rel.replace(os.sep, "/"))
    return sorted(found)


def compare_trees(committed: str, generated: str) -> list[str]:
    """Compares committed fixtures with freshly generated ones, byte for byte.

    Returns one line per problem: a fixture that differs, one the generator produces that is not
    committed (missing), and a committed file that no case produces (extra).
    """
    have, want = set(list_files(committed)), set(list_files(generated))
    problems = [f"missing: {p} (generated, not committed)" for p in sorted(want - have)]
    problems += [f"extra: {p} (committed, no case generates it)" for p in sorted(have - want)]
    for rel in sorted(have & want):
        with open(os.path.join(committed, rel), "rb") as a, open(os.path.join(generated, rel), "rb") as b:
            if a.read() != b.read():
                problems.append(f"changed: {rel} (differs from the generator's output)")
    return problems


def fixture_key(rel: str) -> str:
    """The fixture a file belongs to: its path up to the first `.` of the file name. A case's
    `.safetensors` and `.json` are one fixture, and so are a model's `model.graph.json` and
    `model.safetensors` (docs/testing-strategy.md §3.1)."""
    head, name = os.path.split(rel)
    return os.path.join(head, name.split(".", 1)[0]).replace(os.sep, "/")


def fixture_sizes(root: str) -> dict[str, int]:
    """Bytes per fixture under `root` (`fixture_key`)."""
    sizes: dict[str, int] = {}
    for rel in list_files(root):
        key = fixture_key(rel)
        sizes[key] = sizes.get(key, 0) + os.path.getsize(os.path.join(root, rel))
    return sizes


def check_sizes(
    root: str,
    max_fixture: int = MAX_FIXTURE_BYTES,
    max_total: int = MAX_SMALL_TOTAL_BYTES,
) -> list[str]:
    """Problems with the size budget of `fixtures/small/`: a fixture too large, or the total."""
    sizes = fixture_sizes(root)
    problems = [
        f"{key}: {size} bytes, budget is {max_fixture} per fixture"
        for key, size in sorted(sizes.items())
        if size >= max_fixture
    ]
    total = sum(sizes.values())
    if total >= max_total:
        problems.append(f"{root}: {total} bytes in total, budget is {max_total}")
    return problems
