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


# ---------------------------------------------------------------------------------------------
# Gradient fixtures: op sidecar schema version 2 (docs/testing-strategy.md §3.6), in
# `fixtures/small/grads/<op>/<case>.{json,safetensors}`.

GRAD_SCHEMA_VERSION = 2
GRAD_MODES = ("inference", "training")
# The only ops with training-mode cases, and the key each of them adds in training mode.
GRAD_TRAINING_KEYS = {"nn.dropout": "mask", "nn.batch_norm": "running_stats"}
# batch_norm's running statistics: params that never get a gradient (graph-ir.md §3.1.11).
RUNNING_STATS = ("running_mean", "running_var")


def grad_sidecar_keys(op: Any, mode: Any, with_params: bool) -> tuple[str, ...]:
    """The keys of a schema-2 sidecar, in order: §3.2's keys plus `mode`, `mask` (dropout in training
    mode), `upstream`, `running_stats` (batch_norm in training mode), `gradients` and
    `gradient_tolerance`."""
    extra = GRAD_TRAINING_KEYS.get(op) if mode == "training" else None
    keys = ["schema_version", "op", "case", "description", "mode", "attributes", "inputs"]
    keys += ["params"] if with_params else []
    keys += ["mask"] if extra == "mask" else []
    keys += ["upstream", "outputs"]
    keys += ["running_stats"] if extra == "running_stats" else []
    keys += ["gradients", "tolerance", "gradient_tolerance", "reference", "notes", "seed", "generator", "versions"]
    return tuple(keys)


def _named_entries(entries: Any, extra: tuple[str, ...] = ()) -> list[dict[str, Any]] | None:
    """`entries` if it is a list of `{"name", *extra, "dtype", "shape"}` objects (keys in that order)
    with a string name, a known dtype and a shape of non-negative integers; otherwise None."""
    keys = ["name", *extra, "dtype", "shape"]
    if not isinstance(entries, list):
        return None
    for e in entries:
        if not (
            isinstance(e, dict)
            and list(e) == keys
            and all(isinstance(e[k], str) for k in ("name", *extra))
            and e["dtype"] in DTYPES
            and isinstance(e["shape"], list)
            and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in e["shape"])
        ):
            return None
    return entries


def output_port_name(output: str) -> str:
    """The port of an output tensor (§3.2 names): `out` -> `out`, `out0` -> `0` (core.split), `values`."""
    return output[3:] if output.startswith("out") and output[3:].isdigit() else output


def validate_grad_case(json_path: str) -> list[str]:
    """Checks a gradient sidecar (schema version 2, docs/testing-strategy.md §3.6) against the schema
    and its safetensors file. Returns problems."""
    sidecar, text = read_sidecar(json_path)
    if sidecar is None:
        return [text or f"{json_path}: cannot read"]
    if not isinstance(sidecar, dict):
        return [f"{json_path}: sidecar is not a JSON object"]
    op, mode = sidecar.get("op"), sidecar.get("mode")
    want = grad_sidecar_keys(op, mode, "params" in sidecar)
    if tuple(sidecar.keys()) != want:
        return [f"{json_path}: keys {list(sidecar.keys())} differ from schema 2 for this op and mode {list(want)} "
                "(`mask` only for nn.dropout and `running_stats` only for nn.batch_norm, in training mode)"]
    problems: list[str] = []
    if sidecar["schema_version"] != GRAD_SCHEMA_VERSION:
        problems.append(f"{json_path}: schema_version {sidecar['schema_version']!r} is not {GRAD_SCHEMA_VERSION}")
    if text != canonical_json(sidecar):
        problems.append(f"{json_path}: not in canonical form (rerun the generator)")
    stem = os.path.splitext(os.path.basename(json_path))[0]
    if sidecar["case"] != stem:
        problems.append(f"{json_path}: case {sidecar['case']!r} does not match the file name")
    if not stem.endswith("_f32"):
        problems.append(f"{json_path}: gradient case names end in _f32 (gradients are f32 only in M2)")
    if op != os.path.basename(os.path.dirname(json_path)):
        problems.append(f"{json_path}: op {op!r} does not match the directory name")
    if mode not in GRAD_MODES:
        problems.append(f"{json_path}: mode {mode!r} is not one of {list(GRAD_MODES)}")
    elif mode == "training" and op not in GRAD_TRAINING_KEYS:
        problems.append(f"{json_path}: only {sorted(GRAD_TRAINING_KEYS)} have training-mode cases")
    problems += tolerance_problems(json_path, sidecar["tolerance"])
    problems += tolerance_problems(json_path, sidecar["gradient_tolerance"])

    inputs = _named_entries(sidecar["inputs"])
    params = _named_entries(sidecar.get("params", []))
    outputs = _named_entries(sidecar["outputs"])
    upstream = _named_entries(sidecar["upstream"], ("port",))
    gradients = _named_entries(sidecar["gradients"], ("of",))
    running = _named_entries(sidecar.get("running_stats", []), ("param",))
    mask = _named_entries([sidecar["mask"]]) if "mask" in sidecar else []
    lists = {"inputs": inputs, "params": params, "outputs": outputs, "upstream": upstream, "gradients": gradients,
             "running_stats": running, "mask": mask}
    bad = [k for k, v in lists.items() if v is None]
    if bad:
        return problems + [f"{json_path}: malformed {', '.join(bad)} (entries are {{\"name\", …, \"dtype\", \"shape\"}})"]
    assert inputs is not None and params is not None and outputs is not None and upstream is not None
    assert gradients is not None and running is not None and mask is not None  # checked just above
    if not outputs:
        problems.append(f"{json_path}: outputs must not be empty")
    floats = ("f32",)
    for role, entries in (("inputs", inputs), ("params", params), ("outputs", outputs)):
        for e in entries:
            if e["dtype"] in ("f16", "bf16"):
                problems.append(f"{json_path}: {role} tensor {e['name']!r} is {e['dtype']}; gradient fixtures are f32")
    for role, entries in (("upstream", upstream), ("gradients", gradients), ("running_stats", running), ("mask", mask)):
        for e in entries:
            if e["dtype"] not in floats:
                problems.append(f"{json_path}: {role} tensor {e['name']!r} is {e['dtype']}, not f32")

    by_name = {e["name"]: e for e in inputs + params}
    x = by_name.get("x")
    if mask and (mask[0]["name"] != "mask" or x is None or mask[0]["shape"] != x["shape"]):
        problems.append(f"{json_path}: mask must be named mask and have x's shape")
    # Upstream: one per float output port, in port order, named grad_<port>, with the output's shape.
    float_outputs = [e for e in outputs if e["dtype"] in floats]
    ports = [output_port_name(e["name"]) for e in float_outputs]
    if [u["port"] for u in upstream] != ports:
        problems.append(f"{json_path}: upstream ports {[u['port'] for u in upstream]} differ from the float output "
                        f"ports {ports} (one per float output, in order)")
    for u, o in zip(upstream, float_outputs):
        if u["name"] != "grad_" + u["port"] or u["shape"] != o["shape"]:
            problems.append(f"{json_path}: upstream {u['name']!r} must be grad_{u['port']} with {o['name']}'s shape")
    # Running statistics: new values of running_mean, then running_var, with the params' shapes.
    if running:
        if [(r["name"], r["param"]) for r in running] != [("new_" + p, p) for p in RUNNING_STATS]:
            problems.append(f"{json_path}: running_stats must be new_running_mean, new_running_var of those params")
        for r in running:
            p = by_name.get(r["param"])
            if p is None or p not in params or r["shape"] != p["shape"]:
                problems.append(f"{json_path}: running statistic {r['name']!r} has no param {r['param']!r} of its shape")
    # Gradients: every float input in order, then every param but the running statistics.
    want_of = [e["name"] for e in inputs if e["dtype"] in floats]
    want_of += [e["name"] for e in params if e["name"] not in RUNNING_STATS]
    if [g["of"] for g in gradients] != want_of:
        problems.append(f"{json_path}: gradients are of {[g['of'] for g in gradients]}, expected {want_of} "
                        "(every float input in order, then every param but the running statistics)")
    for g in gradients:
        target = by_name.get(g["of"])
        if g["name"] != "grad_" + g["of"] or target is None or g["shape"] != target["shape"]:
            problems.append(f"{json_path}: gradient {g['name']!r} must be grad_{g['of']} with {g['of']}'s shape")

    st_path = os.path.splitext(json_path)[0] + ".safetensors"
    try:
        header = read_safetensors_header(st_path)
    except (OSError, FixtureError) as e:
        return problems + [str(e)]
    plain = {role: [{"name": e["name"], "dtype": e["dtype"], "shape": e["shape"]} for e in entries]
             for role, entries in lists.items()}
    problems += entry_problems(json_path, st_path, plain, header,
                               ("inputs", "params", "mask", "upstream", "outputs", "running_stats", "gradients"))
    return problems + subnormal_problems(st_path, header)


# ---------------------------------------------------------------------------------------------
# Initial-value fixtures (docs/testing-strategy.md §3.9), in `fixtures/small/init/<kind>/<case>`.

INIT_SCHEMA_VERSION = 1
INIT_SIDECAR_KEYS = (
    "schema_version",
    "kind",
    "case",
    "description",
    "seed",
    "param",
    "outputs",
    "tolerance",
    "reference",
    "notes",
    "generator",
    "versions",
)
INIT_PARAM_KEYS = ("name", "shape", "init", "role", "fans")
INIT_KINDS = (
    "zeros", "ones", "constant", "uniform", "normal",
    "kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal",
)
INIT_FAN_KINDS = ("kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal")
INIT_NORMAL_KINDS = ("normal", "kaiming_normal", "xavier_normal")
# The roles with fans (graph-ir.md §2.4.1, "Fans").
FAN_ROLES = {("nn.linear", "weight"), ("nn.embedding", "weight"), ("nn.conv1d", "weight"), ("nn.conv2d", "weight")}


def _is_uint(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def validate_init_case(json_path: str) -> list[str]:
    """Checks an initial-value sidecar (docs/testing-strategy.md §3.9) against the schema and its
    safetensors file: one `values` tensor, F32, with the param's shape. Returns problems."""
    sidecar, text = read_sidecar(json_path)
    if sidecar is None:
        return [text or f"{json_path}: cannot read"]
    if not isinstance(sidecar, dict):
        return [f"{json_path}: sidecar is not a JSON object"]
    if tuple(sidecar.keys()) != INIT_SIDECAR_KEYS:
        return [f"{json_path}: keys {list(sidecar.keys())} differ from the schema {list(INIT_SIDECAR_KEYS)}"]
    problems: list[str] = []
    if sidecar["schema_version"] != INIT_SCHEMA_VERSION:
        problems.append(f"{json_path}: schema_version {sidecar['schema_version']!r} is not {INIT_SCHEMA_VERSION}")
    if text != canonical_json(sidecar):
        problems.append(f"{json_path}: not in canonical form (rerun the generator)")
    stem = os.path.splitext(os.path.basename(json_path))[0]
    if sidecar["case"] != stem:
        problems.append(f"{json_path}: case {sidecar['case']!r} does not match the file name")
    kind = sidecar["kind"]
    if kind not in INIT_KINDS:
        problems.append(f"{json_path}: kind {kind!r} is not one of {list(INIT_KINDS)}")
    if kind != os.path.basename(os.path.dirname(json_path)):
        problems.append(f"{json_path}: kind {kind!r} does not match the directory name")
    if not _is_uint(sidecar["seed"]) or sidecar["seed"] >= 1 << 64:
        problems.append(f"{json_path}: seed must be a u64")
    param = sidecar["param"]
    if not (isinstance(param, dict) and tuple(param) == INIT_PARAM_KEYS):
        return problems + [f"{json_path}: param must have exactly the keys {list(INIT_PARAM_KEYS)}"]
    shape = param["shape"]
    if not (isinstance(param["name"], str) and isinstance(shape, list) and shape and all(_is_uint(d) and d > 0 for d in shape)):
        problems.append(f"{json_path}: param needs a name and a literal shape")
        shape = None
    init = param["init"]
    if not (isinstance(init, dict) and init.get("kind") == kind):
        problems.append(f"{json_path}: param.init must be an init object of kind {kind!r}")
    role = param["role"]
    if role is not None and not (isinstance(role, dict) and list(role) == ["op", "param"]
                                 and all(isinstance(v, str) for v in role.values())):
        problems.append(f"{json_path}: param.role must be null or {{\"op\", \"param\"}}")
        role = None
    fans = param["fans"]
    has_fans = role is not None and (role["op"], role["param"]) in FAN_ROLES
    if kind in INIT_FAN_KINDS and not has_fans:
        problems.append(f"{json_path}: a fan-based kind needs a role with fans")
    if has_fans and shape is not None:
        rf = math.prod(shape[2:]) if len(shape) >= 2 else 0
        want = {"fan_in": shape[1] * rf, "fan_out": shape[0] * rf} if len(shape) >= 2 else None
        if fans != want:
            problems.append(f"{json_path}: param.fans {fans!r} differ from the fans of its shape {want!r}")
    elif fans is not None:
        problems.append(f"{json_path}: param.fans must be null for a role without fans")
    tol = sidecar["tolerance"]
    want_ulp = 1 if kind in INIT_NORMAL_KINDS else 0
    if not (isinstance(tol, dict) and list(tol) == ["ulp"] and tol["ulp"] in (0, want_ulp)):
        problems.append(f"{json_path}: tolerance must be {{\"ulp\": {want_ulp}}} for {kind} (or 0)")
    outputs = _named_entries(sidecar["outputs"])
    if outputs is None or [e["name"] for e in outputs] != ["values"] or outputs[0]["dtype"] != "f32" or (
        shape is not None and outputs[0]["shape"] != shape
    ):
        return problems + [f"{json_path}: outputs must be one f32 tensor `values` with the param's shape"]
    st_path = os.path.splitext(json_path)[0] + ".safetensors"
    try:
        header = read_safetensors_header(st_path)
    except (OSError, FixtureError) as e:
        return problems + [str(e)]
    problems += entry_problems(json_path, st_path, sidecar, header, ("outputs",))
    return problems + subnormal_problems(st_path, header)


# ---------------------------------------------------------------------------------------------
# Training-mode graph fixtures: model fixtures with case schema version 2 (docs/testing-strategy.md
# §3.10), in `fixtures/small/training/<model>/`.

TRAINING_SCHEMA_VERSION = 2


def training_sidecar_keys(mode: Any) -> tuple[str, ...]:
    """The keys of a schema-2 model case, in order (`training`, `running_stats` and `masks` only in
    training mode)."""
    keys = ["schema_version", "model", "case", "description", "compute_dtype", "mode"]
    keys += ["training"] if mode == "training" else []
    keys += ["inputs", "outputs"]
    keys += ["running_stats", "masks"] if mode == "training" else []
    keys += ["upstream", "gradients", "tolerance", "gradient_tolerance", "reference", "notes", "seed",
             "generator", "versions"]
    return tuple(keys)


def graph_param_uses(graph: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Every (node path, op, op param, graph param) of a graph after expanding repeats, in document
    order, iterations in order."""
    out: list[tuple[str, str, str, str]] = []

    def walk(nodes: Any, path: str, prefix: str) -> None:
        for node in nodes if isinstance(nodes, list) else []:
            if not isinstance(node, dict):
                continue
            if node.get("op") == "repeat":
                body = node.get("body") or {}
                count = node.get("count")
                for i in range(count if _is_uint(count) and count <= 65536 else 0):
                    inner = prefix + str(body.get("params_prefix", "")).replace("{i}", str(i))
                    walk(body.get("nodes"), f"{path}{node.get('id')}[{i}]/", inner)
            else:
                for role, name in (node.get("params") or {}).items():
                    out.append((f"{path}{node.get('id')}", str(node.get("op")), role, prefix + str(name)))

    walk(graph.get("nodes"), "", "")
    return out


def graph_dropouts(graph: dict[str, Any]) -> list[str]:
    """The node paths of every `nn.dropout` with `p > 0`, iterations in order (document order)."""
    out: list[str] = []

    def walk(nodes: Any, path: str) -> None:
        for node in nodes if isinstance(nodes, list) else []:
            if not isinstance(node, dict):
                continue
            if node.get("op") == "repeat":
                count = node.get("count")
                for i in range(count if _is_uint(count) and count <= 65536 else 0):
                    walk((node.get("body") or {}).get("nodes"), f"{path}{node.get('id')}[{i}]/")
            elif node.get("op") == "nn.dropout" and (node.get("attrs") or {}).get("p", 0.5) > 0:
                out.append(f"{path}{node.get('id')}")

    walk(graph.get("nodes"), "")
    return out


def trainable_params(graph: dict[str, Any]) -> list[str]:
    """Trainable params in declaration order (training.md §1.1): `trainable` true (the default) and
    not used as a batch_norm running statistic."""
    stats = {name for _, op, role, name in graph_param_uses(graph) if op == "nn.batch_norm" and role in RUNNING_STATS}
    return [p["name"] for p in graph["params"] if p.get("trainable", True) and p["name"] not in stats]


def validate_training_case(json_path: str, graph: dict[str, Any]) -> list[str]:
    """Checks a training-mode model case (schema version 2, docs/testing-strategy.md §3.10) against
    the schema, the model's graph and its safetensors file. Returns problems."""
    sidecar, text = read_sidecar(json_path)
    if sidecar is None:
        return [text or f"{json_path}: cannot read"]
    if not isinstance(sidecar, dict):
        return [f"{json_path}: sidecar is not a JSON object"]
    mode = sidecar.get("mode")
    want = training_sidecar_keys(mode)
    if tuple(sidecar.keys()) != want:
        return [f"{json_path}: keys {list(sidecar.keys())} differ from schema 2 for mode {mode!r} {list(want)}"]
    problems: list[str] = []
    if sidecar["schema_version"] != TRAINING_SCHEMA_VERSION:
        problems.append(f"{json_path}: schema_version {sidecar['schema_version']!r} is not {TRAINING_SCHEMA_VERSION}")
    if text != canonical_json(sidecar):
        problems.append(f"{json_path}: not in canonical form (rerun the generator)")
    stem = os.path.splitext(os.path.basename(json_path))[0]
    if sidecar["case"] != stem:
        problems.append(f"{json_path}: case {sidecar['case']!r} does not match the file name")
    if sidecar["model"] != os.path.basename(os.path.dirname(json_path)):
        problems.append(f"{json_path}: model {sidecar['model']!r} does not match the directory name")
    if mode not in GRAD_MODES:
        problems.append(f"{json_path}: mode {mode!r} is not one of {list(GRAD_MODES)}")
    if sidecar["compute_dtype"] != "f32":
        problems.append(f"{json_path}: compute_dtype must be f32 (gradients and training mode are f32 only in M2)")
    if mode == "training":
        t = sidecar["training"]
        if not (isinstance(t, dict) and list(t) == ["seed", "step"] and all(_is_uint(v) and v < 1 << 64 for v in t.values())):
            problems.append(f"{json_path}: training must be {{\"seed\", \"step\"}}, both u64")
    problems += tolerance_problems(json_path, sidecar["tolerance"])
    problems += tolerance_problems(json_path, sidecar["gradient_tolerance"])
    lists = {
        "inputs": _named_entries(sidecar["inputs"]),
        "outputs": _named_entries(sidecar["outputs"]),
        "running_stats": _named_entries(sidecar.get("running_stats", []), ("param",)),
        "masks": _named_entries(sidecar.get("masks", []), ("node",)),
        "upstream": _named_entries(sidecar["upstream"], ("output",)),
        "gradients": _named_entries(sidecar["gradients"], ("param",)),
    }
    bad = [k for k, v in lists.items() if v is None]
    if bad:
        return problems + [f"{json_path}: malformed {', '.join(bad)} (entries are {{\"name\", …, \"dtype\", \"shape\"}})"]
    ins, outs = lists["inputs"], lists["outputs"]
    assert all(v is not None for v in lists.values())  # checked just above
    for role, key in (("inputs", "inputs"), ("outputs", "outputs")):
        names, want_names = [e["name"] for e in lists[role]], [e["name"] for e in graph[key]]
        if names != want_names:
            problems.append(f"{json_path}: {role} {names} differ from the graph's {key} {want_names}")
    for e in ins + outs:
        if e["dtype"] in ("f16", "bf16"):
            problems.append(f"{json_path}: tensor {e['name']!r} is {e['dtype']}; training fixtures are f32")
    declared = {p["name"]: p["shape"] for p in graph["params"]}
    # Running statistics: those of every batch_norm node in document order, mean before var.
    uses = graph_param_uses(graph)
    # The order is the order the nodes run in (runtime.md §3.3), which the runtime test checks.
    want_stats = [name for _, op, role, name in uses if op == "nn.batch_norm" and role in RUNNING_STATS] if mode == "training" else []
    if sorted((r["name"], r["param"]) for r in lists["running_stats"]) != sorted(("new/" + p, p) for p in want_stats):
        problems.append(f"{json_path}: running_stats must be new/<param> of every running statistic {want_stats}")
    for r in lists["running_stats"]:
        if r["dtype"] != "f32" or r["shape"] != declared.get(r["param"]):
            problems.append(f"{json_path}: running statistic {r['name']!r} must be f32 with its param's shape")
    # Masks: one bool mask per dropout node path with p > 0.
    want_masks = graph_dropouts(graph) if mode == "training" else []
    if [(m["name"], m["node"]) for m in lists["masks"]] != [("mask/" + p, p) for p in want_masks]:
        problems.append(f"{json_path}: masks must be mask/<node path> of {want_masks}, in order")
    for m in lists["masks"]:
        if m["dtype"] != "bool":
            problems.append(f"{json_path}: mask {m['name']!r} must be bool")
    # Upstream: one per float graph output, named grad/<output>, with its shape.
    floats = [o for o in outs if o["dtype"] == "f32"]
    if [(u["name"], u["output"], u["shape"]) for u in lists["upstream"]] != [("grad/" + o["name"], o["name"], o["shape"]) for o in floats]:
        problems.append(f"{json_path}: upstream must be grad/<output> for every float output, with its shape")
    # Gradients: trainable params only, in declaration order, with their shapes.
    trainable = trainable_params(graph)
    grads = [(g["param"], g["name"]) for g in lists["gradients"]]
    if any(p not in trainable or n != "grad/" + p for p, n in grads) or [p for p, _ in grads] != [p for p in trainable if p in dict(grads)]:
        problems.append(f"{json_path}: gradients must be grad/<param> of trainable params {trainable}, in declaration order")
    for g in lists["gradients"]:
        if g["dtype"] != "f32" or g["shape"] != declared.get(g["param"]):
            problems.append(f"{json_path}: gradient {g['name']!r} must be f32 with its param's shape")
    st_path = os.path.splitext(json_path)[0] + ".safetensors"
    try:
        header = read_safetensors_header(st_path)
    except (OSError, FixtureError) as e:
        return problems + [str(e)]
    plain = {role: [{"name": e["name"], "dtype": e["dtype"], "shape": e["shape"]} for e in entries]
             for role, entries in lists.items()}
    problems += entry_problems(json_path, st_path, plain, header,
                               ("inputs", "outputs", "running_stats", "masks", "upstream", "gradients"))
    return problems + subnormal_problems(st_path, header)


def validate_training_dir(model_dir: str) -> list[str]:
    """Checks one `training/<model>/` directory: graph, weights (as in §3.5) and every case."""
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
            problems.append(f"{path}: unexpected file in a training model directory")
        elif ext == ".json":
            cases += 1
            problems += validate_training_case(path, graph)
        elif stem + ".json" not in names:
            problems.append(f"{path}: no JSON sidecar")
    if cases == 0:
        problems.append(f"{model_dir}: no cases")
    return problems


# ---------------------------------------------------------------------------------------------
# Training component fixtures (docs/testing-strategy.md §3.7), in
# `fixtures/small/train/<component>/<case>.{json,safetensors}`: the optimizers, gradient clipping
# and the schedules (#79). The losses (`loss.*`) are added with #80.

TRAIN_SCHEMA_VERSION = 1
TRAIN_SIDECAR_KEYS = (
    "schema_version",
    "component",
    "case",
    "description",
    "config",
    "inputs",
    "outputs",
    "tolerance",
    "reference",
    "notes",
    "seed",
    "generator",
    "versions",
)
# Schedule sidecars add `values` after `outputs`.
TRAIN_SCHEDULE_KEYS = TRAIN_SIDECAR_KEYS[:7] + ("values",) + TRAIN_SIDECAR_KEYS[7:]
# The fields of each optimizer kind (training.md §2.8) and its state names (§5.2).
TRAIN_OPTIMIZER_FIELDS = {
    "sgd": ("lr", "momentum", "dampening", "nesterov", "weight_decay"),
    "adam": ("lr", "betas", "eps", "weight_decay"),
    "adamw": ("lr", "betas", "eps", "weight_decay"),
}
TRAIN_SCHEDULE_FIELDS = {
    "constant": (),
    "step": ("every", "gamma"),
    "cosine": ("warmup_steps", "min_lr"),
    "linear_warmup": ("warmup_steps",),
}
TRAIN_OPTIMIZER_STEPS = 3
# testing-strategy.md §2.
TRAIN_OPTIMIZER_TOLERANCE = {"exact": False, "atol": 1e-6, "rtol": 1e-6}
TRAIN_CLIP_TOLERANCE = {"exact": False, "atol": 1e-5, "rtol": 1e-3}
TRAIN_COSINE_ATOL_PER_LR = 1e-14


def train_components() -> tuple[str, ...]:
    return (
        tuple(f"optimizer.{k}" for k in TRAIN_OPTIMIZER_FIELDS)
        + ("gradient_clip",)
        + tuple(f"schedule.{k}" for k in TRAIN_SCHEDULE_FIELDS)
    )


def optimizer_state_names(kind: str, config: dict[str, Any]) -> tuple[str, ...]:
    if kind == "sgd":
        return ("momentum_buffer",) if config.get("momentum", 0) != 0 else ()
    return ("exp_avg", "exp_avg_sq")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _train_config_problems(json_path: str, component: str, config: Any) -> list[str]:
    """The `config` object of a training component sidecar: the kind and the field names of
    training.md §2.8 (the ranges are the Rust config types' business; their tests check them)."""
    bad = f"{json_path}: bad config {config!r}"
    if not isinstance(config, dict):
        return [bad]
    if component.startswith("optimizer."):
        kind = component.split(".", 1)[1]
        fields = set(config) - {"kind"}
        if config.get("kind") != kind or not fields <= set(TRAIN_OPTIMIZER_FIELDS[kind]):
            return [f"{bad}: needs kind {kind!r} and only the fields {list(TRAIN_OPTIMIZER_FIELDS[kind])}"]
        lr = config.get("lr", 1e-3)
        if not (_is_number(lr) and lr >= 0):
            return [f"{bad}: lr must be a number >= 0"]
        return []
    if component == "gradient_clip":
        if list(config) != ["max_norm"] or not (_is_number(config["max_norm"]) and config["max_norm"] > 0):
            return [f"{bad}: must be {{\"max_norm\": number > 0}}"]
        return []
    kind = component.split(".", 1)[1]
    if list(config) != ["lr", "total_steps", "schedule"]:
        return [f"{bad}: must have exactly the keys lr, total_steps, schedule"]
    lr, total, schedule = config["lr"], config["total_steps"], config["schedule"]
    if not (_is_number(lr) and lr >= 0) or not (_is_uint(total) and total >= 1):
        return [f"{bad}: lr must be a number >= 0 and total_steps an integer >= 1"]
    if not (isinstance(schedule, dict) and schedule.get("kind") == kind
            and set(schedule) - {"kind"} <= set(TRAIN_SCHEDULE_FIELDS[kind])):
        return [f"{bad}: schedule needs kind {kind!r} and only the fields {list(TRAIN_SCHEDULE_FIELDS[kind])}"]
    return []


def validate_train_case(json_path: str) -> list[str]:
    """Checks a training component sidecar (docs/testing-strategy.md §3.7) against the schema and,
    except for schedules, its safetensors file: keys and their order, the component, the config's
    kind, the tensor names of the component, the tolerance of testing-strategy.md §2 and, for
    schedules, one value per step and no tensors. Returns problems."""
    sidecar, text = read_sidecar(json_path)
    if sidecar is None:
        return [text or f"{json_path}: cannot read"]
    if not isinstance(sidecar, dict):
        return [f"{json_path}: sidecar is not a JSON object"]
    component = sidecar.get("component")
    if component not in train_components():
        return [f"{json_path}: component {component!r} is not one of {list(train_components())}"]
    schedule = component.startswith("schedule.")
    want_keys = TRAIN_SCHEDULE_KEYS if schedule else TRAIN_SIDECAR_KEYS
    if tuple(sidecar.keys()) != want_keys:
        return [f"{json_path}: keys {list(sidecar.keys())} differ from the schema {list(want_keys)}"]
    problems: list[str] = []
    if sidecar["schema_version"] != TRAIN_SCHEMA_VERSION:
        problems.append(f"{json_path}: schema_version {sidecar['schema_version']!r} is not {TRAIN_SCHEMA_VERSION}")
    if text != canonical_json(sidecar):
        problems.append(f"{json_path}: not in canonical form (rerun the generator)")
    stem = os.path.splitext(os.path.basename(json_path))[0]
    if sidecar["case"] != stem:
        problems.append(f"{json_path}: case {sidecar['case']!r} does not match the file name")
    if component != os.path.basename(os.path.dirname(json_path)):
        problems.append(f"{json_path}: component {component!r} does not match the directory name")
    if not _is_uint(sidecar["seed"]) or sidecar["seed"] >= 1 << 64:
        problems.append(f"{json_path}: seed must be a u64")
    config = sidecar["config"]
    config_problems = _train_config_problems(json_path, component, config)
    problems += config_problems
    problems += tolerance_problems(json_path, sidecar["tolerance"])
    st_path = os.path.splitext(json_path)[0] + ".safetensors"
    if schedule:
        if sidecar["inputs"] != [] or sidecar["outputs"] != []:
            problems.append(f"{json_path}: a schedule case has no inputs and no outputs")
        if os.path.exists(st_path):
            problems.append(f"{st_path}: a schedule case has no safetensors file")
        if not config_problems:
            values = sidecar["values"]
            if not (isinstance(values, list) and len(values) == config["total_steps"]
                    and all(_is_number(v) for v in values)):
                problems.append(f"{json_path}: values must be total_steps ({config['total_steps']}) finite numbers")
            want_tol = (
                {"exact": False, "atol": TRAIN_COSINE_ATOL_PER_LR * config["lr"], "rtol": 0}
                if component == "schedule.cosine" else {"exact": True, "atol": 0, "rtol": 0}
            )
            if sidecar["tolerance"] != want_tol:
                problems.append(f"{json_path}: tolerance must be {want_tol} (testing-strategy.md §2)")
        return problems
    inputs, outputs = _named_entries(sidecar["inputs"]), _named_entries(sidecar["outputs"])
    if inputs is None or outputs is None:
        return problems + [f"{json_path}: inputs and outputs must be lists of {{\"name\", \"dtype\", \"shape\"}}"]
    if any(e["dtype"] != "f32" for e in inputs + outputs):
        problems.append(f"{json_path}: training component tensors are f32")
    in_names = [e["name"] for e in inputs]
    out_names = [e["name"] for e in outputs]
    shapes = {e["name"]: e["shape"] for e in inputs + outputs}
    if component == "gradient_clip":
        want_tol = TRAIN_CLIP_TOLERANCE
        n = len(inputs)
        want_in = [f"grad_{i}" for i in range(n)]
        want_out = ["total_norm"] + [f"clipped_{i}" for i in range(n)]
        if n == 0 or in_names != want_in or out_names != want_out:
            problems.append(f"{json_path}: inputs must be {want_in or ['grad_0', '...']} and outputs {want_out}")
        elif shapes["total_norm"] != [1] or any(shapes[f"clipped_{i}"] != shapes[f"grad_{i}"] for i in range(n)):
            problems.append(f"{json_path}: total_norm has shape [1] and clipped_i the shape of grad_i")
    else:
        want_tol = TRAIN_OPTIMIZER_TOLERANCE
        states = optimizer_state_names(component.split(".", 1)[1], config if isinstance(config, dict) else {})
        want_in = ["param"] + [f"grad_{s}" for s in range(1, TRAIN_OPTIMIZER_STEPS + 1)]
        want_out = [f"{name}_{s}" for s in range(1, TRAIN_OPTIMIZER_STEPS + 1) for name in ("param",) + states]
        if in_names != want_in or out_names != want_out:
            problems.append(f"{json_path}: inputs must be {want_in} and outputs {want_out} (training.md §5.2's state names)")
        elif any(shape != shapes["param"] for shape in shapes.values()):
            problems.append(f"{json_path}: every tensor has the param's shape")
    if sidecar["tolerance"] != want_tol:
        problems.append(f"{json_path}: tolerance must be {want_tol} (testing-strategy.md §2)")
    try:
        header = read_safetensors_header(st_path)
    except (OSError, FixtureError) as e:
        return problems + [str(e)]
    problems += entry_problems(json_path, st_path, sidecar, header, ("inputs", "outputs"))
    return problems + subnormal_problems(st_path, header)
