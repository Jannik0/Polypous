# Reference fixture generators

Python scripts that produce ground-truth fixtures from established implementations (PyTorch,
transformers, diffusers, gguf-py, librosa, llama.cpp). Layout, sidecar schema and tolerances:
[docs/testing-strategy.md §3](../../docs/testing-strategy.md#3-reference-fixtures).

Rules:
- Deterministic: fixed seeds, pinned library versions (`requirements.txt`), recorded in each sidecar.
- Output: `*.safetensors` + `*.json` into `fixtures/small/<area>/` (small) or a manifest entry (large).
- Python is a development dependency only. Nothing in Polypous runs Python.

Setup: [docs/dev-environment.md](../../docs/dev-environment.md#python-reference-environment)
(Python 3.14, `pip install -r tools/reference/requirements.txt`).

| File | Purpose |
|---|---|
| `gen_ops.py` | Op fixtures in `fixtures/small/ops/<op>/<case>.{safetensors,json}`, from PyTorch |
| `fetch_fixtures.py` | Downloads `fixtures/manifest.json` entries into `fixtures/cache/` (size limit, SHA-256) |
| `fixturelib.py` | Shared, standard library only: sidecar format, safetensors header check, `--check` comparison, size budget |
| `test_*.py` | Unit tests; the ones that need PyTorch skip without it |

## Regenerate and check

```bash
python tools/reference/gen_ops.py            # rewrites fixtures/small/ops/ (removes cases that no longer exist)
python tools/reference/gen_ops.py --check    # regenerates into a temp dir, fails on changed/missing/extra files
python -m unittest discover -s tools/reference -v
python tools/ci/check_fixtures.py            # size budget and sidecar schema, what CI runs
```

Commit the regenerated files together with the generator change. Never edit a fixture by hand.

## Add an op case

1. Add a `Case` to `build_cases()` in `gen_ops.py`, usually with a helper such as `add_case` or
   `matmul_case`, or write a new helper for a new op:
   - `op`: the IR op name (`core.add`), `name`: unique per op, ending in the dtype (`broadcast_row_f32`).
   - `seed`: a new fixed number, not shared with another case.
   - `inputs`: builds the named inputs in the op's input order from the seeded NumPy generator
     (`uniform`, `integers`, `from_list`). Pick values the reference defines: no NaN or
     out-of-range values where PyTorch's result is undefined, and say so in `notes`.
   - `reference`: the PyTorch computation, in the dtype under test. If its bytes depend on the CPU
     (reductions, matmul), compute in f64 and round once, as `ref_matmul` does, and check the
     native result against the tolerance.
   - `attributes`: the op's attributes, named as in the op catalog.
   - `tolerance`: the defaults from testing-strategy.md §2 (`elementwise_tol`, `matmul_tol`, `EXACT`).
     A looser tolerance needs a comment here and an explanation in the PR.
2. Run the generator, then `--check` twice, the unit tests and `tools/ci/check_fixtures.py`.
3. Keep fixtures small: each case under 1 MiB, `fixtures/small/` under 50 MiB. A few KB per case is enough.

## Large fixtures

Add an entry to `fixtures/manifest.json` (`name`, `url`, `sha256`, `size`, `license`, `path`;
testing-strategy.md §3.4), then `python tools/reference/fetch_fixtures.py NAME`.
