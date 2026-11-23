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
| `gen_grads.py` | Gradient fixtures in `fixtures/small/grads/<op>/<case>.{safetensors,json}` (op sidecar schema 2), from PyTorch autograd |
| `gen_models.py` | End-to-end model fixtures in `fixtures/small/models/<model>/` (graph, weights, cases), from PyTorch modules |
| `gen_training.py` | Initial-value fixtures in `fixtures/small/init/<kind>/<case>` (from `polyrng.py`) and training-mode graph fixtures in `fixtures/small/training/<model>/` (model case schema 2: outputs, running statistics, dropout masks, gradients of the trainable params), from PyTorch in f64 (testing-strategy.md §3.9, §3.10) |
| `polyrng.py` | Standard library only: a Python implementation of the Polypous generator (graph-ir.md §3.1.12: streams, `u24`, `U`, `N`, dropout masks) and of the initial values of every `init` kind (§2.4.1), written from the spec; tested against the spec's known values and testing-strategy.md §3.8's statistics (`test_polyrng.py`) |
| `fetch_fixtures.py` | Downloads `fixtures/manifest.json` entries into `fixtures/cache/` (size limit, SHA-256) |
| `fixturelib.py` | Shared, standard library only: sidecar format, safetensors header check, `--check` comparison, size budget |
| `test_*.py` | Unit tests; the ones that need PyTorch skip without it |

## Regenerate and check

```bash
python tools/reference/gen_ops.py            # rewrites fixtures/small/ops/ (removes cases that no longer exist)
python tools/reference/gen_ops.py --check    # regenerates into a temp dir, fails on changed/missing/extra files
python tools/reference/gen_grads.py          # rewrites fixtures/small/grads/
python tools/reference/gen_grads.py --check
python tools/reference/gen_models.py         # rewrites fixtures/small/models/
python tools/reference/gen_models.py --check
python tools/reference/gen_models.py --report  # the measured errors behind the model tolerances
python tools/reference/gen_training.py       # rewrites fixtures/small/init/ and fixtures/small/training/
python tools/reference/gen_training.py --check
python tools/reference/gen_training.py --report  # PyTorch f32's errors against the training references
python -m unittest discover -s tools/reference -v
python tools/ci/check_fixtures.py            # size budget and sidecar schema, what CI runs
```

Commit the regenerated files together with the generator change. Never edit a fixture by hand.

## Add an op case

1. Add a `Case` to `build_cases()` in `gen_ops.py`, usually through a helper:
   - Elementwise, comparison and logic ops: an `OpSpec` (op name, input names, PyTorch call) and
     `ew_case(spec, name, seed, dtype, description, shapes=..., values=...)`. The family helpers
     (`binary_arith_cases`, `unary_float_cases`, `minmax_cases`, `compare_cases`, …) build the
     standard case lists.
   - `matmul_case`, `reshape_case`, `cast_case` for those ops, or a new helper for a new op.
   - Reductions, `cumsum`, softmax and `top_k`: `reduce_case`, `arg_case`, `cumsum_case`,
     `softmax_case`, `topk_case` (`reduction_cases()`). They take `values=` or `make=` (`ties`,
     `positive`, `below_one`) and honor the layout prefixes below. A `matmul_case` name with a
     layout prefix feeds `a` as that view.
   - Shape, indexing and creation ops: `exact_case`, or `x_case` for one input `x`. A case name
     starting with `permuted_`, `transposed_`, `expanded_` or `strided_` builds its first input as
     that PyTorch view (`layout_input`), and the Rust harness feeds the same kind of view on the
     device (`feed_layout` in `crates/polypous-backend/tests/reference/common.rs`).
   - `nn` ops: `act_case`, `linear_case`, `embedding_case`, `conv_case`, `pool_case`, `norm_case`
     (`nn_cases()`), all built on `nn_case`. **Params** (graph-ir.md §2.4) are built after the
     inputs and listed in `Case.params` (the present ones, in catalog order; `()` if an op's
     optional params are all left out, `None` for ops without params). `write_case` puts them in
     the sidecar's `params` key (testing-strategy.md §3.2). Name a case without an optional param
     after it (`no_bias_f32`).
   - `op`: the IR op name (`core.add`), `name`: unique per op, ending in the dtype (`broadcast_row_f32`).
   - `seed`: a new fixed number, not shared with another case.
   - `inputs`: builds the inputs in the op's input order, **named like the op's inputs** in
     graph-ir.md §3 (`a`, `b`, `x`, `cond`, `index`, `xs0`… for `concat`; the output is `out`,
     `out0`… for `split`, `values` and `indices` for `top_k`), from the seeded NumPy generator
     (`uniform`, `integers`, `from_list`). Pick values the reference defines: no NaN or
     out-of-range values where PyTorch's result is undefined, and say so in `notes`.
   - `reference`: the PyTorch computation, in the dtype under test. If its bytes depend on the CPU
     (reductions, matmul, transcendentals), compute in f64 and round once (`ref_matmul`,
     `OpSpec(f64=True)`), and set `native` so the generator checks PyTorch's own result in the
     dtype against the tolerance.
   - `attributes`: the op's attributes, named as in the op catalog and written as a graph writes
     them (negative axes, `-1`, `"nan"`/`"inf"` strings; defaults may be left out).
   - `tolerance`: the defaults from testing-strategy.md §2 (`elementwise_tol`, `matmul_tol`,
     `reduction_tol`, `EXACT`).
     A looser tolerance needs a comment here and an explanation in the PR.
2. Run the generator, then `--check` twice, the unit tests and `tools/ci/check_fixtures.py`.
3. Add the op to a Rust reference test (`crates/polypous-backend/tests/reference_*.rs`), which
   loads every case of the op directory.
4. Keep fixtures small: each case under 1 MiB, `fixtures/small/` under 50 MiB. A few KB per case is enough.

## Add a gradient case

Layout, sidecar (schema version 2) and tolerances: [testing-strategy.md §3.6 and §2](../../docs/testing-strategy.md#36-gradient-fixtures-op-sidecar-schema-version-2);
conventions: [graph-ir.md §3.1.11](../../docs/specs/graph-ir.md#3111-gradients).

1. Add a `GradCase` to the op's family function in `gen_grads.py` (`elementwise_cases`, `shape_cases`,
   `reduction_cases`, `layer_cases`, `pool_cases`, `norm_cases`, `activation_cases`), usually through
   `case`, `unary`, `x_case` or a family helper:
   - `tensors`: the op's inputs in order, then its present params in catalog order (`params=`), then
     `mask` for `nn.dropout` in training mode, each from a builder (`rnd`, `signed`, `vals`, `choice`,
     `ints`, `bools`, `view`, `copy_of`, `keep_mask`). All floats are f32. Names as in graph-ir.md §3.
   - `fn`: the forward call on a dict of the tensors, returning one tensor (port `out`) or a dict by
     port; `call`: its PyTorch text. The generator derives the upstream gradients (one per float port,
     `upstream_values` unless `upstream=` gives them), the differentiable tensors (every float input,
     every param but `batch_norm`'s running statistics) and the `reference` string.
   - `seed`: a new fixed number, not shared with another case (the generator refuses duplicates).
   - Tie, kink and special-value cases use exactly representable values and set `spec=` to graph-ir.md
     §3.1.11's formula (`spec_maximum`, `spec_clamp`, …); the generator stops if PyTorch disagrees.
   - No NaN or ±inf in inputs or upstream gradients (unspecified there; the generator refuses them).
   - `mode="training"` only for `nn.batch_norm` and `nn.dropout`.
   - Tolerances: `op_tolerance(op)` for outputs and `GRADIENT` for gradients (§2); a stricter one
     needs a note, a looser one an explanation in the PR.
2. Run the generator, then `--check`, the unit tests and `tools/ci/check_fixtures.py`.

## Add a model or a model case

Layout, sidecar and tolerances: [testing-strategy.md §3.5 and §2](../../docs/testing-strategy.md#35-model-fixtures-schema-version-1).

1. A model is a `ModelSpec` in `MODELS` of `gen_models.py`: an `nn.Sequential` of named layers
   (`layers`), the input's name and shape without the batch dim, and fixed seeds for the weights
   and for calibration data (BatchNorm statistics). `layer_node` exports each layer as one graph
   node (node id = layer name, params named like the `state_dict` keys); a layer type it doesn't
   know is an error, so add its export there first (the op must exist in graph-ir.md §3).
   `extra_outputs` adds outputs computed from the last layer (the MLP's `log_softmax`).
2. A case is a `CaseSpec` in `CASES` (model, `batch<N>_<dtype>`, batch size, compute dtype, a new
   seed). The tolerance comes from the dtype (`TOLERANCES`, testing-strategy.md §2); the generator
   refuses to write a case where PyTorch's own result in the dtype or the runtime emulation misses it.
   Check the margins with `--report`.
3. Run the generator, `--check`, the unit tests and `tools/ci/check_fixtures.py`, then
   `cargo test -p polypous --test models`, which loads every case of every model directory listed
   in `MODELS` there.

## Large fixtures

Add an entry to `fixtures/manifest.json` (`name`, `url`, `sha256`, `size`, `license`, `path`;
testing-strategy.md §3.4), then `python tools/reference/fetch_fixtures.py NAME`.
