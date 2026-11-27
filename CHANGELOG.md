# Changelog

All notable changes to Polypous are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break the API).

## [Unreleased]

### Added
- `Plan::check_training_mode(&Bindings)` in `polypous-runtime` (#112, runtime.md §3.1): the
  training-mode checks (compute dtype `f32`, one use per running statistic, at least 2 values per
  `batch_norm` channel), callable on the bound shapes of a call without tensors, with the same
  order and `RuntimeError` values as a training-mode `Plan::run`, which now calls this function.
  The trainer calls it at training.md §2.10 step 6 instead of its own copy of the checks.
- Autodiff on `DynTensor` in `polypous-backend` (#75, ADR-0014): `DynTensor::require_grad` marks a
  float tensor as a param leaf, `backward` back-propagates from a one-element loss into an opaque
  `Gradients`, and `Gradients::get`/`take` return a leaf's gradient as a plain `DynTensor` (leaf
  dtype, shape and device). `detach`, `without_autodiff`, `is_tracked`, `is_grad_leaf` and
  `has_autodiff` complete the API; no Burn type is public. Misuse (backward of an untracked,
  non-float or many-element tensor, a second backward through the same graph, the gradient of a
  non-leaf) is a `TensorError`. Burn's autodiff is now always compiled in (the off-by-default `autodiff` feature is
  gone): the release `polyp` grows from 13.2 to 16.2 MB; inference tensors carry no autodiff
  context. Gradients of the `core` elementwise ops, `cast`, the shape ops, `gather` and
  `index_select` follow graph-ir.md §3.1.11 on every device, including the tie and kink rules
  where Burn differs (`maximum`/`minimum` ties, `clamp` at its bounds, `pow`'s special values,
  `sqrt`/`rsqrt` at `±0`). A gradient reference harness (`tests/grad_reference_*.rs`) checks the
  107 gradient fixtures of these ops, and the forward reference tests rerun every case with
  tracked inputs and require identical outputs (bit for bit, except NaN payloads, which aren't
  part of op semantics).
- Gradients of `core.sum`, `mean`, `max`, `min`, `matmul`, `softmax`, `log_softmax`, `cumsum`
  and `top_k` (`values`) in `polypous-backend` (#76), checked against their 41 gradient fixtures
  (`tests/grad_reference_reduce.rs`; 148 gradient cases in all). `max`/`min` split a tie's
  gradient evenly (`g / k`, as `torch.amax`/`amin`), where Burn sends it to one element.
  `top_k` still sorts on the host but takes its values with a `gather` on the device, so its
  `values` output is differentiable (it cut the graph before); its results are unchanged.
- Gradients of the `nn` ops, and `nn.batch_norm` and `nn.dropout` in training mode, in
  `polypous-backend` (#77), checked against their 105 gradient fixtures
  (`tests/grad_reference_nn.rs`; all 253 gradient cases in all). `relu`, `leaky_relu` and
  `softplus` follow PyTorch at their kinks (`0` at `relu(±0)`, `negative_slope` at
  `leaky_relu(±0)`, `1/2` at `β·x = 0`), and `sigmoid`/`silu` keep their gradients far below
  zero (they were `0` below `x ≈ -44`). New `DynTensor::batch_norm_training` normalizes with the
  batch statistics (the gradient flows through them) and returns the new running statistics
  without autodiff context; `DynTensor::dropout_training` applies a `bool` keep mask the caller
  gives (`out = x · s` with `s = 1 / (1 - p)` as PyTorch computes it). `batch_norm`'s running
  statistics never get a gradient, in either mode. The gradient harness now also checks that
  inputs without an expected gradient get none.
- The Polypous random generator in `polypous-ir` (`polypous_ir::rng`, graph-ir.md §3.1.12, #77):
  SplitMix64 as a counter-based hash (`mix`, `fnv1a`, `stream_key`, `draw`), `u24`, `u53`,
  `normal`, and `dropout_mask`, the keep mask of a dropout node for a shape (checked element
  count, fallible allocation). Tested against the spec's known values and the statistics of
  testing-strategy.md §3.8.
- Training mode and params given by the caller in `polypous-runtime` (#78, runtime.md §3.1–§3.3):
  `Model::run_with(inputs, state, params, mode, options)` and `Plan::run` (a plan without loaded
  weights, for the trainer's param store) take a `RunMode` (`Inference`, the default, or
  `Training { seed, step }`) and optionally `Params` (param name → `DynTensor`), checked on metadata
  only: missing or extra names, dtype, shape, device (`RuntimeError::Param`). A training-mode call
  requires the compute dtype `f32`, exactly one use per `batch_norm` running statistic and at least
  2 values per channel (`TrainingDtype`, `SharedRunningStat`, `BatchNormValues`), runs
  `nn.batch_norm` with batch statistics and `nn.dropout` with host masks keyed by seed, step and
  node path, and returns the new running statistics in `RunOutput::running_stats`. Tracked params
  record through every op. `Model::run` and every M1 call are unchanged.
- Param analysis in `polypous-ir` (#78, graph-ir.md §2.4, §2.4.1): every param's uses after
  expanding repeats and its fans (`Analysis::params`, `param_fans`, `trainable_params`), the new
  validation error `invalid_init` (a fan-based `init` on an unused param, in a role without fans, or
  with disagreeing fans), and the remaining `invalid_value` checks of `init` values (`low <= high`,
  `std >= 0`, values that round to a finite f32).
- The MNIST training test (#84, testing-strategy.md §3.8, the M2 exit model):
  - `tools/reference/convert_mnist.py` converts MNIST (CC BY-SA 3.0, Yann LeCun and Corinna
    Cortes) into two safetensors datasets, `image` F32 `[N, 1, 28, 28]` (byte / 255) and `label`
    I64 `[N]`, byte-for-byte deterministic. They are the `fixtures/manifest.json` entries
    `mnist-train` and `mnist-test` (220 MB together), hosted on the Polypous release
    `fixtures-mnist-v1` and fetched with `fetch_fixtures.py`.
  - `fixtures/small/mnist/` (written by `tools/reference/gen_mnist.py`): a CNN graph in the
    layer order of §3.8 (149,466 params) and its training config (AdamW, cross-entropy,
    accuracy, the test set for validation, 5 epochs, checkpoints), plus the reference graph and
    config, initial weights and the first 10 training losses from PyTorch in f64.
  - Tests, `#[ignore = "mnist"]` (they need the data; `cargo test --release -p polypous -p
    polypous-cli -- --ignored mnist`): `val.accuracy` ≥ 0.98 within 5 epochs through the
    facade and through `polyp train`, the reference losses within `rtol = 1e-3`, and a run
    cancelled mid-epoch and resumed bit-identical to an uninterrupted one, all on the CPU. The
    lavapipe CI job skips them (`--skip mnist`).
- Training through the facade and the CLI (#83, public-api.md §1.4):
  - `Engine::trainer(&graph, TrainingConfig, TrainerOptions, &cancel)` and
    `Engine::resume_trainer(path, ResumeOptions, &cancel)`, and the `Trainer` handle (`Clone +
    Send + Sync + 'static`) with `info`, `run(&cancel, on_event)`, `export(graph, weights)` and an
    idempotent `close`. A thin layer over `polypous-train`.
  - `TrainingConfig` (`from_json`, `from_file`, canonical `to_json`, public fields, serde),
    `TrainerOptions` and `ResumeOptions` (`Default`, `from_json`/`to_json`, unknown fields
    rejected, at most 64 datasets with unique names), `InMemoryDataset { name, fields:
    Vec<NamedTensor> }`, `TrainingEvent` (`#[non_exhaustive]`), `TrainingOutcome` and
    `TrainerInfo`, each with the JSON of training.md §4.1 and public-api.md §1.4; the device in
    `started` and `TrainerInfo` is the facade's `DeviceId`. The config's nested types
    (`DataConfig`, `LossConfig`, `OptimizerConfig`, …) and `Metrics`, `HistoryEntry`, `Best`,
    `StopReason` are in the new module `polypous::training`. They are plain data: the nested
    config types' range checks in `polypous-train` are crate-private now (`validate`,
    `validate_metrics`; `ConfigError` is no longer re-exported at that crate's root), and a
    config is checked by `TrainingConfig::from_json` or when the trainer is created.
  - The config's `device`: `"auto"` is the device `load_graph` picks by default (the primary
    GPU, else the CPU); an id must be listed and available, else `backend_unavailable` with
    `context.device`.
  - Errors map 1:1 to `ErrorKind` with every context field, the message
    `Could not create the trainer: …`, `Could not resume training: …`, `Training failed: …` or
    `Could not export the model: …`, and `context.operation` `engine_trainer`,
    `engine_resume_trainer`, `trainer_run` or `trainer_export`. A panic, also one in the
    `on_event` callback, is `internal` and fails only that trainer.
  - `polyp train <graph> --config <file> [--weights FILE] [--device ID] [--json]
    [--export-graph FILE --export-weights FILE]` and `polyp train --resume <dir> [--device ID]
    [--discard-newer] …`: one line per event (`EVENT key=value …`) or JSON Lines with `--json`;
    the first Ctrl-C (Ctrl-C or Ctrl-Break on Windows) cancels the run, which writes its
    checkpoint and export and exits with code 4; a second one ends the process with code 130.
    Relative paths in the config are resolved against the config file's directory; a `tensors`
    dataset is an error in the CLI. New dependency of `polypous-cli`: `ctrlc` 3.5 (MIT OR
    Apache-2.0; brings `nix` 0.32 on Unix).
- The trainer of `polypous-train` (#82, training.md §1.2, §2, §4, §5; public-api.md §1.4, for
  the facade in #83):
  - `TrainingConfig` with `from_json`, `from_file` (at most 1 MiB) and a canonical `to_json`
    (every field, defaults filled in; `from_json(to_json(c)) == c`). Unknown fields, a key given
    twice, missing and mistyped fields and every range of §2.1–§2.9 are `InvalidInput` with the
    full JSON pointer from the root (`/optimizer/betas/1`, `/data/train/columns/a~1b/from`);
    `precision` `f16`/`bf16` is `UnsupportedOperation`. Numbers are read as the nearest `f64`
    (`serde_json` with `float_roundtrip`, now a normal dependency).
  - `Trainer::new` makes the checks of §2.10 in their order (config, graph without state, the
    device through a resolver the caller gives, references, the used fields of the data, the
    bound shapes of every batch size with the target values and the runtime's training-mode
    checks, the initial weight file with a subset of the params, the run directory) before it
    uploads anything; the cancellation token is checked between the steps.
  - The loop of §1.2: one training-mode forward and one backward per step, running statistics
    written back right after the forward pass, params without a gradient skipped, clipping,
    `lr(k)`, the optimizer; the validation pass in inference mode; the epoch metrics.
  - `TrainingEvent` (`started`, `epoch_started`, `step`, `validation_started`,
    `epoch_finished`, `checkpoint_saved`, `checkpoint_removed`, `finished`) with the JSON of
    §4.1 (non-finite numbers as `"nan"`, `"inf"`, `"-inf"`), in the order of §4.2, `step`
    throttled to 100 ms on an injectable monotonic clock; cancellation after every step, before
    each validation batch and before each checkpoint, with a checkpoint on cancel; `run` again
    continues.
  - Checkpoints (`step-` + 12 digits) written atomically with `fsync` (the directory flushes on
    Unix only), `state.json` as §5.2, `optimizer.safetensors` as `<param>:<state>`,
    `keep_last`/`keep_best` (only epoch checkpoints have a value, judged from each checkpoint's
    `state.json`; NaN never best, a tie keeps the earlier one), deleting only
    `step-*` and `.tmp-step-*` entries.
  - `Trainer::resume` loads an untrusted checkpoint with every check of §5.4 before the data
    it protects is read (fuzz-tested with truncated, bit-flipped and edited files), re-reads
    and fingerprints the data, and refuses or (`discard_newer`) replaces newer checkpoints.
    Resumed and continued runs are bit-identical to uninterrupted ones on the CPU (params,
    optimizer state, history, best and the checkpoint files).
  - `Trainer::export` writes a graph and weights pair that `polypous-runtime` loads, with
    outputs bit-identical to the trainer's inference-mode outputs on the CPU.
  - The `Trainer` handle (`Clone + Send + Sync`) with `info`, `run`, `export` and an idempotent
    `close`; `run` and `export` are `InvalidInput` while running (also from the run's own
    callback), after a failed run and after `close`. `TrainError` carries the kinds and context
    of §6.
- Datasets, shuffling and batching in `polypous-train` (#81, training.md §2.3, §2.4, §3.5, §3.6):
  `read_datasets` reads a config's `train` and `validation` sets into host `Dataset`s (named
  `f32`/`i64`/`i32`/`bool` fields of shape `[n] + sample shape`, in dataset order) from
  `tensors` (in-memory `TensorData`, `f16`/`bf16` widened exactly), `safetensors` (through
  `polypous-formats`, only the used tensors read), `csv{columns}` (an own RFC 4180 parser: quotes,
  CRLF, BOM, finite `f32` and range-checked integers, errors naming the record and column) and
  `image_folder` (classes and files in byte order, PNG of every color type and bit depth with the
  `png` crate, palette expanded, alpha dropped, `v / (2^d − 1)`), then a seeded `split`. Every size
  is checked before memory is reserved, and reservations are fallible: `data.max_bytes` covers
  both sets (checked from metadata before decoding), with the file, record, class and image
  limits of §3.5. `Batcher` gives epoch `e`'s order (Fisher–Yates on the Polypous generator,
  `stream_key(seed, e, "shuffle:train")`) and batch `b` of it directly, with the smaller last
  batch kept, so a resume needs no saved data state; the `data_fingerprint` of §5.2 is computed
  too. The `data` config types (`DataConfig`, `DatasetConfig`) deserialize from §2.3/§2.4's JSON
  and validate their ranges. New dependency: `png` 0.18.1 (MIT OR Apache-2.0, already in the lock
  file).
- Losses and metrics in `polypous-train` (#80, training.md §2.6, §2.7, §3.3, §3.4): `Loss`
  computes `mse`, `cross_entropy{label_smoothing}` (class indices `i64` or `i32`) and
  `binary_cross_entropy{with_logits}` as the batch mean, a `[1]` `f32` tensor that is
  differentiable with respect to the prediction, in public `DynTensor` ops. It is stable at large
  logits (`[1000, 0, −1000]` gives 2000, logits of ±200 give 200 and 0), finite at probabilities of
  exactly 0 and 1, and its gradient is PyTorch's, including `binary_cross_entropy`'s own backward
  formula at the clamp. Shape and dtype mismatches are errors (no broadcasting). `LossMetric` and
  `Accuracy` are epoch accumulators: the `loss` metric is weighted by batch size and summed in f64,
  and `accuracy` counts exactly, multiclass (ties to the lowest index, the first NaN wins, as
  `torch.argmax`) or binary with a `threshold`. `Loss`/`Accuracy::check_shapes` and
  `check_target_values` check a target field before training (class index outside `[0, C)`, BCE
  target outside `[0, 1]` or NaN, binary target not 0 or 1), naming the sample and the value. The
  config types `LossConfig` and `MetricConfig` deserialize from §2.6/§2.7's JSON and validate
  their rules (`label_smoothing` range, metric names, `{"kind": "loss"}`). 17 loss fixtures
  (`fixtures/small/train/loss.*`) from `gen_train.py`, f64 references rounded once; on the CPU
  and lavapipe the worst error is 2.7e-4 of the tolerance.
- Optimizers, schedules and gradient clipping in `polypous-train` (#79, training.md §3.1, §3.2,
  §5.2): `Optimizer` runs `sgd{momentum, dampening, nesterov, weight_decay}`, `adam` and `adamw`
  as PyTorch 2.15.0's single-tensor algorithm in `DynTensor` ops (scalars in f64, rounded once to
  f32), with its per-param state (`momentum_buffer`, `exp_avg`, `exp_avg_sq`) exported and
  imported for checkpoints, with checks (unknown names, shapes, dtypes, partial state);
  `Schedule` gives `lr(k)` for `constant`, `step`, `cosine` and `linear_warmup`;
  `clip_gradients` clips by the total L2 norm. The config types `OptimizerConfig`,
  `ScheduleConfig` and `GradientClipConfig` deserialize from training.md §2.8's JSON with its
  defaults, and validate its ranges with a JSON pointer per error. 29 reference fixtures
  (`fixtures/small/train/`, testing-strategy.md §3.7) from `tools/reference/gen_train.py`: 13
  optimizer cases against PyTorch's own f32 results (worst error 4.8e-7, 0.14 of the tolerance,
  on the CPU and lavapipe), 5 clipping cases (non-finite totals included) and 11 schedule cases;
  `tools/ci/check_fixtures.py` checks their schema.
- Initial values in `polypous-train` (#78, graph-ir.md §2.4.1): `initial_values(seed, name, shape,
  init, fans)` computes every `init` kind on the host in f64, rounded once to f32. Reference
  fixtures from a Python implementation of the spec (`tools/reference/polyrng.py`,
  `gen_training.py`): 45 initial-value cases (`fixtures/small/init/`) and 3 training-mode graph cases
  against PyTorch (`fixtures/small/training/`: outputs, running statistics, dropout masks and
  gradients), with their schemas in testing-strategy.md §3.9 and §3.10 and checks in
  `tools/ci/check_fixtures.py`.
- End-to-end model reference tests (#46, M1 exit): `tools/reference/gen_models.py` writes an MLP
  (784 → 128 → 10, logits and `log_softmax`) and a small CNN (conv2d, relu, max_pool2d, conv2d,
  batch_norm with non-trivial running statistics, relu, adaptive_avg_pool2d, flatten, linear) as
  graph JSON with PyTorch-named weights, inputs and f64 reference outputs for batch 1 and 4 in f32,
  f16 and bf16 (`fixtures/small/models/`, layout and sidecar in `docs/testing-strategy.md` §3.5,
  end-to-end tolerances in §2). `crates/polypous/tests/models.rs` runs every case through the
  facade on every available device; `crates/polypous-cli/tests/models.rs` runs `polyp run` on the
  MLP. `tools/ci/check_fixtures.py` checks the model fixtures (weights against the graph's params,
  case sidecars against the graph and their tensors, no subnormals).
- `polypous` facade and `polyp` CLI for graphs (#45), specified in `docs/specs/public-api.md` §1.3.
  `polypous::Graph` loads (`from_json`, `from_file`), saves (`to_json`, canonical form) and
  validates graph documents; `ValidationReport` lists every diagnostic with its stable code, node id
  and JSON pointer, and an invalid graph is an `InvalidGraph` error with the first diagnostic's node
  and pointer in `context`. `Engine::load_graph(&graph, GraphLoadOptions, &cancel)` compiles a graph
  for one device (default: the primary GPU, else the CPU; a device that isn't available is
  `BackendUnavailable`) with a compute dtype (`f32`, `f16`, `bf16`) and loads its safetensors weights
  (one file or a sharded index). `Model` is an `Arc` handle: `info()` (device, compute dtype, inputs,
  outputs, state entries and params with dtypes and symbolic shapes), `run(inputs, state, &cancel)`
  with named `TensorData` (validated: byte length, `bool` bytes, no zero-size dims), `save()` to
  `model.graph.json` + `model.safetensors` (params in the compute dtype; unused params copied from
  the original weight file) and `close()`. `polypous::tensor_file::{read, write}` read and write
  named tensors as safetensors files, and `DeviceId::parse` reads the `cuda:0` /
  `vulkan:discrete_gpu:0` / `cpu` syntax. Runtime errors map to the facade's kinds with the node
  path, tensor, file and device in the error `context`, which gains the fields `node`, `pointer`,
  `tensor`, `file` and `device`. There is one `CancellationToken` for probing, loading and running.
- `polyp graph validate <file> [--json] [--compute-dtype D]` (exit code 3 for an invalid graph) and
  `polyp run <graph> [--weights FILE] --input NAME=FILE … --output FILE [--device ID]
  [--compute-dtype D] [--json]` (safetensors inputs and outputs; graphs with state are rejected
  until sessions exist). Errors print the message and context to stderr, or the error JSON with
  `--json`.
- `polypous-ir`: `Analysis::outputs`, the dtype and inferred shape of every graph output.
  `polypous-runtime`: `Model::param`, a loaded param tensor by name.
- `polypous-runtime`: the graph interpreter on one device (#44), specified in the new
  `docs/specs/runtime.md`. `Plan::new` validates a graph with a compute dtype (default `f32`),
  checks the compute dtype and every value dtype against the device before anything runs
  (`UnsupportedOperation` naming the node, op, dtype and device), orders each scope
  topologically and computes value lifetimes, so intermediates are dropped after their last use
  (per iteration inside `repeat` bodies). `Model::load` loads the params from a safetensors file or
  a sharded checkpoint (`WeightFile`): the file must hold exactly the declared params (missing,
  extra, non-float and wrong-shape tensors are errors naming the tensor, checked on the header
  before anything is allocated), `F16`/`BF16`/`F32` storage is cast once to the compute dtype,
  repeat prefixes are expanded and tied params loaded once. `Model::run` takes inputs and state by
  name, binds symbols through `Analysis::bind`, rejects values on another device, runs every node
  (all 78 M1 ops, `repeat` with carries, `params_prefix`, body state and nesting), returns the
  outputs and every state entry, and checks a `CancellationToken` before every node and
  iteration. `RuntimeError` carries the node path (`layers[3]/attn`) and maps to the facade's
  error kinds (`RuntimeError::kind`). Every op fixture runs through the interpreter as a one-node
  graph (CPU and Vulkan).
- `polypous-ir`: the M1 op catalog, dtype inference and symbolic shape inference
  (`docs/specs/graph-ir.md` §3, §4.1 checks 4, 6–8, #39). `catalog::OPS` has one entry per row of
  the §3.2/§3.3 tables (inputs with dtype class and rank, params, attributes with type, default and
  range, output ports with their dtype rule, shape rule, differentiable flag); a test parses the
  spec tables and another checks every op fixture's output dtypes and shapes. Validation now checks
  op inputs, params and attributes (types, ranges, axes, `float` only where allowed, dim attributes
  and their symbols, the `concat`/`split` and `constant` limits), infers every value's dtype
  (`float` resolved to the compute dtype, strict dtypes) and shape as normal forms (broadcasting,
  `reshape` with `-1`, conv/pool sizes, clamped `slice` bounds), infers each `repeat` body once and
  checks carries and state updates (`repeat_carry_mismatch`, `state_update_mismatch`).
  `Graph::analyze` returns what validation proved for the runtime (resolved dtypes, inferred shapes,
  normalized attributes, expanded state entries), and `Analysis::bind` binds the symbols for one
  call (§2.3.4 steps 1–5: input and state checks, `min`/`max`, every inferred dim and dim attribute
  evaluated with checked arithmetic), with a `BindError` that names the input, symbol or node.
- graph-ir.md: clarifications from the inference work: body params must have the same shape in
  every iteration; which code a range error gets (`invalid_attr`, `shape_mismatch`,
  `nonpositive_dim`, `unbound_dim`); a NaN `clamp` bound is allowed; `squeeze` `dim` is non-empty;
  `slice` bounds `≤ 1` aren't clamped; `pad`'s default value is the dtype's zero; variadic input
  errors; the binding order of expanded state.
- `polypous-backend`: the first IR ops on `DynTensor` (`ops` module, graph-ir.md §3.2, #40):
  elementwise `add sub mul div pow maximum minimum neg abs exp log sqrt rsqrt sin cos tanh clamp`,
  `where_` (`core.where`), comparisons `equal not_equal less less_equal greater greater_equal`, logic
  `logical_and logical_or logical_not`, and `cast`. Broadcasting across different ranks, strict
  dtypes (typed errors, no promotion), NaN propagation in `maximum`/`minimum`/`clamp`, NaN in
  comparisons and float → bool casts, IEEE special cases of `pow`, wrapping integer arithmetic, and
  f16/bf16 transcendentals computed in f32 on every backend. New `Scalar` argument type,
  `TensorError::InvalidArgument` and `DynTensor::device_supports`. `add` now broadcasts operands
  of different ranks.
- Op reference tests for these ops against PyTorch fixtures on the CPU and on every GPU device
  (`tests/reference_*.rs`, shared harness in `tests/reference/common.rs`), with fixture cases for
  every op in `fixtures/small/ops/` (`tools/reference/gen_ops.py`).
- `polypous-backend`: shape, indexing and creation ops on `DynTensor` (graph-ir.md §3.2, #41):
  `reshape` (moved into `ops`) `flatten permute transpose squeeze unsqueeze expand concat split
  slice pad`, `gather index_select`, and `constant arange zeros ones full full_like` (associated
  functions with a `DeviceId`). Plain, normalized arguments; every bad argument is a typed error
  (no panics, sizes checked before allocating, zero-size results refused, §4.3 list limits).
  Out-of-range `gather`/`index_select` indices are checked before the op runs and give the new
  `TensorError::IndexOutOfRange` (Burn 0.22.1 panics on the CPU and reads stray memory, or crashes,
  on Vulkan). `pad` and every op work for every dtype, including `bool`. Fill values follow
  §3.1.7 and round through f32 to f16/bf16 like PyTorch.
- Reference fixtures for these ops (190 cases: every op × every allowed dtype, rank 1 and 6,
  size-1 dims, negative axes and `slice` bounds, steps > 1, `i32` and `i64` indices, and inputs fed
  as permuted, transposed, expanded and stepped-slice views), `tests/reference_shape.rs`,
  `reference_indexing.rs` (with out-of-range checks on every device) and `reference_creation.rs`.
  The harness gains integer and list attribute getters, a runner that passes the device to
  creation ops, and `feed_layout` for non-contiguous inputs; existing tests are unchanged.
- `polypous-backend`: reductions, `matmul`, softmax, `cumsum` and `top_k` on `DynTensor`
  (graph-ir.md §3.2, #42): `sum mean max min` over any set of dims (`None`: all, `[1]` for a full
  reduction), `argmax argmin` with an `i64` or `i32` result, `cumsum`, `softmax log_softmax`,
  `top_k` (returns `(values, indices)`) and `matmul` (moved into `ops`; batch dims now broadcast
  across different ranks). NaN and ties follow §3.1.8 on every backend (explicit masks; `top_k`
  sorts stably on the host, one device sync per call), integer `sum`/`cumsum` wrap instead of
  panicking on the CPU, softmax subtracts the row maximum, and f16/bf16 `sum mean cumsum softmax
  log_softmax` accumulate in f32.
  Known limitation: f32 `matmul` on CUDA GPUs with TF32 (Ampere or newer) misses the spec's
  f32 accuracy, because Burn 0.22.1 uses TF32 tensor cores; it waits for an upstream fix
  (docs/status.md §4).
- Reference fixtures for these ops (184 new cases, 192 with the 8 earlier `matmul` cases): every
  op × every allowed dtype, one/several/all dims with and without `keepdim`, `matmul` from 2-D to
  rank 6 with broadcast batch dims of different ranks, `k = 1` and `k = 256`, ties and NaN for
  `argmax`/`argmin`/`top_k`, wrap-around for `sum`/`cumsum`, large values for softmax and f16 sums,
  and non-contiguous inputs; `tests/reference_reduce.rs`.
- `polypous-backend`: the M1 `nn` ops on `DynTensor` (graph-ir.md §3.3, #43): `linear` (on
  `matmul`), `embedding`, `conv1d` and `conv2d` (stride, padding, dilation, groups), `max_pool2d`,
  `avg_pool2d` (`count_include_pad`), `adaptive_avg_pool2d`, `dropout` (the identity in inference),
  `batch_norm` (inference), `layer_norm`, `group_norm`, `rms_norm`, and the activations `relu
  leaky_relu gelu gelu_tanh silu sigmoid softplus mish`. Params are plain `DynTensor` arguments in
  PyTorch's layouts (optional ones `Option`), so PyTorch weights load unchanged; every argument is
  checked (typed errors, output sizes with checked arithmetic). Out-of-range `embedding` ids are a
  `TensorError::IndexOutOfRange`, checked before the lookup. NaN propagates through activations and
  `max_pool2d`, `max_pool2d` pads with `-inf` on every backend (CubeCL pads with `-f32::MAX` and
  dropped NaN on lavapipe), `softplus` and `mish` stay accurate far below zero (CubeCL's
  `log1p` is `ln(1 + x)`), norms use a two-pass variance, and f16/bf16 compute in f32. Known
  limitation: like `matmul`, f32 `linear` on CUDA GPUs with TF32 misses the spec's accuracy (7 of
  8 f32 cases on the RTX 3060; docs/status.md §4); conv passes on CUDA.
- Reference fixtures for the `nn` ops (268 cases, the first with params): every op × f32/f16/bf16
  (`embedding` ids i64 and i32), stride/padding/dilation/groups incl. depthwise and non-square
  kernels, `count_include_pad`, overlapping adaptive cells, all-padding pool windows, NaN/±inf,
  large and tail inputs, large-mean norms, absent optional params and non-contiguous inputs;
  `tests/reference_nn.rs` (with out-of-range ids on every device). Sidecars of ops with params
  gain a `params` key after `inputs` (testing-strategy.md §3.2); existing sidecars are unchanged.
- graph-ir.md §3.1.8: f16/bf16 `sum`, `mean`, `cumsum`, `softmax` and `log_softmax` compute in f32;
  §3.2 `top_k` note: fixtures use a stable sort, since `torch.topk` leaves the order of ties open.
- graph-ir.md §3.1.8: subnormal float values are backend-defined (a device may flush them to
  zero). Fixtures contain none; `tools/ci/check_fixtures.py` rejects them.
- `polypous-ir`: graph document model, canonical JSON and structural validation
  (`docs/specs/graph-ir.md` §2, §4, §6, #38). Typed `Graph` for every M1 document field (dims,
  inputs, outputs, params with `init`, state, nodes, `repeat` bodies, `meta`); loading untrusted JSON
  under the §4.3 limits (64 MiB, nesting depth 128, 4,000,000 JSON values, 64 KiB strings, duplicate
  keys rejected) with
  every error collected as a stable code, message, node id and JSON pointer; canonical save (schema
  key order, `meta` and `"float"` kept as written, load → save → load identity); structural checks
  (versions, names and namespaces, references including `node:port` and `repeat:carry`, cycles, op
  names of the M1 opsets, dim declarations and binding, repeat limits computed without expanding,
  carry keys and state update targets); dim expressions with their normal form, canonical string
  and equality (§2.3).
- `polypous-formats`: safetensors reader and writer (`docs/specs/formats.md` §1, #37). Memory-mapped
  reader with a copy fallback that validates the whole header before exposing tensors (100 MB header
  limit, contiguous offsets covering the data buffer, checked sizes, no duplicate keys, rank ≤ 64),
  zero-copy tensor views, all safetensors dtypes recognized and F32/F16/BF16/I64/I32/BOOL decodable;
  sharded checkpoints (`*.safetensors.index.json`, 64 MiB limit, plain shard file names, every
  tensor in exactly one shard); a writer whose output is ordered by element size, then name, so every
  tensor is aligned to its element size, checked against Python `safetensors`.
- Project brief, architecture, roadmap, specifications, ADRs and contributor/agent guides.
- Cloud environment setup script (`tools/cloud/setup.sh`) and documentation for cloud agents.
- Cargo workspace with empty skeleton crates (`polypous-ir`, `-formats`, `-polyp`, `-backend`, `-quant`,
  `-runtime`, `-train`, `-pipelines`, `polypous`, and `polypous-cli` with the `polyp` binary), workspace
  lints (`unsafe_code` denied, `unwrap_used`/`expect_used` warned outside tests) and the Rust toolchain
  pinned to 1.99.1 in `rust-toolchain.toml`.
- CI on GitHub Actions: fmt, clippy and CPU tests on Linux and Windows, `cargo deny` (licenses and
  advisories, `deny.toml`), a dependency-boundary check (`tools/ci/check_dependency_boundary.py`),
  Vulkan tests on Mesa lavapipe, and a macOS build-only job for Metal (on `main`, manual dispatch, and
  PRs that touch the backend crates or the dependency set).
- CI job "PR title and description" (`tools/ci/check_pr_text.py`): fails a PR whose title or
  description contains a Claude Code session link, a `Claude-Session` trailer or a model name, or an
  agent-authored PR whose description does not end with the fixed Claude Code footer.
- Burn pinned to `=0.22.1` and CubeCL to `=0.11.3` as workspace dependencies, used only by
  `polypous-backend`. Backend features `cpu` (default, Burn's pure-Rust `flex` backend), `cuda`,
  `vulkan` and `metal`. `burn_smoke` example to check a backend on a given machine.
- ADR-0011 (accepted): use Burn's runtime `Dispatch` backend for backend selection; supersedes part of ADR-0003.
- ADR-0012: MPL-2.0 allowed for unmodified dependencies, per crate (`colored`, `buildid`, `option-ext`
  as `deny.toml` exceptions).
- ADR-0013: device probing runs in a separate prober process with a probe cache, crash and hang
  recovery, and an in-process fallback. Apps can re-run the probe through the API (`probe_devices`,
  `Engine::refresh_devices`) or pass in their own report.
- Device discovery and probing in `polypous-backend` (`device` module): `BackendKind`, `DeviceId`,
  `DeviceInfo` (memory excluding Windows' shared GPU memory, PCI/LUID identity to match one GPU
  across CUDA and Vulkan, capabilities, probe status; serde JSON with stable `snake_case` names),
  `enumerate_devices`, `describe_device`, `probe_device` (panics caught) and `select_primary_gpu`.
- `DynTensor` in `polypous-backend` (`tensor` module): tensors with runtime rank (1–6) and dtype
  (`f32`, `f16`, `bf16`, `i64`, `i32`, `bool`) over `burn::Tensor`, with `TensorData` conversion
  (little-endian, row-major), `shape`, `dtype`, `reshape`, `cast`, `add`, `matmul` and `to_device`.
  Bad input gives a `TensorError` instead of a panic.
- `DynTensor::from_data`, `to_device` and `device` are public and take or return a `DeviceId` from
  `list_devices()`. A device whose backend is not compiled in, or that doesn't exist, gives the new
  `TensorError::DeviceUnavailable` instead of a panic. Device names in `TensorError` messages are
  `DeviceId` texts such as `vulkan:discrete_gpu:0`.
- `polypous` facade: `Engine::new(EngineConfig::default())` discovers and probes devices (in process
  until the prober process lands), `engine.devices()` / `engine.primary_gpu()` return the facade's own
  serde device types (placement.md §1 fields plus `primary`), and `polypous::Error` with the stable
  `ErrorKind`; panics at the API boundary become `Internal` errors. Backend features `cpu`, `cuda`,
  `vulkan`, `metal` are forwarded through `polypous` and `polypous-cli`.
- `polyp devices` prints a table of devices (backend, name, kind, dedicated memory, unified memory,
  status) with the primary GPU marked; `polyp devices --json` prints the JSON documented in
  public-api.md §1.2. Logs go to stderr (`RUST_LOG=polypous=debug` shows probe details).
- `EnumerateOptions { software_rasterizers }`, `enumerate_devices_with`, `skipped_device` and
  `DeviceId::is_software_rasterizer` in `polypous-backend`, and the `EngineConfig` field
  `software_rasterizers` (default `false`) in `polypous`.
- Crash-safe device probing (ADR-0013, #14): `Engine::new` takes devices from a probe cache (no
  prober process when every device's driver fingerprint is unchanged), else probes them in a prober
  process (`polyp` or the app itself re-launched via `polypous::probe::run_prober_if_requested()`,
  or `EngineConfig::prober_path`), restarting it after a crash or hang so one bad driver only makes
  its own device unavailable (`crashed during probe` / `probe timed out`), else in process with a
  crash journal so a crashing driver takes the app down at most once. Linux driver fingerprints from
  sysfs, `/proc/driver/nvidia/version` and the Vulkan ICD manifests; Windows driver fingerprints and
  stable device keys from the registry (display adapters' PCI location, hardware id and driver
  version/date/provider, Windows build, Vulkan loader), so the cache also survives reboots there and
  an Intel iGPU, which Vulkan reports without a PCI address, is matched by its PCI ids (#28); macOS
  compares the enumeration instead.
- Probe API: `polypous::probe::{probe_devices, run_prober_if_requested, run_prober}`,
  `Engine::refresh_devices`, `Engine::probe_report`, `ProbeOptions`, `ProbeEvent`, `ProbeReport`,
  `Isolation`, `CancellationToken`, and `EngineConfig` fields `probe_report`, `prober_path`,
  `probe_cache_dir`, `probe_isolation`, `probe_timeout_per_device_ms`.
- `polyp probe --json` runs the prober and prints its JSON-lines protocol. `polyp devices` gets
  `--software-rasterizers`, `--isolation`, `--probe-cache-dir`, `--prober-path`,
  `--timeout-per-device-ms` and `--json --report`, and prints how devices were probed under the table.
- `DeviceInfo::physical_gpu` (JSON `physical_gpu`): devices with the same number are one physical
  GPU reached through different backends (the RTX 3060 through CUDA and Vulkan); `null` for the CPU
  and software rasterizers. `polyp devices` shows it in a new `GPU` column.
- The probe report says why a compiled-in GPU backend has no device, for example
  `cuda: compiled in, but no device found: CUDA_VISIBLE_DEVICES is set to "-1"` or
  `… the driver library nvcuda.dll is not installed`; `polyp devices` prints it as a warning.
- Reference fixture tooling (#36): `tools/reference/gen_ops.py` generates op fixtures from
  PyTorch (pinned CPU-only versions in `tools/reference/requirements.txt`, Python 3.14) into
  `fixtures/small/ops/<op>/<case>.{safetensors,json}`, with `--check` to compare against the
  committed files. First fixtures for `core.add`, `core.matmul`, `core.cast` (all 36 pairs of
  `f32`/`f16`/`bf16`/`i64`/`i32`/`bool`) and `core.reshape`. Sidecar schema and layout in
  testing-strategy.md §3. `tools/reference/fetch_fixtures.py` downloads `fixtures/manifest.json`
  entries into `fixtures/cache/` with a size limit and SHA-256 check. CI checks the fixture size
  budget and sidecars (`tools/ci/check_fixtures.py`) in the "Dependency boundary" job.

### Changed
- The trainer's §2.10 step 6 (#112) runs the runtime's training-mode checks after the target
  values, as training.md orders them, and in the runtime's order (a shared running statistic
  before a batch with one value per channel). Before, a last batch of one sample with
  `nn.batch_norm` was reported ahead of a bad target value. Configs with only one of these
  problems get the same error as before.
- Op fixtures name their tensors after the op's inputs (`a`, `b`, `x`, `cond`; outputs `out`),
  and `core.add/empty_dim_f32` is gone (zero-size dims are not allowed in M1, graph-ir.md §2.3).
- `polypous-backend` no longer depends on `polypous-ir`.
- `polyp devices` and `Engine::new` in apps that call `run_prober_if_requested()` report probe
  results as `isolated: true`. Apps without a prober probe in process as before, now with a crash
  journal and a warning in the report.
- Device discovery lists software rasterizers (Vulkan/Metal adapters of type `cpu`, such as
  lavapipe) but no longer opens or probes them unless `software_rasterizers` is set. They are shown
  as unavailable with the new reason code `software_rasterizer_skipped`. This keeps lavapipe on
  Debian 13 (Mesa 25.0.7) from crashing `polyp devices`, `Engine::new` and the GPU tests; the
  `VK_DRIVER_FILES` workaround is no longer needed for that. The GPU tests opt in with the test-only
  variable `POLYPOUS_TEST_SOFTWARE_VULKAN=1`, which the CI lavapipe job sets.
- Each backend's devices are enumerated at most once per process; `list_devices`,
  `list_backend_devices`, `enumerate_devices*`, `DynTensor` device checks and `Engine` discovery reuse
  the cached list, and unlisted ids fail against it. `describe_device` and `probe_device` reject ids
  that aren't listed without opening them. Devices added after the first enumeration are not seen
  until restart (placement.md §1, "When enumeration happens").
- On Windows the CPU device shows its marketing name from the registry (`ProcessorNameString`)
  instead of `PROCESSOR_IDENTIFIER` (new Windows-only dependency `windows-registry`, MIT OR Apache-2.0).

### Fixed
- No more SIGSEGV on Debian 13 when a Vulkan device is first used while another computes (#118,
  register M2-13): the first open of a Vulkan device creates a temporary Vulkan instance inside
  CubeCL, and Debian's Vulkan loader unloads drivers for it without a lock that another thread's
  Vulkan call takes (seen in debug builds). The first open of any Vulkan device in a process now
  opens every Vulkan device of the open set first, once, under the device lock: by default every
  listed Vulkan device but software rasterizers; with an `Engine`, only devices its probe found
  available and no engine's probe found unavailable. That first Vulkan call costs one device open
  per Vulkan device (about 0.1–0.4 s each on the reference machine in release builds). A device
  that fails this open is unavailable for the rest of the process and never opened again. Before,
  this protection came only from `Engine::new` with in-process probing; with the prober process or
  a probe-cache hit nothing was opened up front. Remaining: an opted-in software rasterizer opened
  later through the backend directly, and Vulkan instances created outside Polypous
  (placement.md §1, "Remaining window").
- Training and inference fit a much smaller thread stack in debug builds (#114): `polyp train`
  overflowed the 1 MB main thread on Windows. The interpreter runs each op family in its own
  function, and the largest dispatch functions of `polypous-backend` (`reshape`, `broadcast_to`,
  `linear`, the `f16`/`bf16` path of `matmul`) and the `repeat` loop are split too. With the
  `vulkan` feature on Linux, a training step needs 388 KiB instead of 992 KiB, inference 352 KiB
  instead of 964 KiB, and 8 nested `repeat`s at most 396 KiB; a facade test runs both on a
  640 KiB stack. Results, errors and the API are unchanged.
- A tracked `nn.max_pool2d` gives the untracked result bit for bit (#77): on lavapipe, a window
  holding `0` and `-0` gave `+0` when the input was tracked and `-0` when it wasn't.
- `core.max`/`core.min` of floats give a zero result the sign of IEEE 754-2019's order
  (`-0 < +0`) on every backend (#76): on lavapipe, `max` of a `0`/`-0` tie returned `-0` and
  `min` returned `0`, where PyTorch gives `0` and `-0`. Other results are unchanged.
- `core.matmul` in `f16`/`bf16` computes in `f32` and rounds once to the dtype, like `nn.linear`,
  conv and the reductions (#67; graph-ir.md §3.1.8). `core.matmul/k256_f16` missed `rtol 1e-2` on
  an Intel Iris Xe with Mesa ANV. `f32` `matmul` is unchanged.
- GPU tests (#65): `polypous`'s backend features now also enable `polypous-runtime`'s, so
  `cargo test --workspace --features polypous-cli/cuda,polypous-cli/vulkan -- --include-ignored`
  runs every crate's GPU tests, and the CI lavapipe job runs those of `polypous-backend`,
  `polypous-runtime`, `polypous` and `polypous-cli` (before: only the backend's).
- When a device could not be opened because a worker thread panicked (CubeCL's device runner, for
  example `Unable to acquire wgpu device: RequestDevice { … "Initialization of an object has
  failed" }`), the `probe failed` reason only said `… RecvError`. The prober process now records
  panics on every thread and appends the ones from other threads to the reason (`… (worker thread
  panicked: …)`, bounded). In-process probing is unchanged: it installs no panic hook.
- With a warm probe cache, removing (or installing, or updating) the CUDA driver library was
  ignored: `cuda:0` was still reported available and chosen as the primary GPU. The library
  (`libcuda.so.1` on Linux, `nvcuda.dll` on Windows: path, size and modification time, or
  `missing`) is now part of every driver fingerprint, so a change re-probes. Existing probe caches
  are re-probed once after the update.
- With a warm probe cache, setting `CUDA_VISIBLE_DEVICES` (or `CUDA_DEVICE_ORDER`) was ignored: a
  hidden CUDA device was still reported available and chosen as the primary GPU. Both variables are
  now part of the driver fingerprint, so changing them re-probes.
- Listing devices, or using a `DynTensor` with an id that doesn't exist, while another thread
  computed on a Vulkan GPU could crash the process with SIGSEGV in the Vulkan loader (Debian 13 with
  stock Mesa, #26). The backend GPU tests no longer need `--test-threads=1` there. The first open of
  each Vulkan device still enumerates once inside CubeCL; open devices before computing on other
  threads (`Engine::new` does).
