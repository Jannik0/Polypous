# Spec: Training

**Status:** Binding for M2: §1–§7. The kinds marked "later" in §3 and the items of §8 are not part of
M2. **Crate:** `polypous-train` (with `polypous-backend`, `polypous-runtime` and the facade, §1.1)
**Decision:** [ADR-0014](../adr/0014-training-on-dyntensor-with-own-optimizers.md)

Training lets apps such as a visual builder train a [Graph IR](graph-ir.md) model from a
declarative configuration, with progress events, cancellation, checkpoints and resume. The scope is
small-to-medium models on a single device, in `f32`. Fine-tuning large pretrained models is out of
scope for v1. The public API is [public-api.md §1.4](public-api.md#14-training-m2).

Owner decisions for M2 (#71) referred to below: 2 (a Polypous host generator for randomness),
5 (the catalog), 6 (`f32` only), 7 (running statistics stay params), 8 (a param without a value
is an error).

## 1. How it works

### 1.1 Who owns what

| Crate | In training |
|---|---|
| `polypous-backend` | `DynTensor` ops, their gradients, and the small autodiff API of ADR-0014 (mark a param as a leaf, back-propagate a one-element loss, read a leaf's gradient as a plain `DynTensor`, drop the autodiff context). Burn's `autodiff` feature lives here and nowhere else ([ADR-0007](../adr/0007-burn-version-pinning.md)) |
| `polypous-runtime` | Runs the graph: one call per batch, in training or inference mode, on params the caller gives ([runtime.md §3.1–§3.3](runtime.md#31-mode)). It keeps no state between calls |
| `polypous-train` | Everything else, with no Burn dependency: the config and its validation (§2), the param store, initial values ([graph-ir.md §2.4.1](graph-ir.md#241-initial-values)), datasets, shuffling and batching (§3.5, §3.6), losses and metrics (§3.3, §3.4), gradient clipping, the optimizers and schedules (§3.1, §3.2), the loop (§1.2), events (§4) and checkpoints (§5) |
| `polypous` (facade) | `Engine::trainer`, `Engine::resume_trainer`, `Trainer`, the plain-data types and the error mapping (public-api.md §1.4); the CLI `polyp train` |

- **The param store** holds every declared param of the graph as an `f32` `DynTensor` on the
  training device, keyed by its param name ([graph-ir.md §2.4](graph-ir.md#24-parameters)). It is
  owned by the trainer. There is no `DynamicModel`, no Burn `Module`, `Param` or `GradientsParams`
  (ADR-0014).
- **Trainable params** are the params with `trainable: true` (the default) that no `nn.batch_norm`
  node uses as `running_mean` or `running_var` (graph-ir.md §2.4, §3.3). Only they get gradients,
  optimizer state and updates.
- **Gradients** are read from the autodiff API once per step, one per trainable param, as plain
  `f32` `DynTensor`s without autodiff context. A trainable param that the backward pass doesn't
  reach (unused, or only behind ops that pass no gradient) has no gradient. It is skipped as
  PyTorch skips a param whose `.grad` is `None`: no weight decay, no update, no optimizer state,
  and it doesn't count for gradient clipping. A graph has no control flow, so in M2 the set of params
  with a gradient is the same at every step.
- **Optimizer state** is `f32` `DynTensor`s on the training device, per trainable param (§3.1).
  Because the set above never changes, a param's optimizer step count always equals the trainer's
  global step counter, and only that counter is stored (§5.2).
- **Randomness** comes only from the Polypous generator of
  [graph-ir.md §3.1.12](graph-ir.md#3112-random-values), on the host (owner decision 2). Its streams
  are keyed by the training `seed`, a step and a label; §1.3 lists every stream the trainer uses.
  The generator keeps no state, so nothing about it is saved besides the seed and the counters.
- **Training never streams** weights from disk and never splits a model across devices: the graph,
  the param store, the optimizer state and one batch live on one device. Datasets live in host
  memory (§3.5).

### 1.2 One step, one epoch

The trainer keeps a **global step counter** `k`: the number of optimizer steps completed since the
start of training, `0` at the start, incremented by one after each optimizer step, across epochs
and across resumes. With `S` = steps per epoch (§3.6) and `E` = `epochs`, training runs `T = E · S`
optimizer steps.

Optimizer step `k` (epoch `e = k div S`, batch `b = k mod S`):

1. **Batch.** Take batch `b` of epoch `e`'s order (§3.6), assemble its fields on the host and upload
   them to the training device.
2. **Forward.** Mark every trainable param as an autodiff leaf. Run the graph through the runtime
   once, in mode `training { seed, step: k }` (`seed` from the config), with the param store as the
   given params (runtime.md §3.2) and the batch fields as the graph inputs (`inputs`, §2.5).
   This is the only training-mode call of step `k`: one forward per optimizer step, and no gradient
   accumulation in M2. So every training-mode call of a training run has a distinct `step`, and
   the dropout masks of different steps are independent streams.
3. **Running statistics.** Write the running statistics that the call returned (runtime.md §3.3)
   into the param store, replacing the old tensors, right away: before the loss, the backward pass
   and the optimizer step. This is when PyTorch updates them (during the forward pass). They carry
   no autodiff tracking and are never trained (graph-ir.md §3.3).
4. **Loss.** Compute the loss (§3.3) from the graph output and the target field. It is a
   one-element `f32` tensor that carries the autodiff tracking of the forward pass. Read its value
   back as the step's loss (for events and metrics, §3.4) and update the training metrics with this
   batch's predictions.
5. **Backward.** Back-propagate the loss, read each trainable param's gradient, and drop the
   autodiff context. Burn keeps no graph after a backward (ADR-0014), so there is exactly one backward
   per forward.
6. **Clip** the gradients if `gradient_clip` is set (§3.1.4).
7. **Update.** Compute the learning rate `lr(k)` (§3.2) and apply the optimizer (§3.1) to every
   trainable param that has a gradient.
8. `k ← k + 1`. Emit a `step` event (§4, throttled), check the cancellation token, and write a
   checkpoint if one is due (§5.1).

After the last step of an epoch: if there is a validation set, run the **validation pass**, then
finish the epoch's metrics, emit `epoch_finished` and write a checkpoint if one is due.

**Validation pass.** One graph call per validation batch (`batch_size` samples, the last one
possibly smaller, in the validation set's order, which is never shuffled), in **inference mode**
(runtime.md §3.1), with the param store as the given params and no param marked as a leaf. It draws
no random values, changes no param and computes the `val.*` metrics (§3.4).

The training metrics (`train.*`) of an epoch come from the training-mode forward passes of its
steps, with the params as they were at each step (as in most PyTorch training loops). They are not
recomputed after the epoch.

### 1.3 Random streams

Every random value of a training run comes from one of these streams of the generator of
graph-ir.md §3.1.12 (`stream_key(seed, step, label)`; `seed` is the config's `seed`):

| Use | `step` | `label` | Spec |
|---|---|---|---|
| Initial values of a param | `0` | `"init:"` + the param name | graph-ir.md §2.4.1 |
| `nn.dropout` masks of step `k` | `k` (the global step counter) | `"dropout:"` + the node path | graph-ir.md §3.3, §1.2 step 2 |
| Order of the training samples in epoch `e` | `e` | `"shuffle:train"` | §3.6 |
| Which samples a `split` dataset takes | `0` | `"split:validation"` | §3.5.5 |

- The four label prefixes end in `:`, which appears in no param name or node path, so no two uses
  share a stream. (`"split:"` is followed by the role the split dataset fills; in M2 only
  `validation` can be a split.)
- Validation passes draw nothing. Nothing else in training is random.

### 1.4 Determinism

- **On the CPU** (`cpu` device), with the same Polypous build, machine, graph, config, initial
  weights and data, two training runs give bit-identical params, optimizer state and metrics. A run
  that is stopped and resumed from a checkpoint gives the same results as one that isn't (§5.4).
- **On GPUs** the random values (init, dropout masks, data order) are the same as on the CPU, because
  they come from the host. The arithmetic isn't: GPU kernels may sum in a different order, and may
  differ from run to run (atomics, tiling, TF32 on CUDA, [status.md §4](../status.md#4-known-limitations-and-open-items)).
  Runs on a GPU are reproducible only within that variation.

## 2. Configuration (JSON)

A training config is a JSON document. `TrainingConfig` in the facade is its Rust form, with
`from_json`/`to_json` (public-api.md §2 rule 3); taking a `TrainingConfig` and taking its JSON are
equivalent everywhere.

```json
{
  "format": "polypous.training",
  "format_version": 1,
  "seed": 42,
  "device": "auto",
  "precision": "f32",
  "epochs": 5,
  "batch_size": 64,

  "data": {
    "train":      { "kind": "image_folder", "path": "data/train", "image_size": [28, 28], "channels": 1 },
    "validation": { "kind": "split", "from": "train", "fraction": 0.1 },
    "shuffle": true
  },

  "inputs":  { "image": "image" },
  "loss":    { "kind": "cross_entropy", "prediction": "logits", "target": "label", "label_smoothing": 0.0 },
  "metrics": [ { "kind": "accuracy", "prediction": "logits", "target": "label" } ],

  "optimizer": { "kind": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01 },
  "schedule":  { "kind": "cosine", "warmup_steps": 100, "min_lr": 1e-5 },
  "gradient_clip": { "max_norm": 1.0 },

  "checkpoint": { "dir": "runs/mnist", "every_epochs": 1, "keep_last": 3, "keep_best": "val.accuracy" }
}
```

### 2.1 General rules

- **Strict objects.** Every object of the config (the root and every nested one) rejects unknown
  fields and fields of the wrong JSON type. A field that the selected `kind` doesn't have is an
  unknown field (`momentum` on `adam`). A misspelled option therefore never goes unnoticed.
- **Required and optional.** Fields marked *required* below must be present. Every other field may
  be left out and then has its default; `null` is accepted only where the type says "or null".
  `to_json` writes every field, with defaults filled in, in the order of the tables.
- **Numbers.** An integer field takes a JSON integer (`5`, not `5.0`) in its range. A float field
  takes any JSON number, read as the nearest `f64`; JSON can't express NaN or infinities. A float
  that is used in `f32` arithmetic (§3) must also round to a finite `f32`.
- **Strings.** UTF-8. Names (fields, metrics, graph inputs and outputs) are 1–256 bytes; paths are
  1–4,096 bytes. Paths may be absolute or relative; relative paths are resolved against the
  trainer's base directory (`TrainerOptions::base_dir`, public-api.md §1.4; the CLI uses the config
  file's directory).
- **Limits.** The document is at most 1 MiB (checked before parsing; `TrainingConfig::from_file`
  doesn't read past it) and its JSON nesting depth at most 32. `metrics` has at most 16 entries, the
  `inputs` object at most 4,096, a CSV `columns` object at most 1,024 fields and 16,384 columns in
  all.
- **Errors.** A config that is not valid JSON, isn't an object, breaks a limit, or has an unknown,
  missing, mistyped or out-of-range field is `InvalidInput`, with `context.pointer` the JSON
  pointer (RFC 6901) of the offending value (`/optimizer/betas/1`; `""` for the whole document) and
  a message naming the field and the allowed values. The one exception is `precision` (§2.2).
  These checks need no graph: `TrainingConfig::from_json` makes them. The checks against the graph
  and the data are §2.10.

### 2.2 Root

| Field | Type, default | Range and meaning |
|---|---|---|
| `format` | string, *required* | `"polypous.training"` |
| `format_version` | integer, *required* | `1`. Any other value is `InvalidInput` (a newer config needs a newer Polypous) |
| `seed` | integer, `0` | `0` to `2^64 − 1` (`u64`). Keys every random stream (§1.3) |
| `device` | string, `"auto"` | `"auto"`, or a device id in the syntax of `DeviceId::parse` (`"cpu"`, `"cuda:0"`, `"vulkan:discrete_gpu:0"`, public-api.md §1.3). See below |
| `precision` | string, `"f32"` | `"f32"` only in M2. `"f16"` and `"bf16"` are `UnsupportedOperation` (owner decision 6; mixed precision is "later", §8); any other string is `InvalidInput` |
| `epochs` | integer, *required* | 1–10,000 |
| `batch_size` | integer, *required* | 1–65,536 |
| `data` | object, *required* | §2.3 |
| `inputs` | object, *required* | §2.5 |
| `loss` | object, *required* | §2.6 |
| `metrics` | array, `[]` | §2.7 |
| `optimizer` | object, *required* | §2.8, §3.1 |
| `schedule` | object, `{"kind": "constant"}` | §2.8, §3.2 |
| `gradient_clip` | object or null, null | `{"max_norm": float}` (*required* field, finite and `> 0`): §3.1.4. Null: no clipping |
| `checkpoint` | object or null, null | §2.9. Null: no checkpoints |

**`device`.**
- `"auto"` is the engine's primary GPU if it has one, else the CPU (the default of
  `GraphLoadOptions::device`, public-api.md §1.3). There is no memory estimate in M2 (placement is
  M4): `auto` doesn't check that the model fits.
- A device id must be in `engine.devices()` with status `available`; otherwise
  `BackendUnavailable` with `context.device`, as for `load_graph`.
- **Out of memory.** If the device runs out of memory while the trainer is created or while it
  runs, the call fails with `OutOfMemory` (where the backend reports an allocation failure as
  such; other backend failures are `Backend`). There is no automatic fallback to another device in
  M2. The app may create the trainer again with `"device": "cpu"`, or resume the last checkpoint on
  the CPU (`ResumeOptions::device`, public-api.md §1.4).

### 2.3 `data`

| Field | Type, default | Range and meaning |
|---|---|---|
| `train` | dataset, *required* | The training set (§3.5). Any kind except `split` |
| `validation` | dataset or null, null | The validation set: any kind. Null: no validation pass, no `val.*` metrics |
| `shuffle` | bool, `true` | Shuffle the training samples every epoch (§3.6). The validation set is never shuffled |
| `max_bytes` | integer, `4294967296` (4 GiB) | 1 to `2^40`. The most host memory the decoded datasets may take together (§3.5) |

### 2.4 Datasets

Every dataset object has a `kind`. The kinds, their fields and how they read data are §3.5:

| `kind` | Fields |
|---|---|
| `tensors` | `name` (string, *required*) |
| `safetensors` | `path` (path, *required*) |
| `csv` | `path` (path, *required*), `columns` (object, *required*) |
| `image_folder` | `path` (path, *required*), `image_size` (`[H, W]`, *required*, each 1–4,096), `channels` (`1` or `3`, `1`), `image_field` (name, `"image"`), `label_field` (name, `"label"`, not equal to `image_field`) |
| `split` | `from` (string, `"train"`; only `"train"`), `fraction` (float, *required*, `0 < fraction < 1`) |

### 2.5 `inputs`

An object mapping **every graph input** to a field of the datasets: `{"<graph input>": "<field>"}`.
A graph input that isn't mapped, or a key that isn't a graph input, is an error (§2.10). The same
field may feed several inputs.

### 2.6 `loss`

| Field | Type, default | Meaning |
|---|---|---|
| `kind` | string, *required* | `mse`, `cross_entropy` or `binary_cross_entropy` (§3.3) |
| `prediction` | name, *required* | A graph output |
| `target` | name, *required* | A dataset field |
| `label_smoothing` | float, `0.0` | `cross_entropy` only: `0 ≤ label_smoothing ≤ 1` |
| `with_logits` | bool, `false` | `binary_cross_entropy` only: the prediction is a logit, not a probability |

There is exactly one loss in M2 (several weighted losses are "later").

### 2.7 `metrics`

An array of metric objects. The loss is always reported as the metric `loss` (§3.4) and is not
listed here; listing `{"kind": "loss"}` is `InvalidInput`.

| Field | Type, default | Meaning |
|---|---|---|
| `kind` | string, *required* | `accuracy` in M2 (§3.4) |
| `prediction` | name, *required* | A graph output |
| `target` | name, *required* | A dataset field |
| `name` | string, the `kind` | The metric's name in events and checkpoints: `[a-z0-9_]`, 1–64 bytes, not `loss`, unique in the list |
| `threshold` | float or null, null | Binary accuracy only (§3.4); null means `0.5`. Given with the multiclass form, it is `InvalidInput` |

**Reported names.** Each epoch reports `train.loss`, then `train.<name>` for each metric in list
order, and, with a validation set, `val.loss` and `val.<name>` in the same order.

### 2.8 `optimizer`, `schedule`

| Object | `kind` | Fields (type, default, range) |
|---|---|---|
| `optimizer` | `sgd` | `lr` (float, `1e-3`, `≥ 0`), `momentum` (float, `0`, `0 ≤ m ≤ 1`), `dampening` (float, `0`, `0 ≤ d ≤ 1`), `nesterov` (bool, `false`), `weight_decay` (float, `0`, `≥ 0`). `nesterov` needs `momentum > 0` and `dampening = 0` |
| `optimizer` | `adam` | `lr` (float, `1e-3`, `≥ 0`), `betas` (`[float, float]`, `[0.9, 0.999]`, each `0 ≤ β < 1`), `eps` (float, `1e-8`, `≥ 0`), `weight_decay` (float, `0`, `≥ 0`) |
| `optimizer` | `adamw` | as `adam`, but `weight_decay` defaults to `0.01` |
| `schedule` | `constant` | — |
| `schedule` | `step` | `every` (integer, *required*, 1 to `2^32`), `gamma` (float, `0.1`, `0 ≤ γ ≤ 1`) |
| `schedule` | `cosine` | `warmup_steps` (integer, `0`, 0 to `2^32`), `min_lr` (float, `0`, `0 ≤ min_lr ≤ optimizer.lr`) |
| `schedule` | `linear_warmup` | `warmup_steps` (integer, *required*, 1 to `2^32`) |

- The defaults are PyTorch's (`torch.optim.SGD`, `Adam`, `AdamW` 2.15.0), and so are the ranges,
  except where noted: PyTorch accepts any `momentum ≥ 0`, any `dampening` and any `gamma`; these
  ranges keep the formulas convergent and are what models use.
- `min_lr > lr` is `InvalidInput` at `/schedule/min_lr`, because the cosine would then rise.

### 2.9 `checkpoint`

| Field | Type, default | Meaning |
|---|---|---|
| `dir` | path, *required* | The run directory (§5.1) |
| `every_epochs` | integer or null, `1` | 1–10,000: a checkpoint after every `every_epochs`-th epoch. Null: none by epoch |
| `every_steps` | integer or null, null | 1 to `2^32`: a checkpoint after every `every_steps`-th optimizer step. Null: none by step |
| `keep_last` | integer or null, `3` | 1–10,000: how many of the newest checkpoints to keep. Null: keep all |
| `keep_best` | string or null, null | A reported metric name (§2.7: `val.accuracy`, `train.loss`, …): also keep the checkpoint with its best value (§5.3) |
| `on_cancel` | bool, `true` | Write a checkpoint when the run is cancelled (§4.3) |

Whatever these say, a checkpoint is also written after the last step of training (§5.1).

### 2.10 Validation against the graph and the data

`Engine::trainer` (public-api.md §1.4) checks, in this order, and stops at the first failure.
No param is uploaded and nothing runs on the device before steps 1–8 have passed.

1. **Config** (§2.1–§2.9): `InvalidInput` with `pointer`; `precision` → `UnsupportedOperation`.
2. **Graph.** The graph must have no state entries (graph-ir.md §2.5): training a graph with state
   is `UnsupportedOperation` in M2. (The graph is `f32`: `Graph` validates with compute dtype
   `f32`, and the trainer always uses it.)
3. **Device** (§2.2): `BackendUnavailable` with `device`.
4. **References.** Every key of `inputs` must be a graph input and every graph input must be a key;
   `loss.prediction` and every metric's `prediction` must be graph outputs; `keep_best` must be a
   reported name (§2.7), and a `val.*` name needs a validation set. Each failure is `InvalidInput`
   with `pointer` and `context.tensor` naming the input or output.
5. **Data.** Read the datasets (§3.5): file errors are `Io` (can't read) or `InvalidInput`
   (malformed, over a limit) with `context.file` and, in `details`, the row, column or tensor. Then,
   each `InvalidInput` with `pointer` into the config and `context.tensor` naming the field:
   - every field named by `inputs`, `loss.target` and the metrics' `target` exists in the training
     set and, with a validation set, in it too, with the same dtype and sample shape;
   - each field's dtype fits its use: a field fed to a graph input has the input's dtype (`float`
     inputs take `f32` fields); targets follow §3.3 and §3.4.
6. **Shapes.** Bind the graph (graph-ir.md §2.3.4) with the shapes of a full batch (`batch_size`
   samples) and, if a set's last batch is smaller, with that size too. A binding error is
   `InvalidInput` naming the input. Then check each loss and metric's prediction dtype and shape,
   as binding gives them, against its target (§3.3, §3.4): `InvalidInput` naming the output and the field.
   Then the target values (§3.3, §3.4: class indices below `C`, binary targets 0 or 1, …), on every
   sample of both sets: `InvalidInput` naming the field, the sample and the value.
   Then the runtime's training-mode checks (runtime.md §3.1) on the same shapes: a running
   statistic with more than one use is `UnsupportedOperation`, a `batch_norm` with one value per
   channel is `InvalidInput` naming the node. So a last batch of one sample fails here, not in the
   middle of an epoch.
7. **Params.** If `TrainerOptions::weights` names a weight file, read it (runtime.md §2: header
   first, storage dtype, shape; `F16`/`BF16` widened exactly to `f32`). Unlike `load_graph`, the
   file may hold a subset of the params; a tensor that no param declares is `InvalidInput`
   (`context.tensor`, `context.file`), as in runtime.md §2. Then **every param that the file
   doesn't hold must have an `init`** (graph-ir.md §2.4; owner decision 8). Otherwise the error is
   `InvalidInput` with the message
   `These params have no init and no weights: <names>` (the first 10, then `and N more`),
   `context.tensor` the first of them, `context.details` every one of them, one per line, in
   declaration order. This includes params with `trainable: false` and running statistics: every
   param needs a value.
8. **Checkpoint directory** (§5.1): a new run's `checkpoint.dir` must not contain checkpoints
   (`InvalidInput` with `context.file`); it is created if it doesn't exist (`Io` if that fails).
9. **Upload.** Compute the initial values (graph-ir.md §2.4.1) on the host and upload every param:
   `OutOfMemory` or `Backend`.

The token given to `Engine::trainer` is checked between these steps and between files while reading
datasets; a cancelled creation is `Cancelled` and keeps nothing.

## 3. Catalog

M2 implements exactly the kinds of the "M2" column (owner decision 5). The "Later" column is not
part of M2; a config that uses a later kind or field is `InvalidInput` (unknown `kind` or field).

| Category | M2 | Later |
|---|---|---|
| Optimizers | `sgd{momentum, dampening, nesterov, weight_decay}`, `adam`, `adamw` | `rmsprop`; AMSGrad, `maximize`, per-param-group options |
| Schedules | `constant`, `step{every, gamma}`, `cosine{warmup_steps, min_lr}`, `linear_warmup` | `exponential`; schedules stepped per epoch |
| Gradient clipping | `gradient_clip{max_norm}` (L2 norm) | other norm types, clipping by value |
| Losses | `mse`, `cross_entropy{label_smoothing}`, `binary_cross_entropy{with_logits}` | `l1`, `huber`; `cross_entropy` with probability targets, more than 2 dims, class weights or `ignore_index`; `pos_weight`; several weighted losses |
| Metrics | `loss`, `accuracy` | `top_k_accuracy`, `precision`, `recall`, `f1`, `mae`, `rmse` |
| Datasets | `tensors`, `safetensors`, `csv{columns}`, `image_folder` (PNG), `split` | other image formats (JPEG, …), resizing, nested class folders, a `test` role, datasets larger than memory |
| Augmentation (images) | — | `random_flip`, `random_crop`, `normalize{mean,std}` |
| Loop | §1.2 | early stopping, gradient accumulation, mixed precision (§8) |

### 3.1 Optimizers

Each optimizer follows PyTorch 2.15.0's **single-tensor** algorithm (`torch.optim.SGD`, `Adam`,
`AdamW` with `foreach=False`, `fused=None`, `maximize=False`, `differentiable=False`,
`capturable=False`, `amsgrad=False`) step by step, per trainable param that has a gradient (§1.1),
in the order below (ADR-0014). It is written in `DynTensor` ops on tensors without autodiff
context, in `f32`.

**Scalars.** Every scalar (`lr(k)`, bias corrections, step sizes, `1 - β`, decay factors) is
computed in `f64` from the config values exactly as written, then rounded once to `f32` (round to
nearest, ties to even) where it meets a tensor. PyTorch computes them as Python floats (`f64`) and
its kernels round them to `f32` the same way. `x^y` is `pow` in `f64` (Rust `f64::powf`, C `pow`),
also where `y` is an integer or `0.5`, because that is what PyTorch's `**` computes.

**Order of operations.** PyTorch's CPU kernels fuse some of these operations (`add(alpha=…)`,
`addcmul`, `addcdiv`, `lerp` use FMA), which separate `DynTensor` ops can't reproduce, so results
differ from PyTorch's by a few roundings; testing-strategy.md §2 states the tolerance. `lerp(a, b, w)`
below is ATen's formula: `a + w · (b − a)` if `|w| < 0.5`, else `b − (b − a) · (1 − w)`.

`p` is the param, `g` its gradient (after clipping), `t = k + 1` the param's step count after this
step, and `η = lr(k)` (§3.2).

#### 3.1.1 `sgd` (`torch.optim.SGD(lr, momentum, dampening, weight_decay, nesterov)`)

```text
if weight_decay ≠ 0:  g ← g + weight_decay · p
if momentum ≠ 0:
    if the param has no buffer:  buf ← g              (a copy; this is the first step)
    else:                        buf ← buf · momentum + (1 − dampening) · g
    g ← nesterov ? g + momentum · buf : buf
p ← p + f32(−η) · g
```

State: `momentum_buffer` (only with `momentum ≠ 0`, created at the param's first step).

#### 3.1.2 `adam` (`torch.optim.Adam(lr, betas, eps, weight_decay)`)

```text
if weight_decay ≠ 0:  g ← g + weight_decay · p                  (L2, added to the gradient)
m ← lerp(m, g, 1 − β1)
v ← v · β2 + (1 − β2) · (g · g)
bias_correction1 = 1 − β1^t                    (f64)
bias_correction2 = 1 − β2^t                    (f64)
step_size        = η / bias_correction1        (f64)
denom = sqrt(v) / f32(bias_correction2^0.5) + f32(eps)
p ← p + f32(−step_size) · (m / denom)
```

State: `exp_avg` (`m`) and `exp_avg_sq` (`v`), both zero before the param's first step.

#### 3.1.3 `adamw` (`torch.optim.AdamW(lr, betas, eps, weight_decay)`)

As `adam`, except that weight decay is decoupled and comes first, instead of the L2 term:

```text
if weight_decay ≠ 0:  p ← p · f32(1 − η · weight_decay)        (the product and difference in f64)
```

then the `m`, `v` and `p` updates of `adam` with the undecayed gradient. (In PyTorch 2.15.0,
`AdamW` is `Adam` with `decoupled_weight_decay=True`.)

#### 3.1.4 Gradient clipping (`torch.nn.utils.clip_grad_norm_(params, max_norm)`, norm type 2)

Over the gradients of every trainable param that has one, after the backward pass and before the
optimizer:

```text
n_i   = sqrt(Σ g_i²)                    the L2 norm of each gradient, f32
total = sqrt(Σ n_i²)                    f32
c     = min(f32(max_norm) / (total + f32(1e-6)), 1)          f32
g_i  ← g_i · c                          for every gradient, also when c = 1
```

This is PyTorch's formula (`error_if_nonfinite=False`), so a non-finite `total` gives `c = 0` or
NaN, and the gradients follow, as in PyTorch. The norms are reductions; their summation order is
not specified, and the tolerance is testing-strategy.md §2's. PyTorch 2.15.0 computes the quotient
as `reciprocal(total + 1e-6) · max_norm` (`Tensor.__rdiv__`), so its `c` is rounded twice; the
single division above differs from it by at most about one ulp of `c`, well within that tolerance.

### 3.2 Schedules

A schedule gives the learning rate `lr(k)` of optimizer step `k` (§1.2), in `f64`, from the
optimizer's `lr` (`lr₀` below) and `T` (total steps, §1.2). Schedules step **per optimizer step**,
never per epoch. They have no state beyond `k` and `T`, so a checkpoint saves none (§5.2).

Each equals PyTorch's scheduler stepped once after every `optimizer.step()`, so that step `k` uses
the scheduler's value after `k` scheduler steps:

| Kind | `lr(k)` (f64, evaluated left to right as written) | PyTorch reference |
|---|---|---|
| `constant` | `lr₀` | no scheduler |
| `step{every, gamma}` | `lr₀`, multiplied by `gamma` `⌊k / every⌋` times, one rounding per multiplication | `StepLR(step_size=every, gamma)`, exactly (its chainable form multiplies the current `lr` by `gamma` at each multiple of `every`; `lr₀ · gamma^n` with one `pow` rounds differently) |
| `linear_warmup{warmup_steps = W}` | `lr₀ · ((k + 1) / W)` for `k < W`, then `lr₀` | `LambdaLR(lambda k: min(1, (k + 1) / W))`, exactly. Equals `LinearLR(start_factor=1/W, end_factor=1, total_iters=W−1)` up to `f64` rounding |
| `cosine{warmup_steps = W, min_lr}` | `k < W`: as `linear_warmup`. `k ≥ W`: `min_lr + (lr₀ − min_lr) · (1 + cos(π · (k − W) / (T − W))) / 2` | `CosineAnnealingLR(T_max=T−W, eta_min=min_lr)`'s closed form (`_get_closed_form_lr` at `last_epoch = k − W`), exactly; its chainable form and `SequentialLR([LinearLR(…), CosineAnnealingLR(…)], milestones=[W])` agree within `f64` rounding |

- `π` is the `f64` nearest to π (`std::f64::consts::PI`, Python's `math.pi`); `cos` is the
  platform's (as for graph-ir.md §3.1.12, it isn't required to round correctly).
- `linear_warmup` and the warmup of `cosine` start at `lr₀ / W` (step 0) and reach `lr₀` at step
  `W − 1`, so no step has learning rate 0.
- `cosine` with `W = 0` has no warmup. With `W ≥ T` every step is a warmup step (`T − W` is then
  never used). The last step, `k = T − 1`, is just above `min_lr`; `min_lr` itself would be step `T`.
- `step` with `gamma = 0` sets the learning rate to 0 from step `every` on.

### 3.3 Losses

The **prediction** is a graph output of dtype `f32`; the **target** a dataset field. The loss is a
one-element `f32` tensor (shape `[1]`), the **mean over the batch** (PyTorch's
`reduction="mean"`), and is differentiable with respect to the prediction. Shapes include the batch
dim `B`, as binding gives them (§2.10 step 6). There is no broadcasting between prediction and
target: PyTorch's `mse_loss` broadcasts (with a warning), here a shape mismatch is an error.

| Kind | Prediction | Target | Loss | PyTorch reference |
|---|---|---|---|---|
| `mse` | `f32`, any shape `[B, …]` | `f32`, the same shape | `mean((x − y)²)` over every element | `F.mse_loss(x, y)` |
| `cross_entropy{label_smoothing = ε}` | logits `f32` `[B, C]`, `C ≥ 1` | class indices `i64` or `i32`, `[B]`, each in `0 ≤ y < C` | `mean_n[(1 − ε) · (lse_n − x_{n,y_n}) + (ε / C) · Σ_c (lse_n − x_{n,c})]` with `lse_n = log Σ_c exp(x_{n,c})` | `F.cross_entropy(x, y.long(), label_smoothing=ε)` |
| `binary_cross_entropy` | probabilities `f32`, any shape `[B, …]` | `f32`, the same shape, each in `[0, 1]` | `mean(−(y · max(ln x, −100) + (1 − y) · max(ln(1 − x), −100)))` | `F.binary_cross_entropy(x, y)` |
| `binary_cross_entropy{with_logits}` | logits `f32`, any shape `[B, …]` | `f32`, the same shape, each in `[0, 1]` | `mean((1 − y) · x − log_sigmoid(x))`, `log_sigmoid(x) = min(x, 0) − log1p(exp(−|x|))` | `F.binary_cross_entropy_with_logits(x, y)` |

- **Numerics.** `lse` is computed stably (subtract the row maximum), so `cross_entropy` of logits
  `[1000, 0, −1000]` is finite (`2000` for class 2, as in PyTorch). The `with_logits` form is the
  stable one: logits of ±200 give losses `200` and `0`, not inf. The `−100` clamp is PyTorch's, so
  a probability of exactly 0 or 1 gives a finite loss. The formulas are written as the reference
  computes them; an implementation may use any formulation that meets the tolerance
  (testing-strategy.md §2).
- **Gradients** are PyTorch autograd's for the reference call. For `binary_cross_entropy` that is
  `(x − y) / max(x · (1 − x), 1e-12)` divided by the element count, so finite where the loss is
  clamped.
- **Targets are checked when the trainer is created** (§2.10 step 6), on every sample: a class
  index outside `[0, C)` (PyTorch: `IndexError`), or a `binary_cross_entropy` target outside
  `[0, 1]` or NaN (PyTorch: `RuntimeError`), is `InvalidInput` naming the field, the sample and the
  value. PyTorch's `cross_entropy` takes only `int64` targets; `i32` targets are read as `i64` here.
- **Predictions are not checked.** A `binary_cross_entropy` probability outside `[0, 1]` or a NaN
  prediction gives a NaN loss for that batch. (PyTorch raises for a probability outside `[0, 1]`.)
  Checking would read every prediction back to the host each step.
- **A non-finite loss doesn't stop training**, as in PyTorch: it goes into the metrics and events
  (§4.1 writes non-finite numbers as strings), and the app may cancel. Early stopping is "later".

### 3.4 Metrics

Metrics are computed per epoch, separately for the training steps (`train.*`) and the validation
pass (`val.*`). Values are `f64`.

- **`loss`**, always reported: the batch losses weighted by the batch's sample count `B_b`,
  `Σ_b loss_b · B_b / Σ_b B_b`, summed in `f64` in batch order from the `f32` loss values. For
  `cross_entropy` this is the mean loss per sample even with a smaller last batch.
- **`accuracy`**: correct predictions divided by predictions, counted exactly (integers) over the
  epoch, then divided once in `f64`. Two forms, chosen by the target's dtype:
  - **multiclass** (target `i64`/`i32` `[B]`, prediction `f32` `[B, C]`): sample `n` is correct if
    `argmax_c x_{n,c} = y_n`. Ties go to the **lowest** class index, and NaN counts as larger than
    every number (the first NaN wins), which is `torch.argmax` (2.15.0). Target values are checked
    as for `cross_entropy`. `threshold` must be null.
  - **binary** (target `f32`, the prediction's shape, every value exactly `0` or `1`, checked when
    the trainer is created): each element counts, and is correct if `(x > threshold) = (y = 1)`.
    `threshold` defaults to `0.5`, for probabilities; with logits use `0`. The comparison is in
    `f32`, with the threshold rounded once to `f32` (`x > f32(threshold)`), as PyTorch's
    `x > threshold` on an `f32` tensor. A NaN prediction is never above the threshold.
- **No samples:** a metric over zero samples (an epoch or validation pass without samples) has no
  value, not the NaN of `0 / 0`, so it can't be mistaken for a non-finite loss. A batch of zero
  samples adds nothing.
- Metrics don't take part in the backward pass and never change the params.
- Unlike the loss, `accuracy` is exact: there is no tolerance (testing-strategy.md §2).

### 3.5 Datasets

A dataset is a list of **samples**; each sample has the same named **fields**, each a tensor of a
fixed dtype and **sample shape**. A batch stacks `B` samples into tensors of shape
`[B] + sample shape` (a field with sample shape `[]` gives `[B]`). Datasets are read completely
into host memory when the trainer is created, in their **dataset order** (below), and are
untrusted input: every size is checked before the memory it needs is reserved, and reservations
are fallible.

- **Field dtypes** are `f32`, `i64`, `i32` or `bool`. `f16`/`bf16` data is widened to `f32`
  exactly when read.
- **Memory.** The decoded fields of both sets together may take at most `data.max_bytes`;
  otherwise `InvalidInput` naming the dataset, before the data is decoded where the size is known
  from metadata (safetensors, image count and size). Failing to reserve memory below that limit is
  `OutOfMemory`.
- **Sample count.** Each dataset has 1 to `2^32` samples; all fields of a dataset have the same
  count.
- A dataset's fields that no input, loss or metric uses are ignored (and, where the format allows,
  not read).

#### 3.5.1 `tensors`

In-memory data from the app: `TrainerOptions::datasets` (public-api.md §1.4) holds named datasets,
each a list of `NamedTensor` fields; `name` selects one. Dim 0 of every field is the sample dim
(rank 1–7: a sample has rank 0–6), and all fields must agree on it. Each `TensorData` is checked
(public-api.md §2 rule 4). A `name` that `TrainerOptions` doesn't have is `InvalidInput`.

#### 3.5.2 `safetensors`

A safetensors file read through `polypous-formats` ([formats.md §1](formats.md)): the whole header
is validated first, and every tensor's sizes are checked against the file before any data is read.
Each tensor is a field named by its tensor name; dim 0 is the sample dim (rank 1–7). Storage dtypes:
`F32`, `F16`, `BF16` (read as `f32`), `I64`, `I32`, `BOOL`. A used tensor of another dtype
(`U8`, `F64`, …) is `InvalidInput` naming the tensor. Sharded checkpoints are not datasets.

#### 3.5.3 `csv{path, columns}`

A UTF-8 text file in RFC 4180 form: records separated by LF or CRLF, fields by `,`, fields may be
enclosed in `"` with `""` for a quote inside (and may then contain `,` and line breaks); no other
escapes, no comments, no whitespace trimming. The first record is the **header** of column names,
which must be unique and non-empty. Every record has as many fields as the header. A UTF-8 BOM at
the start is skipped; an empty last line is allowed. One record is one sample, in file order.

`columns` maps each field name to `{"from": <column> | [<column>, …], "dtype": "f32" | "i64" | "i32"}`:

- `"from": "label"` (one column name) gives sample shape `[]`; `"from": ["a", "b", "c"]` gives
  sample shape `[3]` in that order (1–16,384 columns). Unknown column names are `InvalidInput`.
  Columns that no field uses are ignored.
- `f32` values are decimal numbers in Rust's `f32::from_str` syntax, rounded to nearest (`1`,
  `-0.5`, `1e-3`), but must be finite: `inf`, `NaN` and values that round to infinity are errors.
  `i64`/`i32` values are decimal integers with an optional sign, in the dtype's range.
- An empty field, a malformed or out-of-range value, a record with the wrong field count, an
  unterminated quote or invalid UTF-8 is `InvalidInput` with `context.file` and, in the message,
  the record number (the header is record 1) and the column name.
- **Limits:** the file is at most 4 GiB (checked before reading), a record at most 1 MiB, at most
  16,384 columns, and the decoded data counts against `data.max_bytes`.

#### 3.5.4 `image_folder{path, image_size, channels, image_field, label_field}`

```text
path/
  cat/  001.png  002.png …
  dog/  a.png    b.png   …
```

- **Classes** are the sub-directories directly inside `path`, **sorted by the bytes of their
  names** (byte order of the UTF-8 name; a name that isn't valid UTF-8 is `InvalidInput`). The label
  of class `i` in that order is `i` (`i64`). So the labels don't depend on the OS's listing order.
  Byte order is case-sensitive: `Zebra` sorts before `cat`.
- **Images** are the regular files directly inside a class folder whose name ends in `.png`
  (ASCII case-insensitive), sorted by the bytes of their names. Other files (`.DS_Store`,
  `Thumbs.db`, `.jpg`) and nested directories are ignored; the number of ignored files is logged at
  `warn`. Entries directly in `path` that aren't directories are ignored too. Symbolic links are
  followed (there is no recursion, so they can't loop); a link that doesn't lead to a regular file
  or directory is ignored like the other entries. A regular file whose name ends in `.png` but
  isn't valid UTF-8 is `InvalidInput`, as for class folders.
- **Dataset order:** class 0's images in name order, then class 1's, and so on.
- A class folder with no PNG image is `InvalidInput` naming it (a folder of JPEGs fails here, with
  a clear message). There must be at least one class.
- **Decoding** (PNG only in M2; the decoder dependency is chosen in #81, MIT-compatible): every
  color type and bit depth of the PNG standard. A palette is expanded, alpha is dropped (not
  composited), and the image then has 1 channel (gray) or 3 (RGB). With `channels: 1`, an RGB image
  is `InvalidInput` (no color conversion in M2); with `channels: 3`, a gray image is replicated to
  three equal channels. Interlacing and ancillary chunks don't change the pixels; color-management
  chunks (`gAMA`, `iCCP`, …) are ignored.
- **Tensor:** field `image_field`, `f32`, sample shape `[channels, H, W]` (row-major, channel
  first), value `v / (2^d − 1)` for a sample `v` of bit depth `d` (`v / 255` for 8-bit), computed as
  one `f32` division of the exact integer values. This is `torchvision.transforms.ToTensor` for
  8-bit images. Values are in `[0, 1]`; there is no normalization in M2 (do it in the graph, or with
  the later `normalize`). Field `label_field`, `i64`, sample shape `[]`.
- **Size:** every image must be exactly `image_size = [H, W]`; another size is `InvalidInput`
  naming the file and both sizes, read from the header before decoding. Images are not resized.
- **Limits:** at most 10,000 classes and 1,000,000 images (counted while listing, an error as soon
  as one is exceeded), each file at most 64 MiB, and `count · channels · H · W · 4` bytes within
  `data.max_bytes`, checked before decoding. A file that isn't a valid PNG, or is truncated, is
  `InvalidInput` with `context.file`.

#### 3.5.5 `split{from, fraction}`

Only as `validation`, and `from` is `"train"`: it moves a seeded random subset of the training set
into the validation set. With `n` training samples, the validation set gets
`n_val = ⌊n · fraction⌋` samples (the product in `f64`), which must be `1 ≤ n_val ≤ n − 1`, else
`InvalidInput`. Let `π` be the permutation of `0 … n−1` given by the shuffle of §3.6 with the key
`stream_key(seed, 0, "split:validation")`. The validation set is the samples `π(0) … π(n_val − 1)`,
the training set the rest; **both keep the training set's dataset order** (sorted by original
index). The split is the same in every epoch and after a resume.

### 3.6 Shuffling and batching

**Permutation.** With a key `key`, the permutation of `0 … n−1` is a Fisher–Yates shuffle driven
by `draw(key, c)` (graph-ir.md §3.1.12), with a draw counter `c` starting at 0:

```text
order = [0, 1, …, n − 1]
c = 0
for i = n − 1 down to 1:
    m = i + 1
    t = (2^64 − m) mod m            (= 2^64 mod m, on u64: m.wrapping_neg() % m)
    repeat:  z = draw(key, c);  c = c + 1
    until z ≥ t
    j = z mod m                     (uniform in 0 … i, unbiased)
    swap order[i] and order[j]
```

Rejecting `z < t` leaves `2^64 − t` accepted values, a multiple of `m`, so every `j` is equally
likely; a rejection happens with probability below `m / 2^64`. Only integer arithmetic is used, so
the permutation is identical on every OS, device and implementation.

**Epoch order.** With `shuffle: true`, epoch `e`'s training order is the training set's dataset
order permuted with `key = stream_key(seed, e, "shuffle:train")`: position `i` of the epoch holds
the training sample `order[i]`. With `shuffle: false` it is the dataset order. Each epoch's order is
computed from `seed` and `e` alone, so a resumed run continues with the same order (§5.4).

**Batches.** Batch `b` of an epoch holds positions `b · batch_size` to
`min((b + 1) · batch_size, n) − 1`. The **last batch is kept even if it is smaller**
(PyTorch's `DataLoader(drop_last=False)`), so there are `S = ⌈n / batch_size⌉` steps per epoch.
The validation set is batched the same way, in its dataset order.

**Known values** (for tests of an implementation):

| Quantity | Value |
|---|---|
| `stream_key(42, 0, "shuffle:train")` | `0xB220B916172EA32D` |
| its `draw` 0–2 | `0x1C9672ADB1570786`, `0x9AB71BAF68C62E49`, `0x66475AAADA252698` |
| the permutation of `n = 10` with it (epoch 0, seed 42) | `[8, 3, 4, 1, 5, 7, 2, 9, 6, 0]` |
| the permutation of `n = 10`, epoch 1 (`stream_key(42, 1, "shuffle:train")` = `0x310547A3638A4BFA`) | `[8, 5, 4, 6, 3, 0, 1, 2, 9, 7]` |
| `stream_key(42, 0, "split:validation")` | `0x412B830F89CE0DDF` |
| its permutation of `n = 10` | `[7, 4, 8, 2, 3, 5, 0, 1, 9, 6]`; with `fraction = 0.3` (`n_val = 3`), the validation set is samples `[4, 7, 8]` and the training set `[0, 1, 2, 3, 5, 6, 9]` |

## 4. Events and cancellation

### 4.1 Events

`Trainer::run` reports through its callback (public-api.md §2 rules 5–6, §4). Each event is a
`TrainingEvent`, whose JSON is an object with an `"event"` tag, like `ProbeEvent`
(public-api.md §1.1):

```jsonc
{"event": "started", "device": {"backend": "cpu"}, "epochs": 5, "steps_per_epoch": 844, "total_steps": 4220,
 "train_samples": 54000, "validation_samples": 6000, "trainable_elements": 1199882, "step": 0}
{"event": "epoch_started", "epoch": 0, "batch": 0}
{"event": "step", "epoch": 0, "batch": 0, "step": 0, "loss": 2.3051, "lr": 1e-05}
{"event": "validation_started", "epoch": 0}
{"event": "epoch_finished", "epoch": 0, "step": 844,
 "metrics": {"train.loss": 0.2412, "train.accuracy": 0.9271, "val.loss": 0.0711, "val.accuracy": 0.9779}}
{"event": "checkpoint_saved", "path": "runs/mnist/step-000000000844", "step": 844}
{"event": "checkpoint_removed", "path": "runs/mnist/step-000000000422"}
{"event": "finished", "outcome": { /* TrainingOutcome, public-api.md §1.4 */ }}
```

| Event | Fields |
|---|---|
| `started` | `device` (DeviceId JSON, public-api.md §1.2), `epochs`, `steps_per_epoch` (`S`), `total_steps` (`T`), `train_samples`, `validation_samples` (`0` without a validation set), `trainable_elements` (the number of values in the trainable params), `step` (the step the run starts at: `0`, or where a resumed or earlier run stopped) |
| `epoch_started` | `epoch` (0-based), `batch` (the first batch this run trains in the epoch: `0`, or the resume position) |
| `step` | `epoch`, `batch`, `step` (`k`, the step's 0-based index, the `step` of its training-mode call), `loss` (the batch loss), `lr` (`lr(k)`) |
| `validation_started` | `epoch` |
| `epoch_finished` | `epoch`, `step` (steps completed), `metrics` (reported names → values, in the order of §2.7) |
| `checkpoint_saved` | `path` (the checkpoint directory, as resolved), `step` |
| `checkpoint_removed` | `path`: a checkpoint deleted by `keep_last` (§5.3) |
| `finished` | `outcome`: the `TrainingOutcome` that `run` returns |

- **Numbers.** Integers are JSON integers. `loss`, `lr` and metric values are `f64` written in the
  shortest form that reads back as the same `f64`. A non-finite value is the string `"nan"`, `"inf"`
  or `"-inf"` (JSON has no such numbers). The same rule holds in `state.json` (§5.2).
- Readers must accept new fields and new event kinds (public-api.md §1.2).

### 4.2 Order and throttling

One `run` emits:

```text
started
for each epoch, from the current one:
    epoch_started
    step …                              (throttled)
    [checkpoint_saved, checkpoint_removed …]     after a step, for every_steps (§5.1)
    [validation_started]                with a validation set
    epoch_finished
    [checkpoint_saved, checkpoint_removed …]     if one is due (§5.1)
finished                                or an error return instead
```

- **`step` throttling:** a `step` event is emitted for the first step of the run, for the last step
  of every epoch, and otherwise only if at least 100 ms (monotonic clock) have passed since the last
  emitted `step` event: at most about 10 per second. The steps left out are in the epoch's metrics.
- `finished` is the last event of a run that returns `Ok`. A run that returns an error emits no
  `finished`, and the error return is the last thing that happens (public-api.md §4).
- Callbacks are never concurrent and always in this order (public-api.md §2 rule 6). A callback that
  calls `run`, `export` or `close` on the same trainer gets `InvalidInput` (the trainer is
  running).

### 4.3 Cancellation

`run` checks its `CancellationToken` after every optimizer step (§1.2 step 8), before each
validation batch, and before each checkpoint is written. A step that has begun always finishes,
including its optimizer update, so the params are never half-updated.

When the token is set:
1. If `checkpoint` is set and `checkpoint.on_cancel` is true, a checkpoint of the current state is
   written (§5.1), unless one was just written for this step. A run cancelled during the validation
   pass saves the state before validation (§5.2, `batch = S`); the resumed run repeats the whole
   validation pass.
2. `finished` is emitted, and `run` returns `Ok(TrainingOutcome)` with `stop_reason: "cancelled"`.

Unlike `Model::run`, a cancelled training run returns its outcome and keeps its state: the trainer
can be exported (public-api.md §1.4), and calling `run` again continues where it stopped, with the
same results as an uninterrupted run (§5.4). Cancelling before the first step returns an outcome
with `step: 0`.

## 5. Checkpoints, resume and export

### 5.1 The run directory and when checkpoints are written

`checkpoint.dir` is the **run directory**. Each checkpoint is a sub-directory named
`step-` + the number of optimizer steps it contains, as decimal zero-padded to 12 digits
(`step-000000000844`; wider if the number has more digits). The trainer creates, replaces and
deletes only entries named `step-<digits>` and `.tmp-step-<digits>`; it never touches other files.

A checkpoint is written:
- after epoch `e`'s `epoch_finished` if `(e + 1)` is a multiple of `every_epochs`, or if a multiple
  of `every_steps` was reached by the epoch's last step, or if `e` is the last epoch (always, so
  every completed run ends with a checkpoint);
- after step `k` if `k + 1` is a multiple of `every_steps` and `k` is not the last step of its epoch
  (that one is covered above, after the validation pass);
- on cancellation, as §4.3 says.

A new run (not a resume) refuses a run directory that already contains a `step-*` entry
(`InvalidInput` with `context.file`), so two runs never mix their checkpoints.

**Writing is atomic:**
1. Delete any `.tmp-step-*` directories in the run directory (left by a crash).
2. Write the four files of §5.2 into `.tmp-step-<N>`, and flush each to disk (`fsync`).
3. If `step-<N>` exists (a run resumed from an earlier checkpoint reached the same step), delete it.
4. Rename `.tmp-step-<N>` to `step-<N>`, then flush the run directory where the OS supports it.
5. Apply `keep_last` and `keep_best` (§5.3).

A crash leaves either the previous checkpoints, or the new one complete, plus at most one
`.tmp-step-*` directory, which resume ignores. Only step 5 deletes complete checkpoints, after the
new one exists. A failure to write is `Io` with `context.file`, and `run` returns it.

### 5.2 Checkpoint contents

```text
step-000000000844/
  state.json             the training state (below)
  graph.json             the graph, canonical JSON (graph-ir.md §6)
  weights.safetensors    every declared param, F32, by param name, no __metadata__
  optimizer.safetensors  the optimizer state, F32 (below)
```

`graph.json` and `weights.safetensors` are an inference model as they are: `Engine::load_graph`
loads them unchanged (the weights hold every param, running statistics included).

**`optimizer.safetensors`** holds the state of every trainable param that has state, as
`<param name>:<state name>`, with the param's shape, `F32`, no `__metadata__`. `:` appears in no
param name, so the names can't collide. The state names are PyTorch's (`optimizer.state_dict()`):

| Optimizer | State names |
|---|---|
| `sgd` with `momentum ≠ 0` | `momentum_buffer` |
| `sgd` with `momentum = 0` | none (the file holds no tensors) |
| `adam`, `adamw` | `exp_avg`, `exp_avg_sq` |

**`state.json`** (UTF-8 JSON, unknown fields rejected; numbers as §4.1 says):

```jsonc
{
  "format": "polypous.checkpoint",
  "format_version": 1,
  "polypous_version": "0.2.0",          // the version that wrote it (informative)
  "config": { /* the TrainingConfig, canonical JSON (to_json) */ },
  "base_dir": "/home/me/project",       // the absolute base directory relative config paths were resolved against
  "step": 844,                          // optimizer steps completed (k)
  "epoch": 1,                           // the epoch in progress, 0-based
  "batch": 0,                           // batches of that epoch completed, 0 … S
  "steps_per_epoch": 844,
  "train_samples": 54000,
  "validation_samples": 6000,
  "data_fingerprint": "8f3c2a01d4e5b6c7",
  "epoch_sums": {                       // the current epoch's training metrics so far
    "loss_sum": 0.0, "samples": 0,      // Σ loss_b · B_b (f64) and Σ B_b
    "correct": {"accuracy": 0}, "counted": {"accuracy": 0}
  },
  "history": [                          // one entry per finished epoch, in order
    {"epoch": 0, "step": 844, "metrics": {"train.loss": 0.2412, "train.accuracy": 0.9271,
                                          "val.loss": 0.0711, "val.accuracy": 0.9779}}
  ],
  "best": {"metric": "val.accuracy", "value": 0.9779, "epoch": 0, "step": 844,
           "checkpoint": "step-000000000844"},   // or null (no keep_best, or no value yet)
  "elapsed_seconds": 191.4              // wall-clock time of every run so far (informative)
}
```

- `epoch` and `batch` say where training continues: batch `batch` of epoch `epoch`. `batch = S`
  means the epoch's steps are done and its validation pass is not (§4.3).
- The **RNG state** is `config.seed` and the counters `step` and `epoch` (§1.3): the generator keeps
  no state. The **schedule state** is `step` (§3.2). The **data position** is `epoch` and `batch`;
  the epoch's order is recomputed (§3.6).
- `data_fingerprint` is 16 lowercase hex digits of `fnv1a` (graph-ir.md §3.1.12, over bytes) of the
  datasets as read (before a split), with only their used fields (§3.5), in role order (`train`,
  then `validation` if it isn't a `split`), field by field in name order: the field name, `0x00`, the dtype name, `0x00`, the shape
  as 8-byte little-endian integers, and the data bytes (little-endian, row-major). It detects data
  that changed between a run and its resume.
- `epoch_sums` holds the exact accumulators of §3.4 (`f64` sum, integer counts), so a resumed epoch
  reports the same metrics.

### 5.3 `keep_last` and `keep_best`

After each checkpoint is written (§5.1 step 5):
- **`keep_best`**: the best checkpoint is the one whose `state.json` `history` has the best value of
  the metric in its last entry: the largest for `accuracy`, the smallest for `loss`. A NaN is never
  best; on a tie the earlier checkpoint stays best. A checkpoint written before the first epoch
  finished has no value. Only a checkpoint that §5.1's first rule writes, after an
  `epoch_finished`, has a value: one written in the middle of an epoch (`every_steps`) or on
  cancellation has none, even if its last history entry is that of the previous epoch, because
  its weights are not the ones that entry measured. The best checkpoint is recorded in `best` and
  in the outcome.
- **`keep_last = n`**: every `step-*` checkpoint of the run directory except the `n` with the
  largest step and the best one is deleted, oldest first, each with a `checkpoint_removed` event. A
  failure to delete is logged at `warn` and doesn't stop training.

### 5.4 Resume

`Engine::resume_trainer(path, ResumeOptions, &cancel)` (public-api.md §1.4) takes a checkpoint
directory, or a run directory, in which case the checkpoint with the largest step is used
(`.tmp-step-*` is ignored; a run directory without checkpoints is `InvalidInput`).

**Loading an untrusted checkpoint.** Every check runs before the data it protects is read or
allocated; a failure is `InvalidInput` with `context.file` (or `Io` if a file can't be read):

1. `state.json` is at most 64 MiB and parses as above. `format` must be `polypous.checkpoint` and
   `format_version` `1`. `config` is a valid config (§2.1–§2.9). The counters agree:
   `step = epoch · steps_per_epoch + batch`, `batch ≤ steps_per_epoch`,
   `epoch < config.epochs` (or `epoch = config.epochs` with `batch = 0` for a finished run),
   `history` has exactly `epoch` entries with epochs `0 … epoch − 1`, and `best` names an existing
   history entry.
2. `graph.json` loads like `Graph::from_file` (graph-ir.md §4.3 limits, validation).
3. `weights.safetensors` is checked as runtime.md §2 checks a weight file (header first, exactly the
   declared params, declared shapes), but only `F32` storage is accepted.
4. `optimizer.safetensors`: header first; every tensor name is `<trainable param>:<state name>` of
   the optimizer kind, `F32`, with the param's shape; a param has all of its state names or none.
   After loading, a param that has a gradient but no state while `step > 0` (a checkpoint that
   doesn't fit its graph) is `InvalidInput` naming it, at the first step.
5. The datasets are read again from the config (relative paths against `base_dir`, or
   `ResumeOptions::base_dir`), and the app gives the `tensors` datasets again
   (`ResumeOptions::datasets`). Then §2.10 steps 3–6 run. The sample counts and
   `data_fingerprint` must equal the checkpoint's; otherwise `InvalidInput` ("the training data
   changed since the checkpoint").
6. The run directory is the checkpoint's parent directory, whatever `config.checkpoint.dir` says
   (so a moved run directory still works). If it holds checkpoints with a larger step than the one
   resumed, resume fails with `InvalidInput`, unless `ResumeOptions::discard_newer` is true; then
   they are deleted when the first new checkpoint is written (they belong to the continuation that is
   being replaced).

**What resume restores:** the params (all of them, running statistics included), the optimizer
state, `step`, `epoch`, `batch`, `epoch_sums`, `history`, `best` and `elapsed_seconds`; with them
the schedule, the data order and the random streams continue unchanged (§1.3, §3.2, §3.6). The
config can't change on resume, except the device (`ResumeOptions::device`).

**The guarantee.** On the CPU, a run that is cancelled and resumed — through `resume_trainer` from
any checkpoint it wrote, mid-epoch or at an epoch boundary, or by calling `run` again on the same
trainer — gives **bit-identical** params, optimizer state, `step`, `history`, `best` and
checkpoint files (`weights.safetensors`, `optimizer.safetensors`) to an uninterrupted run with the
same config, graph, initial weights and data, on the same machine and Polypous build. Events (the
`step` throttling depends on time) and `elapsed_seconds` are not covered. On a GPU, resume restores
the same state, but the following steps may differ from an uninterrupted run within the GPU's
run-to-run variation (§1.4); resuming a GPU run on the CPU or the other way round is allowed, with
the same caveat.

### 5.5 Export

`Trainer::export(graph_path, weights_path)` (public-api.md §1.4) writes the trainer's current
model: the graph's canonical JSON to `graph_path` and every declared param, running statistics
included, as `F32` to `weights_path` (one safetensors file, by param name, no `__metadata__`;
temporary file and rename, as `Model::save`). `Engine::load_graph` loads the pair, and with compute
dtype `f32` on the same device its outputs equal the trainer's inference-mode outputs bit for bit.
The graph is written unchanged (its `init` stays; `load_graph` ignores `init` when weights are
given). Exporting a checkpoint needs no call: its `graph.json` and `weights.safetensors` are the
same pair (§5.2).

## 6. Errors

Every error of training is a `polypous::Error` with the kinds of public-api.md §2 rule 7:

| Failure | Kind | Context |
|---|---|---|
| Config: malformed JSON, over a limit, unknown, missing, mistyped or out-of-range field, an unsupported `format_version` (§2.1) | `InvalidInput` | `pointer` |
| `precision` `f16` or `bf16` (§2.2) | `UnsupportedOperation` | `pointer` |
| Graph with state; a running statistic with more than one use (§2.10) | `UnsupportedOperation` | `tensor` or `node` |
| Unknown or unavailable device (§2.2) | `BackendUnavailable` | `device` |
| References, fields, dtypes, shapes, target values and binding against the graph and data (§2.10 steps 4–6); `batch_norm` with one value per channel | `InvalidInput` | `pointer`, `tensor`, `node` |
| A dataset file, image folder or weight file that is malformed, over a limit or unsupported (§3.5) | `InvalidInput` | `file`, `tensor`, `details` |
| Params with neither `init` nor weights (§2.10 step 7) | `InvalidInput` | `tensor`, `details` (all of them) |
| A run directory that already holds checkpoints; an invalid or inconsistent checkpoint; data that changed since the checkpoint (§5) | `InvalidInput` | `file` |
| A file or directory can't be read, created or written | `Io` | `file` |
| The device or host runs out of memory | `OutOfMemory` | `device` |
| An out-of-range index while running (graph-ir.md §3.1.8, for example `nn.embedding` ids from data) | `InvalidInput` | `node` |
| Backend failure | `Backend` | `node` |
| `run` while running, after the run finished or failed, or from its own callback; any call after `close` | `InvalidInput` | |
| Cancelled while the trainer is created | `Cancelled` | |
| A broken invariant or caught panic | `Internal` | `details` |

A cancelled `run` is not an error (§4.3). After `run` returns an error, the trainer's state is not
reliable (a step may have stopped half-way): `run` and `export` then return `InvalidInput`, and
training continues from the last checkpoint with `resume_trainer`.

## 7. Correctness tests

The test rules, tolerances and fixture formats are in [testing-strategy.md](../testing-strategy.md)
(§2, §3.6–§3.8). In short:

- The gradient of every differentiable op matches PyTorch autograd on fixtures (graph-ir.md §3.1.11).
- Optimizer steps match PyTorch for 3 steps from identical state, gradient clipping matches
  `clip_grad_norm_`, the schedules match PyTorch's schedulers, and the losses and their gradients
  match `torch.nn.functional`.
- The generator, dropout masks, initial values and shuffles match their known values and pass the
  statistical tests.
- An end-to-end MNIST CNN reaches ≥ 98% test accuracy within 5 epochs on the CPU, and its first
  training losses match PyTorch from identical initial weights.
- Resuming from a checkpoint produces bit-identical results to uninterrupted training (CPU, fixed seed).

## 8. Later (not M2)

Besides the "later" kinds of §3: mixed-precision training (`f16`/`bf16` compute with `f32` master
weights and loss scaling; `precision` is reserved for it), early stopping, gradient accumulation,
several losses, datasets larger than memory, data loading in parallel with training, an automatic
device fallback on out-of-memory, and graphs with state or `functions`/`call` (graph-ir.md §2.7).
