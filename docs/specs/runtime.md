# Spec: Runtime (graph interpreter)

**Status:** Binding for M1: §1–§6. Binding for M2 (training): the call's mode (§3.1), params given by
the caller (§3.2), the running-statistics updates (§3.3) and their rows in §5. Placement across
devices, streaming (M4), sessions and KV caches (M3) extend it later. Parameter initialization is not
the runtime's job: graph-ir.md §2.4.1 defines the values and the trainer applies them
([training.md](training.md)). **Crate:** `polypous-runtime`

The interpreter runs a validated graph ([graph-ir.md](graph-ir.md)) on **one device**. graph-ir.md
is the source for what a graph means (dtypes, shapes, state, `repeat`, op semantics); this spec says
how the runtime prepares and executes it: the plan, weight loading, the call contract and the
errors. It is binding for the facade (#45) and the M1 exit check (#47).

`polypous-runtime` maps IR nodes onto `polypous-backend`'s `DynTensor` ops. No Burn type appears in
its API or its code ([ADR-0007](../adr/0007-burn-version-pinning.md)).

## 1. Plan

A **plan** is a graph compiled for one device and one compute dtype (graph-ir.md §2.2), the
default being `f32`. Building it:

1. **Validates** the graph with that compute dtype (`Graph::analyze`). A graph with validation
   errors gives `InvalidGraph` with the full report.
2. **Checks the device** (graph-ir.md §2.2, "Device support"): the compute dtype, then every graph
   input and state entry, then every output port of every node (depth-first, body nodes included)
   in document order. Each dtype must be one the device supports (`DynTensor::device_supports`).
   The first failure is `UnsupportedOperation` naming the value or node (with its `repeat` path),
   the op, the dtype and the device. Every op input is the output of a node, a graph input or a
   state entry, and params have the compute dtype, so this covers the (op, dtype) pairs of the
   graph: the backend has no per-op dtype table in M1, and an op exists for every dtype a device
   supports. An op that needs a dtype internally that the device lacks (`core.arange` builds
   float values above `i32::MAX` through `i64`) reports it when it runs; that error has the same
   kind and names the node (§5).
3. **Orders** each scope (the root and each `repeat` body) topologically. Validation guarantees a
   DAG; the runtime still detects a cycle and returns an `Internal` error instead of looping or
   panicking. Ties keep document order. A `repeat` node runs after every node of the enclosing
   scope that its carries or any node of its body (at any depth) reference.
4. **Computes value lifetimes** per scope. A value is *owned* by the scope that defines it: graph
   inputs, root state entries and root nodes by the root; carry names, body state entries and body
   nodes by their body. The last use of an owned value is the last step of its scope that reads it;
   a `repeat` step reads every value of the enclosing scope that its carries or its body reference,
   so those values live until the whole `repeat` has finished. After each step, the values whose
   last use it was are dropped; a value that is never read is dropped right after the step that
   produced it (an input: before the first step). Graph outputs and root state updates stay alive
   until the call returns; carry-out and body state update values until the end of the iteration.
   Inside a body this happens per iteration, and at the end of every iteration every value the body
   owns is dropped, so intermediates don't accumulate across iterations.
   An op input whose last use is this step is moved into the op rather than copied.

A plan is immutable and is built once; it doesn't hold weights.

## 2. Weights

A **model** is a plan with its params loaded. Loading reads one safetensors file or a sharded
checkpoint (`*.safetensors.index.json`) through `polypous-formats` ([formats.md](formats.md) §1),
which validates the whole header first. The weight file is untrusted.

1. **Strict match.** The file's tensor names must be exactly the graph's declared param names
   (graph-ir.md §2.4). A declared param that the file lacks is `MissingWeight`, a file tensor
   that no param declares is `ExtraWeight` (both name the tensor). There is no option to allow
   extra tensors in M1: renaming and dropping tensors is the job of a `.polyp` weight map (M3).
   Declared but unused params must be present and are checked like the others, but not uploaded.
2. **Storage dtype.** Every param tensor must be stored as `F32`, `F16` or `BF16`. Another float
   storage (`F64`, `F8_*`, …) is `UnsupportedWeightDtype`; a non-float tensor (`I64`, `BOOL`, …) is
   `WeightDtype`. Quantized storage arrives in M3.
3. **Shape.** The stored shape must equal the declared shape exactly (no broadcasting, no
   squeezing): `WeightShape` naming the tensor, the declared and the stored shape.
4. **Checks before allocation.** Steps 1–3 run for every tensor, using only header metadata,
   before any tensor data is read or uploaded. Each tensor's element count and byte sizes (stored
   and, for the cast below, `f32`) are computed with checked arithmetic and compared with the
   tensor's byte range; host buffers are reserved fallibly.
5. **Cast once.** A param stored in the compute dtype is uploaded as it is. Otherwise `F16`/`BF16`
   data is widened to `f32` on the host (exact), uploaded, and cast on the device to the compute
   dtype, so every value is rounded once (round to nearest, ties to even). The device never needs
   the storage dtype.
6. **Expansion and tying.** Body param references expand to `params_prefix(i) + n` with the
   prefixes of nested repeats concatenated, outer first (graph-ir.md §2.6). Each graph param is
   loaded once, however many nodes or iterations use it.

Weights are loaded once per model and shared, read-only, by every call that doesn't give its own
params (§3.2).

## 3. Calls

A call takes the graph inputs and the state values (graph-ir.md §2.5) by name, its mode (§3.1,
default `inference`), optionally the params (§3.2), and an optional cancellation token. It returns
the graph outputs by output name, in declaration order, every root-level state entry (expanded
names, `Analysis::state` order) in a separate list, and, in training mode, the updated running
statistics in a third list (§3.3).

Before the first node runs, in this order:

1. **Binding** (graph-ir.md §2.3.4 steps 1–5) through `Analysis::bind` on the given dtypes and
   shapes: a missing, extra or duplicate input or state value, a wrong dtype (`float` resolved to
   the compute dtype) or rank, a literal dim that differs, an inconsistent or out-of-range symbol,
   an inferred dim below 1 and overflow are `InvalidInput` naming the input, state entry, symbol or
   node.
2. **Devices:** every input and state value must live on the plan's device; otherwise
   `InvalidInput` naming the value and both devices. The runtime never moves a value between
   devices silently.
3. **Params:** if the call gives the params, the checks of §3.2.
4. **Mode:** in training mode, the checks of §3.1, in the order listed there.

Then every node runs in plan order. Op nodes call their `DynTensor` function with the
attributes of the analysis (defaults filled in) and the dim attributes evaluated by the binding;
params are the loaded model tensors, or the tensors the call gives (§3.2). In training mode
`nn.batch_norm` and `nn.dropout` run their training-mode computation (graph-ir.md §3.3). A `repeat` node runs its body `count` times: iteration 0
reads the carry values, iteration `i + 1` the `carry_out` values of iteration `i`; the repeat's
ports are the last iteration's `carry_out` values.

**State.** A node that reads a state entry sees the value given to the call, in every iteration
and after any update (graph-ir.md §2.5). The returned state of an entry is the value of its
update, or the given value unchanged if it has none. A body's `state_updates` of iteration `i`
update the expanded entry `state_prefix(i) + n`.

**Outputs** are returned only if the whole call succeeds; a failed or cancelled call returns no
partial results (no outputs, state or running statistics). A graph output may be a graph input or
state value; it is returned as given.

### 3.1 Mode

A call runs in `inference` mode (the default, the behavior of M1) or in
`training { seed: u64, step: u64 }` mode (graph-ir.md §3.1.10). The mode changes only what
`nn.batch_norm` and `nn.dropout` compute.

- `seed` and `step` key the dropout masks (graph-ir.md §3.1.12); the runtime uses them for nothing
  else. Every `u64` is valid. Two calls with the same graph, inputs, params, `seed` and `step` draw
  the same masks, on every device. Giving each forward pass its own `step` is the caller's job; the
  trainer's rule is in [training.md](training.md).
- The interpreter stays stateless, as for state until sessions exist (graph-ir.md §2.5): it keeps
  no step counter, no masks and no running statistics between calls.

A training-mode call checks, after binding (§3 steps 1–3) and before any node runs, in this order:

1. **Compute dtype:** the plan's compute dtype must be `f32` in M2 (owner decision 6 for M2;
   graph-ir.md §3.1.11). Otherwise `UnsupportedOperation` naming the mode and the compute dtype.
2. **One use per running statistic:** after expanding repeats, every param that some
   `nn.batch_norm` node uses as `running_mean` or `running_var` must have exactly one use in the
   whole graph: one node, one role, one iteration. A param shared between `batch_norm` nodes, used in
   both roles, used by another op or role, or used by every iteration of a `repeat` (a
   `params_prefix` without `{i}`) is `UnsupportedOperation` naming the param and two of its uses
   (node paths, §5). PyTorch would update a shared running statistic once per use, in order; M2
   doesn't support that.
3. **Values per channel:** every `nn.batch_norm` node, in plan order (iterations in order), must
   have at least 2 values per channel, computed from the bound shape of its `x` (graph-ir.md §3.3).
   Otherwise `InvalidInput` naming the node path.

An inference-mode call makes none of these checks: a graph whose running statistics are shared
runs in inference mode as in M1.

### 3.2 Params given by the caller

By default a call reads the params loaded with the model (§2). A call may instead give the params
itself, as a map from param name to tensor. This is how the trainer runs the interpreter on its
param store ([ADR-0014](../adr/0014-training-on-dyntensor-with-own-optimizers.md)). Either mode may
use either source: the trainer's validation passes run in inference mode on its params.

Given params are checked before any node runs (§3 step 3), with the same strictness as a weight
file (§2):
- the map holds exactly the graph's declared params, unused ones included: a missing or extra name
  is `InvalidInput` naming it;
- each tensor has the compute dtype and exactly the declared shape, and lives on the plan's device;
  otherwise `InvalidInput` naming the param, the expected and the given dtype, shape or device.

Nothing is cast, copied or moved, and the tensor values aren't inspected. The checks use only
metadata, so they allocate nothing.

The runtime reads the given tensors and never changes them. If a param, input or state tensor
carries autodiff tracking (ADR-0014; for example a param the trainer marked as a leaf), the ops
record through it, so the outputs carry it too and the caller can back-propagate a loss computed
from them. The runtime itself never marks, detaches or back-propagates, except for the running
statistics (§3.3), and dropout masks are constants without tracking.

### 3.3 Running statistics

A training-mode call returns the new running statistics of every `nn.batch_norm` node it ran
(graph-ir.md §3.3): a list of (param name, tensor) pairs, in the order the nodes ran (iterations in
order), with each node's `running_mean` before its `running_var`. §3.1 check 2 makes every name
appear once. Each tensor has the compute dtype and the declared shape, lives on the plan's device
and carries no autodiff tracking.

The params themselves don't change: the caller writes the values back (the trainer, at the point
[training.md](training.md) specifies). An inference-mode call returns an empty list.

## 4. Cancellation

`CancellationToken` is a cheap-to-clone flag shared between clones. The runtime checks it before
every node (op or `repeat`) and before every iteration of a `repeat`. If it is set, the call stops
with `Cancelled`, naming the node or iteration it stopped before. Nodes that already ran are not
undone; nothing is returned.

## 5. Errors

`RuntimeError` has one variant per failure below. `RuntimeError::kind()` maps it to the
`polypous::ErrorKind` of [public-api.md](public-api.md) §2 rule 7; graph-ir.md §4.4 lists the
kinds of binding and device failures.

| Failure | Kind |
|---|---|
| Graph validation (§1 step 1) | `InvalidGraph` |
| Device lacks the compute dtype or a value dtype (§1 step 2), or an op reports an unsupported dtype or operation while running | `UnsupportedOperation` |
| The device is not available (backend not compiled in, no such device) | `BackendUnavailable` |
| Missing, extra, non-float or wrong-shape weight (§2) | `InvalidInput` |
| Float storage other than `F32`/`F16`/`BF16` (§2 step 2) | `UnsupportedModel` |
| Weight file can't be read (I/O) | `Io` |
| Weight file malformed or truncated (header, offsets, index) | `InvalidInput` |
| Binding or device check of a call (§3) | `InvalidInput` |
| Given params with a missing or extra name, or a wrong dtype, shape or device (§3.2) | `InvalidInput` |
| Training mode with a compute dtype other than `f32`, or a running statistic with more than one use (§3.1) | `UnsupportedOperation` |
| Training-mode `nn.batch_norm` with one value per channel (§3.1) | `InvalidInput` |
| Out-of-range index in `core.gather`, `core.index_select` or `nn.embedding` (graph-ir.md §3.1.8) | `InvalidInput` |
| Cancelled (§4) | `Cancelled` |
| Backend failure while running or uploading | `Backend` |
| A broken invariant (cycle, missing param or attribute after validation, a backend shape error after binding) | `Internal` |

- **Every error raised while running a node carries the node path**: the ids of the enclosing
  `repeat`s with their iteration, then the node id, joined by `/`: `fc1`, `layers[3]/attn`,
  `outer[1]/inner[0]/mlp`.
- Backend errors are kept as `TensorError` values (`source()`), not only as text, so the facade
  can inspect them.
- Weight errors name the tensor; binding errors name the input, state entry, symbol or node.
- The runtime never panics on a graph, a weight file, call inputs or given params.

## 6. Tracing

A call can record, after each executed step (each op node, each `repeat` as a whole, and body
nodes per iteration), the node path and the names of the values alive at that moment. This is for
tests and debugging (it shows §1 step 4 at work); it doesn't change results.
