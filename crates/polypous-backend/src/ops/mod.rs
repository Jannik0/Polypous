//! The IR ops of `docs/specs/graph-ir.md` §3 as [`DynTensor`] functions.
//!
//! Every op is a method (or associated function) on [`DynTensor`] with plain Rust arguments: no
//! `polypous-ir` type appears here (`docs/architecture.md` §2). `polypous-runtime` maps IR nodes
//! onto these functions, after validation has proven dtypes and shapes (graph-ir.md §2.3.4).
//! The functions still check every argument and return a [`TensorError`] instead of panicking
//! (CLAUDE.md, "No panics on untrusted input").
//!
//! # Layout
//!
//! One file per op family, each with its own `impl DynTensor` block, so that op families can be
//! added in parallel without touching each other's files:
//!
//! | File | Ops (graph-ir.md §3.2, §3.3) |
//! |---|---|
//! | `mod.rs` | Shared pieces: [`Scalar`], dtype classes, broadcasting, operand checks |
//! | `elementwise.rs` | `core.add sub mul div pow maximum minimum neg abs exp log sqrt rsqrt sin cos tanh clamp where` |
//! | `compare.rs` | `core.eq ne lt le gt ge`, and the logic ops `core.and or not` |
//! | `cast.rs` | `core.cast` |
//! | `shape.rs` | `core.reshape flatten permute transpose squeeze unsqueeze expand concat split slice pad` |
//! | `indexing.rs` | `core.gather index_select` |
//! | `creation.rs` | `core.constant arange zeros ones full full_like`, and the fill values of `pad` |
//! | `reduce.rs` | `core.sum mean max min argmax argmin cumsum` |
//! | `linalg.rs` | `core.matmul` |
//! | `softmax.rs` | `core.softmax log_softmax` |
//! | `topk.rs` | `core.top_k` |
//! | `nn_layers.rs` | `nn.linear embedding dropout`, and the param checks shared by the `nn` ops |
//! | `conv.rs` | `nn.conv1d conv2d`, and the window output size shared with the pools |
//! | `pool.rs` | `nn.max_pool2d avg_pool2d adaptive_avg_pool2d` |
//! | `norm.rs` | `nn.batch_norm layer_norm group_norm rms_norm` |
//! | `activation.rs` | `nn.relu leaky_relu gelu gelu_tanh silu sigmoid softplus mish` |
//!
//! Reference tests live in `tests/reference_*.rs`, one test binary per op family, and share the
//! fixture harness in `tests/reference/common.rs` (testing-strategy.md §6). A new family adds
//! its own `tests/reference_<family>.rs` with `mod common;` pointing at that file.
//!
//! # Method names
//!
//! | IR op | Method |
//! |---|---|
//! | `core.add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum` | [`DynTensor::add`], [`sub`](DynTensor::sub), [`mul`](DynTensor::mul), [`div`](DynTensor::div), [`pow`](DynTensor::pow), [`maximum`](DynTensor::maximum), [`minimum`](DynTensor::minimum) |
//! | `core.neg`, `abs`, `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` | The same names |
//! | `core.clamp` | [`DynTensor::clamp`] |
//! | `core.where` | [`DynTensor::where_`] (`where` is a Rust keyword) |
//! | `core.eq`, `ne`, `lt`, `le`, `gt`, `ge` | [`equal`](DynTensor::equal), [`not_equal`](DynTensor::not_equal), [`less`](DynTensor::less), [`less_equal`](DynTensor::less_equal), [`greater`](DynTensor::greater), [`greater_equal`](DynTensor::greater_equal) (not `eq`/`ne`, which read like `PartialEq`) |
//! | `core.and`, `or`, `not` | [`logical_and`](DynTensor::logical_and), [`logical_or`](DynTensor::logical_or), [`logical_not`](DynTensor::logical_not) (PyTorch's names) |
//! | `core.cast` | [`DynTensor::cast`] |
//! | `core.reshape`, `flatten`, `permute`, `transpose`, `squeeze`, `unsqueeze`, `expand`, `split`, `slice`, `pad` | The same names: [`reshape`](DynTensor::reshape), [`flatten`](DynTensor::flatten), [`permute`](DynTensor::permute), [`transpose`](DynTensor::transpose), [`squeeze`](DynTensor::squeeze), [`unsqueeze`](DynTensor::unsqueeze), [`expand`](DynTensor::expand), [`split`](DynTensor::split), [`slice`](DynTensor::slice), [`pad`](DynTensor::pad) |
//! | `core.concat` | [`DynTensor::concat`] (an associated function over a list) |
//! | `core.gather`, `index_select` | [`gather`](DynTensor::gather), [`index_select`](DynTensor::index_select) |
//! | `core.constant`, `arange`, `zeros`, `ones`, `full` | Associated functions with a [`DeviceId`] argument: [`constant`](DynTensor::constant), [`arange`](DynTensor::arange), [`zeros`](DynTensor::zeros), [`ones`](DynTensor::ones), [`full`](DynTensor::full) |
//! | `core.full_like` | [`DynTensor::full_like`] |
//! | `core.sum`, `mean`, `max`, `min` | The same names: [`sum`](DynTensor::sum), [`mean`](DynTensor::mean), [`max`](DynTensor::max), [`min`](DynTensor::min) (`dims: None` reduces every dim) |
//! | `core.argmax`, `argmin`, `cumsum` | [`argmax`](DynTensor::argmax), [`argmin`](DynTensor::argmin), [`cumsum`](DynTensor::cumsum) |
//! | `core.matmul` | [`DynTensor::matmul`] |
//! | `core.softmax`, `log_softmax` | [`softmax`](DynTensor::softmax), [`log_softmax`](DynTensor::log_softmax) |
//! | `core.top_k` | [`DynTensor::top_k`], returning `(values, indices)` |
//! | `nn.linear`, `conv1d`, `conv2d`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, `dropout` | The same names: [`linear`](DynTensor::linear), [`conv1d`](DynTensor::conv1d), [`conv2d`](DynTensor::conv2d), [`max_pool2d`](DynTensor::max_pool2d), [`avg_pool2d`](DynTensor::avg_pool2d), [`adaptive_avg_pool2d`](DynTensor::adaptive_avg_pool2d), [`dropout`](DynTensor::dropout) |
//! | `nn.embedding` | [`DynTensor::embedding`], called on `ids` with the `weight` param |
//! | `nn.batch_norm`, `layer_norm`, `group_norm`, `rms_norm` | The same names: [`batch_norm`](DynTensor::batch_norm), [`layer_norm`](DynTensor::layer_norm), [`group_norm`](DynTensor::group_norm), [`rms_norm`](DynTensor::rms_norm) |
//! | `nn.relu`, `leaky_relu`, `gelu`, `gelu_tanh`, `silu`, `sigmoid`, `softplus`, `mish` | The same names |
//!
//! # Arguments of the shape, indexing and creation ops
//!
//! The runtime calls these after `polypous_ir::Analysis` has bound the graph, so every argument
//! arrives **concrete and normalized**: axes are `0…r-1` (`unsqueeze`: `0…r`), `-1` in `reshape`
//! and `expand` is replaced by the size it stands for, `squeeze`'s axes are filled in, `slice`
//! bounds are resolved and clamped (`0 ≤ start < end ≤ n`), and defaults are filled in. The
//! functions still check every argument and return a [`TensorError`] instead of panicking:
//! wrong rank, axes out of range or repeated, element counts or shapes that don't fit, ranks
//! above 6 ([`TensorError::UnsupportedRank`]), zero-size results (graph-ir.md §2.3: not in M1),
//! lists over the limits of graph-ir.md §4.3, and fill values that don't fit the dtype.
//!
//! - **Views.** `reshape`, `permute`, `transpose`, `expand`, `slice` and `split` return Burn views
//!   where Burn makes them (strided, or stride 0 for `expand`); no op here assumes a contiguous
//!   input. The reference tests feed permuted, transposed, expanded and stepped-slice views to
//!   every shape and indexing op and to `full_like` (`tests/reference/common.rs`, `feed_layout`).
//! - **Bool `gather`/`index_select`** run on an `i32` copy and convert back: Burn 0.22.1's `flex`
//!   returns them in `u8` storage, which isn't a bool dtype. Done on every backend.
//! - **Out-of-range indices** (§3.1.8): `gather` and `index_select` read the minimum and maximum of
//!   the index tensor back to the host before the op runs (one reduction and **one device sync**
//!   per call; on the CPU a pass over the indices) and return
//!   [`TensorError::IndexOutOfRange`]. Burn 0.22.1 does not check: `flex` panics, and the CubeCL
//!   backends read whatever memory the index points at (0, a neighbouring element, or garbage on
//!   lavapipe; `index_select` with index `-1` crashed the lavapipe process with SIGSEGV).
//! - **`pad`** is built from `full` blocks and `concat`, for every dtype (Burn's `pad` exists only
//!   for numeric tensors).
//! - **Fill values** (`full`, `full_like`, `constant`, `pad`; §3.1.7): `bool` takes
//!   [`Scalar::Bool`]; `i32`/`i64` take an in-range [`Scalar::Int`]; float dtypes take
//!   [`Scalar::Float`] or [`Scalar::Int`] (as `f64`), rounded on the host from `f64` to `f32` and
//!   then to `f16`/`bf16`. That is how PyTorch (`torch.full`, `torch.tensor`) rounds Python floats,
//!   so the two agree where a single rounding from `f64` would differ. NaN and infinities are
//!   allowed for float dtypes.
//!
//! # Reductions, matmul, softmax and top_k
//!
//! Arguments arrive normalized as above: reduction axes are `0…r-1` (`dims: None` when the
//! graph leaves `dim` out, meaning every dim), `top_k`'s `k` is a concrete size, and the
//! defaults (`keepdim: false`, `dtype: i64`, `top_k` `dim: -1` → `r-1`, `largest: true`) are
//! filled in. Empty, repeated or out-of-range axes, `k` outside `1…d(dim)`, a non-integer index
//! dtype, an `i32` index dtype for a dim above `i32::MAX`, and matmul operands whose `k` or batch
//! dims don't fit are [`TensorError::InvalidArgument`] or [`TensorError::ShapeMismatch`]; an index
//! dtype the device lacks is [`TensorError::UnsupportedDType`].
//!
//! - **Explicit NaN and tie rules** (§3.1.8), so results don't depend on a backend's reduction
//!   order: `max`/`min` replace NaN by `∓inf` and set NaN where a reduced slice held one;
//!   `argmax`/`argmin` take the smallest index among the positions equal to the extremum (the
//!   NaN positions, in a slice with a NaN), with an integer `min`. Burn's own `argmax` got NaN and
//!   ties right on `flex` and lavapipe in our probes, but nothing in Burn promises it.
//! - **Integer `sum` and `cumsum` wrap** (§3.1.8). `flex` sums `i64` with plain `+`, and computes
//!   `cumsum` of `i32` and `i64` with plain `+`; all three panic on overflow in debug builds
//!   (`attempt to add with overflow`). On the CPU they run on the host with `wrapping_add`
//!   (`i32` `sum` widens to `i64` in `flex` and wraps already). GPU integer arithmetic wraps
//!   (lavapipe: `i64::MAX + 1` and `i32::MAX + 1` wrap in `sum` and `cumsum`).
//! - **Half-precision accumulation:** `sum`, `mean`, `cumsum`, `softmax` and `log_softmax` of
//!   `f16`/`bf16` compute in `f32` and round once to the dtype, on every backend (§3.1.8). On
//!   lavapipe, CubeCL's `f16` `cumsum` accumulates in `f16`: the cumulative sum of 4096 ones
//!   stops at 2048. `matmul` keeps the backend's kernel; with `k = 256` it meets `rtol 1e-2` in
//!   `f16` and `bf16` on `flex` and in `f16` on lavapipe (which has no `bf16`).
//! - **`softmax`/`log_softmax`** subtract the NaN-propagating row maximum first, so large inputs
//!   don't overflow; rows with NaN or `+inf`, and rows of only `-inf`, are NaN, as in PyTorch.
//! - **`top_k` runs on the host** on every backend, with one device sync per call: Burn 0.22.1
//!   has no GPU sort (its CubeCL sort reads back to the host too), its sorts are unstable, and
//!   CubeCL's `topk_with_indices` returned garbage after a NaN on lavapipe. See `topk.rs`.
//! - **`matmul`** broadcasts the batch dims explicitly (different ranks allowed) before Burn's
//!   `matmul` sees them.
//!
//! # The `nn` ops (graph-ir.md §3.3)
//!
//! Called on `x` (`embedding`: on `ids`). **Params are plain [`DynTensor`] arguments** after it, in
//! catalog order (graph-ir.md §2.4), already in the compute dtype and in PyTorch's layouts
//! (`weight` `[out, in]`, `[C_out, C_in/groups, k…]`), so PyTorch weights load unchanged; optional
//! params are `Option<DynTensor>`. Attributes arrive with defaults filled in (`stride` of the
//! pools = `kernel_size`; `layer_norm`/`rms_norm` get `normalized_shape` even where the graph took
//! it from `weight`). Every argument is still checked: dtype class, rank, the param dtypes
//! ([`TensorError::DTypeMismatch`]), devices and shapes, `stride`/`dilation`/`groups`/`kernel_size`
//! ≥ 1, `groups` dividing `C_in` and `C_out`, pool `padding` ≤ `kernel_size / 2`, output sizes ≥ 1
//! with checked arithmetic, `0 ≤ p < 1`, finite `eps`/`beta`/`threshold` > 0, `num_groups` dividing
//! `C`, `normalized_shape` equal to the trailing dims; otherwise [`TensorError::InvalidArgument`],
//! before anything is allocated.
//!
//! Burn 0.22.1 findings (CPU `flex` and lavapipe), and what is done about them:
//!
//! - **f16/bf16 compute in `f32`** for `linear`, conv, the average pools, the norms and every
//!   activation except `relu` (graph-ir.md §3.1.8), on every backend. Without it, in f16:
//!   `nn.linear/transposed_f16` missed `rtol 1e-2` on both devices (expected `9.928e-4`, got
//!   `9.766e-4`), conv2d f16 reached a relative error of `1.07e-1` on `flex` and
//!   `4.39` on lavapipe (outputs that cancel to near zero), `layer_norm/large_mean_f16` failed on 114
//!   of 128 elements and `rms_norm/large_f16` on all (`mean(x²)` overflows f16), `gelu` f16 tails
//!   came out `-0`, and `adaptive_avg_pool2d/long_f16` missed by `1.2e-2` on `flex`.
//! - **`max_pool2d`** pads with `-inf` and propagates NaN explicitly (one extra max-pool of a code
//!   tensor, see `pool.rs`). On lavapipe, CubeCL pads with `-f32::MAX` (an all-padding window, or
//!   a padded window of only `-inf` inputs, gave `-3.4028235e38`) and dropped NaN (the window
//!   `1, NaN, 5, 6` gave 6). `flex` got both right.
//! - **`embedding`** checks its ids before the lookup like `index_select` (one reduction and one
//!   device sync per call; [`TensorError::IndexOutOfRange`]).
//! - **Activations** are written out from elementary ops rather than taken from
//!   `burn::tensor::activation`, so NaN, infinities and large inputs don't depend on a backend
//!   (`relu` is a selection, not `max(x, 0)`, whose NaN behavior GPU drivers leave open; `softplus`
//!   doesn't overflow; `gelu_tanh` is `x · sigmoid(2u)`). Burn's own functions passed every `f32`
//!   case on `flex` and lavapipe; in f16 its `gelu_approximate` overflowed (`x³` for `x = 6e4`).
//! - **Norms** use a two-pass variance (a one-pass `E[x²] - E[x]²` misses the f32 tolerance at mean
//!   100 ± 1).
//! - Conv with groups and dilation, `avg_pool2d` with and without `count_include_pad`,
//!   `adaptive_avg_pool2d` with overlapping cells and `erf` in `f32` (incl. `|x|` up to 10) matched
//!   PyTorch with Burn's kernels as they are, on both devices. CubeCL's `erf` is the
//!   Abramowitz–Stegun 7.1.26 approximation (absolute error 1.5e-7) and its `log1p` is
//!   `ln(1 + x)`; both are inside the tolerances for these ops.
//! - **CUDA:** `linear` uses [`DynTensor::matmul`], so f32 `linear` shares matmul's TF32
//!   limitation on CUDA (docs/status.md §4); conv may use TF32 too. No workaround, by design.
//!
//! # Semantics (graph-ir.md §3.1)
//!
//! - **Strict dtypes** (§3.1.2): operands that must agree have the same dtype, or the op fails
//!   with [`TensorError::DTypeMismatch`]. A dtype outside the op's class is
//!   [`TensorError::UnsupportedOperation`]. There is no promotion and no silent conversion.
//! - **Broadcasting** (§3.1.3): shapes are right-aligned, the shorter one is padded with leading
//!   `1`s, and each pair of dims must be equal or contain a `1`. Ranks may differ; the output has
//!   the largest rank. Every operand is broadcast explicitly (reshape to the output rank, then
//!   `expand`) before Burn sees it, so no backend's own broadcasting rules matter.
//! - **NaN and special values** (§3.1.8) are set explicitly with masks wherever a backend was
//!   found to get them wrong, so results don't depend on the backend:
//!   - `maximum`, `minimum` and `clamp` propagate NaN (`flex`'s `max_pair` returns the non-NaN
//!     operand).
//!   - Comparisons: NaN compares false, except in `ne` (the CubeCL backends compare ordered, so
//!     `ne` gave `false` on lavapipe). Float → `bool` casts map NaN to `true` for the same reason.
//!   - `pow` follows C99 Annex F like `torch.pow` (`pow(1, NaN) = pow(NaN, 0) = 1`, zero and
//!     infinite bases and exponents); CubeCL's `exp2(b · log2(a))` got four of them wrong on
//!     lavapipe.
//! - **Internal precision:** the transcendental ops and `pow` on `f16`/`bf16` compute in `f32`
//!   and round once to the dtype, on every backend (`flex` does this anyway; CubeCL's `f16` `tanh`
//!   missed `rtol 1e-2` near zero on lavapipe). The output dtype is unchanged.
//! - **Integer overflow wraps** (§3.1.8). Burn's CPU backend (`flex`) computes `i64` `add`, `sub`
//!   and `mul` with plain `+ - *`, which panic on overflow when overflow checks are on (debug
//!   builds). On the CPU those three ops on `i64` therefore run on the host with
//!   `wrapping_add`/`wrapping_sub`/`wrapping_mul`. `flex` computes `i32` in `i64` and truncates,
//!   and uses `wrapping_neg`/`wrapping_abs`, so those wrap already. GPU integer arithmetic wraps.
//! - **Devices** (graph-ir.md §2.2): a tensor only exists on a device that supports its dtype
//!   ([`DynTensor::from_data`] and [`DynTensor::to_device`] check), and [`DynTensor::cast`] checks
//!   the target dtype, so no op here can reach a device with an unsupported dtype.
//!   [`DynTensor::device_supports`] answers the question up front.

mod activation;
mod cast;
mod compare;
mod conv;
mod creation;
mod elementwise;
mod indexing;
mod linalg;
mod nn_layers;
mod norm;
mod pool;
mod reduce;
mod shape;
mod softmax;
mod topk;

use crate::device::DeviceId;
use burn::tensor::Shape;

use crate::tensor::{DType, DynTensor, Ranked, Result, TensorError, map_rank, num_elements};

/// Most inputs of `core.concat` and outputs of `core.split` (graph-ir.md §4.3).
pub(crate) const MAX_CONCAT_SPLIT: usize = 1024;

/// `UnsupportedRank` unless `shape` has 1–6 dims; `InvalidArgument` for a zero-size dim
/// (graph-ir.md §2.3: not in M1) or a shape whose byte size overflows. Every op checks its output
/// shape with this before it allocates anything.
pub(crate) fn check_shape(op: &'static str, shape: &[usize], dtype: DType) -> Result<()> {
    if !(1..=DynTensor::MAX_RANK).contains(&shape.len()) {
        return Err(TensorError::UnsupportedRank { rank: shape.len() });
    }
    if shape.contains(&0) {
        return Err(TensorError::InvalidArgument {
            op,
            reason: format!("shape {shape:?} has a zero-size dim (not supported in M1)"),
        });
    }
    let bytes = num_elements(shape).and_then(|n| n.checked_mul(dtype.size_bytes()));
    if bytes.is_none_or(|b| isize::try_from(b).is_err()) {
        return Err(TensorError::InvalidArgument {
            op,
            reason: format!("shape {shape:?} is too large"),
        });
    }
    Ok(())
}

/// `InvalidArgument` unless `axis < rank`. `name` is the argument's name in the IR (`dim`, …).
pub(crate) fn check_axis(op: &'static str, name: &str, axis: usize, rank: usize) -> Result<()> {
    if axis < rank {
        Ok(())
    } else {
        Err(TensorError::InvalidArgument {
            op,
            reason: format!("`{name}` {axis} is out of range for rank {rank}"),
        })
    }
}

/// A scalar argument of an op, for example a `clamp` bound (graph-ir.md §3.1.7, type `scalar`).
///
/// Which variants an op accepts is part of its documentation. In general an integer fits every
/// dtype it is in range for, a float fits float dtypes (it is rounded to the dtype), and a bool
/// fits `bool`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Scalar {
    /// A float value; may be NaN or ±inf.
    Float(f64),
    /// An integer value.
    Int(i64),
    /// A boolean value.
    Bool(bool),
}

/// The dtype classes of graph-ir.md §3.1.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DTypeClass {
    /// `N`: float or integer.
    Numeric,
    /// `F`: `f32`, `f16`, `bf16`.
    Float,
    /// `B`: `bool`.
    Bool,
    /// `I`: `i64`, `i32` (index inputs, §3.1.2).
    Index,
    /// `A`: any dtype.
    Any,
}

impl DTypeClass {
    fn contains(self, dtype: DType) -> bool {
        match self {
            DTypeClass::Numeric => dtype.is_float() || dtype.is_int(),
            DTypeClass::Float => dtype.is_float(),
            DTypeClass::Bool => dtype == DType::Bool,
            DTypeClass::Index => dtype.is_int(),
            DTypeClass::Any => true,
        }
    }

    /// `UnsupportedOperation` unless `dtype` is in the class.
    pub(crate) fn check(self, op: &'static str, dtype: DType) -> Result<()> {
        if self.contains(dtype) {
            Ok(())
        } else {
            Err(TensorError::UnsupportedOperation { op, dtype })
        }
    }
}

/// The broadcast shape of `shapes` (graph-ir.md §3.1.3): right-aligned, padded with leading `1`s,
/// each dim equal or `1`. `None` if two dims are incompatible.
pub(crate) fn broadcast_shapes(shapes: &[&[usize]]) -> Option<Vec<usize>> {
    let rank = shapes.iter().map(|shape| shape.len()).max()?;
    let mut out = vec![1usize; rank];
    for shape in shapes {
        let offset = rank - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let slot = &mut out[offset + i];
            if *slot == 1 {
                *slot = dim;
            } else if dim != 1 && dim != *slot {
                return None;
            }
        }
    }
    Some(out)
}

/// `DeviceMismatch` unless both devices are the same.
pub(crate) fn check_same_device(op: &'static str, lhs: DeviceId, rhs: DeviceId) -> Result<()> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(TensorError::DeviceMismatch {
            op,
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
        })
    }
}

/// Checks the operands of a broadcasting binary op with dtype class `class` and broadcasts both to
/// the output shape. The checks run in this order: dtype mismatch, dtype class, device, shape.
pub(crate) fn prepare_binary(
    op: &'static str,
    class: DTypeClass,
    lhs: DynTensor,
    rhs: DynTensor,
) -> Result<(DynTensor, DynTensor)> {
    if lhs.dtype() != rhs.dtype() {
        return Err(TensorError::DTypeMismatch {
            op,
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    class.check(op, lhs.dtype())?;
    check_same_device(op, lhs.device(), rhs.device())?;
    let (lhs_shape, rhs_shape) = (lhs.shape(), rhs.shape());
    let Some(shape) = broadcast_shapes(&[&lhs_shape, &rhs_shape]) else {
        return Err(TensorError::ShapeMismatch {
            op,
            lhs: lhs_shape,
            rhs: rhs_shape,
        });
    };
    Ok((lhs.broadcast_to(&shape)?, rhs.broadcast_to(&shape)?))
}

impl DynTensor {
    /// Broadcasts the tensor to `shape`, which must be a broadcast of its shape (callers compute
    /// it with [`broadcast_shapes`]): leading `1`s are added with a reshape, then size-1 dims are
    /// expanded. Burn's `expand` makes a view, not a copy.
    pub(crate) fn broadcast_to(self, shape: &[usize]) -> Result<Self> {
        let current = self.shape();
        if current == shape {
            return Ok(self);
        }
        if current.len() > shape.len() {
            return Err(TensorError::ShapeMismatch {
                op: "broadcast",
                lhs: current,
                rhs: shape.to_vec(),
            });
        }
        let compatible = current
            .iter()
            .zip(&shape[shape.len() - current.len()..])
            .all(|(&have, &want)| have == want || have == 1);
        if !compatible {
            return Err(TensorError::ShapeMismatch {
                op: "broadcast",
                lhs: current,
                rhs: shape.to_vec(),
            });
        }
        let aligned = if current.len() < shape.len() {
            let mut padded = vec![1usize; shape.len() - current.len()];
            padded.extend_from_slice(&current);
            self.reshape(&padded)?
        } else {
            self
        };
        if aligned.shape() == shape {
            return Ok(aligned);
        }
        let (inner, dtype, device) = aligned.into_parts();
        let inner = map_rank!(inner, t => t.expand(shape)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

impl<const D: usize> Ranked<D> {
    /// Burn's `expand` to a shape of the same rank. The caller checked that every dim is equal
    /// or `1` in `self`; a wrong rank is an error, not a panic.
    pub(crate) fn expand(self, shape: &[usize]) -> Result<Self> {
        if shape.len() != D {
            return Err(TensorError::UnsupportedRank { rank: shape.len() });
        }
        let shape = Shape::from(shape.to_vec());
        Ok(match self {
            Ranked::Float(t) => Ranked::Float(t.expand(shape)),
            Ranked::Int(t) => Ranked::Int(t.expand(shape)),
            Ranked::Bool(t) => Ranked::Bool(t.expand(shape)),
        })
    }
}

/// Maps a Burn read-back error to [`TensorError::Backend`].
pub(crate) fn backend_error(err: impl std::fmt::Debug) -> TensorError {
    TensorError::Backend {
        message: format!("{err:?}"),
    }
}

#[cfg(test)]
mod tests;
