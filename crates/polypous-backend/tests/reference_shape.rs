//! Reference tests of the shape ops (`core.reshape flatten permute transpose squeeze unsqueeze
//! expand concat split slice pad`) against the PyTorch fixtures in `fixtures/small/ops/`
//! (graph-ir.md §3.2, testing-strategy.md §6). The harness is `tests/reference/common.rs`.
//!
//! The sidecars hold the attributes as written in a graph (negative axes, `-1`, negative `slice`
//! bounds, defaults left out). [`run`] normalizes them the way `polypous_ir::Analysis` does
//! before the runtime calls the backend, and cases named `permuted_…`, `transposed_…`,
//! `expanded_…` or `strided_…` get their first input as a non-contiguous view
//! (`common::feed_layout`).
//!
//! ```text
//! cargo test -p polypous-backend --test reference_shape                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_shape -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

use common::Case;
use polypous_backend::{DType, DynTensor, Scalar, TensorError};

/// An axis attribute normalized for rank `rank` (graph-ir.md §3.1.4).
fn axis(case: &Case, value: i64, rank: usize) -> usize {
    let rank = rank as i64;
    let normalized = if value < 0 { value + rank } else { value };
    assert!(
        (0..rank).contains(&normalized),
        "{}/{}: axis {value} out of range for rank {rank}",
        case.op,
        case.name
    );
    normalized as usize
}

fn required_axis(case: &Case, name: &str, rank: usize) -> usize {
    let value = case
        .int(name)
        .unwrap_or_else(|| panic!("{}/{}: missing {name}", case.op, case.name));
    axis(case, value, rank)
}

/// The zero of a dtype: the default `pad` value (graph-ir.md §3.2 `pad` note).
fn zero(dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(false),
        d if d.is_int() => Scalar::Int(0),
        _ => Scalar::Float(0.0),
    }
}

/// `reshape`'s target with the `-1` entry resolved.
fn reshape_target(case: &Case, x: &DynTensor) -> Vec<usize> {
    let shape = case.ints("shape").expect("reshape shape");
    let count: usize = x.shape().iter().product();
    let known: usize = shape
        .iter()
        .filter(|&&d| d != -1)
        .map(|&d| d as usize)
        .product();
    shape
        .iter()
        .map(|&d| if d == -1 { count / known } else { d as usize })
        .collect()
}

/// `expand`'s target with `-1` entries replaced by the input dim (right-aligned).
fn expand_target(case: &Case, x: &DynTensor) -> Vec<usize> {
    let shape = case.ints("shape").expect("expand shape");
    let input = x.shape();
    let offset = shape.len() - input.len();
    shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == -1 {
                input[i - offset]
            } else {
                d as usize
            }
        })
        .collect()
}

/// A `slice` bound resolved like the IR: negative literals count from the end, then clamped to `n`.
fn slice_bound(value: i64, n: usize) -> usize {
    let n = n as i64;
    let v = if value < 0 { n + value } else { value };
    v.clamp(0, n) as usize
}

fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    let mut inputs = common::feed_layout(case, inputs)?;
    if case.op == "core.concat" {
        let dim = required_axis(case, "dim", inputs[0].rank());
        return Ok(vec![DynTensor::concat(&inputs, dim)?]);
    }
    assert_eq!(inputs.len(), 1, "{}/{}: one input", case.op, case.name);
    let x = inputs.remove(0);
    let rank = x.rank();
    let out = match case.op.as_str() {
        "core.reshape" => {
            let target = reshape_target(case, &x);
            x.reshape(&target)?
        }
        "core.flatten" => {
            let start = axis(case, case.int("start_dim").unwrap_or(0), rank);
            let end = axis(case, case.int("end_dim").unwrap_or(-1), rank);
            x.flatten(start, end)?
        }
        "core.permute" => {
            let dims: Vec<usize> = case
                .ints("dims")
                .expect("permute dims")
                .into_iter()
                .map(|d| axis(case, d, rank))
                .collect();
            x.permute(&dims)?
        }
        "core.transpose" => {
            let dim0 = required_axis(case, "dim0", rank);
            let dim1 = required_axis(case, "dim1", rank);
            x.transpose(dim0, dim1)?
        }
        "core.squeeze" => {
            let dims: Vec<usize> = match case.ints("dim") {
                Some(dims) => dims.into_iter().map(|d| axis(case, d, rank)).collect(),
                None => (0..rank).filter(|&d| x.shape()[d] == 1).collect(),
            };
            x.squeeze(&dims)?
        }
        "core.unsqueeze" => {
            let dim = axis(case, case.int("dim").expect("unsqueeze dim"), rank + 1);
            x.unsqueeze(dim)?
        }
        "core.expand" => {
            let target = expand_target(case, &x);
            x.expand(&target)?
        }
        "core.split" => {
            let dim = required_axis(case, "dim", rank);
            let sizes: Vec<usize> = case
                .ints("sizes")
                .expect("split sizes")
                .into_iter()
                .map(|s| s as usize)
                .collect();
            return x.split(dim, &sizes);
        }
        "core.slice" => {
            let dim = required_axis(case, "dim", rank);
            let n = x.shape()[dim];
            let start = slice_bound(case.int("start").unwrap_or(0), n);
            let end = case.int("end").map_or(n, |e| slice_bound(e, n));
            let step = case.int("step").unwrap_or(1) as usize;
            x.slice(dim, start, end, step)?
        }
        "core.pad" => {
            if let Some(mode) = case.attributes.get("mode") {
                assert_eq!(mode, "constant", "{}/{}: pad mode", case.op, case.name);
            }
            let pad: Vec<usize> = case
                .ints("pad")
                .expect("pad widths")
                .into_iter()
                .map(|p| p as usize)
                .collect();
            let value = case.scalar("value").unwrap_or(zero(x.dtype()));
            x.pad(&pad, value)?
        }
        other => panic!("no runner for {other}"),
    };
    Ok(vec![out])
}

/// One CPU test and one GPU test (`#[ignore = "gpu"]`) per op.
macro_rules! reference_tests {
    ($($test:ident => $op:literal),* $(,)?) => {
        #[cfg(feature = "cpu")]
        mod cpu {
            $(
                #[test]
                fn $test() {
                    super::common::run_cpu($op, super::run);
                }
            )*
        }

        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        mod gpu {
            $(
                #[test]
                #[ignore = "gpu"]
                fn $test() {
                    super::common::run_gpu($op, super::run);
                }
            )*
        }
    };
}

reference_tests! {
    core_reshape => "core.reshape",
    core_flatten => "core.flatten",
    core_permute => "core.permute",
    core_transpose => "core.transpose",
    core_squeeze => "core.squeeze",
    core_unsqueeze => "core.unsqueeze",
    core_expand => "core.expand",
    core_concat => "core.concat",
    core_split => "core.split",
    core_slice => "core.slice",
    core_pad => "core.pad",
}

#[test]
fn layouts_are_chosen_by_prefix() {
    use common::Layout;
    assert_eq!(Layout::of("permuted_rank3_f32"), Layout::Permuted);
    assert_eq!(Layout::of("transposed_i64"), Layout::Transposed);
    assert_eq!(Layout::of("expanded_bool"), Layout::Expanded);
    assert_eq!(Layout::of("strided_f16"), Layout::Strided);
    assert_eq!(Layout::of("rank1_f32"), Layout::Contiguous);
    assert_eq!(Layout::of("my_permuted_f32"), Layout::Contiguous);
}
