//! Reference tests of the reductions, `matmul`, `softmax`, `log_softmax`, `cumsum` and `top_k`
//! (`core.sum mean max min argmax argmin matmul softmax log_softmax cumsum top_k`) against the
//! PyTorch fixtures in `fixtures/small/ops/` (graph-ir.md §3.2, testing-strategy.md §6). The
//! harness is `tests/reference/common.rs`.
//!
//! The sidecars hold the attributes as written in a graph (negative axes, defaults left out).
//! [`run`] normalizes them the way `polypous_ir::Analysis` does before the runtime calls the
//! backend, and cases named `permuted_…`, `transposed_…`, `expanded_…` or `strided_…` get their
//! first input as a non-contiguous view (`common::feed_layout`).
//!
//! ```text
//! cargo test -p polypous-backend --test reference_reduce                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_reduce -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

use common::Case;
use polypous_backend::{DType, DynTensor, TensorError};

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

fn flag(case: &Case, name: &str, default: bool) -> bool {
    case.attributes
        .get(name)
        .map_or(default, |v| v.as_bool().expect("a bool attribute"))
}

/// The `dtype` attribute of `argmax`, `argmin` and `top_k`, default `i64`.
fn index_dtype(case: &Case) -> DType {
    if case.attributes.contains_key("dtype") {
        case.dtype("dtype")
    } else {
        DType::I64
    }
}

fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    let mut inputs = common::feed_layout(case, inputs)?;
    if case.op == "core.matmul" {
        assert_eq!(inputs.len(), 2, "{}/{}: two inputs", case.op, case.name);
        let b = inputs.pop().unwrap();
        let a = inputs.pop().unwrap();
        return Ok(vec![a.matmul(b)?]);
    }
    assert_eq!(inputs.len(), 1, "{}/{}: one input", case.op, case.name);
    let x = inputs.remove(0);
    let rank = x.rank();
    let keepdim = flag(case, "keepdim", false);
    let dim = |default: Option<i64>| {
        let value = case.int("dim").or(default).expect("a dim attribute");
        axis(case, value, rank)
    };
    let out = match case.op.as_str() {
        "core.sum" | "core.mean" | "core.max" | "core.min" => {
            let dims: Option<Vec<usize>> = case
                .ints("dim")
                .map(|dims| dims.into_iter().map(|d| axis(case, d, rank)).collect());
            let dims = dims.as_deref();
            match case.op.as_str() {
                "core.sum" => x.sum(dims, keepdim)?,
                "core.mean" => x.mean(dims, keepdim)?,
                "core.max" => x.max(dims, keepdim)?,
                _ => x.min(dims, keepdim)?,
            }
        }
        "core.argmax" => x.argmax(dim(None), keepdim, index_dtype(case))?,
        "core.argmin" => x.argmin(dim(None), keepdim, index_dtype(case))?,
        "core.cumsum" => x.cumsum(dim(None))?,
        "core.softmax" => x.softmax(dim(None))?,
        "core.log_softmax" => x.log_softmax(dim(None))?,
        "core.top_k" => {
            let k = usize::try_from(case.int("k").expect("k")).expect("k ≥ 0");
            let largest = flag(case, "largest", true);
            let (values, indices) = x.top_k(k, dim(Some(-1)), largest, index_dtype(case))?;
            return Ok(vec![values, indices]);
        }
        op => panic!("not an op of this test: {op}"),
    };
    Ok(vec![out])
}

macro_rules! reference_tests {
    ($($name:ident => $op:literal),* $(,)?) => {
        mod cpu {
            $(
                #[test]
                fn $name() {
                    super::common::run_cpu($op, super::run);
                }
            )*
        }

        /// Every GPU device of every compiled-in GPU backend; lavapipe only with
        /// `POLYPOUS_TEST_SOFTWARE_VULKAN=1`.
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        mod gpu {
            $(
                #[test]
                #[ignore = "gpu"]
                fn $name() {
                    super::common::run_gpu($op, super::run);
                }
            )*
        }
    };
}

reference_tests! {
    core_sum => "core.sum",
    core_mean => "core.mean",
    core_max => "core.max",
    core_min => "core.min",
    core_argmax => "core.argmax",
    core_argmin => "core.argmin",
    core_matmul => "core.matmul",
    core_softmax => "core.softmax",
    core_log_softmax => "core.log_softmax",
    core_cumsum => "core.cumsum",
    core_top_k => "core.top_k",
}
