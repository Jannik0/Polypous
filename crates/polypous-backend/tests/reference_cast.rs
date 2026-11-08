//! Reference tests of `core.cast` against the PyTorch fixtures in `fixtures/small/ops/core.cast/`
//! (graph-ir.md §3.2 and §3.1.8, testing-strategy.md §6). The harness is
//! `tests/reference/common.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

use common::Case;
use polypous_backend::{DynTensor, TensorError};

fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    let [x]: [DynTensor; 1] = inputs.try_into().expect("cast has one input");
    Ok(vec![x.cast(case.dtype("dtype"))?])
}

#[cfg(feature = "cpu")]
#[test]
fn core_cast_cpu() {
    common::run_cpu("core.cast", run);
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn core_cast_gpu() {
    common::run_gpu("core.cast", run);
}
