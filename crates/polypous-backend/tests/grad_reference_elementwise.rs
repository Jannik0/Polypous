//! Gradient reference tests of the `core` elementwise ops and `core.cast` against the PyTorch fixtures in
//! `fixtures/small/grads/` (graph-ir.md §3.1.11, testing-strategy.md §3.6). The harness is
//! `tests/reference/grads.rs`; the ops run through the forward test's runner
//! (`tests/reference/run_elementwise.rs`).
//!
//! ```text
//! cargo test -p polypous-backend --test grad_reference_elementwise                     # CPU
//! cargo test -p polypous-backend --features vulkan --test grad_reference_elementwise -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;
#[path = "reference/grads.rs"]
mod grads;
#[path = "reference/run_elementwise.rs"]
mod runner;

use common::Case;
use polypous_backend::{DynTensor, TensorError};

fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    if case.op == "core.cast" {
        let [x]: [DynTensor; 1] = inputs.try_into().expect("cast has one input");
        return Ok(vec![x.cast(case.dtype("dtype"))?]);
    }
    runner::run(case, inputs)
}

/// One CPU test and one GPU test (`#[ignore = "gpu"]`) per op.
macro_rules! grad_tests {
    ($($test:ident => $op:literal),* $(,)?) => {
        #[cfg(feature = "cpu")]
        mod cpu {
            $(
                #[test]
                fn $test() {
                    super::grads::run_grad_cpu($op, super::run);
                }
            )*
        }

        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        mod gpu {
            $(
                #[test]
                #[ignore = "gpu"]
                fn $test() {
                    super::grads::run_grad_gpu($op, super::run);
                }
            )*
        }
    };
}

grad_tests! {
    core_add => "core.add",
    core_sub => "core.sub",
    core_mul => "core.mul",
    core_div => "core.div",
    core_pow => "core.pow",
    core_maximum => "core.maximum",
    core_minimum => "core.minimum",
    core_neg => "core.neg",
    core_abs => "core.abs",
    core_exp => "core.exp",
    core_log => "core.log",
    core_sqrt => "core.sqrt",
    core_rsqrt => "core.rsqrt",
    core_sin => "core.sin",
    core_cos => "core.cos",
    core_tanh => "core.tanh",
    core_clamp => "core.clamp",
    core_where => "core.where",
    core_cast => "core.cast",
}
