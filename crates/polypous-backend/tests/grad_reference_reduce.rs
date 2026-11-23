//! Gradient reference tests of the reductions, `matmul`, `softmax`, `log_softmax`, `cumsum` and
//! `top_k` (`core.sum mean max min matmul softmax log_softmax cumsum top_k`; `top_k`: `values`
//! only) against the PyTorch fixtures in `fixtures/small/grads/` (graph-ir.md §3.1.11,
//! testing-strategy.md §3.6). The harness is `tests/reference/grads.rs`; the ops run through the
//! forward test's runner (`tests/reference/run_reduce.rs`). `argmax` and `argmin` are not
//! differentiable and have no gradient fixtures.
//!
//! ```text
//! cargo test -p polypous-backend --test grad_reference_reduce                     # CPU
//! cargo test -p polypous-backend --features vulkan --test grad_reference_reduce -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;
#[path = "reference/grads.rs"]
mod grads;
#[path = "reference/run_reduce.rs"]
mod runner;

use runner::run;

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
    core_sum => "core.sum",
    core_mean => "core.mean",
    core_max => "core.max",
    core_min => "core.min",
    core_matmul => "core.matmul",
    core_softmax => "core.softmax",
    core_log_softmax => "core.log_softmax",
    core_cumsum => "core.cumsum",
    core_top_k => "core.top_k",
}
