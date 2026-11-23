//! Reference tests of the reductions, `matmul`, `softmax`, `log_softmax`, `cumsum` and `top_k`
//! (`core.sum mean max min argmax argmin matmul softmax log_softmax cumsum top_k`) against the
//! PyTorch fixtures in `fixtures/small/ops/` (graph-ir.md §3.2, testing-strategy.md §6). The
//! harness is `tests/reference/common.rs`.
//!
//! The ops run through the runner shared with the gradient test, `tests/reference/run_reduce.rs`,
//! which normalizes the sidecar attributes and builds the layout views.
//!
//! ```text
//! cargo test -p polypous-backend --test reference_reduce                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_reduce -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

#[path = "reference/run_reduce.rs"]
mod runner;

use runner::run;

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
