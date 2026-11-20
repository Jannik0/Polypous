//! Gradient reference tests of `core.gather` and `core.index_select` against the PyTorch fixtures in
//! `fixtures/small/grads/` (graph-ir.md §3.1.11, testing-strategy.md §3.6). The harness is
//! `tests/reference/grads.rs`; the ops run through the forward test's
//! runner (`tests/reference/run_indexing.rs`). The index checks (one host read-back) run on
//! tracked inputs too and leave the float graph intact.
//!
//! ```text
//! cargo test -p polypous-backend --test grad_reference_indexing                     # CPU
//! cargo test -p polypous-backend --features vulkan --test grad_reference_indexing -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;
#[path = "reference/grads.rs"]
mod grads;
#[path = "reference/run_indexing.rs"]
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
    core_gather => "core.gather",
    core_index_select => "core.index_select",
}
