//! Gradient reference tests of the shape ops against the PyTorch fixtures in
//! `fixtures/small/grads/` (graph-ir.md §3.1.11, testing-strategy.md §3.6). The harness is
//! `tests/reference/grads.rs`; the ops run through the forward test's runner
//! (`tests/reference/run_shape.rs`).
//!
//! ```text
//! cargo test -p polypous-backend --test grad_reference_shape                     # CPU
//! cargo test -p polypous-backend --features vulkan --test grad_reference_shape -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;
#[path = "reference/grads.rs"]
mod grads;
#[path = "reference/run_shape.rs"]
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
