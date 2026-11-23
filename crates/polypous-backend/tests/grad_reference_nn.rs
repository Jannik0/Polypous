//! Gradient reference tests of the `nn` ops (graph-ir.md §3.3: layers, normalization,
//! activations; `nn.embedding` to `weight`) against the PyTorch fixtures in
//! `fixtures/small/grads/` (graph-ir.md §3.1.11, testing-strategy.md §3.6), including the
//! training-mode forward passes of `nn.batch_norm` (with its updated running statistics) and
//! `nn.dropout` (with the case's keep mask). The harness is `tests/reference/grads.rs`; the ops run
//! through the forward test's runner (`tests/reference/run_nn.rs`).
//!
//! ```text
//! cargo test -p polypous-backend --test grad_reference_nn                     # CPU
//! cargo test -p polypous-backend --features vulkan --test grad_reference_nn -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;
#[path = "reference/grads.rs"]
mod grads;
#[path = "reference/run_nn.rs"]
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
    nn_linear => "nn.linear",
    nn_embedding => "nn.embedding",
    nn_conv1d => "nn.conv1d",
    nn_conv2d => "nn.conv2d",
    nn_max_pool2d => "nn.max_pool2d",
    nn_avg_pool2d => "nn.avg_pool2d",
    nn_adaptive_avg_pool2d => "nn.adaptive_avg_pool2d",
    nn_dropout => "nn.dropout",
    nn_batch_norm => "nn.batch_norm",
    nn_layer_norm => "nn.layer_norm",
    nn_group_norm => "nn.group_norm",
    nn_rms_norm => "nn.rms_norm",
    nn_relu => "nn.relu",
    nn_leaky_relu => "nn.leaky_relu",
    nn_gelu => "nn.gelu",
    nn_gelu_tanh => "nn.gelu_tanh",
    nn_silu => "nn.silu",
    nn_sigmoid => "nn.sigmoid",
    nn_softplus => "nn.softplus",
    nn_mish => "nn.mish",
}
