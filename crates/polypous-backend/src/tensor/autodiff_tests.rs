//! A crash reproduction with Burn's API (needs crate-internal access to Burn types). The autodiff
//! API tests are `tests/autodiff.rs`.
#![cfg(feature = "cpu")]

/// Reproduction of a Burn 0.22.1 / CubeCL crash on lavapipe (Mesa, Ubuntu; #75, docs/status.md
/// §4), with Burn's API only: after some GPU work, a tracked `x` through a NaN-masked comparison
/// (as `DynTensor::greater` builds it), `mask_where` and `select`, back-propagated, makes reading
/// `x`'s gradient crash with SIGSEGV inside lavapipe's compiled kernel code (3 of 3 runs). Without
/// the earlier work, without the `bool_or` of the two NaN masks, or with `gather` instead of
/// `select`, it passes. Not run by `--include-ignored` (it would take the test binary down): set
/// `POLYPOUS_REPRO_LVP_CRASH=1`.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn lavapipe_where_select_backward_crash() {
    use burn::tensor::{Int, Tensor, TensorData as BurnData};

    use crate::device::{BackendKind, list_devices};
    use crate::tensor::resolve;

    if std::env::var("POLYPOUS_REPRO_LVP_CRASH").is_err() {
        eprintln!("SKIP crash reproduction: set POLYPOUS_REPRO_LVP_CRASH=1 to run it");
        return;
    }
    for id in list_devices()
        .into_iter()
        .filter(|id| id.backend() != BackendKind::Cpu)
    {
        let device = resolve(id).unwrap();
        // Earlier GPU work: the backward passes of `tests/autodiff.rs`'s non-finite test.
        for _ in 0..20 {
            let x = Tensor::<1>::from_data(
                BurnData::new(vec![f32::NAN, f32::INFINITY, -0.0, 0.0, 1.0, -1.5], [6]),
                &device,
            )
            .autodiff()
            .require_grad();
            let y = x
                .clone()
                .sqrt()
                .recip()
                .max_pair(x.clone().clamp(-1.0, 1.0));
            let grads = y.sum().backward();
            let _ = x.grad(&grads).unwrap().into_data();
        }
        for _ in 0..3 {
            let x =
                Tensor::<1>::from_data(BurnData::new(vec![1.0f32, -2.0, 3.0, 0.5], [4]), &device)
                    .autodiff()
                    .require_grad();
            let zeros = x.zeros_like();
            let nan = x.clone().is_nan().bool_or(zeros.clone().is_nan());
            let positive = x.clone().greater(zeros.clone()).bool_and(nan.bool_not());
            let y = zeros.mask_where(positive, x.clone());
            let index = Tensor::<1, Int>::from_data(BurnData::new(vec![2i64, 2, 0], [3]), &device);
            let grads = y.select(0, index).sum().backward();
            let grad = x.grad(&grads).unwrap().into_data();
            assert_eq!(
                grad.try_to_vec::<f32>().unwrap(),
                vec![1.0, 0.0, 2.0, 0.0],
                "{id}"
            );
        }
        eprintln!("SUMMARY crash reproduction {id}: no crash");
    }
}
