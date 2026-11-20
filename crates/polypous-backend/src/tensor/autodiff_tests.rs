//! Reproductions of a lavapipe (Mesa) crash with Burn's API (they need crate-internal access to
//! Burn types). The autodiff API tests are `tests/autodiff.rs`.
#![cfg(feature = "cpu")]

/// The crash as #75 found it (docs/status.md §4; cause found in #92): after some GPU work, a
/// tracked `x` through a NaN-masked comparison (as `DynTensor::greater` builds it), `mask_where`
/// and `select`, back-propagated, makes reading `x`'s gradient crash with SIGSEGV inside lavapipe's
/// compiled kernel code (3 of 3 runs). The faulting kernel is `select`'s backward, Burn's
/// `select_assign_kernel`; the masks and the earlier work only decide which stale data lies after
/// the index buffer (see [`lavapipe_select_assign_stale_index_crash`]). Not run by
/// `--include-ignored` (it would take the test binary down): set `POLYPOUS_REPRO_LVP_CRASH=1`.
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

/// The cause of the crash above, reduced (#92, docs/status.md §4): one `select_assign` (`Add`), no
/// autodiff and no other op. Lavapipe's JIT runs the rest of a loop body once more after the last
/// invocation has left the loop (Mesa's gallivm doesn't branch on an empty exec mask), and its
/// load from a uniform global address (`visit_load_global` in `lp_bld_nir_soa.c`) ignores the
/// mask. CubeCL's SPIR-V reaches buffers through buffer device addresses, which Vulkan's
/// robustness doesn't cover. So in that extra pass, Burn's `select_assign_kernel` reads
/// `indices[len]` (whatever lies after the index buffer) and then `tensor[that value]`. Its store
/// stays masked, so the values are right whenever the address is mapped. Here the index buffer
/// reuses a pool slot that held `0xFFC00000` (`-NaN`) after its three indices, and the second load
/// faults: SIGSEGV in 3 of 3 runs on Mesa 25.2.8, 26.0.3 and 26.0.8. With `0`, `1`, `3`, `4`,
/// `0x100` or `0x10000` there instead, it passes with the right values. A hardware GPU doesn't
/// execute loads for invocations that have left the loop; the expected values are checked there.
/// Not run by `--include-ignored`: set `POLYPOUS_REPRO_LVP_CRASH=1`.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn lavapipe_select_assign_stale_index_crash() {
    use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData as BurnData};

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
        // Leave the bits 0xFFC00000 in many 16-byte pool slots, so that the 12-byte index buffer
        // allocated next has them right after its last element.
        let stale = f32::from_bits(0xFFC0_0000);
        let junk: Vec<Tensor<1>> = (0..256)
            .map(|_| Tensor::<1>::from_data(BurnData::new(vec![stale; 4], [4]), &device) * 1.0)
            .collect();
        for tensor in junk {
            let _ = tensor.into_data();
        }
        let index = Tensor::<1, Int>::from_data(BurnData::new(vec![2i64, 2, 0], [3]), &device);
        let out = Tensor::<1>::zeros([4], &device).select_assign(
            0,
            index,
            Tensor::<1>::ones([3], &device),
            IndexingUpdateOp::Add,
        );
        assert_eq!(
            out.into_data().try_to_vec::<f32>().unwrap(),
            vec![1.0, 0.0, 2.0, 0.0],
            "{id}"
        );
        eprintln!("SUMMARY select_assign reproduction {id}: no crash");
    }
}
