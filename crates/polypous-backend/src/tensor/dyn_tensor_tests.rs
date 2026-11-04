//! Tests for [`DynTensor`]. CPU tests use Burn's `flex` backend; the GPU test is
//! `#[ignore = "gpu"]` and runs on every GPU device that `list_devices()` returns, except software
//! rasterizers unless `POLYPOUS_TEST_SOFTWARE_VULKAN=1` is set.
// The tests need a CPU device: Burn's `flex` backend, the default `cpu` feature.
#![cfg(feature = "cpu")]

use burn::tensor::{bf16, f16};

use super::*;
use crate::device::{AdapterKind, BackendKind, list_devices};

/// Shapes used for the round-trip test: the first `rank` entries of `[2, 3, 1, 4, 2, 2]`.
const DIMS: [u64; 6] = [2, 3, 1, 4, 2, 2];

fn cpu() -> DeviceId {
    DeviceId::Cpu
}

/// Encodes `values` as little-endian elements of `dtype`. Values must be exact in `dtype`.
fn encode(dtype: DType, values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * dtype.size_bytes());
    for &v in values {
        match dtype {
            DType::F32 => bytes.extend((v as f32).to_le_bytes()),
            DType::F16 => bytes.extend(f16::from_f64(v).to_le_bytes()),
            DType::BF16 => bytes.extend(bf16::from_f64(v).to_le_bytes()),
            DType::I64 => bytes.extend((v as i64).to_le_bytes()),
            DType::I32 => bytes.extend((v as i32).to_le_bytes()),
            DType::Bool => bytes.push(u8::from(v != 0.0)),
        }
    }
    bytes
}

/// Decodes little-endian elements of `data` as `f64`.
fn decode(data: &TensorData) -> Vec<f64> {
    let width = data.dtype.size_bytes();
    data.bytes
        .chunks_exact(width)
        .map(|b| match data.dtype {
            DType::F32 => f64::from(f32::from_le_bytes(b.try_into().unwrap())),
            DType::F16 => f16::from_le_bytes(b.try_into().unwrap()).to_f64(),
            DType::BF16 => bf16::from_le_bytes(b.try_into().unwrap()).to_f64(),
            DType::I64 => i64::from_le_bytes(b.try_into().unwrap()) as f64,
            DType::I32 => f64::from(i32::from_le_bytes(b.try_into().unwrap())),
            DType::Bool => f64::from(b[0]),
        })
        .collect()
}

fn data(dtype: DType, shape: &[u64], values: &[f64]) -> TensorData {
    TensorData::new(dtype, shape.to_vec(), encode(dtype, values)).unwrap()
}

fn tensor(dtype: DType, shape: &[u64], values: &[f64], device: DeviceId) -> DynTensor {
    DynTensor::from_data(&data(dtype, shape, values), device).unwrap()
}

/// Distinct values that are exact in every dtype (bool: alternating, with a few `true`s).
fn sample_values(dtype: DType, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| match dtype {
            DType::Bool => f64::from(u8::from(i % 3 != 1)),
            _ if dtype.is_float() => i as f64 * 0.5 - 7.0,
            _ => i as f64 * 3.0 - 20.0,
        })
        .collect()
}

/// Every dtype the device supports, by asking Burn directly.
fn supported_dtypes(device: DeviceId) -> Vec<DType> {
    let device = resolve_device(device).unwrap();
    DType::ALL
        .into_iter()
        .filter(|dtype| *dtype == DType::Bool || device.supports_dtype(dtype.to_burn()))
        .collect()
}

fn round_trip_all(device: DeviceId) {
    for dtype in supported_dtypes(device) {
        for rank in 1..=DynTensor::MAX_RANK {
            let shape = &DIMS[..rank];
            let count = shape.iter().product::<u64>() as usize;
            let input = data(dtype, shape, &sample_values(dtype, count));

            let t = DynTensor::from_data(&input, device).unwrap();
            assert_eq!(t.dtype(), dtype);
            assert_eq!(t.rank(), rank);
            let expected_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            assert_eq!(t.shape(), expected_shape);

            let output = t.into_data().unwrap();
            assert_eq!(
                output, input,
                "round trip of {dtype} rank {rank} on {device}"
            );
        }
    }
}

#[test]
fn round_trip_every_dtype_and_rank_on_cpu() {
    let device = cpu();
    // flex supports every value dtype, so this covers all of them.
    assert_eq!(supported_dtypes(device), DType::ALL.to_vec());
    round_trip_all(device);
}

#[test]
fn round_trip_empty_tensor() {
    let input = TensorData::new(DType::F32, vec![2, 0, 3], vec![]).unwrap();
    let t = DynTensor::from_data(&input, cpu()).unwrap();
    assert_eq!(t.shape(), vec![2, 0, 3]);
    assert_eq!(t.into_data().unwrap(), input);
}

#[test]
fn round_trip_preserves_special_float_values() {
    let values = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = TensorData::new(DType::F32, vec![5], bytes).unwrap();
    let output = DynTensor::from_data(&input, cpu())
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(output, input);
}

/// `[[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]] = [[58, 64], [139, 154]]`, computed by
/// hand: 1·7 + 2·9 + 3·11 = 58, 1·8 + 2·10 + 3·12 = 64, 4·7 + 5·9 + 6·11 = 139,
/// 4·8 + 5·10 + 6·12 = 154.
fn check_matmul_2d(device: DeviceId, dtype: DType) {
    let a = tensor(dtype, &[2, 3], &[1., 2., 3., 4., 5., 6.], device);
    let b = tensor(dtype, &[3, 2], &[7., 8., 9., 10., 11., 12.], device);
    let c = a.matmul(b).unwrap();
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.dtype(), dtype);
    assert_eq!(decode(&c.into_data().unwrap()), vec![58., 64., 139., 154.]);
}

#[test]
fn matmul_2d_matches_hand_computed_result() {
    // All products and sums are small integers, exact in f16 and bf16 too.
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        check_matmul_2d(cpu(), dtype);
    }
}

#[test]
fn matmul_batched_with_broadcast() {
    let device = cpu();
    // Batch 0: identity, batch 1: 2·identity, applied to the same rhs (batch dim 1, broadcast).
    let a = tensor(
        DType::F32,
        &[2, 2, 2],
        &[1., 0., 0., 1., 2., 0., 0., 2.],
        device,
    );
    let b = tensor(DType::F32, &[1, 2, 2], &[1., 2., 3., 4.], device);
    let c = a.matmul(b).unwrap();
    assert_eq!(c.shape(), vec![2, 2, 2]);
    assert_eq!(
        decode(&c.into_data().unwrap()),
        vec![1., 2., 3., 4., 2., 4., 6., 8.]
    );
}

#[test]
fn matmul_vector_shapes() {
    // [1, 3] @ [3, 1]: dot product 1·4 + 2·5 + 3·6 = 32.
    let device = cpu();
    let a = tensor(DType::F32, &[1, 3], &[1., 2., 3.], device);
    let b = tensor(DType::F32, &[3, 1], &[4., 5., 6.], device);
    let c = a.matmul(b).unwrap();
    assert_eq!(c.shape(), vec![1, 1]);
    assert_eq!(decode(&c.into_data().unwrap()), vec![32.]);
}

#[test]
fn add_same_shape_and_broadcast() {
    let device = cpu();
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I64, DType::I32] {
        let a = tensor(dtype, &[2, 3], &[1., 2., 3., 4., 5., 6.], device);
        let b = tensor(dtype, &[2, 3], &[10., 20., 30., 40., 50., 60.], device);
        let c = a.clone().add(b).unwrap();
        assert_eq!(c.dtype(), dtype);
        assert_eq!(
            decode(&c.into_data().unwrap()),
            vec![11., 22., 33., 44., 55., 66.]
        );

        // Row vector [1, 3] broadcast over both rows. All sums are exact in bf16 (< 256).
        let row = tensor(dtype, &[1, 3], &[100., 120., 140.], device);
        let c = a.add(row).unwrap();
        assert_eq!(c.shape(), vec![2, 3]);
        assert_eq!(
            decode(&c.into_data().unwrap()),
            vec![101., 122., 143., 104., 125., 146.]
        );
    }
}

#[test]
fn reshape_keeps_row_major_order() {
    let device = cpu();
    let values: Vec<f64> = (0..24).map(f64::from).collect();
    let t = tensor(DType::I32, &[2, 3, 4], &values, device);
    let r = t.reshape(&[4, 6]).unwrap();
    assert_eq!(r.shape(), vec![4, 6]);
    let r = r.reshape(&[1, 2, 1, 3, 2, 2]).unwrap();
    assert_eq!(r.rank(), 6);
    let r = r.reshape(&[24]).unwrap();
    assert_eq!(decode(&r.into_data().unwrap()), values);
}

#[test]
fn cast_between_dtypes() {
    let device = cpu();
    let f = tensor(DType::F32, &[4], &[-1.5, 0.0, 2.75, 3.0], device);

    let i = f.clone().cast(DType::I32).unwrap();
    assert_eq!(i.dtype(), DType::I32);
    // Rounds toward zero.
    assert_eq!(decode(&i.to_data().unwrap()), vec![-1., 0., 2., 3.]);

    let b = f.clone().cast(DType::Bool).unwrap();
    assert_eq!(b.dtype(), DType::Bool);
    assert_eq!(b.to_data().unwrap().bytes, vec![1, 0, 1, 1]);

    let back = b.cast(DType::F16).unwrap();
    assert_eq!(decode(&back.into_data().unwrap()), vec![1., 0., 1., 1.]);

    let h = f.cast(DType::BF16).unwrap();
    assert_eq!(decode(&h.to_data().unwrap()), vec![-1.5, 0.0, 2.75, 3.0]);

    let l = i.cast(DType::I64).unwrap();
    assert_eq!(decode(&l.to_data().unwrap()), vec![-1., 0., 2., 3.]);
    let lb = l.cast(DType::Bool).unwrap();
    assert_eq!(lb.to_data().unwrap().bytes, vec![1, 0, 1, 1]);

    // Same dtype: no-op.
    let same = tensor(DType::I64, &[1], &[5.], device)
        .cast(DType::I64)
        .unwrap();
    assert_eq!(decode(&same.into_data().unwrap()), vec![5.]);
}

#[test]
fn to_device_on_same_backend_keeps_data() {
    let device = cpu();
    let input = data(DType::I64, &[2, 2], &[1., -2., 3., -4.]);
    let t = DynTensor::from_data(&input, device).unwrap();
    let moved = t.to_device(device).unwrap();
    assert_eq!(moved.into_data().unwrap(), input);
}

/// `from_data(&data, id)?.device() == id`, and the Burn device underneath maps back to `id`.
fn assert_round_trips_device_id(id: DeviceId) {
    let t = tensor(DType::F32, &[2], &[1., 2.], id);
    assert_eq!(t.device(), id);
    assert_eq!(device_id_of(&t.burn_device()), Some(id));
    let moved = t.to_device(id).unwrap();
    assert_eq!(moved.device(), id);
    // Results of operations stay on the device.
    let sum = moved.clone().add(moved).unwrap();
    assert_eq!(sum.device(), id);
    assert_eq!(
        sum.reshape(&[1, 2])
            .unwrap()
            .cast(DType::I32)
            .unwrap()
            .device(),
        id
    );
}

#[test]
fn device_id_round_trips_on_cpu() {
    assert!(list_devices().contains(&DeviceId::Cpu));
    assert_round_trips_device_id(DeviceId::Cpu);
}

// ---------------------------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------------------------

#[test]
fn rank_zero_and_above_six_are_rejected() {
    let device = cpu();
    let seven = TensorData::new(DType::F32, vec![1; 7], vec![0; 4]).unwrap();
    assert_eq!(
        DynTensor::from_data(&seven, device).unwrap_err(),
        TensorError::UnsupportedRank { rank: 7 }
    );
    let scalar = TensorData::new(DType::F32, vec![], vec![0; 4]).unwrap();
    assert_eq!(
        DynTensor::from_data(&scalar, device).unwrap_err(),
        TensorError::UnsupportedRank { rank: 0 }
    );

    let t = tensor(DType::F32, &[1], &[1.], device);
    assert_eq!(
        t.clone().reshape(&[1; 7]).unwrap_err(),
        TensorError::UnsupportedRank { rank: 7 }
    );
    assert_eq!(
        t.reshape(&[]).unwrap_err(),
        TensorError::UnsupportedRank { rank: 0 }
    );
}

#[test]
fn malformed_data_is_rejected() {
    let device = cpu();
    // `TensorData` has public fields, so `from_data` validates again.
    let short = TensorData {
        dtype: DType::F32,
        shape: vec![2, 2],
        bytes: vec![0; 15],
    };
    assert!(matches!(
        DynTensor::from_data(&short, device),
        Err(TensorError::InvalidData { .. })
    ));
    let bad_bool = TensorData {
        dtype: DType::Bool,
        shape: vec![2],
        bytes: vec![0, 7],
    };
    assert!(matches!(
        DynTensor::from_data(&bad_bool, device),
        Err(TensorError::InvalidData { .. })
    ));
}

#[test]
fn dtype_mismatch_is_rejected() {
    let device = cpu();
    let a = tensor(DType::F32, &[2, 2], &[1.; 4], device);
    let b = tensor(DType::F16, &[2, 2], &[1.; 4], device);
    let expected = |op| TensorError::DTypeMismatch {
        op,
        lhs: DType::F32,
        rhs: DType::F16,
    };
    assert_eq!(a.clone().add(b.clone()).unwrap_err(), expected("add"));
    assert_eq!(a.matmul(b).unwrap_err(), expected("matmul"));
}

#[test]
fn shape_mismatch_is_rejected() {
    let device = cpu();
    let a = tensor(DType::F32, &[2, 3], &[1.; 6], device);
    let b = tensor(DType::F32, &[3, 2], &[1.; 6], device);
    let c = tensor(DType::F32, &[2, 3], &[1.; 6], device);
    let v = tensor(DType::F32, &[6], &[1.; 6], device);
    let batched = tensor(DType::F32, &[2, 2, 3], &[1.; 12], device);
    let other_batch = tensor(DType::F32, &[3, 3, 2], &[1.; 18], device);

    // add: incompatible dims, different ranks.
    assert!(matches!(
        a.clone().add(b.clone()),
        Err(TensorError::ShapeMismatch { op: "add", .. })
    ));
    assert!(matches!(
        a.clone().add(v.clone()),
        Err(TensorError::ShapeMismatch { op: "add", .. })
    ));
    // matmul: inner dims differ, rank 1, batch dims incompatible.
    assert_eq!(
        a.clone().matmul(c).unwrap_err(),
        TensorError::ShapeMismatch {
            op: "matmul",
            lhs: vec![2, 3],
            rhs: vec![2, 3]
        }
    );
    assert!(matches!(
        v.clone().matmul(v.clone()),
        Err(TensorError::ShapeMismatch { op: "matmul", .. })
    ));
    assert!(matches!(
        batched.matmul(other_batch),
        Err(TensorError::ShapeMismatch { op: "matmul", .. })
    ));
    // reshape: element count differs.
    assert_eq!(
        a.reshape(&[4, 2]).unwrap_err(),
        TensorError::ShapeMismatch {
            op: "reshape",
            lhs: vec![2, 3],
            rhs: vec![4, 2]
        }
    );
}

#[test]
fn unsupported_operations_are_rejected() {
    let device = cpu();
    let b = tensor(DType::Bool, &[2, 2], &[1., 0., 1., 0.], device);
    assert_eq!(
        b.clone().add(b.clone()).unwrap_err(),
        TensorError::UnsupportedOperation {
            op: "add",
            dtype: DType::Bool
        }
    );
    assert_eq!(
        b.clone().matmul(b).unwrap_err(),
        TensorError::UnsupportedOperation {
            op: "matmul",
            dtype: DType::Bool
        }
    );
    let i = tensor(DType::I64, &[2, 2], &[1.; 4], device);
    assert_eq!(
        i.clone().matmul(i).unwrap_err(),
        TensorError::UnsupportedOperation {
            op: "matmul",
            dtype: DType::I64
        }
    );
}

/// Ids whose backend isn't compiled in, and ids that address no device of a compiled-in backend
/// (index past the last device, past CubeCL's `u16` range, or a second software adapter).
fn unavailable_ids() -> Vec<DeviceId> {
    let beyond = usize::from(u16::MAX) + 1;
    let mut ids = vec![
        DeviceId::Cuda { index: 4096 },
        DeviceId::Cuda { index: beyond },
        DeviceId::Cuda { index: usize::MAX },
        DeviceId::Vulkan {
            adapter: AdapterKind::Cpu,
            index: 1,
        },
    ];
    for adapter in [
        AdapterKind::DiscreteGpu,
        AdapterKind::IntegratedGpu,
        AdapterKind::VirtualGpu,
        AdapterKind::Other,
    ] {
        for index in [4096, beyond, usize::MAX] {
            ids.push(DeviceId::Vulkan { adapter, index });
            ids.push(DeviceId::Metal { adapter, index });
        }
    }
    let listed = list_devices();
    ids.retain(|id| !listed.contains(id));
    ids
}

#[test]
fn unavailable_devices_are_errors_not_panics() {
    let input = data(DType::F32, &[2], &[1., 2.]);
    let on_cpu = DynTensor::from_data(&input, cpu()).unwrap();
    for id in unavailable_ids() {
        let compiled_in = id.backend().is_compiled_in();
        for result in [DynTensor::from_data(&input, id), on_cpu.to_device(id)] {
            match result {
                Err(TensorError::DeviceUnavailable { device, reason }) => {
                    assert_eq!(device, id);
                    if compiled_in {
                        assert!(reason.starts_with("no such "), "{id}: {reason}");
                    } else {
                        assert_eq!(
                            reason,
                            format!("the {} backend is not compiled in", id.backend())
                        );
                    }
                }
                other => panic!("{id}: expected DeviceUnavailable, got {other:?}"),
            }
        }
    }
}

#[test]
fn backend_not_compiled_in_message() {
    let Some(backend) = BackendKind::ALL
        .into_iter()
        .find(|backend| !backend.is_compiled_in())
    else {
        return;
    };
    let id = match backend {
        BackendKind::Cuda => DeviceId::Cuda { index: 0 },
        BackendKind::Metal => DeviceId::Metal {
            adapter: AdapterKind::IntegratedGpu,
            index: 0,
        },
        BackendKind::Vulkan => DeviceId::Vulkan {
            adapter: AdapterKind::DiscreteGpu,
            index: 0,
        },
        BackendKind::Cpu => DeviceId::Cpu,
    };
    let err = DynTensor::from_data(&data(DType::F32, &[1], &[1.]), id).unwrap_err();
    assert_eq!(
        err.to_string(),
        format!("device {id} is not available: the {backend} backend is not compiled in")
    );
}

#[test]
fn errors_have_readable_messages() {
    let err = TensorError::UnsupportedDType {
        dtype: DType::BF16,
        device: "vulkan:0".into(),
    };
    assert_eq!(
        err.to_string(),
        "dtype bf16 is not supported on device vulkan:0"
    );
    let err = TensorError::UnsupportedRank { rank: 7 };
    assert_eq!(
        err.to_string(),
        "tensor rank 7 is not supported (supported ranks are 1 to 6)"
    );
}

// ---------------------------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------------------------

/// Every device of every compiled-in GPU backend, from the public device list.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
fn gpu_devices() -> Vec<DeviceId> {
    list_devices()
        .into_iter()
        .filter(|id| id.backend() != BackendKind::Cpu)
        .collect()
}

/// Discovery options for the GPU tests: software rasterizers (lavapipe, …) are skipped as in
/// discovery's default, unless the test-only environment variable
/// `POLYPOUS_TEST_SOFTWARE_VULKAN=1` is set (the CI lavapipe job sets it).
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
fn gpu_test_options() -> crate::device::EnumerateOptions {
    crate::device::EnumerateOptions {
        software_rasterizers: std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN")
            .is_ok_and(|v| v == "1"),
    }
}

/// CPU → GPU → CPU for every dtype and rank, plus ops on the GPU, on every GPU device that
/// `list_devices()` returns. A dtype the GPU doesn't support must fail with `UnsupportedDType`,
/// not panic.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn cpu_to_gpu_to_cpu() {
    let listed = gpu_devices();
    assert!(
        !listed.is_empty(),
        "no GPU device found for the compiled-in GPU backends"
    );
    let options = gpu_test_options();
    let (skipped, devices): (Vec<DeviceId>, Vec<DeviceId>) =
        listed.into_iter().partition(|&id| options.skips(id));
    for id in &skipped {
        eprintln!(
            "skipped {id}: software rasterizer (set POLYPOUS_TEST_SOFTWARE_VULKAN=1 to test it)"
        );
    }
    if devices.is_empty() {
        eprintln!("every GPU device is a skipped software rasterizer; nothing to test");
        return;
    }
    let host = cpu();
    for &gpu in &devices {
        let supported = supported_dtypes(gpu);
        let names: Vec<&str> = supported.iter().map(|d| d.name()).collect();
        eprintln!("{gpu}: supported dtypes {names:?}");

        for dtype in DType::ALL {
            for rank in 1..=DynTensor::MAX_RANK {
                let shape = &DIMS[..rank];
                let count = shape.iter().product::<u64>() as usize;
                let input = data(dtype, shape, &sample_values(dtype, count));
                let on_cpu = DynTensor::from_data(&input, host).unwrap();

                if !supported.contains(&dtype) {
                    assert!(matches!(
                        on_cpu.to_device(gpu),
                        Err(TensorError::UnsupportedDType { dtype: d, .. }) if d == dtype
                    ));
                    assert!(matches!(
                        DynTensor::from_data(&input, gpu),
                        Err(TensorError::UnsupportedDType { .. })
                    ));
                    continue;
                }

                let on_gpu = on_cpu.to_device(gpu).unwrap();
                assert_eq!(on_gpu.device(), gpu);
                assert_eq!(device_id_of(&on_gpu.burn_device()), Some(gpu));
                let back = on_gpu.to_device(host).unwrap();
                assert_eq!(back.device(), host);
                assert_eq!(
                    back.into_data().unwrap(),
                    input,
                    "{dtype} rank {rank} via {gpu}"
                );
            }
        }

        // Direct upload/readback and ops on the GPU itself.
        round_trip_all(gpu);
        assert_round_trips_device_id(gpu);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            if supported.contains(&dtype) {
                check_matmul_2d(gpu, dtype);
            }
        }
        let a = tensor(DType::F32, &[2, 2], &[1., 2., 3., 4.], gpu);
        let b = tensor(DType::F32, &[2, 2], &[1., 2., 3., 4.], host);
        assert_eq!(
            a.add(b).unwrap_err(),
            TensorError::DeviceMismatch {
                op: "add",
                lhs: gpu.to_string(),
                rhs: "cpu".into(),
            }
        );
        // float → int rounds toward zero on the GPU too.
        let f = tensor(DType::F32, &[4], &[-1.5, 0.0, 2.75, 3.0], gpu);
        let i = f.cast(DType::I32).unwrap();
        assert_eq!(decode(&i.to_data().unwrap()), vec![-1., 0., 2., 3.]);
        let b = i.cast(DType::Bool).unwrap();
        assert_eq!(b.to_data().unwrap().bytes, vec![1, 0, 1, 1]);
    }
}
