//! Helpers shared by the optimizer tests: tensors, devices and the comparison rule of
//! testing-strategy.md §3.2.

#![allow(dead_code, reason = "each test binary uses a different part")]

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorData};

pub fn f32_tensor(shape: &[usize], values: &[f32], device: DeviceId) -> DynTensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape = shape.iter().map(|&d| d as u64).collect();
    DynTensor::from_data(&TensorData::new(DType::F32, shape, bytes).unwrap(), device).unwrap()
}

/// The f32 values of a tensor.
pub fn read_f32(t: &DynTensor) -> Vec<f32> {
    let data = t.to_data().unwrap();
    assert_eq!(data.dtype, DType::F32);
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

/// Deterministic values in `[-scale, scale)` (a small LCG), different per `seed`.
pub fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The GPU devices of the compiled-in backends (lavapipe only with
/// `POLYPOUS_TEST_SOFTWARE_VULKAN=1`). Fails if a GPU test runs without one.
pub fn gpu_devices() -> Vec<DeviceId> {
    use polypous_backend::device::{BackendKind, EnumerateOptions, list_devices};
    let listed: Vec<DeviceId> = list_devices()
        .into_iter()
        .filter(|id| id.backend() != BackendKind::Cpu)
        .collect();
    assert!(
        !listed.is_empty(),
        "no GPU device found for the compiled-in GPU backends"
    );
    let options = EnumerateOptions {
        software_rasterizers: std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN")
            .is_ok_and(|v| v == "1"),
    };
    let (skipped, devices): (Vec<DeviceId>, Vec<DeviceId>) =
        listed.into_iter().partition(|&id| options.skips(id));
    for id in skipped {
        eprintln!(
            "SKIP device {id}: software rasterizer (set POLYPOUS_TEST_SOFTWARE_VULKAN=1 to test it)"
        );
    }
    devices
}

/// The distance in ulps between two finite f32 values.
pub fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let bits = x.to_bits();
        if bits >> 31 == 1 {
            -i64::from(bits & 0x7FFF_FFFF)
        } else {
            i64::from(bits)
        }
    };
    key(a).abs_diff(key(b))
}

/// The worst error of a comparison: absolute, in ulps, and as a fraction of the allowed error.
#[derive(Clone, Copy, Debug, Default)]
pub struct Worst {
    pub abs: f64,
    pub ulp: u64,
    pub ratio: f64,
}

impl Worst {
    pub fn merge(&mut self, other: Worst) {
        self.abs = self.abs.max(other.abs);
        self.ulp = self.ulp.max(other.ulp);
        self.ratio = self.ratio.max(other.ratio);
    }
}

/// Compares under testing-strategy.md §3.2's rule: NaN exactly where expected, an expected inf
/// must be the same inf, every other element within `atol + rtol · |expected|`. Panics (a test
/// failure) naming `what` and the first element that breaks it.
pub fn assert_close(what: &str, got: &[f32], want: &[f32], atol: f64, rtol: f64) -> Worst {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    let mut worst = Worst::default();
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{what}[{i}]: got {g}, expected NaN");
            continue;
        }
        assert!(!g.is_nan(), "{what}[{i}]: got NaN, expected {w}");
        if w.is_infinite() {
            assert_eq!(g, w, "{what}[{i}]");
            continue;
        }
        let err = (f64::from(g) - f64::from(w)).abs();
        let allowed = atol + rtol * f64::from(w).abs();
        assert!(
            err <= allowed,
            "{what}[{i}]: got {g:e}, expected {w:e} (|error| {err:e}, allowed {allowed:e}, {} ulp)",
            ulps(g, w)
        );
        let ratio = if allowed > 0.0 { err / allowed } else { 0.0 };
        worst.merge(Worst {
            abs: err,
            ulp: ulps(g, w),
            ratio,
        });
    }
    worst
}
