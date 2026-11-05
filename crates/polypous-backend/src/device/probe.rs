//! Device probing: initialize, allocate, run a tiny kernel, compare the result (placement.md §1).

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::tensor::{Device, Tensor};

use super::discover::{lock_devices, panic_message};
use super::info::{DeviceId, DeviceStatus, UnavailableReason};
use super::resolve::resolve_device;

/// `[[1, 2], [3, 4]] @ [[5, 6], [7, 8]] + 1`. Small integers, exact in every float format a
/// backend might compute with (f32, tf32, f16), so a correct device gives exactly this.
const EXPECTED: [f32; 4] = [20.0, 23.0, 44.0, 51.0];

/// Probes one device in this process: initializes it, allocates two small tensors, runs a matmul
/// and an elementwise add, reads the result back and compares it with the expected values.
///
/// Never panics: errors, wrong results and Rust panics inside Burn/CubeCL (caught with
/// `catch_unwind`) become [`DeviceStatus::Unavailable`] with
/// [`UnavailableReason::ProbeFailed`]. The status says `isolated: false`; the prober process
/// (ADR-0013, #14) marks its results as isolated with [`DeviceStatus::with_isolated`]. Native
/// driver crashes and hangs cannot be caught here: that is why the prober runs this in a separate
/// process. A caught panic is still printed by the panic hook (to stderr).
///
/// An id that [`list_backend_devices`](super::list_backend_devices) doesn't list fails without
/// being opened (no driver is touched once the backend's list is cached).
///
/// The first probe of a GPU compiles kernels and can take seconds.
pub fn probe_device(id: DeviceId) -> DeviceStatus {
    // Checked before taking the device lock: enumeration takes it too.
    let device = resolve_device(id);
    let _guard = lock_devices();
    guarded(|| probe(&device?))
}

/// Runs a probe body, turning errors and panics into a status.
fn guarded(body: impl FnOnce() -> Result<(), String>) -> DeviceStatus {
    let failed = |message: String| DeviceStatus::Unavailable {
        reason: UnavailableReason::ProbeFailed { message },
        isolated: false,
    };
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => DeviceStatus::Available { isolated: false },
        Ok(Err(message)) => failed(message),
        Err(payload) => failed(format!("panic: {}", panic_message(&*payload))),
    }
}

fn probe(device: &Device) -> Result<(), String> {
    let a = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], device);
    let b = Tensor::<2>::from_floats([[5.0, 6.0], [7.0, 8.0]], device);
    let values = a
        .matmul(b)
        .add_scalar(1.0)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|err| format!("could not read the result back: {err:?}"))?;
    check_result(&values)
}

fn check_result(values: &[f32]) -> Result<(), String> {
    if values == EXPECTED {
        Ok(())
    } else {
        Err(format!("wrong result {values:?}, expected {EXPECTED:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panics_become_probe_failed() {
        let status = guarded(|| panic!("driver said no"));
        assert_eq!(
            status,
            DeviceStatus::Unavailable {
                reason: UnavailableReason::ProbeFailed {
                    message: "panic: driver said no".into()
                },
                isolated: false,
            }
        );
    }

    #[test]
    fn errors_become_probe_failed() {
        let status = guarded(|| Err("no memory".into()));
        assert_eq!(
            status.to_string(),
            "unavailable: probe failed: no memory, not isolated"
        );
        assert_eq!(
            guarded(|| Ok(())),
            DeviceStatus::Available { isolated: false }
        );
    }

    #[test]
    fn wrong_results_are_rejected() {
        assert!(check_result(&EXPECTED).is_ok());
        assert!(check_result(&[20.0, 23.0, 44.0, 50.0]).is_err());
        assert!(check_result(&[20.0, 23.0, 44.0]).is_err());
        assert!(check_result(&[f32::NAN; 4]).is_err());
    }

    #[test]
    fn missing_backend_is_unavailable() {
        for id in [DeviceId::Cuda { index: 0 }, DeviceId::Cpu] {
            if !id.backend().is_compiled_in() {
                assert!(!probe_device(id).is_available());
            }
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_probe_passes() {
        assert_eq!(
            probe_device(DeviceId::Cpu),
            DeviceStatus::Available { isolated: false }
        );
    }
}
