//! From a [`DeviceId`] to the `burn::Device` that work runs on (placement.md §1), for the `tensor`
//! module and the functions that open devices. Crate-internal: no Burn type leaves the crate
//! (ADR-0007).
//!
//! # Why ids are checked against the enumeration
//!
//! [`burn_device`] builds a Burn device for any id of a compiled-in backend without opening it.
//! CubeCL opens the device lazily, on the first tensor, and **panics** if it doesn't exist
//! (cubecl-wgpu 0.11.3 `DeviceService::init`: "Unable to acquire wgpu device"; cubecl-cuda fails
//! creating the context). CubeCL also clamps indices to `u16::MAX` (cubecl-runtime
//! `device::index_id`), so an index above that would silently address another device. And trying
//! to open a Vulkan id makes CubeCL create a temporary wgpu instance, which races with Vulkan
//! compute on other threads (issue #26).
//!
//! Catching that panic with `catch_unwind` was rejected: it fires inside CubeCL's device registry,
//! which we can't prove is consistent afterwards, and the panic hook would still print it. Instead
//! an id is accepted only if Burn's enumeration lists it ([`list_backend_devices`], the same list
//! [`list_devices`](super::list_devices) returns), so Burn never sees an id that addresses no device.
//!
//! The enumeration runs once per backend and process and is cached ([`list_backend_devices`]), so
//! checking an id, listed or not, never touches the driver again. A device that disappears later
//! (an unplugged eGPU) fails inside the backend; CubeCL keeps a device's client for the whole
//! process too.

use burn::tensor::Device;

use super::discover::{EnumerationError, burn_device, list_backend_devices};
use super::info::DeviceId;

/// The Burn device for `id`, or why `id` can't be used: its backend isn't compiled in, or the
/// backend has no such device. Never panics; enumeration panics are caught by
/// [`list_backend_devices`]. Must not be called while holding the device lock (the first
/// enumeration of a backend takes it).
pub(crate) fn resolve_device(id: DeviceId) -> Result<Device, String> {
    let backend = id.backend();
    let device =
        burn_device(id).ok_or_else(|| format!("the {backend} backend is not compiled in"))?;
    check_listed(id, list_backend_devices(backend))?;
    Ok(device)
}

/// Whether `listed`, the enumeration of `id`'s backend, contains `id`. The error names the devices
/// that do exist.
pub(crate) fn check_listed(
    id: DeviceId,
    listed: Result<Vec<DeviceId>, EnumerationError>,
) -> Result<(), String> {
    let listed = listed.map_err(|err| err.to_string())?;
    if listed.contains(&id) {
        return Ok(());
    }
    let found = if listed.is_empty() {
        "none".to_string()
    } else {
        listed
            .iter()
            .map(DeviceId::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(format!(
        "no such {} device (devices found: {found})",
        id.backend()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{AdapterKind, BackendKind};

    const VULKAN_0: DeviceId = DeviceId::Vulkan {
        adapter: AdapterKind::DiscreteGpu,
        index: 0,
    };

    #[test]
    fn listed_ids_pass() {
        assert_eq!(check_listed(VULKAN_0, Ok(vec![VULKAN_0])), Ok(()));
    }

    #[test]
    fn unlisted_ids_name_the_devices_found() {
        let lavapipe = DeviceId::Vulkan {
            adapter: AdapterKind::Cpu,
            index: 0,
        };
        // An index past `u16::MAX` that CubeCL would clamp onto another device.
        let aliased = DeviceId::Vulkan {
            adapter: AdapterKind::DiscreteGpu,
            index: usize::from(u16::MAX) + 1,
        };
        assert_eq!(
            check_listed(aliased, Ok(vec![VULKAN_0, lavapipe])),
            Err(
                "no such vulkan device (devices found: vulkan:discrete_gpu:0, vulkan:cpu:0)".into()
            )
        );
        assert_eq!(
            check_listed(VULKAN_0, Ok(vec![])),
            Err("no such vulkan device (devices found: none)".into())
        );
    }

    #[test]
    fn enumeration_errors_are_passed_on() {
        let err = EnumerationError {
            backend: BackendKind::Vulkan,
            message: "panic: loader".into(),
        };
        assert_eq!(
            check_listed(VULKAN_0, Err(err)),
            Err("enumerating vulkan devices failed: panic: loader".into())
        );
    }

    #[test]
    fn backends_not_compiled_in_are_rejected() {
        for backend in BackendKind::ALL {
            if backend.is_compiled_in() {
                continue;
            }
            let id = match backend {
                BackendKind::Cuda => DeviceId::Cuda { index: 0 },
                BackendKind::Metal => DeviceId::Metal {
                    adapter: AdapterKind::IntegratedGpu,
                    index: 0,
                },
                BackendKind::Vulkan => VULKAN_0,
                BackendKind::Cpu => DeviceId::Cpu,
            };
            assert_eq!(
                resolve_device(id).err(),
                Some(format!("the {backend} backend is not compiled in"))
            );
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_resolves() {
        assert!(resolve_device(DeviceId::Cpu).is_ok());
    }
}
