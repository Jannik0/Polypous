//! Device discovery for `Engine::new` (placement.md §1, ADR-0013).
//!
//! For now discovery runs **in the app process**: enumerate every compiled-in backend, then
//! describe and probe each device, in GPU priority order. Software rasterizers (lavapipe, …) are
//! listed but not opened unless [`EngineConfig::software_rasterizers`] is set; the rule is the
//! backend's [`EnumerateOptions`](backend::EnumerateOptions). A broken GPU driver can still
//! crash the process natively. The prober process, the probe cache and the crash journal (#14)
//! replace [`in_process`] behind the [`Discovery`] type without changing the engine.

use std::time::Instant;

use polypous_backend::device::{
    self as backend, AdapterKind, BackendKind, DeviceId, DeviceInfo, DeviceStatus,
};

use crate::EngineConfig;

/// How an engine discovers its devices: a function from the config to the backend's device
/// descriptions, each with a probe result. `Engine::new` uses [`in_process`]; tests inject
/// synthetic devices; #14 adds the prober process and the cache.
pub(crate) type Discovery = dyn FnOnce(&EngineConfig) -> Vec<DeviceInfo>;

/// Probe order of a device: GPU priority order of placement.md §1 (CUDA, Metal, Vulkan discrete,
/// Vulkan integrated), then other adapters and software rasterizers, then the CPU. Devices that
/// are likely to become the primary GPU are probed first.
fn probe_rank(id: &DeviceId) -> u8 {
    match id {
        DeviceId::Cuda { .. } => 0,
        DeviceId::Metal { .. } => 1,
        DeviceId::Vulkan {
            adapter: AdapterKind::DiscreteGpu,
            ..
        } => 2,
        DeviceId::Vulkan {
            adapter: AdapterKind::IntegratedGpu,
            ..
        } => 3,
        DeviceId::Vulkan {
            adapter: AdapterKind::VirtualGpu | AdapterKind::Other,
            ..
        } => 4,
        DeviceId::Vulkan {
            adapter: AdapterKind::Cpu,
            ..
        } => 5,
        DeviceId::Cpu => 6,
    }
}

/// Sorts device ids into probe order; the sort is stable, so the backend's order is kept within a
/// rank.
fn probe_order(mut ids: Vec<DeviceId>) -> Vec<DeviceId> {
    ids.sort_by_key(probe_rank);
    ids
}

/// The backend's discovery options for an engine config.
fn enumerate_options(config: &EngineConfig) -> backend::EnumerateOptions {
    backend::EnumerateOptions {
        software_rasterizers: config.software_rasterizers,
    }
}

/// Enumerates, describes and probes every device of every compiled-in backend in this process,
/// except devices the config skips (software rasterizers by default), which are listed without
/// being opened.
///
/// The device lists come from the backend's process-wide enumeration cache, so a second engine
/// doesn't enumerate again; and every device probed here is open afterwards, so later work on it
/// creates no Vulkan instance that could race with compute on other threads (placement.md §1,
/// "When enumeration happens"; issue #26).
///
/// Never fails: a backend whose enumeration fails is logged and skipped, and a device that cannot
/// be opened or fails its probe is reported as unavailable with the reason.
pub(crate) fn in_process(config: &EngineConfig) -> Vec<DeviceInfo> {
    let started = Instant::now();
    let mut ids = Vec::new();
    for backend_kind in BackendKind::compiled_in() {
        match backend::list_backend_devices(backend_kind) {
            Ok(found) => {
                tracing::debug!(
                    backend = %backend_kind,
                    count = found.len(),
                    devices = ?found.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "enumerated devices"
                );
                ids.extend(found);
            }
            Err(error) => {
                tracing::warn!(backend = %backend_kind, %error, "device enumeration failed");
            }
        }
    }

    let devices = discover_listed(ids, &enumerate_options(config), describe_and_probe);
    tracing::debug!(
        count = devices.len(),
        elapsed_ms = started.elapsed().as_millis(),
        isolated = false,
        "device discovery finished (in process)"
    );
    devices
}

/// Sorts `ids` into probe order, then describes and probes each one with `open`, except the
/// devices `options` skip, which get the backend's skipped description without being opened.
fn discover_listed(
    ids: Vec<DeviceId>,
    options: &backend::EnumerateOptions,
    mut open: impl FnMut(DeviceId) -> DeviceInfo,
) -> Vec<DeviceInfo> {
    probe_order(ids)
        .into_iter()
        .map(|id| {
            if options.skips(id) {
                let info = backend::skipped_device(id);
                tracing::info!(
                    device = %id,
                    "software rasterizer listed but not opened (set software_rasterizers to probe it)"
                );
                info
            } else {
                open(id)
            }
        })
        .collect()
}

/// Describes one device and, if it could be opened, probes it.
fn describe_and_probe(id: DeviceId) -> DeviceInfo {
    tracing::debug!(device = %id, "describing device");
    let started = Instant::now();
    let mut info = backend::describe_device(id);
    tracing::debug!(
        device = %id,
        name = %info.name,
        kind = ?info.kind,
        vendor = info.vendor.as_deref().unwrap_or("-"),
        driver = info.driver.as_deref().unwrap_or("-"),
        dedicated_memory = ?info.dedicated_memory,
        free_memory = ?info.free_memory,
        pci_bus_id = info.identity.pci_bus_id.as_deref().unwrap_or("-"),
        luid = info.identity.luid.as_deref().unwrap_or("-"),
        capabilities = ?info.capabilities,
        elapsed_ms = started.elapsed().as_millis(),
        "described device"
    );
    // A device that could not be opened already carries its reason; don't probe it.
    if info.status == DeviceStatus::NotProbed {
        tracing::debug!(device = %id, "probing device");
        let started = Instant::now();
        info.status = backend::probe_device(id);
        tracing::debug!(
            device = %id,
            status = %info.status,
            elapsed_ms = started.elapsed().as_millis(),
            "probed device"
        );
    }
    if !info.status.is_available() {
        tracing::warn!(device = %id, name = %info.name, status = %info.status, "device unavailable");
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_order_follows_gpu_priority() {
        let vulkan = |adapter, index| DeviceId::Vulkan { adapter, index };
        let ids = vec![
            DeviceId::Cpu,
            vulkan(AdapterKind::Cpu, 0),
            vulkan(AdapterKind::IntegratedGpu, 0),
            vulkan(AdapterKind::DiscreteGpu, 1),
            vulkan(AdapterKind::DiscreteGpu, 0),
            DeviceId::Metal {
                adapter: AdapterKind::IntegratedGpu,
                index: 0,
            },
            vulkan(AdapterKind::Other, 0),
            DeviceId::Cuda { index: 0 },
        ];
        assert_eq!(
            probe_order(ids),
            vec![
                DeviceId::Cuda { index: 0 },
                DeviceId::Metal {
                    adapter: AdapterKind::IntegratedGpu,
                    index: 0,
                },
                // Stable: the backend's order within a rank.
                vulkan(AdapterKind::DiscreteGpu, 1),
                vulkan(AdapterKind::DiscreteGpu, 0),
                vulkan(AdapterKind::IntegratedGpu, 0),
                vulkan(AdapterKind::Other, 0),
                vulkan(AdapterKind::Cpu, 0),
                DeviceId::Cpu,
            ]
        );
    }

    /// The reference machine on Debian 13 with stock Mesa, in the backend's listing order.
    fn debian_reference_ids() -> Vec<DeviceId> {
        vec![
            DeviceId::Cuda { index: 0 },
            DeviceId::Vulkan {
                adapter: AdapterKind::DiscreteGpu,
                index: 0,
            },
            DeviceId::Vulkan {
                adapter: AdapterKind::IntegratedGpu,
                index: 0,
            },
            DeviceId::Vulkan {
                adapter: AdapterKind::Cpu,
                index: 0,
            },
            DeviceId::Cpu,
        ]
    }

    /// Runs `discover_listed` with an open step that records what it opened and reports every
    /// opened device as available.
    fn discover(config: &EngineConfig) -> (Vec<DeviceInfo>, Vec<DeviceId>) {
        let mut opened = Vec::new();
        let devices = discover_listed(debian_reference_ids(), &enumerate_options(config), |id| {
            opened.push(id);
            DeviceInfo {
                status: DeviceStatus::Available { isolated: false },
                ..backend::skipped_device(id)
            }
        });
        (devices, opened)
    }

    #[test]
    fn software_rasterizers_are_listed_but_not_opened_by_default() {
        let lavapipe = DeviceId::Vulkan {
            adapter: AdapterKind::Cpu,
            index: 0,
        };
        let (devices, opened) = discover(&EngineConfig::default());
        assert!(!opened.contains(&lavapipe));
        assert_eq!(opened.len(), 4);
        let ids: Vec<DeviceId> = devices.iter().map(|device| device.id).collect();
        assert_eq!(ids, probe_order(debian_reference_ids()));
        let skipped = devices
            .iter()
            .find(|device| device.id == lavapipe)
            .expect("lavapipe is listed");
        assert_eq!(*skipped, backend::skipped_device(lavapipe));

        // Through the facade: the reason is converted, and the primary GPU is still the RTX via
        // CUDA.
        let converted = crate::device::device_list(devices);
        let skipped = converted
            .iter()
            .find(|device| {
                device.backend == crate::BackendKind::Vulkan
                    && device.kind == crate::DeviceKind::Cpu
            })
            .expect("lavapipe is listed");
        assert_eq!(
            serde_json::to_value(&skipped.status).unwrap(),
            serde_json::json!({
                "state": "unavailable",
                "reason": {"code": "software_rasterizer_skipped"},
                "isolated": false
            })
        );
        assert!(!skipped.primary);
        let primary = converted.iter().find(|device| device.primary).unwrap();
        assert_eq!(primary.id, crate::DeviceId::Cuda { index: 0 });
    }

    #[test]
    fn software_rasterizers_are_probed_when_opted_in() {
        let config = EngineConfig {
            software_rasterizers: true,
        };
        let (devices, opened) = discover(&config);
        assert_eq!(opened, probe_order(debian_reference_ids()));
        assert!(devices.iter().all(|device| device.status.is_available()));
        // Probed and available, but still never the primary GPU.
        let converted = crate::device::device_list(devices);
        let primary = converted.iter().find(|device| device.primary).unwrap();
        assert_eq!(primary.id, crate::DeviceId::Cuda { index: 0 });
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn in_process_discovery_probes_the_cpu() {
        let devices = in_process(&EngineConfig::default());
        let cpu = devices
            .iter()
            .find(|device| device.id == DeviceId::Cpu)
            .expect("the cpu device");
        assert_eq!(cpu.status, DeviceStatus::Available { isolated: false });
        assert!(
            devices
                .iter()
                .all(|device| device.status != DeviceStatus::NotProbed)
        );
        assert_eq!(devices.last().map(|device| device.id), Some(DeviceId::Cpu));
    }
}
