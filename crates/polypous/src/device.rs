//! Public device descriptions (public-api.md §1.2, placement.md §1).
//!
//! These types are the facade's own: they mirror `polypous_backend::device` field for field, with
//! the same JSON names, plus [`DeviceInfo::primary`]. The facade converts from the backend's types
//! instead of re-exporting them, so the backend can change without breaking the public JSON that
//! apps rely on. All of them are plain serde data without Burn types (public-api.md §2 rule 3).

use std::fmt;

use polypous_backend::device as backend;
use serde::{Deserialize, Serialize};

/// A compute backend family. JSON: `"cuda"`, `"metal"`, `"vulkan"`, `"cpu"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BackendKind {
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
    /// Vulkan.
    Vulkan,
    /// The pure-Rust CPU backend.
    Cpu,
}

impl BackendKind {
    /// The stable lower-case name, the same as in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Cuda => "cuda",
            BackendKind::Metal => "metal",
            BackendKind::Vulkan => "vulkan",
            BackendKind::Cpu => "cpu",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The adapter type a graphics API (Vulkan, Metal) reports.
///
/// JSON: `"discrete_gpu"`, `"integrated_gpu"`, `"virtual_gpu"`, `"cpu"`, `"other"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AdapterKind {
    /// A discrete GPU.
    DiscreteGpu,
    /// A GPU integrated into the CPU package.
    IntegratedGpu,
    /// A virtual GPU.
    VirtualGpu,
    /// A software rasterizer such as lavapipe.
    Cpu,
    /// Anything else.
    Other,
}

impl AdapterKind {
    fn as_str(self) -> &'static str {
        match self {
            AdapterKind::DiscreteGpu => "discrete_gpu",
            AdapterKind::IntegratedGpu => "integrated_gpu",
            AdapterKind::VirtualGpu => "virtual_gpu",
            AdapterKind::Cpu => "cpu",
            AdapterKind::Other => "other",
        }
    }
}

/// Addresses one device, stable within one boot of one machine.
///
/// JSON: `{"backend":"cuda","index":0}`, `{"backend":"vulkan","adapter":"discrete_gpu","index":0}`,
/// `{"backend":"cpu"}`. Display: `cuda:0`, `vulkan:discrete_gpu:0`, `cpu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DeviceId {
    /// A CUDA device.
    Cuda {
        /// CUDA device ordinal.
        index: usize,
    },
    /// A Metal adapter.
    Metal {
        /// Adapter type.
        adapter: AdapterKind,
        /// Index among Metal adapters of this type.
        index: usize,
    },
    /// A Vulkan adapter.
    Vulkan {
        /// Adapter type.
        adapter: AdapterKind,
        /// Index among Vulkan adapters of this type.
        index: usize,
    },
    /// The CPU backend's single device.
    Cpu,
}

impl DeviceId {
    /// The backend this device belongs to.
    pub fn backend(&self) -> BackendKind {
        match self {
            DeviceId::Cuda { .. } => BackendKind::Cuda,
            DeviceId::Metal { .. } => BackendKind::Metal,
            DeviceId::Vulkan { .. } => BackendKind::Vulkan,
            DeviceId::Cpu => BackendKind::Cpu,
        }
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceId::Cuda { index } => write!(f, "cuda:{index}"),
            DeviceId::Metal { adapter, index } => write!(f, "metal:{}:{index}", adapter.as_str()),
            DeviceId::Vulkan { adapter, index } => {
                write!(f, "vulkan:{}:{index}", adapter.as_str())
            }
            DeviceId::Cpu => f.write_str("cpu"),
        }
    }
}

/// What kind of hardware a device is. JSON: `"discrete"`, `"integrated"`, `"cpu"`, `"other"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DeviceKind {
    /// A GPU with its own memory (VRAM).
    Discrete,
    /// A GPU that shares system RAM with the CPU.
    Integrated,
    /// The CPU, or a software rasterizer running on it (lavapipe).
    Cpu,
    /// A virtual GPU or an adapter of unknown type.
    Other,
}

impl DeviceKind {
    /// The stable lower-case name, the same as in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Discrete => "discrete",
            DeviceKind::Integrated => "integrated",
            DeviceKind::Cpu => "cpu",
            DeviceKind::Other => "other",
        }
    }
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The physical card behind a device. Every field is optional; the CPU and software rasterizers
/// have none.
///
/// JSON: `{"pci_bus_id":"0000:01:00.0","luid":null,"registry_id":null}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PhysicalIdentity {
    /// PCI address `domain:bus:device.function`.
    pub pci_bus_id: Option<String>,
    /// Windows adapter LUID as 16 lower-case hex digits. Changes on reboot.
    pub luid: Option<String>,
    /// macOS I/O Registry entry id.
    pub registry_id: Option<u64>,
}

/// Device capabilities. Fields are `null`/`false` where the backend does not report them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DeviceCapabilities {
    /// Whether `f16` tensors are supported.
    pub f16: bool,
    /// Whether `bf16` tensors are supported.
    pub bf16: bool,
    /// Smallest subgroup (warp) size.
    pub subgroup_size_min: Option<u32>,
    /// Largest subgroup (warp) size.
    pub subgroup_size_max: Option<u32>,
    /// Largest single buffer in bytes.
    pub max_buffer_size: Option<u64>,
    /// Shared (workgroup) memory per workgroup in bytes.
    pub max_shared_memory: Option<u64>,
    /// Compute units, or CPU cores for the CPU.
    pub compute_units: Option<u32>,
}

/// Why a device is unavailable.
///
/// JSON: `{"code":"crashed_during_probe"}`, `{"code":"probe_timed_out"}`,
/// `{"code":"probe_failed","message":"…"}`, `{"code":"library_missing","library":"…"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
#[non_exhaustive]
pub enum UnavailableReason {
    /// The probe crashed the prober process.
    CrashedDuringProbe,
    /// The probe did not finish in time.
    ProbeTimedOut,
    /// The probe ran and failed.
    #[non_exhaustive]
    ProbeFailed {
        /// What went wrong.
        message: String,
    },
    /// A runtime library the backend needs is missing.
    #[non_exhaustive]
    LibraryMissing {
        /// The library's name.
        library: String,
    },
}

impl fmt::Display for UnavailableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnavailableReason::CrashedDuringProbe => f.write_str("crashed during probe"),
            UnavailableReason::ProbeTimedOut => f.write_str("probe timed out"),
            UnavailableReason::ProbeFailed { message } => write!(f, "probe failed: {message}"),
            UnavailableReason::LibraryMissing { library } => {
                write!(f, "library missing: {library}")
            }
        }
    }
}

/// A device's probe status.
///
/// JSON: `{"state":"not_probed"}`, `{"state":"available","isolated":false}`,
/// `{"state":"unavailable","reason":{…},"isolated":false}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DeviceStatus {
    /// Enumerated, not probed.
    NotProbed,
    /// The probe passed.
    #[non_exhaustive]
    Available {
        /// Whether the probe ran in a separate prober process (ADR-0013).
        isolated: bool,
    },
    /// The device must not be used.
    #[non_exhaustive]
    Unavailable {
        /// Why.
        reason: UnavailableReason,
        /// Whether the probe ran in a separate prober process (ADR-0013).
        isolated: bool,
    },
}

impl DeviceStatus {
    /// Whether the device passed its probe.
    pub fn is_available(&self) -> bool {
        matches!(self, DeviceStatus::Available { .. })
    }
}

impl fmt::Display for DeviceStatus {
    /// `available`, `available, not isolated`, `unavailable: probe failed: …, not isolated`,
    /// `not probed` (the wording of placement.md §1).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let isolation = |isolated: bool| if isolated { "" } else { ", not isolated" };
        match self {
            DeviceStatus::NotProbed => f.write_str("not probed"),
            DeviceStatus::Available { isolated } => {
                write!(f, "available{}", isolation(*isolated))
            }
            DeviceStatus::Unavailable { reason, isolated } => {
                write!(f, "unavailable: {reason}{}", isolation(*isolated))
            }
        }
    }
}

/// Everything Polypous knows about one device (public-api.md §1.2).
///
/// The fields and their JSON names are those of placement.md §1, plus `primary`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DeviceInfo {
    /// How to address this device.
    pub id: DeviceId,
    /// Backend family (always `id.backend()`).
    pub backend: BackendKind,
    /// The device's name, for example `NVIDIA GeForce RTX 3060 Laptop GPU`.
    pub name: String,
    /// Vendor, if known.
    pub vendor: Option<String>,
    /// Driver name and version, if known.
    pub driver: Option<String>,
    /// Hardware kind.
    pub kind: DeviceKind,
    /// Memory the device can use for its own allocations, in bytes (VRAM for discrete GPUs, never
    /// Windows' "shared GPU memory"). `null` for the CPU.
    pub dedicated_memory: Option<u64>,
    /// Free device memory in bytes at discovery time, where the backend reports it (CUDA).
    pub free_memory: Option<u64>,
    /// Whether device memory is system RAM (integrated GPUs, Apple Silicon, the CPU).
    pub unified_memory: bool,
    /// The physical card, to match one card across backends.
    pub identity: PhysicalIdentity,
    /// Capabilities.
    pub capabilities: DeviceCapabilities,
    /// Probe status.
    pub status: DeviceStatus,
    /// Whether this is the primary GPU: the one GPU Polypous uses (placement.md §1, "Primary GPU
    /// selection"). At most one device has `true`; none if no GPU passed its probe (CPU only).
    pub primary: bool,
}

// Conversions from the backend's types. They are crate-private functions, not `From` impls, so no
// backend type appears in the public API. The matches are exhaustive over the backend's enums: a
// new backend variant fails to compile here until the public type has it too.

fn backend_kind(value: backend::BackendKind) -> BackendKind {
    match value {
        backend::BackendKind::Cuda => BackendKind::Cuda,
        backend::BackendKind::Metal => BackendKind::Metal,
        backend::BackendKind::Vulkan => BackendKind::Vulkan,
        backend::BackendKind::Cpu => BackendKind::Cpu,
    }
}

fn adapter_kind(value: backend::AdapterKind) -> AdapterKind {
    match value {
        backend::AdapterKind::DiscreteGpu => AdapterKind::DiscreteGpu,
        backend::AdapterKind::IntegratedGpu => AdapterKind::IntegratedGpu,
        backend::AdapterKind::VirtualGpu => AdapterKind::VirtualGpu,
        backend::AdapterKind::Cpu => AdapterKind::Cpu,
        backend::AdapterKind::Other => AdapterKind::Other,
    }
}

fn device_id(value: backend::DeviceId) -> DeviceId {
    match value {
        backend::DeviceId::Cuda { index } => DeviceId::Cuda { index },
        backend::DeviceId::Metal { adapter, index } => DeviceId::Metal {
            adapter: adapter_kind(adapter),
            index,
        },
        backend::DeviceId::Vulkan { adapter, index } => DeviceId::Vulkan {
            adapter: adapter_kind(adapter),
            index,
        },
        backend::DeviceId::Cpu => DeviceId::Cpu,
    }
}

fn device_kind(value: backend::DeviceKind) -> DeviceKind {
    match value {
        backend::DeviceKind::Discrete => DeviceKind::Discrete,
        backend::DeviceKind::Integrated => DeviceKind::Integrated,
        backend::DeviceKind::Cpu => DeviceKind::Cpu,
        backend::DeviceKind::Other => DeviceKind::Other,
    }
}

fn unavailable_reason(value: backend::UnavailableReason) -> UnavailableReason {
    match value {
        backend::UnavailableReason::CrashedDuringProbe => UnavailableReason::CrashedDuringProbe,
        backend::UnavailableReason::ProbeTimedOut => UnavailableReason::ProbeTimedOut,
        backend::UnavailableReason::ProbeFailed { message } => {
            UnavailableReason::ProbeFailed { message }
        }
        backend::UnavailableReason::LibraryMissing { library } => {
            UnavailableReason::LibraryMissing { library }
        }
    }
}

fn device_status(value: backend::DeviceStatus) -> DeviceStatus {
    match value {
        backend::DeviceStatus::NotProbed => DeviceStatus::NotProbed,
        backend::DeviceStatus::Available { isolated } => DeviceStatus::Available { isolated },
        backend::DeviceStatus::Unavailable { reason, isolated } => DeviceStatus::Unavailable {
            reason: unavailable_reason(reason),
            isolated,
        },
    }
}

/// Converts the backend's description of a device; `primary` marks the primary GPU.
pub(crate) fn device_info(value: backend::DeviceInfo, primary: bool) -> DeviceInfo {
    let backend::DeviceInfo {
        id,
        backend,
        name,
        vendor,
        driver,
        kind,
        dedicated_memory,
        free_memory,
        unified_memory,
        identity,
        capabilities,
        status,
    } = value;
    let backend::PhysicalIdentity {
        pci_bus_id,
        luid,
        registry_id,
    } = identity;
    let backend::Capabilities {
        f16,
        bf16,
        subgroup_size_min,
        subgroup_size_max,
        max_buffer_size,
        max_shared_memory,
        compute_units,
    } = capabilities;
    DeviceInfo {
        id: device_id(id),
        backend: backend_kind(backend),
        name,
        vendor,
        driver,
        kind: device_kind(kind),
        dedicated_memory,
        free_memory,
        unified_memory,
        identity: PhysicalIdentity {
            pci_bus_id,
            luid,
            registry_id,
        },
        capabilities: DeviceCapabilities {
            f16,
            bf16,
            subgroup_size_min,
            subgroup_size_max,
            max_buffer_size,
            max_shared_memory,
            compute_units,
        },
        status: device_status(status),
        primary,
    }
}

/// Converts a discovered device list and marks the primary GPU chosen by the backend's
/// `select_primary_gpu` (placement.md §1). The order is kept.
pub(crate) fn device_list(devices: Vec<backend::DeviceInfo>) -> Vec<DeviceInfo> {
    let primary = backend::select_primary_gpu(&devices).and_then(|chosen| {
        devices
            .iter()
            .position(|device| std::ptr::eq(device, chosen))
    });
    devices
        .into_iter()
        .enumerate()
        .map(|(index, device)| device_info(device, Some(index) == primary))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// A synthetic backend device.
    pub(crate) fn synthetic(
        id: backend::DeviceId,
        name: &str,
        kind: backend::DeviceKind,
        memory: Option<u64>,
        pci: Option<&str>,
        status: backend::DeviceStatus,
    ) -> backend::DeviceInfo {
        backend::DeviceInfo {
            id,
            backend: id.backend(),
            name: name.into(),
            vendor: None,
            driver: None,
            kind,
            dedicated_memory: memory,
            free_memory: None,
            unified_memory: kind != backend::DeviceKind::Discrete,
            identity: backend::PhysicalIdentity {
                pci_bus_id: pci.map(Into::into),
                ..Default::default()
            },
            capabilities: backend::Capabilities::default(),
            status,
        }
    }

    fn available() -> backend::DeviceStatus {
        backend::DeviceStatus::Available { isolated: false }
    }

    /// The reference machine: RTX 3060 via CUDA and Vulkan, Iris Xe via Vulkan, CPU.
    pub(crate) fn reference_machine() -> Vec<backend::DeviceInfo> {
        vec![
            synthetic(
                backend::DeviceId::Cuda { index: 0 },
                "NVIDIA GeForce RTX 3060 Laptop GPU",
                backend::DeviceKind::Discrete,
                Some(6 * GIB),
                Some("0000:01:00.0"),
                available(),
            ),
            synthetic(
                backend::DeviceId::Vulkan {
                    adapter: backend::AdapterKind::DiscreteGpu,
                    index: 0,
                },
                "NVIDIA GeForce RTX 3060 Laptop GPU",
                backend::DeviceKind::Discrete,
                Some(6 * GIB),
                Some("0000:01:00.0"),
                available(),
            ),
            synthetic(
                backend::DeviceId::Vulkan {
                    adapter: backend::AdapterKind::IntegratedGpu,
                    index: 0,
                },
                "Intel(R) Iris(R) Xe Graphics",
                backend::DeviceKind::Integrated,
                Some(7 * GIB),
                Some("0000:00:02.0"),
                backend::DeviceStatus::Unavailable {
                    reason: backend::UnavailableReason::ProbeFailed {
                        message: "wrong result".into(),
                    },
                    isolated: false,
                },
            ),
            synthetic(
                backend::DeviceId::Cpu,
                "12th Gen Intel(R) Core(TM) i7-12800H",
                backend::DeviceKind::Cpu,
                None,
                None,
                available(),
            ),
        ]
    }

    #[test]
    fn every_field_is_converted() {
        let mut device = synthetic(
            backend::DeviceId::Vulkan {
                adapter: backend::AdapterKind::DiscreteGpu,
                index: 1,
            },
            "gpu",
            backend::DeviceKind::Discrete,
            Some(4 * GIB),
            Some("0000:01:00.0"),
            available(),
        );
        device.vendor = Some("NVIDIA".into());
        device.driver = Some("NVIDIA 580.95.05".into());
        device.free_memory = Some(3 * GIB);
        device.identity.luid = Some("a1b2c3d400000000".into());
        device.identity.registry_id = Some(42);
        device.capabilities = backend::Capabilities {
            f16: true,
            bf16: false,
            subgroup_size_min: Some(32),
            subgroup_size_max: Some(32),
            max_buffer_size: Some(GIB),
            max_shared_memory: Some(49152),
            compute_units: Some(30),
        };
        // The backend's JSON and the facade's JSON agree on every field placement.md §1 names;
        // the facade adds `primary`.
        let mut expected = serde_json::to_value(&device).unwrap();
        expected["primary"] = serde_json::Value::Bool(true);
        let converted = device_info(device, true);
        assert_eq!(serde_json::to_value(&converted).unwrap(), expected);
    }

    #[test]
    fn every_status_is_converted() {
        let statuses = [
            backend::DeviceStatus::NotProbed,
            backend::DeviceStatus::Available { isolated: true },
            backend::DeviceStatus::Unavailable {
                reason: backend::UnavailableReason::CrashedDuringProbe,
                isolated: true,
            },
            backend::DeviceStatus::Unavailable {
                reason: backend::UnavailableReason::ProbeTimedOut,
                isolated: true,
            },
            backend::DeviceStatus::Unavailable {
                reason: backend::UnavailableReason::ProbeFailed {
                    message: "m".into(),
                },
                isolated: false,
            },
            backend::DeviceStatus::Unavailable {
                reason: backend::UnavailableReason::LibraryMissing {
                    library: "libcuda.so".into(),
                },
                isolated: false,
            },
        ];
        for status in statuses {
            let converted = device_status(status.clone());
            assert_eq!(
                serde_json::to_value(&converted).unwrap(),
                serde_json::to_value(&status).unwrap()
            );
            assert_eq!(converted.to_string(), status.to_string());
            assert_eq!(converted.is_available(), status.is_available());
        }
    }

    #[test]
    fn every_id_is_converted() {
        let adapters = [
            backend::AdapterKind::DiscreteGpu,
            backend::AdapterKind::IntegratedGpu,
            backend::AdapterKind::VirtualGpu,
            backend::AdapterKind::Cpu,
            backend::AdapterKind::Other,
        ];
        let mut ids = vec![backend::DeviceId::Cuda { index: 3 }, backend::DeviceId::Cpu];
        for adapter in adapters {
            ids.push(backend::DeviceId::Vulkan { adapter, index: 1 });
            ids.push(backend::DeviceId::Metal { adapter, index: 0 });
        }
        for id in ids {
            let converted = device_id(id);
            assert_eq!(
                serde_json::to_value(converted).unwrap(),
                serde_json::to_value(id).unwrap()
            );
            assert_eq!(converted.to_string(), id.to_string());
            assert_eq!(converted.backend().as_str(), id.backend().as_str());
        }
        for kind in [
            backend::DeviceKind::Discrete,
            backend::DeviceKind::Integrated,
            backend::DeviceKind::Cpu,
            backend::DeviceKind::Other,
        ] {
            assert_eq!(
                serde_json::to_value(device_kind(kind)).unwrap(),
                serde_json::to_value(kind).unwrap()
            );
        }
    }

    #[test]
    fn reference_machine_marks_the_rtx_via_cuda_as_primary() {
        let devices = device_list(reference_machine());
        let primary: Vec<_> = devices.iter().filter(|device| device.primary).collect();
        assert_eq!(primary.len(), 1);
        assert_eq!(primary[0].id, DeviceId::Cuda { index: 0 });
        // Order is kept.
        assert_eq!(devices[3].id, DeviceId::Cpu);
    }

    #[test]
    fn cpu_only_has_no_primary() {
        let mut devices = reference_machine();
        devices.drain(..3);
        let devices = device_list(devices);
        assert_eq!(devices.len(), 1);
        assert!(!devices[0].primary);
        assert!(device_list(Vec::new()).is_empty());
    }

    /// The JSON of `polyp devices --json` and `Engine::devices()` (public-api.md §1.2), as a
    /// snapshot of a synthetic device list.
    #[test]
    fn device_list_json_snapshot() {
        let mut devices = reference_machine();
        devices.remove(1);
        devices.remove(1);
        let json = serde_json::to_string_pretty(&device_list(devices)).unwrap();
        let expected = r#"[
  {
    "id": {
      "backend": "cuda",
      "index": 0
    },
    "backend": "cuda",
    "name": "NVIDIA GeForce RTX 3060 Laptop GPU",
    "vendor": null,
    "driver": null,
    "kind": "discrete",
    "dedicated_memory": 6442450944,
    "free_memory": null,
    "unified_memory": false,
    "identity": {
      "pci_bus_id": "0000:01:00.0",
      "luid": null,
      "registry_id": null
    },
    "capabilities": {
      "f16": false,
      "bf16": false,
      "subgroup_size_min": null,
      "subgroup_size_max": null,
      "max_buffer_size": null,
      "max_shared_memory": null,
      "compute_units": null
    },
    "status": {
      "state": "available",
      "isolated": false
    },
    "primary": true
  },
  {
    "id": {
      "backend": "cpu"
    },
    "backend": "cpu",
    "name": "12th Gen Intel(R) Core(TM) i7-12800H",
    "vendor": null,
    "driver": null,
    "kind": "cpu",
    "dedicated_memory": null,
    "free_memory": null,
    "unified_memory": true,
    "identity": {
      "pci_bus_id": null,
      "luid": null,
      "registry_id": null
    },
    "capabilities": {
      "f16": false,
      "bf16": false,
      "subgroup_size_min": null,
      "subgroup_size_max": null,
      "max_buffer_size": null,
      "max_shared_memory": null,
      "compute_units": null
    },
    "status": {
      "state": "available",
      "isolated": false
    },
    "primary": false
  }
]"#;
        assert_eq!(json, expected);
        // It reads back.
        let parsed: Vec<DeviceInfo> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].primary);
    }

    #[test]
    fn unavailable_status_json() {
        let status = device_status(backend::DeviceStatus::Unavailable {
            reason: backend::UnavailableReason::ProbeFailed {
                message: "wrong result".into(),
            },
            isolated: false,
        });
        assert_eq!(
            serde_json::to_string(&status).unwrap(),
            r#"{"state":"unavailable","reason":{"code":"probe_failed","message":"wrong result"},"isolated":false}"#
        );
        assert_eq!(
            status.to_string(),
            "unavailable: probe failed: wrong result, not isolated"
        );
    }
}
