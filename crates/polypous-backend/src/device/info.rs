//! Device description types: what discovery and probing report about a device.
//!
//! Spec: `docs/specs/placement.md` §1. All types here are plain data with stable `snake_case` serde
//! names, so the prober process (ADR-0013, issue #14) can send them as JSON lines. They contain no
//! Burn types (ADR-0007).

use std::fmt;

use serde::{Deserialize, Serialize};

/// A compute backend family as Polypous names it (ADR-0003, ADR-0011).
///
/// This is a Polypous value type. `polypous-backend` maps it to and from Burn's devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// NVIDIA CUDA (CubeCL CUDA runtime).
    Cuda,
    /// Apple Metal (CubeCL wgpu runtime pinned to Metal).
    Metal,
    /// Vulkan (CubeCL wgpu runtime pinned to Vulkan).
    Vulkan,
    /// The pure-Rust CPU backend (Burn's `flex`).
    Cpu,
}

impl BackendKind {
    /// All backend kinds, in the GPU priority order of placement.md §1 (CPU last).
    pub const ALL: [BackendKind; 4] = [
        BackendKind::Cuda,
        BackendKind::Metal,
        BackendKind::Vulkan,
        BackendKind::Cpu,
    ];

    /// The stable lower-case name (`cuda`, `metal`, `vulkan`, `cpu`), the same as in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Cuda => "cuda",
            BackendKind::Metal => "metal",
            BackendKind::Vulkan => "vulkan",
            BackendKind::Cpu => "cpu",
        }
    }

    /// Whether this backend is compiled into this build (cargo features `cuda`, `metal`, `vulkan`,
    /// `cpu`).
    pub fn is_compiled_in(self) -> bool {
        match self {
            BackendKind::Cuda => cfg!(feature = "cuda"),
            BackendKind::Metal => cfg!(feature = "metal"),
            BackendKind::Vulkan => cfg!(feature = "vulkan"),
            BackendKind::Cpu => cfg!(feature = "cpu"),
        }
    }

    /// The backends compiled into this build, in [`BackendKind::ALL`] order.
    pub fn compiled_in() -> Vec<BackendKind> {
        Self::ALL
            .into_iter()
            .filter(|kind| kind.is_compiled_in())
            .collect()
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The adapter type a graphics API (Vulkan, Metal) reports. Together with an index it addresses
/// one adapter of that API, the way Burn's `DeviceKind` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    /// A discrete GPU.
    DiscreteGpu,
    /// A GPU integrated into the CPU package, sharing system RAM.
    IntegratedGpu,
    /// A virtual GPU (for example in a VM).
    VirtualGpu,
    /// A software rasterizer such as lavapipe, llvmpipe or WARP.
    Cpu,
    /// Anything else the API reports.
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

    /// The [`DeviceKind`] of an adapter of this type.
    pub fn device_kind(self) -> DeviceKind {
        match self {
            AdapterKind::DiscreteGpu => DeviceKind::Discrete,
            AdapterKind::IntegratedGpu => DeviceKind::Integrated,
            AdapterKind::Cpu => DeviceKind::Cpu,
            AdapterKind::VirtualGpu | AdapterKind::Other => DeviceKind::Other,
        }
    }
}

/// Addresses one device of one backend, stable within one boot of one machine.
///
/// It is what [`probe_device`](super::probe_device) and [`describe_device`](super::describe_device)
/// take, so the prober process (#14) can be told which devices to (re-)probe. Indices follow
/// Burn's enumeration: the CUDA ordinal, or for Vulkan/Metal the index among adapters of the same
/// [`AdapterKind`] on that API.
///
/// JSON: `{"backend":"cuda","index":0}`, `{"backend":"vulkan","adapter":"discrete_gpu","index":0}`,
/// `{"backend":"cpu"}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum DeviceId {
    /// A CUDA device by ordinal.
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
    /// `cuda:0`, `vulkan:discrete_gpu:0`, `metal:integrated_gpu:0`, `cpu`.
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

/// What kind of hardware a device is (placement.md §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    /// A GPU with its own memory (VRAM).
    Discrete,
    /// A GPU that shares system RAM with the CPU.
    Integrated,
    /// The CPU, or a software rasterizer that runs on it (for example lavapipe through Vulkan).
    Cpu,
    /// A virtual GPU or an adapter of unknown type. Never chosen as the primary GPU.
    Other,
}

/// The physical card behind a device, used to recognize one card reached through two backends
/// (for example an NVIDIA GPU through CUDA and Vulkan).
///
/// Every field is optional because each backend and OS reports a different subset. Compare with
/// [`PhysicalIdentity::is_same_card`], not `==`. Devices that are no card (the CPU, software
/// rasterizers) have an empty identity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhysicalIdentity {
    /// PCI address `domain:bus:device.function`, for example `0000:01:00.0`. Stable across
    /// restarts; reported alike by every backend on Linux.
    pub pci_bus_id: Option<String>,
    /// Windows adapter LUID as 16 lower-case hex digits (the 8 bytes, low part first). Reported
    /// alike by every backend on Windows. It changes on reboot, so never use it as a cache key.
    pub luid: Option<String>,
    /// macOS I/O Registry entry id (Metal `registryID`). Not stable across restarts.
    pub registry_id: Option<u64>,
}

impl PhysicalIdentity {
    /// Whether both identities are known to be the same card: by PCI bus id if both have one,
    /// otherwise by LUID, otherwise by registry id. An empty identity matches nothing, itself
    /// included. (The same rule as CubeCL's `PhysicalDevice::is_same_card`, which Burn's
    /// `Device::enumerate_physical` uses.)
    pub fn is_same_card(&self, other: &Self) -> bool {
        if let (Some(mine), Some(theirs)) = (&self.pci_bus_id, &other.pci_bus_id) {
            return mine.eq_ignore_ascii_case(theirs);
        }
        if let (Some(mine), Some(theirs)) = (&self.luid, &other.luid) {
            return mine.eq_ignore_ascii_case(theirs);
        }
        matches!((self.registry_id, other.registry_id), (Some(mine), Some(theirs)) if mine == theirs)
    }

    /// Whether no field is known.
    pub fn is_empty(&self) -> bool {
        self.pci_bus_id.is_none() && self.luid.is_none() && self.registry_id.is_none()
    }
}

/// Device capabilities relevant to kernels and placement. Fields are `None`/`false` where the
/// backend does not report them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Whether `f16` tensors are supported.
    pub f16: bool,
    /// Whether `bf16` tensors are supported.
    pub bf16: bool,
    /// Smallest subgroup (warp, plane) size.
    pub subgroup_size_min: Option<u32>,
    /// Largest subgroup (warp, plane) size.
    pub subgroup_size_max: Option<u32>,
    /// Largest single buffer (allocation) in bytes.
    pub max_buffer_size: Option<u64>,
    /// Shared (workgroup) memory per workgroup in bytes.
    pub max_shared_memory: Option<u64>,
    /// Streaming multiprocessors / compute units, or CPU cores for the CPU.
    pub compute_units: Option<u32>,
}

/// Why a device is unavailable (placement.md §1).
///
/// JSON: `{"code":"probe_failed","message":"…"}`, `{"code":"crashed_during_probe"}`, ….
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum UnavailableReason {
    /// The prober process died while probing this device (set by the prober's parent, #14).
    CrashedDuringProbe,
    /// The probe did not finish within its timeout (set by the prober's parent, #14).
    ProbeTimedOut,
    /// The probe ran and failed: an error, a wrong result, or a caught panic.
    ProbeFailed {
        /// What went wrong, for logs and `polyp devices`.
        message: String,
    },
    /// A runtime library the backend needs (CUDA driver, Vulkan loader) is missing.
    LibraryMissing {
        /// The library's name, for example `libcuda.so`.
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

/// A device's probe status (placement.md §1).
///
/// JSON: `{"state":"not_probed"}`, `{"state":"available","isolated":true}`,
/// `{"state":"unavailable","reason":{"code":"probe_failed","message":"…"},"isolated":false}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum DeviceStatus {
    /// Enumerated, but not probed yet.
    NotProbed,
    /// The probe passed: the device can be used.
    Available {
        /// Whether the probe ran in a separate prober process (ADR-0013).
        isolated: bool,
    },
    /// The device must not be used.
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

    /// The same status with `isolated` set, for the prober process (#14), which runs
    /// [`probe_device`](super::probe_device) in its own process and reports the result as isolated.
    #[must_use]
    pub fn with_isolated(self, value: bool) -> Self {
        match self {
            DeviceStatus::NotProbed => DeviceStatus::NotProbed,
            DeviceStatus::Available { .. } => DeviceStatus::Available { isolated: value },
            DeviceStatus::Unavailable { reason, .. } => DeviceStatus::Unavailable {
                reason,
                isolated: value,
            },
        }
    }
}

impl fmt::Display for DeviceStatus {
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

/// Everything discovery knows about one device (placement.md §1).
///
/// Serialized with stable `snake_case` field names for the prober protocol (#14).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// How to address this device again.
    pub id: DeviceId,
    /// Backend family (always `id.backend()`; repeated for readers of the JSON).
    pub backend: BackendKind,
    /// The device's name, for example `NVIDIA GeForce RTX 3060 Laptop GPU`.
    pub name: String,
    /// Vendor, for example `NVIDIA` or `Intel`, if known.
    pub vendor: Option<String>,
    /// Driver name and version as the backend reports it, if known.
    pub driver: Option<String>,
    /// Hardware kind.
    pub kind: DeviceKind,
    /// Memory the device can use for its own allocations, in bytes. For discrete GPUs this is
    /// VRAM only: Windows' host-visible "shared GPU memory" is never counted. For integrated GPUs
    /// and Apple Silicon it is the share of system RAM the driver exposes ([`unified_memory`]).
    /// `None` for the CPU, whose RAM is measured by the OS queries of the placement planner (M4).
    ///
    /// [`unified_memory`]: DeviceInfo::unified_memory
    pub dedicated_memory: Option<u64>,
    /// Free device memory in bytes at discovery time, where the backend reports it (CUDA). The
    /// planner queries it again at plan time.
    pub free_memory: Option<u64>,
    /// Whether device memory is system RAM (integrated GPUs, Apple Silicon, the CPU).
    pub unified_memory: bool,
    /// The physical card, to match one card across backends.
    pub identity: PhysicalIdentity,
    /// Capabilities.
    pub capabilities: Capabilities,
    /// Probe status.
    pub status: DeviceStatus,
}

impl DeviceInfo {
    /// Whether `other` is the same physical device reached through another (or the same) backend.
    pub fn is_same_physical_device(&self, other: &DeviceInfo) -> bool {
        self.id == other.id || self.identity.is_same_card(&other.identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_id_json_is_stable() {
        let cases = [
            (
                DeviceId::Cuda { index: 0 },
                r#"{"backend":"cuda","index":0}"#,
            ),
            (
                DeviceId::Vulkan {
                    adapter: AdapterKind::DiscreteGpu,
                    index: 1,
                },
                r#"{"backend":"vulkan","adapter":"discrete_gpu","index":1}"#,
            ),
            (DeviceId::Cpu, r#"{"backend":"cpu"}"#),
        ];
        for (id, json) in cases {
            assert_eq!(serde_json::to_string(&id).unwrap(), json);
            assert_eq!(serde_json::from_str::<DeviceId>(json).unwrap(), id);
        }
    }

    #[test]
    fn status_json_is_stable() {
        let status = DeviceStatus::Unavailable {
            reason: UnavailableReason::ProbeFailed {
                message: "wrong result".into(),
            },
            isolated: false,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(
            json,
            r#"{"state":"unavailable","reason":{"code":"probe_failed","message":"wrong result"},"isolated":false}"#
        );
        assert_eq!(serde_json::from_str::<DeviceStatus>(&json).unwrap(), status);
        assert_eq!(
            serde_json::to_string(&DeviceStatus::NotProbed).unwrap(),
            r#"{"state":"not_probed"}"#
        );
        assert_eq!(
            serde_json::to_string(&DeviceStatus::Unavailable {
                reason: UnavailableReason::CrashedDuringProbe,
                isolated: true
            })
            .unwrap(),
            r#"{"state":"unavailable","reason":{"code":"crashed_during_probe"},"isolated":true}"#
        );
    }

    #[test]
    fn status_display_matches_spec_wording() {
        let status = DeviceStatus::Unavailable {
            reason: UnavailableReason::ProbeTimedOut,
            isolated: true,
        };
        assert_eq!(status.to_string(), "unavailable: probe timed out");
        assert_eq!(
            DeviceStatus::Available { isolated: false }.to_string(),
            "available, not isolated"
        );
        assert_eq!(
            status.with_isolated(false).to_string(),
            "unavailable: probe timed out, not isolated"
        );
    }

    #[test]
    fn device_id_display() {
        assert_eq!(DeviceId::Cuda { index: 2 }.to_string(), "cuda:2");
        assert_eq!(
            DeviceId::Vulkan {
                adapter: AdapterKind::IntegratedGpu,
                index: 0
            }
            .to_string(),
            "vulkan:integrated_gpu:0"
        );
        assert_eq!(DeviceId::Cpu.to_string(), "cpu");
    }

    #[test]
    fn same_card_prefers_pci_then_luid() {
        let pci = |s: &str| PhysicalIdentity {
            pci_bus_id: Some(s.into()),
            ..Default::default()
        };
        assert!(pci("0000:01:00.0").is_same_card(&pci("0000:01:00.0")));
        assert!(!pci("0000:01:00.0").is_same_card(&pci("0000:00:02.0")));

        // CUDA on Windows reports PCI + LUID, Vulkan maybe only the LUID.
        let cuda = PhysicalIdentity {
            pci_bus_id: Some("0000:01:00.0".into()),
            luid: Some("a1b2c3d400000000".into()),
            registry_id: None,
        };
        let vulkan = PhysicalIdentity {
            luid: Some("A1B2C3D400000000".into()),
            ..Default::default()
        };
        assert!(cuda.is_same_card(&vulkan));

        // Two PCI ids that differ win over equal LUIDs.
        let other = PhysicalIdentity {
            pci_bus_id: Some("0000:02:00.0".into()),
            luid: cuda.luid.clone(),
            registry_id: None,
        };
        assert!(!cuda.is_same_card(&other));

        let empty = PhysicalIdentity::default();
        assert!(!empty.is_same_card(&empty));
        assert!(empty.is_empty());
    }
}
