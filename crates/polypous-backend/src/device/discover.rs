//! Device enumeration and description on top of Burn (placement.md §1, ADR-0011).
//!
//! Enumeration uses Burn's `Device::enumerate` per compiled-in backend. Memory, identity and
//! capabilities come from the CubeCL client of the device, which Burn's `Device::identity` and
//! `Device::enumerate_physical` also use, plus a few facts Burn does not expose: driver strings
//! from wgpu's adapter info and free memory from the CUDA driver (cudarc). Every Burn call that can
//! panic runs under `catch_unwind`; native crashes are the prober process's job (ADR-0013, #14).

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::tensor::{DType, Device, DeviceType};
use serde::{Deserialize, Serialize};

use super::info::{
    BackendKind, Capabilities, DeviceId, DeviceInfo, DeviceKind, DeviceStatus, PhysicalIdentity,
    UnavailableReason,
};

/// Enumeration of one backend failed (Burn panicked). The backend's devices are unknown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnumerationError {
    /// The backend whose enumeration failed.
    pub backend: BackendKind,
    /// What went wrong.
    pub message: String,
}

impl std::fmt::Display for EnumerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "enumerating {} devices failed: {}",
            self.backend, self.message
        )
    }
}

impl std::error::Error for EnumerationError {}

/// Serializes everything that opens devices. CubeCL guards its own wgpu instances against
/// concurrent creation (the Vulkan loader crashes when instances come and go on several threads at
/// once), but [`describe_device`] creates one more for wgpu's adapter info, so device work here runs
/// one call at a time.
static DEVICE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Holds [`DEVICE_LOCK`]. The lock protects no data, so a poisoned lock is fine to take.
pub(crate) fn lock_devices() -> std::sync::MutexGuard<'static, ()> {
    DEVICE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Text of a caught panic payload.
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panic with a non-text payload".to_string()
    }
}

/// Lists the devices of one backend without opening them.
///
/// A backend that is not compiled in, or whose runtime library is missing (no `libcuda`, no
/// Vulkan loader or ICD), has no devices. A Burn panic during enumeration becomes an
/// [`EnumerationError`].
pub fn list_backend_devices(backend: BackendKind) -> Result<Vec<DeviceId>, EnumerationError> {
    let Some(device_type) = device_type(backend) else {
        return Ok(Vec::new());
    };
    let _guard = lock_devices();
    catch_unwind(AssertUnwindSafe(|| {
        Device::enumerate(device_type)
            .into_iter()
            .filter_map(|device| device_id_of(&device))
            .filter(|id| id.backend() == backend)
            .collect()
    }))
    .map_err(|payload| EnumerationError {
        backend,
        message: format!("panic: {}", panic_message(&*payload)),
    })
}

/// Lists the devices of every compiled-in backend, in [`BackendKind::ALL`] order, without
/// opening them. Backends whose enumeration fails are skipped; use [`list_backend_devices`] to see
/// why.
pub fn list_devices() -> Vec<DeviceId> {
    BackendKind::compiled_in()
        .into_iter()
        .filter_map(|backend| list_backend_devices(backend).ok())
        .flatten()
        .collect()
}

/// Opens one device and reads its name, memory, identity and capabilities. The status is
/// [`DeviceStatus::NotProbed`].
///
/// Never panics: if opening the device fails (an error or a caught Burn panic), the result carries
/// what is known from `id` and the status is unavailable with the reason. Opening a GPU loads its
/// driver, which can crash the process natively; run it in the prober process (ADR-0013).
pub fn describe_device(id: DeviceId) -> DeviceInfo {
    let _guard = lock_devices();
    let result = catch_unwind(AssertUnwindSafe(|| describe(id)))
        .unwrap_or_else(|payload| Err(format!("panic: {}", panic_message(&*payload))));
    match result {
        Ok(info) => info,
        Err(message) => DeviceInfo {
            status: DeviceStatus::Unavailable {
                reason: UnavailableReason::ProbeFailed {
                    message: format!("could not open the device: {message}"),
                },
                isolated: false,
            },
            ..placeholder(id)
        },
    }
}

/// Lists and describes the devices of every compiled-in backend (placement.md §1). Never fails
/// or panics; see [`list_devices`] and [`describe_device`].
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    list_devices().into_iter().map(describe_device).collect()
}

/// A description from the id alone, for devices that could not be opened.
fn placeholder(id: DeviceId) -> DeviceInfo {
    let kind = match id {
        DeviceId::Cuda { .. } => DeviceKind::Discrete,
        DeviceId::Metal { adapter, .. } | DeviceId::Vulkan { adapter, .. } => adapter.device_kind(),
        DeviceId::Cpu => DeviceKind::Cpu,
    };
    DeviceInfo {
        id,
        backend: id.backend(),
        name: id.to_string(),
        vendor: None,
        driver: None,
        kind,
        dedicated_memory: None,
        free_memory: None,
        unified_memory: kind != DeviceKind::Discrete,
        identity: PhysicalIdentity::default(),
        capabilities: Capabilities::default(),
        status: DeviceStatus::NotProbed,
    }
}

/// Burn's device filter for a backend, if it is compiled in.
fn device_type(backend: BackendKind) -> Option<DeviceType> {
    match backend {
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => Some(DeviceType::Cuda),
        #[cfg(feature = "metal")]
        BackendKind::Metal => Some(DeviceType::Metal),
        #[cfg(feature = "vulkan")]
        BackendKind::Vulkan => Some(DeviceType::Vulkan),
        #[cfg(feature = "cpu")]
        BackendKind::Cpu => Some(DeviceType::Flex),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

/// Our id for a Burn device, if it belongs to a backend Polypous uses.
pub(crate) fn device_id_of(device: &Device) -> Option<DeviceId> {
    #[allow(unused_imports)]
    use burn::backend::DispatchDevice;
    match device.as_dispatch() {
        #[cfg(feature = "cpu")]
        DispatchDevice::Flex(_) => Some(DeviceId::Cpu),
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        DispatchDevice::Cube(cube) => gpu::device_id_of(cube),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

/// The Burn device for an id, if its backend is compiled in.
pub(crate) fn burn_device(id: DeviceId) -> Option<Device> {
    match id {
        #[cfg(feature = "cpu")]
        DeviceId::Cpu => Some(Device::flex()),
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        _ => gpu::cube_device(id).map(Device::new),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

fn describe(id: DeviceId) -> Result<DeviceInfo, String> {
    if !id.backend().is_compiled_in() {
        return Err(format!("the {} backend is not compiled in", id.backend()));
    }
    match id {
        DeviceId::Cpu => describe_cpu(),
        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        _ => gpu::describe(id),
        #[allow(unreachable_patterns)]
        _ => Err(format!("the {} backend is not compiled in", id.backend())),
    }
}

/// `f16` / `bf16` support as Burn reports it for the device.
fn dtype_capabilities(device: &Device) -> (bool, bool) {
    (
        device.supports_dtype(DType::F16),
        device.supports_dtype(DType::BF16),
    )
}

fn describe_cpu() -> Result<DeviceInfo, String> {
    let device = burn_device(DeviceId::Cpu).ok_or("the cpu backend is not compiled in")?;
    let (f16, bf16) = dtype_capabilities(&device);
    let compute_units = std::thread::available_parallelism()
        .ok()
        .and_then(|n| u32::try_from(n.get()).ok());
    Ok(DeviceInfo {
        name: cpu_name().unwrap_or_else(|| format!("CPU ({})", std::env::consts::ARCH)),
        capabilities: Capabilities {
            f16,
            bf16,
            compute_units,
            ..Capabilities::default()
        },
        ..placeholder(DeviceId::Cpu)
    })
}

/// The CPU's model name from the OS, without loading any driver.
fn cpu_name() -> Option<String> {
    let name = if cfg!(target_os = "linux") {
        // Read at most 64 KiB: /proc/cpuinfo is small, but don't trust its size.
        use std::io::Read;
        let mut text = String::new();
        std::fs::File::open("/proc/cpuinfo")
            .ok()?
            .take(64 * 1024)
            .read_to_string(&mut text)
            .ok()?;
        text.lines()
            .find(|line| line.starts_with("model name"))
            .and_then(|line| line.split_once(':'))
            .map(|(_, value)| value.trim().to_string())
    } else if cfg!(windows) {
        std::env::var("PROCESSOR_IDENTIFIER").ok()
    } else {
        None
    };
    name.filter(|name| !name.is_empty())
}

/// CubeCL devices (CUDA, Vulkan, Metal).
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
mod gpu {
    use cubecl::Device as CubeDevice;
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    use cubecl::device::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

    use super::super::info::{AdapterKind, Capabilities, DeviceId, DeviceInfo, PhysicalIdentity};
    use super::{DeviceKind, dtype_capabilities, placeholder};

    pub(super) fn device_id_of(cube: &CubeDevice) -> Option<DeviceId> {
        match cube {
            #[cfg(feature = "cuda")]
            CubeDevice::Cuda(device) => Some(DeviceId::Cuda {
                index: device.index,
            }),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            CubeDevice::Wgpu(device) => {
                let (adapter, index) = match device.kind {
                    WgpuDeviceKind::DiscreteGpu(index) => (AdapterKind::DiscreteGpu, index),
                    WgpuDeviceKind::IntegratedGpu(index) => (AdapterKind::IntegratedGpu, index),
                    WgpuDeviceKind::VirtualGpu(index) => (AdapterKind::VirtualGpu, index),
                    WgpuDeviceKind::Cpu => (AdapterKind::Cpu, 0),
                    WgpuDeviceKind::Other(index) => (AdapterKind::Other, index),
                    _ => return None,
                };
                match device.backend {
                    WgpuBackend::Vulkan => Some(DeviceId::Vulkan { adapter, index }),
                    WgpuBackend::Metal => Some(DeviceId::Metal { adapter, index }),
                    _ => None,
                }
            }
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    pub(super) fn cube_device(id: DeviceId) -> Option<CubeDevice> {
        #[cfg(any(feature = "vulkan", feature = "metal"))]
        let wgpu = |adapter: AdapterKind, index: usize, backend: WgpuBackend| {
            let kind = match adapter {
                AdapterKind::DiscreteGpu => WgpuDeviceKind::DiscreteGpu(index),
                AdapterKind::IntegratedGpu => WgpuDeviceKind::IntegratedGpu(index),
                AdapterKind::VirtualGpu => WgpuDeviceKind::VirtualGpu(index),
                AdapterKind::Cpu => WgpuDeviceKind::Cpu,
                AdapterKind::Other => WgpuDeviceKind::Other(index),
            };
            CubeDevice::Wgpu(WgpuDevice::new(kind).on(backend))
        };
        match id {
            #[cfg(feature = "cuda")]
            DeviceId::Cuda { index } => {
                Some(CubeDevice::Cuda(cubecl::device::CudaDevice { index }))
            }
            #[cfg(feature = "vulkan")]
            DeviceId::Vulkan { adapter, index } => Some(wgpu(adapter, index, WgpuBackend::Vulkan)),
            #[cfg(feature = "metal")]
            DeviceId::Metal { adapter, index } => Some(wgpu(adapter, index, WgpuBackend::Metal)),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    pub(super) fn describe(id: DeviceId) -> Result<DeviceInfo, String> {
        let cube = cube_device(id).ok_or_else(|| format!("{id} is not a CubeCL device"))?;
        let burn_device = burn::tensor::Device::new(cube.clone());
        // Opens the device (creates its CubeCL client, loads the driver).
        let client = cube.client();
        let properties = client.properties();
        let (f16, bf16) = dtype_capabilities(&burn_device);

        let mut info = placeholder(id);
        let identity = &properties.identity;
        if !identity.name.trim().is_empty() {
            info.name = identity.name.trim().to_string();
        }
        if let Some(physical) = &identity.physical {
            info.vendor = physical.vendor.map(|vendor| vendor.to_string());
            info.identity = PhysicalIdentity {
                pci_bus_id: physical.pci_address.map(|address| address.to_string()),
                luid: physical.luid.map(|luid| {
                    luid.bytes()
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect()
                }),
                registry_id: physical.registry_entry_id.map(|id| id.get()),
            };
        }
        // CUDA: `cuDeviceTotalMem`. Vulkan: the largest DEVICE_LOCAL heap, so the host-visible
        // "shared GPU memory" heap of a discrete GPU on Windows is never counted (it is not
        // DEVICE_LOCAL). Metal: `recommendedMaxWorkingSetSize`. (cubecl-cuda runtime.rs,
        // cubecl-wgpu backend/vulkan.rs `register_features`, backend/metal.rs.)
        info.dedicated_memory = properties.memory.max_memory();
        let hardware = &properties.hardware;
        info.capabilities = Capabilities {
            f16,
            bf16,
            subgroup_size_min: Some(hardware.plane_size_min).filter(|&size| size > 0),
            subgroup_size_max: Some(hardware.plane_size_max).filter(|&size| size > 0),
            max_buffer_size: Some(properties.memory.max_page_size).filter(|&size| size > 0),
            max_shared_memory: u64::try_from(hardware.max_shared_memory_size)
                .ok()
                .filter(|&size| size > 0),
            compute_units: hardware.num_streaming_multiprocessors,
        };

        match id {
            #[cfg(feature = "cuda")]
            DeviceId::Cuda { index } => cuda::fill(index, &mut info),
            #[cfg(feature = "vulkan")]
            DeviceId::Vulkan { adapter, index } => wgpu_info::fill(
                cubecl::wgpu::wgpu::Backend::Vulkan,
                adapter,
                index,
                &mut info,
            ),
            #[cfg(feature = "metal")]
            DeviceId::Metal { adapter, index } => wgpu_info::fill(
                cubecl::wgpu::wgpu::Backend::Metal,
                adapter,
                index,
                &mut info,
            ),
            #[allow(unreachable_patterns)]
            _ => {}
        }
        info.unified_memory = info.kind != DeviceKind::Discrete;
        Ok(info)
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use cudarc::driver::CudaContext;
        use cudarc::driver::sys::CUdevice_attribute;

        use super::{DeviceInfo, DeviceKind};

        /// Free memory and integrated-ness from the CUDA driver. Burn/CubeCL report neither.
        /// Failures leave the fields as they are: they are optional facts, not the probe.
        pub(super) fn fill(index: usize, info: &mut DeviceInfo) {
            // Retains the primary context, the same one CubeCL uses.
            let Ok(context) = CudaContext::new(index) else {
                return;
            };
            if let Ok((free, total)) = context.mem_get_info() {
                info.free_memory = u64::try_from(free).ok();
                if info.dedicated_memory.is_none() {
                    info.dedicated_memory = u64::try_from(total).ok();
                }
            }
            if context
                .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
                .is_ok_and(|integrated| integrated != 0)
            {
                info.kind = DeviceKind::Integrated;
            }
        }
    }

    #[cfg(any(feature = "vulkan", feature = "metal"))]
    mod wgpu_info {
        use cubecl::wgpu::wgpu;

        use super::{AdapterKind, DeviceInfo};

        fn adapter_kind(device_type: wgpu::DeviceType) -> AdapterKind {
            match device_type {
                wgpu::DeviceType::DiscreteGpu => AdapterKind::DiscreteGpu,
                wgpu::DeviceType::IntegratedGpu => AdapterKind::IntegratedGpu,
                wgpu::DeviceType::VirtualGpu => AdapterKind::VirtualGpu,
                wgpu::DeviceType::Cpu => AdapterKind::Cpu,
                wgpu::DeviceType::Other => AdapterKind::Other,
            }
        }

        /// Driver name/version (and vendor, PCI bus id, name where CubeCL had none) from wgpu's
        /// adapter info. Burn/CubeCL keep the adapter info to themselves, so this lists the
        /// adapters of `backend` again and picks the `index`-th of type `adapter`, the same
        /// addressing CubeCL uses (cubecl-wgpu runtime.rs, `adapter_device_ids`).
        pub(super) fn fill(
            backend: wgpu::Backend,
            adapter: AdapterKind,
            index: usize,
            info: &mut DeviceInfo,
        ) {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: backend.into(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapters = cubecl::future::block_on(instance.enumerate_adapters(backend.into()));
            // CubeCL gives every software adapter index 0.
            let index = if adapter == AdapterKind::Cpu {
                0
            } else {
                index
            };
            let Some(adapter_info) = adapters
                .iter()
                .map(wgpu::Adapter::get_info)
                .filter(|candidate| adapter_kind(candidate.device_type) == adapter)
                .nth(index)
            else {
                return;
            };
            let driver = [adapter_info.driver.trim(), adapter_info.driver_info.trim()]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !driver.is_empty() {
                info.driver = Some(driver);
            }
            if info.vendor.is_none() && adapter_info.vendor != 0 {
                info.vendor = Some(cubecl::ir::PciVendor::from(adapter_info.vendor).to_string());
            }
            if info.identity.pci_bus_id.is_none() && !adapter_info.device_pci_bus_id.is_empty() {
                info.identity.pci_bus_id = Some(adapter_info.device_pci_bus_id.clone());
            }
            if info.name.starts_with("vulkan:") || info.name.starts_with("metal:") {
                info.name = adapter_info.name.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_messages_are_extracted() {
        let payload = catch_unwind(|| panic!("static text")).unwrap_err();
        assert_eq!(panic_message(&*payload), "static text");
        let payload = catch_unwind(|| panic!("formatted {}", 42)).unwrap_err();
        assert_eq!(panic_message(&*payload), "formatted 42");
        let payload = catch_unwind(|| std::panic::panic_any(7_u8)).unwrap_err();
        assert_eq!(panic_message(&*payload), "panic with a non-text payload");
    }

    #[test]
    fn backends_not_compiled_in_have_no_devices() {
        for backend in BackendKind::ALL {
            if !backend.is_compiled_in() {
                assert_eq!(list_backend_devices(backend), Ok(Vec::new()));
            }
        }
    }

    #[test]
    fn describing_a_device_of_a_missing_backend_reports_unavailable() {
        for id in [
            DeviceId::Cuda { index: 0 },
            DeviceId::Metal {
                adapter: crate::device::AdapterKind::IntegratedGpu,
                index: 0,
            },
        ] {
            if id.backend().is_compiled_in() {
                continue;
            }
            let info = describe_device(id);
            assert_eq!(info.id, id);
            assert!(matches!(info.status, DeviceStatus::Unavailable { .. }));
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_is_enumerated_and_described() {
        assert_eq!(
            list_backend_devices(BackendKind::Cpu),
            Ok(vec![DeviceId::Cpu])
        );
        assert!(list_devices().contains(&DeviceId::Cpu));
        let info = describe_device(DeviceId::Cpu);
        assert_eq!(info.backend, BackendKind::Cpu);
        assert_eq!(info.kind, DeviceKind::Cpu);
        assert!(info.unified_memory);
        assert!(info.identity.is_empty());
        assert_eq!(info.status, DeviceStatus::NotProbed);
        assert!(!info.name.is_empty());
    }
}
