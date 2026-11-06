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

/// Serializes everything that enumerates or opens devices. CubeCL guards its own wgpu instances
/// against concurrent creation (the Vulkan loader crashes when instances come and go on several
/// threads at once), but the first enumeration of a backend creates one more for wgpu's adapter
/// info, so device work here runs one call at a time. Compute doesn't take this lock; see
/// [`ENUMERATED`] for how enumeration stays away from it.
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

/// The successful enumeration of each backend, kept for the rest of the process (placement.md §1,
/// "When enumeration happens").
///
/// # Why enumeration is cached
///
/// Enumerating Vulkan devices creates a temporary wgpu instance (Burn's `Device::enumerate` creates
/// five, one per adapter type), so the Vulkan loader loads every installed ICD and unloads the
/// device-less ones again (`dlclose`). That races with Vulkan compute on other threads: on Debian 13
/// with stock Mesa the process crashes with SIGSEGV inside `libvulkan.so.1` (issue #26). Taking
/// [`DEVICE_LOCK`] doesn't help, because compute never takes it. So each backend is enumerated at
/// most once per process, and everything that needs the device list ([`list_backend_devices`],
/// [`describe_device`], [`probe_device`](super::probe_device), the `DynTensor` device checks,
/// the facade's discovery) reuses it. Only successful results are cached: a failed enumeration is
/// retried on the next call.
///
/// Devices that appear later (a newly plugged-in eGPU) are not seen until the process restarts;
/// CubeCL keeps a device's client for the whole process anyway. The cache holds hardware facts
/// only, the same for every `Engine`, so it is not engine state (public-api.md §2 rule 9).
static ENUMERATED: EnumerationCache = EnumerationCache::new();

/// Per-backend device lists. A handful of entries; lookups are linear.
pub(crate) struct EnumerationCache {
    lists: std::sync::Mutex<Vec<(BackendKind, Vec<DeviceId>)>>,
}

impl EnumerationCache {
    pub(crate) const fn new() -> Self {
        Self {
            lists: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn cached(&self, backend: BackendKind) -> Option<Vec<DeviceId>> {
        // Entries are pushed whole, so a panic elsewhere can't leave one half-written.
        self.lists
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|(cached, _)| *cached == backend)
            .map(|(_, ids)| ids.clone())
    }

    /// The cached list of `backend`, or the result of `enumerate`, which is cached if it
    /// succeeded. `enumerate` runs under [`DEVICE_LOCK`], at most once per backend unless it
    /// fails: a second thread that misses the cache waits for the first one's result.
    pub(crate) fn list_with(
        &self,
        backend: BackendKind,
        enumerate: impl FnOnce(BackendKind) -> Result<Vec<DeviceId>, EnumerationError>,
    ) -> Result<Vec<DeviceId>, EnumerationError> {
        if let Some(ids) = self.cached(backend) {
            return Ok(ids);
        }
        let _guard = lock_devices();
        if let Some(ids) = self.cached(backend) {
            return Ok(ids);
        }
        let ids = enumerate(backend)?;
        self.lists
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((backend, ids.clone()));
        Ok(ids)
    }
}

/// Number of real enumerations per backend in this process, in [`BackendKind::ALL`] order. Tests
/// check that the cache keeps it at one.
#[cfg(test)]
pub(crate) static REAL_ENUMERATIONS: [std::sync::atomic::AtomicUsize; 4] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 4];

/// Lists the devices of one backend without opening them.
///
/// The backend is enumerated on the first successful call only; later calls return the same list
/// without touching the driver (see [`ENUMERATED`] for why). A backend that is not compiled in, or
/// whose runtime library is missing (no `libcuda`, no Vulkan loader or ICD), has no devices. A Burn
/// panic during enumeration becomes an [`EnumerationError`] and is not cached.
pub fn list_backend_devices(backend: BackendKind) -> Result<Vec<DeviceId>, EnumerationError> {
    if device_type(backend).is_none() {
        return Ok(Vec::new());
    }
    ENUMERATED.list_with(backend, enumerate_backend)
}

/// Enumerates one compiled-in backend through Burn. For Vulkan and Metal it also records wgpu's
/// adapter info for [`describe_device`], so that no later call creates a wgpu instance of ours.
/// Runs under [`DEVICE_LOCK`].
fn enumerate_backend(backend: BackendKind) -> Result<Vec<DeviceId>, EnumerationError> {
    #[cfg(test)]
    if let Some(index) = BackendKind::ALL.iter().position(|&kind| kind == backend) {
        REAL_ENUMERATIONS[index].fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    let Some(device_type) = device_type(backend) else {
        return Ok(Vec::new());
    };
    catch_unwind(AssertUnwindSafe(|| {
        let ids = Device::enumerate(device_type)
            .into_iter()
            .filter_map(|device| device_id_of(&device))
            .filter(|id| id.backend() == backend)
            .collect();
        #[cfg(any(feature = "vulkan", feature = "metal"))]
        gpu::wgpu_info::record_adapters(backend);
        ids
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
/// what is known from `id` and the status is unavailable with the reason. An id that
/// [`list_backend_devices`] doesn't list is not opened (no driver is touched once the backend's
/// list is cached). Opening a GPU loads its driver, which can crash the process natively; run it
/// in the prober process (ADR-0013).
pub fn describe_device(id: DeviceId) -> DeviceInfo {
    // Checked before taking the device lock: enumeration takes it too.
    let result = super::resolve::resolve_device(id).and_then(|_| {
        let _guard = lock_devices();
        catch_unwind(AssertUnwindSafe(|| describe(id)))
            .unwrap_or_else(|payload| Err(format!("panic: {}", panic_message(&*payload))))
    });
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

/// Options for device discovery ([`enumerate_devices_with`]; placement.md §1).
///
/// JSON: an object, `{}` for the defaults, for example `{"software_rasterizers":true}`. Unknown
/// fields are rejected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnumerateOptions {
    /// Whether to open and probe software rasterizers ([`DeviceId::is_software_rasterizer`]:
    /// lavapipe, SwiftShader, … through Vulkan or Metal). Default `false`: they are listed with
    /// the status unavailable / [`UnavailableReason::SoftwareRasterizerSkipped`] and never opened,
    /// because they are never the primary GPU and some (lavapipe with Mesa 25.0 on Debian 13)
    /// crash the process on their first compute work.
    pub software_rasterizers: bool,
}

impl EnumerateOptions {
    /// Whether discovery with these options lists `id` without opening it. Such a device is
    /// described by [`skipped_device`] instead of [`describe_device`] and is not probed.
    pub fn skips(&self, id: DeviceId) -> bool {
        id.is_software_rasterizer() && !self.software_rasterizers
    }
}

/// The description of a device that discovery lists but does not open
/// ([`EnumerateOptions::skips`]): what is known from `id` alone, with the status unavailable /
/// [`UnavailableReason::SoftwareRasterizerSkipped`], `isolated: false`. Opens nothing.
pub fn skipped_device(id: DeviceId) -> DeviceInfo {
    DeviceInfo {
        status: DeviceStatus::Unavailable {
            reason: UnavailableReason::SoftwareRasterizerSkipped,
            isolated: false,
        },
        ..placeholder(id)
    }
}

/// Lists and describes the devices of every compiled-in backend with the default
/// [`EnumerateOptions`] (placement.md §1): software rasterizers are listed but not opened. Never
/// fails or panics; see [`enumerate_devices_with`].
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    enumerate_devices_with(&EnumerateOptions::default())
}

/// Lists the devices of every compiled-in backend ([`list_devices`]) and describes each one
/// ([`describe_device`]), except those `options` skip, which get [`skipped_device`]. Never fails
/// or panics.
///
/// Explicit [`describe_device`] / [`probe_device`](super::probe_device) calls are not affected by
/// the options: they open the device they are given.
pub fn enumerate_devices_with(options: &EnumerateOptions) -> Vec<DeviceInfo> {
    describe_listed(list_devices(), options, describe_device)
}

/// [`enumerate_devices_with`] on a given id list, with the describe step injected (tests).
fn describe_listed(
    ids: Vec<DeviceId>,
    options: &EnumerateOptions,
    describe: impl Fn(DeviceId) -> DeviceInfo,
) -> Vec<DeviceInfo> {
    ids.into_iter()
        .map(|id| {
            if options.skips(id) {
                skipped_device(id)
            } else {
                describe(id)
            }
        })
        .collect()
}

/// What enumeration alone tells about a listed device, without opening it (see
/// [`enumeration_details`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct EnumerationDetails {
    /// Stable key (`pci:<address>`) if the graphics API reported the PCI address.
    pub(crate) key: Option<String>,
    /// Adapter name, PCI vendor/device ids and driver strings.
    pub(crate) fingerprint: Option<String>,
    /// PCI vendor and device id, for finding the key elsewhere when the address is missing.
    pub(crate) pci_ids: Option<(u32, u32)>,
}

/// What enumeration alone tells about a listed device, without opening it, from wgpu's adapter info
/// recorded with the backend's enumeration. Empty for CUDA, the CPU, and when nothing was recorded.
/// Used by the prober and the probe cache (placement.md §1, "Probe cache").
pub(crate) fn enumeration_details(id: DeviceId) -> EnumerationDetails {
    match id {
        #[cfg(feature = "vulkan")]
        DeviceId::Vulkan { adapter, index } => {
            gpu::wgpu_info::details(cubecl::wgpu::wgpu::Backend::Vulkan, adapter, index)
        }
        #[cfg(feature = "metal")]
        DeviceId::Metal { adapter, index } => {
            gpu::wgpu_info::details(cubecl::wgpu::wgpu::Backend::Metal, adapter, index)
        }
        #[allow(unreachable_patterns)]
        _ => EnumerationDetails::default(),
    }
}

/// The description of a device from its id alone, with the status [`DeviceStatus::NotProbed`]:
/// name = the id, kind from the adapter type, everything else unknown. Opens nothing. The prober's
/// supervisor uses it for devices that crashed or hung the prober before they could be described.
pub fn unprobed_device(id: DeviceId) -> DeviceInfo {
    placeholder(id)
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

/// The CPU's model name from the OS, without loading any driver. Surrounding whitespace is
/// removed (Intel pads the Windows registry value); an empty name counts as unknown.
fn cpu_name() -> Option<String> {
    os_cpu_name()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
}

/// Linux: the first `model name` line of `/proc/cpuinfo`.
#[cfg(target_os = "linux")]
fn os_cpu_name() -> Option<String> {
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
        .map(|(_, value)| value.to_string())
}

/// Windows: the marketing name from the registry (for example
/// `12th Gen Intel(R) Core(TM) i7-12800H`), else `PROCESSOR_IDENTIFIER`
/// (`Intel64 Family 6 Model 154 …`).
#[cfg(windows)]
fn os_cpu_name() -> Option<String> {
    windows_registry_cpu_name()
        .filter(|name| !name.trim().is_empty())
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
}

/// `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString`.
#[cfg(windows)]
fn windows_registry_cpu_name() -> Option<String> {
    windows_registry::LOCAL_MACHINE
        .open(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        .ok()?
        .get_string("ProcessorNameString")
        .ok()
}

/// Other systems: unknown (the caller falls back to `CPU (<arch>)`).
#[cfg(not(any(target_os = "linux", windows)))]
fn os_cpu_name() -> Option<String> {
    None
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
    pub(super) mod wgpu_info {
        use std::sync::{Mutex, PoisonError};

        use cubecl::wgpu::wgpu;

        use super::super::super::info::BackendKind;
        use super::{AdapterKind, DeviceInfo};

        /// wgpu's adapter info per graphics API, read once with the backend's enumeration
        /// ([`super::super::ENUMERATED`] explains why nothing enumerates twice).
        static ADAPTERS: Mutex<Vec<(wgpu::Backend, Vec<wgpu::AdapterInfo>)>> =
            Mutex::new(Vec::new());

        /// Reads and keeps the adapter info of `backend`'s graphics API, if it has one. Called
        /// once per backend, from its enumeration (under the device lock).
        pub(in super::super) fn record_adapters(backend: BackendKind) {
            let api = match backend {
                #[cfg(feature = "vulkan")]
                BackendKind::Vulkan => wgpu::Backend::Vulkan,
                #[cfg(feature = "metal")]
                BackendKind::Metal => wgpu::Backend::Metal,
                _ => return,
            };
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: api.into(),
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapters = cubecl::future::block_on(instance.enumerate_adapters(api.into()))
                .iter()
                .map(wgpu::Adapter::get_info)
                .collect();
            // The list is replaced whole, so a panic elsewhere can't leave it half-written.
            let mut cached = ADAPTERS.lock().unwrap_or_else(PoisonError::into_inner);
            cached.retain(|(cached_api, _)| *cached_api != api);
            cached.push((api, adapters));
        }

        /// Enumeration details of the `index`-th adapter of type `adapter` (see
        /// `enumeration_details`). Empty if nothing was recorded.
        pub(in super::super) fn details(
            backend: wgpu::Backend,
            adapter: AdapterKind,
            index: usize,
        ) -> super::super::EnumerationDetails {
            let adapters = recorded_adapters(backend);
            // CubeCL gives every software adapter index 0.
            let index = if adapter == AdapterKind::Cpu {
                0
            } else {
                index
            };
            let Some(info) = adapters
                .iter()
                .filter(|candidate| adapter_kind(candidate.device_type) == adapter)
                .nth(index)
            else {
                return super::super::EnumerationDetails::default();
            };
            let pci = info.device_pci_bus_id.trim();
            let key = (!pci.is_empty()).then(|| format!("pci:{}", pci.to_ascii_lowercase()));
            let fingerprint = format!(
                "name={};vendor={:#06x};device={:#06x};driver={};driver_info={}",
                info.name.trim(),
                info.vendor,
                info.device,
                info.driver.trim(),
                info.driver_info.trim()
            );
            super::super::EnumerationDetails {
                key,
                fingerprint: Some(fingerprint),
                pci_ids: Some((info.vendor, info.device)),
            }
        }

        fn recorded_adapters(api: wgpu::Backend) -> Vec<wgpu::AdapterInfo> {
            ADAPTERS
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .iter()
                .find(|(cached_api, _)| *cached_api == api)
                .map(|(_, adapters)| adapters.clone())
                .unwrap_or_default()
        }

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
        /// adapter info. Burn/CubeCL keep the adapter info to themselves, so this uses the list
        /// [`record_adapters`] read with the backend's enumeration and picks the `index`-th
        /// adapter of type `adapter`, the same addressing CubeCL uses (cubecl-wgpu runtime.rs,
        /// `adapter_device_ids`). Without a recorded list the fields stay as they are.
        pub(super) fn fill(
            backend: wgpu::Backend,
            adapter: AdapterKind,
            index: usize,
            info: &mut DeviceInfo,
        ) {
            let adapters = recorded_adapters(backend);
            // CubeCL gives every software adapter index 0.
            let index = if adapter == AdapterKind::Cpu {
                0
            } else {
                index
            };
            let Some(adapter_info) = adapters
                .iter()
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
    use super::super::info::AdapterKind;
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

    /// An enumeration step that counts its calls and returns `result`.
    fn counting(
        calls: &std::cell::Cell<usize>,
        result: Result<Vec<DeviceId>, EnumerationError>,
    ) -> impl FnOnce(BackendKind) -> Result<Vec<DeviceId>, EnumerationError> + '_ {
        move |_| {
            calls.set(calls.get() + 1);
            result
        }
    }

    #[test]
    fn enumeration_is_cached_per_backend() {
        let cache = EnumerationCache::new();
        let calls = std::cell::Cell::new(0);
        let gpus = vec![
            vulkan(AdapterKind::DiscreteGpu, 0),
            vulkan(AdapterKind::IntegratedGpu, 0),
        ];
        for _ in 0..5 {
            assert_eq!(
                cache.list_with(BackendKind::Vulkan, counting(&calls, Ok(gpus.clone()))),
                Ok(gpus.clone())
            );
        }
        assert_eq!(calls.get(), 1);

        // Each backend has its own entry; an empty list is a result too.
        for _ in 0..3 {
            assert_eq!(
                cache.list_with(BackendKind::Cuda, counting(&calls, Ok(Vec::new()))),
                Ok(Vec::new())
            );
        }
        assert_eq!(calls.get(), 2);
        assert_eq!(cache.cached(BackendKind::Vulkan), Some(gpus));
    }

    #[test]
    fn failed_enumerations_are_not_cached() {
        let cache = EnumerationCache::new();
        let calls = std::cell::Cell::new(0);
        let err = EnumerationError {
            backend: BackendKind::Vulkan,
            message: "panic: loader".into(),
        };
        for attempt in 1..=2 {
            assert_eq!(
                cache.list_with(BackendKind::Vulkan, counting(&calls, Err(err.clone()))),
                Err(err.clone())
            );
            assert_eq!(calls.get(), attempt);
        }
        assert_eq!(cache.cached(BackendKind::Vulkan), None);

        let gpus = vec![vulkan(AdapterKind::DiscreteGpu, 0)];
        for _ in 0..3 {
            assert_eq!(
                cache.list_with(BackendKind::Vulkan, counting(&calls, Ok(gpus.clone()))),
                Ok(gpus.clone())
            );
        }
        assert_eq!(calls.get(), 3);
    }

    /// Unlisted ids fail against the cached list, many times, with one enumeration.
    #[test]
    fn unlisted_ids_fail_without_enumerating_again() {
        let cache = EnumerationCache::new();
        let calls = std::cell::Cell::new(0);
        let gpus = vec![vulkan(AdapterKind::DiscreteGpu, 0)];
        let unlisted = [
            vulkan(AdapterKind::DiscreteGpu, 1),
            vulkan(AdapterKind::IntegratedGpu, 3),
            vulkan(AdapterKind::DiscreteGpu, usize::from(u16::MAX) + 1),
        ];
        for _ in 0..100 {
            for id in unlisted {
                let listed =
                    cache.list_with(BackendKind::Vulkan, counting(&calls, Ok(gpus.clone())));
                assert_eq!(
                    super::super::resolve::check_listed(id, listed),
                    Err("no such vulkan device (devices found: vulkan:discrete_gpu:0)".into())
                );
            }
        }
        assert_eq!(calls.get(), 1);
    }

    /// Two threads that miss the cache at the same time enumerate once.
    #[test]
    fn concurrent_first_calls_enumerate_once() {
        let cache = EnumerationCache::new();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    cache.list_with(BackendKind::Cuda, |_| {
                        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        Ok(vec![DeviceId::Cuda { index: 0 }])
                    })
                });
            }
        });
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// The real entry points (listing, describing, probing and `DynTensor` with listed and
    /// unlisted ids) enumerate each backend at most once in this process, whatever other tests
    /// run in parallel.
    #[test]
    fn public_calls_enumerate_each_backend_at_most_once() {
        use crate::{DType, DynTensor, TensorData};
        use std::sync::atomic::Ordering;

        let data = TensorData::new(DType::F32, vec![1], 1f32.to_le_bytes().to_vec()).unwrap();
        let unlisted = [
            DeviceId::Cuda { index: 4096 },
            vulkan(AdapterKind::DiscreteGpu, 4096),
            vulkan(AdapterKind::Other, 4096),
            DeviceId::Metal {
                adapter: AdapterKind::IntegratedGpu,
                index: 4096,
            },
        ];
        for _ in 0..3 {
            let _ = list_devices();
            for backend in BackendKind::ALL {
                let _ = list_backend_devices(backend);
            }
            for id in unlisted {
                assert!(DynTensor::from_data(&data, id).is_err(), "{id}");
                assert!(!describe_device(id).status.is_available(), "{id}");
                assert!(!super::super::probe_device(id).is_available(), "{id}");
            }
        }
        for (backend, count) in BackendKind::ALL.iter().zip(&REAL_ENUMERATIONS) {
            let count = count.load(Ordering::SeqCst);
            assert!(count <= 1, "{backend} enumerated {count} times");
            if backend.is_compiled_in() && list_backend_devices(*backend).is_ok() {
                assert_eq!(count, 1, "{backend}");
            }
        }
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

    fn vulkan(adapter: AdapterKind, index: usize) -> DeviceId {
        DeviceId::Vulkan { adapter, index }
    }

    /// The device list of the reference machine on Debian 13 with stock Mesa: lavapipe is listed
    /// next to the real GPUs (docs/dev-environment.md, Debian notes).
    fn debian_reference_ids() -> Vec<DeviceId> {
        vec![
            DeviceId::Cuda { index: 0 },
            vulkan(AdapterKind::DiscreteGpu, 0),
            vulkan(AdapterKind::IntegratedGpu, 0),
            vulkan(AdapterKind::Cpu, 0),
            DeviceId::Cpu,
        ]
    }

    /// A describe step that records which devices it opened.
    fn recording_describe(
        opened: &std::cell::RefCell<Vec<DeviceId>>,
    ) -> impl Fn(DeviceId) -> DeviceInfo + '_ {
        move |id| {
            opened.borrow_mut().push(id);
            placeholder(id)
        }
    }

    #[test]
    fn software_rasterizers_are_skipped_by_default() {
        let options = EnumerateOptions::default();
        assert!(!options.software_rasterizers);
        let opened = std::cell::RefCell::new(Vec::new());
        let devices = describe_listed(
            debian_reference_ids(),
            &options,
            recording_describe(&opened),
        );

        // Listed in the same order, but lavapipe was not opened.
        let ids: Vec<DeviceId> = devices.iter().map(|info| info.id).collect();
        assert_eq!(ids, debian_reference_ids());
        assert!(!opened.borrow().contains(&vulkan(AdapterKind::Cpu, 0)));
        assert_eq!(opened.borrow().len(), 4);

        let lavapipe = &devices[3];
        assert_eq!(lavapipe.backend, BackendKind::Vulkan);
        assert_eq!(lavapipe.kind, DeviceKind::Cpu);
        assert_eq!(lavapipe.name, "vulkan:cpu:0");
        assert!(lavapipe.unified_memory);
        assert!(lavapipe.identity.is_empty());
        assert_eq!(
            lavapipe.status,
            DeviceStatus::Unavailable {
                reason: UnavailableReason::SoftwareRasterizerSkipped,
                isolated: false,
            }
        );
        assert_eq!(skipped_device(lavapipe.id), *lavapipe);
        assert_eq!(
            serde_json::to_value(&lavapipe.status).unwrap(),
            serde_json::json!({
                "state": "unavailable",
                "reason": {"code": "software_rasterizer_skipped"},
                "isolated": false
            })
        );
        // The other devices went through the describe step unchanged.
        assert!(
            devices
                .iter()
                .filter(|info| info.id != lavapipe.id)
                .all(|info| info.status == DeviceStatus::NotProbed)
        );
    }

    #[test]
    fn software_rasterizers_are_opened_when_opted_in() {
        let options = EnumerateOptions {
            software_rasterizers: true,
        };
        let opened = std::cell::RefCell::new(Vec::new());
        let devices = describe_listed(
            debian_reference_ids(),
            &options,
            recording_describe(&opened),
        );
        assert_eq!(*opened.borrow(), debian_reference_ids());
        assert!(
            devices
                .iter()
                .all(|info| info.status == DeviceStatus::NotProbed)
        );
    }

    #[test]
    fn only_software_rasterizers_are_skipped() {
        let default = EnumerateOptions::default();
        let opt_in = EnumerateOptions {
            software_rasterizers: true,
        };
        for id in [
            vulkan(AdapterKind::Cpu, 0),
            DeviceId::Metal {
                adapter: AdapterKind::Cpu,
                index: 0,
            },
        ] {
            assert!(default.skips(id), "{id}");
            assert!(!opt_in.skips(id), "{id}");
        }
        for id in [
            DeviceId::Cpu,
            DeviceId::Cuda { index: 0 },
            vulkan(AdapterKind::DiscreteGpu, 0),
            vulkan(AdapterKind::IntegratedGpu, 0),
            vulkan(AdapterKind::VirtualGpu, 0),
            vulkan(AdapterKind::Other, 0),
        ] {
            assert!(!default.skips(id), "{id}");
            assert!(!opt_in.skips(id), "{id}");
        }
    }

    #[test]
    fn options_json_is_stable() {
        assert_eq!(
            serde_json::to_string(&EnumerateOptions::default()).unwrap(),
            r#"{"software_rasterizers":false}"#
        );
        assert_eq!(
            serde_json::from_str::<EnumerateOptions>("{}").unwrap(),
            EnumerateOptions::default()
        );
        assert!(
            serde_json::from_str::<EnumerateOptions>(r#"{"software_rasterizers":true}"#)
                .unwrap()
                .software_rasterizers
        );
        assert!(serde_json::from_str::<EnumerateOptions>(r#"{"software":true}"#).is_err());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_is_enumerated_and_described() {
        assert!(
            enumerate_devices()
                .iter()
                .any(|info| info.id == DeviceId::Cpu && info.status == DeviceStatus::NotProbed)
        );
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

    /// The Windows CI runner and the reference machine have the registry value; the name must be
    /// the marketing name, not `PROCESSOR_IDENTIFIER`'s "Intel64 Family 6 Model …".
    #[cfg(windows)]
    #[test]
    fn windows_cpu_name_comes_from_the_registry() {
        let registry = windows_registry_cpu_name().expect("ProcessorNameString exists");
        let name = cpu_name().expect("a CPU name");
        assert_eq!(name, registry.trim());
        assert!(!name.contains(" Family "), "{name}");
    }
}
