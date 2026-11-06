//! Where probed devices come from: the real backends, or, in test builds with the
//! `test-fault-injection` feature, fake devices and injected faults ([`super::faults`]).

use std::time::Instant;

use crate::device::{
    self, BackendKind, DeviceId, DeviceInfo, DeviceStatus, EnumerationError, enumeration_details,
};

use super::fingerprint;
use super::protocol::EnumeratedDevice;

/// The device operations a probe session needs. The real implementation is [`RealDevices`];
/// fault injection wraps it or replaces it with fakes.
pub(crate) trait DeviceSource {
    /// Backends to enumerate, in [`BackendKind::ALL`] order.
    fn backends(&self) -> Vec<BackendKind>;
    /// The devices of one backend, without opening them.
    fn list(&self, backend: BackendKind) -> Result<Vec<DeviceId>, EnumerationError>;
    /// What enumeration tells about a listed device, without opening it.
    fn details(&self, id: DeviceId) -> EnumeratedDevice;
    /// Opens, describes and probes a device. Never panics.
    fn describe_and_probe(&self, id: DeviceId) -> DeviceInfo;
}

/// The compiled-in backends, through `crate::device`.
pub(crate) struct RealDevices;

impl DeviceSource for RealDevices {
    fn backends(&self) -> Vec<BackendKind> {
        BackendKind::compiled_in()
    }

    fn list(&self, backend: BackendKind) -> Result<Vec<DeviceId>, EnumerationError> {
        device::list_backend_devices(backend)
    }

    fn details(&self, id: DeviceId) -> EnumeratedDevice {
        let details = enumeration_details(id);
        // No PCI address from the graphics API (Intel iGPUs through Vulkan on Windows): the key of
        // the one GPU in the OS fingerprint with the same PCI ids (placement.md §1).
        let key = details.key.or_else(|| {
            let (vendor, device) = details.pci_ids?;
            fingerprint::current()?.key_for_pci_ids(vendor, device)
        });
        EnumeratedDevice {
            id,
            key,
            enum_fingerprint: details.fingerprint,
        }
    }

    fn describe_and_probe(&self, id: DeviceId) -> DeviceInfo {
        describe_and_probe(id)
    }
}

/// The device source of this process: [`RealDevices`], unless a test build sets the fault-injection
/// variable.
pub(crate) fn device_source() -> Box<dyn DeviceSource> {
    #[cfg(feature = "test-fault-injection")]
    if let Some(faults) = super::faults::Faults::from_env() {
        return faults.into_source(Box::new(RealDevices));
    }
    Box::new(RealDevices)
}

/// Describes one device and, if it could be opened, probes it, logging both steps.
pub(crate) fn describe_and_probe(id: DeviceId) -> DeviceInfo {
    tracing::debug!(device = %id, "describing device");
    let started = Instant::now();
    let mut info = device::describe_device(id);
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
        info.status = device::probe_device(id);
        tracing::debug!(
            device = %id,
            status = %info.status,
            elapsed_ms = started.elapsed().as_millis(),
            "probed device"
        );
    }
    info
}
