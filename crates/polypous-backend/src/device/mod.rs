//! Backend kinds, device discovery and probing (spec: `docs/specs/placement.md` §1; ADR-0003,
//! ADR-0011, ADR-0013).
//!
//! These are the in-process building blocks. The prober process (ADR-0013, issue #14) runs
//! [`list_devices`], [`describe_device`] and [`probe_device`] in a separate process and sends the
//! results as JSON lines; every type here is serde-serializable with stable `snake_case` names.
//! Calling the functions that open devices in the app process is only safe for tests and the
//! in-process fallback: a broken GPU driver can crash the process natively.
//!
//! - [`list_devices`]: device ids of every compiled-in backend, without opening devices.
//! - [`describe_device`]: opens a device and reads name, memory, identity, capabilities.
//! - [`enumerate_devices`] / [`enumerate_devices_with`]: both, for all devices. Software
//!   rasterizers (lavapipe, …) are listed but not opened unless [`EnumerateOptions`] opts in.
//! - [`probe_device`]: initialize, allocate, run a tiny kernel, compare the result.
//! - [`select_primary_gpu`]: the priority rule of placement.md §1, on a device list.
//!
//! None of these functions fails or panics: problems become a [`DeviceStatus::Unavailable`].
//! Calls that touch devices are serialized by a process-wide lock, so they may be called from any
//! thread.

mod discover;
mod info;
mod probe;
mod resolve;
mod select;

pub(crate) use discover::device_id_of;
pub use discover::{
    EnumerateOptions, EnumerationError, describe_device, enumerate_devices, enumerate_devices_with,
    list_backend_devices, list_devices, skipped_device,
};
pub use info::{
    AdapterKind, BackendKind, Capabilities, DeviceId, DeviceInfo, DeviceKind, DeviceStatus,
    PhysicalIdentity, UnavailableReason,
};
pub use probe::probe_device;
pub(crate) use resolve::resolve_device;
pub use select::{group_physical_devices, select_primary_gpu};
