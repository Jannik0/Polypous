//! Polypous: build, train and run neural networks reliably on consumer hardware.
//!
//! Responsibility: public facade: `Engine`, `Model`, `Session`, `Pipeline`, `Trainer`, events,
//! errors.
//!
//! See `docs/architecture.md` §2 and `docs/specs/public-api.md`. Every public item follows the
//! wrapper-friendliness rules of public-api.md §2: no generics or lifetimes, handles are cheap
//! clones, data is plain serde, errors are [`Error`], and panics are caught at the boundary.
//!
//! ```no_run
//! use polypous::{Engine, EngineConfig};
//!
//! let engine = Engine::new(EngineConfig::default())?;
//! for device in engine.devices() {
//!     println!("{} {} ({})", device.id, device.name, device.status);
//! }
//! # Ok::<(), polypous::Error>(())
//! ```

mod device;
mod discovery;
mod engine;
mod error;

pub use device::{
    AdapterKind, BackendKind, DeviceCapabilities, DeviceId, DeviceInfo, DeviceKind, DeviceStatus,
    PhysicalIdentity, UnavailableReason,
};
pub use engine::{Engine, EngineConfig};
pub use error::{Error, ErrorContext, ErrorKind};
