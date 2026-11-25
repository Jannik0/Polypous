//! Polypous: build, train and run neural networks reliably on consumer hardware.
//!
//! Responsibility: public facade: `Engine`, `Model`, `Session`, `Pipeline`, `Trainer`, events,
//! errors.
//!
//! Training (public-api.md §1.4): [`Engine::trainer`], [`Engine::resume_trainer`] and
//! [`Trainer`]; the nested config and result types are in [`training`].
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

// `Error` is the documented public shape `{ kind, message, context }` (public-api.md §2 rule 7)
// with optional context strings; it is returned by value on the cold error path, so its size
// doesn't matter. Boxing it would change the public type.
#![allow(clippy::result_large_err)]

mod cancel;
mod device;
mod engine;
mod error;
mod graph;
mod model;
pub mod probe;
mod tensor;
pub mod tensor_file;
pub mod training;

pub use cancel::CancellationToken;
pub use device::{
    AdapterKind, BackendKind, DeviceCapabilities, DeviceId, DeviceInfo, DeviceKind, DeviceStatus,
    PhysicalIdentity, UnavailableReason,
};
pub use engine::{Engine, EngineConfig};
pub use error::{Error, ErrorContext, ErrorKind};
pub use graph::{Diagnostic, Graph, GraphValidateOptions, ValidationReport};
pub use model::{Dim, GraphLoadOptions, Model, ModelInfo, RunOutput, TensorSpec};
pub use probe::{Isolation, ProbeEvent, ProbeOptions, ProbeReport};
pub use tensor::{ComputeDtype, Dtype, NamedTensor, TensorData};
pub use training::{
    InMemoryDataset, ResumeOptions, Trainer, TrainerInfo, TrainerOptions, TrainingConfig,
    TrainingEvent, TrainingOutcome,
};
