//! safetensors reader (memory-mapped, single file and sharded) and writer.
//!
//! Implements `docs/specs/formats.md` §1. The file layout is an 8-byte little-endian header length
//! `N`, an `N`-byte JSON header, and the data buffer. [`SafetensorsFile`] validates the whole header
//! (§1.2) before it exposes any tensor; [`TensorView`] borrows a tensor's bytes from the map without
//! copying. [`ShardedSafetensors`] opens a Hugging Face `*.safetensors.index.json` (§1.5).
//! [`serialize`], [`write`] and [`write_file`] produce files (§1.6).
//!
//! ```no_run
//! use polypous_formats::safetensors::SafetensorsFile;
//!
//! # fn main() -> Result<(), polypous_formats::safetensors::Error> {
//! let file = SafetensorsFile::open("model.safetensors")?;
//! for name in file.names() {
//!     let view = file.tensor(name)?;
//!     println!("{name}: {:?} {:?}, {} bytes", view.dtype(), view.shape(), view.data().len());
//! }
//! # Ok(())
//! # }
//! ```

mod dtype;
mod error;
mod file;
mod header;
mod json;
mod sharded;
mod view;
mod writer;

pub use dtype::Dtype;
pub use error::{Error, ErrorKind};
pub use file::SafetensorsFile;
pub use header::TensorInfo;
pub use sharded::ShardedSafetensors;
pub use view::{TensorValues, TensorView, bf16_to_f32, f16_to_f32};
pub use writer::{WriteTensor, serialize, write, write_file};

/// Largest accepted header length `N` in bytes (100 MB), the reference implementation's limit
/// (`docs/specs/formats.md` §1.2, rule 2). The writer refuses to produce a larger header.
pub const MAX_HEADER_LEN: u64 = 100_000_000;

/// Largest accepted number of dimensions of a tensor shape (`docs/specs/formats.md` §1.2, rule 10).
pub const MAX_RANK: usize = 64;

/// Largest accepted size of a sharded index file in bytes (64 MiB, `docs/specs/formats.md` §1.5).
pub const MAX_INDEX_LEN: u64 = 64 * 1024 * 1024;
