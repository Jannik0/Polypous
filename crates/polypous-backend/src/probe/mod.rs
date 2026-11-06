//! Crash-safe device probing (ADR-0013; spec: `docs/specs/placement.md` §1, "Crash safety").
//!
//! - [`protocol`]: the request the app sends and the JSON lines the prober writes, shared by both
//!   sides.
//! - [`run_prober`]: the prober side (`polyp probe --json`, or an app re-launched with
//!   [`protocol::PROBER_FLAG`]).
//! - [`probe_devices`]: the app side: probe cache → prober process (timeouts, restarts,
//!   untrusted-output limits) → in-process fallback with a crash journal.
//!
//! Test builds with the `test-fault-injection` feature can inject crashes, hangs and malformed
//! output (`faults`); that code is not compiled otherwise.

mod absence;
mod cache;
mod collect;
#[cfg(feature = "test-fault-injection")]
mod faults;
mod fingerprint;
mod journal;
mod prober;
pub mod protocol;
mod run;
mod session;
mod source;
mod supervisor;

pub use cache::default_cache_dir;
pub use prober::{
    EXIT_BAD_REQUEST, EXIT_PARENT_GONE, EXIT_STDOUT_FAILED, is_prober_invocation,
    parse_prober_args, parse_request, run_prober,
};
pub use run::{
    DEFAULT_TIMEOUT_PER_DEVICE, Isolation, ProbeConfig, ProbeError, ProbeEvent, ProbeReport,
    probe_devices,
};
pub use supervisor::ProberCommand;
