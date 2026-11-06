//! Device probing (public-api.md §1.1; ADR-0013; placement.md §1, "Crash safety").
//!
//! [`Engine::new`](crate::Engine::new) probes automatically: probe cache, else a prober process,
//! else in this process with a crash journal. This module lets apps run the same probe themselves
//! ([`probe_devices`], [`Engine::refresh_devices`](crate::Engine::refresh_devices)) and lets a Rust
//! app serve as its own prober ([`run_prober_if_requested`]).
//!
//! # Which program is the prober
//!
//! 1. [`ProbeOptions::prober_path`] / [`EngineConfig::prober_path`](crate::EngineConfig::prober_path),
//!    if set: started as `<path> --polypous-prober --request <json>`. `polyp` and every app that
//!    calls [`run_prober_if_requested`] understand that command line (the tiny `polypous-probe`
//!    executable of M7 will too).
//! 2. Otherwise the current executable, **if it called [`run_prober_if_requested`]** (the first
//!    line of `main`); `polyp` does.
//! 3. Otherwise there is no prober: with [`Isolation::Auto`] the probe runs in process with the
//!    crash journal and the report says it was not isolated.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use polypous_backend::probe as backend;
use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;
use crate::device::{self, DeviceId, DeviceInfo};
use crate::error::{Error, ErrorKind, guard};

/// Where device probes run (ADR-0013 §4). JSON: `"auto"`, `"out_of_process"`, `"in_process"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Isolation {
    /// In a prober process if one is available, else in this process with the crash journal.
    #[default]
    Auto,
    /// Only in a prober process. If none can be started, probing fails with
    /// [`ErrorKind::BackendUnavailable`].
    OutOfProcess,
    /// In this process, with the crash journal: a driver that crashes takes the app down once,
    /// and the device is reported `crashed during probe` from the next start on.
    InProcess,
}

/// Default of [`ProbeOptions::timeout_per_device_ms`]: 60 s (the first probe of a GPU compiles
/// kernels).
pub const DEFAULT_TIMEOUT_PER_DEVICE_MS: u64 = 60_000;

/// Longest accepted timeout: one day.
const MAX_TIMEOUT_MS: u64 = 24 * 60 * 60 * 1000;

/// Options of [`probe_devices`] and [`Engine::refresh_devices`](crate::Engine::refresh_devices).
///
/// JSON: an object, `{}` for the defaults, for example `{"force":true}`. Unknown fields are an
/// [`ErrorKind::InvalidInput`] error. The struct is `#[non_exhaustive]`: start from
/// `ProbeOptions::default()` and set fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ProbeOptions {
    /// Ignore the probe cache and probe every device again, including devices recorded as
    /// `crashed during probe` / `probe timed out` (out of process only: in process, such a device
    /// is never opened again under the same driver). Default `false`.
    pub force: bool,
    /// Probe only these devices; `null`/`None` (default): all. The report lists only them.
    pub devices: Option<Vec<DeviceId>>,
    /// Where probes run. Default [`Isolation::Auto`].
    pub isolation: Isolation,
    /// How long the prober may stay silent, per device (and for start-up and enumeration), in
    /// milliseconds; then it is killed and the device is `probe timed out`. Default 60000.
    /// Must be between 1 and 86400000.
    pub timeout_per_device_ms: u64,
    /// Open and probe software rasterizers (lavapipe, SwiftShader, …), with the meaning of
    /// [`EngineConfig::software_rasterizers`](crate::EngineConfig::software_rasterizers).
    /// Default `false`.
    pub software_rasterizers: bool,
    /// The prober executable (see the module docs). `None` (default): the current executable if
    /// it called [`run_prober_if_requested`]; for [`Engine::refresh_devices`](crate::Engine::refresh_devices)
    /// the engine's [`EngineConfig::prober_path`](crate::EngineConfig::prober_path) first.
    pub prober_path: Option<PathBuf>,
    /// The probe cache directory. `None` (default): the platform cache directory
    /// (`~/.cache/polypous` on Linux, `%LOCALAPPDATA%\polypous\cache` on Windows,
    /// `~/Library/Caches/polypous` on macOS); for `refresh_devices` the engine's
    /// [`EngineConfig::probe_cache_dir`](crate::EngineConfig::probe_cache_dir) first.
    pub probe_cache_dir: Option<PathBuf>,
}

impl Default for ProbeOptions {
    fn default() -> Self {
        ProbeOptions {
            force: false,
            devices: None,
            isolation: Isolation::Auto,
            timeout_per_device_ms: DEFAULT_TIMEOUT_PER_DEVICE_MS,
            software_rasterizers: false,
            prober_path: None,
            probe_cache_dir: None,
        }
    }
}

impl ProbeOptions {
    /// Reads options from their JSON form (public-api.md §2 rule 3).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] if `json` is not valid JSON, not an object, or has an unknown
    /// field.
    pub fn from_json(json: &str) -> Result<ProbeOptions, Error> {
        guard("probe_options_from_json", || {
            crate::engine::object_from_json(json, "The probe options are not valid.")
        })
    }

    /// The options' JSON form.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Progress of a probe, in order (public-api.md §4): `probe_started` first, `probe_finished`
/// last (unless the probe returns an error).
///
/// JSON: `{"event":"probe_started"}`, `{"event":"device_enumerated","id":{…}}`,
/// `{"event":"device_probed","device":{…},"cached":false}`,
/// `{"event":"probe_finished","report":{…}}`.
// A handful of events per probe: their size doesn't matter, and unboxed variants are easier to
// match for apps.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProbeEvent {
    /// The probe started.
    ProbeStarted,
    /// A device was enumerated (by the prober, or in process). Devices taken from the cache
    /// without enumeration don't get this event.
    DeviceEnumerated {
        /// The device.
        id: DeviceId,
    },
    /// A device's result is known. `device.primary` is always `false` here: the primary GPU is
    /// chosen when all devices are known (see the report).
    DeviceProbed {
        /// The device with its status.
        device: DeviceInfo,
        /// Whether the result came from the probe cache.
        cached: bool,
    },
    /// The probe finished.
    ProbeFinished {
        /// The result, the same as the return value.
        report: ProbeReport,
    },
}

/// The result of a probe (public-api.md §1.1). Plain JSON data; an app may produce one itself and
/// pass it as [`EngineConfig::probe_report`](crate::EngineConfig::probe_report).
///
/// JSON: `{"devices":[…],"isolated":true,"cache_hits":[…],"prober_starts":1,
/// "fingerprint_known":true,"warnings":[]}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ProbeReport {
    /// The devices in probe order, with status and reason; the primary GPU has `primary: true`.
    pub devices: Vec<DeviceInfo>,
    /// Whether every probe result in `devices`, cached or new, came from a prober process. `false`
    /// means some device was probed in process (the fallback); its status says `isolated: false`.
    pub isolated: bool,
    /// Devices whose result came from the probe cache, without being opened.
    pub cache_hits: Vec<DeviceId>,
    /// Prober processes started by this probe: 0 for a complete cache hit or in-process probing.
    pub prober_starts: u32,
    /// Whether the OS driver fingerprint was known. If not, a prober enumerated the devices to
    /// compare them with the cache (macOS for now).
    pub fingerprint_known: bool,
    /// Problems worth showing: a prober crash or hang, a fallback to in-process probing, a
    /// device remembered from a crash.
    pub warnings: Vec<String>,
}

impl Default for ProbeReport {
    fn default() -> Self {
        ProbeReport {
            devices: Vec::new(),
            isolated: true,
            cache_hits: Vec::new(),
            prober_starts: 0,
            fingerprint_known: false,
            warnings: Vec::new(),
        }
    }
}

impl ProbeReport {
    /// Converts the backend's report and marks the primary GPU.
    pub(crate) fn from_backend(report: backend::ProbeReport) -> Self {
        let backend::ProbeReport {
            devices,
            isolated,
            cache_hits,
            prober_starts,
            fingerprint_known,
            warnings,
        } = report;
        ProbeReport {
            devices: device::device_list(devices),
            isolated,
            cache_hits: cache_hits
                .into_iter()
                .map(device::from_backend_id)
                .collect(),
            prober_starts,
            fingerprint_known,
            warnings,
        }
    }
}

/// Set by [`run_prober_if_requested`]: this executable can serve as the prober. A fact about the
/// executable, the same for every engine, so not engine state (public-api.md §2 rule 9).
static PROBER_HOOK: AtomicBool = AtomicBool::new(false);

/// Lets Polypous use this executable as its prober (ADR-0013 §3). Call it as the **first line of
/// `main`** in Rust apps that link Polypous.
///
/// When Polypous starts this executable as the prober (`--polypous-prober --request <json>`),
/// this call runs the probe, writes the protocol to stdout and **exits the process**. Otherwise it
/// returns normally and records that this executable can be the prober, so `Engine::new` probes
/// devices in a separate process instead of in process.
pub fn run_prober_if_requested() {
    PROBER_HOOK.store(true, Ordering::SeqCst);
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    if !backend::is_prober_invocation(&args) {
        return;
    }
    let code = match backend::parse_prober_args(&args) {
        Ok(request) => backend::run_prober(&request, true),
        Err(message) => {
            use std::io::Write;
            let _ = writeln!(std::io::stderr(), "polypous prober: {message}");
            backend::EXIT_BAD_REQUEST
        }
    };
    std::process::exit(code);
}

/// Runs the prober in this process for a request in JSON (`{}`: probe everything) and returns the
/// exit code. For prober executables such as `polyp probe --json`; it writes the protocol to
/// stdout (placement.md §1, "Prober protocol"). Unlike a prober started by Polypous it doesn't
/// watch stdin.
pub fn run_prober(request_json: &str) -> i32 {
    match backend::parse_request(request_json) {
        Ok(request) => backend::run_prober(&request, false),
        Err(message) => {
            use std::io::Write;
            let _ = writeln!(std::io::stderr(), "polypous prober: {message}");
            backend::EXIT_BAD_REQUEST
        }
    }
}

/// Runs a device probe without an [`Engine`](crate::Engine): the probe cache, else a prober
/// process (restarted after a crash or hang for the devices not yet reported), else in process
/// with the crash journal (ADR-0013). Blocks until done; events arrive in order on this thread.
///
/// # Errors
///
/// [`ErrorKind::InvalidInput`] for a timeout out of range, [`ErrorKind::Cancelled`] if `cancel`
/// was triggered, [`ErrorKind::BackendUnavailable`] for [`Isolation::OutOfProcess`] without a
/// working prober, [`ErrorKind::Internal`] for a caught panic. Device problems are statuses in the
/// report, never errors.
pub fn probe_devices(
    options: ProbeOptions,
    cancel: &CancellationToken,
    mut on_event: impl FnMut(ProbeEvent),
) -> Result<ProbeReport, Error> {
    guard("probe_devices", || {
        run_probe(&options, cancel, &mut on_event)
    })
}

/// The probe behind [`probe_devices`], `Engine::new` and `Engine::refresh_devices`.
pub(crate) fn run_probe(
    options: &ProbeOptions,
    cancel: &CancellationToken,
    on_event: &mut dyn FnMut(ProbeEvent),
) -> Result<ProbeReport, Error> {
    let config = backend_config(options)?;
    on_event(ProbeEvent::ProbeStarted);
    let result = backend::probe_devices(&config, &|| cancel.is_cancelled(), &mut |event| {
        on_event(match event {
            backend::ProbeEvent::DeviceEnumerated { id } => ProbeEvent::DeviceEnumerated {
                id: device::from_backend_id(id),
            },
            backend::ProbeEvent::DeviceProbed { device, cached } => ProbeEvent::DeviceProbed {
                device: device::device_info(device, false),
                cached,
            },
        });
    });
    let report = match result {
        Ok(report) => ProbeReport::from_backend(report),
        Err(backend::ProbeError::Cancelled) => {
            return Err(Error::new(
                ErrorKind::Cancelled,
                "The device probe was cancelled.",
            ));
        }
        Err(error @ backend::ProbeError::ProberUnavailable(_)) => {
            return Err(Error::new(
                ErrorKind::BackendUnavailable,
                "Devices could not be probed in a separate process.",
            )
            .with_details(error.to_string()));
        }
    };
    on_event(ProbeEvent::ProbeFinished {
        report: report.clone(),
    });
    Ok(report)
}

/// The backend's probe configuration for `options`.
fn backend_config(options: &ProbeOptions) -> Result<backend::ProbeConfig, Error> {
    if !(1..=MAX_TIMEOUT_MS).contains(&options.timeout_per_device_ms) {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "The probe timeout per device must be between 1 ms and one day.",
        )
        .with_details(format!(
            "timeout_per_device_ms = {}",
            options.timeout_per_device_ms
        )));
    }
    let cache_dir = options
        .probe_cache_dir
        .clone()
        .or_else(backend::default_cache_dir);
    if cache_dir.is_none() {
        tracing::warn!("no probe cache directory: every start probes again");
    }
    Ok(backend::ProbeConfig {
        force: options.force,
        devices: options
            .devices
            .as_ref()
            .map(|ids| ids.iter().copied().map(device::to_backend_id).collect()),
        isolation: match options.isolation {
            Isolation::Auto => backend::Isolation::Auto,
            Isolation::OutOfProcess => backend::Isolation::OutOfProcess,
            Isolation::InProcess => backend::Isolation::InProcess,
        },
        timeout_per_device: Duration::from_millis(options.timeout_per_device_ms),
        software_rasterizers: options.software_rasterizers,
        prober: prober_command(options.prober_path.as_ref()),
        cache_dir,
    })
}

/// The prober to start (see the module docs).
fn prober_command(path: Option<&PathBuf>) -> Option<backend::ProberCommand> {
    if let Some(path) = path {
        return Some(backend::ProberCommand {
            program: path.clone(),
        });
    }
    if !PROBER_HOOK.load(Ordering::SeqCst) {
        return None;
    }
    match std::env::current_exe() {
        Ok(program) => Some(backend::ProberCommand { program }),
        Err(error) => {
            tracing::warn!(%error, "cannot locate the current executable to use it as the prober");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_json_round_trips() {
        let defaults = ProbeOptions::default();
        assert_eq!(
            defaults.to_json(),
            r#"{"force":false,"devices":null,"isolation":"auto","timeout_per_device_ms":60000,"software_rasterizers":false,"prober_path":null,"probe_cache_dir":null}"#
        );
        assert_eq!(ProbeOptions::from_json("{}"), Ok(defaults.clone()));
        let options = ProbeOptions::from_json(
            r#"{"force":true,"devices":[{"backend":"cpu"}],"isolation":"in_process"}"#,
        )
        .unwrap();
        assert!(options.force);
        assert_eq!(options.devices, Some(vec![DeviceId::Cpu]));
        assert_eq!(options.isolation, Isolation::InProcess);
        assert_eq!(ProbeOptions::from_json(&options.to_json()), Ok(options));
        for bad in [r#"{"isolation":"sandbox"}"#, r#"{"forse":true}"#, "[]"] {
            assert_eq!(
                ProbeOptions::from_json(bad).unwrap_err().kind,
                ErrorKind::InvalidInput
            );
        }
    }

    #[test]
    fn timeouts_are_validated() {
        for timeout in [0, MAX_TIMEOUT_MS + 1] {
            let options = ProbeOptions {
                timeout_per_device_ms: timeout,
                ..ProbeOptions::default()
            };
            let error = probe_devices(options, &CancellationToken::new(), |_| {}).unwrap_err();
            assert_eq!(error.kind, ErrorKind::InvalidInput);
        }
    }

    #[test]
    fn events_and_report_json() {
        let event = ProbeEvent::DeviceEnumerated { id: DeviceId::Cpu };
        assert_eq!(
            serde_json::to_string(&event).unwrap(),
            r#"{"event":"device_enumerated","id":{"backend":"cpu"}}"#
        );
        assert_eq!(
            serde_json::to_string(&ProbeEvent::ProbeStarted).unwrap(),
            r#"{"event":"probe_started"}"#
        );
        let report = ProbeReport::default();
        assert_eq!(
            serde_json::to_string(&report).unwrap(),
            r#"{"devices":[],"isolated":true,"cache_hits":[],"prober_starts":0,"fingerprint_known":false,"warnings":[]}"#
        );
    }

    #[test]
    fn out_of_process_without_a_prober_is_backend_unavailable() {
        let options = ProbeOptions {
            isolation: Isolation::OutOfProcess,
            prober_path: Some(PathBuf::from("/nonexistent/polypous-test-prober")),
            probe_cache_dir: Some(std::env::temp_dir().join("polypous-test-facade-oop")),
            ..ProbeOptions::default()
        };
        let mut events = Vec::new();
        let error = probe_devices(options, &CancellationToken::new(), |event| {
            events.push(event);
        })
        .unwrap_err();
        assert_eq!(error.kind, ErrorKind::BackendUnavailable);
        assert_eq!(events, vec![ProbeEvent::ProbeStarted]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn in_process_probe_reports_the_cpu_not_isolated() {
        let dir =
            std::env::temp_dir().join(format!("polypous-test-facade-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let options = ProbeOptions {
            isolation: Isolation::InProcess,
            probe_cache_dir: Some(dir.clone()),
            devices: Some(vec![DeviceId::Cpu]),
            ..ProbeOptions::default()
        };
        let mut events = Vec::new();
        let report = probe_devices(options, &CancellationToken::new(), |event| {
            events.push(event);
        })
        .unwrap();
        assert!(!report.isolated);
        assert_eq!(report.devices.len(), 1);
        assert_eq!(report.devices[0].id, DeviceId::Cpu);
        assert_eq!(events.first(), Some(&ProbeEvent::ProbeStarted));
        assert!(matches!(
            events.last(),
            Some(ProbeEvent::ProbeFinished { report: finished }) if *finished == report
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
