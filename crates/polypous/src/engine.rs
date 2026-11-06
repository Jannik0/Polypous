//! [`Engine`] and [`EngineConfig`] (public-api.md §1, §1.1, §1.2).

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, PoisonError, RwLock};

use polypous_backend::device as backend;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;
use crate::device::{self, DeviceInfo};
use crate::error::{Error, ErrorKind, guard};
use crate::probe::{
    self, DEFAULT_TIMEOUT_PER_DEVICE_MS, Isolation, ProbeEvent, ProbeOptions, ProbeReport,
};

/// Configuration of an [`Engine`]. Every field is optional; `EngineConfig::default()` is the
/// recommended configuration.
///
/// JSON (public-api.md §1.2): an object, `{}` for the defaults, for example
/// `{"software_rasterizers":true}`. Unknown fields are an [`ErrorKind::InvalidInput`] error, so a
/// misspelled option is never silently ignored. Fields are added as the engine grows (placement
/// options with M4); each one is optional with a documented default.
///
/// The struct is `#[non_exhaustive]`: start from `EngineConfig::default()` and set fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct EngineConfig {
    /// Whether discovery opens and probes software rasterizers: Vulkan or Metal adapters of type
    /// `cpu` such as lavapipe or SwiftShader (placement.md §1). Default `false`: they are listed
    /// as unavailable with the reason
    /// [`SoftwareRasterizerSkipped`](crate::UnavailableReason::SoftwareRasterizerSkipped) and
    /// never opened. They are never the primary GPU (the CPU backend is faster), and some crash
    /// the process on their first compute work (lavapipe with Mesa 25.0 on Debian 13). Turn it on
    /// to test them.
    pub software_rasterizers: bool,
    /// A probe result the app produced itself (for example with [`probe::probe_devices`] in its
    /// own child process). Polypous then uses its devices as they are and starts no probe.
    /// Default `None`.
    pub probe_report: Option<ProbeReport>,
    /// The prober executable ([`probe`] module docs). Default `None`: the current executable if it
    /// called [`probe::run_prober_if_requested`], else in-process probing with the crash journal.
    pub prober_path: Option<PathBuf>,
    /// The probe cache directory (also holds crash journals). Default `None`: the platform cache
    /// directory (`~/.cache/polypous` on Linux).
    pub probe_cache_dir: Option<PathBuf>,
    /// Where `Engine::new` runs device probes. Default [`Isolation::Auto`].
    pub probe_isolation: Isolation,
    /// How long the prober may stay silent per device, in milliseconds. Default 60000.
    pub probe_timeout_per_device_ms: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            software_rasterizers: false,
            probe_report: None,
            prober_path: None,
            probe_cache_dir: None,
            probe_isolation: Isolation::Auto,
            probe_timeout_per_device_ms: DEFAULT_TIMEOUT_PER_DEVICE_MS,
        }
    }
}

/// Reads a config struct from a JSON object (public-api.md §2 rule 3): anything but an object, and
/// unknown fields, are [`ErrorKind::InvalidInput`] with `message`.
pub(crate) fn object_from_json<T: DeserializeOwned>(json: &str, message: &str) -> Result<T, Error> {
    let invalid =
        |details: String| Error::new(ErrorKind::InvalidInput, message).with_details(details);
    // serde reads a struct from an array too; the documented form is an object only.
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|error| invalid(error.to_string()))?;
    if !value.is_object() {
        return Err(invalid("expected a JSON object".to_string()));
    }
    serde_json::from_value(value).map_err(|error| invalid(error.to_string()))
}

impl EngineConfig {
    /// Reads a config from its JSON form, the equivalent of constructing it in Rust
    /// (public-api.md §2 rule 3).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] if `json` is not valid JSON, not an object, or has an unknown
    /// field.
    pub fn from_json(json: &str) -> Result<EngineConfig, Error> {
        guard("engine_config_from_json", || {
            object_from_json(json, "The engine configuration is not valid.")
        })
    }

    /// The config's JSON form.
    pub fn to_json(&self) -> String {
        // Serializing a struct of plain fields cannot fail (a path that is not UTF-8 can); fall
        // back to the defaults' form then.
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// The probe options `Engine::new` uses.
    fn probe_options(&self) -> ProbeOptions {
        ProbeOptions {
            isolation: self.probe_isolation,
            timeout_per_device_ms: self.probe_timeout_per_device_ms,
            software_rasterizers: self.software_rasterizers,
            prober_path: self.prober_path.clone(),
            probe_cache_dir: self.probe_cache_dir.clone(),
            ..ProbeOptions::default()
        }
    }
}

/// The entry point to Polypous: owns the devices of this machine, and later the loaded models.
///
/// A cheap-to-clone handle (`Arc` inside), `Send + Sync + 'static`: clones share one engine, and
/// dropping the last clone frees it (public-api.md §2 rule 2). Several engines may coexist; there
/// is no global state (rule 9).
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    config: EngineConfig,
    state: RwLock<EngineState>,
}

/// What the engine knows about its devices. Replaced whole, so a panic can't leave it half
/// updated.
#[derive(Clone)]
struct EngineState {
    /// The devices in the backend's types, in probe order (the primary GPU is recomputed from it).
    devices: Vec<backend::DeviceInfo>,
    /// The last probe report.
    report: ProbeReport,
}

impl EngineState {
    fn new(report: ProbeReport) -> Self {
        let devices = report
            .devices
            .iter()
            .cloned()
            .map(device::to_backend_info)
            .collect();
        let mut state = EngineState { devices, report };
        state.mark_primary();
        state
    }

    /// Marks the report's devices with the engine's primary GPU.
    fn mark_primary(&mut self) {
        let primary = backend::select_primary_gpu(&self.devices).map(|device| device.id);
        for device in &mut self.report.devices {
            device.primary = primary.is_some_and(|id| device::to_backend_id(device.id) == id);
        }
    }
}

/// How an engine discovers its devices: `Engine::new` probes ([`probe::run_probe`]); tests inject
/// synthetic devices.
pub(crate) type Discovery = dyn FnOnce(&EngineConfig) -> Result<ProbeReport, Error>;

impl Engine {
    /// Creates an engine: discovers every device of every compiled-in backend and probes it
    /// (placement.md §1, ADR-0013), then selects the primary GPU.
    ///
    /// Devices come from [`EngineConfig::probe_report`] if set; otherwise from the probe cache if
    /// it is valid for every device (no prober is started then); otherwise a prober process probes
    /// them, so a crashing or hanging driver takes down only the prober; otherwise (no prober
    /// available) they are probed in this process with a crash journal. A device that fails its
    /// probe is reported as unavailable, never as an error, so this succeeds on any machine.
    ///
    /// The first probe of a GPU compiles kernels and can take seconds.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::BackendUnavailable`] if [`EngineConfig::probe_isolation`] is
    /// [`Isolation::OutOfProcess`] and no prober works, [`ErrorKind::InvalidInput`] for a probe
    /// timeout out of range, [`ErrorKind::Internal`] if Polypous panics (a bug).
    pub fn new(config: EngineConfig) -> Result<Engine, Error> {
        Self::with_discovery(config, Box::new(discover))
    }

    /// [`Engine::new`] with another discovery step (tests).
    pub(crate) fn with_discovery(
        config: EngineConfig,
        discover: Box<Discovery>,
    ) -> Result<Engine, Error> {
        guard("engine_new", || {
            let state = EngineState::new(discover(&config)?);
            log_primary(&state);
            Ok(Engine {
                inner: Arc::new(EngineInner {
                    config,
                    state: RwLock::new(state),
                }),
            })
        })
    }

    /// The configuration this engine was created with.
    pub fn config(&self) -> EngineConfig {
        self.inner.config.clone()
    }

    fn state(&self) -> EngineState {
        self.inner
            .state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Every discovered device with its probe status, in probe order (GPU priority order, the CPU
    /// last). The primary GPU, if any, has [`DeviceInfo::primary`] set.
    ///
    /// JSON: an array of [`DeviceInfo`] (public-api.md §1.2).
    pub fn devices(&self) -> Vec<DeviceInfo> {
        device::device_list(self.state().devices)
    }

    /// The primary GPU: the one GPU Polypous uses, or `None` to run on the CPU only.
    pub fn primary_gpu(&self) -> Option<DeviceInfo> {
        self.devices().into_iter().find(|device| device.primary)
    }

    /// The report of the probe that produced the current device list: whether it was isolated,
    /// which devices came from the cache, how many prober processes were started, warnings.
    pub fn probe_report(&self) -> ProbeReport {
        self.state().report
    }

    /// Probes devices again, for example for a "Re-scan hardware" button or after a driver update
    /// (ADR-0013 §4), and updates this engine's device list: all devices, or with
    /// [`ProbeOptions::devices`] only those. Models that are already loaded keep their placement;
    /// new loads use the new list.
    ///
    /// The options are used as given, except that `prober_path` and `probe_cache_dir` default to
    /// the engine's. Blocks until done; `on_event` gets the events in order on this thread. It may
    /// read the engine (`devices()`), which still shows the old list until the probe finished.
    ///
    /// # Errors
    ///
    /// As [`probe::probe_devices`]; on an error the device list is unchanged.
    pub fn refresh_devices(
        &self,
        options: ProbeOptions,
        cancel: &CancellationToken,
        mut on_event: impl FnMut(ProbeEvent),
    ) -> Result<ProbeReport, Error> {
        guard("engine_refresh_devices", || {
            let mut options = options;
            if options.prober_path.is_none() {
                options.prober_path = self.inner.config.prober_path.clone();
            }
            if options.probe_cache_dir.is_none() {
                options.probe_cache_dir = self.inner.config.probe_cache_dir.clone();
            }
            let partial = options.devices.is_some();
            let report = probe::run_probe(&options, cancel, &mut on_event)?;

            let mut guard = self
                .inner
                .state
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            let mut state = guard.clone();
            let probed: Vec<backend::DeviceInfo> = report
                .devices
                .iter()
                .cloned()
                .map(device::to_backend_info)
                .collect();
            if partial {
                for device in probed {
                    match state.devices.iter_mut().find(|known| known.id == device.id) {
                        Some(known) => *known = device,
                        None => state.devices.push(device),
                    }
                }
                state
                    .devices
                    .sort_by_key(|device| backend::probe_rank(&device.id));
            } else {
                state.devices = probed;
            }
            state.report = report;
            state.mark_primary();
            let report = state.report.clone();
            *guard = state;
            drop(guard);
            log_primary(&self.state());
            Ok(report)
        })
    }
}

/// `Engine::new`'s discovery: the app's report, or a probe.
fn discover(config: &EngineConfig) -> Result<ProbeReport, Error> {
    if let Some(report) = &config.probe_report {
        tracing::info!(
            devices = report.devices.len(),
            "using the probe report from the engine config; no probe"
        );
        return Ok(report.clone());
    }
    let mut on_event = |event: ProbeEvent| {
        if let ProbeEvent::DeviceProbed { device, cached } = &event {
            let skipped = matches!(
                device.status,
                crate::DeviceStatus::Unavailable {
                    reason: crate::UnavailableReason::SoftwareRasterizerSkipped,
                    ..
                }
            );
            if skipped {
                tracing::info!(device = %device.id, cached, "software rasterizer listed but not opened (set software_rasterizers to probe it)");
            } else if device.status.is_available() {
                tracing::debug!(device = %device.id, name = %device.name, status = %device.status, cached, "device probed");
            } else {
                tracing::warn!(device = %device.id, name = %device.name, status = %device.status, cached, "device unavailable");
            }
        }
    };
    probe::run_probe(
        &config.probe_options(),
        &CancellationToken::new(),
        &mut on_event,
    )
}

fn log_primary(state: &EngineState) {
    match state.report.devices.iter().find(|device| device.primary) {
        Some(primary) => {
            tracing::info!(device = %primary.id, name = %primary.name, "primary GPU selected");
        }
        None => tracing::info!("no usable GPU, running on the CPU only"),
    }
}

impl fmt::Debug for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("config", &self.inner.config)
            .field("devices", &self.state().devices.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::tests::reference_machine;
    use crate::{DeviceId, ErrorKind};

    /// A discovery step that reports `devices` as an isolated probe would.
    pub(crate) fn synthetic(devices: Vec<backend::DeviceInfo>) -> Box<Discovery> {
        Box::new(move |_| {
            Ok(ProbeReport {
                devices: device::device_list(devices),
                ..ProbeReport::default()
            })
        })
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "polypous-test-engine-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn engine_is_a_send_sync_static_handle() {
        fn assert_handle<T: Clone + Send + Sync + 'static>() {}
        assert_handle::<Engine>();
        assert_handle::<EngineConfig>();
        assert_handle::<DeviceInfo>();
        assert_handle::<Error>();
        assert_handle::<ProbeReport>();
        assert_handle::<CancellationToken>();
    }

    #[test]
    fn config_json_round_trips() {
        assert_eq!(
            EngineConfig::default().to_json(),
            r#"{"software_rasterizers":false,"probe_report":null,"prober_path":null,"probe_cache_dir":null,"probe_isolation":"auto","probe_timeout_per_device_ms":60000}"#
        );
        assert_eq!(EngineConfig::from_json("{}"), Ok(EngineConfig::default()));
        assert_eq!(
            EngineConfig::from_json(" { } "),
            Ok(EngineConfig::default())
        );
        let mut opt_in = EngineConfig::default();
        assert!(!opt_in.software_rasterizers);
        opt_in.software_rasterizers = true;
        assert_eq!(
            EngineConfig::from_json(r#"{"software_rasterizers":true}"#),
            Ok(opt_in.clone())
        );
        assert_eq!(EngineConfig::from_json(&opt_in.to_json()), Ok(opt_in));

        let probing = EngineConfig::from_json(
            r#"{"prober_path":"/opt/app/polypous-probe","probe_cache_dir":"/tmp/cache",
                "probe_isolation":"in_process","probe_timeout_per_device_ms":5000,
                "probe_report":{"devices":[],"isolated":false}}"#,
        )
        .unwrap();
        assert_eq!(
            probing.prober_path,
            Some(PathBuf::from("/opt/app/polypous-probe"))
        );
        assert_eq!(probing.probe_isolation, Isolation::InProcess);
        assert_eq!(probing.probe_timeout_per_device_ms, 5000);
        assert!(!probing.probe_report.as_ref().unwrap().isolated);
        assert_eq!(EngineConfig::from_json(&probing.to_json()), Ok(probing));
    }

    #[test]
    fn bad_config_json_is_invalid_input() {
        for json in [
            r#"{"devcie":"auto"}"#,
            r#"{"software_rasterizers":"yes"}"#,
            r#"{"probe_isolation":"sandboxed"}"#,
            "[]",
            "null",
            "{",
            "",
        ] {
            let error = EngineConfig::from_json(json).unwrap_err();
            assert_eq!(error.kind, ErrorKind::InvalidInput, "{json}");
            assert_eq!(
                error.context.operation.as_deref(),
                Some("engine_config_from_json")
            );
            assert!(error.context.details.is_some());
        }
    }

    #[test]
    fn engine_reports_converted_devices_and_the_primary_gpu() {
        let engine =
            Engine::with_discovery(EngineConfig::default(), synthetic(reference_machine()))
                .unwrap();
        let devices = engine.devices();
        assert_eq!(devices.len(), 4);
        assert_eq!(
            engine.primary_gpu().map(|device| device.id),
            Some(DeviceId::Cuda { index: 0 })
        );
        assert_eq!(devices.iter().filter(|device| device.primary).count(), 1);
        assert_eq!(engine.probe_report().devices, devices);

        // Clones share the engine.
        let clone = engine.clone();
        assert_eq!(clone.devices(), devices);
        assert!(format!("{engine:?}").contains("devices: 4"));
    }

    #[test]
    fn probe_report_from_the_config_is_used_as_is() {
        let report = ProbeReport {
            devices: device::device_list(reference_machine()),
            isolated: true,
            ..ProbeReport::default()
        };
        // Primary flags in the input are recomputed.
        let mut tampered = report.clone();
        for device in &mut tampered.devices {
            device.primary = true;
        }
        let config = EngineConfig {
            probe_report: Some(tampered),
            ..EngineConfig::default()
        };
        let engine = Engine::new(config).unwrap();
        assert_eq!(engine.devices(), report.devices);
        assert_eq!(engine.probe_report().devices, report.devices);
    }

    #[test]
    fn panic_during_discovery_is_an_internal_error() {
        let error = Engine::with_discovery(
            EngineConfig::default(),
            Box::new(|_| panic!("driver exploded")),
        )
        .unwrap_err();
        assert_eq!(error.kind, ErrorKind::Internal);
        assert_eq!(error.context.operation.as_deref(), Some("engine_new"));
        assert_eq!(
            error.context.details.as_deref(),
            Some("panic: driver exploded")
        );
    }

    #[test]
    fn out_of_process_without_a_prober_fails_engine_new() {
        let config = EngineConfig {
            probe_isolation: Isolation::OutOfProcess,
            prober_path: Some(PathBuf::from("/nonexistent/polypous-test-prober")),
            probe_cache_dir: Some(temp_dir("oop")),
            ..EngineConfig::default()
        };
        let error = Engine::new(config).unwrap_err();
        assert_eq!(error.kind, ErrorKind::BackendUnavailable);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn engine_new_lists_the_cpu_and_refreshes() {
        let dir = temp_dir("cpu");
        let config = EngineConfig {
            probe_cache_dir: Some(dir.clone()),
            ..EngineConfig::default()
        };
        // No prober hook in a test binary: probing runs in process, not isolated.
        let engine = Engine::new(config).unwrap();
        let cpu = engine
            .devices()
            .into_iter()
            .find(|device| device.id == DeviceId::Cpu)
            .expect("the cpu device");
        assert!(cpu.status.is_available());
        assert!(!cpu.primary);
        assert!(!engine.probe_report().isolated);

        // A partial refresh of the CPU only, from the cache (written by `Engine::new`).
        let mut events = Vec::new();
        let options = ProbeOptions {
            devices: Some(vec![DeviceId::Cpu]),
            ..ProbeOptions::default()
        };
        let report = engine
            .refresh_devices(options, &CancellationToken::new(), |event| {
                events.push(event);
            })
            .unwrap();
        assert_eq!(report.devices.len(), 1);
        assert_eq!(engine.devices().len(), engine.state().devices.len());
        assert!(
            engine
                .devices()
                .iter()
                .any(|device| device.id == DeviceId::Cpu)
        );
        assert!(matches!(
            events.last(),
            Some(ProbeEvent::ProbeFinished { .. })
        ));

        // Cancelled before it starts: an error, the list is unchanged.
        let before = engine.devices();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let options = ProbeOptions {
            force: true,
            ..ProbeOptions::default()
        };
        let error = engine
            .refresh_devices(options, &cancel, |_| {})
            .unwrap_err();
        assert_eq!(error.kind, ErrorKind::Cancelled);
        assert_eq!(engine.devices(), before);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
