//! [`Engine`] and [`EngineConfig`] (public-api.md §1, §1.2).

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::device::{self, DeviceInfo};
use crate::discovery::{self, Discovery};
use crate::error::{Error, ErrorKind, guard};

/// Configuration of an [`Engine`]. Every field is optional; `EngineConfig::default()` is the
/// recommended configuration.
///
/// JSON (public-api.md §1.2): an object, `{}` for the defaults, for example
/// `{"software_rasterizers":true}`. Unknown fields are an [`ErrorKind::InvalidInput`] error, so a
/// misspelled option is never silently ignored. Fields are added as the engine grows
/// (`probe_report`, `prober_path`, `probe_cache_dir` with #14; placement options with M4); each one
/// is optional with a documented default.
///
/// The struct is `#[non_exhaustive]`: start from `EngineConfig::default()` and set fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
            let invalid = |details: String| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "The engine configuration is not valid.",
                )
                .with_details(details)
            };
            // serde reads a struct from an array too; the documented form is an object only.
            let value: serde_json::Value =
                serde_json::from_str(json).map_err(|error| invalid(error.to_string()))?;
            if !value.is_object() {
                return Err(invalid("expected a JSON object".to_string()));
            }
            serde_json::from_value(value).map_err(|error| invalid(error.to_string()))
        })
    }

    /// The config's JSON form.
    pub fn to_json(&self) -> String {
        // Serializing a struct of plain fields cannot fail; fall back to the defaults' form anyway.
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
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
    devices: Vec<DeviceInfo>,
}

impl Engine {
    /// Creates an engine: discovers every device of every compiled-in backend and probes it
    /// (placement.md §1), then selects the primary GPU.
    ///
    /// For now discovery and probing run in this process, in GPU priority order. A broken GPU
    /// driver can crash the process natively; the prober process of ADR-0013 (#14) will isolate
    /// that. A device that fails its probe is reported as unavailable, never as an error, so this
    /// succeeds on any machine.
    ///
    /// The first probe of a GPU compiles kernels and can take seconds.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Internal`] if Polypous panics while discovering devices (a bug).
    pub fn new(config: EngineConfig) -> Result<Engine, Error> {
        Self::with_discovery(config, Box::new(discovery::in_process))
    }

    /// [`Engine::new`] with another discovery step (tests now; the prober process with #14).
    pub(crate) fn with_discovery(
        config: EngineConfig,
        discover: Box<Discovery>,
    ) -> Result<Engine, Error> {
        guard("engine_new", || {
            let devices = device::device_list(discover(&config));
            match devices.iter().find(|device| device.primary) {
                Some(primary) => {
                    tracing::info!(device = %primary.id, name = %primary.name, "primary GPU selected");
                }
                None => tracing::info!("no usable GPU, running on the CPU only"),
            }
            Ok(Engine {
                inner: Arc::new(EngineInner { config, devices }),
            })
        })
    }

    /// The configuration this engine was created with.
    pub fn config(&self) -> EngineConfig {
        self.inner.config.clone()
    }

    /// Every discovered device with its probe status, in probe order (GPU priority order, the CPU
    /// last). The primary GPU, if any, has [`DeviceInfo::primary`] set.
    ///
    /// JSON: an array of [`DeviceInfo`] (public-api.md §1.2).
    pub fn devices(&self) -> Vec<DeviceInfo> {
        self.inner.devices.clone()
    }

    /// The primary GPU: the one GPU Polypous uses, or `None` to run on the CPU only.
    pub fn primary_gpu(&self) -> Option<DeviceInfo> {
        self.inner
            .devices
            .iter()
            .find(|device| device.primary)
            .cloned()
    }
}

impl fmt::Debug for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("config", &self.inner.config)
            .field("devices", &self.inner.devices.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::tests::reference_machine;
    use crate::{DeviceId, ErrorKind};

    #[test]
    fn engine_is_a_send_sync_static_handle() {
        fn assert_handle<T: Clone + Send + Sync + 'static>() {}
        assert_handle::<Engine>();
        assert_handle::<EngineConfig>();
        assert_handle::<DeviceInfo>();
        assert_handle::<Error>();
    }

    #[test]
    fn config_json_round_trips() {
        assert_eq!(
            EngineConfig::default().to_json(),
            r#"{"software_rasterizers":false}"#
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
    }

    #[test]
    fn bad_config_json_is_invalid_input() {
        for json in [
            r#"{"devcie":"auto"}"#,
            r#"{"software_rasterizers":"yes"}"#,
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
            Engine::with_discovery(EngineConfig::default(), Box::new(|_| reference_machine()))
                .unwrap();
        let devices = engine.devices();
        assert_eq!(devices.len(), 4);
        assert_eq!(
            engine.primary_gpu().map(|device| device.id),
            Some(DeviceId::Cuda { index: 0 })
        );
        assert_eq!(devices.iter().filter(|device| device.primary).count(), 1);

        // Clones share the engine.
        let clone = engine.clone();
        assert_eq!(clone.devices(), devices);
        assert!(format!("{engine:?}").contains("devices: 4"));
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

    #[cfg(feature = "cpu")]
    #[test]
    fn engine_new_lists_the_cpu() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let cpu = engine
            .devices()
            .into_iter()
            .find(|device| device.id == DeviceId::Cpu)
            .expect("the cpu device");
        assert!(cpu.status.is_available());
        assert!(!cpu.primary);
    }
}
