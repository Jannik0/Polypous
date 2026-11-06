//! Test-only fault injection for the prober and the in-process fallback (issue #14, "Tests").
//!
//! **Compiled only with the `test-fault-injection` cargo feature**, which no release build turns
//! on: the workspace enables it only through dev-dependencies (`polypous-cli`'s tests), so a
//! `cargo build`/`cargo install` of `polyp` or of an app does not contain this module, and the
//! variables below are not read. See placement.md §1, "Testing the prober".
//!
//! `POLYPOUS_TEST_PROBER_FAULTS` holds `;`-separated `key=value` items:
//!
//! | Item | Effect |
//! |---|---|
//! | `fake_devices=<id>,<id>,…` | Replace the real backends by these fake devices (no driver is loaded) |
//! | `fake_driver=<text>` | Enumeration fingerprint of the fake devices (default `fake`) |
//! | `crash=<id>,…` | Abort the process when probing these devices |
//! | `hang=<id>,…` | Never return when probing these devices |
//! | `fail=<id>,…` | Report these devices as `probe_failed` |
//! | `crash_enumeration=<backend>` | Abort the process while enumerating this backend |
//! | `garbage=<id>` | Prober only: after `probing` for this device, write a malformed line and hang |
//! | `oversized=<id>` | Prober only: … a line longer than the limit … |
//! | `spam=<id>` | Prober only: … more lines than the limit … |
//! | `exit_before_header` | Prober only: exit with code 7 before the header |
//! | `pid_file=<path>` | Prober only: append the prober's process id to this file at start |
//!
//! `POLYPOUS_TEST_FINGERPRINT` (read by [`super::fingerprint`]) replaces the OS driver fingerprint
//! of the app process: `unknown`, or any text used as the fingerprint of every device.

use std::io::{self, Write};
use std::time::Duration;

use crate::device::{
    BackendKind, DeviceId, DeviceInfo, DeviceStatus, EnumerationError, UnavailableReason,
    unprobed_device,
};

use super::protocol::{EnumeratedDevice, MAX_LINE_BYTES, MAX_LINES, ProberLine};
use super::source::DeviceSource;

/// The variable with the faults to inject.
pub(crate) const FAULTS_VAR: &str = "POLYPOUS_TEST_PROBER_FAULTS";

/// The variable that replaces the OS driver fingerprint.
pub(crate) const FINGERPRINT_VAR: &str = "POLYPOUS_TEST_FINGERPRINT";

/// Parsed [`FAULTS_VAR`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Faults {
    fake_devices: Option<Vec<DeviceId>>,
    fake_driver: Option<String>,
    crash: Vec<DeviceId>,
    hang: Vec<DeviceId>,
    fail: Vec<DeviceId>,
    crash_enumeration: Vec<BackendKind>,
    garbage: Vec<DeviceId>,
    oversized: Vec<DeviceId>,
    spam: Vec<DeviceId>,
    pub(crate) exit_before_header: bool,
    pub(crate) pid_file: Option<String>,
}

impl Faults {
    /// The faults of this process's environment, if the variable is set. A malformed value is
    /// reported on stderr and ignored, so a typo fails the test that relies on it.
    pub(crate) fn from_env() -> Option<Faults> {
        let value = std::env::var(FAULTS_VAR).ok()?;
        match Faults::parse(&value) {
            Ok(faults) => {
                tracing::warn!(faults = %value, "test-only fault injection is active");
                Some(faults)
            }
            Err(message) => {
                tracing::error!(%message, "ignoring {FAULTS_VAR}");
                None
            }
        }
    }

    pub(crate) fn parse(text: &str) -> Result<Faults, String> {
        let ids = |value: &str| -> Result<Vec<DeviceId>, String> {
            value
                .split(',')
                .filter(|part| !part.trim().is_empty())
                .map(str::parse)
                .collect()
        };
        let backend = |value: &str| match value.trim() {
            "cuda" => Ok(BackendKind::Cuda),
            "metal" => Ok(BackendKind::Metal),
            "vulkan" => Ok(BackendKind::Vulkan),
            "cpu" => Ok(BackendKind::Cpu),
            other => Err(format!("unknown backend `{other}`")),
        };
        let mut faults = Faults::default();
        for item in text
            .split(';')
            .map(str::trim)
            .filter(|item| !item.is_empty())
        {
            let (key, value) = item.split_once('=').unwrap_or((item, ""));
            match key.trim() {
                "fake_devices" => faults.fake_devices = Some(ids(value)?),
                "fake_driver" => faults.fake_driver = Some(value.to_string()),
                "crash" => faults.crash.extend(ids(value)?),
                "hang" => faults.hang.extend(ids(value)?),
                "fail" => faults.fail.extend(ids(value)?),
                "crash_enumeration" => faults.crash_enumeration.push(backend(value)?),
                "garbage" => faults.garbage.extend(ids(value)?),
                "oversized" => faults.oversized.extend(ids(value)?),
                "spam" => faults.spam.extend(ids(value)?),
                "exit_before_header" => faults.exit_before_header = true,
                "pid_file" => faults.pid_file = Some(value.to_string()),
                other => return Err(format!("unknown fault `{other}`")),
            }
        }
        Ok(faults)
    }

    /// Wraps `real`, or replaces it by the fake devices.
    pub(crate) fn into_source(self, real: Box<dyn DeviceSource>) -> Box<dyn DeviceSource> {
        Box::new(Faulty { faults: self, real })
    }

    /// Prober only: what to write instead of `line`. Returns whether `line` was replaced.
    pub(crate) fn replace_line(&self, line: &ProberLine, out: &mut dyn Write) -> io::Result<bool> {
        let ProberLine::Probing { id } = line else {
            return Ok(false);
        };
        if !(self.garbage.contains(id) || self.oversized.contains(id) || self.spam.contains(id)) {
            return Ok(false);
        }
        // The `probing` line goes out first: the bad output comes while the device is in
        // progress, as when a driver writes to stdout.
        out.write_all(line.to_json().as_bytes())?;
        out.write_all(b"\n")?;
        if self.garbage.contains(id) {
            out.write_all(b"this is not JSON\n")?;
        } else if self.oversized.contains(id) {
            let mut text = vec![b'x'; MAX_LINE_BYTES + 100];
            text.push(b'\n');
            out.write_all(&text)?;
        } else if self.spam.contains(id) {
            let line = ProberLine::Enumerating {
                backend: BackendKind::Cpu,
            }
            .to_json();
            for _ in 0..=MAX_LINES {
                out.write_all(line.as_bytes())?;
                out.write_all(b"\n")?;
            }
        }
        out.flush()?;
        // Keep the process alive; the app must kill it.
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }
}

/// The real devices or fakes, with faults.
struct Faulty {
    faults: Faults,
    real: Box<dyn DeviceSource>,
}

impl DeviceSource for Faulty {
    fn backends(&self) -> Vec<BackendKind> {
        if self.faults.fake_devices.is_some() {
            BackendKind::ALL.to_vec()
        } else {
            self.real.backends()
        }
    }

    fn list(&self, backend: BackendKind) -> Result<Vec<DeviceId>, EnumerationError> {
        if self.faults.crash_enumeration.contains(&backend) {
            eprintln!("fault injection: crashing while enumerating {backend}");
            std::process::abort();
        }
        match &self.faults.fake_devices {
            Some(fakes) => Ok(fakes
                .iter()
                .copied()
                .filter(|id| id.backend() == backend)
                .collect()),
            None => self.real.list(backend),
        }
    }

    fn details(&self, id: DeviceId) -> EnumeratedDevice {
        if self.faults.fake_devices.is_some() {
            EnumeratedDevice {
                id,
                key: None,
                enum_fingerprint: Some(format!(
                    "fake:{}",
                    self.faults.fake_driver.as_deref().unwrap_or("fake")
                )),
            }
        } else {
            self.real.details(id)
        }
    }

    fn describe_and_probe(&self, id: DeviceId) -> DeviceInfo {
        if self.faults.crash.contains(&id) {
            eprintln!("fault injection: crashing while probing {id}");
            std::process::abort();
        }
        if self.faults.hang.contains(&id) {
            eprintln!("fault injection: hanging while probing {id}");
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        if self.faults.fail.contains(&id) {
            return DeviceInfo {
                status: DeviceStatus::Unavailable {
                    reason: UnavailableReason::ProbeFailed {
                        message: "fault injection".into(),
                    },
                    isolated: false,
                },
                ..unprobed_device(id)
            };
        }
        if self.faults.fake_devices.is_some() {
            DeviceInfo {
                name: format!("Fake {id}"),
                status: DeviceStatus::Available { isolated: false },
                ..unprobed_device(id)
            }
        } else {
            self.real.describe_and_probe(id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::AdapterKind;

    #[test]
    fn parses_faults() {
        let faults = Faults::parse(
            "fake_devices=cuda:0,vulkan:cpu:0,cpu; crash=vulkan:cpu:0; hang=cuda:0;\
             crash_enumeration=vulkan;exit_before_header",
        )
        .unwrap();
        assert_eq!(
            faults.fake_devices,
            Some(vec![
                DeviceId::Cuda { index: 0 },
                DeviceId::Vulkan {
                    adapter: AdapterKind::Cpu,
                    index: 0
                },
                DeviceId::Cpu
            ])
        );
        assert_eq!(faults.hang, vec![DeviceId::Cuda { index: 0 }]);
        assert_eq!(faults.crash_enumeration, vec![BackendKind::Vulkan]);
        assert!(faults.exit_before_header);
        assert!(Faults::parse("explode=cpu").is_err());
        assert!(Faults::parse("crash=gpu:0").is_err());
    }
}
