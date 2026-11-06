//! Collects the results of one probe across prober processes (placement.md §1, "Prober protocol"):
//! checks every line against the protocol's order, keeps what was reported, and knows which device
//! or backend was in progress when a prober died or hung.

use crate::device::{BackendKind, DeviceId, DeviceInfo, DeviceStatus, UnavailableReason};

use super::protocol::{EnumeratedDevice, MAX_DEVICES, PROTOCOL_VERSION, ProbeRequest, ProberLine};

/// What became of one device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Described and probed by a prober (or in process).
    Probed(Box<DeviceInfo>),
    /// Matched skip rule `rule` of the request: known from the cache, not opened.
    Skipped { rule: usize },
    /// The prober died while probing it.
    Crashed,
    /// The prober sent nothing within the timeout while probing it.
    TimedOut,
    /// The prober misbehaved while probing it, or never got to it.
    Failed(String),
}

impl Outcome {
    /// The status of an outcome that is not a full description.
    pub(crate) fn blame_status(&self) -> Option<DeviceStatus> {
        let reason = match self {
            Outcome::Crashed => UnavailableReason::CrashedDuringProbe,
            Outcome::TimedOut => UnavailableReason::ProbeTimedOut,
            Outcome::Failed(message) => UnavailableReason::ProbeFailed {
                message: message.clone(),
            },
            Outcome::Probed(_) | Outcome::Skipped { .. } => return None,
        };
        Some(DeviceStatus::Unavailable {
            reason,
            isolated: true,
        })
    }
}

/// What an accepted line means to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Accepted {
    /// Nothing new to report.
    Nothing,
    /// A device seen for the first time.
    Enumerated(DeviceId),
    /// A device's outcome is known.
    Outcome(DeviceId),
    /// The `done` line.
    Done,
}

/// Per-process protocol state.
#[derive(Debug, Default)]
struct RunState {
    header: bool,
    done: bool,
    enumerating: Option<BackendKind>,
    in_progress: Option<DeviceId>,
    enumerated_here: Vec<DeviceId>,
}

/// Results of one probe, across prober restarts.
#[derive(Debug)]
pub(crate) struct Collector {
    /// The original request (filters and skip rules).
    request: ProbeRequest,
    /// Whether results came from a prober process (statuses are forced to `isolated: true`).
    isolated: bool,
    /// Every device enumerated so far, in first-seen order.
    pub(crate) enumerated: Vec<EnumeratedDevice>,
    /// The outcome of each reported device, in report order.
    pub(crate) outcomes: Vec<(DeviceId, Outcome)>,
    /// Backends whose enumeration failed or killed a prober, with the reason.
    pub(crate) failed_backends: Vec<(BackendKind, String)>,
    run: RunState,
}

impl Collector {
    pub(crate) fn new(request: ProbeRequest, isolated: bool) -> Self {
        Collector {
            request,
            isolated,
            enumerated: Vec::new(),
            outcomes: Vec::new(),
            failed_backends: Vec::new(),
            run: RunState::default(),
        }
    }

    /// Resets the per-process state before a (re)started prober.
    pub(crate) fn start_run(&mut self) {
        self.run = RunState::default();
    }

    /// Whether the current process sent its header.
    pub(crate) fn header_seen(&self) -> bool {
        self.run.header
    }

    /// Whether the current process sent `done`.
    pub(crate) fn done_seen(&self) -> bool {
        self.run.done
    }

    /// The device the current process was probing, if any.
    pub(crate) fn in_progress(&self) -> Option<DeviceId> {
        self.run.in_progress
    }

    /// The backend the current process was enumerating (no device line after it yet), if any.
    pub(crate) fn enumerating(&self) -> Option<BackendKind> {
        self.run.enumerating
    }

    /// Whether `id` has an outcome.
    pub(crate) fn is_done(&self, id: DeviceId) -> bool {
        self.outcomes.iter().any(|(done, _)| *done == id)
    }

    /// Devices the request wants that have no outcome yet.
    pub(crate) fn remaining(&self) -> Vec<DeviceId> {
        self.enumerated
            .iter()
            .map(|device| device.id)
            .filter(|id| self.request.wants(*id) && !self.is_done(*id))
            .collect()
    }

    /// The enumeration details of `id`, if it was enumerated.
    pub(crate) fn details(&self, id: DeviceId) -> Option<&EnumeratedDevice> {
        self.enumerated.iter().find(|device| device.id == id)
    }

    /// Records an outcome the supervisor determined (crash, timeout, protocol failure).
    pub(crate) fn blame(&mut self, id: DeviceId, outcome: Outcome) {
        if !self.is_done(id) {
            self.outcomes.push((id, outcome));
        }
        if self.run.in_progress == Some(id) {
            self.run.in_progress = None;
        }
    }

    /// Records that enumerating `backend` failed (for example the prober died doing it).
    pub(crate) fn fail_backend(&mut self, backend: BackendKind, message: String) {
        if !self
            .failed_backends
            .iter()
            .any(|(known, _)| *known == backend)
        {
            self.failed_backends.push((backend, message));
        }
        if self.run.enumerating == Some(backend) {
            self.run.enumerating = None;
        }
    }

    /// Checks one line against the protocol and records it.
    ///
    /// # Errors
    ///
    /// A description of the violation. The caller kills the prober.
    pub(crate) fn accept(&mut self, line: ProberLine) -> Result<Accepted, String> {
        if self.run.done {
            return Err("output after the `done` line".into());
        }
        if !self.run.header && !matches!(line, ProberLine::Header { .. }) {
            return Err("the first line is not the header".into());
        }
        match line {
            ProberLine::Header { protocol, .. } => {
                if self.run.header {
                    return Err("a second header".into());
                }
                if protocol != PROTOCOL_VERSION {
                    return Err(format!(
                        "the prober speaks protocol version {protocol}, expected {PROTOCOL_VERSION}"
                    ));
                }
                self.run.header = true;
                Ok(Accepted::Nothing)
            }
            ProberLine::Enumerating { backend } => {
                self.check_not_probing("enumerating")?;
                self.run.enumerating = Some(backend);
                Ok(Accepted::Nothing)
            }
            ProberLine::EnumerationFailed { backend, message } => {
                self.check_not_probing("enumeration_failed")?;
                self.run.enumerating = None;
                self.fail_backend(backend, message);
                Ok(Accepted::Nothing)
            }
            ProberLine::Enumerated { device } => {
                self.check_not_probing("enumerated")?;
                if self.run.enumerated_here.contains(&device.id) {
                    return Err(format!("{} enumerated twice", device.id));
                }
                if self.run.enumerated_here.len() >= MAX_DEVICES {
                    return Err(format!("more than {MAX_DEVICES} devices"));
                }
                if self.run.enumerating != Some(device.id.backend()) {
                    return Err(format!(
                        "{} enumerated outside its backend's enumeration",
                        device.id
                    ));
                }
                self.run.enumerated_here.push(device.id);
                if self.details(device.id).is_some() {
                    // Already known from an earlier prober process of this probe.
                    return Ok(Accepted::Nothing);
                }
                let id = device.id;
                self.enumerated.push(device);
                Ok(if self.request.wants(id) {
                    Accepted::Enumerated(id)
                } else {
                    Accepted::Nothing
                })
            }
            ProberLine::Skipped { id, rule } => {
                self.check_reportable(id, "skipped")?;
                if rule >= self.request.skip.len() {
                    return Err(format!("{id} skipped by rule {rule}, which doesn't exist"));
                }
                self.run.enumerating = None;
                self.outcomes.push((id, Outcome::Skipped { rule }));
                Ok(Accepted::Outcome(id))
            }
            ProberLine::Probing { id } => {
                self.check_reportable(id, "probing")?;
                self.run.enumerating = None;
                self.run.in_progress = Some(id);
                Ok(Accepted::Nothing)
            }
            ProberLine::Probed { mut device } => {
                let id = device.id;
                if self.run.in_progress != Some(id) {
                    return Err(format!("{id} reported without a `probing` line"));
                }
                if device.status == DeviceStatus::NotProbed {
                    return Err(format!("{id} reported as not probed"));
                }
                // Don't trust the prober with these: the backend follows from the id, and whether
                // a result is isolated is decided here.
                device.backend = id.backend();
                let skipped = matches!(
                    device.status,
                    DeviceStatus::Unavailable {
                        reason: UnavailableReason::SoftwareRasterizerSkipped,
                        ..
                    }
                );
                device.status = device.status.with_isolated(self.isolated && !skipped);
                self.run.in_progress = None;
                self.outcomes.push((id, Outcome::Probed(device)));
                Ok(Accepted::Outcome(id))
            }
            ProberLine::Done => {
                self.check_not_probing("done")?;
                self.run.done = true;
                self.run.enumerating = None;
                Ok(Accepted::Done)
            }
        }
    }

    fn check_not_probing(&self, what: &str) -> Result<(), String> {
        match self.run.in_progress {
            Some(id) => Err(format!("`{what}` line while probing {id}")),
            None => Ok(()),
        }
    }

    /// A device may get a `skipped` or `probing` line once: enumerated by this process, wanted,
    /// not reported before, and no other device in progress.
    fn check_reportable(&self, id: DeviceId, what: &str) -> Result<(), String> {
        self.check_not_probing(what)?;
        if !self.run.enumerated_here.contains(&id) {
            return Err(format!("`{what}` line for {id}, which was not enumerated"));
        }
        if !self.request.wants(id) || self.request.exclude.contains(&id) {
            return Err(format!("`{what}` line for {id}, which was not requested"));
        }
        if self.is_done(id) {
            return Err(format!("{id} reported twice"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::unprobed_device;
    use crate::probe::protocol::SkipRule;

    fn header() -> ProberLine {
        ProberLine::Header {
            protocol: PROTOCOL_VERSION,
            polypous_version: "0".into(),
        }
    }

    fn enumerated(id: DeviceId) -> ProberLine {
        ProberLine::Enumerated {
            device: EnumeratedDevice {
                id,
                key: None,
                enum_fingerprint: None,
            },
        }
    }

    fn probed(id: DeviceId) -> ProberLine {
        ProberLine::Probed {
            device: Box::new(DeviceInfo {
                status: DeviceStatus::Available { isolated: false },
                ..unprobed_device(id)
            }),
        }
    }

    fn feed(collector: &mut Collector, lines: Vec<ProberLine>) -> Result<(), String> {
        for line in lines {
            collector.accept(line)?;
        }
        Ok(())
    }

    #[test]
    fn accepts_a_complete_run_and_forces_isolation() {
        let mut collector = Collector::new(ProbeRequest::default(), true);
        let cuda = DeviceId::Cuda { index: 0 };
        feed(
            &mut collector,
            vec![
                header(),
                ProberLine::Enumerating {
                    backend: BackendKind::Cuda,
                },
                enumerated(cuda),
                ProberLine::Enumerating {
                    backend: BackendKind::Cpu,
                },
                enumerated(DeviceId::Cpu),
                ProberLine::Probing { id: cuda },
                probed(cuda),
            ],
        )
        .unwrap();
        assert_eq!(collector.remaining(), vec![DeviceId::Cpu]);
        assert_eq!(
            collector.accept(ProberLine::Probing { id: DeviceId::Cpu }),
            Ok(Accepted::Nothing)
        );
        assert_eq!(collector.in_progress(), Some(DeviceId::Cpu));
        assert_eq!(
            collector.accept(probed(DeviceId::Cpu)),
            Ok(Accepted::Outcome(DeviceId::Cpu))
        );
        assert_eq!(collector.accept(ProberLine::Done), Ok(Accepted::Done));
        let Outcome::Probed(device) = &collector.outcomes[0].1 else {
            panic!("probed")
        };
        assert_eq!(device.status, DeviceStatus::Available { isolated: true });
        assert!(collector.accept(ProberLine::Done).is_err());
    }

    #[test]
    fn rejects_out_of_order_lines() {
        let cpu_enum = || ProberLine::Enumerating {
            backend: BackendKind::Cpu,
        };
        let cases: Vec<Vec<ProberLine>> = vec![
            // No header first.
            vec![ProberLine::Done],
            // Wrong protocol version.
            vec![ProberLine::Header {
                protocol: 99,
                polypous_version: "x".into(),
            }],
            // Probing a device that was never enumerated.
            vec![header(), ProberLine::Probing { id: DeviceId::Cpu }],
            // Probed without probing.
            vec![
                header(),
                cpu_enum(),
                enumerated(DeviceId::Cpu),
                probed(DeviceId::Cpu),
            ],
            // Probed result for another device.
            vec![
                header(),
                cpu_enum(),
                enumerated(DeviceId::Cpu),
                ProberLine::Probing { id: DeviceId::Cpu },
                probed(DeviceId::Cuda { index: 0 }),
            ],
            // Enumerated twice.
            vec![
                header(),
                cpu_enum(),
                enumerated(DeviceId::Cpu),
                enumerated(DeviceId::Cpu),
            ],
            // Enumerated outside its backend.
            vec![header(), enumerated(DeviceId::Cpu)],
            // A skip rule that doesn't exist.
            vec![
                header(),
                cpu_enum(),
                enumerated(DeviceId::Cpu),
                ProberLine::Skipped {
                    id: DeviceId::Cpu,
                    rule: 0,
                },
            ],
            // Done while probing.
            vec![
                header(),
                cpu_enum(),
                enumerated(DeviceId::Cpu),
                ProberLine::Probing { id: DeviceId::Cpu },
                ProberLine::Done,
            ],
        ];
        for lines in cases {
            let mut collector = Collector::new(ProbeRequest::default(), true);
            assert!(feed(&mut collector, lines.clone()).is_err(), "{lines:?}");
        }
    }

    #[test]
    fn restarts_keep_results_and_ignore_known_devices() {
        let request = ProbeRequest {
            skip: vec![SkipRule::Id { id: DeviceId::Cpu }],
            ..Default::default()
        };
        let lavapipe = crate::probe::session::tests::lavapipe();
        let mut collector = Collector::new(request, true);
        feed(
            &mut collector,
            vec![
                header(),
                ProberLine::Enumerating {
                    backend: BackendKind::Vulkan,
                },
                enumerated(lavapipe),
                ProberLine::Enumerating {
                    backend: BackendKind::Cpu,
                },
                enumerated(DeviceId::Cpu),
                ProberLine::Probing { id: lavapipe },
            ],
        )
        .unwrap();
        // The prober dies here.
        collector.blame(lavapipe, Outcome::Crashed);
        assert_eq!(collector.remaining(), vec![DeviceId::Cpu]);
        collector.start_run();
        feed(
            &mut collector,
            vec![
                header(),
                ProberLine::Enumerating {
                    backend: BackendKind::Vulkan,
                },
                enumerated(lavapipe),
                ProberLine::Enumerating {
                    backend: BackendKind::Cpu,
                },
                enumerated(DeviceId::Cpu),
                ProberLine::Skipped {
                    id: DeviceId::Cpu,
                    rule: 0,
                },
                ProberLine::Done,
            ],
        )
        .unwrap();
        assert_eq!(collector.enumerated.len(), 2);
        assert_eq!(
            collector.outcomes,
            vec![
                (lavapipe, Outcome::Crashed),
                (DeviceId::Cpu, Outcome::Skipped { rule: 0 })
            ]
        );
        assert_eq!(
            Outcome::Crashed.blame_status(),
            Some(DeviceStatus::Unavailable {
                reason: UnavailableReason::CrashedDuringProbe,
                isolated: true
            })
        );
    }
}
