//! One probe session: enumerate, then describe and probe each device in probe order, reporting
//! every step as a [`ProberLine`] (placement.md §1, "Prober protocol").
//!
//! The prober process runs this and writes the lines to stdout; the in-process fallback runs the
//! same code and feeds the lines straight to the collector, with the crash journal written on
//! `probing` lines. So both paths probe identically.

use std::io;

use crate::device::{DeviceId, probe_rank, skipped_device};

use super::protocol::{EnumeratedDevice, MAX_DEVICES, PROTOCOL_VERSION, ProbeRequest, ProberLine};
use super::source::DeviceSource;
use super::worker_panics::{self, WorkerPanics};

/// Runs one session for `request` on `source`. `isolated` is what probe results report
/// (`true` in the prober process). `emit` receives every line in protocol order; an error from it
/// (stdout closed) ends the session.
pub(crate) fn run_session(
    request: &ProbeRequest,
    source: &dyn DeviceSource,
    isolated: bool,
    emit: &mut dyn FnMut(ProberLine) -> io::Result<()>,
) -> io::Result<()> {
    emit(ProberLine::Header {
        protocol: PROTOCOL_VERSION,
        polypous_version: env!("CARGO_PKG_VERSION").to_string(),
    })?;

    let mut enumerated: Vec<EnumeratedDevice> = Vec::new();
    for backend in source.backends() {
        if request.skip_backends.contains(&backend) {
            continue;
        }
        emit(ProberLine::Enumerating { backend })?;
        match source.list(backend) {
            Ok(ids) => {
                for id in ids {
                    if enumerated.len() >= MAX_DEVICES
                        || enumerated.iter().any(|known| known.id == id)
                    {
                        continue;
                    }
                    let device = source.details(id);
                    emit(ProberLine::Enumerated {
                        device: device.clone(),
                    })?;
                    enumerated.push(device);
                }
            }
            Err(error) => emit(ProberLine::EnumerationFailed {
                backend,
                message: error.message,
            })?,
        }
    }

    enumerated.sort_by_key(|device| probe_rank(&device.id));
    for device in enumerated {
        let id = device.id;
        if !request.wants(id) || request.exclude.contains(&id) {
            continue;
        }
        if let Some(rule) = request.skip_rule_for(&device) {
            emit(ProberLine::Skipped { id, rule })?;
            continue;
        }
        emit(ProberLine::Probing { id })?;
        emit(ProberLine::Probed {
            device: Box::new(probe_one(request, source, id, isolated)),
        })?;
    }
    emit(ProberLine::Done)
}

/// Probes one device, or describes it as skipped if it is a software rasterizer the request does
/// not opt in to (never opened then).
fn probe_one(
    request: &ProbeRequest,
    source: &dyn DeviceSource,
    id: DeviceId,
    isolated: bool,
) -> crate::device::DeviceInfo {
    if id.is_software_rasterizer() && !request.software_rasterizers {
        return skipped_device(id);
    }
    // Only the prober process collects panics of other threads (it installed the hook; in-process
    // probing never does, see `worker_panics`).
    let (mut info, panics) = if isolated {
        worker_panics::during_step(|| source.describe_and_probe(id))
    } else {
        (source.describe_and_probe(id), WorkerPanics::default())
    };
    info.status = worker_panics::attach(info.status, &panics).with_isolated(isolated);
    info
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::device::{
        AdapterKind, BackendKind, DeviceInfo, DeviceStatus, EnumerationError, UnavailableReason,
        unprobed_device,
    };
    use crate::probe::protocol::SkipRule;

    /// A scripted source: these devices, every probe passes, and it records what it opened.
    pub(crate) struct Scripted {
        pub(crate) ids: Vec<DeviceId>,
        pub(crate) opened: RefCell<Vec<DeviceId>>,
    }

    impl Scripted {
        pub(crate) fn new(ids: Vec<DeviceId>) -> Self {
            Scripted {
                ids,
                opened: RefCell::new(Vec::new()),
            }
        }
    }

    impl DeviceSource for Scripted {
        fn backends(&self) -> Vec<BackendKind> {
            BackendKind::ALL.to_vec()
        }
        fn list(&self, backend: BackendKind) -> Result<Vec<DeviceId>, EnumerationError> {
            Ok(self
                .ids
                .iter()
                .copied()
                .filter(|id| id.backend() == backend)
                .collect())
        }
        fn details(&self, id: DeviceId) -> EnumeratedDevice {
            EnumeratedDevice {
                id,
                key: None,
                enum_fingerprint: None,
            }
        }
        fn describe_and_probe(&self, id: DeviceId) -> DeviceInfo {
            self.opened.borrow_mut().push(id);
            DeviceInfo {
                status: DeviceStatus::Available { isolated: false },
                ..unprobed_device(id)
            }
        }
    }

    pub(crate) fn lavapipe() -> DeviceId {
        DeviceId::Vulkan {
            adapter: AdapterKind::Cpu,
            index: 0,
        }
    }

    pub(crate) fn debian_reference_ids() -> Vec<DeviceId> {
        vec![
            DeviceId::Cpu,
            DeviceId::Cuda { index: 0 },
            DeviceId::Vulkan {
                adapter: AdapterKind::DiscreteGpu,
                index: 0,
            },
            DeviceId::Vulkan {
                adapter: AdapterKind::IntegratedGpu,
                index: 0,
            },
            lavapipe(),
        ]
    }

    fn lines(request: &ProbeRequest, source: &Scripted) -> Vec<ProberLine> {
        let mut lines = Vec::new();
        run_session(request, source, true, &mut |line| {
            lines.push(line);
            Ok(())
        })
        .unwrap();
        lines
    }

    #[test]
    fn session_order_and_isolation() {
        let source = Scripted::new(debian_reference_ids());
        let lines = lines(&ProbeRequest::default(), &source);
        assert!(matches!(lines[0], ProberLine::Header { protocol: 1, .. }));
        assert_eq!(lines.last(), Some(&ProberLine::Done));
        let probing: Vec<DeviceId> = lines
            .iter()
            .filter_map(|line| match line {
                ProberLine::Probing { id } => Some(*id),
                _ => None,
            })
            .collect();
        // Probe order: CUDA, Vulkan discrete, Vulkan integrated, lavapipe, CPU.
        assert_eq!(probing, crate::device::probe_order(debian_reference_ids()));
        // Lavapipe is reported skipped and never opened; the others are isolated.
        assert!(!source.opened.borrow().contains(&lavapipe()));
        for line in &lines {
            if let ProberLine::Probed { device } = line {
                if device.id == lavapipe() {
                    assert_eq!(
                        device.status,
                        DeviceStatus::Unavailable {
                            reason: UnavailableReason::SoftwareRasterizerSkipped,
                            isolated: false
                        }
                    );
                } else {
                    assert_eq!(device.status, DeviceStatus::Available { isolated: true });
                }
            }
        }
    }

    #[test]
    fn skip_rules_filters_and_exclusions_are_honored() {
        let source = Scripted::new(debian_reference_ids());
        let request = ProbeRequest {
            software_rasterizers: true,
            skip: vec![SkipRule::Id {
                id: DeviceId::Cuda { index: 0 },
            }],
            exclude: vec![DeviceId::Cpu],
            skip_backends: vec![BackendKind::Metal],
            ..Default::default()
        };
        let output = lines(&request, &source);
        assert!(output.contains(&ProberLine::Skipped {
            id: DeviceId::Cuda { index: 0 },
            rule: 0
        }));
        assert!(!output.contains(&ProberLine::Probing { id: DeviceId::Cpu }));
        // Excluded devices are still enumerated.
        assert!(output.iter().any(|line| matches!(
            line,
            ProberLine::Enumerated { device } if device.id == DeviceId::Cpu
        )));
        assert!(!output.contains(&ProberLine::Enumerating {
            backend: BackendKind::Metal
        }));
        assert_eq!(
            *source.opened.borrow(),
            vec![
                DeviceId::Vulkan {
                    adapter: AdapterKind::DiscreteGpu,
                    index: 0
                },
                DeviceId::Vulkan {
                    adapter: AdapterKind::IntegratedGpu,
                    index: 0
                },
                lavapipe(),
            ]
        );

        let only = ProbeRequest {
            devices: Some(vec![DeviceId::Cpu]),
            ..Default::default()
        };
        let output = lines(&only, &Scripted::new(debian_reference_ids()));
        let probed = output
            .iter()
            .filter(|line| matches!(line, ProberLine::Probing { .. }))
            .count();
        assert_eq!(probed, 1);
    }
}
