//! One probe: probe cache → prober process (with restarts) → in-process fallback with crash
//! journal (ADR-0013; placement.md §1, "Crash safety").

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::device::{
    BackendKind, DeviceId, DeviceInfo, DeviceStatus, UnavailableReason, probe_rank, skipped_device,
    unprobed_device,
};

use super::cache::{self, CacheEntry, CacheFile};
use super::collect::{Accepted, Collector, Outcome};
use super::fingerprint::{self, SystemFingerprint};
use super::journal::{self, Journal};
use super::protocol::{MAX_DEVICES, ProbeRequest, SkipRule};
use super::session::run_session;
use super::source::device_source;
use super::supervisor::{self, ProberCommand, RunEnd, StartError};

/// Where probes run (ADR-0013 §4). JSON: `"auto"`, `"out_of_process"`, `"in_process"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Isolation {
    /// In a prober process if one can be started, else in process with the crash journal.
    #[default]
    Auto,
    /// Only in a prober process; an error if none can be started.
    OutOfProcess,
    /// In this process, with the crash journal.
    InProcess,
}

/// Default timeout per device (ADR-0013 §1).
pub const DEFAULT_TIMEOUT_PER_DEVICE: Duration = Duration::from_secs(60);

/// What to probe and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeConfig {
    /// Ignore the cache: probe every device again (a device recorded as crashed is still not
    /// opened again in process).
    pub force: bool,
    /// Probe only these devices; `None`: all.
    pub devices: Option<Vec<DeviceId>>,
    /// Where probes run.
    pub isolation: Isolation,
    /// How long the prober may stay silent (per device, and for start-up and enumeration).
    pub timeout_per_device: Duration,
    /// Open and probe software rasterizers (placement.md §1).
    pub software_rasterizers: bool,
    /// The prober executable, if one is available.
    pub prober: Option<ProberCommand>,
    /// The probe cache directory (also holds crash journals); `None`: no cache, no journal.
    pub cache_dir: Option<PathBuf>,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        ProbeConfig {
            force: false,
            devices: None,
            isolation: Isolation::Auto,
            timeout_per_device: DEFAULT_TIMEOUT_PER_DEVICE,
            software_rasterizers: false,
            prober: None,
            cache_dir: None,
        }
    }
}

/// Progress of a probe, in order.
// A handful of events per probe: their size doesn't matter, and unboxed variants are easier to
// match.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeEvent {
    /// A device was enumerated (not sent for devices taken from the cache without enumerating).
    DeviceEnumerated {
        /// The device.
        id: DeviceId,
    },
    /// A device's result is known.
    DeviceProbed {
        /// The device with its status.
        device: DeviceInfo,
        /// Whether the result came from the probe cache.
        cached: bool,
    },
}

/// The result of a probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeReport {
    /// Every device, in probe order, with its status.
    pub devices: Vec<DeviceInfo>,
    /// Whether every probe result in `devices` (cached or new) came from a prober process.
    pub isolated: bool,
    /// Devices whose result came from the probe cache, without opening them.
    pub cache_hits: Vec<DeviceId>,
    /// How many prober processes this probe started (0: everything came from the cache, or
    /// probing ran in process).
    pub prober_starts: u32,
    /// Whether the OS driver fingerprint was known (else the enumerate-only comparison ran).
    pub fingerprint_known: bool,
    /// Problems worth showing: a prober crash, a fallback to in-process probing, …
    pub warnings: Vec<String>,
}

/// Why a probe failed as a whole. Device problems are statuses, never errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProbeError {
    /// The caller cancelled; a running prober was killed.
    #[error("the device probe was cancelled")]
    Cancelled,
    /// `Isolation::OutOfProcess` was requested and no prober could be started or spoke the
    /// protocol.
    #[error("no device prober could be used: {0}")]
    ProberUnavailable(String),
}

/// Runs a probe: the cache, else the prober, else the in-process fallback. Never panics on
/// device problems; `cancelled` is polled while waiting.
///
/// # Errors
///
/// [`ProbeError::Cancelled`], or [`ProbeError::ProberUnavailable`] for
/// [`Isolation::OutOfProcess`] without a working prober.
pub fn probe_devices(
    config: &ProbeConfig,
    cancelled: &dyn Fn() -> bool,
    on_event: &mut dyn FnMut(ProbeEvent),
) -> Result<ProbeReport, ProbeError> {
    let started = Instant::now();
    let fingerprint = fingerprint::current();
    let cache_path = config
        .cache_dir
        .as_ref()
        .map(|dir| dir.join(cache::file_name()));
    let mut cache = cache_path
        .as_deref()
        .and_then(cache::load)
        .unwrap_or_else(CacheFile::empty);
    let mut warnings = Vec::new();

    if let Some(dir) = &config.cache_dir {
        let crashed = journal::recover(dir);
        if !crashed.is_empty() {
            for entry in crashed {
                warnings.push(format!(
                    "{} crashed this app during an earlier in-process probe; it stays unavailable \
                     until its driver changes",
                    entry.device.id
                ));
                cache.upsert(entry);
            }
            if let Some(path) = &cache_path {
                cache::save(path, &cache);
            }
        }
    }

    if !config.force
        && let Some(fingerprint) = &fingerprint
        && let Some(devices) = full_hit(&cache, fingerprint, config)
    {
        tracing::info!(
            devices = devices.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "probe cache hit: every device from the cache, no prober started"
        );
        let cache_hits = devices.iter().map(|device| device.id).collect();
        for device in &devices {
            on_event(ProbeEvent::DeviceProbed {
                device: device.clone(),
                cached: true,
            });
        }
        return Ok(report(devices, cache_hits, 0, true, warnings));
    }

    let in_process = match (config.isolation, &config.prober) {
        (Isolation::InProcess, _) => true,
        (Isolation::OutOfProcess, None) => {
            return Err(ProbeError::ProberUnavailable(
                "no prober executable is configured".into(),
            ));
        }
        (Isolation::Auto, None) => {
            let message = "no device prober is available (set prober_path, or call \
                           polypous::probe::run_prober_if_requested() first in main): probing in \
                           this process, not isolated";
            tracing::warn!("{message}");
            warnings.push(message.into());
            true
        }
        (_, Some(_)) => false,
    };
    let rules = skip_rules(&cache, fingerprint.as_ref(), config, in_process);
    let request = ProbeRequest {
        software_rasterizers: config.software_rasterizers,
        devices: config.devices.clone(),
        skip: rules.iter().map(|(rule, _)| rule.clone()).collect(),
        ..ProbeRequest::default()
    };
    let planned: Vec<CacheEntry> = rules.into_iter().map(|(_, entry)| entry).collect();

    let mut run = Run {
        config,
        fingerprint: fingerprint.as_ref(),
        planned: &planned,
        cache,
        cache_path,
        warnings,
        prober_starts: 0,
    };
    let collector = if in_process {
        run.in_process(request, cancelled, on_event)?
    } else {
        match run.out_of_process(&request, cancelled, on_event) {
            Ok(collector) => collector,
            Err(Fallback::Cancelled) => return Err(ProbeError::Cancelled),
            Err(Fallback::Unavailable(message)) => {
                if config.isolation == Isolation::OutOfProcess {
                    return Err(ProbeError::ProberUnavailable(message));
                }
                let message = format!(
                    "the device prober could not be used ({message}): probing in this process, \
                     not isolated"
                );
                tracing::warn!("{message}");
                run.warnings.push(message);
                run.in_process(request, cancelled, on_event)?
            }
        }
    };
    let report = run.finish(&collector);
    tracing::info!(
        devices = report.devices.len(),
        cache_hits = report.cache_hits.len(),
        prober_starts = report.prober_starts,
        isolated = report.isolated,
        elapsed_ms = started.elapsed().as_millis(),
        "device probe finished"
    );
    Ok(report)
}

/// Builds a report; `isolated` is derived from the statuses.
fn report(
    mut devices: Vec<DeviceInfo>,
    cache_hits: Vec<DeviceId>,
    prober_starts: u32,
    fingerprint_known: bool,
    warnings: Vec<String>,
) -> ProbeReport {
    devices.sort_by_key(|device| probe_rank(&device.id));
    let isolated = devices.iter().all(|device| match &device.status {
        DeviceStatus::NotProbed => true,
        DeviceStatus::Available { isolated } => *isolated,
        DeviceStatus::Unavailable {
            reason: UnavailableReason::SoftwareRasterizerSkipped,
            ..
        } => true,
        DeviceStatus::Unavailable { isolated, .. } => *isolated,
    });
    ProbeReport {
        devices,
        isolated,
        cache_hits,
        prober_starts,
        fingerprint_known,
        warnings,
    }
}

/// A cached device as reported now: a software rasterizer the caller doesn't opt in to is
/// reported skipped, whatever was cached.
fn as_reported(id: DeviceId, device: DeviceInfo, config: &ProbeConfig) -> DeviceInfo {
    if id.is_software_rasterizer() && !config.software_rasterizers {
        return skipped_device(id);
    }
    DeviceInfo {
        id,
        backend: id.backend(),
        ..device
    }
}

/// Whether a cache entry is still valid under the current OS fingerprint.
fn entry_valid(entry: &CacheEntry, fingerprint: &SystemFingerprint) -> bool {
    entry.fingerprint.is_some()
        && entry.fingerprint
            == fingerprint.for_device(
                entry.key.as_deref(),
                entry.device.identity.pci_bus_id.as_deref(),
            )
}

/// The devices of a complete cache hit, or `None` if anything needs probing: the GPU set or a
/// device's fingerprint changed, a requested device is unknown, the last probe was incomplete, or
/// a software rasterizer the caller now opts in to was only skipped before.
fn full_hit(
    cache: &CacheFile,
    fingerprint: &SystemFingerprint,
    config: &ProbeConfig,
) -> Option<Vec<DeviceInfo>> {
    if !cache.complete || cache.devices.is_empty() || cache.gpu_keys != fingerprint.gpu_keys() {
        return None;
    }
    let entries: Vec<&CacheEntry> = match &config.devices {
        None => cache.devices.iter().collect(),
        Some(ids) => ids
            .iter()
            .map(|id| cache.entry(*id))
            .collect::<Option<Vec<_>>>()?,
    };
    let usable = |entry: &&CacheEntry| {
        entry_valid(entry, fingerprint)
            && !(config.software_rasterizers && entry.is_skipped_rasterizer())
    };
    if !entries.iter().all(usable) {
        return None;
    }
    Some(
        entries
            .into_iter()
            .map(|entry| as_reported(entry.device.id, entry.device.clone(), config))
            .collect(),
    )
}

/// Skip rules for the cached devices that need no new probe, each with its entry.
///
/// With a known fingerprint, a valid entry is matched by stable key (or by id if it has none);
/// with an unknown fingerprint, by id plus enumeration key and fingerprint (enumerate-only
/// comparison). `force` drops all rules, except that in-process probing never re-opens a device
/// recorded as crashed or timed out under the same driver.
fn skip_rules(
    cache: &CacheFile,
    fingerprint: Option<&SystemFingerprint>,
    config: &ProbeConfig,
    in_process: bool,
) -> Vec<(SkipRule, CacheEntry)> {
    cache
        .devices
        .iter()
        .filter(|entry| !config.force || (in_process && entry.is_crash()))
        .filter(|entry| !(config.software_rasterizers && entry.is_skipped_rasterizer()))
        .filter_map(|entry| {
            let rule = match fingerprint {
                Some(fingerprint) => {
                    if !entry_valid(entry, fingerprint) {
                        return None;
                    }
                    match &entry.key {
                        Some(key) => SkipRule::Key { key: key.clone() },
                        None => SkipRule::Id {
                            id: entry.device.id,
                        },
                    }
                }
                None => SkipRule::Enumeration {
                    id: entry.device.id,
                    key: entry.key.clone(),
                    enum_fingerprint: entry.enum_fingerprint.clone(),
                },
            };
            Some((rule, entry.clone()))
        })
        .take(MAX_DEVICES)
        .collect()
}

/// Why the prober path gave up before any device result.
enum Fallback {
    Cancelled,
    Unavailable(String),
}

/// State of one probe after the cache check.
struct Run<'a> {
    config: &'a ProbeConfig,
    fingerprint: Option<&'a SystemFingerprint>,
    /// The cache entry of each skip rule, by rule index.
    planned: &'a [CacheEntry],
    cache: CacheFile,
    cache_path: Option<PathBuf>,
    warnings: Vec<String>,
    prober_starts: u32,
}

impl Run<'_> {
    /// The device as reported for an outcome. `bool`: from the cache.
    fn final_device(&self, id: DeviceId, outcome: &Outcome) -> (DeviceInfo, bool) {
        let (device, cached) = match outcome {
            Outcome::Probed(device) => ((**device).clone(), false),
            Outcome::Skipped { rule } => match self.planned.get(*rule) {
                Some(entry) => (entry.device.clone(), true),
                // The collector checked the index; unreachable, but never panic on it.
                None => (unprobed_device(id), false),
            },
            blamed => (
                DeviceInfo {
                    status: blamed.blame_status().unwrap_or(DeviceStatus::NotProbed),
                    ..unprobed_device(id)
                },
                false,
            ),
        };
        (as_reported(id, device, self.config), cached)
    }

    /// Sends the events of an accepted line.
    fn emit(
        &self,
        collector: &Collector,
        accepted: &Accepted,
        on_event: &mut dyn FnMut(ProbeEvent),
    ) {
        match accepted {
            Accepted::Enumerated(id) => on_event(ProbeEvent::DeviceEnumerated { id: *id }),
            Accepted::Outcome(id) => {
                if let Some((_, outcome)) = collector.outcomes.iter().find(|(done, _)| done == id) {
                    let (device, cached) = self.final_device(*id, outcome);
                    on_event(ProbeEvent::DeviceProbed { device, cached });
                }
            }
            Accepted::Nothing | Accepted::Done => {}
        }
    }

    /// Blames `id` (or a backend) for a prober that died, hung or misbehaved.
    fn blame(
        &mut self,
        collector: &mut Collector,
        id: Option<DeviceId>,
        backend: Option<BackendKind>,
        outcome: Outcome,
        what: &str,
        on_event: &mut dyn FnMut(ProbeEvent),
    ) -> bool {
        // "was killed by signal 11 (SIGSEGV) while probing vulkan:cpu:0 (last stderr line: …)".
        let (what, note) = what
            .split_once(supervisor::STDERR_NOTE)
            .map_or((what, String::new()), |(what, line)| {
                (what, format!(" (last stderr line: {line})"))
            });
        if let Some(id) = id {
            let message = format!("the device prober {what} while probing {id}{note}");
            tracing::warn!("{message}");
            self.warnings.push(message);
            collector.blame(id, outcome.clone());
            let (device, cached) = self.final_device(id, &outcome);
            on_event(ProbeEvent::DeviceProbed { device, cached });
            true
        } else if let Some(backend) = backend {
            let message =
                format!("the device prober {what} while enumerating {backend} devices{note}");
            tracing::warn!("{message}");
            self.warnings.push(message.clone());
            collector.fail_backend(backend, message);
            true
        } else {
            let message = format!("the device prober {what}{note}");
            tracing::warn!("{message}");
            self.warnings.push(message);
            false
        }
    }

    /// Probes through prober processes, restarting after a crash or hang for the devices not yet
    /// reported (at most once per device or backend that was blamed).
    fn out_of_process(
        &mut self,
        request: &ProbeRequest,
        cancelled: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(ProbeEvent),
    ) -> Result<Collector, Fallback> {
        let Some(command) = self.config.prober.clone() else {
            return Err(Fallback::Unavailable("no prober".into()));
        };
        let mut collector = Collector::new(request.clone(), true);
        let mut next = request.clone();
        loop {
            self.prober_starts += 1;
            let outcomes_before = collector.outcomes.len();
            let end = supervisor::run_once(
                &command,
                &next,
                self.config.timeout_per_device,
                cancelled,
                &mut collector,
                &mut |collector, accepted| self.emit(collector, &accepted, on_event),
            );
            let blamed = match end {
                Ok(RunEnd::Completed) => break,
                Ok(RunEnd::Died {
                    in_progress,
                    enumerating,
                    detail,
                }) => self.blame(
                    &mut collector,
                    in_progress,
                    enumerating,
                    Outcome::Crashed,
                    &detail,
                    on_event,
                ),
                Ok(RunEnd::TimedOut {
                    in_progress,
                    enumerating,
                }) => {
                    let what = format!(
                        "sent nothing for {} s and was killed",
                        self.config.timeout_per_device.as_secs_f32()
                    );
                    self.blame(
                        &mut collector,
                        in_progress,
                        enumerating,
                        Outcome::TimedOut,
                        &what,
                        on_event,
                    )
                }
                Ok(RunEnd::ProtocolError {
                    in_progress,
                    message,
                }) => {
                    let what = format!("sent invalid output ({message}) and was killed");
                    let outcome = Outcome::Failed(format!(
                        "the device prober sent invalid output: {message}"
                    ));
                    self.blame(&mut collector, in_progress, None, outcome, &what, on_event)
                }
                Err(StartError::Cancelled) => return Err(Fallback::Cancelled),
                Err(StartError::Spawn(message) | StartError::NoHeader(message)) => {
                    if self.prober_starts == 1 {
                        return Err(Fallback::Unavailable(message));
                    }
                    let message = format!("the device prober could not be restarted: {message}");
                    tracing::warn!("{message}");
                    self.warnings.push(message);
                    break;
                }
            };
            let progressed = blamed || collector.outcomes.len() > outcomes_before;
            let unknown_backends = collector
                .failed_backends
                .iter()
                .any(|(backend, _)| !next.skip_backends.contains(backend));
            if !progressed
                || (collector.remaining().is_empty() && !unknown_backends)
                || self.prober_starts as usize > MAX_DEVICES + BackendKind::ALL.len()
            {
                break;
            }
            next.exclude = collector.outcomes.iter().map(|(id, _)| *id).collect();
            next.skip_backends = collector
                .failed_backends
                .iter()
                .map(|(backend, _)| *backend)
                .collect();
            tracing::warn!(
                remaining = collector.remaining().len(),
                "restarting the device prober for the devices not yet probed"
            );
        }
        // Devices no prober got to.
        for id in collector.remaining() {
            let outcome = Outcome::Failed("the device prober ended before probing it".into());
            collector.blame(id, outcome.clone());
            let (device, cached) = self.final_device(id, &outcome);
            on_event(ProbeEvent::DeviceProbed { device, cached });
        }
        Ok(collector)
    }

    /// Probes in this process, in probe order, with the crash journal; the cache is saved after
    /// every device so a crash loses nothing already probed.
    fn in_process(
        &mut self,
        request: ProbeRequest,
        cancelled: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(ProbeEvent),
    ) -> Result<Collector, ProbeError> {
        let source = device_source();
        let mut collector = Collector::new(request.clone(), false);
        let mut journal = match &self.config.cache_dir {
            Some(dir) => match Journal::create(dir) {
                Ok(journal) => Some(journal),
                Err(error) => {
                    tracing::warn!(%error, "could not create the crash journal");
                    None
                }
            },
            None => None,
        };
        if journal.is_none() {
            let message = "probing in this process without a crash journal: a driver that \
                           crashes will crash the app again on the next start";
            tracing::warn!("{message}");
            self.warnings.push(message.into());
        }
        let mut was_cancelled = false;
        let result = run_session(&request, source.as_ref(), false, &mut |line| {
            if cancelled() {
                was_cancelled = true;
                return Err(std::io::Error::other("cancelled"));
            }
            if let (super::protocol::ProberLine::Probing { id }, Some(journal)) =
                (&line, journal.as_mut())
            {
                let entry = self.cache_entry(&collector, *id, &Outcome::Crashed, false);
                journal.begin(&entry)?;
            }
            let accepted = collector.accept(line).map_err(std::io::Error::other)?;
            if let Accepted::Outcome(id) = accepted {
                if let Some(journal) = journal.as_mut() {
                    journal.end()?;
                }
                self.store(&collector, id);
            }
            self.emit(&collector, &accepted, on_event);
            Ok(())
        });
        drop(journal);
        if was_cancelled {
            return Err(ProbeError::Cancelled);
        }
        if let Err(error) = result {
            // Only the journal can fail here (the collector accepts what the session writes).
            let message = format!("in-process probing stopped: {error}");
            tracing::warn!("{message}");
            self.warnings.push(message);
        }
        Ok(collector)
    }

    /// The cache entry for a device's outcome.
    fn cache_entry(
        &self,
        collector: &Collector,
        id: DeviceId,
        outcome: &Outcome,
        isolated: bool,
    ) -> CacheEntry {
        let details = collector.details(id);
        let key = details.and_then(|details| details.key.clone());
        let mut device = match outcome {
            Outcome::Probed(device) => (**device).clone(),
            Outcome::Skipped { rule } => self
                .planned
                .get(*rule)
                .map_or_else(|| unprobed_device(id), |entry| entry.device.clone()),
            blamed => DeviceInfo {
                status: blamed
                    .blame_status()
                    .unwrap_or(DeviceStatus::NotProbed)
                    .with_isolated(isolated),
                ..unprobed_device(id)
            },
        };
        device.id = id;
        device.backend = id.backend();
        let fingerprint = self.fingerprint.and_then(|fingerprint| {
            fingerprint.for_device(key.as_deref(), device.identity.pci_bus_id.as_deref())
        });
        CacheEntry {
            device,
            key,
            enum_fingerprint: details.and_then(|details| details.enum_fingerprint.clone()),
            fingerprint,
        }
    }

    /// Stores the outcome of `id` in the cache and saves it, marked incomplete until the probe
    /// finished (in process: after every device, so a crash loses nothing already probed).
    fn store(&mut self, collector: &Collector, id: DeviceId) {
        if self.store_entry(collector, id) {
            let complete = std::mem::replace(&mut self.cache.complete, false);
            self.save();
            self.cache.complete = complete;
        }
    }

    /// Stores the outcome of `id` in the cache. Protocol failures are not cached: they say nothing
    /// about the device.
    fn store_entry(&mut self, collector: &Collector, id: DeviceId) -> bool {
        let Some((_, outcome)) = collector.outcomes.iter().find(|(done, _)| *done == id) else {
            return false;
        };
        if matches!(outcome, Outcome::Failed(_)) {
            return false;
        }
        let entry = self.cache_entry(collector, id, outcome, true);
        self.cache.upsert(entry);
        true
    }

    fn save(&mut self) {
        self.cache.gpu_keys = self
            .fingerprint
            .map(SystemFingerprint::gpu_keys)
            .unwrap_or_default();
        if let Some(path) = &self.cache_path {
            cache::save(path, &self.cache);
        }
    }

    /// Updates and saves the cache, and builds the report.
    fn finish(mut self, collector: &Collector) -> ProbeReport {
        for (id, _) in &collector.outcomes {
            self.store_entry(collector, *id);
        }
        if self.config.devices.is_none() {
            // A probe of all devices: the cache is complete unless a backend's devices are
            // unknown.
            self.cache.complete = collector.failed_backends.is_empty();
            if self.cache.complete {
                // Devices that are gone.
                self.cache
                    .devices
                    .retain(|entry| collector.details(entry.device.id).is_some());
            }
        }
        self.save();

        let mut devices = Vec::new();
        let mut cache_hits = Vec::new();
        for (id, outcome) in &collector.outcomes {
            let (device, cached) = self.final_device(*id, outcome);
            if cached {
                cache_hits.push(*id);
            }
            devices.push(device);
        }
        for (backend, message) in &collector.failed_backends {
            let message = format!("{backend} devices are unknown: {message}");
            if !self.warnings.contains(&message) {
                self.warnings.push(message);
            }
        }
        report(
            devices,
            cache_hits,
            self.prober_starts,
            self.fingerprint.is_some(),
            self.warnings,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::device::unprobed_device;

    fn fingerprint(shared: &str) -> SystemFingerprint {
        SystemFingerprint {
            gpus: BTreeMap::new(),
            shared: shared.into(),
        }
    }

    fn entry(id: DeviceId, status: DeviceStatus, fingerprint: &str) -> CacheEntry {
        CacheEntry {
            device: DeviceInfo {
                status,
                ..unprobed_device(id)
            },
            key: None,
            enum_fingerprint: Some("enum".into()),
            fingerprint: Some(fingerprint.into()),
        }
    }

    fn crashed() -> DeviceStatus {
        DeviceStatus::Unavailable {
            reason: UnavailableReason::CrashedDuringProbe,
            isolated: true,
        }
    }

    fn cache(entries: Vec<CacheEntry>) -> CacheFile {
        CacheFile {
            devices: entries,
            complete: true,
            ..CacheFile::empty()
        }
    }

    #[test]
    fn full_hit_needs_matching_fingerprints_and_options() {
        let lavapipe = crate::probe::session::tests::lavapipe();
        let available = DeviceStatus::Available { isolated: true };
        let cache = cache(vec![
            entry(DeviceId::Cpu, available.clone(), "a"),
            entry(lavapipe, crashed(), "a"),
        ]);
        let config = ProbeConfig::default();
        let hit = full_hit(&cache, &fingerprint("a"), &config).unwrap();
        // Software rasterizers off: lavapipe is reported skipped, whatever the cache says.
        assert_eq!(hit[1], skipped_device(lavapipe));
        let opt_in = ProbeConfig {
            software_rasterizers: true,
            ..ProbeConfig::default()
        };
        let hit = full_hit(&cache, &fingerprint("a"), &opt_in).unwrap();
        assert_eq!(hit[1].status, crashed());

        assert!(full_hit(&cache, &fingerprint("b"), &config).is_none());
        let incomplete = CacheFile {
            complete: false,
            ..cache.clone()
        };
        assert!(full_hit(&incomplete, &fingerprint("a"), &config).is_none());
        assert!(full_hit(&CacheFile::empty(), &fingerprint("a"), &config).is_none());
        let mut new_gpu = fingerprint("a");
        new_gpu.gpus.insert("pci:0000:05:00.0".into(), "x".into());
        assert!(full_hit(&cache, &new_gpu, &config).is_none());

        let only_cpu = ProbeConfig {
            devices: Some(vec![DeviceId::Cpu]),
            ..ProbeConfig::default()
        };
        assert_eq!(
            full_hit(&cache, &fingerprint("a"), &only_cpu)
                .unwrap()
                .len(),
            1
        );
        let unknown = ProbeConfig {
            devices: Some(vec![DeviceId::Cuda { index: 0 }]),
            ..ProbeConfig::default()
        };
        assert!(full_hit(&cache, &fingerprint("a"), &unknown).is_none());
    }

    #[test]
    fn skip_rules_follow_fingerprint_force_and_isolation() {
        let available = DeviceStatus::Available { isolated: true };
        let mut keyed = entry(
            DeviceId::Vulkan {
                adapter: crate::device::AdapterKind::DiscreteGpu,
                index: 0,
            },
            available.clone(),
            "gpu",
        );
        keyed.key = Some("pci:0000:01:00.0".into());
        let cache = cache(vec![
            entry(DeviceId::Cpu, available, "a"),
            entry(DeviceId::Cuda { index: 0 }, crashed(), "a"),
            keyed,
        ]);
        let mut known = fingerprint("a");
        known.gpus.insert("pci:0000:01:00.0".into(), "gpu".into());
        let config = ProbeConfig::default();

        let rules: Vec<SkipRule> = skip_rules(&cache, Some(&known), &config, false)
            .into_iter()
            .map(|(rule, _)| rule)
            .collect();
        assert_eq!(
            rules,
            vec![
                SkipRule::Id { id: DeviceId::Cpu },
                SkipRule::Id {
                    id: DeviceId::Cuda { index: 0 }
                },
                SkipRule::Key {
                    key: "pci:0000:01:00.0".into()
                },
            ]
        );
        // A changed shared fingerprint invalidates the keyless entries only.
        let mut changed = known.clone();
        changed.shared = "b".into();
        assert_eq!(skip_rules(&cache, Some(&changed), &config, false).len(), 1);

        // Unknown fingerprint: enumerate-only comparison for every entry.
        let rules = skip_rules(&cache, None, &config, false);
        assert!(
            rules
                .iter()
                .all(|(rule, _)| matches!(rule, SkipRule::Enumeration { .. }))
        );
        assert_eq!(rules.len(), 3);

        // Force: no rules out of process; in process, crashed devices are still not reopened.
        let force = ProbeConfig {
            force: true,
            ..ProbeConfig::default()
        };
        assert!(skip_rules(&cache, Some(&known), &force, false).is_empty());
        let in_process = skip_rules(&cache, Some(&known), &force, true);
        assert_eq!(in_process.len(), 1);
        assert_eq!(in_process[0].1.device.id, DeviceId::Cuda { index: 0 });
    }

    #[test]
    fn report_isolation_ignores_skipped_rasterizers() {
        let lavapipe = crate::probe::session::tests::lavapipe();
        let isolated = DeviceInfo {
            status: DeviceStatus::Available { isolated: true },
            ..unprobed_device(DeviceId::Cpu)
        };
        let report_isolated = report(
            vec![isolated.clone(), skipped_device(lavapipe)],
            vec![],
            1,
            true,
            vec![],
        );
        assert!(report_isolated.isolated);
        // Probe order: lavapipe before the CPU.
        assert_eq!(report_isolated.devices[0].id, lavapipe);
        let not_isolated = DeviceInfo {
            status: DeviceStatus::Available { isolated: false },
            ..isolated
        };
        assert!(!report(vec![not_isolated], vec![], 0, true, vec![]).isolated);
    }

    /// The in-process path end to end on the CPU backend: the CPU is probed (not isolated), the
    /// cache is written, and the next probe is a full cache hit that opens nothing.
    #[cfg(feature = "cpu")]
    #[test]
    fn in_process_probe_then_cache_hit() {
        let dir = std::env::temp_dir().join(format!("polypous-test-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = ProbeConfig {
            isolation: Isolation::InProcess,
            cache_dir: Some(dir.clone()),
            ..ProbeConfig::default()
        };
        let mut events = Vec::new();
        let first = probe_devices(&config, &|| false, &mut |event| events.push(event)).unwrap();
        let cpu = first
            .devices
            .iter()
            .find(|device| device.id == DeviceId::Cpu)
            .unwrap();
        assert_eq!(cpu.status, DeviceStatus::Available { isolated: false });
        assert!(!first.isolated);
        assert_eq!(first.prober_starts, 0);
        assert!(events.contains(&ProbeEvent::DeviceEnumerated { id: DeviceId::Cpu }));

        let second = probe_devices(&config, &|| false, &mut |_| {}).unwrap();
        assert_eq!(second.devices, first.devices);
        assert_eq!(second.cache_hits.len(), second.devices.len());
        // No journal is left behind.
        let journals = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().contains("journal"))
            .count();
        assert_eq!(journals, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn out_of_process_without_a_prober_is_an_error() {
        let config = ProbeConfig {
            isolation: Isolation::OutOfProcess,
            ..ProbeConfig::default()
        };
        assert!(matches!(
            probe_devices(&config, &|| false, &mut |_| {}),
            Err(ProbeError::ProberUnavailable(_))
        ));
    }
}
