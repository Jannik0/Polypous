//! The up-front open of every Vulkan device on the first Vulkan open (placement.md §1, "Remaining
//! window"; issue #118).
//!
//! # Why
//!
//! The first open of a Vulkan device makes CubeCL create a temporary wgpu instance (cubecl-wgpu
//! 0.11.3 `try_create_setup` → `probe`). Its `vkEnumeratePhysicalDevices` makes the Vulkan loader
//! unload the device-less ICDs, and on Debian 13 (loader 1.4.309) it does so without the lock that
//! another thread's loader-implemented device call (`vkSetDebugUtilsObjectNameEXT`, which wgpu
//! issues in debug builds) takes: that thread crashes with SIGSEGV (register M2-13). Compute never
//! takes [`DEVICE_LOCK`](super::discover::lock_devices), so the lock alone can't keep the two
//! apart. What does: once every Vulkan device the process will use is open, no temporary instance
//! is created again, so none can meet Vulkan compute.
//!
//! # What
//!
//! The first time [`resolve_device`](super::resolve_device) is asked for any Vulkan device, it runs
//! [`OpenSet::prepare_with`]'s step, once per process and under the device lock: every Vulkan
//! device of the **open set** is opened, in enumeration order, before the caller's device is
//! used. The open set is
//!
//! - by default every Vulkan device [`list_backend_devices`](super::list_backend_devices) lists
//!   that is not a software rasterizer ([`DeviceId::is_software_rasterizer`]; the same devices
//!   discovery skips by default);
//! - once the facade has passed a probe's results ([`note_probe_results`]), only the listed Vulkan
//!   devices some probe found usable and no probe rejected (ADR-0013: a device the prober rejected
//!   is never opened in the app's process by this step). A software rasterizer the caller opted
//!   into joins the set then, because the probe opened and checked it in the prober.
//!
//! The set is fixed once the step has run: later probe results don't open or close anything.
//!
//! A device of the set that fails to open (CubeCL panics) is caught and remembered: every later
//! use of it fails with the reason, and it is never opened again, because a retry would create
//! another temporary instance. Catching that panic is acceptable here, unlike in
//! `DynTensor::from_data` (see `resolve.rs`): the panic comes from CubeCL's service
//! initialization, which inserts nothing into its device registry when it fails (cubecl-common
//! 0.11.3 `device/handle/channel.rs` `ChannelDeviceState::init`: the registry lock is a
//! non-poisoning spin lock, released on unwind, and the entry is inserted only after the service
//! was created), so other devices' state is untouched; and the failed device is never addressed
//! again, so whatever its runner thread holds is never used. `describe_device` has caught the same
//! panic since M0.
//!
//! A device outside the set that is addressed later (an opted-in software rasterizer of a direct
//! backend user, a device a probe rejected, a device listed after the step ran) still opens alone:
//! that is the remaining window placement.md §1 documents.
//!
//! The probe session (prober process and in-process fallback) doesn't run the step: it opens one
//! device at a time on purpose, so a crash is attributed to the device that caused it
//! ([`resolve_device_alone`](super::resolve::resolve_device_alone)).
//!
//! The state is process-wide hardware state, like the enumeration cache: it holds which devices
//! are open, which CubeCL keeps for the whole process anyway, not engine state (public-api.md §2
//! rule 9).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::discover::panic_message;
use super::info::{BackendKind, DeviceId, DeviceInfo, DeviceStatus, UnavailableReason};

/// What the up-front Vulkan open of this process did ([`vulkan_open_step`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VulkanOpenStep {
    /// How many times the step ran in this process. One once it has run; tests check it.
    pub runs: u32,
    /// The devices the step opened, in the order it opened them.
    pub opened: Vec<DeviceId>,
    /// The devices that failed to open, with the reason. They are never opened again in this
    /// process; a use of one fails with this reason.
    pub failed: Vec<(DeviceId, String)>,
}

impl VulkanOpenStep {
    fn failure(&self, id: DeviceId) -> Option<&str> {
        self.failed
            .iter()
            .find(|(failed, _)| *failed == id)
            .map(|(_, reason)| reason.as_str())
    }
}

/// What probes said about Vulkan devices ([`OpenSet::note_probe_results`]).
#[derive(Debug, Default)]
struct ProbeVerdicts {
    /// Devices some probe found usable.
    usable: Vec<DeviceId>,
    /// Devices some probe rejected (crashed, timed out, failed, library missing).
    rejected: Vec<DeviceId>,
}

#[derive(Debug, Default)]
struct State {
    /// `None` until a probe's results were passed: the default set then.
    verdicts: Option<ProbeVerdicts>,
    /// `None` until the step has run.
    step: Option<VulkanOpenStep>,
}

/// The open set and the result of the step. One per process ([`OPEN_SET`]); tests make their own.
#[derive(Debug)]
pub(crate) struct OpenSet {
    state: Mutex<State>,
}

/// The open set of this process.
static OPEN_SET: OpenSet = OpenSet::new();

impl OpenSet {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(State {
                verdicts: None,
                step: None,
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        // The state is replaced field by field with complete values, and nothing in this module
        // panics while holding it, so a poisoned lock still holds consistent data.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// What the step did, if it has run.
    pub(crate) fn step(&self) -> Option<VulkanOpenStep> {
        self.state().step.clone()
    }

    /// Takes a probe's results: Vulkan devices that are available become usable, those that are
    /// unavailable for any reason but [`UnavailableReason::SoftwareRasterizerSkipped`] (not opened,
    /// so no verdict) become rejected; devices not probed give no verdict. A device is in the set
    /// if some probe found it usable and none rejected it. Returns `false`, changing nothing, if
    /// the step has already run (the set is fixed then).
    pub(crate) fn note_probe_results(&self, devices: &[DeviceInfo]) -> bool {
        let mut state = self.state();
        if state.step.is_some() {
            return false;
        }
        let verdicts = state.verdicts.get_or_insert_with(ProbeVerdicts::default);
        for device in devices {
            if device.id.backend() != BackendKind::Vulkan {
                continue;
            }
            let list = match &device.status {
                DeviceStatus::Available { .. } => &mut verdicts.usable,
                DeviceStatus::Unavailable {
                    reason: UnavailableReason::SoftwareRasterizerSkipped,
                    ..
                }
                | DeviceStatus::NotProbed => continue,
                DeviceStatus::Unavailable { .. } => &mut verdicts.rejected,
            };
            if !list.contains(&device.id) {
                list.push(device.id);
            }
        }
        true
    }

    /// The devices of `listed` the step opens, in `listed`'s order.
    fn members(verdicts: Option<&ProbeVerdicts>, listed: &[DeviceId]) -> Vec<DeviceId> {
        listed
            .iter()
            .copied()
            .filter(|id| id.backend() == BackendKind::Vulkan)
            .filter(|id| match verdicts {
                None => !id.is_software_rasterizer(),
                Some(verdicts) => verdicts.usable.contains(id) && !verdicts.rejected.contains(id),
            })
            .collect()
    }

    /// Prepares the use of `id`: if `id` is a Vulkan device and the step hasn't run yet, runs it
    /// (under `lock`, which every thread takes, so exactly one runs it and the others wait for
    /// its result); then fails with the remembered reason if `id` failed to open in the step.
    ///
    /// `listed` gives the Vulkan device list (called before `lock`, because the first
    /// enumeration takes the device lock). `open` opens one device; its panics are caught.
    pub(crate) fn prepare_with<G>(
        &self,
        id: DeviceId,
        listed: impl FnOnce() -> Vec<DeviceId>,
        lock: impl FnOnce() -> G,
        mut open: impl FnMut(DeviceId) -> Result<(), String>,
    ) -> Result<(), String> {
        if id.backend() != BackendKind::Vulkan {
            return Ok(());
        }
        if let Some(step) = &self.state().step {
            return Self::check(step, id);
        }
        let listed = listed();
        let _guard = lock();
        // Another thread may have run the step while this one waited for the lock.
        let members = {
            let state = self.state();
            if let Some(step) = &state.step {
                return Self::check(step, id);
            }
            Self::members(state.verdicts.as_ref(), &listed)
        };
        tracing::debug!(
            devices = ?members,
            caller = %id,
            "first Vulkan open: opening every Vulkan device of the open set"
        );
        let mut done = VulkanOpenStep {
            runs: 1,
            ..VulkanOpenStep::default()
        };
        for member in members {
            let started = std::time::Instant::now();
            let result = catch_unwind(AssertUnwindSafe(|| open(member)))
                .unwrap_or_else(|payload| Err(format!("panic: {}", panic_message(&*payload))));
            match result {
                Ok(()) => {
                    tracing::debug!(device = %member, elapsed_ms = started.elapsed().as_millis(), "opened");
                    done.opened.push(member);
                }
                Err(reason) => {
                    tracing::warn!(device = %member, %reason, "could not open the device; it is not used in this process");
                    done.failed.push((member, reason));
                }
            }
        }
        if !done.opened.contains(&id) && done.failure(id).is_none() {
            tracing::debug!(device = %id, "not in the Vulkan open set: opens alone");
        }
        let mut state = self.state();
        // Appended, not replaced: a second run (a bug) shows as `runs > 1` and repeated devices.
        let step = state.step.get_or_insert_with(VulkanOpenStep::default);
        step.runs += done.runs;
        step.opened.extend(done.opened);
        step.failed.extend(done.failed);
        Self::check(step, id)
    }

    /// `Err` with the reason if `id` failed to open in the step.
    fn check(step: &VulkanOpenStep, id: DeviceId) -> Result<(), String> {
        match step.failure(id) {
            Some(reason) => Err(format!(
                "{id} failed to open earlier in this process and is not opened again: {reason}"
            )),
            None => Ok(()),
        }
    }

    /// `Err` with the reason if `id` failed to open in the step; never runs the step.
    pub(crate) fn check_failed(&self, id: DeviceId) -> Result<(), String> {
        match &self.state().step {
            Some(step) => Self::check(step, id),
            None => Ok(()),
        }
    }
}

/// [`OpenSet::prepare_with`] on this process's open set with the real device list, device lock
/// and opener. Must not be called while holding the device lock.
pub(crate) fn prepare(id: DeviceId) -> Result<(), String> {
    OPEN_SET.prepare_with(
        id,
        || super::discover::list_backend_devices(BackendKind::Vulkan).unwrap_or_default(),
        super::discover::lock_devices,
        super::discover::open_device,
    )
}

/// Whether `id` failed to open in this process's step ([`OpenSet::check_failed`]).
pub(crate) fn check_failed(id: DeviceId) -> Result<(), String> {
    OPEN_SET.check_failed(id)
}

/// Restricts the Vulkan open set of this process to the devices a probe found usable
/// ([`OpenSet::note_probe_results`]): a device is opened by the up-front open only if some probe
/// found it usable and no probe rejected it. Returns `false`, changing nothing, if the up-front
/// open has already run.
///
/// Workspace-internal: the `polypous` facade calls it with every probe report an `Engine` gets,
/// before any of its devices is used. Not part of the facade's API.
#[doc(hidden)]
pub fn note_probe_results(devices: &[DeviceInfo]) -> bool {
    OPEN_SET.note_probe_results(devices)
}

/// What the up-front Vulkan open of this process did, or `None` if no Vulkan device has been used
/// yet. Workspace-internal, for tests and diagnostics.
#[doc(hidden)]
pub fn vulkan_open_step() -> Option<VulkanOpenStep> {
    OPEN_SET.step()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::info::AdapterKind;
    use super::*;

    fn vulkan(adapter: AdapterKind, index: usize) -> DeviceId {
        DeviceId::Vulkan { adapter, index }
    }

    const DGPU: DeviceId = DeviceId::Vulkan {
        adapter: AdapterKind::DiscreteGpu,
        index: 0,
    };
    const IGPU: DeviceId = DeviceId::Vulkan {
        adapter: AdapterKind::IntegratedGpu,
        index: 0,
    };
    const LAVAPIPE: DeviceId = DeviceId::Vulkan {
        adapter: AdapterKind::Cpu,
        index: 0,
    };

    /// The reference machine's list on Debian 13.
    fn listed() -> Vec<DeviceId> {
        vec![
            DeviceId::Cuda { index: 0 },
            DGPU,
            IGPU,
            LAVAPIPE,
            DeviceId::Cpu,
        ]
    }

    /// A lock that protects nothing: these tests use their own `OpenSet` (the real one takes the
    /// device lock).
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    fn lock() -> MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn info(id: DeviceId, status: DeviceStatus) -> DeviceInfo {
        DeviceInfo {
            status,
            ..super::super::unprobed_device(id)
        }
    }

    fn available(id: DeviceId) -> DeviceInfo {
        info(id, DeviceStatus::Available { isolated: true })
    }

    fn rejected(id: DeviceId) -> DeviceInfo {
        info(
            id,
            DeviceStatus::Unavailable {
                reason: UnavailableReason::CrashedDuringProbe,
                isolated: true,
            },
        )
    }

    fn skipped(id: DeviceId) -> DeviceInfo {
        info(
            id,
            DeviceStatus::Unavailable {
                reason: UnavailableReason::SoftwareRasterizerSkipped,
                isolated: false,
            },
        )
    }

    /// Runs `prepare_with` for `id` and records every open in `log`, with a marker for the
    /// caller's use after it returned.
    fn use_device(
        set: &OpenSet,
        id: DeviceId,
        log: &Mutex<Vec<String>>,
        fail: &[DeviceId],
    ) -> Result<(), String> {
        let result = set.prepare_with(id, listed, lock, |member| {
            log.lock().unwrap().push(format!("open {member}"));
            if fail.contains(&member) {
                panic!("Unable to acquire wgpu device");
            }
            Ok(())
        });
        if result.is_ok() {
            log.lock().unwrap().push(format!("use {id}"));
        }
        result
    }

    /// By default the step opens every listed Vulkan device but software rasterizers, before the
    /// caller's device is used, once.
    #[test]
    fn default_set_is_opened_once_before_the_caller() {
        let set = OpenSet::new();
        let log = Mutex::new(Vec::new());
        assert_eq!(set.step(), None);
        assert_eq!(use_device(&set, IGPU, &log, &[]), Ok(()));
        assert_eq!(
            *log.lock().unwrap(),
            [
                "open vulkan:discrete_gpu:0",
                "open vulkan:integrated_gpu:0",
                "use vulkan:integrated_gpu:0"
            ]
        );
        for id in [DGPU, IGPU, LAVAPIPE] {
            assert_eq!(use_device(&set, id, &log, &[]), Ok(()));
        }
        assert_eq!(log.lock().unwrap().len(), 6, "no device opened again");
        assert_eq!(
            set.step(),
            Some(VulkanOpenStep {
                runs: 1,
                opened: vec![DGPU, IGPU],
                failed: vec![],
            })
        );
    }

    /// Non-Vulkan devices don't run the step.
    #[test]
    fn other_backends_do_not_run_the_step() {
        let set = OpenSet::new();
        let log = Mutex::new(Vec::new());
        for id in [DeviceId::Cpu, DeviceId::Cuda { index: 0 }] {
            assert_eq!(use_device(&set, id, &log, &[]), Ok(()));
        }
        assert_eq!(set.step(), None);
        assert!(
            log.lock()
                .unwrap()
                .iter()
                .all(|line| line.starts_with("use"))
        );
    }

    /// Many threads make their first Vulkan call at once: one runs the step, the others wait for
    /// it and then find their device open.
    #[test]
    fn concurrent_first_calls_run_the_step_once() {
        let set = OpenSet::new();
        let opens = AtomicUsize::new(0);
        let step_done = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for thread in 0..16 {
                let (set, opens, step_done) = (&set, &opens, &step_done);
                scope.spawn(move || {
                    let id = if thread % 2 == 0 { DGPU } else { IGPU };
                    let result = set.prepare_with(id, listed, lock, |_| {
                        opens.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        Ok(())
                    });
                    assert_eq!(result, Ok(()));
                    // Every caller returns only after the step opened both devices.
                    assert_eq!(opens.load(Ordering::SeqCst), 2);
                    step_done.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(step_done.load(Ordering::SeqCst), 16);
        assert_eq!(opens.load(Ordering::SeqCst), 2);
        assert_eq!(set.step().map(|step| step.runs), Some(1));
    }

    /// A device that fails to open is remembered: its uses fail with the reason, it is never
    /// opened again, and the other devices are unaffected.
    #[test]
    fn failed_devices_are_remembered_and_never_retried() {
        let set = OpenSet::new();
        let log = Mutex::new(Vec::new());
        // The caller asks for the healthy device; the broken one fails inside the step.
        assert_eq!(use_device(&set, DGPU, &log, &[IGPU]), Ok(()));
        for _ in 0..3 {
            let err = use_device(&set, IGPU, &log, &[IGPU]).unwrap_err();
            assert!(err.contains("Unable to acquire wgpu device"), "{err}");
            assert!(err.contains("not opened again"), "{err}");
            assert!(set.check_failed(IGPU).is_err());
        }
        assert_eq!(set.check_failed(DGPU), Ok(()));
        assert_eq!(
            log.lock()
                .unwrap()
                .iter()
                .filter(|line| line.starts_with("open"))
                .count(),
            2
        );
        let step = set.step().unwrap();
        assert_eq!(step.opened, [DGPU]);
        assert_eq!(step.failed.len(), 1);
        assert_eq!(step.failed[0].0, IGPU);
    }

    /// The caller's own device failing in the step is an error for that first use too.
    #[test]
    fn the_callers_device_failing_is_an_error() {
        let set = OpenSet::new();
        let log = Mutex::new(Vec::new());
        assert!(use_device(&set, DGPU, &log, &[DGPU]).is_err());
        assert_eq!(set.check_failed(IGPU), Ok(()));
    }

    /// With probe results, only devices a probe found usable and none rejected are opened; a
    /// software rasterizer a probe found usable joins; a skipped one gives no verdict.
    #[test]
    fn probe_results_restrict_the_set() {
        let set = OpenSet::new();
        // Engine 1 opted into software rasterizers; engine 2 didn't and its prober crashed on the
        // iGPU.
        assert!(set.note_probe_results(&[
            available(DGPU),
            available(IGPU),
            available(LAVAPIPE),
            available(DeviceId::Cpu),
        ]));
        assert!(set.note_probe_results(&[available(DGPU), rejected(IGPU), skipped(LAVAPIPE)]));
        let log = Mutex::new(Vec::new());
        assert_eq!(use_device(&set, IGPU, &log, &[]), Ok(()));
        assert_eq!(
            *log.lock().unwrap(),
            [
                "open vulkan:discrete_gpu:0",
                "open vulkan:cpu:0",
                // Outside the set: the caller opens it alone (the remaining window).
                "use vulkan:integrated_gpu:0"
            ]
        );
        // Fixed once the step has run.
        assert!(!set.note_probe_results(&[rejected(DGPU)]));
        assert_eq!(set.step().unwrap().opened, [DGPU, LAVAPIPE]);
    }

    /// A probe report that rejects every Vulkan device, or lists none, opens nothing.
    #[test]
    fn probe_results_can_empty_the_set() {
        for report in [
            vec![rejected(DGPU), rejected(IGPU), rejected(LAVAPIPE)],
            vec![available(DeviceId::Cpu)],
            vec![info(DGPU, DeviceStatus::NotProbed)],
        ] {
            let set = OpenSet::new();
            assert!(set.note_probe_results(&report));
            let log = Mutex::new(Vec::new());
            assert_eq!(use_device(&set, DGPU, &log, &[]), Ok(()));
            assert_eq!(*log.lock().unwrap(), ["use vulkan:discrete_gpu:0"]);
            assert_eq!(set.step().unwrap().opened, []);
        }
    }

    #[test]
    fn members_keep_the_enumeration_order() {
        let ids = vec![
            vulkan(AdapterKind::IntegratedGpu, 0),
            vulkan(AdapterKind::DiscreteGpu, 1),
            vulkan(AdapterKind::DiscreteGpu, 0),
        ];
        assert_eq!(OpenSet::members(None, &ids), ids);
    }
}
