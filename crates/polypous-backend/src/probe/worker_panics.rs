//! Panics of other threads during a device step, **in the prober process only** (placement.md §1,
//! "Probe failure reasons").
//!
//! When CubeCL can't create a device, the panic with the real cause happens on its device-runner
//! thread; the calling thread's `catch_unwind` only sees the closed channel (`RecvError`). The
//! prober installs a panic hook ([`install`]) that records the message of every panic, then calls
//! the previous hook. [`during_step`] collects the panics of *other* threads while one device is
//! described and probed, and [`attach`] appends them to a `probe failed` reason.
//!
//! In-process probing never installs the hook: the panic hook is the app's process-wide state
//! (public-api.md §2 rule 9), so there the reason stays as the calling thread saw it. The hook is
//! installed once per process and only by [`super::run_prober`], whose process exists to probe;
//! that makes it a fact about the process, the same for every probe in it.

use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread::ThreadId;

use crate::device::{DeviceStatus, UnavailableReason};

/// At most this many panics are kept per step; later ones are only counted.
const MAX_RECORDS: usize = 8;
/// At most this many bytes of one panic message are kept (cut at a character boundary).
const MAX_MESSAGE_BYTES: usize = 400;
/// At most this many distinct messages are attached to a reason.
const MAX_ATTACHED: usize = 3;

/// One recorded panic.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Record {
    thread: ThreadId,
    message: String,
}

#[derive(Debug, Default)]
struct Records {
    kept: Vec<Record>,
    dropped: usize,
}

/// The panics recorded since the current step began. Bounded: [`MAX_RECORDS`] messages of at most
/// [`MAX_MESSAGE_BYTES`] bytes.
#[derive(Debug, Default)]
pub(crate) struct Recorder {
    records: Mutex<Records>,
}

impl Recorder {
    fn lock(&self) -> std::sync::MutexGuard<'_, Records> {
        // The records stay consistent even if a thread panicked while holding the lock.
        self.records.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Records a panic of `thread`.
    fn record(&self, thread: ThreadId, message: &str) {
        let mut records = self.lock();
        if records.kept.len() < MAX_RECORDS {
            records.kept.push(Record {
                thread,
                message: bounded(message),
            });
        } else {
            records.dropped += 1;
        }
    }

    /// Forgets what was recorded before a step.
    fn begin(&self) {
        let mut records = self.lock();
        records.kept.clear();
        records.dropped = 0;
    }

    /// The panics of threads other than `current` since [`Recorder::begin`]: distinct messages in
    /// order, and how many more panics there were.
    fn take_others(&self, current: ThreadId) -> WorkerPanics {
        let mut records = self.lock();
        let mut messages: Vec<String> = Vec::new();
        let mut more = records.dropped;
        for record in records.kept.drain(..) {
            if record.thread == current || messages.contains(&record.message) {
                continue;
            }
            if messages.len() < MAX_ATTACHED {
                messages.push(record.message);
            } else {
                more += 1;
            }
        }
        records.dropped = 0;
        WorkerPanics { messages, more }
    }

    /// Runs one device step and returns its result with the panics of other threads during it.
    fn during<T>(&self, step: impl FnOnce() -> T) -> (T, WorkerPanics) {
        self.begin();
        let result = step();
        (result, self.take_others(std::thread::current().id()))
    }
}

/// Panics of other threads during one device step.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct WorkerPanics {
    /// Distinct messages, at most [`MAX_ATTACHED`], each at most [`MAX_MESSAGE_BYTES`] bytes.
    messages: Vec<String>,
    /// Panics not listed (beyond the limits; repeats of a listed message are not counted).
    more: usize,
}

/// The recorder of this process, set by [`install`].
static RECORDER: OnceLock<Recorder> = OnceLock::new();

/// Installs the recording panic hook (once per process; later calls do nothing). The previous
/// hook still runs after recording, so panics are printed as before. Call it only in the prober
/// process.
pub(crate) fn install() {
    RECORDER.get_or_init(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(recorder) = RECORDER.get() {
                let message = info
                    .payload_as_str()
                    .unwrap_or("panic with a non-text payload");
                recorder.record(std::thread::current().id(), message);
            }
            previous(info);
        }));
        Recorder::default()
    });
}

/// Runs one device step, collecting the panics of other threads during it if [`install`] was
/// called in this process; otherwise none are collected.
pub(crate) fn during_step<T>(step: impl FnOnce() -> T) -> (T, WorkerPanics) {
    match RECORDER.get() {
        Some(recorder) => recorder.during(step),
        None => (step(), WorkerPanics::default()),
    }
}

/// Appends the worker panics to a `probe failed` reason, for example `could not open the device:
/// panic: … RecvError (worker thread panicked: Unable to acquire wgpu device: …)`. Every other
/// status is returned unchanged.
pub(crate) fn attach(status: DeviceStatus, panics: &WorkerPanics) -> DeviceStatus {
    if panics.messages.is_empty() {
        return status;
    }
    match status {
        DeviceStatus::Unavailable {
            reason: UnavailableReason::ProbeFailed { message },
            isolated,
        } => {
            let threads = if panics.messages.len() + panics.more > 1 {
                "worker threads panicked"
            } else {
                "worker thread panicked"
            };
            let mut list = panics.messages.join("; ");
            if panics.more > 0 {
                list.push_str(&format!("; and {} more", panics.more));
            }
            DeviceStatus::Unavailable {
                reason: UnavailableReason::ProbeFailed {
                    message: format!("{message} ({threads}: {list})"),
                },
                isolated,
            }
        }
        other => other,
    }
}

/// `message` on one line (control characters become spaces), cut to at most
/// [`MAX_MESSAGE_BYTES`] bytes at a character boundary, with `…` if it was cut.
fn bounded(message: &str) -> String {
    let line: String = message
        .trim()
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    if line.len() <= MAX_MESSAGE_BYTES {
        return line;
    }
    let ellipsis = '…';
    let mut end = MAX_MESSAGE_BYTES - ellipsis.len_utf8();
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{ellipsis}", &line[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(message: &str) -> DeviceStatus {
        DeviceStatus::Unavailable {
            reason: UnavailableReason::ProbeFailed {
                message: message.into(),
            },
            isolated: true,
        }
    }

    fn reason(status: &DeviceStatus) -> String {
        match status {
            DeviceStatus::Unavailable {
                reason: UnavailableReason::ProbeFailed { message },
                ..
            } => message.clone(),
            other => panic!("not probe_failed: {other:?}"),
        }
    }

    /// A thread that panics during the step: its message is attached; the calling thread's own
    /// panic is not (it is already the reason).
    #[test]
    fn a_worker_panic_during_the_step_is_attached() {
        let recorder = Recorder::default();
        recorder.record(std::thread::current().id(), "before the step");
        let (status, panics) = recorder.during(|| {
            let worker = std::thread::spawn(|| std::thread::current().id())
                .join()
                .unwrap();
            recorder.record(
                worker,
                "Unable to acquire wgpu device: RequestDevice { message: \"Initialization of an \
                 object has failed\" }",
            );
            recorder.record(
                std::thread::current().id(),
                "called `Result::unwrap()` on an `Err` value: RecvError",
            );
            failed(
                "could not open the device: panic: called `Result::unwrap()` on an `Err` value: \
                 RecvError",
            )
        });
        assert_eq!(
            reason(&attach(status, &panics)),
            "could not open the device: panic: called `Result::unwrap()` on an `Err` value: \
             RecvError (worker thread panicked: Unable to acquire wgpu device: RequestDevice { \
             message: \"Initialization of an object has failed\" })"
        );
        // The next step starts empty.
        let ((), panics) = recorder.during(|| ());
        assert_eq!(panics, WorkerPanics::default());
    }

    #[test]
    fn only_probe_failures_get_the_panics() {
        let panics = WorkerPanics {
            messages: vec!["boom".into()],
            more: 0,
        };
        let available = DeviceStatus::Available { isolated: true };
        assert_eq!(attach(available.clone(), &panics), available);
        let crashed = DeviceStatus::Unavailable {
            reason: UnavailableReason::CrashedDuringProbe,
            isolated: true,
        };
        assert_eq!(attach(crashed.clone(), &panics), crashed);
        // No panics: unchanged.
        assert_eq!(attach(failed("x"), &WorkerPanics::default()), failed("x"));
    }

    #[test]
    fn panics_are_bounded_and_deduplicated() {
        let recorder = Recorder::default();
        let worker = std::thread::spawn(|| std::thread::current().id())
            .join()
            .unwrap();
        let ((), panics) = recorder.during(|| {
            for n in 0..20 {
                // Repeats of the first message, then distinct ones.
                let message = if n < 3 {
                    "same".to_string()
                } else {
                    format!("m{n}")
                };
                recorder.record(worker, &message);
            }
        });
        // 8 kept (3 × "same", m3..m7), 12 dropped; listed: same, m3, m4; more: m5, m6, m7 + 12.
        assert_eq!(panics.messages, vec!["same", "m3", "m4"]);
        assert_eq!(panics.more, 15);
        assert_eq!(
            reason(&attach(failed("panic: x"), &panics)),
            "panic: x (worker threads panicked: same; m3; m4; and 15 more)"
        );
    }

    #[test]
    fn long_messages_are_cut_at_a_character_boundary() {
        // 'é' is two bytes, so the cut lands inside one unless the boundary is respected.
        let long = "é".repeat(MAX_MESSAGE_BYTES);
        let cut = bounded(&long);
        assert!(cut.len() <= MAX_MESSAGE_BYTES, "{}", cut.len());
        assert!(cut.ends_with('…'));
        assert!(cut.trim_end_matches('…').chars().all(|c| c == 'é'));
        assert_eq!(bounded("  two\nlines\t "), "two lines");
        assert_eq!(bounded("short"), "short");
    }

    #[test]
    fn without_the_hook_nothing_is_collected() {
        // The library's unit tests never call `install` (in-process probing must not either).
        if RECORDER.get().is_none() {
            let (value, panics) = during_step(|| 7);
            assert_eq!(value, 7);
            assert_eq!(panics, WorkerPanics::default());
        }
    }
}
