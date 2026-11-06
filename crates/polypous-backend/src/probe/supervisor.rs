//! The app side of the prober protocol: starts one prober process, reads its output as untrusted
//! input, and reports how the process ended (ADR-0013 §1; placement.md §1, "Prober protocol").
//!
//! - stdout is read on its own thread, line by line, at most [`MAX_LINE_BYTES`] per line and
//!   [`MAX_LINES`] lines; anything else is a protocol error.
//! - stderr is drained on another thread (so the prober never blocks on a full pipe) and forwarded
//!   to the log at `debug`; the last lines are kept for the warning when the prober dies.
//! - stdin stays open and is never written: the prober's watchdog exits when it closes, so a prober
//!   never outlives the app.
//! - No line within the timeout, a protocol error or cancellation kills the prober. Every path
//!   waits for the child ([`ChildGuard`]), so no zombie is left behind.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::device::{BackendKind, DeviceId};

use super::collect::{Accepted, Collector};
use super::protocol::{
    MAX_LINE_BYTES, MAX_LINES, PROBER_FLAG, ProbeRequest, ProberLine, REQUEST_FLAG,
};

/// How often a waiting supervisor checks for cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(50);

/// How long to wait for a prober to exit after its `done` line before it is killed.
const EXIT_GRACE: Duration = Duration::from_secs(5);

/// Separates the exit description from the prober's last stderr line in [`RunEnd::Died`].
pub(crate) const STDERR_NOTE: &str = "; last stderr line: ";

/// stderr lines kept for diagnostics.
const STDERR_TAIL: usize = 20;

/// The executable to start as the prober. It is started as
/// `<program> --polypous-prober --request <json>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProberCommand {
    /// The executable.
    pub program: PathBuf,
}

/// How one prober process ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RunEnd {
    /// It sent `done`.
    Completed,
    /// It exited without `done`. `in_progress`/`enumerating` say what it was doing.
    Died {
        in_progress: Option<DeviceId>,
        enumerating: Option<BackendKind>,
        detail: String,
    },
    /// No line within the timeout; it was killed.
    TimedOut {
        in_progress: Option<DeviceId>,
        enumerating: Option<BackendKind>,
    },
    /// It sent something the protocol doesn't allow; it was killed.
    ProtocolError {
        in_progress: Option<DeviceId>,
        message: String,
    },
}

/// Why no prober result could be read at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StartError {
    /// The process could not be started.
    Spawn(String),
    /// It ended, hung or misbehaved before its header line.
    NoHeader(String),
    /// The caller cancelled; the prober was killed.
    Cancelled,
}

/// What the stdout reader thread sends.
enum ReadEvent {
    Line(String),
    Invalid(String),
    Eof,
}

/// Kills (if still running) and waits for the child on every exit path.
struct ChildGuard {
    child: Child,
    stdin: Option<ChildStdin>,
    waited: Option<ExitStatus>,
}

impl ChildGuard {
    fn kill_and_wait(&mut self) -> Option<ExitStatus> {
        if self.waited.is_none() {
            // Errors mean it already exited; `wait` reaps it either way.
            let _ = self.child.kill();
            self.waited = self.child.wait().ok();
        }
        self.waited
    }

    /// Closes stdin, then waits up to `grace` for the child to exit before killing it.
    fn finish(&mut self, grace: Duration) -> Option<ExitStatus> {
        self.stdin = None;
        let deadline = Instant::now() + grace;
        while self.waited.is_none() {
            match self.child.try_wait() {
                Ok(Some(status)) => self.waited = Some(status),
                Ok(None) if Instant::now() < deadline => std::thread::sleep(CANCEL_POLL),
                _ => return self.kill_and_wait(),
            }
        }
        self.waited
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.stdin = None;
        self.kill_and_wait();
    }
}

/// Starts one prober for `request` and feeds its lines to `collector` until it ends.
/// `on_accepted` is called for every accepted line, in order.
pub(crate) fn run_once(
    command: &ProberCommand,
    request: &ProbeRequest,
    timeout: Duration,
    cancelled: &dyn Fn() -> bool,
    collector: &mut Collector,
    on_accepted: &mut dyn FnMut(&Collector, Accepted),
) -> Result<RunEnd, StartError> {
    let json = serde_json::to_string(request)
        .map_err(|error| StartError::Spawn(format!("cannot encode the request: {error}")))?;
    let mut child = Command::new(&command.program)
        .args([
            OsString::from(PROBER_FLAG),
            OsString::from(REQUEST_FLAG),
            OsString::from(json),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            StartError::Spawn(format!(
                "cannot start `{}`: {error}",
                command.program.display()
            ))
        })?;
    let pid = child.id();
    tracing::info!(prober = %command.program.display(), pid, "started device prober");

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdin = child.stdin.take();
    let mut guard = ChildGuard {
        child,
        stdin,
        waited: None,
    };
    let (Some(stdout), Some(stderr)) = (stdout, stderr) else {
        return Err(StartError::Spawn("the prober's pipes are missing".into()));
    };
    let lines = spawn_stdout_reader(stdout);
    let stderr_tail = spawn_stderr_drain(stderr, pid);

    collector.start_run();
    let outcome = read_lines(&lines, timeout, cancelled, collector, on_accepted);
    let status = match &outcome {
        ReadOutcome::Eof => guard.finish(EXIT_GRACE),
        _ => guard.kill_and_wait(),
    };
    let detail = describe_exit(status, &stderr_tail);
    let in_progress = collector.in_progress();
    let enumerating = collector.enumerating();

    if !collector.header_seen() {
        return Err(match outcome {
            ReadOutcome::Cancelled => StartError::Cancelled,
            ReadOutcome::Eof => StartError::NoHeader(format!("the prober {detail}")),
            ReadOutcome::TimedOut => StartError::NoHeader(format!(
                "the prober sent nothing within {} s and was killed",
                timeout.as_secs_f32()
            )),
            ReadOutcome::Invalid(message) => {
                StartError::NoHeader(format!("the prober sent invalid output: {message}"))
            }
        });
    }
    Ok(match outcome {
        ReadOutcome::Cancelled => return Err(StartError::Cancelled),
        ReadOutcome::Eof if collector.done_seen() => RunEnd::Completed,
        ReadOutcome::Eof => RunEnd::Died {
            in_progress,
            enumerating,
            detail,
        },
        ReadOutcome::TimedOut => RunEnd::TimedOut {
            in_progress,
            enumerating,
        },
        ReadOutcome::Invalid(message) => RunEnd::ProtocolError {
            in_progress,
            message,
        },
    })
}

enum ReadOutcome {
    Eof,
    TimedOut,
    Cancelled,
    Invalid(String),
}

fn read_lines(
    lines: &Receiver<ReadEvent>,
    timeout: Duration,
    cancelled: &dyn Fn() -> bool,
    collector: &mut Collector,
    on_accepted: &mut dyn FnMut(&Collector, Accepted),
) -> ReadOutcome {
    let mut deadline = Instant::now() + timeout;
    loop {
        if cancelled() {
            return ReadOutcome::Cancelled;
        }
        let now = Instant::now();
        if now >= deadline {
            return ReadOutcome::TimedOut;
        }
        match lines.recv_timeout(CANCEL_POLL.min(deadline - now)) {
            Ok(ReadEvent::Line(text)) => {
                let line = match serde_json::from_str::<ProberLine>(&text) {
                    Ok(line) => line,
                    Err(error) => {
                        return ReadOutcome::Invalid(format!(
                            "malformed line ({error}): {}",
                            excerpt(&text)
                        ));
                    }
                };
                match collector.accept(line) {
                    Ok(accepted) => on_accepted(collector, accepted),
                    Err(message) => return ReadOutcome::Invalid(message),
                }
                deadline = Instant::now() + timeout;
            }
            Ok(ReadEvent::Invalid(message)) => return ReadOutcome::Invalid(message),
            Ok(ReadEvent::Eof) | Err(RecvTimeoutError::Disconnected) => return ReadOutcome::Eof,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

/// The start of an untrusted line, for messages.
fn excerpt(text: &str) -> String {
    let mut excerpt: String = text.chars().take(120).collect();
    if excerpt.len() < text.len() {
        excerpt.push('…');
    }
    excerpt
}

/// Reads stdout on a thread: complete lines up to the limits, then end of file. Stops reading
/// after the first invalid event; the supervisor kills the prober then.
fn spawn_stdout_reader(stdout: impl Read + Send + 'static) -> Receiver<ReadEvent> {
    let (sender, receiver) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("polypous-prober-stdout".into())
        .spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut count = 0_usize;
            loop {
                let event = match read_bounded_line(&mut reader) {
                    Ok(None) => ReadEvent::Eof,
                    Ok(Some(line)) => {
                        count += 1;
                        if count > MAX_LINES {
                            ReadEvent::Invalid(format!("more than {MAX_LINES} lines"))
                        } else {
                            ReadEvent::Line(line)
                        }
                    }
                    Err(message) => ReadEvent::Invalid(message),
                };
                let last = !matches!(event, ReadEvent::Line(_));
                if sender.send(event).is_err() || last {
                    return;
                }
            }
        });
    if spawned.is_err() {
        // Without a reader the supervisor sees a disconnected channel, i.e. end of file.
        tracing::warn!("could not start the prober's stdout reader thread");
    }
    receiver
}

/// One line without its newline (`\n` or `\r\n`), `None` at end of file.
///
/// # Errors
///
/// A line longer than [`MAX_LINE_BYTES`], not UTF-8, a partial last line, or a read error.
fn read_bounded_line(reader: &mut impl BufRead) -> Result<Option<String>, String> {
    let mut buffer = Vec::new();
    let limit = u64::try_from(MAX_LINE_BYTES + 2).unwrap_or(u64::MAX);
    let read = reader
        .by_ref()
        .take(limit)
        .read_until(b'\n', &mut buffer)
        .map_err(|error| format!("reading the prober's output failed: {error}"))?;
    if read == 0 {
        return Ok(None);
    }
    if buffer.last() != Some(&b'\n') {
        return Err(if buffer.len() > MAX_LINE_BYTES {
            format!("a line longer than {MAX_LINE_BYTES} bytes")
        } else {
            "the output ends in the middle of a line".into()
        });
    }
    buffer.pop();
    if buffer.last() == Some(&b'\r') {
        buffer.pop();
    }
    if buffer.len() > MAX_LINE_BYTES {
        return Err(format!("a line longer than {MAX_LINE_BYTES} bytes"));
    }
    String::from_utf8(buffer)
        .map(Some)
        .map_err(|_| "a line that is not UTF-8".into())
}

/// Drains stderr on a thread into the log; keeps the last lines.
fn spawn_stderr_drain(
    stderr: impl Read + Send + 'static,
    pid: u32,
) -> Arc<Mutex<VecDeque<String>>> {
    let tail = Arc::new(Mutex::new(VecDeque::new()));
    let kept = Arc::clone(&tail);
    let spawned = std::thread::Builder::new()
        .name("polypous-prober-stderr".into())
        .spawn(move || {
            let mut reader = BufReader::new(stderr);
            loop {
                let mut buffer = Vec::new();
                // Bounded per line; longer lines are split.
                let limit = u64::try_from(MAX_LINE_BYTES).unwrap_or(u64::MAX);
                match reader.by_ref().take(limit).read_until(b'\n', &mut buffer) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {
                        let line = String::from_utf8_lossy(&buffer).trim_end().to_string();
                        tracing::debug!(target: "polypous::prober", pid, "{line}");
                        let mut tail = kept.lock().unwrap_or_else(PoisonError::into_inner);
                        if tail.len() == STDERR_TAIL {
                            tail.pop_front();
                        }
                        tail.push_back(line);
                    }
                }
            }
        });
    if spawned.is_err() {
        tracing::warn!("could not start the prober's stderr reader thread");
    }
    tail
}

/// "exited with code 1" / "was killed by signal 11 (SIGSEGV)", plus the last stderr line.
fn describe_exit(status: Option<ExitStatus>, stderr_tail: &Mutex<VecDeque<String>>) -> String {
    let mut text = match status {
        None => "ended (exit status unknown)".to_string(),
        Some(status) => exit_text(status),
    };
    // Give the drain thread a moment to read what the prober wrote last.
    std::thread::sleep(Duration::from_millis(20));
    let tail = stderr_tail.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(last) = tail.iter().rev().find(|line| !line.is_empty()) {
        text.push_str(&format!("{STDERR_NOTE}{}", excerpt(last)));
    }
    text
}

fn exit_text(status: ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            let name = match signal {
                4 => " (SIGILL)",
                6 => " (SIGABRT)",
                7 => " (SIGBUS)",
                8 => " (SIGFPE)",
                9 => " (SIGKILL)",
                11 => " (SIGSEGV)",
                15 => " (SIGTERM)",
                _ => "",
            };
            return format!("was killed by signal {signal}{name}");
        }
    }
    match status.code() {
        // Windows reports crashes as NTSTATUS exit codes, for example 0xC0000005.
        Some(code) if code < 0 => format!("exited with code {code:#010x}"),
        Some(code) => format!("exited with code {code}"),
        None => format!("ended: {status}"),
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[test]
    fn bounded_lines() {
        let mut reader = io::Cursor::new(b"one\r\ntwo\n".to_vec());
        assert_eq!(read_bounded_line(&mut reader), Ok(Some("one".into())));
        assert_eq!(read_bounded_line(&mut reader), Ok(Some("two".into())));
        assert_eq!(read_bounded_line(&mut reader), Ok(None));

        let mut long = vec![b'x'; MAX_LINE_BYTES + 10];
        long.push(b'\n');
        assert!(
            read_bounded_line(&mut io::Cursor::new(long))
                .unwrap_err()
                .contains("longer than")
        );
        let mut exact = vec![b'x'; MAX_LINE_BYTES];
        exact.push(b'\n');
        assert!(read_bounded_line(&mut io::Cursor::new(exact)).is_ok());

        assert!(read_bounded_line(&mut io::Cursor::new(b"partial".to_vec())).is_err());
        assert!(read_bounded_line(&mut io::Cursor::new(b"\xff\xfe\n".to_vec())).is_err());
    }

    #[test]
    fn the_reader_stops_after_the_line_limit() {
        let text = "{}\n".repeat(MAX_LINES + 5);
        let lines = spawn_stdout_reader(io::Cursor::new(text.into_bytes()));
        let mut count = 0;
        loop {
            match lines.recv().unwrap() {
                ReadEvent::Line(_) => count += 1,
                ReadEvent::Invalid(message) => {
                    assert_eq!(message, format!("more than {MAX_LINES} lines"));
                    break;
                }
                ReadEvent::Eof => panic!("the limit was not enforced"),
            }
        }
        assert_eq!(count, MAX_LINES);
        // Nothing after the invalid event: the reader stopped.
        assert!(lines.recv().is_err());
    }

    #[test]
    fn missing_program_is_a_spawn_error() {
        let mut collector = Collector::new(ProbeRequest::default(), true);
        let result = run_once(
            &ProberCommand {
                program: PathBuf::from("/nonexistent/polypous-prober-test"),
            },
            &ProbeRequest::default(),
            Duration::from_secs(1),
            &|| false,
            &mut collector,
            &mut |_, _| {},
        );
        assert!(matches!(result, Err(StartError::Spawn(_))), "{result:?}");
    }

    #[test]
    fn excerpts_are_short() {
        assert_eq!(excerpt("abc"), "abc");
        assert_eq!(excerpt(&"y".repeat(500)).chars().count(), 121);
    }
}
