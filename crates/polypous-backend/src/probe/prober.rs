//! The prober side: runs a probe session and writes the protocol to stdout (ADR-0013 §1).
//!
//! Entered through `polypous::probe::run_prober_if_requested` (the app or `polyp` re-launched with
//! [`PROBER_FLAG`]) or through `polyp probe --json`. Logs go to stderr: the app's supervisor drains
//! them into its own log at `debug`.

use std::ffi::OsString;
use std::io::{self, Read, Write};

use super::protocol::{PROBER_FLAG, PROTOCOL_VERSION, ProbeRequest, ProberLine, REQUEST_FLAG};
use super::session::run_session;
use super::source::device_source;

/// Exit code of a prober whose request was invalid (wrong protocol version, bad JSON).
pub const EXIT_BAD_REQUEST: i32 = 2;
/// Exit code of a supervised prober whose parent went away (stdin closed).
pub const EXIT_PARENT_GONE: i32 = 3;
/// Exit code of a prober that could not write to stdout.
pub const EXIT_STDOUT_FAILED: i32 = 4;

/// Whether `args` (without the program name) are the arguments the supervisor starts a prober
/// with: [`PROBER_FLAG`] first.
pub fn is_prober_invocation(args: &[OsString]) -> bool {
    args.first().is_some_and(|arg| arg == PROBER_FLAG)
}

/// Reads the [`ProbeRequest`] from prober arguments: `--polypous-prober --request <json>`.
///
/// # Errors
///
/// A message for stderr if the arguments are not of that form or the JSON is invalid.
pub fn parse_prober_args(args: &[OsString]) -> Result<ProbeRequest, String> {
    match args {
        [flag, request_flag, json] if flag == PROBER_FLAG && request_flag == REQUEST_FLAG => {
            let json = json
                .to_str()
                .ok_or_else(|| "the probe request is not UTF-8".to_string())?;
            parse_request(json)
        }
        _ => Err(format!(
            "usage: <program> {PROBER_FLAG} {REQUEST_FLAG} <probe request JSON>"
        )),
    }
}

/// Parses a probe request and checks its protocol version.
///
/// # Errors
///
/// A message for stderr if `json` is not a valid request of this protocol version.
pub fn parse_request(json: &str) -> Result<ProbeRequest, String> {
    let request: ProbeRequest =
        serde_json::from_str(json).map_err(|error| format!("invalid probe request: {error}"))?;
    if request.protocol != PROTOCOL_VERSION {
        return Err(format!(
            "probe request has protocol version {}, this prober speaks {PROTOCOL_VERSION}",
            request.protocol
        ));
    }
    Ok(request)
}

/// Runs the prober for `request`: writes the protocol lines to stdout and returns the process
/// exit code (0 after the `done` line).
///
/// `supervised`: started by the app's supervisor. Then a watchdog thread exits the process as soon
/// as stdin reaches end of file, which happens when the app closes it or dies, so a hung prober
/// never outlives its app. (`polyp probe --json` from a terminal runs unsupervised.)
///
/// Installs a panic hook for the rest of the process that records panics on every thread (then
/// runs the previous hook), so a device whose worker thread panicked gets that message in its
/// `probe failed` reason (placement.md §1, "Probe failure reasons"). Call this only in a process
/// that exists to probe: the hook is process-wide.
pub fn run_prober(request: &ProbeRequest, supervised: bool) -> i32 {
    super::worker_panics::install();
    #[cfg(feature = "test-fault-injection")]
    let faults = super::faults::Faults::from_env();
    #[cfg(feature = "test-fault-injection")]
    if let Some(faults) = &faults {
        if let Some(path) = &faults.pid_file {
            use std::io::Write as _;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(file, "{}", std::process::id());
            }
        }
        if faults.exit_before_header {
            return 7;
        }
    }

    if supervised {
        // Detached on purpose: it ends with the process.
        let _watchdog = std::thread::Builder::new()
            .name("polypous-prober-watchdog".into())
            .spawn(|| {
                let mut sink = [0_u8; 256];
                let mut stdin = io::stdin().lock();
                // Read until end of file or an error: both mean the parent is gone.
                while matches!(stdin.read(&mut sink), Ok(n) if n > 0) {}
                std::process::exit(EXIT_PARENT_GONE);
            });
    }

    let source = device_source();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut stderr = io::stderr();
    let result = run_session(request, source.as_ref(), true, &mut |line| {
        if let ProberLine::Probing { id } = &line {
            // Best effort: a closed stderr must not stop the probe.
            let _ = writeln!(stderr, "polypous prober: probing {id}");
        }
        #[cfg(feature = "test-fault-injection")]
        if let Some(faults) = &faults
            && faults.replace_line(&line, &mut out)?
        {
            return Ok(());
        }
        // One line, flushed, so the app sees `probing` before the device is opened.
        out.write_all(line.to_json().as_bytes())?;
        out.write_all(b"\n")?;
        out.flush()
    });
    match result {
        Ok(()) => 0,
        Err(error) => {
            let _ = writeln!(stderr, "polypous prober: cannot write to stdout: {error}");
            EXIT_STDOUT_FAILED
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    #[test]
    fn recognizes_and_parses_prober_arguments() {
        let json = r#"{"software_rasterizers":true}"#;
        let full = args(&[PROBER_FLAG, REQUEST_FLAG, json]);
        assert!(is_prober_invocation(&full));
        let request = parse_prober_args(&full).unwrap();
        assert!(request.software_rasterizers);

        assert!(!is_prober_invocation(&args(&["devices"])));
        assert!(!is_prober_invocation(&[]));
        assert!(parse_prober_args(&args(&[PROBER_FLAG])).is_err());
        assert!(parse_prober_args(&args(&[PROBER_FLAG, REQUEST_FLAG, "{"])).is_err());
        let wrong_version = args(&[PROBER_FLAG, REQUEST_FLAG, r#"{"protocol":99}"#]);
        assert!(
            parse_prober_args(&wrong_version)
                .unwrap_err()
                .contains("protocol version 99")
        );
    }
}
