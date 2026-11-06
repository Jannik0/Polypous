//! `polyp`, the Polypous developer CLI.
//!
//! Responsibility: inspect files, validate `.polyp`, probe devices, plan placement, run pipelines,
//! benchmark.
//!
//! See `docs/architecture.md` §2. Commands so far:
//!
//! - `polyp devices [--json] [options]`: the devices Polypous found, with probe status and the
//!   primary GPU (public-api.md §1.2), and how they were probed.
//! - `polyp probe --json`: runs the device prober in this process and writes the prober protocol
//!   (placement.md §1, "Prober protocol"). `Engine::new` starts `polyp` itself as its prober
//!   (`polyp --polypous-prober …`, handled by `run_prober_if_requested` first thing in `main`).
//!
//! Output goes to stdout; logs go to stderr through `tracing` (filter with `RUST_LOG`, for example
//! `RUST_LOG=polypous=debug` for probe details; the default shows warnings).

mod devices;

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use polypous::{Engine, EngineConfig, Error, Isolation};
use tracing_subscriber::EnvFilter;

const USAGE: &str = "\
polyp: the Polypous developer CLI

Usage:
  polyp devices [--json] [OPTIONS]   List devices, their probe status and the primary GPU
  polyp probe --json [--software-rasterizers] [--request JSON]
                                     Run the device prober here and print its protocol lines
  polyp --help                       Show this help
  polyp --version                    Show the version

Options of `devices`:
  --report                     With --json: print the whole probe report, not only the devices
  --software-rasterizers       Open and probe software rasterizers (lavapipe, ...)
  --isolation MODE             auto (default), out_of_process or in_process
  --probe-cache-dir DIR        Probe cache directory (default: the platform cache directory)
  --prober-path PATH           Prober executable (default: polyp itself)
  --timeout-per-device-ms MS   Prober timeout per device (default 60000)

Logs go to stderr. Set RUST_LOG to filter them, for example RUST_LOG=polypous=debug.
";

/// Options of `polyp devices`.
#[derive(Debug, Default, PartialEq, Eq)]
struct DevicesArgs {
    json: bool,
    report: bool,
    software_rasterizers: bool,
    isolation: Option<Isolation>,
    probe_cache_dir: Option<PathBuf>,
    prober_path: Option<PathBuf>,
    timeout_per_device_ms: Option<u64>,
}

/// A parsed command line.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    Devices(DevicesArgs),
    Probe { request: String },
    Help,
    Version,
}

/// Parses the arguments after the program name. `Err` carries a message for stderr.
fn parse_args(args: &[String]) -> Result<Command, String> {
    match args.first().map(String::as_str) {
        None | Some("-h" | "--help" | "help") => Ok(Command::Help),
        Some("-V" | "--version") => Ok(Command::Version),
        Some("devices") => parse_devices(&args[1..]).map(Command::Devices),
        Some("probe") => parse_probe(&args[1..]),
        Some(other) => Err(format!("unknown command: {other}")),
    }
}

fn parse_devices(args: &[String]) -> Result<DevicesArgs, String> {
    let mut parsed = DevicesArgs::default();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .cloned()
                .ok_or_else(|| format!("`{arg}` needs a value"))
        };
        match arg.as_str() {
            "--json" => parsed.json = true,
            "--report" => parsed.report = true,
            "--software-rasterizers" => parsed.software_rasterizers = true,
            "--isolation" => {
                let mode = value()?;
                parsed.isolation = Some(
                    serde_json::from_value(serde_json::Value::String(mode.clone())).map_err(
                        |_| {
                            format!("unknown isolation `{mode}` (auto, out_of_process, in_process)")
                        },
                    )?,
                );
            }
            "--probe-cache-dir" => parsed.probe_cache_dir = Some(PathBuf::from(value()?)),
            "--prober-path" => parsed.prober_path = Some(PathBuf::from(value()?)),
            "--timeout-per-device-ms" => {
                let text = value()?;
                parsed.timeout_per_device_ms = Some(
                    text.parse()
                        .map_err(|_| format!("not a number of milliseconds: {text}"))?,
                );
            }
            other => return Err(format!("unknown argument for `devices`: {other}")),
        }
    }
    if parsed.report && !parsed.json {
        return Err("`--report` needs `--json`".into());
    }
    Ok(parsed)
}

fn parse_probe(args: &[String]) -> Result<Command, String> {
    let mut json = false;
    let mut software_rasterizers = false;
    let mut request = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--software-rasterizers" => software_rasterizers = true,
            "--request" => {
                request = Some(
                    args.next()
                        .cloned()
                        .ok_or_else(|| "`--request` needs a value".to_string())?,
                );
            }
            other => return Err(format!("unknown argument for `probe`: {other}")),
        }
    }
    if !json {
        return Err("`probe` writes JSON lines only: pass `--json`".into());
    }
    let request = request.unwrap_or_else(|| {
        serde_json::json!({ "software_rasterizers": software_rasterizers }).to_string()
    });
    Ok(Command::Probe { request })
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .init();
    // When `Engine::new` (in another `polyp` process) starts this one as its prober, this runs the
    // probe and exits. Otherwise it records that `polyp` can be its own prober.
    polypous::probe::run_prober_if_requested();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = match parse_args(&args) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    let output = match command {
        Command::Help => Ok(USAGE.to_string()),
        Command::Version => Ok(format!("polyp {}\n", env!("CARGO_PKG_VERSION"))),
        Command::Devices(args) => devices(&args),
        Command::Probe { request } => {
            let code = polypous::probe::run_prober(&request);
            return ExitCode::from(u8::try_from(code).unwrap_or(1));
        }
    };
    match output {
        Ok(text) => write_stdout(&text),
        Err(error) => {
            eprintln!("error: {error}");
            if let Ok(json) = serde_json::to_string(&error.context) {
                eprintln!("context: {json}");
            }
            ExitCode::FAILURE
        }
    }
}

/// `polyp devices [--json] [options]`.
fn devices(args: &DevicesArgs) -> Result<String, Error> {
    let mut config = EngineConfig::default();
    config.software_rasterizers = args.software_rasterizers;
    if let Some(isolation) = args.isolation {
        config.probe_isolation = isolation;
    }
    config.probe_cache_dir = args.probe_cache_dir.clone();
    config.prober_path = args.prober_path.clone();
    if let Some(timeout) = args.timeout_per_device_ms {
        config.probe_timeout_per_device_ms = timeout;
    }
    let engine = Engine::new(config)?;
    let devices = engine.devices();
    // Plain serde data; serializing it cannot fail. Pretty, with a final newline.
    if args.report {
        let text = serde_json::to_string_pretty(&engine.probe_report())
            .unwrap_or_else(|_| "{}".to_string());
        Ok(format!("{text}\n"))
    } else if args.json {
        let text = serde_json::to_string_pretty(&devices).unwrap_or_else(|_| "[]".to_string());
        Ok(format!("{text}\n"))
    } else {
        Ok(format!(
            "{}{}",
            devices::render_table(&devices),
            devices::render_probe_summary(&engine.probe_report())
        ))
    }
}

/// Writes the command's output. A closed pipe (`polyp devices | head`) is not an error.
fn write_stdout(text: &str) -> ExitCode {
    let mut stdout = std::io::stdout().lock();
    match stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.flush())
    {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: could not write the output: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Command, String> {
        parse_args(&args.iter().map(ToString::to_string).collect::<Vec<_>>())
    }

    #[test]
    fn parses_commands() {
        assert_eq!(parse(&[]), Ok(Command::Help));
        assert_eq!(parse(&["--help"]), Ok(Command::Help));
        assert_eq!(parse(&["-V"]), Ok(Command::Version));
        assert_eq!(
            parse(&["devices"]),
            Ok(Command::Devices(DevicesArgs::default()))
        );
        assert_eq!(
            parse(&["devices", "--json"]),
            Ok(Command::Devices(DevicesArgs {
                json: true,
                ..DevicesArgs::default()
            }))
        );
        assert_eq!(
            parse(&[
                "devices",
                "--json",
                "--report",
                "--software-rasterizers",
                "--isolation",
                "in_process",
                "--probe-cache-dir",
                "/tmp/c",
                "--prober-path",
                "/bin/p",
                "--timeout-per-device-ms",
                "500",
            ]),
            Ok(Command::Devices(DevicesArgs {
                json: true,
                report: true,
                software_rasterizers: true,
                isolation: Some(Isolation::InProcess),
                probe_cache_dir: Some(PathBuf::from("/tmp/c")),
                prober_path: Some(PathBuf::from("/bin/p")),
                timeout_per_device_ms: Some(500),
            }))
        );
        assert_eq!(
            parse(&["probe", "--json"]),
            Ok(Command::Probe {
                request: r#"{"software_rasterizers":false}"#.into()
            })
        );
        assert_eq!(
            parse(&["probe", "--json", "--request", "{}"]),
            Ok(Command::Probe {
                request: "{}".into()
            })
        );
    }

    #[test]
    fn rejects_unknown_arguments() {
        assert!(parse(&["explode"]).unwrap_err().contains("unknown command"));
        assert!(
            parse(&["devices", "--yaml"])
                .unwrap_err()
                .contains("--yaml")
        );
        assert!(parse(&["devices", "--isolation", "sandbox"]).is_err());
        assert!(parse(&["devices", "--isolation"]).is_err());
        assert!(parse(&["devices", "--report"]).is_err());
        assert!(parse(&["devices", "--timeout-per-device-ms", "x"]).is_err());
        assert!(parse(&["probe"]).unwrap_err().contains("--json"));
    }
}
