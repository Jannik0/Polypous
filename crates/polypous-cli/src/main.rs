//! `polyp`, the Polypous developer CLI.
//!
//! Responsibility: inspect files, validate `.polyp`, probe devices, plan placement, run pipelines,
//! benchmark.
//!
//! See `docs/architecture.md` §2. Commands so far:
//!
//! - `polyp devices [--json]`: the devices Polypous found, with probe status and the primary GPU
//!   (public-api.md §1.2).
//!
//! Output goes to stdout; logs go to stderr through `tracing` (filter with `RUST_LOG`, for example
//! `RUST_LOG=polypous=debug` for probe details; the default shows warnings).

mod devices;

use std::io::{IsTerminal, Write};
use std::process::ExitCode;

use polypous::{Engine, EngineConfig, Error};
use tracing_subscriber::EnvFilter;

const USAGE: &str = "\
polyp: the Polypous developer CLI

Usage:
  polyp devices [--json]   List devices, their probe status and the primary GPU
  polyp --help             Show this help
  polyp --version          Show the version

Logs go to stderr. Set RUST_LOG to filter them, for example RUST_LOG=polypous=debug.
";

/// A parsed command line.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    Devices { json: bool },
    Help,
    Version,
}

/// Parses the arguments after the program name. `Err` carries a message for stderr.
fn parse_args(args: &[String]) -> Result<Command, String> {
    match args.first().map(String::as_str) {
        None | Some("-h" | "--help" | "help") => Ok(Command::Help),
        Some("-V" | "--version") => Ok(Command::Version),
        Some("devices") => {
            let mut json = false;
            for arg in &args[1..] {
                match arg.as_str() {
                    "--json" => json = true,
                    other => return Err(format!("unknown argument for `devices`: {other}")),
                }
            }
            Ok(Command::Devices { json })
        }
        Some(other) => Err(format!("unknown command: {other}")),
    }
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .init();

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
        Command::Devices { json } => devices(json),
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

/// `polyp devices [--json]`.
fn devices(json: bool) -> Result<String, Error> {
    let engine = Engine::new(EngineConfig::default())?;
    let devices = engine.devices();
    if json {
        // Plain serde data; serializing it cannot fail. Pretty, with a final newline.
        let text = serde_json::to_string_pretty(&devices).unwrap_or_else(|_| "[]".to_string());
        Ok(format!("{text}\n"))
    } else {
        Ok(devices::render_table(&devices))
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
        assert_eq!(parse(&["devices"]), Ok(Command::Devices { json: false }));
        assert_eq!(
            parse(&["devices", "--json"]),
            Ok(Command::Devices { json: true })
        );
    }

    #[test]
    fn rejects_unknown_arguments() {
        assert!(parse(&["probe"]).unwrap_err().contains("unknown command"));
        assert!(
            parse(&["devices", "--yaml"])
                .unwrap_err()
                .contains("--yaml")
        );
    }
}
