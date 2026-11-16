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
//! - `polyp graph validate <file> [--json]`: every validation error of a graph document, or the
//!   `ValidationReport` JSON (public-api.md §1.3).
//! - `polyp run <graph> [--weights FILE] --input NAME=FILE … --output FILE [options]`: loads a graph
//!   with safetensors weights, runs it once on safetensors inputs and writes its outputs
//!   (public-api.md §1.3).
//!
//! Output goes to stdout; errors and logs go to stderr, logs through `tracing` (filter with
//! `RUST_LOG`, for example `RUST_LOG=polypous=debug` for probe details; the default shows
//! warnings). Exit codes: `0` success, `1` an error, `2` a usage error, `3` `graph validate` found
//! the graph invalid.

// `polypous::Error` is large (public-api.md §2 rule 7: optional context strings) and only returned
// on the error path; see the same allowance in the `polypous` crate.
#![allow(clippy::result_large_err)]

mod devices;
mod graphs;

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use graphs::{CliError, EXIT_INVALID_GRAPH, RunArgs, ValidateArgs};
use polypous::{ComputeDtype, DeviceId, Engine, EngineConfig, Error, Isolation};
use tracing_subscriber::EnvFilter;

const USAGE: &str = "\
polyp: the Polypous developer CLI

Usage:
  polyp devices [--json] [OPTIONS]   List devices, their probe status and the primary GPU
  polyp graph validate FILE [--json] [--compute-dtype D]
                                     Validate a graph document and list every error
  polyp run GRAPH [--weights FILE] --input NAME=FILE... --output FILE [OPTIONS]
                                     Run a graph once on safetensors inputs
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

Options of `graph validate`:
  --json                       Print the validation report as JSON
  --compute-dtype D            f32 (default), f16 or bf16: what `float` resolves to

Options of `run`:
  --weights FILE               Weights: a .safetensors file or a sharded *.safetensors.index.json
                               (needed if the graph declares params)
  --input NAME=FILE            Graph input NAME from a safetensors file holding exactly one tensor
                               (of any name); once per input
  --output FILE                Write every graph output, by name, to this safetensors file
  --device ID                  cpu, cuda:0, vulkan:discrete_gpu:0, ... as `polyp devices` prints
                               them (default: the primary GPU, else the CPU)
  --compute-dtype D            f32 (default), f16 or bf16
  --json                       Print a JSON summary, and errors as JSON
  --software-rasterizers       Also open software rasterizers (needed for --device vulkan:cpu:0)
  --probe-cache-dir DIR        Probe cache directory (default: the platform cache directory)
  --prober-path PATH           Prober executable (default: polyp itself)
  Graphs with state are not supported by `run` yet; use the API.

Exit codes: 0 success, 1 an error (message and context on stderr; the error JSON with --json),
2 a usage error, 3 `graph validate` found the graph invalid.

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
    GraphValidate(ValidateArgs),
    Run(RunArgs),
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
        Some("graph") => match args.get(1).map(String::as_str) {
            Some("validate") => parse_validate(&args[2..]).map(Command::GraphValidate),
            Some(other) => Err(format!("unknown command: graph {other}")),
            None => Err("`graph` needs a subcommand: validate".into()),
        },
        Some("run") => parse_run(&args[1..]).map(Command::Run),
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

fn parse_validate(args: &[String]) -> Result<ValidateArgs, String> {
    let mut file = None;
    let mut json = false;
    let mut compute_dtype = ComputeDtype::default();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--compute-dtype" => {
                let text = args
                    .next()
                    .ok_or_else(|| "`--compute-dtype` needs a value".to_string())?;
                compute_dtype = graphs::parse_compute_dtype(text)?;
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown argument for `graph validate`: {other}"));
            }
            other if file.is_none() => file = Some(PathBuf::from(other)),
            other => return Err(format!("`graph validate` takes one file, got also {other}")),
        }
    }
    Ok(ValidateArgs {
        file: file.ok_or_else(|| "`graph validate` needs a graph file".to_string())?,
        json,
        compute_dtype,
    })
}

fn parse_run(args: &[String]) -> Result<RunArgs, String> {
    let mut parsed = RunArgs::default();
    let mut graph = None;
    let mut output = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .cloned()
                .ok_or_else(|| format!("`{arg}` needs a value"))
        };
        match arg.as_str() {
            "--weights" => parsed.weights = Some(PathBuf::from(value()?)),
            "--input" => {
                let text = value()?;
                let (name, file) = text
                    .split_once('=')
                    .filter(|(name, file)| !name.is_empty() && !file.is_empty())
                    .ok_or_else(|| format!("`--input` takes NAME=FILE, got `{text}`"))?;
                if parsed.inputs.iter().any(|(known, _)| known == name) {
                    return Err(format!("input `{name}` is given twice"));
                }
                parsed.inputs.push((name.to_string(), PathBuf::from(file)));
            }
            "--output" => output = Some(PathBuf::from(value()?)),
            "--device" => {
                let text = value()?;
                parsed.device = Some(DeviceId::parse(&text).map_err(|error| error.message)?);
            }
            "--compute-dtype" => parsed.compute_dtype = graphs::parse_compute_dtype(&value()?)?,
            "--json" => parsed.json = true,
            "--software-rasterizers" => parsed.software_rasterizers = true,
            "--probe-cache-dir" => parsed.probe_cache_dir = Some(PathBuf::from(value()?)),
            "--prober-path" => parsed.prober_path = Some(PathBuf::from(value()?)),
            other if other.starts_with("--") => {
                return Err(format!("unknown argument for `run`: {other}"));
            }
            other if graph.is_none() => graph = Some(PathBuf::from(other)),
            other => return Err(format!("`run` takes one graph file, got also {other}")),
        }
    }
    parsed.graph = graph.ok_or_else(|| "`run` needs a graph file".to_string())?;
    parsed.output = output.ok_or_else(|| "`run` needs `--output FILE`".to_string())?;
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
        Command::GraphValidate(args) => {
            return match graphs::validate(&args) {
                Ok((text, valid)) => {
                    let code = write_stdout(&text);
                    if valid || code != ExitCode::SUCCESS {
                        code
                    } else {
                        ExitCode::from(EXIT_INVALID_GRAPH)
                    }
                }
                Err(error) => report_error(&error, args.json),
            };
        }
        Command::Run(args) => {
            return match graphs::run(&args) {
                Ok(text) => write_stdout(&text),
                Err(error) => report_error(&error, args.json),
            };
        }
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

/// Prints a failed command's error to stderr: the message and context, or with `json` the error
/// JSON on one line. Exit code 1.
fn report_error(error: &CliError, json: bool) -> ExitCode {
    if json {
        eprintln!("{}", error.to_json());
    } else {
        eprint!("{}", error.render());
    }
    ExitCode::FAILURE
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
        assert!(parse(&["graph"]).unwrap_err().contains("subcommand"));
        assert!(parse(&["graph", "check", "g.json"]).is_err());
        assert!(parse(&["graph", "validate"]).unwrap_err().contains("file"));
        assert!(parse(&["graph", "validate", "a", "b"]).is_err());
        assert!(parse(&["graph", "validate", "a", "--yaml"]).is_err());
        assert!(parse(&["graph", "validate", "a", "--compute-dtype", "f64"]).is_err());
        assert!(parse(&["run", "g.json"]).unwrap_err().contains("--output"));
        assert!(
            parse(&["run", "--output", "o"])
                .unwrap_err()
                .contains("graph")
        );
        for bad in ["x", "=f", "x=", ""] {
            assert!(
                parse(&["run", "g", "--output", "o", "--input", bad]).is_err(),
                "{bad}"
            );
        }
        assert!(
            parse(&[
                "run", "g", "--output", "o", "--input", "x=a", "--input", "x=b"
            ])
            .unwrap_err()
            .contains("twice")
        );
        assert!(
            parse(&["run", "g", "--output", "o", "--device", "gpu0"])
                .unwrap_err()
                .contains("not a device id")
        );
        assert!(parse(&["run", "g", "--output", "o", "--device"]).is_err());
    }

    #[test]
    fn parses_graph_commands() {
        assert_eq!(
            parse(&["graph", "validate", "g.json"]),
            Ok(Command::GraphValidate(ValidateArgs {
                file: PathBuf::from("g.json"),
                json: false,
                compute_dtype: ComputeDtype::F32,
            }))
        );
        assert_eq!(
            parse(&[
                "graph",
                "validate",
                "--json",
                "g.json",
                "--compute-dtype",
                "bf16"
            ]),
            Ok(Command::GraphValidate(ValidateArgs {
                file: PathBuf::from("g.json"),
                json: true,
                compute_dtype: ComputeDtype::Bf16,
            }))
        );
        assert_eq!(
            parse(&[
                "run",
                "g.json",
                "--weights",
                "w.safetensors",
                "--input",
                "x=x.safetensors",
                "--input",
                "mask=a=b.safetensors",
                "--output",
                "out.safetensors",
                "--device",
                "vulkan:cpu:0",
                "--compute-dtype",
                "f16",
                "--json",
                "--software-rasterizers",
                "--probe-cache-dir",
                "/tmp/c",
                "--prober-path",
                "/bin/p",
            ]),
            Ok(Command::Run(RunArgs {
                graph: PathBuf::from("g.json"),
                weights: Some(PathBuf::from("w.safetensors")),
                inputs: vec![
                    ("x".into(), PathBuf::from("x.safetensors")),
                    ("mask".into(), PathBuf::from("a=b.safetensors")),
                ],
                output: PathBuf::from("out.safetensors"),
                device: Some(DeviceId::parse("vulkan:cpu:0").unwrap()),
                compute_dtype: ComputeDtype::F16,
                json: true,
                software_rasterizers: true,
                probe_cache_dir: Some(PathBuf::from("/tmp/c")),
                prober_path: Some(PathBuf::from("/bin/p")),
            }))
        );
    }
}
