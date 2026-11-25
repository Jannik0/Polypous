//! `polyp train` (public-api.md §1.4, "CLI"): a new run from a graph and a training config, or a
//! resume from a checkpoint or run directory, with progress on stdout, Ctrl-C handling and an
//! optional export.
//!
//! Output: one line per event on stdout, `<event> key=value …` (see [`event_line`]), or with
//! `--json` each event's JSON (JSON Lines), `finished` last. Errors go to stderr (the error JSON
//! with `--json`), as for the other subcommands, so stdout stays valid JSON Lines when a run fails.
//!
//! Ctrl-C (SIGINT; Ctrl-C or Ctrl-Break on Windows): the first cancels the run, which finishes its
//! step, writes a checkpoint if configured and returns its outcome (exit code 4, after the
//! export); the second ends the process at once with exit code 130, without a checkpoint. The
//! handler is installed by the CLI only, never by the library (public-api.md §2 rule 9).

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU32, Ordering};

use polypous::training::{DatasetConfig, StopReason};
use polypous::{
    CancellationToken, DeviceId, Engine, EngineConfig, Graph, ResumeOptions, Trainer,
    TrainerOptions, TrainingConfig, TrainingEvent, TrainingOutcome,
};

use crate::graphs::CliError;

/// Exit code of a cancelled run (public-api.md §1.4, "CLI").
pub(crate) const EXIT_CANCELLED: u8 = 4;
/// Exit code after a second Ctrl-C: the process ends at once (128 + SIGINT, as shells report it).
pub(crate) const EXIT_INTERRUPTED: i32 = 130;

/// What to train: a new run, or a resume.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TrainSource {
    New {
        graph: PathBuf,
        config: PathBuf,
        weights: Option<PathBuf>,
    },
    Resume {
        dir: PathBuf,
        discard_newer: bool,
    },
}

/// Options of `polyp train`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TrainArgs {
    pub source: TrainSource,
    pub device: Option<DeviceId>,
    pub json: bool,
    /// `--export-graph` and `--export-weights`, both or neither.
    pub export: Option<(PathBuf, PathBuf)>,
    pub software_rasterizers: bool,
    pub probe_cache_dir: Option<PathBuf>,
    pub prober_path: Option<PathBuf>,
}

/// Parses the arguments after `train`. `Err` is a usage error.
pub(crate) fn parse(args: &[String]) -> Result<TrainArgs, String> {
    let mut graph = None;
    let mut config = None;
    let mut weights = None;
    let mut resume = None;
    let mut discard_newer = false;
    let mut device = None;
    let mut json = false;
    let mut export_graph = None;
    let mut export_weights = None;
    let mut software_rasterizers = false;
    let mut probe_cache_dir = None;
    let mut prober_path = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .cloned()
                .ok_or_else(|| format!("`{arg}` needs a value"))
        };
        match arg.as_str() {
            "--config" => config = Some(PathBuf::from(value()?)),
            "--weights" => weights = Some(PathBuf::from(value()?)),
            "--resume" => resume = Some(PathBuf::from(value()?)),
            "--discard-newer" => discard_newer = true,
            "--device" => {
                let text = value()?;
                device = Some(DeviceId::parse(&text).map_err(|error| error.message)?);
            }
            "--json" => json = true,
            "--export-graph" => export_graph = Some(PathBuf::from(value()?)),
            "--export-weights" => export_weights = Some(PathBuf::from(value()?)),
            "--software-rasterizers" => software_rasterizers = true,
            "--probe-cache-dir" => probe_cache_dir = Some(PathBuf::from(value()?)),
            "--prober-path" => prober_path = Some(PathBuf::from(value()?)),
            other if other.starts_with("--") => {
                return Err(format!("unknown argument for `train`: {other}"));
            }
            other if graph.is_none() => graph = Some(PathBuf::from(other)),
            other => return Err(format!("`train` takes one graph file, got also {other}")),
        }
    }
    let export = match (export_graph, export_weights) {
        (Some(g), Some(w)) => Some((g, w)),
        (None, None) => None,
        _ => return Err("`--export-graph` and `--export-weights` go together".into()),
    };
    let source = match resume {
        Some(dir) => {
            for (given, what) in [
                (graph.is_some(), "a graph file"),
                (config.is_some(), "`--config`"),
                (weights.is_some(), "`--weights`"),
            ] {
                if given {
                    return Err(format!(
                        "`--resume` takes no {what}: the checkpoint has the graph, the config and the weights"
                    ));
                }
            }
            TrainSource::Resume { dir, discard_newer }
        }
        None => {
            if discard_newer {
                return Err("`--discard-newer` needs `--resume`".into());
            }
            TrainSource::New {
                graph: graph.ok_or_else(|| {
                    "`train` needs a graph file and `--config FILE`, or `--resume DIR`".to_string()
                })?,
                config: config.ok_or_else(|| "`train` needs `--config FILE`".to_string())?,
                weights,
            }
        }
    };
    Ok(TrainArgs {
        source,
        device,
        json,
        export,
        software_rasterizers,
        probe_cache_dir,
        prober_path,
    })
}

/// The first `tensors` dataset of a config, as a pointer: the CLI has no in-memory data.
fn tensors_dataset(config: &TrainingConfig) -> Option<(&'static str, String)> {
    let named = |dataset: &DatasetConfig| match dataset {
        DatasetConfig::Tensors(t) => Some(t.name.clone()),
        _ => None,
    };
    if let Some(name) = named(&config.data.train) {
        return Some(("/data/train", name));
    }
    config
        .data
        .validation
        .as_ref()
        .and_then(named)
        .map(|name| ("/data/validation", name))
}

/// The directory relative config paths are resolved against: the config file's.
fn config_dir(path: &Path) -> PathBuf {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Installs the Ctrl-C handler: the first cancels `cancel`, the second ends the process.
fn install_ctrl_c(cancel: &CancellationToken) {
    let cancel = cancel.clone();
    let count = AtomicU32::new(0);
    // `ctrlc` runs this on its own thread, not in the signal handler, so it may print and exit.
    let result = ctrlc::set_handler(move || {
        if count.fetch_add(1, Ordering::SeqCst) == 0 {
            cancel.cancel();
            eprintln!(
                "Cancelling: the current step finishes and a checkpoint is written if configured. \
                 Press Ctrl-C again to stop at once."
            );
        } else {
            eprintln!("Stopped.");
            std::process::exit(EXIT_INTERRUPTED);
        }
    });
    if let Err(error) = result {
        tracing::warn!(%error, "could not install the Ctrl-C handler; Ctrl-C ends the process at once");
    }
}

/// Runs `polyp train`: prints the events, exports, and returns the exit code.
pub(crate) fn train(args: &TrainArgs) -> ExitCode {
    let mut out = std::io::stdout().lock();
    let json = args.json;
    let result = run(args, &mut |event: &TrainingEvent| {
        let line = if json {
            event.to_json()
        } else {
            event_line(event)
        };
        // A closed pipe (`polyp train … | head`) must not stop training: ignore write errors.
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    });
    drop(out);
    match result {
        Ok((outcome, exported)) => {
            if !json {
                let mut text = String::new();
                if let Some((graph, weights)) = exported {
                    let _ = writeln!(
                        text,
                        "Exported the model to {} and {}.",
                        graph.display(),
                        weights.display()
                    );
                }
                if outcome.stop_reason == StopReason::Cancelled {
                    text.push_str("Training was cancelled; continue it with `polyp train --resume <run directory>`.\n");
                }
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
            }
            match outcome.stop_reason {
                StopReason::Completed => ExitCode::SUCCESS,
                _ => ExitCode::from(EXIT_CANCELLED),
            }
        }
        Err(error) => crate::report_error(&error, json),
    }
}

type Exported = Option<(PathBuf, PathBuf)>;

fn run(
    args: &TrainArgs,
    on_event: &mut dyn FnMut(&TrainingEvent),
) -> Result<(TrainingOutcome, Exported), CliError> {
    let cancel = CancellationToken::new();
    let mut engine_config = EngineConfig::default();
    engine_config.software_rasterizers = args.software_rasterizers;
    engine_config.probe_cache_dir = args.probe_cache_dir.clone();
    engine_config.prober_path = args.prober_path.clone();

    let trainer: Trainer = match &args.source {
        TrainSource::New {
            graph,
            config,
            weights,
        } => {
            let graph_doc = Graph::from_file(graph)?;
            let mut training = TrainingConfig::from_file(config)?;
            if let Some((pointer, name)) = tensors_dataset(&training) {
                return Err(CliError::Cli {
                    kind: "invalid_input",
                    message: format!(
                        "The training config uses the in-memory dataset `{name}` (kind `tensors`), \
                         which only the API can give; use a safetensors, csv or image_folder \
                         dataset with `polyp train`."
                    ),
                    file: Some(config.display().to_string()),
                    pointer: Some(pointer.to_string()),
                });
            }
            if let Some(device) = args.device {
                training.device = device.to_string();
            }
            let mut options = TrainerOptions::default();
            options.base_dir = Some(config_dir(config));
            options.weights = weights.clone();
            let engine = Engine::new(engine_config)?;
            install_ctrl_c(&cancel);
            engine.trainer(&graph_doc, training, options, &cancel)?
        }
        TrainSource::Resume { dir, discard_newer } => {
            let mut options = ResumeOptions::default();
            options.device = args.device.map(|d| d.to_string());
            options.discard_newer = *discard_newer;
            let engine = Engine::new(engine_config)?;
            install_ctrl_c(&cancel);
            engine.resume_trainer(dir, options, &cancel)?
        }
    };
    let outcome = trainer.run(&cancel, |event| on_event(&event))?;
    let exported = match &args.export {
        Some((graph, weights)) => {
            trainer.export(graph, weights)?;
            Some((graph.clone(), weights.clone()))
        }
        None => None,
    };
    Ok((outcome, exported))
}

/// A metric or loss value: Rust's shortest round-trip form, `nan`, `inf` or `-inf`.
fn number(value: f64) -> String {
    if value.is_nan() {
        "nan".into()
    } else {
        format!("{value}")
    }
}

fn metrics_text(metrics: &polypous::training::Metrics) -> String {
    metrics
        .0
        .iter()
        .map(|(name, value)| format!(" {name}={}", number(*value)))
        .collect()
}

/// The human output of one event: its name, then `key=value` pairs (epochs and steps 0-based, as
/// in the JSON).
pub(crate) fn event_line(event: &TrainingEvent) -> String {
    match event {
        TrainingEvent::Started {
            device,
            epochs,
            steps_per_epoch,
            total_steps,
            train_samples,
            validation_samples,
            trainable_elements,
            step,
        } => format!(
            "started device={device} epochs={epochs} steps_per_epoch={steps_per_epoch} \
             total_steps={total_steps} train_samples={train_samples} \
             validation_samples={validation_samples} trainable_elements={trainable_elements} \
             step={step}"
        ),
        TrainingEvent::EpochStarted { epoch, batch } => {
            format!("epoch_started epoch={epoch} batch={batch}")
        }
        TrainingEvent::Step {
            epoch,
            batch,
            step,
            loss,
            lr,
        } => format!(
            "step epoch={epoch} batch={batch} step={step} loss={} lr={}",
            number(*loss),
            number(*lr)
        ),
        TrainingEvent::ValidationStarted { epoch } => format!("validation_started epoch={epoch}"),
        TrainingEvent::EpochFinished {
            epoch,
            step,
            metrics,
        } => format!(
            "epoch_finished epoch={epoch} step={step}{}",
            metrics_text(metrics)
        ),
        TrainingEvent::CheckpointSaved { path, step } => {
            format!("checkpoint_saved step={step} path={path}")
        }
        TrainingEvent::CheckpointRemoved { path } => format!("checkpoint_removed path={path}"),
        TrainingEvent::Finished { outcome } => {
            let stop = match outcome.stop_reason {
                StopReason::Completed => "completed",
                _ => "cancelled",
            };
            let mut line = format!(
                "finished stop_reason={stop} step={} epochs_completed={}{}",
                outcome.step,
                outcome.epochs_completed,
                metrics_text(&outcome.final_metrics)
            );
            if let Some(best) = &outcome.best {
                let _ = write!(
                    line,
                    " best={}={} best_checkpoint={}",
                    best.metric,
                    number(best.value),
                    best.checkpoint
                );
            }
            if let Some(path) = &outcome.last_checkpoint {
                let _ = write!(line, " last_checkpoint={path}");
            }
            let _ = write!(line, " elapsed_seconds={:.3}", outcome.elapsed_seconds);
            line
        }
        // A kind added later: its JSON.
        other => other.to_json(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_strs(args: &[&str]) -> Result<TrainArgs, String> {
        parse(&args.iter().map(ToString::to_string).collect::<Vec<_>>())
    }

    #[test]
    fn parses_both_forms() {
        assert_eq!(
            parse_strs(&[
                "g.json",
                "--config",
                "t.json",
                "--weights",
                "w.safetensors",
                "--device",
                "cpu",
                "--json",
                "--export-graph",
                "m.graph.json",
                "--export-weights",
                "m.safetensors",
                "--software-rasterizers",
                "--probe-cache-dir",
                "/tmp/c",
                "--prober-path",
                "/bin/p",
            ]),
            Ok(TrainArgs {
                source: TrainSource::New {
                    graph: "g.json".into(),
                    config: "t.json".into(),
                    weights: Some("w.safetensors".into()),
                },
                device: Some(DeviceId::Cpu),
                json: true,
                export: Some(("m.graph.json".into(), "m.safetensors".into())),
                software_rasterizers: true,
                probe_cache_dir: Some("/tmp/c".into()),
                prober_path: Some("/bin/p".into()),
            })
        );
        assert_eq!(
            parse_strs(&[
                "--resume",
                "runs/x",
                "--discard-newer",
                "--device",
                "vulkan:cpu:0"
            ]),
            Ok(TrainArgs {
                source: TrainSource::Resume {
                    dir: "runs/x".into(),
                    discard_newer: true,
                },
                device: Some(DeviceId::parse("vulkan:cpu:0").unwrap()),
                json: false,
                export: None,
                software_rasterizers: false,
                probe_cache_dir: None,
                prober_path: None,
            })
        );
    }

    #[test]
    fn usage_errors() {
        for (args, text) in [
            (&["--resume", "r", "g.json"][..], "graph file"),
            (&["--resume", "r", "--config", "t.json"][..], "--config"),
            (&["--resume", "r", "--weights", "w"][..], "--weights"),
            (&["g.json"][..], "--config"),
            (&["--config", "t.json"][..], "graph file"),
            (&[][..], "graph file"),
            (&["g", "--config", "t", "--discard-newer"][..], "--resume"),
            (
                &["g", "--config", "t", "--export-graph", "m"][..],
                "go together",
            ),
            (
                &["--resume", "r", "--export-weights", "m"][..],
                "go together",
            ),
            (
                &["g", "--config", "t", "--device", "gpu"][..],
                "not a device id",
            ),
            (&["g", "--config"][..], "needs a value"),
            (&["g", "--config", "t", "--yaml"][..], "--yaml"),
            (&["g", "h", "--config", "t"][..], "one graph file"),
        ] {
            let error = parse_strs(args).unwrap_err();
            assert!(error.contains(text), "{args:?}: {error}");
        }
    }

    #[test]
    fn the_config_directory_is_the_base() {
        assert_eq!(config_dir(Path::new("train.json")), PathBuf::from("."));
        assert_eq!(
            config_dir(Path::new("a/b/train.json")),
            PathBuf::from("a/b")
        );
    }

    #[test]
    fn numbers_in_lines() {
        assert_eq!(number(0.25), "0.25");
        assert_eq!(number(1e-7), "0.0000001");
        assert_eq!(number(f64::NAN), "nan");
        assert_eq!(number(f64::INFINITY), "inf");
        assert_eq!(number(f64::NEG_INFINITY), "-inf");
        let line = event_line(&TrainingEvent::Step {
            epoch: 1,
            batch: 2,
            step: 10,
            loss: 0.5,
            lr: 0.001,
        });
        assert_eq!(line, "step epoch=1 batch=2 step=10 loss=0.5 lr=0.001");
    }
}
