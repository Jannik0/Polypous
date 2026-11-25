//! `polyp graph validate` and `polyp run` (public-api.md §1.3, "CLI").

use std::path::{Path, PathBuf};

use polypous::{
    CancellationToken, ComputeDtype, DeviceId, Engine, EngineConfig, Error, Graph,
    GraphLoadOptions, GraphValidateOptions, NamedTensor, ValidationReport, tensor_file,
};
use serde_json::json;

/// Exit code of `graph validate` for an invalid graph (`1` is an error, `2` a usage error).
pub(crate) const EXIT_INVALID_GRAPH: u8 = 3;

/// A failed command: a facade error, or a check of the CLI itself (in the same JSON shape).
#[derive(Debug)]
pub(crate) enum CliError {
    Polypous(Error),
    Cli {
        kind: &'static str,
        message: String,
        file: Option<String>,
        /// A JSON pointer into the file (a training config).
        pointer: Option<String>,
    },
}

impl From<Error> for CliError {
    fn from(error: Error) -> Self {
        CliError::Polypous(error)
    }
}

impl CliError {
    /// The message and, if any, the context as a JSON line, for stderr.
    pub(crate) fn render(&self) -> String {
        match self {
            CliError::Polypous(error) => {
                let mut out = format!("error: {}\n", error.message);
                if let Ok(context) = serde_json::to_string(&error.context)
                    && context != "{}"
                {
                    out.push_str(&format!("context: {context}\n"));
                }
                out
            }
            CliError::Cli { message, .. } => {
                let mut out = format!("error: {message}\n");
                let context = self.cli_context();
                if context != json!({}) {
                    out.push_str(&format!("context: {context}\n"));
                }
                out
            }
        }
    }

    /// The error JSON of public-api.md §1.2.
    pub(crate) fn to_json(&self) -> String {
        match self {
            // Plain strings: serializing cannot fail.
            CliError::Polypous(error) => {
                serde_json::to_string(error).unwrap_or_else(|_| "{}".to_string())
            }
            CliError::Cli { kind, message, .. } => {
                json!({ "kind": kind, "message": message, "context": self.cli_context() })
                    .to_string()
            }
        }
    }

    /// The context of a [`CliError::Cli`]: `pointer` and `file` when set (in the facade's order).
    fn cli_context(&self) -> serde_json::Value {
        let mut context = serde_json::Map::new();
        if let CliError::Cli { file, pointer, .. } = self {
            if let Some(pointer) = pointer {
                context.insert("pointer".into(), json!(pointer));
            }
            if let Some(file) = file {
                context.insert("file".into(), json!(file));
            }
        }
        serde_json::Value::Object(context)
    }
}

/// Options of `polyp graph validate`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ValidateArgs {
    pub file: PathBuf,
    pub json: bool,
    pub compute_dtype: ComputeDtype,
}

/// Options of `polyp run`.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct RunArgs {
    pub graph: PathBuf,
    pub weights: Option<PathBuf>,
    pub inputs: Vec<(String, PathBuf)>,
    pub output: PathBuf,
    pub device: Option<DeviceId>,
    pub compute_dtype: ComputeDtype,
    pub json: bool,
    pub software_rasterizers: bool,
    pub probe_cache_dir: Option<PathBuf>,
    pub prober_path: Option<PathBuf>,
}

/// Parses `f32`, `f16` or `bf16`.
pub(crate) fn parse_compute_dtype(text: &str) -> Result<ComputeDtype, String> {
    serde_json::from_value(serde_json::Value::String(text.to_string()))
        .map_err(|_| format!("unknown compute dtype `{text}` (f32, f16, bf16)"))
}

/// The output of `graph validate` and whether the graph is valid.
pub(crate) fn validate(args: &ValidateArgs) -> Result<(String, bool), CliError> {
    let mut options = GraphValidateOptions::default();
    options.compute_dtype = args.compute_dtype;
    let report = Graph::validate_file(&args.file, &options)?;
    let text = if args.json {
        // Plain data: serializing cannot fail.
        let text = serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string());
        format!("{text}\n")
    } else {
        render_report(&args.file, &report)
    };
    Ok((text, report.valid))
}

/// `<file>: valid`, or the number of errors and one line per diagnostic.
pub(crate) fn render_report(file: &Path, report: &ValidationReport) -> String {
    let count = report.diagnostics.len();
    if count == 0 {
        return format!("{}: valid\n", file.display());
    }
    let mut out = format!(
        "{}: invalid, {count} error{}\n",
        file.display(),
        if count == 1 { "" } else { "s" }
    );
    for diagnostic in &report.diagnostics {
        // `code (node `id`) at /pointer: message`.
        out.push_str(&format!("  {diagnostic}\n"));
    }
    out
}

fn shape_text(shape: &[u64]) -> String {
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    format!("[{}]", dims.join(", "))
}

/// `polyp run`: loads the graph, reads the inputs, runs it once and writes the outputs.
pub(crate) fn run(args: &RunArgs) -> Result<String, CliError> {
    let graph = Graph::from_file(&args.graph)?;

    let mut config = EngineConfig::default();
    config.software_rasterizers = args.software_rasterizers;
    config.probe_cache_dir = args.probe_cache_dir.clone();
    config.prober_path = args.prober_path.clone();
    let engine = Engine::new(config)?;
    let mut options = GraphLoadOptions::default();
    options.weights = args.weights.clone();
    options.device = args.device;
    options.compute_dtype = args.compute_dtype;
    let model = engine.load_graph(&graph, options, &CancellationToken::new())?;
    let info = model.info();
    if !info.state.is_empty() {
        let names: Vec<&str> = info.state.iter().map(|s| s.name.as_str()).collect();
        return Err(CliError::Cli {
            kind: "unsupported_operation",
            message: format!(
                "The graph has state entries ({}); `polyp run` doesn't pass state in this version, \
                 because the initial state's shapes depend on the bound dims. Use the API \
                 (`Model::run` with `state`).",
                names.join(", ")
            ),
            file: Some(args.graph.display().to_string()),
            pointer: None,
        });
    }

    let mut inputs = Vec::with_capacity(args.inputs.len());
    for (name, path) in &args.inputs {
        let mut tensors = tensor_file::read(path)?;
        if tensors.len() != 1 {
            return Err(CliError::Cli {
                kind: "invalid_input",
                message: format!(
                    "The input file {} for `{name}` holds {} tensors; it must hold exactly one.",
                    path.display(),
                    tensors.len()
                ),
                file: Some(path.display().to_string()),
                pointer: None,
            });
        }
        // `tensors` has exactly one element.
        let tensor = tensors.remove(0).tensor;
        inputs.push(NamedTensor::new(name.clone(), tensor));
    }

    let out = model.run(inputs, vec![], &CancellationToken::new())?;
    tensor_file::write(&args.output, &out.outputs)?;

    if args.json {
        let outputs: Vec<serde_json::Value> = out
            .outputs
            .iter()
            .map(|o| json!({ "name": o.name, "dtype": o.tensor.dtype, "shape": o.tensor.shape }))
            .collect();
        let summary = json!({
            "device": info.device,
            "compute_dtype": info.compute_dtype,
            "output_file": args.output.display().to_string(),
            "outputs": outputs,
        });
        // Plain data: serializing cannot fail.
        let text = serde_json::to_string_pretty(&summary).unwrap_or_else(|_| "{}".to_string());
        return Ok(format!("{text}\n"));
    }
    let mut text = format!(
        "Ran {} on {} (compute dtype {}).\nWrote {} output{} to {}:\n",
        args.graph.display(),
        info.device,
        info.compute_dtype,
        out.outputs.len(),
        if out.outputs.len() == 1 { "" } else { "s" },
        args.output.display()
    );
    let width = out
        .outputs
        .iter()
        .map(|o| o.name.chars().count())
        .max()
        .unwrap_or(0);
    for o in &out.outputs {
        text.push_str(&format!(
            "  {:<width$}  {:<4}  {}\n",
            o.name,
            o.tensor.dtype.as_str(),
            shape_text(&o.tensor.shape)
        ));
    }
    Ok(text)
}
