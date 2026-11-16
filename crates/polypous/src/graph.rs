//! [`Graph`]: loading, saving and validating graph documents (public-api.md §1.3, graph-ir.md §4).

use std::fmt;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use polypous_ir::limits::MAX_DOCUMENT_BYTES;
use serde::{Deserialize, Serialize};

use crate::engine::object_from_json;
use crate::error::{Error, ErrorKind, guard};
use crate::tensor::ComputeDtype;

/// Options of graph validation (public-api.md §1.3).
///
/// JSON: an object, `{}` for the defaults, for example `{"compute_dtype":"f16"}`. Unknown fields
/// are an [`ErrorKind::InvalidInput`] error. The struct is `#[non_exhaustive]`: start from
/// `GraphValidateOptions::default()` and set fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct GraphValidateOptions {
    /// The compute dtype `float` resolves to (graph-ir.md §2.2). Default `f32`.
    pub compute_dtype: ComputeDtype,
}

impl GraphValidateOptions {
    /// Reads options from their JSON form (public-api.md §2 rule 3).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] if `json` is not a JSON object of known fields.
    pub fn from_json(json: &str) -> Result<GraphValidateOptions, Error> {
        guard("graph_validate_options_from_json", || {
            object_from_json(json, "The graph validation options are not valid.")
        })
    }

    /// The options' JSON form.
    pub fn to_json(&self) -> String {
        // Plain fields: serializing cannot fail.
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    fn to_ir(&self) -> polypous_ir::ValidateOptions {
        polypous_ir::ValidateOptions {
            compute_dtype: self.compute_dtype.to_ir(),
        }
    }
}

/// One validation error (graph-ir.md §4.2).
///
/// JSON: `{"code":"unknown_reference","node":"fc1","pointer":"/nodes/0/inputs/x","message":"…"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Diagnostic {
    /// The stable `snake_case` error code of graph-ir.md §4.2, for example `unknown_reference`.
    pub code: String,
    /// The id of the node the error is about, as written (not qualified by enclosing repeats).
    pub node: Option<String>,
    /// JSON pointer (RFC 6901) into the document; `""` is the whole document.
    pub pointer: String,
    /// A human-readable description.
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code)?;
        if let Some(node) = &self.node {
            write!(f, " (node `{node}`)")?;
        }
        let pointer = if self.pointer.is_empty() {
            "/"
        } else {
            &self.pointer
        };
        write!(f, " at {pointer}: {}", self.message)
    }
}

/// The result of validating a graph: every error, in the order the checks ran.
///
/// JSON: `{"valid":false,"diagnostics":[…]}` (public-api.md §1.3); `valid` is `true` exactly when
/// `diagnostics` is empty.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ValidationReport {
    /// Whether the graph is valid (no diagnostics).
    pub valid: bool,
    /// The errors.
    pub diagnostics: Vec<Diagnostic>,
}

impl ValidationReport {
    pub(crate) fn from_ir(report: &polypous_ir::ValidationReport) -> ValidationReport {
        let diagnostics: Vec<Diagnostic> = report
            .diagnostics
            .iter()
            .map(|d| Diagnostic {
                code: d.code.as_str().to_string(),
                node: d.node_id.clone(),
                pointer: d.pointer.clone(),
                message: d.message.clone(),
            })
            .collect();
        ValidationReport {
            valid: diagnostics.is_empty(),
            diagnostics,
        }
    }

    /// The report's JSON form.
    pub fn to_json(&self) -> String {
        // Plain strings and a bool: serializing cannot fail.
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// The `InvalidGraph` error for a report with errors (public-api.md §1.3, "Invalid graphs").
pub(crate) fn invalid_graph_error(report: &ValidationReport) -> Error {
    let Some(first) = report.diagnostics.first() else {
        return Error::new(ErrorKind::InvalidGraph, "The graph is not valid.");
    };
    let more = report.diagnostics.len() - 1;
    let message = match more {
        0 => format!("The graph is not valid: {}", first.message),
        1 => format!(
            "The graph is not valid: {} (and 1 more error)",
            first.message
        ),
        n => format!(
            "The graph is not valid: {} (and {n} more errors)",
            first.message
        ),
    };
    let details: Vec<String> = report.diagnostics.iter().map(ToString::to_string).collect();
    let mut error = Error::new(ErrorKind::InvalidGraph, message)
        .with_details(details.join("\n"))
        .with_pointer(first.pointer.clone());
    if let Some(node) = &first.node {
        error = error.with_node(node.clone());
    }
    error
}

/// Reads a graph document, at most one byte more than the document limit (graph-ir.md §4.3), so a
/// huge file is never read whole; the IR reports `document_too_large` for it.
fn read_document(path: &Path) -> Result<Vec<u8>, Error> {
    let io_error = |error: std::io::Error| {
        Error::new(
            ErrorKind::Io,
            format!("Could not read the graph file {}: {error}.", path.display()),
        )
        .with_details(error.to_string())
        .with_file(path)
    };
    let file = std::fs::File::open(path).map_err(io_error)?;
    let mut bytes = Vec::new();
    file.take(MAX_DOCUMENT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    Ok(bytes)
}

/// A loaded, valid graph document (graph-ir.md).
///
/// A cheap-to-clone, immutable handle (`Arc` inside), `Send + Sync + 'static`. Load it with
/// [`Graph::from_json`] or [`Graph::from_file`], run it with
/// [`Engine::load_graph`](crate::Engine::load_graph).
#[derive(Clone)]
pub struct Graph {
    inner: Arc<polypous_ir::Graph>,
}

impl Graph {
    fn from_bytes(bytes: &[u8]) -> Result<Graph, Error> {
        let outcome = polypous_ir::load(bytes, &polypous_ir::ValidateOptions::default());
        match outcome.graph {
            Some(graph) if outcome.report.is_ok() => Ok(Graph {
                inner: Arc::new(graph),
            }),
            _ => Err(invalid_graph_error(&ValidationReport::from_ir(
                &outcome.report,
            ))),
        }
    }

    /// Loads and validates a graph document (graph-ir.md §4, compute dtype `f32`).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidGraph`] if the document doesn't parse or has any validation error;
    /// `context` holds the first diagnostic's node id and JSON pointer, and every diagnostic in
    /// `details`. [`Graph::validate_json`] returns them as data.
    pub fn from_json(json: &str) -> Result<Graph, Error> {
        guard("graph_from_json", || Graph::from_bytes(json.as_bytes()))
    }

    /// Loads and validates the graph document in the file at `path`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Io`] if the file can't be read; otherwise as [`Graph::from_json`], with the
    /// file in `context.file`. A file larger than 64 MiB is `document_too_large`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Graph, Error> {
        let path = path.as_ref();
        guard("graph_from_file", || {
            let bytes = read_document(path)?;
            Graph::from_bytes(&bytes).map_err(|error| error.with_file(path))
        })
    }

    /// The graph's canonical JSON form (graph-ir.md §6).
    pub fn to_json(&self) -> String {
        self.inner.to_json()
    }

    /// Validates the graph with `options`, for example another compute dtype. The pointers point
    /// into the canonical form ([`Graph::to_json`]).
    pub fn validate(&self, options: &GraphValidateOptions) -> ValidationReport {
        let graph = self.inner.clone();
        let options = options.to_ir();
        guard("graph_validate", || {
            Ok(ValidationReport::from_ir(&graph.validate(&options)))
        })
        .unwrap_or_else(|error| internal_report(&error))
    }

    /// Every validation error of a document that may not load (editors, `polyp graph validate`).
    pub fn validate_json(json: &str, options: &GraphValidateOptions) -> ValidationReport {
        let options = options.to_ir();
        guard("graph_validate_json", || {
            Ok(ValidationReport::from_ir(
                &polypous_ir::load(json.as_bytes(), &options).report,
            ))
        })
        .unwrap_or_else(|error| internal_report(&error))
    }

    /// Every validation error of the document in the file at `path`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Io`] if the file can't be read.
    pub fn validate_file(
        path: impl AsRef<Path>,
        options: &GraphValidateOptions,
    ) -> Result<ValidationReport, Error> {
        let path = path.as_ref();
        let options = options.to_ir();
        guard("graph_validate_file", || {
            let bytes = read_document(path)?;
            Ok(ValidationReport::from_ir(
                &polypous_ir::load(&bytes, &options).report,
            ))
        })
    }

    /// The IR graph.
    pub(crate) fn ir(&self) -> &polypous_ir::Graph {
        &self.inner
    }
}

/// A report for a validation that panicked (a bug): one diagnostic, so the graph never looks
/// valid.
fn internal_report(error: &Error) -> ValidationReport {
    ValidationReport {
        valid: false,
        diagnostics: vec![Diagnostic {
            code: "internal_error".to_string(),
            node: None,
            pointer: String::new(),
            message: format!(
                "{} ({})",
                error.message,
                error.context.details.as_deref().unwrap_or_default()
            ),
        }],
    }
}

impl PartialEq for Graph {
    fn eq(&self, other: &Graph) -> bool {
        self.inner == other.inner
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("name", &self.inner.name)
            .field("inputs", &self.inner.inputs.len())
            .field("outputs", &self.inner.outputs.len())
            .field("params", &self.inner.params.len())
            .field("nodes", &self.inner.nodes.len())
            .finish_non_exhaustive()
    }
}
