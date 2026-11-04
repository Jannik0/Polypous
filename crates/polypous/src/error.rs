//! The facade's error type (public-api.md §2 rules 7 and 8).
//!
//! Every fallible public function returns [`Error`]. Errors of the lower crates are mapped to an
//! [`ErrorKind`] here, and panics are caught at the facade boundary ([`guard`]) and become
//! [`ErrorKind::Internal`].

use std::any::Any;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::{Deserialize, Serialize};

/// What went wrong, as a stable category apps can match on (public-api.md §2 rule 7).
///
/// JSON: the variant name in `snake_case`, for example `"invalid_input"`. New variants may be
/// added, so match with a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorKind {
    /// An argument, config or request is malformed or out of range.
    InvalidInput,
    /// A graph failed validation.
    InvalidGraph,
    /// The model's architecture or file format is not supported.
    UnsupportedModel,
    /// An operation is not supported (for example on this backend).
    UnsupportedOperation,
    /// Reading or writing a file failed.
    Io,
    /// Not enough device or system memory.
    OutOfMemory,
    /// No usable device for the requested backend.
    BackendUnavailable,
    /// A backend failed while running.
    Backend,
    /// The operation was cancelled through its `CancellationToken`.
    Cancelled,
    /// A bug in Polypous (including a caught panic).
    Internal,
}

impl ErrorKind {
    /// The stable `snake_case` name, the same as in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidInput => "invalid_input",
            ErrorKind::InvalidGraph => "invalid_graph",
            ErrorKind::UnsupportedModel => "unsupported_model",
            ErrorKind::UnsupportedOperation => "unsupported_operation",
            ErrorKind::Io => "io",
            ErrorKind::OutOfMemory => "out_of_memory",
            ErrorKind::BackendUnavailable => "backend_unavailable",
            ErrorKind::Backend => "backend",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Technical details of an error, for logs and bug reports rather than for end users.
///
/// Every field is optional and omitted from JSON when absent. More fields (node id, file, tensor
/// name, device) are added as the facade grows.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ErrorContext {
    /// The facade operation that failed, for example `engine_new`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    /// Technical detail: the underlying error or the panic message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// The error of every fallible public function (public-api.md §2 rule 7).
///
/// `message` is suitable to show to end users; `context` carries the technical details.
///
/// JSON: `{"kind":"invalid_input","message":"…","context":{"operation":"…","details":"…"}}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("{message}")]
#[non_exhaustive]
pub struct Error {
    /// The category.
    pub kind: ErrorKind,
    /// A human-readable message, suitable for end users.
    pub message: String,
    /// Technical details.
    #[serde(default)]
    pub context: ErrorContext,
}

impl Error {
    /// An error of `kind` with `message` and an empty context.
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
            context: ErrorContext::default(),
        }
    }

    /// The same error with the failing operation recorded, unless one is already set.
    pub(crate) fn in_operation(mut self, operation: &str) -> Self {
        if self.context.operation.is_none() {
            self.context.operation = Some(operation.to_string());
        }
        self
    }

    /// The same error with technical details.
    pub(crate) fn with_details(mut self, details: impl Into<String>) -> Self {
        self.context.details = Some(details.into());
        self
    }

    /// The error for a panic caught at the facade boundary.
    fn from_panic(payload: &(dyn Any + Send)) -> Self {
        Error::new(
            ErrorKind::Internal,
            "Polypous ran into an internal error. Please report this as a bug.",
        )
        .with_details(format!("panic: {}", panic_message(payload)))
    }
}

/// Text of a caught panic payload.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panic with a non-text payload".to_string()
    }
}

/// Runs the body of a public facade function: a panic becomes an [`ErrorKind::Internal`] error
/// (public-api.md §2 rule 8), and every error records `operation`.
///
/// The panic hook still runs, so a caught panic is also printed to stderr (or wherever the app's
/// hook sends it).
pub(crate) fn guard<T>(
    operation: &str,
    body: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    // AssertUnwindSafe: a panicking body leaves no state behind that the caller can observe; the
    // facade builds its handles only after the body returned successfully.
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(result) => result,
        Err(payload) => {
            let error = Error::from_panic(&*payload);
            tracing::error!(
                operation,
                details = error.context.details.as_deref().unwrap_or_default(),
                "caught a panic at the API boundary"
            );
            Err(error)
        }
    }
    .map_err(|error| error.in_operation(operation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_json_shape_is_stable() {
        let error = Error::new(ErrorKind::InvalidInput, "The engine config is not valid.")
            .with_details("unknown field `foo`")
            .in_operation("engine_config_from_json");
        let json = serde_json::to_string(&error).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"invalid_input","message":"The engine config is not valid.","context":{"operation":"engine_config_from_json","details":"unknown field `foo`"}}"#
        );
        assert_eq!(serde_json::from_str::<Error>(&json).unwrap(), error);

        let bare = Error::new(ErrorKind::Internal, "x");
        assert_eq!(
            serde_json::to_string(&bare).unwrap(),
            r#"{"kind":"internal","message":"x","context":{}}"#
        );
        // `context` may be omitted when reading.
        assert_eq!(
            serde_json::from_str::<Error>(r#"{"kind":"internal","message":"x"}"#).unwrap(),
            bare
        );
    }

    #[test]
    fn error_kind_names_match_json() {
        let kinds = [
            ErrorKind::InvalidInput,
            ErrorKind::InvalidGraph,
            ErrorKind::UnsupportedModel,
            ErrorKind::UnsupportedOperation,
            ErrorKind::Io,
            ErrorKind::OutOfMemory,
            ErrorKind::BackendUnavailable,
            ErrorKind::Backend,
            ErrorKind::Cancelled,
            ErrorKind::Internal,
        ];
        for kind in kinds {
            assert_eq!(
                serde_json::to_string(&kind).unwrap(),
                format!("\"{}\"", kind.as_str())
            );
            assert_eq!(kind.to_string(), kind.as_str());
        }
    }

    #[test]
    fn display_is_the_user_message() {
        let error = Error::new(ErrorKind::Io, "Could not read the file.").with_details("ENOENT");
        assert_eq!(error.to_string(), "Could not read the file.");
    }

    #[test]
    fn panics_become_internal_errors() {
        let result: Result<(), Error> = guard("test_op", || panic!("boom {}", 42));
        let error = result.unwrap_err();
        assert_eq!(error.kind, ErrorKind::Internal);
        assert_eq!(error.context.operation.as_deref(), Some("test_op"));
        assert_eq!(error.context.details.as_deref(), Some("panic: boom 42"));

        let result: Result<(), Error> = guard("test_op", || std::panic::panic_any(7_u8));
        assert_eq!(
            result.unwrap_err().context.details.as_deref(),
            Some("panic: panic with a non-text payload")
        );
    }

    #[test]
    fn guard_passes_values_and_records_the_operation_on_errors() {
        assert_eq!(guard("op", || Ok(5)), Ok(5));
        let error =
            guard::<()>("outer", || Err(Error::new(ErrorKind::Cancelled, "stop"))).unwrap_err();
        assert_eq!(error.context.operation.as_deref(), Some("outer"));
        // An operation set further inside is kept.
        let error = guard::<()>("outer", || {
            Err(Error::new(ErrorKind::Cancelled, "stop").in_operation("inner"))
        })
        .unwrap_err();
        assert_eq!(error.context.operation.as_deref(), Some("inner"));
    }
}
