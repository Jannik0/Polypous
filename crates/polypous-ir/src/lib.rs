//! Graph IR for Polypous.
//!
//! Responsibility: graph IR data model, op catalog (opset), validation, symbolic shape inference,
//! JSON (de)serialization, versioning/migration.
//!
//! Pure Rust, with no GPU or Burn dependency. See `docs/architecture.md` §2 and
//! `docs/specs/graph-ir.md`.
//!
//! - [`model`]: the typed graph document (§2), [`Graph`].
//! - [`load`](mod@load): loading untrusted JSON under the limits of §4.3, collecting every error with its JSON
//!   pointer, then validating ([`load::load`], [`Graph::from_json`]).
//! - [`validate`]: structural validation (§4.1 checks 2–5, 9 and the structural parts of 8), then
//!   op signatures, dtype and shape inference (checks 4, 6–8; `infer.rs`).
//! - [`catalog`]: the M1 op catalog (§3.2, §3.3).
//! - [`analysis`]: what validation proves, for the runtime ([`Graph::analyze`]), and the symbol
//!   binding of §2.3.4 ([`Analysis::bind`]).
//! - [`save`]: the canonical JSON form (§6), [`Graph::to_json`].
//! - [`dim`]: dim expressions, their normal form and equality (§2.3).
//! - [`diag`]: error codes (§4.2) and the [`ValidationReport`].
//!
//! Validation problems are data: a [`ValidationReport`] with a list of [`Diagnostic`]s, not `Err`.
//!
//! ```
//! use polypous_ir::{Graph, ValidateOptions};
//!
//! let json = br#"{
//!   "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
//!   "dims": { "batch": {} },
//!   "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
//!   "outputs": [ { "name": "y", "value": "act" } ],
//!   "nodes": [ { "id": "act", "op": "nn.relu", "inputs": { "x": "x" } } ]
//! }"#;
//! let graph = Graph::from_json(json).expect("valid graph");
//! assert!(graph.validate(&ValidateOptions::default()).is_ok());
//! let saved = graph.to_json();
//! assert_eq!(Graph::from_json(saved.as_bytes()).unwrap(), graph);
//! ```

pub mod analysis;
pub mod catalog;
pub mod diag;
pub mod dim;
pub mod dtype;
pub(crate) mod infer;
pub mod json;
pub mod limits;
pub mod load;
pub mod model;
pub mod names;
pub mod ops;
pub mod save;
pub mod validate;

pub use analysis::{
    Analysis, AttrValue, Attrs, BindError, Bindings, BoundDim, BoundNode, BoundOp, BoundRepeat,
    DimCheck, NamedValue, NodeInfo, OpInfo, RepeatInfo, Scalar, SymbolInfo, TensorDesc, ValueInfo,
    ValueKind,
};
pub use catalog::OpDef;
pub use diag::{Diagnostic, ErrorCode, Item, ValidationReport};
pub use dim::{DimError, DimExpr};
pub use dtype::{ComputeDtype, Dtype, DtypeSpec};
pub use load::{LoadOutcome, load};
pub use model::Graph;
pub use validate::ValidateOptions;

impl Graph {
    /// Loads a graph document and validates it with the default options (compute dtype `f32`).
    ///
    /// Returns the graph only if it has no errors at all; otherwise the report lists every error.
    /// Use [`load()`] to get a graph that converted but has validation errors (for editors).
    pub fn from_json(bytes: &[u8]) -> Result<Graph, ValidationReport> {
        let outcome = load(bytes, &ValidateOptions::default());
        match outcome.graph {
            Some(graph) if outcome.report.is_ok() => Ok(graph),
            _ => Err(outcome.report),
        }
    }

    /// Validates the graph (§4.1: structure, op signatures, dtypes and shapes; see [`validate`]).
    ///
    /// The graph is checked as its canonical JSON ([`Graph::to_json`]) would be loaded, so the JSON
    /// limits of §4.3 (document size, nesting depth, string lengths, in `meta` too) apply as well,
    /// and the JSON pointers of the diagnostics point into the canonical form.
    pub fn validate(&self, options: &ValidateOptions) -> ValidationReport {
        load(self.to_json().as_bytes(), options).report
    }

    /// Writes the graph in the canonical JSON form (§6).
    pub fn to_json(&self) -> String {
        save::to_json(self)
    }
}
