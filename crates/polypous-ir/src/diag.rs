//! Validation diagnostics: the stable error codes of `docs/specs/graph-ir.md` §4.2 and the report
//! that collects them.

use std::fmt;

macro_rules! error_codes {
    ($( $(#[$doc:meta])* $variant:ident => $code:literal, )*) => {
        /// A stable validation error code (`docs/specs/graph-ir.md` §4.2).
        ///
        /// The string form ([`ErrorCode::as_str`]) is the stable identifier; new codes may be added.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[non_exhaustive]
        pub enum ErrorCode {
            $( $(#[$doc])* $variant, )*
        }

        impl ErrorCode {
            /// Every code, in the order of the spec table.
            pub const ALL: &'static [ErrorCode] = &[ $( ErrorCode::$variant, )* ];

            /// The stable `snake_case` code string.
            pub fn as_str(self) -> &'static str {
                match self {
                    $( ErrorCode::$variant => $code, )*
                }
            }

            /// Looks a code up by its string.
            pub fn from_code(code: &str) -> Option<ErrorCode> {
                match code {
                    $( $code => Some(ErrorCode::$variant), )*
                    _ => None,
                }
            }
        }
    };
}

error_codes! {
    /// Not valid UTF-8 JSON, or a field has the wrong JSON type (or a duplicate key).
    InvalidJson => "invalid_json",
    /// The document exceeds the size limit.
    DocumentTooLarge => "document_too_large",
    /// JSON nesting exceeds the depth limit.
    JsonTooDeep => "json_too_deep",
    /// A JSON string exceeds its length limit.
    StringTooLong => "string_too_long",
    /// `format` isn't `polypous.graph`, or `format_version` is unknown.
    UnsupportedFormat => "unsupported_format",
    /// An opset is unknown or its version isn't supported.
    UnsupportedOpset => "unsupported_opset",
    /// A field the schema doesn't define (outside `meta`).
    UnknownField => "unknown_field",
    /// A required field is missing.
    MissingField => "missing_field",
    /// A field has a value outside its allowed set.
    InvalidValue => "invalid_value",
    /// A name or id breaks the naming rules.
    InvalidName => "invalid_name",
    /// Two values, params, dims, state entries or outputs share a name.
    DuplicateName => "duplicate_name",
    /// A reference to an undeclared value, port, param, state entry or dim symbol.
    UnknownReference => "unknown_reference",
    /// A bare node id refers to a node without an `out` port.
    AmbiguousOutput => "ambiguous_output",
    /// The nodes form a cycle.
    Cycle => "cycle",
    /// The op isn't in the catalog of the declared opset version, or its opset isn't declared.
    UnknownOp => "unknown_op",
    /// A feature this version specifies but doesn't implement.
    UnsupportedInVersion => "unsupported_in_version",
    /// A required op input is absent.
    MissingInput => "missing_input",
    /// A required op param is absent.
    MissingParam => "missing_param",
    /// A required op attribute is absent.
    MissingAttr => "missing_attr",
    /// The op defines no input of that name.
    UnknownInput => "unknown_input",
    /// The op defines no param of that name.
    UnknownParam => "unknown_param",
    /// The op defines no attribute of that name.
    UnknownAttr => "unknown_attr",
    /// An attribute has the wrong type, is out of range, or has a wrong list length.
    InvalidAttr => "invalid_attr",
    /// An input's or param's dtype is outside the op's allowed class.
    UnsupportedDtype => "unsupported_dtype",
    /// Inputs or params that must have the same dtype don't.
    DtypeMismatch => "dtype_mismatch",
    /// An input or param has a rank the op doesn't allow.
    RankMismatch => "rank_mismatch",
    /// A declared or inferred rank exceeds 6.
    RankTooLarge => "rank_too_large",
    /// Dims that must be equal are not provably equal, or a literal bound is out of range.
    ShapeMismatch => "shape_mismatch",
    /// Two dims can't be broadcast.
    BroadcastMismatch => "broadcast_mismatch",
    /// A dim expression is malformed or exceeds the expression limits.
    InvalidDimExpr => "invalid_dim_expr",
    /// `i64` overflow while normalizing a dim expression.
    DimOverflow => "dim_overflow",
    /// A literal or literal-folded dim is ≤ 0.
    NonpositiveDim => "nonpositive_dim",
    /// A used symbol doesn't appear as a plain dim in any input or state shape.
    UnboundDim => "unbound_dim",
    /// A state update has a different dtype or shape, or updates an entry twice.
    StateUpdateMismatch => "state_update_mismatch",
    /// `carry` and `carry_out` have different keys, or a carry's dtype or shape differs.
    RepeatCarryMismatch => "repeat_carry_mismatch",
    /// `count` is out of range, `carry` is empty, or a body with state lacks `{i}` in `state_prefix`.
    InvalidRepeat => "invalid_repeat",
    /// Too many nodes (or param bindings) after expanding repeats.
    TooManyNodes => "too_many_nodes",
    /// `repeat` nesting exceeds 8 levels.
    RepeatTooDeep => "repeat_too_deep",
    /// An element count doesn't fit in `u64` or exceeds its limit.
    TooManyElements => "too_many_elements",
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The op input, param or attribute a diagnostic is about, if any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    /// An op input name.
    Input(String),
    /// An op param name.
    Param(String),
    /// An op attribute name.
    Attr(String),
}

/// One validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Stable error code.
    pub code: ErrorCode,
    /// Human-readable description.
    pub message: String,
    /// Id of the node the error is about (as written, not qualified by enclosing repeats).
    pub node_id: Option<String>,
    /// The op input, param or attribute the error is about.
    pub item: Option<Item>,
    /// JSON pointer (RFC 6901) to the offending part of the document; `""` is the whole document.
    pub pointer: String,
}

impl Diagnostic {
    /// A diagnostic without node or item.
    pub fn new(code: ErrorCode, pointer: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            code,
            message: message.into(),
            node_id: None,
            item: None,
            pointer: pointer.into(),
        }
    }

    /// Sets the node id.
    pub fn with_node(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Sets the input/param/attr item.
    pub fn with_item(mut self, item: Item) -> Self {
        self.item = Some(item);
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code)?;
        if let Some(node) = &self.node_id {
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

/// The result of validation: every error found, in the order the checks ran.
///
/// Validation problems are data, not `Err`: an empty report means the checks passed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationReport {
    /// The errors.
    pub diagnostics: Vec<Diagnostic>,
}

impl ValidationReport {
    /// Whether no error was found.
    pub fn is_ok(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Whether some diagnostic has `code`.
    pub fn has(&self, code: ErrorCode) -> bool {
        self.diagnostics.iter().any(|d| d.code == code)
    }

    pub(crate) fn push(&mut self, d: Diagnostic) {
        self.diagnostics.push(d);
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.diagnostics.is_empty() {
            return f.write_str("no errors");
        }
        for (i, d) in self.diagnostics.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{d}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_and_round_trip() {
        assert_eq!(ErrorCode::ALL.len(), 39);
        let mut seen = std::collections::HashSet::new();
        for &c in ErrorCode::ALL {
            assert!(seen.insert(c.as_str()));
            assert_eq!(ErrorCode::from_code(c.as_str()), Some(c));
        }
    }
}
