//! Resource limits for untrusted graph documents, `docs/specs/graph-ir.md` §2.1, §2.3.5 and §4.3.

/// Maximum document size in bytes (64 MiB).
pub const MAX_DOCUMENT_BYTES: usize = 64 * 1024 * 1024;
/// Maximum nesting depth of JSON arrays and objects, `meta` included.
pub const MAX_JSON_DEPTH: usize = 128;
/// Maximum number of JSON values (scalars, arrays and objects; keys don't count) in a document.
/// Bounds the parsed tree (about 32 bytes per value) to about 128 MiB.
pub const MAX_JSON_VALUES: usize = 4_000_000;
/// Maximum length in bytes of any JSON string (and object key) without a tighter limit.
pub const MAX_STRING_BYTES: usize = 64 * 1024;
/// Maximum length of node ids and graph input, output and state names.
pub const MAX_VALUE_NAME_BYTES: usize = 128;
/// Maximum length of parameter names (and of `params_prefix`/`state_prefix` as written).
pub const MAX_PARAM_NAME_BYTES: usize = 256;
/// Maximum length of a dim symbol.
pub const MAX_SYMBOL_BYTES: usize = 64;
/// Maximum length of a dim expression string.
pub const MAX_DIM_EXPR_BYTES: usize = 256;
/// Maximum nesting of parentheses and `min`/`max` calls in a dim expression.
pub const MAX_DIM_EXPR_DEPTH: usize = 16;
/// Maximum number of terms of a dim normal form.
pub const MAX_DIM_TERMS: usize = 64;
/// Maximum degree of a monomial of a dim normal form.
pub const MAX_DIM_DEGREE: usize = 8;
/// Maximum rank of any value, param or state entry (the minimum is 1).
pub const MAX_RANK: usize = 6;
/// Maximum number of nodes after expanding repeats.
pub const MAX_EXPANDED_NODES: u64 = 1_000_000;
/// Maximum number of param bindings (node param entries) after expanding repeats.
pub const MAX_EXPANDED_PARAM_BINDINGS: u64 = 4_000_000;
/// Maximum `repeat` nesting depth.
pub const MAX_REPEAT_DEPTH: usize = 8;
/// Smallest `repeat` `count`.
pub const MIN_REPEAT_COUNT: u64 = 1;
/// Largest `repeat` `count`.
pub const MAX_REPEAT_COUNT: u64 = 65_536;
/// Maximum number of declared `dims` symbols.
pub const MAX_DIMS: usize = 256;
/// Maximum number of graph inputs, of graph outputs, and of state entries (after expanding repeats).
pub const MAX_GRAPH_IO: usize = 4_096;
/// Maximum literal dim (2^32).
pub const MAX_LITERAL_DIM: i64 = 1 << 32;
