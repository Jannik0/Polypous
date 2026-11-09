//! Op names and output ports, `docs/specs/graph-ir.md` §3.1.6, for structural validation (§4.1
//! checks 3–4: references to ports, the op exists in the declared opset). The signatures are in the
//! op catalog ([`crate::catalog`]).

pub use crate::catalog::SUPPORTED_OPSETS;

/// Whether `op` (a full `opset.op_name`) is an M1 op.
pub fn is_known_op(op: &str) -> bool {
    crate::catalog::lookup(op).is_some()
}

/// The opset part of a full op name (`nn` for `nn.linear`), if it has one.
pub fn opset_of(op: &str) -> Option<&str> {
    op.split_once('.').map(|(opset, _)| opset)
}

/// The output ports of a node, as far as they can be told without the op catalog's attribute
/// checks (§3.1.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Ports {
    /// The single port `out`.
    Out,
    /// Named ports (`core.top_k`: `values`, `indices`), none of them `out`.
    Named(&'static [&'static str]),
    /// `core.split` with `n` sizes: ports `0` … `n-1`.
    Indexed(usize),
    /// `core.split` whose `sizes` attribute is malformed: the attribute error is reported by the
    /// catalog check, and any canonical decimal port is accepted here.
    AnyIndex,
    /// The op is unknown: reported as `unknown_op`; every port is accepted to avoid follow-up errors.
    Unknown,
}

impl Ports {
    /// Whether the node has port `port`.
    pub(crate) fn has(&self, port: &str) -> bool {
        match self {
            Ports::Out => port == "out",
            Ports::Named(names) => names.contains(&port),
            Ports::Indexed(n) => parse_index_port(port).is_some_and(|i| i < *n),
            Ports::AnyIndex => parse_index_port(port).is_some(),
            Ports::Unknown => true,
        }
    }

    /// Whether a bare node id (meaning `:out`) is allowed.
    pub(crate) fn has_out(&self) -> bool {
        matches!(self, Ports::Out | Ports::Unknown)
    }
}

/// Parses a decimal port index without leading zeros.
fn parse_index_port(port: &str) -> Option<usize> {
    if port.is_empty()
        || (port.len() > 1 && port.starts_with('0'))
        || !port.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    port.parse().ok()
}

/// The ports of an op node. `split_sizes` is the length of `core.split`'s `sizes` attribute if it is
/// an array.
pub(crate) fn ports_of(op: &str, split_sizes: Option<usize>) -> Ports {
    match op {
        "core.split" => match split_sizes {
            Some(n) => Ports::Indexed(n),
            None => Ports::AnyIndex,
        },
        "core.top_k" => Ports::Named(&["values", "indices"]),
        op if is_known_op(op) => Ports::Out,
        _ => Ports::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ports() {
        assert!(ports_of("nn.linear", None).has("out"));
        assert!(!ports_of("nn.linear", None).has("0"));
        let split = ports_of("core.split", Some(3));
        assert!(split.has("0") && split.has("2") && !split.has("3") && !split.has("01"));
        assert!(!split.has_out());
        assert!(ports_of("core.top_k", None).has("indices"));
        assert!(!ports_of("core.top_k", None).has_out());
        assert!(ports_of("core.split", None).has("7"));
        assert!(!ports_of("core.split", None).has("x"));
    }
}
