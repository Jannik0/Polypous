//! JSON values for graph documents: parsing under the limits of `docs/specs/graph-ir.md` §4.3,
//! JSON pointers, and the canonical writer of §6.
//!
//! [`Value`] keeps object members in document order and keeps duplicate keys, so that the loader can
//! report them (with a pointer) instead of silently keeping one. `meta` content is stored as a
//! [`Value`] and therefore round-trips with its key order and values unchanged.

use std::fmt::Write as _;

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

use crate::limits;

/// A JSON number as parsed: integers keep their exact value, everything else is an `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    /// A non-negative integer.
    PosInt(u64),
    /// A negative integer.
    NegInt(i64),
    /// A number with a fraction or exponent, or an integer outside the `i64`/`u64` range.
    Float(f64),
}

impl Number {
    /// The value as an `i64`, if it is an integer that fits.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Number::PosInt(v) => i64::try_from(v).ok(),
            Number::NegInt(v) => Some(v),
            Number::Float(_) => None,
        }
    }

    /// The value as an `f64` (integers are converted, possibly with rounding).
    pub fn as_f64(&self) -> f64 {
        match *self {
            // Rounding of large integers is intended: callers ask for a float.
            Number::PosInt(v) => v as f64,
            Number::NegInt(v) => v as f64,
            Number::Float(v) => v,
        }
    }

    /// Whether this number is an integer (not a float).
    pub fn is_integer(&self) -> bool {
        !matches!(self, Number::Float(_))
    }
}

/// A JSON value that keeps object members in document order, duplicates included.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`
    Null,
    /// `true` or `false`
    Bool(bool),
    /// A number.
    Number(Number),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Value>),
    /// An object, members in document order.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The JSON type name, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "a boolean",
            Value::Number(n) if n.is_integer() => "an integer",
            Value::Number(_) => "a number",
            Value::String(_) => "a string",
            Value::Array(_) => "an array",
            Value::Object(_) => "an object",
        }
    }

    /// Writes this value in the canonical layout (see [`write_value`]).
    pub fn to_canonical_string(&self) -> String {
        let mut out = String::new();
        write_value(&mut out, self, 0);
        out
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ValueVisitor)
    }
}

struct ValueVisitor;

impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(if v < 0 {
            Number::NegInt(v)
        } else {
            Number::PosInt(v.unsigned_abs())
        }))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(Number::PosInt(v)))
    }

    fn visit_f64<E>(self, v: f64) -> Result<Value, E> {
        Ok(Value::Number(Number::Float(v)))
    }

    fn visit_str<E>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        // No size hint is trusted: serde_json gives none, and the vector grows with the input,
        // which is bounded by the document size limit.
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<Value>()? {
            items.push(item);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut members = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, Value>()? {
            members.push((key, value));
        }
        Ok(Value::Object(members))
    }
}

/// Why a document could not be parsed into a [`Value`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// The document is larger than [`limits::MAX_DOCUMENT_BYTES`].
    #[error("document is {len} bytes, more than the limit of {max} bytes")]
    TooLarge {
        /// Document length in bytes.
        len: usize,
        /// The limit.
        max: usize,
    },
    /// Arrays and objects nest deeper than [`limits::MAX_JSON_DEPTH`].
    #[error("JSON nesting exceeds {max} levels at byte offset {offset}")]
    TooDeep {
        /// Byte offset of the bracket that exceeds the limit.
        offset: usize,
        /// The limit.
        max: usize,
    },
    /// Not valid UTF-8 JSON.
    #[error("invalid JSON: {0}")]
    Syntax(String),
}

/// Parses a JSON document, checking the size limit before anything is allocated and the nesting
/// limit before the parser recurses.
///
/// Object members keep document order and duplicates (see [`Value`]). String lengths are checked by
/// the caller on the parsed value, which is bounded by the document size.
pub fn parse(bytes: &[u8]) -> Result<Value, ParseError> {
    if bytes.len() > limits::MAX_DOCUMENT_BYTES {
        return Err(ParseError::TooLarge {
            len: bytes.len(),
            max: limits::MAX_DOCUMENT_BYTES,
        });
    }
    check_depth(bytes)?;
    let mut de = serde_json::Deserializer::from_slice(bytes);
    // serde_json's own limit allows only 127 levels (one less than the spec's 128). The depth was
    // checked above, so the parser's recursion is bounded by MAX_JSON_DEPTH.
    de.disable_recursion_limit();
    let value = Value::deserialize(&mut de).map_err(|e| ParseError::Syntax(e.to_string()))?;
    de.end().map_err(|e| ParseError::Syntax(e.to_string()))?;
    Ok(value)
}

/// Scans the raw bytes for the bracket nesting depth, skipping string contents. The scan does not
/// validate JSON; the parser does that afterwards.
fn check_depth(bytes: &[u8]) -> Result<(), ParseError> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' | b'{' => {
                depth += 1;
                if depth > limits::MAX_JSON_DEPTH {
                    return Err(ParseError::TooDeep {
                        offset,
                        max: limits::MAX_JSON_DEPTH,
                    });
                }
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

/// Escapes one reference token of a JSON pointer (RFC 6901): `~` becomes `~0`, `/` becomes `~1`.
pub fn escape_pointer_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

/// Appends `/token` (escaped) to a JSON pointer.
pub(crate) fn push_pointer(pointer: &str, token: &str) -> String {
    let mut p = String::with_capacity(pointer.len() + token.len() + 1);
    p.push_str(pointer);
    p.push('/');
    p.push_str(&escape_pointer_token(token));
    p
}

/// Writes a JSON string literal with escapes.
pub(crate) fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                // Infallible: writing to a String never fails.
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Writes a number. Floats use the shortest representation that parses back to the same `f64`.
pub(crate) fn write_number(out: &mut String, n: &Number) {
    match *n {
        Number::PosInt(v) => {
            let _ = write!(out, "{v}");
        }
        Number::NegInt(v) => {
            let _ = write!(out, "{v}");
        }
        Number::Float(v) => {
            if v.is_finite() {
                // `{:?}` on f64 is the shortest round-trip representation and always contains a
                // `.` or an exponent, so it parses back as a float.
                let _ = write!(out, "{v:?}");
            } else {
                // Not reachable from parsed JSON (JSON has no infinities); written as null so the
                // output stays valid JSON.
                out.push_str("null");
            }
        }
    }
}

fn is_scalar(v: &Value) -> bool {
    !matches!(v, Value::Array(_) | Value::Object(_))
}

pub(crate) fn write_indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

/// Writes `value` in the canonical layout of `docs/specs/graph-ir.md` §6:
/// - objects one member per line, indented by two spaces per level, members in stored order
/// - arrays that contain only scalars on one line (`[128, 784]`), other arrays one item per line
/// - empty arrays and objects as `[]` and `{}`
pub fn write_value(out: &mut String, value: &Value, level: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
            } else if items.iter().all(is_scalar) {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_value(out, item, level);
                }
                out.push(']');
            } else {
                out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    write_indent(out, level + 1);
                    write_value(out, item, level + 1);
                }
                out.push('\n');
                write_indent(out, level);
                out.push(']');
            }
        }
        Value::Object(members) => {
            if members.is_empty() {
                out.push_str("{}");
            } else {
                out.push_str("{\n");
                for (i, (key, item)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    write_indent(out, level + 1);
                    write_string(out, key);
                    out.push_str(": ");
                    write_value(out, item, level + 1);
                }
                out.push('\n');
                write_indent(out, level);
                out.push('}');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nested(depth: usize) -> String {
        "[".repeat(depth) + &"]".repeat(depth)
    }

    #[test]
    fn depth_limit_is_exactly_128() {
        assert!(parse(nested(128).as_bytes()).is_ok());
        assert!(matches!(
            parse(nested(129).as_bytes()),
            Err(ParseError::TooDeep { max: 128, .. })
        ));
        // Brackets inside strings don't count.
        let s = format!("[\"{}\"]", "[".repeat(1000));
        assert!(parse(s.as_bytes()).is_ok());
        // Escaped quotes don't end a string.
        let s = format!("[\"\\\"{}\"]", "{".repeat(1000));
        assert!(parse(s.as_bytes()).is_ok());
    }

    #[test]
    fn keeps_order_and_duplicates() {
        let v = parse(br#"{"b": 1, "a": 2, "b": 3}"#).unwrap();
        let Value::Object(m) = v else { panic!() };
        let keys: Vec<_> = m.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "b"]);
    }

    #[test]
    fn rejects_trailing_data_and_bad_utf8() {
        assert!(matches!(parse(b"{} {}"), Err(ParseError::Syntax(_))));
        assert!(matches!(parse(b"\"\xff\""), Err(ParseError::Syntax(_))));
        assert!(matches!(parse(b""), Err(ParseError::Syntax(_))));
    }

    #[test]
    fn numbers_round_trip() {
        for text in [
            "0",
            "-1",
            "18446744073709551615",
            "-9223372036854775808",
            "1.0",
            "1e300",
            "0.1",
            "-2.5e-8",
        ] {
            let v = parse(text.as_bytes()).unwrap();
            let written = v.to_canonical_string();
            assert_eq!(parse(written.as_bytes()).unwrap(), v, "{text} -> {written}");
        }
    }

    #[test]
    fn strings_round_trip() {
        let v = Value::String("a\"b\\c\n\u{1}\u{7f}ü😀".into());
        let written = v.to_canonical_string();
        assert_eq!(parse(written.as_bytes()).unwrap(), v);
    }

    #[test]
    fn pointer_escaping() {
        assert_eq!(push_pointer("/a", "b/c~d"), "/a/b~1c~0d");
    }
}
