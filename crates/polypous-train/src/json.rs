//! A strict JSON reader and the number rules of training configs and checkpoint state
//! (`docs/specs/training.md` §2.1, §4.1, §5.2).
//!
//! Serde's derives report an unknown field or a wrong type without saying where it is, but
//! training.md §2.1 requires the JSON pointer (RFC 6901) of the offending value in every error. So
//! a document is first parsed into a [`Json`] tree that keeps every object member in document
//! order (a key given twice included), and the config and state readers walk that tree with
//! [`Obj`], which knows the pointer of every value it hands out.
//!
//! Numbers are parsed by `serde_json` with its `float_roundtrip` feature, which reads a decimal
//! number as the nearest `f64` (§2.1); the default parser can be one ulp off. An integer field
//! takes only a JSON number without fraction or exponent that fits `u64` (`5`, not `5.0`).

use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, Serializer};

/// A parsed JSON value. Objects keep their members in document order, duplicates included, so
/// the readers can report a key given twice with its pointer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    /// A non-negative integer that fits `u64`.
    U64(u64),
    /// A negative integer that fits `i64`.
    I64(i64),
    /// Any other number: with a fraction or exponent, or an integer outside both ranges.
    F64(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// The JSON type, for error messages.
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::U64(_) | Json::I64(_) | Json::F64(_) => "a number",
            Json::Str(_) => "a string",
            Json::Arr(_) => "an array",
            Json::Obj(_) => "an object",
        }
    }

    /// The nesting depth: 0 for a scalar, 1 for an object or array of scalars.
    pub(crate) fn depth(&self) -> usize {
        match self {
            Json::Arr(items) => 1 + items.iter().map(Json::depth).max().unwrap_or(0),
            Json::Obj(members) => 1 + members.iter().map(|(_, v)| v.depth()).max().unwrap_or(0),
            _ => 0,
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
        Ok(Json::Bool(v))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
        Ok(Json::U64(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
        Ok(if v >= 0 {
            Json::U64(v.unsigned_abs())
        } else {
            Json::I64(v)
        })
    }

    fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
        Ok(Json::F64(v))
    }

    fn visit_str<E>(self, v: &str) -> Result<Json, E> {
        Ok(Json::Str(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> Result<Json, E> {
        Ok(Json::Str(v))
    }

    fn visit_unit<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_none<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Json, D::Error> {
        Json::deserialize(d)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<Json>()? {
            items.push(item);
        }
        Ok(Json::Arr(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut members = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            let value = map.next_value::<Json>()?;
            members.push((key, value));
        }
        Ok(Json::Obj(members))
    }
}

/// Parses `text` into a [`Json`] tree. `serde_json` limits the nesting to 128 levels, so the
/// recursion here is bounded; callers check their own, lower depth limit afterwards.
pub(crate) fn parse(text: &str) -> Result<Json, String> {
    serde_json::from_str::<Json>(text).map_err(|e| e.to_string())
}

/// A failed read: the pointer of the offending value and what is wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReadError {
    pub pointer: String,
    pub message: String,
}

pub(crate) type Read<T> = Result<T, ReadError>;

pub(crate) fn error<T>(pointer: &str, message: impl Into<String>) -> Read<T> {
    Err(ReadError {
        pointer: pointer.to_owned(),
        message: message.into(),
    })
}

/// Escapes one reference token of a JSON pointer (RFC 6901 §3: `~` → `~0`, `/` → `~1`).
pub(crate) fn token(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

/// The pointer of member `key` of the value at `at`.
pub(crate) fn child(at: &str, key: &str) -> String {
    format!("{at}/{}", token(key))
}

/// An object being read: every member must be known ([`Obj::allow`]), each key appears once.
pub(crate) struct Obj<'a> {
    at: String,
    members: &'a [(String, Json)],
}

impl<'a> Obj<'a> {
    /// The object at pointer `at`. `what` names it in the message if it isn't an object (`the
    /// config`, `` `data` ``). A key given twice is an error at its second occurrence.
    pub(crate) fn new(value: &'a Json, at: &str, what: &str) -> Read<Self> {
        let Json::Obj(members) = value else {
            return error(
                at,
                format!("{what} must be an object, got {}", value.type_name()),
            );
        };
        for (i, (key, _)) in members.iter().enumerate() {
            if members[..i].iter().any(|(k, _)| k == key) {
                return error(&child(at, key), format!("field `{key}` is given twice"));
            }
        }
        Ok(Self {
            at: at.to_owned(),
            members,
        })
    }

    /// The members, in document order.
    pub(crate) fn members(&self) -> &'a [(String, Json)] {
        self.members
    }

    /// The pointer of member `key`.
    pub(crate) fn ptr(&self, key: &str) -> String {
        child(&self.at, key)
    }

    /// Rejects the first member (in document order) whose key isn't in `allowed`. `what` names
    /// the object (`` `optimizer` of kind `adam` ``) in the message, which lists the allowed fields.
    pub(crate) fn allow(&self, allowed: &[&str], what: &str) -> Read<()> {
        if let Some((key, _)) = self
            .members
            .iter()
            .find(|(k, _)| !allowed.contains(&k.as_str()))
        {
            return error(
                &self.ptr(key),
                format!(
                    "unknown field `{key}` in {what}; the allowed fields are {}",
                    allowed
                        .iter()
                        .map(|f| format!("`{f}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        }
        Ok(())
    }

    /// Member `key`, if present.
    pub(crate) fn get(&self, key: &str) -> Option<&'a Json> {
        self.members.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Member `key`, which must be present.
    pub(crate) fn req(&self, key: &str) -> Read<&'a Json> {
        self.get(key).map_or_else(
            || error(&self.ptr(key), format!("missing required field `{key}`")),
            Ok,
        )
    }

    /// A required string.
    pub(crate) fn req_str(&self, key: &str) -> Read<&'a str> {
        as_str(self.req(key)?, &self.ptr(key), key)
    }

    /// A required integer.
    pub(crate) fn req_u64(&self, key: &str) -> Read<u64> {
        as_u64(self.req(key)?, &self.ptr(key), key)
    }

    /// A required number.
    pub(crate) fn req_f64(&self, key: &str) -> Read<f64> {
        as_f64(self.req(key)?, &self.ptr(key), key)
    }

    /// An optional string (absent → `default`; `null` is a type error).
    pub(crate) fn opt_str(&self, key: &str, default: &str) -> Read<String> {
        self.get(key).map_or_else(
            || Ok(default.to_owned()),
            |v| as_str(v, &self.ptr(key), key).map(str::to_owned),
        )
    }

    /// An optional integer (absent → `default`; `null` is a type error).
    pub(crate) fn opt_u64(&self, key: &str, default: u64) -> Read<u64> {
        self.get(key)
            .map_or(Ok(default), |v| as_u64(v, &self.ptr(key), key))
    }

    /// An optional number (absent → `default`; `null` is a type error).
    pub(crate) fn opt_f64(&self, key: &str, default: f64) -> Read<f64> {
        self.get(key)
            .map_or(Ok(default), |v| as_f64(v, &self.ptr(key), key))
    }

    /// An optional boolean (absent → `default`; `null` is a type error).
    pub(crate) fn opt_bool(&self, key: &str, default: bool) -> Read<bool> {
        self.get(key)
            .map_or(Ok(default), |v| as_bool(v, &self.ptr(key), key))
    }

    /// An optional value that may be `null`: absent → `default`, `null` → `None`.
    pub(crate) fn nullable<T>(
        &self,
        key: &str,
        default: Option<T>,
        read: impl FnOnce(&'a Json, &str) -> Read<T>,
    ) -> Read<Option<T>> {
        match self.get(key) {
            None => Ok(default),
            Some(Json::Null) => Ok(None),
            Some(v) => read(v, &self.ptr(key)).map(Some),
        }
    }
}

pub(crate) fn as_str<'a>(v: &'a Json, at: &str, field: &str) -> Read<&'a str> {
    match v {
        Json::Str(s) => Ok(s),
        _ => error(
            at,
            format!("`{field}` must be a string, got {}", v.type_name()),
        ),
    }
}

pub(crate) fn as_bool(v: &Json, at: &str, field: &str) -> Read<bool> {
    match v {
        Json::Bool(b) => Ok(*b),
        _ => error(
            at,
            format!("`{field}` must be true or false, got {}", v.type_name()),
        ),
    }
}

/// An integer field (§2.1, "Numbers"): a JSON integer from 0 to `2^64 − 1`.
pub(crate) fn as_u64(v: &Json, at: &str, field: &str) -> Read<u64> {
    match v {
        Json::U64(n) => Ok(*n),
        Json::I64(n) => error(at, format!("`{field}` must not be negative, got {n}")),
        Json::F64(x) => error(
            at,
            format!(
                "`{field}` must be an integer written without a fraction or exponent, from 0 to \
                 2^64 − 1, got {x}"
            ),
        ),
        _ => error(
            at,
            format!("`{field}` must be an integer, got {}", v.type_name()),
        ),
    }
}

/// A float field (§2.1): any JSON number, as the nearest `f64`.
#[allow(
    clippy::cast_precision_loss,
    reason = "an integer read as a float field is rounded to the nearest f64 (§2.1)"
)]
pub(crate) fn as_f64(v: &Json, at: &str, field: &str) -> Read<f64> {
    match v {
        Json::U64(n) => Ok(*n as f64),
        Json::I64(n) => Ok(*n as f64),
        Json::F64(x) => Ok(*x),
        _ => error(
            at,
            format!("`{field}` must be a number, got {}", v.type_name()),
        ),
    }
}

/// A float that may be non-finite (training.md §4.1): a JSON number, or the string `"nan"`,
/// `"inf"` or `"-inf"`.
pub(crate) fn as_f64_or_special(v: &Json, at: &str, field: &str) -> Read<f64> {
    match v {
        Json::Str(s) => match s.as_str() {
            "nan" => Ok(f64::NAN),
            "inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            _ => error(
                at,
                format!("`{field}` must be a number, \"nan\", \"inf\" or \"-inf\", got {s:?}"),
            ),
        },
        _ => as_f64(v, at, field),
    }
}

pub(crate) fn as_array<'a>(v: &'a Json, at: &str, field: &str) -> Read<&'a [Json]> {
    match v {
        Json::Arr(items) => Ok(items),
        _ => error(
            at,
            format!("`{field}` must be an array, got {}", v.type_name()),
        ),
    }
}

/// An `f64` written as training.md §4.1 says: the shortest digits that read back as the same
/// `f64` (`serde_json` writes them with `ryu`: `2.3051`, `0.00001`, `1e-7`), or the string
/// `"nan"`, `"inf"` or `"-inf"`. Serializes through serde.
#[derive(Clone, Copy, Debug)]
pub struct Number(pub f64);

impl PartialEq for Number {
    /// Bitwise equality, so that NaN equals NaN (for comparing events and outcomes).
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Serialize for Number {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let x = self.0;
        if x.is_nan() {
            s.serialize_str("nan")
        } else if x == f64::INFINITY {
            s.serialize_str("inf")
        } else if x == f64::NEG_INFINITY {
            s.serialize_str("-inf")
        } else {
            // serde_json writes an f64 with `ryu`: the shortest form that round-trips.
            s.serialize_f64(x)
        }
    }
}

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Json::deserialize(d)?;
        as_f64_or_special(&v, "", "number")
            .map(Number)
            .map_err(|e| de::Error::custom(e.message))
    }
}

/// Writes `value` as compact JSON. The types written here have string keys and no fallible
/// `Serialize` impls, so `serde_json` can't fail on them.
pub(crate) fn to_string<T: Serialize>(value: &T) -> String {
    match serde_json::to_string(value) {
        Ok(s) => s,
        // Unreachable for the crate's types (see above); an empty string would fail any reader.
        Err(e) => {
            tracing::error!(error = %e, "serializing a training value failed");
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_read_as_the_nearest_f64() {
        for text in [
            "0.1",
            "1e-5",
            "0.00010000000000000003",
            "2.2250738585072014e-308",
        ] {
            let Json::F64(x) = parse(text).unwrap() else {
                panic!("{text}")
            };
            assert_eq!(x, text.parse::<f64>().unwrap(), "{text}");
            // And they are written back in the shortest form that reads back bit for bit.
            let written = to_string(&Number(x));
            assert_eq!(written.parse::<f64>().unwrap().to_bits(), x.to_bits());
        }
        assert_eq!(parse("5").unwrap(), Json::U64(5));
        assert_eq!(parse("-5").unwrap(), Json::I64(-5));
        assert_eq!(parse("5.0").unwrap(), Json::F64(5.0));
        assert_eq!(parse("18446744073709551615").unwrap(), Json::U64(u64::MAX));
        assert!(matches!(
            parse("18446744073709551616").unwrap(),
            Json::F64(_)
        ));
    }

    #[test]
    fn special_numbers() {
        assert_eq!(to_string(&Number(f64::NAN)), "\"nan\"");
        assert_eq!(to_string(&Number(f64::INFINITY)), "\"inf\"");
        assert_eq!(to_string(&Number(f64::NEG_INFINITY)), "\"-inf\"");
        // The shortest round-trip digits, as serde_json (ryu) writes them.
        assert_eq!(to_string(&Number(1e-5)), "0.00001");
        assert_eq!(to_string(&Number(1e-7)), "1e-7");
        assert_eq!(to_string(&Number(1e300)), "1e+300");
        assert_eq!(to_string(&Number(2.3051)), "2.3051");
        assert_eq!(to_string(&Number(-0.0)), "-0.0");
        for s in ["\"nan\"", "\"inf\"", "\"-inf\"", "0.25"] {
            let n: Number = serde_json::from_str(s).unwrap();
            assert_eq!(to_string(&n), s);
        }
        assert!(serde_json::from_str::<Number>("\"NaN\"").is_err());
    }

    #[test]
    fn objects_keep_duplicates_and_order() {
        let v = parse(r#"{"b": 1, "a": 2, "b": 3}"#).unwrap();
        let err = Obj::new(&v, "/x", "x").err().unwrap();
        assert_eq!(err.pointer, "/x/b");
        let v = parse(r#"{"b": 1, "a~/": 2}"#).unwrap();
        let o = Obj::new(&v, "", "x").unwrap();
        let err = o.allow(&["b"], "x").unwrap_err();
        assert_eq!(err.pointer, "/a~0~1");
        assert_eq!(o.req("c").unwrap_err().pointer, "/c");
        assert_eq!(parse("[[1], {\"a\": [2]}]").unwrap().depth(), 3);
    }
}
