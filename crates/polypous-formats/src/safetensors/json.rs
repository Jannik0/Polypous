//! JSON parsing of safetensors headers and sharded indices.
//!
//! Hand-written `serde` visitors instead of `serde_json::Value` or derives, because the format
//! rules need more than the defaults give (`docs/specs/formats.md` §1.2 and §1.5):
//!
//! - duplicate keys are rejected (rule 8; `serde_json` maps keep the last one silently),
//! - shapes are bounded to [`MAX_RANK`] while parsing, before allocating (rule 10),
//! - violations map to specific [`ErrorKind`]s with the tensor name attached.
//!
//! A visitor that detects a specific violation stores it in a [`Slot`] and aborts parsing with a
//! generic `serde` error; [`finish`] then prefers the stored error. Syntax errors always win, since
//! a slot entry set on the way out of a syntax error would be misleading.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::fmt;

use serde::de::{self, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

use super::error::truncate_for_message;
use super::{Error, ErrorKind, MAX_RANK};

/// The first specific error found while parsing.
#[derive(Default)]
struct Slot(RefCell<Option<Error>>);

impl Slot {
    /// Stores `err` unless an error is already stored.
    fn set(&self, err: Error) {
        let mut slot = self.0.borrow_mut();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    /// Stores `err` (unless one is stored) and returns a `serde` error that aborts parsing.
    fn fail<E: de::Error>(&self, err: Error) -> E {
        let msg = err.kind().to_string();
        self.set(err);
        E::custom(msg)
    }

    fn take(&self) -> Option<Error> {
        self.0.borrow_mut().take()
    }
}

/// Parses `text` with `seed`, requires only whitespace after the value, and maps errors.
/// `generic` builds the error for failures without a more specific slot entry.
fn finish<'de, S>(
    text: &'de str,
    seed: S,
    slot: &Slot,
    generic: fn(String) -> ErrorKind,
) -> Result<S::Value, Error>
where
    S: DeserializeSeed<'de>,
{
    let mut de = serde_json::Deserializer::from_str(text);
    let result = seed.deserialize(&mut de).and_then(|v| de.end().map(|()| v));
    match result {
        Ok(v) => Ok(v),
        Err(e) if e.is_syntax() || e.is_eof() || e.is_io() => Err(generic(e.to_string()).into()),
        Err(e) => Err(slot.take().unwrap_or_else(|| generic(e.to_string()).into())),
    }
}

// ---------------------------------------------------------------------------------------------
// safetensors header
// ---------------------------------------------------------------------------------------------

/// A tensor entry as written in the header, before the checks of `header.rs`.
#[derive(Debug)]
pub(crate) struct RawTensor {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) data_offsets: (u64, u64),
}

/// The parsed header: `__metadata__` and the tensor entries, sorted by name.
#[derive(Debug, Default)]
pub(crate) struct RawHeader {
    pub(crate) metadata: BTreeMap<String, String>,
    pub(crate) tensors: BTreeMap<String, RawTensor>,
}

/// Parses a header (`docs/specs/formats.md` §1.2 rules 5–8 and 10).
pub(crate) fn parse_header(text: &str) -> Result<RawHeader, Error> {
    let slot = Slot::default();
    finish(
        text,
        HeaderSeed { slot: &slot },
        &slot,
        ErrorKind::InvalidHeader,
    )
}

struct HeaderSeed<'s> {
    slot: &'s Slot,
}

impl<'de> DeserializeSeed<'de> for HeaderSeed<'_> {
    type Value = RawHeader;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<RawHeader, D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for HeaderSeed<'_> {
    type Value = RawHeader;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<RawHeader, A::Error> {
        let slot = self.slot;
        let mut header = RawHeader::default();
        let mut seen_metadata = false;
        while let Some(key) = map.next_key::<String>()? {
            if key == "__metadata__" {
                if seen_metadata {
                    return Err(slot.fail(ErrorKind::DuplicateKey { key }.into()));
                }
                seen_metadata = true;
                header.metadata =
                    map.next_value_seed(StringMapSeed { slot })
                        .inspect_err(|e: &A::Error| {
                            slot.set(ErrorKind::InvalidMetadata(e.to_string()).into());
                        })?;
                continue;
            }
            match header.tensors.entry(key) {
                Entry::Occupied(o) => {
                    let key = o.key().clone();
                    return Err(slot.fail(ErrorKind::DuplicateKey { key }.into()));
                }
                Entry::Vacant(v) => {
                    let name = v.key().clone();
                    let tensor = map
                        .next_value_seed(TensorSeed { slot, name: &name })
                        .inspect_err(|e: &A::Error| {
                            slot.set(
                                Error::from(ErrorKind::InvalidHeader(e.to_string()))
                                    .with_tensor(&name),
                            );
                        })?;
                    v.insert(tensor);
                }
            }
        }
        Ok(header)
    }
}

/// An object of string to string, without duplicate keys (`__metadata__`).
struct StringMapSeed<'s> {
    slot: &'s Slot,
}

impl<'de> DeserializeSeed<'de> for StringMapSeed<'_> {
    type Value = BTreeMap<String, String>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for StringMapSeed<'_> {
    type Value = BTreeMap<String, String>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an object of string to string")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut out = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            let value = map.next_value::<String>()?;
            match out.entry(key) {
                Entry::Occupied(o) => {
                    let key = o.key().clone();
                    return Err(self.slot.fail(ErrorKind::DuplicateKey { key }.into()));
                }
                Entry::Vacant(v) => {
                    v.insert(value);
                }
            }
        }
        Ok(out)
    }
}

/// One tensor entry: `{"dtype": …, "shape": […], "data_offsets": [b, e]}`.
struct TensorSeed<'s, 'n> {
    slot: &'s Slot,
    name: &'n str,
}

impl<'de> DeserializeSeed<'de> for TensorSeed<'_, '_> {
    type Value = RawTensor;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<RawTensor, D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for TensorSeed<'_, '_> {
    type Value = RawTensor;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a tensor entry object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<RawTensor, A::Error> {
        let slot = self.slot;
        let fail =
            |kind: ErrorKind| -> A::Error { slot.fail(Error::from(kind).with_tensor(self.name)) };
        let duplicate = |key: &str| ErrorKind::DuplicateKey {
            key: key.to_owned(),
        };
        let mut dtype = None;
        let mut shape = None;
        let mut data_offsets = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "dtype" => {
                    if dtype.is_some() {
                        return Err(fail(duplicate("dtype")));
                    }
                    dtype = Some(map.next_value::<String>()?);
                }
                "shape" => {
                    if shape.is_some() {
                        return Err(fail(duplicate("shape")));
                    }
                    shape = Some(map.next_value_seed(ShapeSeed {
                        slot,
                        name: self.name,
                    })?);
                }
                "data_offsets" => {
                    if data_offsets.is_some() {
                        return Err(fail(duplicate("data_offsets")));
                    }
                    data_offsets = Some(map.next_value_seed(OffsetsSeed)?);
                }
                // Unknown fields are ignored, as in the reference implementation (rule 7).
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        let missing = |field: &str| ErrorKind::InvalidHeader(format!("missing field `{field}`"));
        let dtype = dtype.ok_or_else(|| fail(missing("dtype")))?;
        let shape = shape.ok_or_else(|| fail(missing("shape")))?;
        let data_offsets = data_offsets.ok_or_else(|| fail(missing("data_offsets")))?;
        Ok(RawTensor {
            dtype,
            shape,
            data_offsets,
        })
    }
}

/// A shape: an array of at most [`MAX_RANK`] non-negative integers.
struct ShapeSeed<'s, 'n> {
    slot: &'s Slot,
    name: &'n str,
}

impl<'de> DeserializeSeed<'de> for ShapeSeed<'_, '_> {
    type Value = Vec<u64>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Vec<u64>, D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for ShapeSeed<'_, '_> {
    type Value = Vec<u64>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an array of non-negative integers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u64>, A::Error> {
        let mut shape = Vec::new();
        while let Some(dim) = seq.next_element::<u64>()? {
            if shape.len() == MAX_RANK {
                return Err(self.slot.fail(
                    Error::from(ErrorKind::RankTooLarge { max: MAX_RANK }).with_tensor(self.name),
                ));
            }
            shape.push(dim);
        }
        Ok(shape)
    }
}

/// `data_offsets`: an array of exactly two non-negative integers.
struct OffsetsSeed;

impl<'de> DeserializeSeed<'de> for OffsetsSeed {
    type Value = (u64, u64);
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(u64, u64), D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for OffsetsSeed {
    type Value = (u64, u64);

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an array of two non-negative integers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(u64, u64), A::Error> {
        let begin = seq
            .next_element::<u64>()?
            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
        let end = seq
            .next_element::<u64>()?
            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
        if seq.next_element::<IgnoredAny>()?.is_some() {
            return Err(de::Error::invalid_length(3, &self));
        }
        Ok((begin, end))
    }
}

// ---------------------------------------------------------------------------------------------
// Sharded index
// ---------------------------------------------------------------------------------------------

/// A parsed `*.safetensors.index.json` (`docs/specs/formats.md` §1.5).
#[derive(Debug, Default)]
pub(crate) struct RawIndex {
    /// `metadata.total_size`, if it is a non-negative integer.
    pub(crate) total_size: Option<u64>,
    /// `weight_map`: tensor name to shard path, sorted by tensor name.
    pub(crate) weight_map: BTreeMap<String, String>,
}

/// Parses a sharded index.
pub(crate) fn parse_index(text: &str) -> Result<RawIndex, Error> {
    let slot = Slot::default();
    finish(
        text,
        IndexSeed { slot: &slot },
        &slot,
        ErrorKind::InvalidIndex,
    )
}

struct IndexSeed<'s> {
    slot: &'s Slot,
}

impl<'de> DeserializeSeed<'de> for IndexSeed<'_> {
    type Value = RawIndex;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<RawIndex, D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for IndexSeed<'_> {
    type Value = RawIndex;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<RawIndex, A::Error> {
        let slot = self.slot;
        let mut weight_map = None;
        let mut metadata = None;
        let mut seen = std::collections::BTreeSet::<String>::new();
        while let Some(key) = map.next_key::<String>()? {
            if seen.contains(&key) {
                return Err(slot.fail(ErrorKind::DuplicateKey { key }.into()));
            }
            match key.as_str() {
                "weight_map" => weight_map = Some(map.next_value_seed(StringMapSeed { slot })?),
                "metadata" => metadata = Some(map.next_value_seed(IndexMetadataSeed { slot })?),
                // Other top-level keys are ignored (§1.5).
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            // Keys are kept only to detect duplicates; their total size is bounded by the index
            // size limit.
            seen.insert(key);
        }
        let weight_map = weight_map.ok_or_else(|| {
            slot.fail(ErrorKind::InvalidIndex("missing field `weight_map`".into()).into())
        })?;
        Ok(RawIndex {
            total_size: metadata.flatten(),
            weight_map,
        })
    }
}

/// The index's `metadata` object: returns `total_size` if it is a non-negative integer.
struct IndexMetadataSeed<'s> {
    slot: &'s Slot,
}

impl<'de> DeserializeSeed<'de> for IndexMetadataSeed<'_> {
    type Value = Option<u64>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Option<u64>, D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for IndexMetadataSeed<'_> {
    type Value = Option<u64>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Option<u64>, A::Error> {
        let mut total_size = None;
        let mut seen = std::collections::BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if key == "total_size" {
                total_size = map.next_value::<MaybeU64>()?.0;
            } else {
                map.next_value::<IgnoredAny>()?;
            }
            if !seen.insert(key.clone()) {
                return Err(self.slot.fail(ErrorKind::DuplicateKey { key }.into()));
            }
        }
        Ok(total_size)
    }
}

/// Any JSON value; `Some` if it is a non-negative integer.
struct MaybeU64(Option<u64>);

impl<'de> de::Deserialize<'de> for MaybeU64 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = MaybeU64;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<MaybeU64, E> {
                Ok(MaybeU64(Some(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<MaybeU64, E> {
                Ok(MaybeU64(u64::try_from(v).ok()))
            }
            fn visit_f64<E: de::Error>(self, _: f64) -> Result<MaybeU64, E> {
                Ok(MaybeU64(None))
            }
            fn visit_bool<E: de::Error>(self, _: bool) -> Result<MaybeU64, E> {
                Ok(MaybeU64(None))
            }
            fn visit_str<E: de::Error>(self, _: &str) -> Result<MaybeU64, E> {
                Ok(MaybeU64(None))
            }
            fn visit_unit<E: de::Error>(self) -> Result<MaybeU64, E> {
                Ok(MaybeU64(None))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<MaybeU64, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(MaybeU64(None))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<MaybeU64, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(MaybeU64(None))
            }
        }
        d.deserialize_any(V)
    }
}

/// Truncates a dtype name for [`ErrorKind::UnknownDtype`].
pub(crate) fn dtype_name_for_message(name: &str) -> String {
    truncate_for_message(name, 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(text: &str) -> ErrorKind {
        parse_header(text).unwrap_err().into_kind()
    }

    #[test]
    fn parses_header() {
        let h = parse_header(
            r#"{"__metadata__":{"a":"b"},"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"extra":[1,{"k":null}]}}  "#,
        )
        .unwrap();
        assert_eq!(h.metadata.get("a").map(String::as_str), Some("b"));
        let x = &h.tensors["x"];
        assert_eq!(x.dtype, "F32");
        assert_eq!(x.shape, vec![2]);
        assert_eq!(x.data_offsets, (0, 8));
    }

    #[test]
    fn rejects_structure_errors() {
        assert!(matches!(kind("[]"), ErrorKind::InvalidHeader(_)));
        assert!(matches!(kind("{} x"), ErrorKind::InvalidHeader(_)));
        assert!(matches!(kind("{"), ErrorKind::InvalidHeader(_)));
        assert!(matches!(
            kind(r#"{"__metadata__":{"a":1}}"#),
            ErrorKind::InvalidMetadata(_)
        ));
        assert!(matches!(
            kind(r#"{"__metadata__":[]}"#),
            ErrorKind::InvalidMetadata(_)
        ));
        assert!(matches!(
            kind(r#"{"__metadata__":{"a":"b","a":"c"}}"#),
            ErrorKind::DuplicateKey { .. }
        ));
        assert!(matches!(
            kind(r#"{"__metadata__":{},"__metadata__":{}}"#),
            ErrorKind::DuplicateKey { .. }
        ));
        let e =
            parse_header(r#"{"t":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}"#).unwrap_err();
        assert_eq!(e.tensor(), Some("t"));
        assert!(matches!(e.kind(), ErrorKind::InvalidHeader(_)));
        let e = parse_header(r#"{"t":{"dtype":"F32","shape":[]}}"#).unwrap_err();
        assert_eq!(e.tensor(), Some("t"));
        assert!(matches!(
            kind(r#"{"t":{"dtype":"F32","shape":[],"data_offsets":[0,0,0]}}"#),
            ErrorKind::InvalidHeader(_)
        ));
        assert!(matches!(
            kind(r#"{"t":{"dtype":"F32","dtype":"F32","shape":[],"data_offsets":[0,4]}}"#),
            ErrorKind::DuplicateKey { .. }
        ));
    }

    #[test]
    fn rank_is_bounded_while_parsing() {
        let dims = vec!["1"; MAX_RANK + 1].join(",");
        let text = format!(r#"{{"t":{{"dtype":"F32","shape":[{dims}],"data_offsets":[0,4]}}}}"#);
        assert!(matches!(kind(&text), ErrorKind::RankTooLarge { .. }));
        let dims = vec!["1"; MAX_RANK].join(",");
        let text = format!(r#"{{"t":{{"dtype":"F32","shape":[{dims}],"data_offsets":[0,4]}}}}"#);
        assert!(parse_header(&text).is_ok());
    }

    #[test]
    fn deep_nesting_is_an_error() {
        let text = format!(
            r#"{{"t":{{"x":{}{}}}}}"#,
            "[".repeat(10_000),
            "]".repeat(10_000)
        );
        assert!(matches!(kind(&text), ErrorKind::InvalidHeader(_)));
    }

    #[test]
    fn parses_index() {
        let idx = parse_index(
            r#"{"metadata":{"total_size":12,"x":[1]},"weight_map":{"a":"s1","b":"s2"},"other":1}"#,
        )
        .unwrap();
        assert_eq!(idx.total_size, Some(12));
        assert_eq!(idx.weight_map.len(), 2);
        let idx = parse_index(r#"{"metadata":{"total_size":"12"},"weight_map":{}}"#).unwrap();
        assert_eq!(idx.total_size, None);
        assert!(matches!(
            parse_index(r#"{"metadata":{}}"#).unwrap_err().kind(),
            ErrorKind::InvalidIndex(_)
        ));
        assert!(matches!(
            parse_index(r#"{"weight_map":{"a":"s","a":"t"}}"#)
                .unwrap_err()
                .kind(),
            ErrorKind::DuplicateKey { .. }
        ));
        assert!(matches!(
            parse_index(r#"{"weight_map":{},"weight_map":{}}"#)
                .unwrap_err()
                .kind(),
            ErrorKind::DuplicateKey { .. }
        ));
        assert!(matches!(
            parse_index(r#"{"weight_map":{"a":1}}"#).unwrap_err().kind(),
            ErrorKind::InvalidIndex(_)
        ));
    }
}
