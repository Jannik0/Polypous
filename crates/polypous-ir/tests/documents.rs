//! Graph documents: round trips, one test per implemented error code (with its JSON pointer), the
//! resource limits of `docs/specs/graph-ir.md` §4.3, and a deterministic fuzz-style test.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous_ir::json::Value;
use polypous_ir::model::{Dim, NodeKind};
use polypous_ir::{ErrorCode, Graph, ValidateOptions, load};
use serde_json::json;

const MLP: &str = include_str!("data/mlp.graph.json");
const REPEAT: &str = include_str!("data/repeat.graph.json");

fn opts() -> ValidateOptions {
    ValidateOptions::default()
}

/// (code, pointer) of every diagnostic.
fn errors(doc: &[u8]) -> Vec<(String, String)> {
    load(doc, &opts())
        .report
        .diagnostics
        .iter()
        .map(|d| (d.code.as_str().to_owned(), d.pointer.clone()))
        .collect()
}

#[track_caller]
fn assert_error(doc: &serde_json::Value, code: ErrorCode, pointer: &str) {
    assert_error_bytes(doc.to_string().as_bytes(), code, pointer);
}

#[track_caller]
fn assert_error_bytes(doc: &[u8], code: ErrorCode, pointer: &str) {
    let errs = errors(doc);
    assert!(
        errs.iter().any(|(c, p)| c == code.as_str() && p == pointer),
        "expected {code} at {pointer:?}, got {errs:?}"
    );
}

#[track_caller]
fn assert_only(doc: &serde_json::Value, code: ErrorCode, pointer: &str) {
    let errs = errors(doc.to_string().as_bytes());
    assert_eq!(errs, vec![(code.as_str().to_owned(), pointer.to_owned())]);
}

#[track_caller]
fn assert_valid(doc: &serde_json::Value) {
    let errs = errors(doc.to_string().as_bytes());
    assert!(errs.is_empty(), "{errs:?}");
}

fn mlp() -> serde_json::Value {
    serde_json::from_str(MLP).unwrap()
}

fn rep() -> serde_json::Value {
    serde_json::from_str(REPEAT).unwrap()
}

/// A minimal valid graph to build test cases from.
fn base() -> serde_json::Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "a" } ],
        "nodes": [ { "id": "a", "op": "nn.relu", "inputs": { "x": "x" } } ]
    })
}

/// A graph with one repeat around `body_nodes` (carry `h` from `x`, carry_out `h` → `out_ref`).
fn with_repeat(count: u64, body: serde_json::Value) -> serde_json::Value {
    let mut g = base();
    g["nodes"] = json!([ { "id": "r", "op": "repeat", "count": count, "carry": { "h": "x" }, "body": body } ]);
    g["outputs"] = json!([ { "name": "y", "value": "r:h" } ]);
    g
}

// ---------------------------------------------------------------------------------------------
// Example documents and round trips (§2, §6)

#[test]
fn spec_examples_validate() {
    for doc in [MLP, REPEAT] {
        let out = load(doc.as_bytes(), &opts());
        assert!(out.report.is_ok(), "{}", out.report);
        assert!(out.graph.is_some());
    }
}

#[test]
fn load_save_load_is_identity() {
    for doc in [MLP, REPEAT] {
        let g1 = Graph::from_json(doc.as_bytes()).unwrap();
        let saved = g1.to_json();
        let g2 = Graph::from_json(saved.as_bytes()).unwrap();
        assert_eq!(g1, g2);
        // Saving is a fixed point.
        assert_eq!(g2.to_json(), saved);
    }
}

#[test]
fn meta_and_float_survive_unchanged() {
    let g = Graph::from_json(REPEAT.as_bytes()).unwrap();
    let saved = g.to_json();
    // meta keeps its keys in document order and its values.
    let original: serde_json::Value = serde_json::from_str(REPEAT).unwrap();
    let reloaded: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(original["meta"], reloaded["meta"]);
    assert_eq!(original["nodes"][0]["meta"], reloaded["nodes"][0]["meta"]);
    assert_eq!(
        original["outputs"][1]["meta"],
        reloaded["outputs"][1]["meta"]
    );
    let order = |s: &str| {
        s.find("\"zoom\"").unwrap() < s.find("\"tags\"").unwrap()
            && s.find("\"tags\"").unwrap() < s.find("\"nested\"").unwrap()
            && s.find("\"z\": null").unwrap() < s.find("\"a\": [").unwrap()
    };
    assert!(order(REPEAT) && order(&saved));
    // `float` is saved as written, never as the resolved dtype.
    assert!(saved.contains(r#""dtype": "float""#));
    let mlp = Graph::from_json(MLP.as_bytes()).unwrap().to_json();
    assert_eq!(mlp.matches(r#""dtype": "float""#).count(), 5);
    // Dims stay as written.
    assert!(saved.contains(r#""4 * 4""#));
}

#[test]
fn canonical_layout() {
    let saved = Graph::from_json(MLP.as_bytes()).unwrap().to_json();
    let expected_start = "{\n  \"format\": \"polypous.graph\",\n  \"format_version\": 1,\n  \"opsets\": {\n    \"core\": 1,\n    \"nn\": 1\n  },\n  \"name\": \"mnist-mlp\",\n";
    assert!(saved.starts_with(expected_start), "{saved}");
    assert!(saved.contains("\"shape\": [\"batch\", 784]"));
    assert!(saved.ends_with("}\n"));
    // Schema fields are written in schema order whatever the input order; maps keyed by names
    // (here node `params`) keep their document order.
    let shuffled = mlp().to_string(); // serde_json sorts keys alphabetically
    let from_shuffled = Graph::from_json(shuffled.as_bytes()).unwrap().to_json();
    assert!(from_shuffled.starts_with(expected_start), "{from_shuffled}");
    assert!(from_shuffled.contains("\"bias\": \"fc1.bias\",\n        \"weight\""));
    assert!(saved.contains("\"weight\": \"fc1.weight\",\n        \"bias\""));
    let a: serde_json::Value = serde_json::from_str(&saved).unwrap();
    let b: serde_json::Value = serde_json::from_str(&from_shuffled).unwrap();
    assert_eq!(a, b);
}

#[test]
fn typed_graph_validates_like_its_json() {
    let mut g = Graph::from_json(MLP.as_bytes()).unwrap();
    assert!(g.validate(&opts()).is_ok());
    g.nodes[1].id = "fc1".into();
    let report = g.validate(&opts());
    assert!(report.has(ErrorCode::DuplicateName), "{report}");
    assert_eq!(report.diagnostics[0].pointer, "/nodes/1/id");
    // A typed graph whose meta would break the JSON limits doesn't validate.
    let mut g = Graph::from_json(MLP.as_bytes()).unwrap();
    let mut deep = Value::Null;
    for _ in 0..130 {
        deep = Value::Array(vec![deep]);
    }
    g.meta = Some(deep);
    assert!(g.validate(&opts()).has(ErrorCode::JsonTooDeep));
}

#[test]
fn load_returns_graph_with_validation_errors() {
    let mut doc = base();
    doc["nodes"][0]["inputs"]["x"] = json!("nope");
    let out = load(doc.to_string().as_bytes(), &opts());
    assert!(out.graph.is_some());
    assert!(out.report.has(ErrorCode::UnknownReference));
    assert!(Graph::from_json(doc.to_string().as_bytes()).is_err());
    // Schema errors lose information: no graph.
    let mut doc = base();
    doc["bogus"] = json!(1);
    assert!(load(doc.to_string().as_bytes(), &opts()).graph.is_none());
}

#[test]
fn errors_are_collected() {
    let mut doc = base();
    doc["nodes"] = json!([
        { "id": "a", "op": "nn.relu", "inputs": { "x": "missing1" } },
        { "id": "b", "op": "nn.nope", "inputs": { "x": "missing2" } },
        { "id": "a", "op": "nn.relu", "inputs": { "x": "x" } }
    ]);
    let errs = errors(doc.to_string().as_bytes());
    assert!(errs.len() >= 4, "{errs:?}");
    let report = load(doc.to_string().as_bytes(), &opts()).report;
    let d = report
        .diagnostics
        .iter()
        .find(|d| d.code == ErrorCode::UnknownOp)
        .unwrap();
    assert_eq!(d.node_id.as_deref(), Some("b"));
}

#[test]
fn repeat_model() {
    let g = Graph::from_json(REPEAT.as_bytes()).unwrap();
    let NodeKind::Repeat(r) = &g.nodes[0].kind else {
        panic!()
    };
    assert_eq!(r.count, 2);
    assert_eq!(r.body.state[0].shape[2], Dim::Expr("4 * 4".into()));
}

// ---------------------------------------------------------------------------------------------
// One test per error code

#[test]
fn invalid_json() {
    assert_error_bytes(b"{ \"format\": ", ErrorCode::InvalidJson, "");
    assert_error_bytes(b"[]", ErrorCode::InvalidJson, "");
    assert_error_bytes(b"\xef\xbb\xbf{}", ErrorCode::InvalidJson, "");
    let mut doc = base();
    doc["nodes"][0]["inputs"] = json!(["x"]);
    assert_error(&doc, ErrorCode::InvalidJson, "/nodes/0/inputs");
    let mut doc = base();
    doc["inputs"][0]["shape"][1] = json!(4.5);
    assert_error(&doc, ErrorCode::InvalidJson, "/inputs/0/shape/1");
    // Duplicate keys are rejected everywhere, `meta` included.
    let dup = MLP.replacen(
        "\"name\": \"mnist-mlp\",",
        "\"name\": \"a\", \"name\": \"b\",",
        1,
    );
    assert_error_bytes(dup.as_bytes(), ErrorCode::InvalidJson, "/name");
    let dup = MLP.replacen("\"zoom\": 1.0", "\"zoom\": 1.0, \"zoom\": 2.0", 1);
    assert_error_bytes(
        dup.as_bytes(),
        ErrorCode::InvalidJson,
        "/meta/builder/canvas/zoom",
    );
}

#[test]
fn document_too_large() {
    let mut doc = MLP.as_bytes().to_vec();
    doc.resize(64 * 1024 * 1024 + 1, b' ');
    assert_eq!(errors(&doc), vec![("document_too_large".into(), "".into())]);
    // Exactly at the limit is fine.
    doc.truncate(64 * 1024 * 1024);
    assert!(errors(&doc).is_empty());
    // At most 4,000,000 JSON values, checked before parsing.
    let with_meta = |zeros: usize| {
        let mut doc = mlp();
        doc["meta"] = serde_json::Value::Array(vec![json!(0); zeros]);
        doc.to_string()
    };
    let small = with_meta(0);
    let other = 4_000_000 - values_of(&small); // values besides the zeros in `meta`
    assert!(errors(with_meta(other).as_bytes()).is_empty());
    assert_eq!(
        errors(with_meta(other + 1).as_bytes()),
        vec![("document_too_large".into(), "".into())]
    );
}

/// Counts JSON values (not keys) with serde_json, as an independent check of the pre-scan.
fn values_of(doc: &str) -> usize {
    fn count(v: &serde_json::Value) -> usize {
        1 + match v {
            serde_json::Value::Array(a) => a.iter().map(count).sum(),
            serde_json::Value::Object(o) => o.values().map(count).sum(),
            _ => 0,
        }
    }
    count(&serde_json::from_str(doc).unwrap())
}

#[test]
fn json_too_deep() {
    let nest = |n: usize| {
        let mut doc = mlp();
        let mut v = json!(1);
        for _ in 0..n {
            v = json!([v]);
        }
        doc["meta"] = v;
        doc.to_string()
    };
    // The root object is level 1, so `meta` content may nest 127 levels.
    assert!(errors(nest(127).as_bytes()).is_empty());
    assert_eq!(
        errors(nest(128).as_bytes()),
        vec![("json_too_deep".into(), "".into())]
    );
}

#[test]
fn string_too_long() {
    let mut doc = mlp();
    doc["meta"] = json!({ "note": "x".repeat(64 * 1024) });
    assert_valid(&doc);
    doc["meta"] = json!({ "note": "x".repeat(64 * 1024 + 1) });
    assert_only(&doc, ErrorCode::StringTooLong, "/meta/note");
    let mut doc = mlp();
    doc["meta"] = json!({ "x".repeat(64 * 1024 + 1): 1 });
    assert_only(&doc, ErrorCode::StringTooLong, "/meta");
}

#[test]
fn unsupported_format() {
    let mut doc = base();
    doc["format"] = json!("onnx");
    assert_only(&doc, ErrorCode::UnsupportedFormat, "/format");
    let mut doc = base();
    doc["format_version"] = json!(2);
    assert_only(&doc, ErrorCode::UnsupportedFormat, "/format_version");
}

#[test]
fn unsupported_opset() {
    let mut doc = base();
    doc["opsets"]["nn"] = json!(2);
    assert_only(&doc, ErrorCode::UnsupportedOpset, "/opsets/nn");
    let mut doc = base();
    doc["opsets"]["transformer"] = json!(1);
    assert_only(&doc, ErrorCode::UnsupportedOpset, "/opsets/transformer");
}

#[test]
fn unknown_field() {
    let mut doc = base();
    doc["nodes"][0]["input"] = json!({ "x": "x" });
    assert_error(&doc, ErrorCode::UnknownField, "/nodes/0/input");
    let mut doc = base();
    doc["Nodes"] = json!([]);
    assert_error(&doc, ErrorCode::UnknownField, "/Nodes");
    let mut doc = rep();
    doc["nodes"][0]["body"]["carry"] = json!({});
    assert_error(&doc, ErrorCode::UnknownField, "/nodes/0/body/carry");
    let mut doc = rep();
    doc["params"][0]["init"]["value"] = json!(1);
    assert_error(&doc, ErrorCode::UnknownField, "/params/0/init/value");
    // Anything goes inside meta.
    let mut doc = base();
    doc["nodes"][0]["meta"] = json!({ "whatever": [1, { "x": null }] });
    assert_valid(&doc);
}

#[test]
fn missing_field() {
    let mut doc = base();
    doc["nodes"][0].as_object_mut().unwrap().remove("op");
    assert_only(&doc, ErrorCode::MissingField, "/nodes/0/op");
    let mut doc = mlp();
    doc["params"][0].as_object_mut().unwrap().remove("shape");
    assert_only(&doc, ErrorCode::MissingField, "/params/0/shape");
    let mut doc = base();
    doc.as_object_mut().unwrap().remove("outputs");
    assert_only(&doc, ErrorCode::MissingField, "/outputs");
    let mut doc = rep();
    doc["nodes"][0]["body"]
        .as_object_mut()
        .unwrap()
        .remove("carry_out");
    assert_error(&doc, ErrorCode::MissingField, "/nodes/0/body/carry_out");
}

#[test]
fn invalid_value() {
    let mut doc = base();
    doc["inputs"][0]["dtype"] = json!("q4_k");
    assert_only(&doc, ErrorCode::InvalidValue, "/inputs/0/dtype");
    let mut doc = mlp();
    doc["params"][0]["init"]["kind"] = json!("orthogonal");
    assert_only(&doc, ErrorCode::InvalidValue, "/params/0/init/kind");
    let mut doc = base();
    doc["dims"]["batch"] = json!({ "min": 0 });
    assert_only(&doc, ErrorCode::InvalidValue, "/dims/batch/min");
    let mut doc = base();
    doc["dims"]["batch"] = json!({ "min": 4, "max": 3 });
    assert_only(&doc, ErrorCode::InvalidValue, "/dims/batch/max");
    let mut doc = base();
    doc["inputs"][0]["shape"][0] = json!("batch * 2");
    assert_only(&doc, ErrorCode::InvalidValue, "/inputs/0/shape/0");
    let mut doc = mlp();
    doc["params"][0]["shape"][0] = json!("batch");
    assert_only(&doc, ErrorCode::InvalidValue, "/params/0/shape/0");
    let mut doc = rep();
    doc["state"][0]["init"] = json!("ones");
    assert_only(&doc, ErrorCode::InvalidValue, "/state/0/init");
    let mut doc = base();
    doc["inputs"][0]["shape"] = json!([]);
    assert_only(&doc, ErrorCode::InvalidValue, "/inputs/0/shape");
    let mut doc = base();
    doc["outputs"] = json!([]);
    assert_only(&doc, ErrorCode::InvalidValue, "/outputs");
    // Literal dims are at most 2^32.
    let mut doc = base();
    doc["inputs"][0]["shape"][1] = json!(1u64 << 32);
    assert_valid(&doc);
    doc["inputs"][0]["shape"][1] = json!((1u64 << 32) + 1);
    assert_only(&doc, ErrorCode::InvalidValue, "/inputs/0/shape/1");
}

#[test]
fn invalid_name() {
    let mut doc = base();
    doc["nodes"][0]["id"] = json!("a:b");
    doc["outputs"][0]["value"] = json!("a");
    assert_error(&doc, ErrorCode::InvalidName, "/nodes/0/id");
    let mut doc = base();
    doc["dims"] = json!({ "1x": {} });
    doc["inputs"][0]["shape"] = json!([4]);
    assert_only(&doc, ErrorCode::InvalidName, "/dims/1x");
    let mut doc = base();
    doc["dims"] = json!({ "max": {} });
    doc["inputs"][0]["shape"] = json!([4]);
    assert_only(&doc, ErrorCode::InvalidName, "/dims/max");
    let mut doc = mlp();
    doc["params"][0]["name"] = json!("fc1 weight");
    assert_error(&doc, ErrorCode::InvalidName, "/params/0/name");
    let mut doc = rep();
    doc["nodes"][0]["body"]["params_prefix"] = json!("layers.{j}.");
    assert_only(&doc, ErrorCode::InvalidName, "/nodes/0/body/params_prefix");
    let mut doc = base();
    doc["outputs"][0]["name"] = json!("");
    assert_only(&doc, ErrorCode::InvalidName, "/outputs/0/name");
}

#[test]
fn name_length_limits() {
    let mut doc = base();
    let id = "n".repeat(128);
    doc["nodes"][0]["id"] = json!(id);
    doc["outputs"][0]["value"] = json!(id);
    doc["outputs"][0]["name"] = json!("o".repeat(128));
    doc["inputs"][0]["name"] = json!("i".repeat(128));
    doc["nodes"][0]["inputs"]["x"] = json!("i".repeat(128));
    assert_valid(&doc);
    let id = "n".repeat(129);
    doc["nodes"][0]["id"] = json!(id);
    doc["outputs"][0]["value"] = json!(id);
    assert_error(&doc, ErrorCode::InvalidName, "/nodes/0/id");
    // Param names: 256 bytes.
    let mut doc = mlp();
    let name = "p".repeat(256);
    doc["params"][0]["name"] = json!(name);
    doc["nodes"][0]["params"]["weight"] = json!(name);
    assert_valid(&doc);
    let name = "p".repeat(257);
    doc["params"][0]["name"] = json!(name);
    doc["nodes"][0]["params"]["weight"] = json!(name);
    assert_error(&doc, ErrorCode::InvalidName, "/params/0/name");
    // Dim symbols: 64 bytes.
    let mut doc = base();
    doc["dims"] = json!({ "s".repeat(64): {} });
    doc["inputs"][0]["shape"][0] = json!("s".repeat(64));
    assert_valid(&doc);
    doc["dims"] = json!({ "s".repeat(65): {} });
    doc["inputs"][0]["shape"][0] = json!("s".repeat(65));
    assert_error(
        &doc,
        ErrorCode::InvalidName,
        &format!("/dims/{}", "s".repeat(65)),
    );
    // Expanded state names: 128 bytes with the largest index.
    let body = |prefix: &str| {
        json!({ "state_prefix": prefix, "state": [ { "name": "k", "dtype": "f32", "shape": [1] } ],
                "nodes": [], "carry_out": { "h": "h" } })
    };
    let ok = format!("{}{{i}}", "s".repeat(126)); // "s"*126 + "9" + "k" = 128 bytes
    assert_valid(&with_repeat(10, body(&ok)));
    assert_only(
        &with_repeat(11, body(&ok)),
        ErrorCode::InvalidName,
        "/nodes/0/body/state/0/name",
    );
}

#[test]
fn duplicate_name() {
    let mut doc = base();
    doc["nodes"][0]["id"] = json!("x");
    doc["nodes"][0]["inputs"]["x"] = json!("x");
    doc["outputs"][0]["value"] = json!("x");
    assert_error(&doc, ErrorCode::DuplicateName, "/nodes/0/id");
    let mut doc = mlp();
    doc["params"][1]["name"] = json!("fc1.weight");
    doc["nodes"][0]["params"]
        .as_object_mut()
        .unwrap()
        .remove("bias");
    assert_only(&doc, ErrorCode::DuplicateName, "/params/1/name");
    let mut doc = base();
    doc["outputs"] = json!([ { "name": "y", "value": "a" }, { "name": "y", "value": "x" } ]);
    assert_only(&doc, ErrorCode::DuplicateName, "/outputs/1/name");
    // Body names must not reuse a name of the enclosing scope.
    let mut doc = rep();
    doc["nodes"][0]["body"]["nodes"][0]["id"] = json!("head");
    doc["nodes"][0]["body"]["nodes"][1]["inputs"]["x"] = json!("head");
    assert_error(&doc, ErrorCode::DuplicateName, "/nodes/0/body/nodes/0/id");
    // Expanded state names must be unique.
    let mut doc = rep();
    doc["state"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "name": "layers.1.cache", "dtype": "f32", "shape": [1] }));
    assert_only(&doc, ErrorCode::DuplicateName, "/nodes/0/body/state/0/name");
}

#[test]
fn unknown_reference() {
    let mut doc = base();
    doc["nodes"][0]["inputs"]["x"] = json!("y");
    assert_only(&doc, ErrorCode::UnknownReference, "/nodes/0/inputs/x");
    let mut doc = base();
    doc["outputs"][0]["value"] = json!("a:values");
    assert_only(&doc, ErrorCode::UnknownReference, "/outputs/0/value");
    let mut doc = base();
    doc["nodes"][0]["inputs"]["x"] = json!("x:out");
    assert_only(&doc, ErrorCode::UnknownReference, "/nodes/0/inputs/x");
    let mut doc = mlp();
    doc["nodes"][0]["params"]["weight"] = json!("fc9.weight");
    assert_only(&doc, ErrorCode::UnknownReference, "/nodes/0/params/weight");
    let mut doc = base();
    doc["inputs"][0]["shape"][0] = json!("seq");
    assert_only(&doc, ErrorCode::UnknownReference, "/inputs/0/shape/0");
    let mut doc = rep();
    doc["outputs"][1]["value"] = json!("split:2");
    assert_only(&doc, ErrorCode::UnknownReference, "/outputs/1/value");
    let mut doc = rep();
    doc["outputs"][1]["value"] = json!("layers:g");
    assert_only(&doc, ErrorCode::UnknownReference, "/outputs/1/value");
    // Body nodes are not visible outside the body.
    let mut doc = rep();
    doc["outputs"][1]["value"] = json!("fc");
    assert_only(&doc, ErrorCode::UnknownReference, "/outputs/1/value");
    // Every iteration's param must exist.
    let mut doc = rep();
    doc["params"].as_array_mut().unwrap().remove(4); // layers.1.fc.weight
    assert_only(
        &doc,
        ErrorCode::UnknownReference,
        "/nodes/0/body/nodes/1/params/weight",
    );
    // State update targets.
    let mut doc = rep();
    doc["state_updates"] = json!({ "nope": "next_step" });
    assert_only(&doc, ErrorCode::UnknownReference, "/state_updates/nope");
    let mut doc = rep();
    doc["nodes"][0]["body"]["state_updates"] = json!({ "step": "res" });
    assert_only(
        &doc,
        ErrorCode::UnknownReference,
        "/nodes/0/body/state_updates/step",
    );
    let report = load(doc.to_string().as_bytes(), &opts()).report;
    assert_eq!(report.diagnostics[0].node_id.as_deref(), Some("layers"));
}

#[test]
fn ambiguous_output() {
    let mut doc = rep();
    doc["outputs"][1]["value"] = json!("split");
    assert_only(&doc, ErrorCode::AmbiguousOutput, "/outputs/1/value");
    let mut doc = rep();
    doc["outputs"][1]["value"] = json!("layers");
    assert_only(&doc, ErrorCode::AmbiguousOutput, "/outputs/1/value");
}

#[test]
fn cycle() {
    let mut doc = base();
    doc["nodes"] = json!([
        { "id": "a", "op": "nn.relu", "inputs": { "x": "b" } },
        { "id": "b", "op": "nn.relu", "inputs": { "x": "a" } }
    ]);
    assert_only(&doc, ErrorCode::Cycle, "/nodes/0");
    let mut doc = base();
    doc["nodes"][0]["inputs"]["x"] = json!("a");
    assert_only(&doc, ErrorCode::Cycle, "/nodes/0");
    // Nodes may be listed in any order.
    let mut doc = base();
    doc["nodes"] = json!([
        { "id": "b", "op": "nn.relu", "inputs": { "x": "a" } },
        { "id": "a", "op": "nn.relu", "inputs": { "x": "x" } }
    ]);
    doc["outputs"][0]["value"] = json!("b");
    assert_valid(&doc);
    // A body may not use an enclosing node that depends on the repeat.
    let mut doc = rep();
    doc["nodes"][0]["body"]["nodes"][2]["inputs"]["b"] = json!("head");
    assert_only(&doc, ErrorCode::Cycle, "/nodes/0");
    // A cycle inside a body.
    let mut doc = rep();
    doc["nodes"][0]["body"]["nodes"][0]["inputs"]["x"] = json!("res");
    assert_only(&doc, ErrorCode::Cycle, "/nodes/0/body/nodes/0");
}

#[test]
fn unknown_op() {
    let mut doc = base();
    doc["nodes"][0]["op"] = json!("nn.swish");
    assert_only(&doc, ErrorCode::UnknownOp, "/nodes/0/op");
    let mut doc = base();
    doc["opsets"] = json!({ "core": 1 });
    assert_only(&doc, ErrorCode::UnknownOp, "/nodes/0/op");
    let mut doc = base();
    doc["nodes"][0]["op"] = json!("relu");
    assert_only(&doc, ErrorCode::UnknownOp, "/nodes/0/op");
    let mut doc = base();
    doc["opsets"]["transformer"] = json!(1);
    doc["nodes"][0]["op"] = json!("transformer.rope");
    assert_only(&doc, ErrorCode::UnsupportedOpset, "/opsets/transformer");
}

#[test]
fn unsupported_in_version() {
    let mut doc = base();
    doc["functions"] = json!({ "f": {} });
    assert_only(&doc, ErrorCode::UnsupportedInVersion, "/functions");
    let mut doc = base();
    doc["nodes"][0] = json!({ "id": "a", "op": "call", "function": "f", "inputs": { "x": "x" } });
    assert_only(&doc, ErrorCode::UnsupportedInVersion, "/nodes/0/op");
    let mut doc = mlp();
    doc["params"][0]["dtype"] = json!("f16");
    assert_only(&doc, ErrorCode::UnsupportedInVersion, "/params/0/dtype");
}

#[test]
fn rank_too_large() {
    let mut doc = base();
    doc["inputs"][0]["shape"] = json!(["batch", 1, 1, 1, 1, 4]);
    assert_valid(&doc);
    doc["inputs"][0]["shape"] = json!(["batch", 1, 1, 1, 1, 1, 4]);
    assert_only(&doc, ErrorCode::RankTooLarge, "/inputs/0/shape");
}

#[test]
fn dim_errors() {
    let state = |shape: serde_json::Value| {
        let mut doc = base();
        doc["dims"]["seq"] = json!({});
        doc["state"] = json!([ { "name": "s", "dtype": "f32", "shape": shape } ]);
        doc
    };
    assert_valid(&state(json!(["batch", "seq", "batch * seq + 1"])));
    assert_only(
        &state(json!(["batch", "seq", "batch +"])),
        ErrorCode::InvalidDimExpr,
        "/state/0/shape/2",
    );
    assert_only(
        &state(json!(["batch", "seq", "nope + 1"])),
        ErrorCode::InvalidDimExpr,
        "/state/0/shape/2",
    );
    assert_only(
        &state(json!(["batch", "seq", "seq // batch"])),
        ErrorCode::InvalidDimExpr,
        "/state/0/shape/2",
    );
    assert_only(
        &state(json!(["batch", "nope"])),
        ErrorCode::UnknownReference,
        "/state/0/shape/1",
    );
    assert_only(
        &state(json!([
            "batch",
            "seq",
            "9223372036854775807 + seq * 9223372036854775807 * 2"
        ])),
        ErrorCode::DimOverflow,
        "/state/0/shape/2",
    );
    assert_only(
        &state(json!(["batch", "seq", "seq - seq"])),
        ErrorCode::NonpositiveDim,
        "/state/0/shape/2",
    );
    assert_only(
        &state(json!(["batch", 0])),
        ErrorCode::NonpositiveDim,
        "/state/0/shape/1",
    );
    let mut doc = base();
    doc["inputs"][0]["shape"][1] = json!(-3);
    assert_only(&doc, ErrorCode::NonpositiveDim, "/inputs/0/shape/1");
    // Expression limits apply: length, depth, terms, degree.
    let long = format!("seq{}", " ".repeat(254));
    assert_only(
        &state(json!(["batch", "seq", long])),
        ErrorCode::InvalidDimExpr,
        "/state/0/shape/2",
    );
    let deep = format!("{}seq{}", "(".repeat(17), ")".repeat(17));
    assert_only(
        &state(json!(["batch", "seq", deep])),
        ErrorCode::InvalidDimExpr,
        "/state/0/shape/2",
    );
    let degree = ["seq"; 9].join("*");
    assert_only(
        &state(json!(["batch", "seq", degree])),
        ErrorCode::InvalidDimExpr,
        "/state/0/shape/2",
    );
    // (batch + seq + c + d + 1)^4 has 70 terms, more than 64.
    let mut doc = state(json!(["batch", "seq", "c", "d", 1]));
    doc["dims"]["c"] = json!({});
    doc["dims"]["d"] = json!({});
    assert_valid(&doc);
    let sum = "(batch + seq + c + d + 1)";
    doc["state"][0]["shape"][4] = json!(format!("{sum}*{sum}*{sum}"));
    assert_valid(&doc);
    doc["state"][0]["shape"][4] = json!(format!("{sum}*{sum}*{sum}*{sum}"));
    assert_only(&doc, ErrorCode::InvalidDimExpr, "/state/0/shape/4");
}

#[test]
fn unbound_dim() {
    let mut doc = base();
    doc["dims"]["seq"] = json!({});
    doc["state"] = json!([ { "name": "s", "dtype": "f32", "shape": ["batch", "seq + 1"] } ]);
    assert_only(&doc, ErrorCode::UnboundDim, "/state/0/shape/1");
    // Bound by a plain dim of a state shape.
    doc["state"] = json!([
        { "name": "s", "dtype": "f32", "shape": ["batch", "seq + 1"] },
        { "name": "t", "dtype": "f32", "shape": ["seq"] }
    ]);
    assert_valid(&doc);
    // Declared but unused is fine.
    let mut doc = base();
    doc["dims"]["unused"] = json!({});
    assert_valid(&doc);
}

#[test]
fn repeat_carry_mismatch() {
    let mut doc = rep();
    doc["nodes"][0]["body"]["carry_out"] = json!({ "g": "res" });
    let errs = errors(doc.to_string().as_bytes());
    assert_eq!(
        errs,
        vec![
            (
                "repeat_carry_mismatch".into(),
                "/nodes/0/body/carry_out/g".into()
            ),
            (
                "repeat_carry_mismatch".into(),
                "/nodes/0/body/carry_out".into()
            ),
        ]
    );
}

#[test]
fn invalid_repeat() {
    let body = json!({ "nodes": [], "carry_out": { "h": "h" } });
    assert_valid(&with_repeat(1, body.clone()));
    assert_valid(&with_repeat(65_536, body.clone()));
    assert_only(
        &with_repeat(0, body.clone()),
        ErrorCode::InvalidRepeat,
        "/nodes/0/count",
    );
    assert_only(
        &with_repeat(65_537, body.clone()),
        ErrorCode::InvalidRepeat,
        "/nodes/0/count",
    );
    let mut doc = with_repeat(2, json!({ "nodes": [], "carry_out": {} }));
    doc["nodes"][0]["carry"] = json!({});
    doc["outputs"][0]["value"] = json!("x");
    assert_only(&doc, ErrorCode::InvalidRepeat, "/nodes/0/carry");
    let mut doc = rep();
    doc["nodes"][0]["body"]["state_prefix"] = json!("layers.");
    assert_only(&doc, ErrorCode::InvalidRepeat, "/nodes/0/body/state_prefix");
    let mut doc = rep();
    doc["nodes"][0]["body"]
        .as_object_mut()
        .unwrap()
        .remove("state_prefix");
    assert_only(&doc, ErrorCode::InvalidRepeat, "/nodes/0/body/state");
}

/// Nested repeats: `n_levels` levels of `count` each, innermost body one relu.
fn nested(n_levels: usize, count: u64) -> serde_json::Value {
    let mut body = json!({ "nodes": [ { "id": "n", "op": "nn.relu", "inputs": { "x": format!("h{}", n_levels - 1) } } ],
                           "carry_out": { format!("h{}", n_levels - 1): "n" } });
    for level in (0..n_levels).rev() {
        let carry_in = if level == 0 {
            "x".to_owned()
        } else {
            format!("h{}", level - 1)
        };
        let node = json!({ "id": format!("r{level}"), "op": "repeat", "count": count,
                           "carry": { format!("h{level}"): carry_in }, "body": body });
        if level == 0 {
            let mut g = base();
            g["nodes"] = json!([node]);
            g["outputs"] = json!([ { "name": "y", "value": "r0:h0" } ]);
            return g;
        }
        body = json!({ "nodes": [node], "carry_out": { format!("h{}", level - 1): format!("r{level}:h{level}") } });
    }
    unreachable!()
}

#[test]
fn nested_repeats_and_prefixes() {
    assert_valid(&nested(2, 3));
    // Prefixes concatenate, outer first.
    let mut doc = nested(2, 2);
    doc["nodes"][0]["body"]["params_prefix"] = json!("a{i}.");
    doc["nodes"][0]["body"]["nodes"][0]["body"]["params_prefix"] = json!("b{i}.");
    doc["nodes"][0]["body"]["nodes"][0]["body"]["nodes"][0] = json!({ "id": "n", "op": "nn.linear", "inputs": { "x": "h1" }, "params": { "weight": "w" } });
    let params: Vec<_> = ["a0.b0.w", "a0.b1.w", "a1.b0.w", "a1.b1.w"]
        .iter()
        .map(|n| json!({ "name": n, "dtype": "float", "shape": [4, 4] }))
        .collect();
    doc["params"] = json!(params);
    assert_valid(&doc);
    doc["params"].as_array_mut().unwrap().remove(2);
    let report = load(doc.to_string().as_bytes(), &opts()).report;
    assert_eq!(report.diagnostics.len(), 1, "{report}");
    assert!(
        report.diagnostics[0].message.contains("a1.b0.w"),
        "{report}"
    );
}

#[test]
fn repeat_too_deep() {
    assert_valid(&nested(8, 1));
    let pointer = format!("/nodes/0{}/body", "/body/nodes/0".repeat(8));
    assert_only(&nested(9, 1), ErrorCode::RepeatTooDeep, &pointer);
}

#[test]
fn too_many_nodes() {
    // 4 levels of 32 = 1,048,576 nodes, computed without expanding.
    let start = std::time::Instant::now();
    assert_only(&nested(4, 32), ErrorCode::TooManyNodes, "/nodes");
    // 8 levels of 65,536 would overflow u64: still an error, not a panic.
    assert_only(&nested(8, 65_536), ErrorCode::TooManyNodes, "/nodes");
    assert!(start.elapsed().as_secs() < 5);
    // 1,000,000 exactly is fine (10^6 = 100^3).
    assert_valid(&nested(3, 100));
}

#[test]
fn too_many_param_bindings() {
    // 4 params per node would be fine; 5 × 1,000,000 isn't.
    let mut doc = nested(3, 100);
    let inner = &mut doc["nodes"][0]["body"]["nodes"][0]["body"]["nodes"][0]["body"]["nodes"][0];
    inner["params"] = json!({ "a": "p", "b": "p", "c": "p", "d": "p", "e": "p" });
    doc["params"] = json!([ { "name": "p", "dtype": "float", "shape": [4] } ]);
    assert_only(&doc, ErrorCode::TooManyNodes, "/nodes");
}

#[test]
fn too_many_elements() {
    let mut doc = mlp();
    doc["params"][0]["shape"] = json!([1u64 << 32, 1u64 << 32, 1u64 << 32]);
    assert_only(&doc, ErrorCode::TooManyElements, "/params/0/shape");
    let mut doc = mlp();
    let big = json!([1u64 << 32, 1u64 << 31]); // 2^63 elements each
    doc["params"][0]["shape"] = big.clone();
    doc["params"][1]["shape"] = big;
    assert_only(&doc, ErrorCode::TooManyElements, "/params");
}

#[test]
fn list_limits() {
    let io = |n: usize| -> Vec<serde_json::Value> {
        (0..n)
            .map(|i| json!({ "name": format!("i{i}"), "dtype": "f32", "shape": [1] }))
            .collect()
    };
    let mut doc = base();
    let mut inputs = io(4095);
    inputs.push(json!({ "name": "x", "dtype": "float", "shape": ["batch", 4] }));
    doc["inputs"] = json!(inputs);
    assert_valid(&doc);
    doc["inputs"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "name": "z", "dtype": "f32", "shape": [1] }));
    assert_only(&doc, ErrorCode::InvalidValue, "/inputs");

    let mut doc = base();
    let outputs: Vec<_> = (0..4097)
        .map(|i| json!({ "name": format!("o{i}"), "value": "a" }))
        .collect();
    doc["outputs"] = json!(outputs);
    assert_only(&doc, ErrorCode::InvalidValue, "/outputs");

    let mut doc = base();
    doc["state"] = json!(io(4097));
    assert_only(&doc, ErrorCode::InvalidValue, "/state");

    // Expanded body state counts too: 4096 root-level entries in all.
    let body = json!({ "state_prefix": "l{i}.", "state": [ { "name": "k", "dtype": "f32", "shape": [1] } ],
                       "nodes": [], "carry_out": { "h": "h" } });
    assert_valid(&with_repeat(4096, body.clone()));
    assert_only(&with_repeat(4097, body), ErrorCode::InvalidValue, "/state");

    let mut doc = base();
    let dims: serde_json::Map<_, _> = (0..257).map(|i| (format!("d{i}"), json!({}))).collect();
    doc["dims"] = serde_json::Value::Object(dims);
    assert_only(&doc, ErrorCode::InvalidValue, "/dims");
}

// ---------------------------------------------------------------------------------------------
// Fuzz-style robustness

/// xorshift64*: deterministic, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn validation_never_panics() {
    const SNIPPETS: &[&str] = &[
        "{",
        "}",
        "[",
        "]",
        "\"",
        ":",
        ",",
        "null",
        "-1",
        "0",
        "1e999",
        "\"repeat\"",
        "\"x:out\"",
        "\"{i}\"",
        "\"batch * seq\"",
        "\"((\"",
        "65537",
        "\"float\"",
        "true",
        "\\u0000",
        "\"min(\"",
        "\"layers:h\"",
        "\"\"",
        "9223372036854775808",
        "\"core.split\"",
    ];
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    let start = std::time::Instant::now();
    let docs = [MLP.as_bytes().to_vec(), REPEAT.as_bytes().to_vec()];
    let mut runs = 0;
    for doc in &docs {
        // Every truncation point (stepping to keep the time bounded).
        for cut in (0..doc.len()).step_by(7) {
            let _ = load(&doc[..cut], &opts());
            runs += 1;
        }
    }
    for _ in 0..3000 {
        let mut d = docs[rng.below(2)].clone();
        for _ in 0..1 + rng.below(4) {
            let pos = rng.below(d.len() + 1);
            match rng.below(4) {
                0 if !d.is_empty() => {
                    let end = (pos + 1 + rng.below(16)).min(d.len());
                    d.drain(pos.min(end)..end);
                }
                1 => {
                    let s = SNIPPETS[rng.below(SNIPPETS.len())].as_bytes();
                    d.splice(pos..pos, s.iter().copied());
                }
                2 if pos < d.len() => d[pos] = (rng.next() & 0xff) as u8,
                _ => {
                    // Duplicate a slice somewhere else.
                    let a = rng.below(d.len() + 1);
                    let b = (a + rng.below(64)).min(d.len());
                    let piece = d[a..b].to_vec();
                    d.splice(pos..pos, piece);
                }
            }
        }
        let out = load(&d, &opts());
        if let Some(g) = out.graph {
            // A converted graph saves and reloads to the same graph.
            let again = load(g.to_json().as_bytes(), &opts());
            assert_eq!(again.graph.as_ref(), Some(&g));
        }
        runs += 1;
    }
    assert!(runs > 3000);
    assert!(
        start.elapsed().as_secs() < 60,
        "fuzz test took {:?}",
        start.elapsed()
    );
}

#[test]
fn dim_parser_never_panics() {
    const TOKENS: &[&str] = &[
        "a",
        "b",
        "seq",
        "1",
        "0",
        "2",
        "9223372036854775807",
        "+",
        "-",
        "*",
        "//",
        "%",
        "(",
        ")",
        ",",
        "min",
        "max",
        " ",
        "/",
        "x",
        "_",
        "65",
    ];
    let mut rng = Rng(42);
    for _ in 0..20_000 {
        let n = 1 + rng.below(24);
        let s: String = (0..n).map(|_| TOKENS[rng.below(TOKENS.len())]).collect();
        if let Ok(e) = polypous_ir::DimExpr::parse(&s, &|_| true) {
            // The canonical string parses back to the same normal form.
            // The canonical string parses back to the same normal form, unless it is longer or
            // nests deeper than the input limits allow.
            match polypous_ir::DimExpr::parse(&e.to_string(), &|_| true) {
                Ok(back) => assert_eq!(back, e, "{s} -> {e}"),
                Err(err) => assert!(
                    matches!(
                        err,
                        polypous_ir::DimError::TooLong { .. }
                            | polypous_ir::DimError::TooDeep { .. }
                    ),
                    "{s} -> {e}: {err}"
                ),
            }
        }
    }
}
