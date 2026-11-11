//! The op catalog against the spec tables (`docs/specs/graph-ir.md` §3.2, §3.3) and against the op
//! fixtures (`fixtures/small/ops`, testing-strategy.md §3.2).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use std::path::{Path, PathBuf};

use polypous_ir::catalog::{self, InDtype, OpDef, Outputs, Presence};
use polypous_ir::{ComputeDtype, Dtype, NodeInfo, ValidateOptions};
use serde_json::{Value, json};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// One row of an op table: the cells after the op name, keyed by column header.
struct Row {
    op: String,
    cells: Vec<(String, String)>,
}

impl Row {
    fn cell(&self, header: &str) -> &str {
        self.cells
            .iter()
            .find(|(h, _)| h == header)
            .map_or("—", |(_, c)| c.as_str())
    }
}

/// The rows of every table in §3.2 and §3.3 whose first cell is a backticked op name.
fn spec_rows() -> Vec<Row> {
    let text = std::fs::read_to_string(repo_root().join("docs/specs/graph-ir.md")).unwrap();
    let start = text.find("\n### 3.2 ").expect("§3.2");
    let end = text.find("\n### 3.4 ").expect("§3.4");
    let mut rows = Vec::new();
    let mut headers: Vec<String> = Vec::new();
    for line in text[start..end].lines() {
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .trim()
            .trim_matches('|')
            .split(" | ")
            .map(|c| c.trim().to_owned())
            .collect();
        if cells[0] == "Op" {
            headers = cells;
            continue;
        }
        if !cells[0].starts_with('`') {
            continue;
        }
        assert_eq!(cells.len(), headers.len(), "{line}");
        rows.push(Row {
            op: cells[0].trim_matches('`').to_owned(),
            cells: headers.iter().cloned().zip(cells).collect(),
        });
    }
    rows
}

/// Backticked spans of a cell.
fn spans(cell: &str) -> Vec<&str> {
    cell.split('`').skip(1).step_by(2).collect()
}

/// `(name, class)` pairs of the Inputs column: `` `a`: `N`, `b`: `T` ``.
fn spec_inputs(cell: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let parts: Vec<&str> = cell.split('`').collect();
    // parts: text, span, text, span, …; an input is a span followed by ": " and a class span.
    let mut i = 1;
    while i + 2 < parts.len() {
        if parts[i + 1] == ": " && parts[i + 2].len() == 1 {
            out.push((parts[i].to_owned(), parts[i + 2].to_owned()));
            i += 4;
        } else {
            i += 2;
        }
    }
    out
}

/// `(name, optional)` of the Params column: `` `weight` `[out, in]`, `bias?` `[out]` ``.
fn spec_params(cell: &str) -> Vec<(String, bool)> {
    if cell == "—" {
        return Vec::new();
    }
    let parts: Vec<&str> = cell.split('`').collect();
    let mut out = Vec::new();
    let mut i = 1;
    while i + 2 < parts.len() {
        if parts[i + 1] == " " {
            let name = parts[i];
            out.push((name.trim_end_matches('?').to_owned(), name.ends_with('?')));
            i += 4;
        } else {
            i += 2;
        }
    }
    out
}

/// `(name, type, presence)` of the Attributes column: `` `start_dim: int = 0` ``,
/// `` `dim: int[]?` ``, `` `shape: dim[]` ``.
fn spec_attrs(cell: &str) -> Vec<(String, String, String)> {
    spans(cell)
        .into_iter()
        .filter_map(|s| {
            let (name, rest) = s.split_once(": ")?;
            if !name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
            {
                return None;
            }
            let (ty, presence) = match rest.split_once(" = ") {
                Some((ty, default)) => (ty.to_owned(), format!("= {default}")),
                None if rest.ends_with('?') => (rest.trim_end_matches('?').to_owned(), "?".into()),
                None => (rest.to_owned(), "required".into()),
            };
            Some((name.to_owned(), ty, presence))
        })
        .collect()
}

fn catalog_inputs(op: &OpDef) -> Vec<(String, String)> {
    op.inputs
        .iter()
        .map(|i| (i.name.to_owned(), i.dtype.letter().to_owned()))
        .collect()
}

fn catalog_attrs(op: &OpDef) -> Vec<(String, String, String)> {
    op.attrs
        .iter()
        .map(|a| {
            let presence = match a.presence {
                Presence::Required => "required".to_owned(),
                Presence::Optional => "?".to_owned(),
                Presence::Default(d) => format!("= {d}"),
            };
            (a.name.to_owned(), a.ty.spec_type(), presence)
        })
        .collect()
}

fn catalog_outputs(op: &OpDef) -> String {
    match op.outputs {
        Outputs::Fixed(outs) => outs
            .iter()
            .map(|o| format!("`{}`: `{}`", o.name, o.dtype.spec()))
            .collect::<Vec<_>>()
            .join(", "),
        Outputs::Split => "`0`…`n-1`: `T`".to_owned(),
    }
}

#[test]
fn catalog_matches_spec_tables() {
    let rows = spec_rows();
    let spec_names: Vec<&str> = rows.iter().map(|r| r.op.as_str()).collect();
    let catalog_names: Vec<&str> = catalog::OPS.iter().map(|o| o.name).collect();
    assert_eq!(
        catalog_names, spec_names,
        "the catalog and the spec tables list different ops (or a different order)"
    );
    for row in &rows {
        let op = catalog::lookup(&row.op).unwrap();
        assert_eq!(
            catalog_inputs(op),
            spec_inputs(row.cell("Inputs")),
            "{}: inputs",
            row.op
        );
        let params: Vec<(String, bool)> = op
            .params
            .iter()
            .map(|p| (p.name.to_owned(), p.optional))
            .collect();
        assert_eq!(
            params,
            spec_params(row.cell("Params")),
            "{}: params",
            row.op
        );
        let shapes: Vec<&str> = op.params.iter().map(|p| p.shape).collect();
        let spec_shapes: Vec<&str> = spans(row.cell("Params"))
            .into_iter()
            .skip(1)
            .step_by(2)
            .collect();
        assert_eq!(shapes, spec_shapes, "{}: param shapes", row.op);
        assert_eq!(
            catalog_attrs(op),
            spec_attrs(row.cell("Attributes")),
            "{}: attributes",
            row.op
        );
        assert_eq!(
            catalog_outputs(op),
            row.cell("Outputs"),
            "{}: outputs",
            row.op
        );
        assert_eq!(
            op.differentiable.spec(),
            row.cell("Diff."),
            "{}: differentiable",
            row.op
        );
    }
}

#[test]
fn spec_row_parsers() {
    assert_eq!(
        spec_inputs("`cond`: `B`, `a`: `A`, `b`: `T`"),
        vec![
            ("cond".into(), "B".into()),
            ("a".into(), "A".into()),
            ("b".into(), "T".into())
        ]
    );
    assert_eq!(
        spec_params("`weight` `[out, in]`, `bias?` `[out]`"),
        vec![("weight".into(), false), ("bias".into(), true)]
    );
    assert_eq!(
        spec_attrs("`dim: int[]?` (non-empty), `keepdim: bool = false`"),
        vec![
            ("dim".into(), "int[]".into(), "?".into()),
            ("keepdim".into(), "bool".into(), "= false".into())
        ]
    );
    // Catalog letters: `C` for compute-dtype inputs, `T` for same-as.
    assert_eq!(InDtype::Compute.letter(), "C");
}

// ---- Fixtures ----

fn dtype_of(s: &str) -> Dtype {
    Dtype::parse(s).unwrap_or_else(|| panic!("unknown dtype {s}"))
}

/// A one-node graph from a sidecar: graph inputs named like the op inputs, graph params from the
/// sidecar's `params` (testing-strategy.md §3.2), the sidecar's attributes, one graph output per
/// op output.
fn graph_for(sidecar: &Value) -> (Value, ValidateOptions) {
    let op_name = sidecar["op"].as_str().unwrap();
    let op = catalog::lookup(op_name).unwrap_or_else(|| panic!("{op_name} is not in the catalog"));
    let mut inputs = Vec::new();
    let mut params = Vec::new();
    let mut node_inputs = serde_json::Map::new();
    let mut node_params = serde_json::Map::new();
    let mut variadic = Vec::new();
    let mut compute = None;
    for t in sidecar["inputs"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap();
        let dtype = dtype_of(t["dtype"].as_str().unwrap());
        if op.param(name).is_some() {
            params.push(json!({ "name": name, "dtype": "float", "shape": t["shape"] }));
            node_params.insert(name.into(), json!(name));
            compute.get_or_insert(dtype);
            continue;
        }
        inputs.push(json!({ "name": name, "dtype": t["dtype"], "shape": t["shape"] }));
        match op.inputs.iter().find(|i| i.name == name) {
            Some(i) if i.variadic => variadic.push(json!(name)),
            Some(i) => {
                if i.dtype == InDtype::Compute {
                    compute.get_or_insert(dtype);
                }
                node_inputs.insert(name.into(), json!(name));
            }
            None if op.inputs.iter().any(|i| i.variadic) => variadic.push(json!(name)),
            None => panic!(
                "{op_name}/{}: sidecar input `{name}` is not an input or param of the catalog entry",
                sidecar["case"]
            ),
        }
    }
    for t in sidecar
        .get("params")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = t["name"].as_str().unwrap();
        assert!(
            op.param(name).is_some(),
            "{op_name}/{}: sidecar param `{name}` is not a param of the catalog entry",
            sidecar["case"]
        );
        params.push(json!({ "name": name, "dtype": "float", "shape": t["shape"] }));
        node_params.insert(name.into(), json!(name));
        compute.get_or_insert(dtype_of(t["dtype"].as_str().unwrap()));
    }
    if let Some(v) = op.inputs.iter().find(|i| i.variadic) {
        node_inputs.insert(v.name.into(), json!(variadic));
    }
    let outputs: Vec<Value> = sidecar["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(k, o)| {
            let port = match op.outputs {
                Outputs::Fixed(outs) => outs[k].name.to_owned(),
                Outputs::Split => k.to_string(),
            };
            json!({ "name": o["name"], "value": format!("n:{port}") })
        })
        .collect();
    let opset = op.opset();
    let graph = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { opset: 1 },
        "inputs": inputs, "outputs": outputs, "params": params,
        "nodes": [ { "id": "n", "op": op_name, "inputs": node_inputs, "params": node_params,
                     "attrs": sidecar["attributes"] } ]
    });
    let compute_dtype = match compute {
        Some(Dtype::F16) => ComputeDtype::F16,
        Some(Dtype::Bf16) => ComputeDtype::Bf16,
        _ => ComputeDtype::F32,
    };
    (graph, ValidateOptions { compute_dtype })
}

/// The `params` key of a sidecar (testing-strategy.md §3.2): present exactly for ops with params,
/// listing the params that are present in catalog order, every required one included, each in the
/// dtype of the case's float input (the compute dtype).
fn check_params_key(case: &str, sidecar: &Value) {
    let op = catalog::lookup(sidecar["op"].as_str().unwrap()).unwrap();
    let Some(listed) = sidecar.get("params") else {
        assert!(
            op.params.is_empty(),
            "{case}: no `params` key, but the op has params"
        );
        return;
    };
    assert!(
        !op.params.is_empty(),
        "{case}: a `params` key, but the op has none"
    );
    let names: Vec<&str> = listed
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    let expected: Vec<&str> = op
        .params
        .iter()
        .map(|p| p.name)
        .filter(|name| names.contains(name))
        .collect();
    assert_eq!(names, expected, "{case}: params not in catalog order");
    for p in op.params.iter().filter(|p| !p.optional) {
        assert!(
            names.contains(&p.name),
            "{case}: required param `{}` missing",
            p.name
        );
    }
    for p in listed.as_array().unwrap() {
        assert!(
            matches!(p["dtype"].as_str(), Some("f32" | "f16" | "bf16")),
            "{case}: param {p} is not a float tensor"
        );
    }
}

#[test]
fn catalog_matches_fixtures() {
    let dir = repo_root().join("fixtures/small/ops");
    let mut sidecars: Vec<PathBuf> = Vec::new();
    for op_dir in std::fs::read_dir(&dir).unwrap() {
        let op_dir = op_dir.unwrap().path();
        for f in std::fs::read_dir(&op_dir).unwrap() {
            let f = f.unwrap().path();
            if f.extension().is_some_and(|e| e == "json") {
                sidecars.push(f);
            }
        }
    }
    sidecars.sort();
    assert!(!sidecars.is_empty(), "no sidecars under {}", dir.display());
    let mut checked = 0;
    for path in &sidecars {
        let sidecar: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let case = format!("{}", path.display());
        check_params_key(&case, &sidecar);
        let (doc, options) = graph_for(&sidecar);
        let bytes = serde_json::to_vec(&doc).unwrap();
        let outcome = polypous_ir::load(&bytes, &options);
        assert!(outcome.report.is_ok(), "{case}:\n{}", outcome.report);
        let graph = outcome.graph.unwrap();
        let analysis = graph
            .analyze(&options)
            .unwrap_or_else(|r| panic!("{case}: {r}"));
        let NodeInfo::Op(info) = &analysis.nodes[0] else {
            panic!("{case}: not an op node");
        };
        let expected = sidecar["outputs"].as_array().unwrap();
        assert_eq!(info.outputs.len(), expected.len(), "{case}: output count");
        for ((port, got), want) in info.outputs.iter().zip(expected) {
            assert_eq!(
                got.dtype,
                dtype_of(want["dtype"].as_str().unwrap()),
                "{case}: dtype of `{port}`"
            );
            let shape: Vec<String> = want["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.to_string())
                .collect();
            assert_eq!(
                got.shape_string(),
                format!("[{}]", shape.join(", ")),
                "{case}: shape of `{port}`"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, sidecars.len());
}
