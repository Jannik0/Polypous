//! Plan compilation (`docs/specs/runtime.md` §1): validation, the device check, topological order
//! and value lifetimes.
//!
//! Every value of the graph (graph input, state entry, node output port, carry) gets a **slot**,
//! an index into the value table of a call. Each scope (the root and every `repeat` body) becomes
//! a [`Block`] of [`Step`]s in topological order. A step lists the slots to drop after it
//! ([`Step::frees`]); op operands whose last use is the step are moved into the op
//! ([`Operand::take`]).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use polypous_backend::DynTensor;
use polypous_backend::device::DeviceId;
use polypous_ir::catalog::OpDef;
use polypous_ir::model::{Dim, InputRef, Node, NodeKind};
use polypous_ir::{Analysis, ComputeDtype, Dtype, Graph, NodeInfo, ValidateOptions};

use crate::error::{Result, RuntimeError};
use crate::ops::{self, to_backend_dtype};

/// An index into the value table of a call.
pub(crate) type Slot = usize;

/// Options for [`Plan::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanOptions {
    /// The device every node runs on.
    pub device: DeviceId,
    /// The compute dtype (graph-ir.md §2.2): `float` params, inputs, state entries and `dtype`
    /// attributes resolve to it. Default `f32`.
    pub compute_dtype: ComputeDtype,
}

impl PlanOptions {
    /// Options for `device` with the default compute dtype `f32`.
    pub fn new(device: DeviceId) -> Self {
        PlanOptions {
            device,
            compute_dtype: ComputeDtype::F32,
        }
    }
}

/// A graph compiled for one device and one compute dtype (runtime.md §1). Immutable; holds no
/// weights. Load weights with [`Model::load`](crate::Model::load).
#[derive(Debug)]
pub struct Plan {
    pub(crate) device: DeviceId,
    pub(crate) analysis: Analysis,
    pub(crate) program: Program,
    /// Declared params: name and shape, in declaration order.
    pub(crate) params: Vec<(String, Vec<usize>)>,
    /// Root-level state entry name → index in `analysis.state`.
    pub(crate) state_index: HashMap<String, usize>,
}

impl Plan {
    /// Validates `graph` with the compute dtype of `options`, checks every dtype of the graph
    /// against the device, and compiles it (runtime.md §1).
    ///
    /// Errors: [`RuntimeError::InvalidGraph`] with the validation report,
    /// [`RuntimeError::UnsupportedDtype`] naming the value or node, op, dtype and device,
    /// [`RuntimeError::DeviceUnavailable`].
    pub fn new(graph: &Graph, options: PlanOptions) -> Result<Plan> {
        let validate = ValidateOptions {
            compute_dtype: options.compute_dtype,
        };
        let analysis = graph
            .analyze(&validate)
            .map_err(RuntimeError::InvalidGraph)?;
        check_device(&analysis, options.device)?;
        let program = compile(graph, &analysis)?;
        let params = graph
            .params
            .iter()
            .map(|p| Ok((p.name.clone(), literal_shape(&p.name, &p.shape)?)))
            .collect::<Result<Vec<_>>>()?;
        let state_index = analysis
            .state
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.clone(), i))
            .collect();
        tracing::debug!(
            device = %options.device,
            compute_dtype = %options.compute_dtype.dtype(),
            slots = program.slots.len(),
            "compiled graph plan"
        );
        Ok(Plan {
            device: options.device,
            analysis,
            program,
            params,
            state_index,
        })
    }

    /// The device the plan runs on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The compute dtype the graph was validated with.
    pub fn compute_dtype(&self) -> ComputeDtype {
        self.analysis.compute_dtype
    }

    /// What validation proved about the graph: dtypes and shapes of inputs, state entries
    /// (expanded) and nodes.
    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    /// The root-level steps in execution order, with the values each one drops (runtime.md §1
    /// step 4). Body steps are not listed; [`RunOptions::trace`](crate::RunOptions::trace) shows
    /// them per iteration.
    pub fn schedule(&self) -> Vec<ScheduledStep> {
        let names = |slots: &[Slot]| -> Vec<String> {
            slots
                .iter()
                .map(|&s| self.program.slots[s].clone())
                .collect()
        };
        let mut out = Vec::with_capacity(self.program.root.steps.len() + 1);
        if !self.program.root.frees_at_start.is_empty() {
            out.push(ScheduledStep {
                node: None,
                drops: names(&self.program.root.frees_at_start),
            });
        }
        for step in &self.program.root.steps {
            out.push(ScheduledStep {
                node: Some(step.id.clone()),
                drops: names(&step.frees),
            });
        }
        out
    }
}

/// One root-level step of a [`Plan::schedule`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledStep {
    /// The node id; `None` for the drops before the first node (inputs nobody reads).
    pub node: Option<String>,
    /// The values dropped after the step (body values as `repeat/value`, ports other than
    /// `out` as `node:port`).
    pub drops: Vec<String>,
}

/// The compiled graph.
#[derive(Debug)]
pub(crate) struct Program {
    /// A display name per slot, for tracing.
    pub slots: Vec<String>,
    pub root: Block,
    /// Parallel to `analysis.inputs`.
    pub inputs: Vec<Slot>,
    /// Root state entries: index in `analysis.state` and slot.
    pub root_state: Vec<(usize, Slot)>,
    /// Graph outputs: name and slot.
    pub outputs: Vec<(String, Slot)>,
    /// Root state updates: index in `analysis.state` and slot of the new value.
    pub state_updates: Vec<(usize, Slot)>,
}

/// The steps of one scope.
#[derive(Debug)]
pub(crate) struct Block {
    pub steps: Vec<Step>,
    /// Owned values defined before the first step (inputs, carries, state) that no step reads.
    pub frees_at_start: Vec<Slot>,
    /// Every slot the scope owns; a body drops them all at the end of each iteration.
    pub owned: Vec<Slot>,
}

#[derive(Debug)]
pub(crate) struct Step {
    /// Index of the node in its scope (graph, analysis and bindings).
    pub index: usize,
    pub id: String,
    pub kind: StepKind,
    /// Slots to drop after the step.
    pub frees: Vec<Slot>,
}

#[derive(Debug)]
pub(crate) enum StepKind {
    Op(OpStep),
    Repeat(Box<RepeatStep>),
}

/// A value read by a step.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Operand {
    pub slot: Slot,
    /// The step is the value's last use (and this its last occurrence in the step): move it.
    pub take: bool,
}

#[derive(Debug)]
pub(crate) enum OpInput {
    Absent,
    One(Operand),
    Many(Vec<Operand>),
}

#[derive(Debug)]
pub(crate) struct OpStep {
    pub op: &'static OpDef,
    /// Parallel to `op.inputs`.
    pub inputs: Vec<OpInput>,
    /// Parallel to `op.params`: the param name as written on the node (body-local in a body).
    pub params: Vec<Option<String>>,
    /// Parallel to the node's output ports.
    pub outputs: Vec<Slot>,
}

#[derive(Debug)]
pub(crate) struct RepeatStep {
    pub count: u64,
    pub carry: Vec<CarryStep>,
    pub body: Block,
    /// As written (may contain `{i}`; absent = "").
    pub params_prefix: String,
    pub state_prefix: String,
    /// Body state entries: local name and slot.
    pub state: Vec<(String, Slot)>,
    /// Body state updates: local name and slot of the new value.
    pub state_updates: Vec<(String, Slot)>,
}

#[derive(Debug)]
pub(crate) struct CarryStep {
    /// The initial value (enclosing scope).
    pub init: Operand,
    /// The carry inside the body.
    pub slot: Slot,
    /// The `carry_out` value (body scope).
    pub carry_out: Slot,
    /// The repeat's output port.
    pub port: Slot,
}

// ---- Device check (runtime.md §1 step 2) ----

fn check_device(analysis: &Analysis, device: DeviceId) -> Result<()> {
    let supports = |dtype: Dtype| -> Result<bool> {
        DynTensor::device_supports(device, to_backend_dtype(dtype)).map_err(|source| {
            RuntimeError::DeviceUnavailable {
                device: device.to_string(),
                source,
            }
        })
    };
    let unsupported = |what: String, node: Option<String>, op: Option<&str>, dtype: Dtype| {
        RuntimeError::UnsupportedDtype {
            what,
            node,
            op: op.map(str::to_owned),
            dtype,
            device: device.to_string(),
        }
    };
    let compute = analysis.compute_dtype.dtype();
    if !supports(compute)? {
        return Err(unsupported("the compute dtype".into(), None, None, compute));
    }
    for v in &analysis.inputs {
        if !supports(v.info.dtype)? {
            return Err(unsupported(
                format!("input `{}`", v.name),
                None,
                None,
                v.info.dtype,
            ));
        }
    }
    for v in &analysis.state {
        if !supports(v.info.dtype)? {
            return Err(unsupported(
                format!("state entry `{}`", v.name),
                None,
                None,
                v.info.dtype,
            ));
        }
    }
    let mut cache: HashMap<Dtype, bool> = HashMap::new();
    let mut check = |dtype: Dtype| -> Result<bool> {
        if let Some(&ok) = cache.get(&dtype) {
            return Ok(ok);
        }
        let ok = supports(dtype)?;
        cache.insert(dtype, ok);
        Ok(ok)
    };
    check_nodes(&analysis.nodes, "", &mut check, &unsupported)
}

type Unsupported<'a> = dyn Fn(String, Option<String>, Option<&str>, Dtype) -> RuntimeError + 'a;

fn check_nodes(
    nodes: &[NodeInfo],
    prefix: &str,
    check: &mut dyn FnMut(Dtype) -> Result<bool>,
    unsupported: &Unsupported<'_>,
) -> Result<()> {
    for node in nodes {
        let path = format!("{prefix}{}", node.id());
        match node {
            NodeInfo::Op(op) => {
                for (port, info) in &op.outputs {
                    if !check(info.dtype)? {
                        return Err(unsupported(
                            format!("node `{path}` ({}) output `{port}`", op.op.name),
                            Some(path.clone()),
                            Some(op.op.name),
                            info.dtype,
                        ));
                    }
                }
            }
            NodeInfo::Repeat(r) => {
                for (carry, info) in &r.carry {
                    if !check(info.dtype)? {
                        return Err(unsupported(
                            format!("node `{path}` (repeat) carry `{carry}`"),
                            Some(path.clone()),
                            Some("repeat"),
                            info.dtype,
                        ));
                    }
                }
                check_nodes(&r.nodes, &format!("{path}/"), check, unsupported)?;
            }
        }
    }
    Ok(())
}

fn literal_shape(name: &str, shape: &[Dim]) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|d| match d {
            Dim::Literal(v) => usize::try_from(*v).ok().filter(|&v| v >= 1),
            Dim::Expr(_) => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            RuntimeError::internal(format!(
                "param `{name}` has a non-literal or non-positive shape after validation"
            ))
        })
}

// ---- Compilation (runtime.md §1 steps 3–4) ----

/// What a name of a scope refers to.
#[derive(Debug, Clone)]
enum Name {
    /// A graph input, state entry or carry: referred to by its name only.
    Value(Slot),
    /// A node: its ports.
    Node(Vec<(String, Slot)>),
}

/// An unordered step with what it reads.
struct Pending {
    step: Step,
    /// Every slot read, in operand order (repeat: carry inits, then the body's outside reads).
    reads: Vec<Slot>,
}

struct Compiler<'g> {
    slots: Vec<String>,
    /// Innermost last.
    scopes: Vec<HashMap<&'g str, Name>>,
}

fn compile(graph: &Graph, analysis: &Analysis) -> Result<Program> {
    let mut c = Compiler {
        slots: Vec::new(),
        scopes: vec![HashMap::new()],
    };
    let mut owned = Vec::new();
    let mut inputs = Vec::with_capacity(graph.inputs.len());
    for input in &graph.inputs {
        let slot = c.alloc(input.name.clone());
        c.define(&input.name, Name::Value(slot))?;
        inputs.push(slot);
        owned.push(slot);
    }
    if inputs.len() != analysis.inputs.len() {
        return Err(RuntimeError::internal(
            "the analysis lists a different number of inputs than the graph",
        ));
    }
    let mut root_state = Vec::with_capacity(graph.state.len());
    for (i, entry) in graph.state.iter().enumerate() {
        let slot = c.alloc(entry.name.clone());
        c.define(&entry.name, Name::Value(slot))?;
        if analysis.state.get(i).map(|s| s.name.as_str()) != Some(entry.name.as_str()) {
            return Err(RuntimeError::internal(format!(
                "root state entry `{}` is not at position {i} of the analysis",
                entry.name
            )));
        }
        root_state.push((i, slot));
        owned.push(slot);
    }
    let state_pos = |name: &str| {
        root_state
            .iter()
            .zip(&graph.state)
            .find(|(_, e)| e.name == name)
            .map(|((i, _), _)| *i)
            .ok_or_else(|| {
                RuntimeError::internal(format!("state update of an undeclared entry `{name}`"))
            })
    };
    let ports = c.declare_nodes(&graph.nodes, &analysis.nodes, "")?;
    owned.extend(ports.iter().flatten().copied());
    let pending = c.compile_nodes(&graph.nodes, &analysis.nodes, &ports, "")?;
    let outputs = graph
        .outputs
        .iter()
        .map(|o| Ok((o.name.clone(), c.resolve(&o.value)?)))
        .collect::<Result<Vec<_>>>()?;
    let state_updates = graph
        .state_updates
        .iter()
        .map(|(name, value)| Ok((state_pos(name)?, c.resolve(value)?)))
        .collect::<Result<Vec<_>>>()?;
    let end_uses: Vec<Slot> = outputs
        .iter()
        .map(|(_, s)| *s)
        .chain(state_updates.iter().map(|(_, s)| *s))
        .collect();
    let early: Vec<Slot> = inputs
        .iter()
        .copied()
        .chain(root_state.iter().map(|(_, s)| *s))
        .collect();
    let (root, _) = schedule(pending, &ports, &early, owned, &end_uses, "the graph")?;
    Ok(Program {
        slots: c.slots,
        root,
        inputs,
        root_state,
        outputs,
        state_updates,
    })
}

impl<'g> Compiler<'g> {
    fn alloc(&mut self, name: String) -> Slot {
        self.slots.push(name);
        self.slots.len() - 1
    }

    fn define(&mut self, name: &'g str, value: Name) -> Result<()> {
        let scope = self
            .scopes
            .last_mut()
            .ok_or_else(|| RuntimeError::internal("no scope"))?;
        if scope.insert(name, value).is_some() {
            return Err(RuntimeError::internal(format!(
                "name `{name}` is defined twice in one scope after validation"
            )));
        }
        Ok(())
    }

    /// Resolves a value reference (`node`, `node:port`, an input, state or carry name) in the
    /// current scope chain.
    fn resolve(&self, reference: &str) -> Result<Slot> {
        let (base, port) = match reference.split_once(':') {
            Some((b, p)) => (b, Some(p)),
            None => (reference, None),
        };
        let name = self
            .scopes
            .iter()
            .rev()
            .find_map(|s| s.get(base))
            .ok_or_else(|| {
                RuntimeError::internal(format!(
                    "reference `{reference}` doesn't resolve after validation"
                ))
            })?;
        match (name, port) {
            (Name::Value(slot), None) => Ok(*slot),
            (Name::Node(ports), port) => {
                let port = port.unwrap_or("out");
                ports
                    .iter()
                    .find(|(p, _)| p == port)
                    .map(|(_, s)| *s)
                    .ok_or_else(|| {
                        RuntimeError::internal(format!(
                            "reference `{reference}`: no port `{port}` after validation"
                        ))
                    })
            }
            (Name::Value(_), Some(_)) => Err(RuntimeError::internal(format!(
                "reference `{reference}`: a value has no ports"
            ))),
        }
    }

    /// Allocates the output slots of every node of a scope and defines the node ids, so that
    /// references to later nodes resolve. Returns the port slots per node.
    fn declare_nodes(
        &mut self,
        nodes: &'g [Node],
        infos: &[NodeInfo],
        prefix: &str,
    ) -> Result<Vec<Vec<Slot>>> {
        if nodes.len() != infos.len() {
            return Err(RuntimeError::internal(
                "the analysis lists a different number of nodes than the graph",
            ));
        }
        let mut all = Vec::with_capacity(nodes.len());
        for (node, info) in nodes.iter().zip(infos) {
            if node.id != info.id() {
                return Err(RuntimeError::internal(format!(
                    "node `{}` doesn't match the analysis",
                    node.id
                )));
            }
            let mut ports = Vec::new();
            let mut slots = Vec::new();
            for (port, _) in info.outputs() {
                let display = if port == "out" {
                    format!("{prefix}{}", node.id)
                } else {
                    format!("{prefix}{}:{port}", node.id)
                };
                let slot = self.alloc(display);
                ports.push((port.clone(), slot));
                slots.push(slot);
            }
            self.define(&node.id, Name::Node(ports))?;
            all.push(slots);
        }
        Ok(all)
    }

    fn compile_nodes(
        &mut self,
        nodes: &'g [Node],
        infos: &[NodeInfo],
        ports: &[Vec<Slot>],
        prefix: &str,
    ) -> Result<Vec<Pending>> {
        let mut out = Vec::with_capacity(nodes.len());
        for (index, ((node, info), ports)) in nodes.iter().zip(infos).zip(ports).enumerate() {
            let pending = match (&node.kind, info) {
                (NodeKind::Op(op), NodeInfo::Op(op_info)) => {
                    let def = op_info.op;
                    if !ops::is_mapped(def.name) {
                        return Err(RuntimeError::internal(format!(
                            "op `{}` has no backend mapping",
                            def.name
                        )));
                    }
                    let mut reads = Vec::new();
                    let mut inputs = Vec::with_capacity(def.inputs.len());
                    for input in def.inputs {
                        let given = op.inputs.iter().find(|(n, _)| n == input.name);
                        inputs.push(match given {
                            None => OpInput::Absent,
                            Some((_, InputRef::One(r))) => {
                                let slot = self.resolve(r)?;
                                reads.push(slot);
                                OpInput::One(Operand { slot, take: false })
                            }
                            Some((_, InputRef::Many(rs))) => {
                                let mut list = Vec::with_capacity(rs.len());
                                for r in rs {
                                    let slot = self.resolve(r)?;
                                    reads.push(slot);
                                    list.push(Operand { slot, take: false });
                                }
                                OpInput::Many(list)
                            }
                        });
                    }
                    let params = def
                        .params
                        .iter()
                        .map(|p| {
                            op.params
                                .iter()
                                .find(|(n, _)| n == p.name)
                                .map(|(_, v)| v.clone())
                        })
                        .collect();
                    Pending {
                        step: Step {
                            index,
                            id: node.id.clone(),
                            kind: StepKind::Op(OpStep {
                                op: def,
                                inputs,
                                params,
                                outputs: ports.clone(),
                            }),
                            frees: Vec::new(),
                        },
                        reads,
                    }
                }
                (NodeKind::Repeat(r), NodeInfo::Repeat(r_info)) => {
                    // Carry inits resolve in the enclosing scope.
                    let inits = r
                        .carry
                        .iter()
                        .map(|(_, v)| self.resolve(v))
                        .collect::<Result<Vec<_>>>()?;
                    let body_prefix = format!("{prefix}{}/", node.id);
                    self.scopes.push(HashMap::new());
                    let compiled = self.compile_body(r, r_info, ports, &inits, &body_prefix);
                    self.scopes.pop();
                    let (step, outside) = compiled?;
                    let mut reads = inits;
                    reads.extend(outside);
                    Pending {
                        step: Step {
                            index,
                            id: node.id.clone(),
                            kind: StepKind::Repeat(Box::new(step)),
                            frees: Vec::new(),
                        },
                        reads,
                    }
                }
                _ => {
                    return Err(RuntimeError::internal(format!(
                        "node `{}` doesn't match the analysis",
                        node.id
                    )));
                }
            };
            out.push(pending);
        }
        Ok(out)
    }

    /// Compiles a body in a fresh scope (pushed by the caller). Returns the step and the slots
    /// of enclosing scopes that the body reads.
    fn compile_body(
        &mut self,
        r: &'g polypous_ir::model::Repeat,
        info: &polypous_ir::RepeatInfo,
        ports: &[Slot],
        inits: &[Slot],
        prefix: &str,
    ) -> Result<(RepeatStep, Vec<Slot>)> {
        let count = u64::try_from(r.count)
            .ok()
            .filter(|&c| c >= 1 && c == info.count)
            .ok_or_else(|| RuntimeError::internal("repeat count doesn't match the analysis"))?;
        let mut owned = Vec::new();
        let mut carry_slots = Vec::with_capacity(r.carry.len());
        for (name, _) in &r.carry {
            let slot = self.alloc(format!("{prefix}{name}"));
            self.define(name, Name::Value(slot))?;
            carry_slots.push(slot);
            owned.push(slot);
        }
        let mut state = Vec::with_capacity(r.body.state.len());
        for entry in &r.body.state {
            let slot = self.alloc(format!("{prefix}{}", entry.name));
            self.define(&entry.name, Name::Value(slot))?;
            state.push((entry.name.clone(), slot));
            owned.push(slot);
        }
        let node_ports = self.declare_nodes(&r.body.nodes, &info.nodes, prefix)?;
        owned.extend(node_ports.iter().flatten().copied());
        let pending = self.compile_nodes(&r.body.nodes, &info.nodes, &node_ports, prefix)?;
        let state_updates = r
            .body
            .state_updates
            .iter()
            .map(|(name, v)| Ok((name.clone(), self.resolve(v)?)))
            .collect::<Result<Vec<_>>>()?;
        if r.carry.len() != ports.len() || r.carry.len() != inits.len() {
            return Err(RuntimeError::internal(
                "repeat carries don't match the analysis",
            ));
        }
        let mut carry = Vec::with_capacity(r.carry.len());
        for (k, (name, _)) in r.carry.iter().enumerate() {
            let out_ref = r
                .body
                .carry_out
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v)
                .ok_or_else(|| {
                    RuntimeError::internal(format!("carry `{name}` has no carry_out"))
                })?;
            carry.push(CarryStep {
                init: Operand {
                    slot: inits[k],
                    take: false,
                },
                slot: carry_slots[k],
                carry_out: self.resolve(out_ref)?,
                port: ports[k],
            });
        }
        let end_uses: Vec<Slot> = carry
            .iter()
            .map(|c| c.carry_out)
            .chain(state_updates.iter().map(|(_, s)| *s))
            .collect();
        let early: Vec<Slot> = carry_slots
            .iter()
            .copied()
            .chain(state.iter().map(|(_, s)| *s))
            .collect();
        let owned_set: HashSet<Slot> = owned.iter().copied().collect();
        let outside_end: Vec<Slot> = end_uses
            .iter()
            .copied()
            .filter(|s| !owned_set.contains(s))
            .collect();
        let (body, mut outside) = schedule(
            pending,
            &node_ports,
            &early,
            owned,
            &end_uses,
            &format!("the body of `{}`", prefix.trim_end_matches('/')),
        )?;
        for s in outside_end {
            if !outside.contains(&s) {
                outside.push(s);
            }
        }
        Ok((
            RepeatStep {
                count,
                carry,
                body,
                params_prefix: r.body.params_prefix.clone().unwrap_or_default(),
                state_prefix: r.body.state_prefix.clone().unwrap_or_default(),
                state,
                state_updates,
            },
            outside,
        ))
    }
}

/// Orders the steps of a scope topologically and computes the drops (runtime.md §1 steps 3–4).
/// `ports[i]` are the slots node `i` produces, `early` the owned slots defined before the first
/// step, `owned` every slot the scope owns, `end_uses` the slots read after the last step.
/// Returns the block and the slots of enclosing scopes that the steps read (deduplicated).
fn schedule(
    pending: Vec<Pending>,
    ports: &[Vec<Slot>],
    early: &[Slot],
    owned: Vec<Slot>,
    end_uses: &[Slot],
    what: &str,
) -> Result<(Block, Vec<Slot>)> {
    let n = pending.len();
    let mut producer: HashMap<Slot, usize> = HashMap::new();
    for (i, slots) in ports.iter().enumerate() {
        for &s in slots {
            producer.insert(s, i);
        }
    }
    // Kahn's algorithm; ties in document order.
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indegree = vec![0usize; n];
    for (i, p) in pending.iter().enumerate() {
        let mut deps: Vec<usize> = p
            .reads
            .iter()
            .filter_map(|s| producer.get(s).copied())
            .collect();
        deps.sort_unstable();
        deps.dedup();
        for d in deps {
            dependents[d].push(i);
            indegree[i] += 1;
        }
    }
    let mut ready: BinaryHeap<Reverse<usize>> =
        (0..n).filter(|&i| indegree[i] == 0).map(Reverse).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse(i)) = ready.pop() {
        order.push(i);
        for &j in &dependents[i] {
            indegree[j] -= 1;
            if indegree[j] == 0 {
                ready.push(Reverse(j));
            }
        }
    }
    if order.len() != n {
        return Err(RuntimeError::internal(format!(
            "the nodes of {what} form a cycle after validation"
        )));
    }

    let owned_set: HashSet<Slot> = owned.iter().copied().collect();
    let end_set: HashSet<Slot> = end_uses.iter().copied().collect();
    let mut slots_of: Vec<Option<Pending>> = pending.into_iter().map(Some).collect();
    let mut steps: Vec<Pending> = Vec::with_capacity(n);
    for &i in &order {
        let p = slots_of[i]
            .take()
            .ok_or_else(|| RuntimeError::internal("a node is scheduled twice"))?;
        steps.push(p);
    }
    // Last use (position in `steps`) of every owned slot; outside reads.
    let mut last_use: HashMap<Slot, usize> = HashMap::new();
    let mut outside = Vec::new();
    for (pos, p) in steps.iter().enumerate() {
        for &s in &p.reads {
            if owned_set.contains(&s) {
                last_use.insert(s, pos);
            } else if !outside.contains(&s) {
                outside.push(s);
            }
        }
    }
    let position: HashMap<usize, usize> = steps
        .iter()
        .enumerate()
        .map(|(pos, p)| (p.step.index, pos))
        .collect();
    let mut frees: Vec<Vec<Slot>> = vec![Vec::new(); steps.len()];
    let mut frees_at_start = Vec::new();
    for &s in &owned {
        if end_set.contains(&s) {
            continue;
        }
        match last_use.get(&s) {
            Some(&pos) => frees[pos].push(s),
            None => match producer.get(&s).and_then(|i| position.get(i)) {
                Some(&pos) => frees[pos].push(s),
                None if early.contains(&s) => frees_at_start.push(s),
                None => {
                    return Err(RuntimeError::internal(format!(
                        "slot {s} of {what} has no producer"
                    )));
                }
            },
        }
    }
    let mut block_steps = Vec::with_capacity(steps.len());
    for (pos, mut p) in steps.into_iter().enumerate() {
        let free: HashSet<Slot> = frees[pos].iter().copied().collect();
        mark_takes(&mut p.step.kind, &free);
        p.step.frees = std::mem::take(&mut frees[pos]);
        block_steps.push(p.step);
    }
    Ok((
        Block {
            steps: block_steps,
            frees_at_start,
            owned,
        },
        outside,
    ))
}

/// Sets [`Operand::take`] on the last occurrence of every operand that the step frees. A carry
/// init is only taken if the body doesn't read the same value.
fn mark_takes(kind: &mut StepKind, free: &HashSet<Slot>) {
    let mut seen: HashSet<Slot> = HashSet::new();
    match kind {
        StepKind::Op(op) => {
            for input in op.inputs.iter_mut().rev() {
                let operands: Vec<&mut Operand> = match input {
                    OpInput::Absent => Vec::new(),
                    OpInput::One(o) => vec![o],
                    OpInput::Many(list) => list.iter_mut().rev().collect(),
                };
                for o in operands {
                    o.take = free.contains(&o.slot) && seen.insert(o.slot);
                }
            }
        }
        StepKind::Repeat(r) => {
            let mut read_by_body = body_reads(&r.body);
            read_by_body.extend(r.carry.iter().map(|c| c.carry_out));
            read_by_body.extend(r.state_updates.iter().map(|(_, s)| *s));
            for c in r.carry.iter_mut().rev() {
                c.init.take = free.contains(&c.init.slot)
                    && !read_by_body.contains(&c.init.slot)
                    && seen.insert(c.init.slot);
            }
        }
    }
}

/// Every slot read by the steps of a block (deep) or at the end of its iterations.
fn body_reads(block: &Block) -> HashSet<Slot> {
    fn walk(block: &Block, out: &mut HashSet<Slot>) {
        for step in &block.steps {
            match &step.kind {
                StepKind::Op(op) => {
                    for input in &op.inputs {
                        match input {
                            OpInput::Absent => {}
                            OpInput::One(o) => {
                                out.insert(o.slot);
                            }
                            OpInput::Many(list) => out.extend(list.iter().map(|o| o.slot)),
                        }
                    }
                }
                StepKind::Repeat(r) => {
                    for c in &r.carry {
                        out.insert(c.init.slot);
                        out.insert(c.carry_out);
                    }
                    out.extend(r.state_updates.iter().map(|(_, s)| *s));
                    walk(&r.body, out);
                }
            }
        }
    }
    let mut out = HashSet::new();
    walk(block, &mut out);
    out
}
