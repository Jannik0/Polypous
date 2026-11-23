//! Running a plan (`docs/specs/runtime.md` §3, §4, §6): binding, the checks of params given by
//! the caller (§3.2) and of training mode (§3.1), then the nodes in plan order. In training mode
//! `nn.batch_norm` and `nn.dropout` run their training-mode computation, and the call returns the
//! new running statistics (§3.3).

use std::collections::HashMap;

use polypous_backend::DynTensor;
use polypous_ir::names::expand_prefix;
use polypous_ir::{BoundNode, ComputeDtype, NodeInfo, TensorDesc};

use crate::cancel::CancellationToken;
use crate::error::{ParamProblem, Result, RuntimeError};
use crate::ops::{self, Arg, OpFailure, OpMode};
use crate::plan::{Block, OpInput, OpStep, Operand, Plan, RepeatStep, Slot, StepKind};
use crate::weights::{self, WeightFile};

/// A plan with its params loaded on the plan's device (runtime.md §2). Calls share the params
/// read-only; [`Model::run`] takes `&self`.
#[derive(Debug)]
pub struct Model {
    plan: Plan,
    params: HashMap<String, DynTensor>,
}

/// Params given to a call by the caller (runtime.md §3.2): param name → tensor, for every
/// declared param of the graph (unused ones included). The call reads them and never changes,
/// moves or copies them.
pub type Params = HashMap<String, DynTensor>;

/// The mode of a call (runtime.md §3.1, graph-ir.md §3.1.10).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum RunMode {
    /// The default, and M1's behavior.
    #[default]
    Inference,
    /// `nn.batch_norm` normalizes with the batch statistics and returns new running statistics;
    /// `nn.dropout` with `p > 0` drops elements with the masks of `seed` and `step`
    /// (graph-ir.md §3.1.12). Requires the compute dtype `f32` in M2.
    Training {
        /// Keys the dropout masks, with `step`.
        seed: u64,
        /// Keys the dropout masks, with `seed`.
        step: u64,
    },
}

impl RunMode {
    /// Whether this is training mode.
    pub fn is_training(self) -> bool {
        matches!(self, RunMode::Training { .. })
    }
}

/// Options of one call.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunOptions<'a> {
    /// Checked before every node and every `repeat` iteration (runtime.md §4).
    pub cancel: Option<&'a CancellationToken>,
    /// Record the live values after every step ([`RunOutput::trace`], runtime.md §6).
    pub trace: bool,
}

/// The result of a call (runtime.md §3).
#[derive(Debug)]
pub struct RunOutput {
    /// Graph outputs by output name, in declaration order.
    pub outputs: Vec<(String, DynTensor)>,
    /// Every root-level state entry (expanded names, in `Analysis::state` order): its update, or
    /// the given value if it has none.
    pub state: Vec<(String, DynTensor)>,
    /// In training mode: the new running statistics of every `nn.batch_norm` node that ran
    /// (runtime.md §3.3), as (param name, tensor), in the order the nodes ran (iterations in
    /// order), each node's `running_mean` before its `running_var`. Each tensor has the compute
    /// dtype and the declared shape, lives on the plan's device and carries no autodiff context.
    /// Empty in inference mode.
    pub running_stats: Vec<(String, DynTensor)>,
    /// With [`RunOptions::trace`]: one entry per executed step.
    pub trace: Vec<TraceStep>,
}

impl RunOutput {
    /// The graph output `name`.
    pub fn output(&self, name: &str) -> Option<&DynTensor> {
        self.outputs.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// The returned state entry `name`.
    pub fn state_entry(&self, name: &str) -> Option<&DynTensor> {
        self.state.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// The new running statistic `name` (training mode, runtime.md §3.3).
    pub fn running_stat(&self, name: &str) -> Option<&DynTensor> {
        self.running_stats
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
    }
}

/// One executed step of a traced call (runtime.md §6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceStep {
    /// The node path (`fc1`, `layers[3]/attn`, or `layers` after the whole repeat).
    pub node: String,
    /// The values alive after the step and its drops, sorted (body values as `layers/attn`).
    pub live: Vec<String>,
}

impl Model {
    /// Loads the params of `plan` from `weights` (runtime.md §2).
    ///
    /// Errors name the tensor: [`RuntimeError::Weight`] (missing, extra, wrong dtype or shape),
    /// [`RuntimeError::WeightFile`] (unreadable or malformed data), [`RuntimeError::WeightUpload`].
    pub fn load(plan: Plan, weights: &WeightFile) -> Result<Model> {
        let params = weights::load(&plan, Some(weights))?;
        Ok(Model { plan, params })
    }

    /// A model of a graph that declares no params. Fails with a missing-weight error naming the
    /// first param otherwise.
    pub fn without_weights(plan: Plan) -> Result<Model> {
        let params = weights::load(&plan, None)?;
        Ok(Model { plan, params })
    }

    /// The plan.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The loaded tensor of graph param `name`, in the compute dtype on the plan's device. `None`
    /// for a name the graph doesn't declare, and for a declared param that no node uses: those are
    /// checked but not uploaded (runtime.md §2 step 1).
    pub fn param(&self, name: &str) -> Option<&DynTensor> {
        self.params.get(name)
    }

    /// Runs the graph once (runtime.md §3): `inputs` are the graph inputs by name, `state` the
    /// values of every root-level state entry by (expanded) name.
    ///
    /// Errors: [`RuntimeError::Bind`] and [`RuntimeError::WrongDevice`] before anything runs;
    /// [`RuntimeError::Cancelled`]; [`RuntimeError::Node`] naming the node path. A failed call
    /// returns no partial outputs.
    pub fn run(
        &self,
        inputs: Vec<(String, DynTensor)>,
        state: Vec<(String, DynTensor)>,
        options: &RunOptions<'_>,
    ) -> Result<RunOutput> {
        execute(
            &self.plan,
            Source::Loaded(&self.params),
            RunMode::Inference,
            inputs,
            state,
            options,
        )
    }

    /// Runs the graph once in `mode` (runtime.md §3, §3.1), with the loaded params or, if
    /// `params` is `Some`, with the params the caller gives (§3.2). [`Model::run`] is
    /// `run_with(inputs, state, None, RunMode::Inference, options)`.
    ///
    /// Errors, in this order (runtime.md §3): binding and devices as [`Model::run`]; then, for
    /// given params, [`RuntimeError::Param`]; then, in training mode,
    /// [`RuntimeError::TrainingDtype`], [`RuntimeError::SharedRunningStat`] and
    /// [`RuntimeError::BatchNormValues`] (§3.1); then the errors of running the nodes.
    pub fn run_with(
        &self,
        inputs: Vec<(String, DynTensor)>,
        state: Vec<(String, DynTensor)>,
        params: Option<&Params>,
        mode: RunMode,
        options: &RunOptions<'_>,
    ) -> Result<RunOutput> {
        let source = match params {
            Some(p) => Source::Given(p),
            None => Source::Loaded(&self.params),
        };
        execute(&self.plan, source, mode, inputs, state, options)
    }
}

impl Plan {
    /// Runs the plan once in `mode` on params the caller gives (runtime.md §3, §3.2), without a
    /// [`Model`]: this is how the trainer runs the graph on its param store. A graph without
    /// params takes an empty map. Errors as [`Model::run_with`].
    pub fn run(
        &self,
        inputs: Vec<(String, DynTensor)>,
        state: Vec<(String, DynTensor)>,
        params: &Params,
        mode: RunMode,
        options: &RunOptions<'_>,
    ) -> Result<RunOutput> {
        execute(self, Source::Given(params), mode, inputs, state, options)
    }
}

/// Where a call's params come from.
#[derive(Clone, Copy)]
enum Source<'a> {
    /// The model's weights: checked when they were loaded; declared but unused params are absent.
    Loaded(&'a HashMap<String, DynTensor>),
    /// Given by the caller: checked by [`check_given_params`] before any node runs.
    Given(&'a Params),
}

/// One call (runtime.md §3).
fn execute(
    plan: &Plan,
    source: Source<'_>,
    mode: RunMode,
    inputs: Vec<(String, DynTensor)>,
    state: Vec<(String, DynTensor)>,
    options: &RunOptions<'_>,
) -> Result<RunOutput> {
    let analysis = &plan.analysis;
    let program = &plan.program;

    // Binding (§3 step 1).
    let shapes: Vec<Vec<usize>> = inputs
        .iter()
        .chain(&state)
        .map(|(_, t)| t.shape())
        .collect();
    let mut descs = Vec::with_capacity(shapes.len());
    for ((name, t), shape) in inputs.iter().chain(&state).zip(&shapes) {
        let dtype = ops::from_backend_dtype(t.dtype()).ok_or_else(|| {
            RuntimeError::internal(format!(
                "value `{name}` has dtype {}, which the IR doesn't know",
                t.dtype()
            ))
        })?;
        descs.push((
            name.as_str(),
            TensorDesc {
                dtype,
                shape: shape.as_slice(),
            },
        ));
    }
    let (input_descs, state_descs) = descs.split_at(inputs.len());
    let bindings = analysis.bind(input_descs, state_descs)?;
    drop(descs);

    // Devices (§3 step 2).
    for (kind, values) in [("input", &inputs), ("state entry", &state)] {
        for (name, t) in values {
            if t.device() != plan.device {
                return Err(RuntimeError::WrongDevice {
                    kind,
                    name: name.clone(),
                    expected: plan.device.to_string(),
                    actual: t.device().to_string(),
                });
            }
        }
    }

    // Params (§3 step 3, §3.2).
    let params = match source {
        Source::Loaded(p) => p,
        Source::Given(p) => {
            check_given_params(plan, p)?;
            p
        }
    };

    // Mode (§3 step 4, §3.1).
    if mode.is_training() {
        check_training(plan, &bindings.nodes)?;
    }

    let mut env: Vec<Option<DynTensor>> = vec![None; program.slots.len()];
    let mut by_name: HashMap<String, DynTensor> = inputs.into_iter().collect();
    for (v, &slot) in analysis.inputs.iter().zip(&program.inputs) {
        env[slot] = by_name.remove(&v.name);
    }
    // Given state values, by position in `analysis.state`; reads clone them, so the given
    // value stays available for entries without an update.
    let mut state_in: Vec<Option<DynTensor>> = vec![None; analysis.state.len()];
    for (name, t) in state {
        match plan.state_index.get(&name) {
            Some(&i) => state_in[i] = Some(t),
            None => {
                return Err(RuntimeError::internal(format!(
                    "state entry `{name}` passed the binding but is unknown"
                )));
            }
        }
    }
    for &(i, slot) in &program.root_state {
        env[slot] = state_in[i].clone();
    }

    let mut run = Run {
        plan,
        params,
        mode,
        env,
        state_in: &state_in,
        state_out: vec![None; analysis.state.len()],
        running_stats: Vec::new(),
        cancel: options.cancel,
        trace: options.trace.then(Vec::new),
        path: String::new(),
        params_prefix: String::new(),
        state_prefix: String::new(),
    };
    run.free(&program.root.frees_at_start);
    run.block(&program.root, &analysis.nodes, &bindings.nodes)?;
    for &(i, slot) in &program.state_updates {
        run.state_out[i] = Some(run.read(slot, "root state update")?);
    }
    let mut outputs = Vec::with_capacity(program.outputs.len());
    for (name, slot) in &program.outputs {
        outputs.push((name.clone(), run.read(*slot, name)?));
    }
    let mut state = Vec::with_capacity(analysis.state.len());
    for (i, entry) in analysis.state.iter().enumerate() {
        let value = run.state_out[i]
            .take()
            .or_else(|| state_in[i].clone())
            .ok_or_else(|| {
                RuntimeError::internal(format!("state entry `{}` has no value", entry.name))
            })?;
        state.push((entry.name.clone(), value));
    }
    Ok(RunOutput {
        outputs,
        state,
        running_stats: run.running_stats,
        trace: run.trace.unwrap_or_default(),
    })
}

/// The checks of params given by the caller (runtime.md §3.2), on metadata only: first the names
/// (a missing param, in declaration order; then an extra name, the first in sorted order), then
/// each param in declaration order: dtype, shape, device.
fn check_given_params(plan: &Plan, given: &Params) -> Result<()> {
    let err = |name: &str, problem| RuntimeError::Param {
        name: name.to_owned(),
        problem,
    };
    for (name, _) in &plan.params {
        if !given.contains_key(name) {
            return Err(err(name, ParamProblem::Missing));
        }
    }
    if given.len() != plan.params.len() {
        let mut extra: Vec<&str> = given
            .keys()
            .map(String::as_str)
            .filter(|n| !plan.params.iter().any(|(d, _)| d == n))
            .collect();
        extra.sort_unstable();
        if let Some(name) = extra.first() {
            return Err(err(name, ParamProblem::Extra));
        }
    }
    let compute = plan.compute_dtype().dtype();
    let compute_backend = ops::to_backend_dtype(compute);
    for (name, declared) in &plan.params {
        let Some(t) = given.get(name) else {
            return Err(err(name, ParamProblem::Missing));
        };
        if t.dtype() != compute_backend {
            return Err(err(
                name,
                ParamProblem::Dtype {
                    expected: compute.to_string(),
                    actual: t.dtype().to_string(),
                },
            ));
        }
        let shape = t.shape();
        if shape != *declared {
            return Err(err(
                name,
                ParamProblem::Shape {
                    expected: declared.clone(),
                    actual: shape,
                },
            ));
        }
        if t.device() != plan.device {
            return Err(err(
                name,
                ParamProblem::Device {
                    expected: plan.device.to_string(),
                    actual: t.device().to_string(),
                },
            ));
        }
    }
    Ok(())
}

/// The checks of a training-mode call (runtime.md §3.1), in order: the compute dtype, one use per
/// running statistic (params in declaration order), at least 2 values per channel for every
/// `nn.batch_norm` (plan order).
fn check_training(plan: &Plan, bound: &[BoundNode]) -> Result<()> {
    let analysis = &plan.analysis;
    if plan.compute_dtype() != ComputeDtype::F32 {
        return Err(RuntimeError::TrainingDtype {
            dtype: plan.compute_dtype().dtype(),
        });
    }
    for p in &analysis.params {
        if p.running_stat && p.uses != 1 {
            let uses: Vec<String> = p.first_uses.iter().map(ToString::to_string).collect();
            let (Some(first), Some(second)) = (uses.first(), uses.get(1)) else {
                return Err(RuntimeError::internal(format!(
                    "running statistic `{}` has {} uses, but fewer are recorded",
                    p.name, p.uses
                )));
            };
            return Err(RuntimeError::SharedRunningStat {
                param: p.name.clone(),
                uses: p.uses,
                first: first.clone(),
                second: second.clone(),
            });
        }
    }
    check_values_per_channel(&plan.program.root, &analysis.nodes, bound, "")
}

/// §3.1 check 3, in plan order. Every iteration of a body has the same shapes, so the first
/// failing iteration is iteration 0.
fn check_values_per_channel(
    block: &Block,
    infos: &[NodeInfo],
    bound: &[BoundNode],
    path: &str,
) -> Result<()> {
    for step in &block.steps {
        match (&step.kind, infos.get(step.index), bound.get(step.index)) {
            (StepKind::Op(op), Some(NodeInfo::Op(_)), Some(BoundNode::Op(b)))
                if op.op.name == "nn.batch_norm" =>
            {
                // `out` has the shape of `x`, `[N, C, …]`.
                let shape = b.outputs.first().map(Vec::as_slice).unwrap_or_default();
                let total = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
                let values = match (total, shape.get(1)) {
                    (Some(t), Some(&c)) if c > 0 => t / c,
                    _ => {
                        return Err(RuntimeError::internal(format!(
                            "node `{path}{}`: batch_norm has the bound shape {shape:?}",
                            step.id
                        )));
                    }
                };
                if values < 2 {
                    return Err(RuntimeError::BatchNormValues {
                        node: format!("{path}{}", step.id),
                        shape: shape.to_vec(),
                        values,
                    });
                }
            }
            (StepKind::Repeat(r), Some(NodeInfo::Repeat(info)), Some(BoundNode::Repeat(b)))
                if r.count > 0 =>
            {
                let inner = format!("{path}{}[0]/", step.id);
                check_values_per_channel(&r.body, &info.nodes, &b.nodes, &inner)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// The mutable state of one call.
struct Run<'a> {
    plan: &'a Plan,
    params: &'a HashMap<String, DynTensor>,
    mode: RunMode,
    env: Vec<Option<DynTensor>>,
    state_in: &'a [Option<DynTensor>],
    state_out: Vec<Option<DynTensor>>,
    running_stats: Vec<(String, DynTensor)>,
    cancel: Option<&'a CancellationToken>,
    trace: Option<Vec<TraceStep>>,
    /// Path of the current scope, `layers[3]/` (empty at the root).
    path: String,
    /// Expanded params and state prefixes of the current scope, outer first.
    params_prefix: String,
    state_prefix: String,
}

impl Run<'_> {
    fn node_path(&self, id: &str) -> String {
        format!("{}{id}", self.path)
    }

    fn check_cancel(&self, before: impl FnOnce() -> String) -> Result<()> {
        match self.cancel {
            Some(token) if token.is_cancelled() => {
                Err(RuntimeError::Cancelled { before: before() })
            }
            _ => Ok(()),
        }
    }

    /// A copy of the value in `slot` (cheap: tensors share their buffer).
    fn read(&self, slot: Slot, what: &str) -> Result<DynTensor> {
        self.env
            .get(slot)
            .and_then(Clone::clone)
            .ok_or_else(|| self.dropped(slot, what))
    }

    fn operand(&mut self, o: Operand, what: &str) -> Result<DynTensor> {
        if o.take {
            self.env
                .get_mut(o.slot)
                .and_then(Option::take)
                .ok_or_else(|| self.dropped(o.slot, what))
        } else {
            self.read(o.slot, what)
        }
    }

    fn dropped(&self, slot: Slot, what: &str) -> RuntimeError {
        let name = self
            .plan
            .program
            .slots
            .get(slot)
            .map_or("?", String::as_str);
        RuntimeError::internal(format!(
            "value `{name}` is not available for {what} in scope `{}`",
            self.path
        ))
    }

    fn free(&mut self, slots: &[Slot]) {
        for &s in slots {
            if let Some(v) = self.env.get_mut(s) {
                *v = None;
            }
        }
    }

    fn record(&mut self, node: String) {
        if let Some(trace) = &mut self.trace {
            let names = &self.plan.program.slots;
            let mut live: Vec<String> = self
                .env
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_some())
                .map(|(s, _)| names[s].clone())
                .collect();
            live.sort();
            trace.push(TraceStep { node, live });
        }
    }

    fn block(&mut self, block: &Block, infos: &[NodeInfo], bound: &[BoundNode]) -> Result<()> {
        for step in &block.steps {
            self.check_cancel(|| self.node_path(&step.id))?;
            let (info, bound) = match (infos.get(step.index), bound.get(step.index)) {
                (Some(i), Some(b)) => (i, b),
                _ => {
                    return Err(RuntimeError::internal(format!(
                        "node `{}` has no analysis or binding",
                        step.id
                    )));
                }
            };
            match (&step.kind, info, bound) {
                (StepKind::Op(op), NodeInfo::Op(info), BoundNode::Op(bound)) => {
                    self.op(&step.id, op, info, bound)?;
                }
                (StepKind::Repeat(r), NodeInfo::Repeat(info), BoundNode::Repeat(bound)) => {
                    self.repeat(&step.id, r, &info.nodes, &bound.nodes)?;
                }
                _ => {
                    return Err(RuntimeError::internal(format!(
                        "node `{}` doesn't match its analysis",
                        step.id
                    )));
                }
            }
            self.free(&step.frees);
            if self.trace.is_some() {
                let node = self.node_path(&step.id);
                self.record(node);
            }
        }
        Ok(())
    }

    fn op(
        &mut self,
        id: &str,
        step: &OpStep,
        info: &polypous_ir::OpInfo,
        bound: &polypous_ir::BoundOp,
    ) -> Result<()> {
        let mut inputs = Vec::with_capacity(step.inputs.len());
        for input in &step.inputs {
            inputs.push(match input {
                OpInput::Absent => None,
                OpInput::One(o) => Some(Arg::One(Box::new(self.operand(*o, id)?))),
                OpInput::Many(list) => {
                    let mut ts = Vec::with_capacity(list.len());
                    for o in list {
                        ts.push(self.operand(*o, id)?);
                    }
                    Some(Arg::Many(ts))
                }
            });
        }
        let mut params = Vec::with_capacity(step.params.len());
        for name in &step.params {
            params.push(match name {
                None => None,
                Some(local) => {
                    let full = format!("{}{local}", self.params_prefix);
                    let t = self.params.get(&full).cloned().ok_or_else(|| {
                        RuntimeError::internal(format!(
                            "node `{}`: param `{full}` is not loaded",
                            self.node_path(id)
                        ))
                    })?;
                    Some(t)
                }
            });
        }
        let node_path = self.node_path(id);
        let op_mode = match self.mode {
            RunMode::Inference => OpMode::Inference,
            RunMode::Training { seed, step } => OpMode::Training {
                seed,
                step,
                node_path: &node_path,
            },
        };
        let mut running = None;
        let outputs = ops::run_op(
            info,
            bound,
            inputs,
            params,
            self.plan.device,
            op_mode,
            &mut running,
        )
        .map_err(|failure| match failure {
            OpFailure::Tensor(source) => RuntimeError::Node {
                node: self.node_path(id),
                op: step.op.name.to_owned(),
                source,
            },
            OpFailure::Internal(message) => {
                RuntimeError::internal(format!("node `{}`: {message}", self.node_path(id)))
            }
        })?;
        if outputs.len() != step.outputs.len() {
            return Err(RuntimeError::internal(format!(
                "node `{}` returned {} outputs, expected {}",
                self.node_path(id),
                outputs.len(),
                step.outputs.len()
            )));
        }
        for (slot, t) in step.outputs.iter().zip(outputs) {
            self.env[*slot] = Some(t);
        }
        if let Some((mean, var)) = running {
            // `nn.batch_norm`'s params in catalog order: running_mean, running_var, …
            let name = |k: usize| -> Result<String> {
                match step.params.get(k) {
                    Some(Some(local)) => Ok(format!("{}{local}", self.params_prefix)),
                    _ => Err(RuntimeError::internal(format!(
                        "node `{}` returned running statistics but has no param {k}",
                        self.node_path(id)
                    ))),
                }
            };
            let (mean_name, var_name) = (name(0)?, name(1)?);
            self.running_stats.push((mean_name, mean));
            self.running_stats.push((var_name, var));
        }
        Ok(())
    }

    fn repeat(
        &mut self,
        id: &str,
        r: &RepeatStep,
        infos: &[NodeInfo],
        bound: &[BoundNode],
    ) -> Result<()> {
        let mut carry: Vec<DynTensor> = Vec::with_capacity(r.carry.len());
        for c in &r.carry {
            carry.push(self.operand(c.init, id)?);
        }
        let (path, params_prefix, state_prefix) = (
            self.path.len(),
            self.params_prefix.len(),
            self.state_prefix.len(),
        );
        let result = (|| -> Result<()> {
            for i in 0..r.count {
                self.path.truncate(path);
                self.path.push_str(&format!("{id}[{i}]"));
                self.check_cancel(|| self.path.clone())?;
                self.path.push('/');
                self.params_prefix.truncate(params_prefix);
                self.params_prefix
                    .push_str(&expand_prefix(&r.params_prefix, i));
                self.state_prefix.truncate(state_prefix);
                self.state_prefix
                    .push_str(&expand_prefix(&r.state_prefix, i));

                for (c, value) in r.carry.iter().zip(carry.drain(..)) {
                    self.env[c.slot] = Some(value);
                }
                for (local, slot) in &r.state {
                    let name = format!("{}{local}", self.state_prefix);
                    let value = self
                        .plan
                        .state_index
                        .get(&name)
                        .and_then(|&k| self.state_in.get(k).cloned().flatten())
                        .ok_or_else(|| {
                            RuntimeError::internal(format!("state entry `{name}` has no value"))
                        })?;
                    self.env[*slot] = Some(value);
                }
                self.free(&r.body.frees_at_start);
                self.block(&r.body, infos, bound)?;

                for c in &r.carry {
                    carry.push(self.read(c.carry_out, "carry_out")?);
                }
                for (local, slot) in &r.state_updates {
                    let name = format!("{}{local}", self.state_prefix);
                    let value = self.read(*slot, "a state update")?;
                    let k = *self.plan.state_index.get(&name).ok_or_else(|| {
                        RuntimeError::internal(format!("state entry `{name}` is unknown"))
                    })?;
                    self.state_out[k] = Some(value);
                }
                // Every value the body owns, intermediates of nested bodies included, is dropped
                // at the end of the iteration.
                self.free(&r.body.owned);
            }
            Ok(())
        })();
        self.path.truncate(path);
        self.params_prefix.truncate(params_prefix);
        self.state_prefix.truncate(state_prefix);
        result?;
        for (c, value) in r.carry.iter().zip(carry) {
            self.env[c.port] = Some(value);
        }
        Ok(())
    }
}
