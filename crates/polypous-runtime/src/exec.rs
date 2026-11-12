//! Running a plan (`docs/specs/runtime.md` §3, §4, §6).

use std::collections::HashMap;

use polypous_backend::DynTensor;
use polypous_ir::names::expand_prefix;
use polypous_ir::{BoundNode, NodeInfo, TensorDesc};

use crate::cancel::CancellationToken;
use crate::error::{Result, RuntimeError};
use crate::ops::{self, Arg, OpFailure};
use crate::plan::{Block, OpInput, OpStep, Operand, Plan, RepeatStep, Slot, StepKind};
use crate::weights::{self, WeightFile};

/// A plan with its params loaded on the plan's device (runtime.md §2). Calls share the params
/// read-only; [`Model::run`] takes `&self`.
#[derive(Debug)]
pub struct Model {
    plan: Plan,
    params: HashMap<String, DynTensor>,
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
        let plan = &self.plan;
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
            model: self,
            env,
            state_in: &state_in,
            state_out: vec![None; analysis.state.len()],
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
            trace: run.trace.unwrap_or_default(),
        })
    }
}

/// The mutable state of one call.
struct Run<'a> {
    model: &'a Model,
    env: Vec<Option<DynTensor>>,
    state_in: &'a [Option<DynTensor>],
    state_out: Vec<Option<DynTensor>>,
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
            .model
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
            let names = &self.model.plan.program.slots;
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
                    let t = self.model.params.get(&full).cloned().ok_or_else(|| {
                        RuntimeError::internal(format!(
                            "node `{}`: param `{full}` is not loaded",
                            self.node_path(id)
                        ))
                    })?;
                    Some(t)
                }
            });
        }
        let outputs = ops::run_op(info, bound, inputs, params, self.model.plan.device).map_err(
            |failure| match failure {
                OpFailure::Tensor(source) => RuntimeError::Node {
                    node: self.node_path(id),
                    op: step.op.name.to_owned(),
                    source,
                },
                OpFailure::Internal(message) => {
                    RuntimeError::internal(format!("node `{}`: {message}", self.node_path(id)))
                }
            },
        )?;
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
                        .model
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
                    let k = *self.model.plan.state_index.get(&name).ok_or_else(|| {
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
