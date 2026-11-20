//! Autodiff on [`DynTensor`] (ADR-0014): param leaves, `backward`, [`Gradients`], `detach`.
//!
//! Burn 0.22.1 records operations when a tensor carries an autodiff context
//! (`burn-tensor` `tensor/api/autodiff.rs`). The context lives on the **tensor**, not on the
//! device: `Tensor::autodiff()` wraps a tensor of a plain device, and its device then reads as
//! `DispatchDevice::Autodiff(inner)`. A `DynTensor` therefore keeps its [`DeviceId`] when it is
//! tracked (`device_id_of` sees through the autodiff wrapper), and every op that builds helper
//! tensors with `t.device()` builds them on the autodiff device, as untracked constants. Tensors
//! without autodiff context (every tensor that was never marked, and everything computed only from
//! such tensors) pay no autodiff dispatch: inference is unchanged.
//!
//! # How a training step uses it
//!
//! 1. Mark each trainable param with [`DynTensor::require_grad`]. The result is a **leaf**; keep
//!    it (or a clone: clones are the same leaf) as the key for its gradient.
//! 2. Run the ops. Every result that depends on a leaf is tracked ([`DynTensor::is_tracked`]).
//!    Untracked tensors (inputs, constants, bool masks, int indices) mix freely with tracked ones.
//! 3. Call [`DynTensor::backward`] on the one-element loss. It returns [`Gradients`].
//! 4. Read each leaf's gradient with [`Gradients::get`] or [`Gradients::take`]: a plain
//!    `DynTensor` without autodiff context, in the leaf's dtype, shape and device.
//! 5. Update the params on [`DynTensor::without_autodiff`] copies, so that the update itself is
//!    not recorded, and mark the new values again for the next step.
//!
//! **One backward per forward.** Burn does not retain the graph: `backward` consumes the
//! recorded steps it walks, and a second `backward` through them is an error (Burn would panic;
//! see [`DynTensor::backward`]). Combine losses before calling it. A fresh forward from the same
//! leaves works.
//!
//! **Gradient conventions.** Gradients follow graph-ir.md §3.1.11 (PyTorch's, with its tie and
//! kink rules). Where Burn's own backward differs, the op fixes it on its tracked path (see the
//! module documentation of [`crate::ops`], "Gradients"); untracked calls run the forward code
//! unchanged, and tracked calls give the same forward values bit for bit.
//!
//! No Burn type appears in this API (ADR-0007): [`Gradients`] holds Burn's container privately.
//!
//! [`DeviceId`]: crate::device::DeviceId

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::{DynTensor, Inner, Ranked, Result, TensorError, map_rank, with_rank};

/// The gradients of one [`DynTensor::backward`] pass.
///
/// Opaque: read a leaf's gradient with [`Gradients::get`] or [`Gradients::take`]. Holds every
/// leaf gradient of the pass until it is dropped or taken.
pub struct Gradients(burn::tensor::Gradients);

impl fmt::Debug for Gradients {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gradients").finish_non_exhaustive()
    }
}

/// Runs a gradient lookup on the float tensor of each rank and wraps the result back into an
/// `Inner` of that rank. Non-float tensors have no gradient (`None`); callers checked for a leaf.
macro_rules! lookup_rank {
    ($inner:expr, $t:ident => $body:expr) => {
        match $inner {
            Inner::R1(Ranked::Float($t)) => $body.map(|g| Inner::R1(Ranked::Float(g))),
            Inner::R2(Ranked::Float($t)) => $body.map(|g| Inner::R2(Ranked::Float(g))),
            Inner::R3(Ranked::Float($t)) => $body.map(|g| Inner::R3(Ranked::Float(g))),
            Inner::R4(Ranked::Float($t)) => $body.map(|g| Inner::R4(Ranked::Float(g))),
            Inner::R5(Ranked::Float($t)) => $body.map(|g| Inner::R5(Ranked::Float(g))),
            Inner::R6(Ranked::Float($t)) => $body.map(|g| Inner::R6(Ranked::Float(g))),
            _ => None,
        }
    };
}

impl Gradients {
    /// The gradient of `leaf`, a tensor from [`DynTensor::require_grad`] (or a clone of it),
    /// without autodiff context, in the leaf's dtype, shape and device. Gradients add up over
    /// every use of the leaf (graph-ir.md §3.1.11).
    ///
    /// `Ok(None)` if the leaf took no part in the backward pass (for example a param the loss
    /// doesn't depend on, or a leaf of another pass). [`TensorError::InvalidArgument`] if
    /// `leaf` is not a leaf: an untracked tensor, a non-float tensor, or the result of an op.
    pub fn get(&self, leaf: &DynTensor) -> Result<Option<DynTensor>> {
        leaf.check_leaf()?;
        let grad = lookup_rank!(leaf.inner_ref(), t => t.grad(&self.0));
        Ok(grad.map(|inner| DynTensor::from_parts(inner, leaf.dtype(), leaf.device())))
    }

    /// Like [`Gradients::get`], but removes the gradient from the container, which saves a copy
    /// when each gradient is read once. A second `take` of the same leaf returns `Ok(None)`.
    pub fn take(&mut self, leaf: &DynTensor) -> Result<Option<DynTensor>> {
        leaf.check_leaf()?;
        let grad = lookup_rank!(leaf.inner_ref(), t => t.grad_remove(&mut self.0));
        Ok(grad.map(|inner| DynTensor::from_parts(inner, leaf.dtype(), leaf.device())))
    }
}

impl DynTensor {
    /// Marks a float tensor as a **param leaf**: ops on it are recorded, and
    /// [`DynTensor::backward`] keeps its gradient for [`Gradients::get`]. The values and the
    /// device are unchanged. A tensor that is already tracked (a leaf, or the result of tracked
    /// ops) becomes a new leaf: earlier ops are not differentiated through, as with
    /// [`DynTensor::detach`].
    ///
    /// [`TensorError::UnsupportedOperation`] for integer and bool tensors, which have no
    /// gradient (graph-ir.md §3.1.11).
    pub fn require_grad(self) -> Result<Self> {
        let dtype = self.dtype();
        if !dtype.is_float() {
            return Err(TensorError::UnsupportedOperation {
                op: "require_grad",
                dtype,
            });
        }
        let (inner, dtype, device) = self.into_parts();
        // `detach` first: Burn's `require_grad` panics on a tracked non-leaf.
        let inner = map_rank!(inner, t => match t {
            Ranked::Float(t) => Ranked::Float(t.autodiff().detach().require_grad()),
            other => other,
        });
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// Whether the tensor takes part in a recorded graph: a leaf from
    /// [`DynTensor::require_grad`], or the float result of an op on a tracked tensor. Integer and
    /// bool tensors are never tracked.
    pub fn is_tracked(&self) -> bool {
        with_rank!(self.inner_ref(), t => match t {
            Ranked::Float(t) => t.is_tracked(),
            _ => false,
        })
    }

    /// Whether the tensor is a param leaf from [`DynTensor::require_grad`] (or a clone of one),
    /// whose gradient [`Gradients::get`] returns.
    pub fn is_grad_leaf(&self) -> bool {
        with_rank!(self.inner_ref(), t => match t {
            Ranked::Float(t) => t.is_autodiff() && t.is_require_grad(),
            _ => false,
        })
    }

    /// Back-propagates from a one-element float tensor (a loss) and returns the gradients of
    /// every leaf it depends on. The gradient of the loss itself is `1`.
    ///
    /// Errors, all [`TensorError::InvalidArgument`] unless noted:
    /// - the tensor is not float ([`TensorError::UnsupportedOperation`]);
    /// - it has more than one element (Burn would silently differentiate the sum; PyTorch
    ///   rejects it too);
    /// - it is not tracked (it depends on no leaf);
    /// - the graph was already consumed by an earlier `backward` through the same ops. Burn
    ///   0.22.1 has no `retain_graph` and panics there; the panic is caught and turned into this
    ///   error. The graph's state is unchanged by the failed call (Burn checks before it consumes
    ///   anything), so fresh forwards from the same leaves still work. Burn's panic message is
    ///   still printed by the panic hook. With `panic = "abort"` the process would abort instead;
    ///   Polypous builds unwind.
    pub fn backward(&self) -> Result<Gradients> {
        const OP: &str = "backward";
        let dtype = self.dtype();
        if !dtype.is_float() {
            return Err(TensorError::UnsupportedOperation { op: OP, dtype });
        }
        if self.shape().iter().product::<usize>() != 1 {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "the loss must have one element, got shape {:?}",
                    self.shape()
                ),
            });
        }
        if !self.is_tracked() {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: "the tensor is not tracked: it depends on no tensor from `require_grad`"
                    .into(),
            });
        }
        let run = || {
            with_rank!(self.inner_ref(), t => match t {
                Ranked::Float(t) => Some(t.backward()),
                _ => None,
            })
        };
        match catch_unwind(AssertUnwindSafe(run)) {
            Ok(Some(grads)) => Ok(Gradients(grads)),
            // The dtype check above proved the tensor is a float tensor.
            Ok(None) => Err(TensorError::Backend {
                message: "non-float tensor in backward".into(),
            }),
            Err(panic) => Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "the graph was already consumed by an earlier backward (one backward per \
                     forward; Burn: {})",
                    panic_message(panic.as_ref())
                ),
            }),
        }
    }

    /// Returns the tensor cut off from its graph: ops on the result are not differentiated
    /// through to earlier ops. Values, dtype and device are unchanged. A tracked tensor keeps its
    /// autodiff context as an untracked constant (cheap to mix into tracked ops); integer, bool
    /// and untracked tensors are returned unchanged.
    pub fn detach(self) -> Self {
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => match t {
            // Burn's `detach` alone keeps `require_grad` on a leaf (a new leaf).
            Ranked::Float(t) => Ranked::Float(t.detach().set_require_grad(false)),
            other => other,
        });
        DynTensor::from_parts(inner, dtype, device)
    }

    /// Returns the tensor without autodiff context: nothing done with the result is recorded,
    /// and ops on it pay no autodiff dispatch (optimizer updates, inference). Values, dtype and
    /// device are unchanged; for tensors without context this is the identity.
    pub fn without_autodiff(self) -> Self {
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => match t {
            Ranked::Float(t) => Ranked::Float(t.without_autodiff()),
            Ranked::Int(t) => Ranked::Int(t.without_autodiff()),
            Ranked::Bool(t) => Ranked::Bool(t.without_autodiff()),
        });
        DynTensor::from_parts(inner, dtype, device)
    }

    /// Whether the tensor carries an autodiff context, tracked or not (a leaf, a result of tracked
    /// ops, a [`DynTensor::detach`]ed copy, or a non-float result of ops on tracked tensors).
    /// Gradients and [`DynTensor::without_autodiff`] results carry none.
    pub fn has_autodiff(&self) -> bool {
        with_rank!(self.inner_ref(), t => match t {
            Ranked::Float(t) => t.is_autodiff(),
            Ranked::Int(t) => t.is_autodiff(),
            Ranked::Bool(t) => t.is_autodiff(),
        })
    }

    /// `InvalidArgument` unless the tensor is a param leaf.
    fn check_leaf(&self) -> Result<()> {
        if self.is_grad_leaf() {
            return Ok(());
        }
        let what = if !self.dtype().is_float() {
            format!("a {} tensor", self.dtype())
        } else if self.is_tracked() {
            "the result of an op, not a leaf".to_string()
        } else {
            "an untracked tensor".to_string()
        };
        Err(TensorError::InvalidArgument {
            op: "gradient",
            reason: format!(
                "gradients are kept only for leaves from `require_grad`; this is {what}"
            ),
        })
    }
}

/// The text of a caught panic.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic".to_string()
    }
}

#[cfg(test)]
#[path = "autodiff_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "autodiff_spike_tests.rs"]
mod spike;
