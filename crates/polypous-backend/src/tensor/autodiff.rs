//! Crate-internal autodiff on [`DynTensor`] (spike for #71; the API #75 builds on).
//!
//! Burn 0.22.1 records operations when a tensor carries an autodiff context
//! (`burn-tensor` `tensor/api/autodiff.rs`). The context lives on the **tensor**, not only on the
//! device: `Tensor::autodiff()` wraps a tensor of a plain device, and its device then reads as
//! `DispatchDevice::Autodiff(inner)`. A `DynTensor` therefore keeps its [`DeviceId`] unchanged
//! when it is tracked (`device_id_of` sees through the autodiff wrapper), and every op that
//! builds helper tensors with `t.device()` builds them on the autodiff device, as untracked
//! constants.
//!
//! Nothing here is public: no Burn type leaves the crate (ADR-0007), and the public API is
//! settled by ADR-0014 / #75.
//!
//! [`DeviceId`]: crate::device::DeviceId

use burn::tensor::Tensor;

use super::{DynTensor, Inner, Ranked, Result, TensorError, map_rank, with_rank};

/// The gradients of one backward pass. Wraps Burn's container; crate-internal.
pub(crate) struct Grads(burn::tensor::Gradients);

impl DynTensor {
    /// Marks a float tensor as a leaf whose gradient is kept by [`DynTensor::backward`]. A tensor
    /// on a plain device gets an autodiff context first. Non-float tensors are
    /// `UnsupportedOperation`; a tracked non-leaf is detached first (Burn would panic).
    pub(crate) fn require_grad(self) -> Result<Self> {
        let dtype = self.dtype();
        if !dtype.is_float() {
            return Err(TensorError::UnsupportedOperation {
                op: "require_grad",
                dtype,
            });
        }
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => match t {
            Ranked::Float(t) => Ranked::Float(t.autodiff().detach().require_grad()),
            other => other,
        });
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// Whether the tensor carries an autodiff context (ops on it are recorded if it is tracked).
    pub(crate) fn is_autodiff(&self) -> bool {
        with_rank!(&self.inner_ref(), t => match t {
            Ranked::Float(t) => t.is_autodiff(),
            Ranked::Int(t) => t.is_autodiff(),
            Ranked::Bool(t) => t.is_autodiff(),
        })
    }

    /// Whether the tensor takes part in a recorded graph.
    pub(crate) fn is_tracked(&self) -> bool {
        with_rank!(&self.inner_ref(), t => match t {
            Ranked::Float(t) => t.is_tracked(),
            _ => false,
        })
    }

    /// Back-propagates from a tracked float tensor with one element (a loss). Burn seeds the
    /// backward pass with ones, so a tensor with more elements would compute the gradient of its
    /// sum; that is rejected here, as in PyTorch. Burn panics on untracked tensors: checked first.
    pub(crate) fn backward(&self) -> Result<Grads> {
        if !self.dtype().is_float() {
            return Err(TensorError::UnsupportedOperation {
                op: "backward",
                dtype: self.dtype(),
            });
        }
        let elements: usize = self.shape().iter().product();
        if elements != 1 {
            return Err(TensorError::InvalidArgument {
                op: "backward",
                reason: format!(
                    "the loss must have one element, got shape {:?}",
                    self.shape()
                ),
            });
        }
        if !self.is_tracked() {
            return Err(TensorError::InvalidArgument {
                op: "backward",
                reason: "the tensor does not depend on a tensor that requires a gradient".into(),
            });
        }
        let grads = with_rank!(&self.inner_ref(), t => match t {
            Ranked::Float(t) => float_backward(t),
            _ => unreachable_kind(),
        })?;
        Ok(Grads(grads))
    }

    /// The gradient of a leaf from [`DynTensor::require_grad`], without autodiff context, in the
    /// tensor's dtype. `None` if the tensor was not part of the backward pass.
    pub(crate) fn grad(&self, grads: &Grads) -> Option<DynTensor> {
        let (inner, dtype, device) = (self.inner_ref(), self.dtype(), self.device());
        let grad: Option<Inner> = match inner {
            Inner::R1(Ranked::Float(t)) => t.grad(&grads.0).map(|g| Inner::R1(Ranked::Float(g))),
            Inner::R2(Ranked::Float(t)) => t.grad(&grads.0).map(|g| Inner::R2(Ranked::Float(g))),
            Inner::R3(Ranked::Float(t)) => t.grad(&grads.0).map(|g| Inner::R3(Ranked::Float(g))),
            Inner::R4(Ranked::Float(t)) => t.grad(&grads.0).map(|g| Inner::R4(Ranked::Float(g))),
            Inner::R5(Ranked::Float(t)) => t.grad(&grads.0).map(|g| Inner::R5(Ranked::Float(g))),
            Inner::R6(Ranked::Float(t)) => t.grad(&grads.0).map(|g| Inner::R6(Ranked::Float(g))),
            _ => None,
        };
        grad.map(|inner| DynTensor::from_parts(inner, dtype, device))
    }

    /// Starts a new graph lineage: earlier ops are not differentiated through. Keeps the autodiff
    /// context. Non-float tensors are returned unchanged.
    pub(crate) fn detach(self) -> Self {
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => match t {
            Ranked::Float(t) => Ranked::Float(t.detach()),
            other => other,
        });
        DynTensor::from_parts(inner, dtype, device)
    }

    /// Drops the autodiff context: ops on the result are not recorded (optimizer updates).
    pub(crate) fn without_autodiff(self) -> Self {
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => match t {
            Ranked::Float(t) => Ranked::Float(t.without_autodiff()),
            Ranked::Int(t) => Ranked::Int(t.without_autodiff()),
            Ranked::Bool(t) => Ranked::Bool(t.without_autodiff()),
        });
        DynTensor::from_parts(inner, dtype, device)
    }

    /// Adds an autodiff context without requiring a gradient (an untracked constant).
    pub(crate) fn with_autodiff(self) -> Self {
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => match t {
            Ranked::Float(t) => Ranked::Float(t.autodiff()),
            Ranked::Int(t) => Ranked::Int(t.autodiff()),
            Ranked::Bool(t) => Ranked::Bool(t.autodiff()),
        });
        DynTensor::from_parts(inner, dtype, device)
    }
}

fn float_backward<const D: usize>(t: &Tensor<D>) -> Result<burn::tensor::Gradients> {
    Ok(t.backward())
}

fn unreachable_kind() -> Result<burn::tensor::Gradients> {
    // The dtype check above proved the tensor is a float tensor.
    Err(TensorError::Backend {
        message: "non-float tensor in backward".into(),
    })
}

#[cfg(test)]
#[path = "autodiff_spike_tests.rs"]
mod spike;
