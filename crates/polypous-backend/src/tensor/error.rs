//! Errors of the `tensor` module.

use super::DType;
use crate::device::DeviceId;

/// Error from creating, converting or operating on a [`DynTensor`](super::DynTensor).
///
/// Every check runs before Burn is called, so bad input gives one of these errors instead of a
/// Burn panic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TensorError {
    /// The rank is outside the supported range 1–6 (`DynTensor::MAX_RANK`).
    #[error("tensor rank {rank} is not supported (supported ranks are 1 to 6)")]
    UnsupportedRank {
        /// The requested rank.
        rank: usize,
    },

    /// The operands of an operation have different dtypes.
    #[error("dtype mismatch in `{op}`: {lhs} and {rhs}")]
    DTypeMismatch {
        /// The operation.
        op: &'static str,
        /// Dtype of the first operand.
        lhs: DType,
        /// Dtype of the second operand.
        rhs: DType,
    },

    /// The shapes of the operands (or the requested shape) don't fit the operation.
    #[error("shape mismatch in `{op}`: {lhs:?} and {rhs:?}")]
    ShapeMismatch {
        /// The operation.
        op: &'static str,
        /// Shape of the first operand (for `reshape`: the tensor's shape).
        lhs: Vec<usize>,
        /// Shape of the second operand (for `reshape`: the requested shape).
        rhs: Vec<usize>,
    },

    /// The operation is not defined for this dtype (for example `matmul` on `bool`).
    #[error("`{op}` is not supported for dtype {dtype}")]
    UnsupportedOperation {
        /// The operation.
        op: &'static str,
        /// The dtype of the operand.
        dtype: DType,
    },

    /// The backend of the target device can't store or compute this dtype (for example `bf16`
    /// on wgpu/Vulkan).
    #[error("dtype {dtype} is not supported on device {device}")]
    UnsupportedDType {
        /// The dtype.
        dtype: DType,
        /// The device, as its [`DeviceId`] text (for example `vulkan:discrete_gpu:0`).
        device: String,
    },

    /// The device can't be used: its backend is not compiled in, or the backend has no such
    /// device (for example an index past the last GPU). Use an id from `list_devices()`.
    #[error("device {device} is not available: {reason}")]
    DeviceUnavailable {
        /// The requested device.
        device: DeviceId,
        /// Why it can't be used.
        reason: String,
    },

    /// The operands of an operation live on different devices. Move one with `to_device` first.
    #[error("tensors of `{op}` are on different devices: {lhs} and {rhs}")]
    DeviceMismatch {
        /// The operation.
        op: &'static str,
        /// Device of the first operand, as its [`DeviceId`] text.
        lhs: String,
        /// Device of the second operand, as its [`DeviceId`] text.
        rhs: String,
    },

    /// The `TensorData` is malformed: byte length doesn't match the shape and dtype, the element
    /// count overflows, or a `bool` byte is not `0` or `1`.
    #[error("invalid tensor data: {reason}")]
    InvalidData {
        /// What is wrong.
        reason: String,
    },

    /// The backend failed while running or reading back a tensor.
    #[error("backend error: {message}")]
    Backend {
        /// The backend's message.
        message: String,
    },
}

/// Result type of the `tensor` module.
pub type Result<T, E = TensorError> = std::result::Result<T, E>;
