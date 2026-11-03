//! [`DynTensor`]: a tensor whose rank and dtype are runtime values.
//!
//! See the module documentation of [`crate::tensor`] for how the rank/kind dispatch works.

use std::fmt;

use burn::tensor::{Bool, DType as BurnDType, Device, Float, Int, Tensor};

use super::data::num_elements;
use super::{DType, Kind, Result, TensorData, TensorError};

/// Highest supported rank.
const MAX_RANK: usize = 6;

/// A Burn tensor of compile-time rank `D`, with the tensor kind as a runtime value.
#[derive(Clone)]
enum Ranked<const D: usize> {
    Float(Tensor<D, Float>),
    Int(Tensor<D, Int>),
    Bool(Tensor<D, Bool>),
}

/// One variant per supported rank. The variant *is* the rank.
#[derive(Clone)]
enum Inner {
    R1(Ranked<1>),
    R2(Ranked<2>),
    R3(Ranked<3>),
    R4(Ranked<4>),
    R5(Ranked<5>),
    R6(Ranked<6>),
}

/// Runs `$body` with `$t` bound to the `Ranked<D>` inside an `Inner`, for whichever rank it has.
/// The body is instantiated six times, once per rank.
macro_rules! with_rank {
    ($inner:expr, $t:ident => $body:expr) => {
        match $inner {
            Inner::R1($t) => $body,
            Inner::R2($t) => $body,
            Inner::R3($t) => $body,
            Inner::R4($t) => $body,
            Inner::R5($t) => $body,
            Inner::R6($t) => $body,
        }
    };
}

/// Like [`with_rank!`], but `$body` returns a `Ranked` of the same rank, which is wrapped back
/// into an `Inner`.
macro_rules! map_rank {
    ($inner:expr, $t:ident => $body:expr) => {
        match $inner {
            Inner::R1($t) => Inner::R1($body),
            Inner::R2($t) => Inner::R2($body),
            Inner::R3($t) => Inner::R3($body),
            Inner::R4($t) => Inner::R4($body),
            Inner::R5($t) => Inner::R5($body),
            Inner::R6($t) => Inner::R6($body),
        }
    };
}

/// Binary version of [`map_rank!`]. Callers check that both ranks are equal first; should they
/// differ anyway, the enclosing function returns a `ShapeMismatch` error instead of panicking.
macro_rules! zip_rank {
    ($op:expr, $lhs:expr, $rhs:expr, $a:ident, $b:ident => $body:expr) => {
        match ($lhs, $rhs) {
            (Inner::R1($a), Inner::R1($b)) => Inner::R1($body),
            (Inner::R2($a), Inner::R2($b)) => Inner::R2($body),
            (Inner::R3($a), Inner::R3($b)) => Inner::R3($body),
            (Inner::R4($a), Inner::R4($b)) => Inner::R4($body),
            (Inner::R5($a), Inner::R5($b)) => Inner::R5($body),
            (Inner::R6($a), Inner::R6($b)) => Inner::R6($body),
            (a, b) => {
                return Err(TensorError::ShapeMismatch {
                    op: $op,
                    lhs: a.dims(),
                    rhs: b.dims(),
                });
            }
        }
    };
}

/// Turns a runtime rank into a compile-time one: evaluates `$body` with the constant `$n` set to
/// `$rank` and wraps the resulting `Ranked<$n>` into an `Inner`. Ranks outside 1–6 return
/// `UnsupportedRank` from the enclosing function.
macro_rules! dispatch_rank {
    ($rank:expr, $n:ident => $body:expr) => {
        match $rank {
            1 => {
                const $n: usize = 1;
                Inner::R1($body)
            }
            2 => {
                const $n: usize = 2;
                Inner::R2($body)
            }
            3 => {
                const $n: usize = 3;
                Inner::R3($body)
            }
            4 => {
                const $n: usize = 4;
                Inner::R4($body)
            }
            5 => {
                const $n: usize = 5;
                Inner::R5($body)
            }
            6 => {
                const $n: usize = 6;
                Inner::R6($body)
            }
            rank => return Err(TensorError::UnsupportedRank { rank }),
        }
    };
}

/// A tensor with runtime rank (1–6) and runtime dtype, on any device of any compiled-in backend.
///
/// It wraps a `burn::Tensor`, so the device it was created on selects the backend (ADR-0011). No
/// Burn type appears in its public API (ADR-0007): data goes in and out as [`TensorData`], and
/// the device-taking functions are crate-internal until the device API (issue #4) exists.
///
/// Operations check dtypes, shapes and devices first and return a [`TensorError`] instead of
/// letting Burn panic. Cloning is cheap: Burn tensors share their buffer.
#[derive(Clone)]
pub struct DynTensor {
    inner: Inner,
    dtype: DType,
}

impl DynTensor {
    /// Highest supported rank.
    pub const MAX_RANK: usize = MAX_RANK;

    /// Uploads host data to `device`.
    ///
    /// Fails if the data is malformed, the rank is not 1–6, or the device's backend doesn't
    /// support the dtype.
    #[cfg_attr(
        not(all(test, feature = "cpu")),
        expect(dead_code, reason = "used once the device API (issue #4) exists")
    )]
    pub(crate) fn from_data(data: &TensorData, device: &Device) -> Result<Self> {
        data.validate()?;
        let dims = data.dims()?;
        check_device_supports(data.dtype, device)?;

        let mut bytes = data.bytes.clone();
        le_to_native(&mut bytes, data.dtype.size_bytes());
        let burn_data =
            burn::tensor::TensorData::try_from_bytes_vec(bytes, dims.clone(), data.dtype.to_burn())
                .map_err(|err| TensorError::InvalidData {
                    reason: err.to_string(),
                })?;

        let dtype = data.dtype;
        let options = (device, dtype.to_burn());
        let inner = dispatch_rank!(dims.len(), N => match dtype.kind() {
            Kind::Float => Ranked::<N>::Float(Tensor::from_data(burn_data, options)),
            Kind::Int => Ranked::<N>::Int(Tensor::from_data(burn_data, options)),
            // Burn picks the device's bool storage for bool tensors itself.
            Kind::Bool => Ranked::<N>::Bool(Tensor::from_data(burn_data, device)),
        });
        Ok(DynTensor { inner, dtype })
    }

    /// Reads the tensor back into host memory.
    pub fn into_data(self) -> Result<TensorData> {
        let dtype = self.dtype;
        let burn_data = with_rank!(self.inner, t => t.into_burn_data())?;
        let shape = burn_data.shape().iter().map(|&dim| dim as u64).collect();
        let (bytes, _, stored) = burn_data.into_parts();

        let bytes = match (dtype, stored) {
            // Devices store bools as 1, 8 or 32 bits; the host layout is one byte, 0 or 1.
            (DType::Bool, BurnDType::Bool(store)) => {
                let width = BurnDType::Bool(store).size();
                bytes
                    .chunks_exact(width)
                    .map(|element| u8::from(element.iter().any(|&byte| byte != 0)))
                    .collect()
            }
            (dtype, stored) if stored == dtype.to_burn() => {
                let mut bytes = bytes.to_vec();
                native_to_le(&mut bytes, dtype.size_bytes());
                bytes
            }
            (dtype, stored) => {
                return Err(TensorError::Backend {
                    message: format!(
                        "backend returned {stored:?} data for a tensor of dtype {dtype}"
                    ),
                });
            }
        };
        TensorData::new(dtype, shape, bytes)
    }

    /// Reads the tensor back into host memory, keeping the tensor.
    pub fn to_data(&self) -> Result<TensorData> {
        self.clone().into_data()
    }

    /// The dimensions, outermost first.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims()
    }

    /// The number of dimensions (1–6).
    pub fn rank(&self) -> usize {
        with_rank!(&self.inner, t => t.rank())
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device the tensor lives on.
    pub(crate) fn device(&self) -> Device {
        with_rank!(&self.inner, t => t.device())
    }

    /// Returns a tensor with the same elements in row-major order and a new shape (rank 1–6).
    ///
    /// Fails if the element counts differ. Dimensions are given explicitly; there is no `-1`
    /// wildcard.
    pub fn reshape(self, shape: &[usize]) -> Result<Self> {
        let current = self.shape();
        if num_elements(shape) != num_elements(&current) {
            return Err(TensorError::ShapeMismatch {
                op: "reshape",
                lhs: current,
                rhs: shape.to_vec(),
            });
        }
        let dtype = self.dtype;
        let inner = with_rank!(self.inner, t => {
            dispatch_rank!(shape.len(), N => t.reshape::<N>(shape)?)
        });
        Ok(DynTensor { inner, dtype })
    }

    /// Converts the elements to `dtype`.
    ///
    /// - float → int rounds toward zero; out-of-range values are backend-defined.
    /// - any → `bool` gives `true` for non-zero elements (NaN is non-zero).
    /// - `bool` → float or int gives `1` and `0`.
    ///
    /// Fails if the device's backend doesn't support `dtype`.
    pub fn cast(self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            return Ok(self);
        }
        let device = self.device();
        check_device_supports(dtype, &device)?;
        let inner = map_rank!(self.inner, t => t.cast(dtype));
        Ok(DynTensor { inner, dtype })
    }

    /// Element-wise sum, with broadcasting.
    ///
    /// Not `std::ops::Add`, because it can fail.
    ///
    /// Both tensors need the same dtype (float or int), the same rank and the same device. Two
    /// dimensions are compatible if they are equal or one of them is `1`.
    #[expect(
        clippy::should_implement_trait,
        reason = "fallible, so `std::ops::Add` does not fit"
    )]
    pub fn add(self, other: Self) -> Result<Self> {
        const OP: &str = "add";
        self.check_binary(OP, &other)?;
        if self.dtype.kind() == Kind::Bool {
            return Err(TensorError::UnsupportedOperation {
                op: OP,
                dtype: self.dtype,
            });
        }
        let (lhs, rhs) = (self.shape(), other.shape());
        if broadcast(&lhs, &rhs).is_none() {
            return Err(TensorError::ShapeMismatch { op: OP, lhs, rhs });
        }
        let dtype = self.dtype;
        let inner = zip_rank!(OP, self.inner, other.inner, a, b => a.add(b)?);
        Ok(DynTensor { inner, dtype })
    }

    /// Matrix product over the last two dimensions; leading dimensions are batch dimensions and
    /// broadcast like in [`DynTensor::add`].
    ///
    /// Both tensors need the same float dtype, the same rank (at least 2) and the same device, and
    /// `self`'s last dimension must equal `other`'s second-to-last. `[.., m, k] @ [.., k, n]` gives
    /// `[.., m, n]`.
    pub fn matmul(self, other: Self) -> Result<Self> {
        const OP: &str = "matmul";
        self.check_binary(OP, &other)?;
        if !self.dtype.is_float() {
            return Err(TensorError::UnsupportedOperation {
                op: OP,
                dtype: self.dtype,
            });
        }
        let (lhs, rhs) = (self.shape(), other.shape());
        let rank = lhs.len();
        let compatible = rank >= 2
            && lhs[rank - 1] == rhs[rank - 2]
            && broadcast(&lhs[..rank - 2], &rhs[..rank - 2]).is_some();
        if !compatible {
            return Err(TensorError::ShapeMismatch { op: OP, lhs, rhs });
        }
        let dtype = self.dtype;
        let inner = zip_rank!(OP, self.inner, other.inner, a, b => a.matmul(b)?);
        Ok(DynTensor { inner, dtype })
    }

    /// Returns a copy of the tensor on `device`. Copies between backends go through host memory.
    ///
    /// Fails if the target backend doesn't support the dtype.
    #[cfg_attr(
        not(all(test, feature = "cpu")),
        expect(dead_code, reason = "used once the device API (issue #4) exists")
    )]
    pub(crate) fn to_device(&self, device: &Device) -> Result<Self> {
        check_device_supports(self.dtype, device)?;
        let dtype = self.dtype;
        let inner = map_rank!(&self.inner, t => t.to_device(device));
        Ok(DynTensor { inner, dtype })
    }

    /// Checks dtype, rank and device of a binary operation's operands.
    fn check_binary(&self, op: &'static str, other: &Self) -> Result<()> {
        if self.dtype != other.dtype {
            return Err(TensorError::DTypeMismatch {
                op,
                lhs: self.dtype,
                rhs: other.dtype,
            });
        }
        if self.rank() != other.rank() {
            return Err(TensorError::ShapeMismatch {
                op,
                lhs: self.shape(),
                rhs: other.shape(),
            });
        }
        let (lhs, rhs) = (self.device(), other.device());
        if lhs != rhs {
            return Err(TensorError::DeviceMismatch {
                op,
                lhs: format!("{lhs:?}"),
                rhs: format!("{rhs:?}"),
            });
        }
        Ok(())
    }
}

impl fmt::Debug for DynTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynTensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape())
            .field("device", &self.device())
            .finish()
    }
}

impl Inner {
    fn dims(&self) -> Vec<usize> {
        with_rank!(self, t => t.dims())
    }
}

impl<const D: usize> Ranked<D> {
    fn dims(&self) -> Vec<usize> {
        let shape = match self {
            Ranked::Float(t) => t.shape(),
            Ranked::Int(t) => t.shape(),
            Ranked::Bool(t) => t.shape(),
        };
        shape.iter().copied().collect()
    }

    fn rank(&self) -> usize {
        D
    }

    fn device(&self) -> Device {
        match self {
            Ranked::Float(t) => t.device(),
            Ranked::Int(t) => t.device(),
            Ranked::Bool(t) => t.device(),
        }
    }

    fn into_burn_data(self) -> Result<burn::tensor::TensorData> {
        let data = match self {
            Ranked::Float(t) => t.try_into_data(),
            Ranked::Int(t) => t.try_into_data(),
            Ranked::Bool(t) => t.try_into_data(),
        };
        data.map_err(|err| TensorError::Backend {
            message: format!("{err:?}"),
        })
    }

    /// The caller checked that the element counts match.
    fn reshape<const N: usize>(self, shape: &[usize]) -> Result<Ranked<N>> {
        let dims: [usize; N] = shape
            .try_into()
            .map_err(|_| TensorError::UnsupportedRank { rank: shape.len() })?;
        Ok(match self {
            Ranked::Float(t) => Ranked::Float(t.reshape(dims)),
            Ranked::Int(t) => Ranked::Int(t.reshape(dims)),
            Ranked::Bool(t) => Ranked::Bool(t.reshape(dims)),
        })
    }

    /// The caller checked that the device supports `dtype`.
    fn cast(self, dtype: DType) -> Self {
        match (self, dtype.kind()) {
            (Ranked::Float(t), Kind::Float) => Ranked::Float(t.cast(dtype.to_burn_float())),
            (Ranked::Float(t), Kind::Int) => Ranked::Int(t.cast(dtype.to_burn_int())),
            (Ranked::Float(t), Kind::Bool) => Ranked::Bool(t.bool()),
            (Ranked::Int(t), Kind::Float) => Ranked::Float(t.cast(dtype.to_burn_float())),
            (Ranked::Int(t), Kind::Int) => Ranked::Int(t.cast(dtype.to_burn_int())),
            (Ranked::Int(t), Kind::Bool) => Ranked::Bool(t.bool()),
            (Ranked::Bool(t), Kind::Float) => Ranked::Float(t.cast(dtype.to_burn_float())),
            (Ranked::Bool(t), Kind::Int) => Ranked::Int(t.cast(dtype.to_burn_int())),
            (Ranked::Bool(t), Kind::Bool) => Ranked::Bool(t),
        }
    }

    /// The caller checked dtypes, shapes and devices.
    fn add(self, other: Self) -> Result<Self> {
        match (self, other) {
            (Ranked::Float(a), Ranked::Float(b)) => Ok(Ranked::Float(a.add(b))),
            (Ranked::Int(a), Ranked::Int(b)) => Ok(Ranked::Int(a.add(b))),
            (a, _) => Err(a.kind_error("add")),
        }
    }

    /// The caller checked dtypes, shapes and devices.
    fn matmul(self, other: Self) -> Result<Self> {
        match (self, other) {
            (Ranked::Float(a), Ranked::Float(b)) => Ok(Ranked::Float(a.matmul(b))),
            (a, _) => Err(a.kind_error("matmul")),
        }
    }

    /// Burn tensors share their buffer, so the clones are cheap.
    fn to_device(&self, device: &Device) -> Self {
        match self {
            Ranked::Float(t) => Ranked::Float(t.clone().to_device(device)),
            Ranked::Int(t) => Ranked::Int(t.clone().to_device(device)),
            Ranked::Bool(t) => Ranked::Bool(t.clone().to_device(device)),
        }
    }

    /// Error for a kind combination that the public checks should have rejected already.
    fn kind_error(&self, op: &'static str) -> TensorError {
        let dtype = match self {
            Ranked::Float(_) => DType::F32,
            Ranked::Int(_) => DType::I64,
            Ranked::Bool(_) => DType::Bool,
        };
        TensorError::UnsupportedOperation { op, dtype }
    }
}

/// Fails if `device`'s backend can't store or compute `dtype`. `bool` is always supported: Burn
/// stores it in whatever form the device has.
fn check_device_supports(dtype: DType, device: &Device) -> Result<()> {
    if dtype == DType::Bool || device.supports_dtype(dtype.to_burn()) {
        Ok(())
    } else {
        Err(TensorError::UnsupportedDType {
            dtype,
            device: format!("{device:?}"),
        })
    }
}

/// Broadcast shape of two shapes of equal rank (dimensions equal, or one of them `1`), or `None`.
fn broadcast(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    if lhs.len() != rhs.len() {
        return None;
    }
    lhs.iter()
        .zip(rhs)
        .map(|(&l, &r)| match (l, r) {
            _ if l == r => Some(l),
            (1, r) => Some(r),
            (l, 1) => Some(l),
            _ => None,
        })
        .collect()
}

/// Converts little-endian elements of `width` bytes to the host's byte order, in place.
fn le_to_native(bytes: &mut [u8], width: usize) {
    if cfg!(target_endian = "big") && width > 1 {
        bytes
            .chunks_exact_mut(width)
            .for_each(|element| element.reverse());
    }
}

/// Converts elements of `width` bytes from the host's byte order to little-endian, in place.
fn native_to_le(bytes: &mut [u8], width: usize) {
    le_to_native(bytes, width);
}

#[cfg(test)]
#[path = "dyn_tensor_tests.rs"]
mod tests;
