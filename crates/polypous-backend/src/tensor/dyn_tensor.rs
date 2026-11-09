//! [`DynTensor`]: a tensor whose rank and dtype are runtime values.
//!
//! See the module documentation of [`crate::tensor`] for how the rank/kind dispatch works.

use std::fmt;

use burn::tensor::{Bool, BoolStore, DType as BurnDType, Device, Float, Int, Tensor};

use super::data::num_elements;
use super::{DType, Kind, Result, TensorData, TensorError};
use crate::device::{DeviceId, device_id_of, resolve_device};

/// Highest supported rank.
const MAX_RANK: usize = 6;

/// A Burn tensor of compile-time rank `D`, with the tensor kind as a runtime value.
#[derive(Clone)]
pub(crate) enum Ranked<const D: usize> {
    Float(Tensor<D, Float>),
    Int(Tensor<D, Int>),
    Bool(Tensor<D, Bool>),
}

/// One variant per supported rank. The variant *is* the rank.
#[derive(Clone)]
pub(crate) enum Inner {
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
            $crate::tensor::Inner::R1($t) => $body,
            $crate::tensor::Inner::R2($t) => $body,
            $crate::tensor::Inner::R3($t) => $body,
            $crate::tensor::Inner::R4($t) => $body,
            $crate::tensor::Inner::R5($t) => $body,
            $crate::tensor::Inner::R6($t) => $body,
        }
    };
}

/// Like [`with_rank!`], but `$body` returns a `Ranked` of the same rank, which is wrapped back
/// into an `Inner`.
macro_rules! map_rank {
    ($inner:expr, $t:ident => $body:expr) => {
        match $inner {
            $crate::tensor::Inner::R1($t) => $crate::tensor::Inner::R1($body),
            $crate::tensor::Inner::R2($t) => $crate::tensor::Inner::R2($body),
            $crate::tensor::Inner::R3($t) => $crate::tensor::Inner::R3($body),
            $crate::tensor::Inner::R4($t) => $crate::tensor::Inner::R4($body),
            $crate::tensor::Inner::R5($t) => $crate::tensor::Inner::R5($body),
            $crate::tensor::Inner::R6($t) => $crate::tensor::Inner::R6($body),
        }
    };
}
pub(crate) use map_rank;

/// Binary version of [`map_rank!`]. Callers check that both ranks are equal first; should they
/// differ anyway, the enclosing function returns a `ShapeMismatch` error instead of panicking.
macro_rules! zip_rank {
    ($op:expr, $lhs:expr, $rhs:expr, $a:ident, $b:ident => $body:expr) => {
        match ($lhs, $rhs) {
            ($crate::tensor::Inner::R1($a), $crate::tensor::Inner::R1($b)) => {
                $crate::tensor::Inner::R1($body)
            }
            ($crate::tensor::Inner::R2($a), $crate::tensor::Inner::R2($b)) => {
                $crate::tensor::Inner::R2($body)
            }
            ($crate::tensor::Inner::R3($a), $crate::tensor::Inner::R3($b)) => {
                $crate::tensor::Inner::R3($body)
            }
            ($crate::tensor::Inner::R4($a), $crate::tensor::Inner::R4($b)) => {
                $crate::tensor::Inner::R4($body)
            }
            ($crate::tensor::Inner::R5($a), $crate::tensor::Inner::R5($b)) => {
                $crate::tensor::Inner::R5($body)
            }
            ($crate::tensor::Inner::R6($a), $crate::tensor::Inner::R6($b)) => {
                $crate::tensor::Inner::R6($body)
            }
            (a, b) => {
                return Err($crate::tensor::TensorError::ShapeMismatch {
                    op: $op,
                    lhs: a.dims(),
                    rhs: b.dims(),
                });
            }
        }
    };
}
pub(crate) use zip_rank;

/// Turns a runtime rank into a compile-time one: evaluates `$body` with the constant `$n` set to
/// `$rank` and wraps the resulting `Ranked<$n>` into an `Inner`. Ranks outside 1–6 return
/// `UnsupportedRank` from the enclosing function.
macro_rules! dispatch_rank {
    ($rank:expr, $n:ident => $body:expr) => {
        match $rank {
            1 => {
                const $n: usize = 1;
                $crate::tensor::Inner::R1($body)
            }
            2 => {
                const $n: usize = 2;
                $crate::tensor::Inner::R2($body)
            }
            3 => {
                const $n: usize = 3;
                $crate::tensor::Inner::R3($body)
            }
            4 => {
                const $n: usize = 4;
                $crate::tensor::Inner::R4($body)
            }
            5 => {
                const $n: usize = 5;
                $crate::tensor::Inner::R5($body)
            }
            6 => {
                const $n: usize = 6;
                $crate::tensor::Inner::R6($body)
            }
            rank => return Err($crate::tensor::TensorError::UnsupportedRank { rank }),
        }
    };
}

/// A tensor with runtime rank (1–6) and runtime dtype, on any device of any compiled-in backend.
///
/// It wraps a `burn::Tensor`, so the device it was created on selects the backend (ADR-0011). No
/// Burn type appears in its public API (ADR-0007): data goes in and out as [`TensorData`], and
/// devices are addressed by [`DeviceId`], as returned by
/// [`list_devices`](crate::device::list_devices).
///
/// Operations check dtypes, shapes and devices first and return a [`TensorError`] instead of
/// letting Burn panic. Cloning is cheap: Burn tensors share their buffer.
#[derive(Clone)]
pub struct DynTensor {
    inner: Inner,
    dtype: DType,
    /// The id of the Burn device `inner` lives on, mapped once when the tensor is created.
    device: DeviceId,
}

impl DynTensor {
    /// Highest supported rank.
    pub const MAX_RANK: usize = MAX_RANK;

    /// Uploads host data to `device`.
    ///
    /// Fails if the data is malformed, the rank is not 1–6, the device is not available (its
    /// backend is not compiled in, or no such device exists), or the device's backend doesn't
    /// support the dtype.
    pub fn from_data(data: &TensorData, device: DeviceId) -> Result<Self> {
        Self::from_data_burn(data, &resolve(device)?)
    }

    /// [`DynTensor::from_data`] on a Burn device. The device must have a [`DeviceId`] (one of
    /// [`resolve_device`]'s results always has).
    pub(crate) fn from_data_burn(data: &TensorData, device: &Device) -> Result<Self> {
        data.validate()?;
        let dims = data.dims()?;
        let id = id_of(device)?;
        check_device_supports(data.dtype, device, id)?;

        let (bytes, burn_dtype) = if data.dtype == DType::Bool {
            // Burn creates bool tensors in the device's bool storage, but its CubeCL backends
            // (Burn 0.22.1, `bool_from_data`) panic unless the data already has that storage.
            let store = bool_store(device);
            (encode_bool(&data.bytes, store), BurnDType::Bool(store))
        } else {
            let mut bytes = data.bytes.clone();
            le_to_native(&mut bytes, data.dtype.size_bytes());
            (bytes, data.dtype.to_burn())
        };
        let burn_data =
            burn::tensor::TensorData::try_from_bytes_vec(bytes, dims.clone(), burn_dtype).map_err(
                |err| TensorError::InvalidData {
                    reason: err.to_string(),
                },
            )?;

        let dtype = data.dtype;
        let options = (device, burn_dtype);
        let inner = dispatch_rank!(dims.len(), N => match dtype.kind() {
            Kind::Float => Ranked::<N>::Float(Tensor::from_data(burn_data, options)),
            Kind::Int => Ranked::<N>::Int(Tensor::from_data(burn_data, options)),
            Kind::Bool => Ranked::<N>::Bool(Tensor::from_data(burn_data, device)),
        });
        Ok(DynTensor {
            inner,
            dtype,
            device: id,
        })
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
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The Burn device the tensor lives on.
    pub(crate) fn burn_device(&self) -> Device {
        with_rank!(&self.inner, t => t.device())
    }

    /// Whether `device` can store and compute `dtype`. `bool` is supported everywhere.
    ///
    /// Ops on a tensor that exists on a device work for its dtype, except where an op's
    /// documentation names a device restriction. Fails if the device is not available (see
    /// [`DynTensor::from_data`]).
    pub fn device_supports(device: DeviceId, dtype: DType) -> Result<bool> {
        let burn = resolve(device)?;
        Ok(check_device_supports(dtype, &burn, device).is_ok())
    }

    /// Builds a tensor from its parts. `inner` must live on the Burn device of `device` and hold
    /// a tensor of `dtype`'s kind; the ops module keeps this invariant.
    pub(crate) fn from_parts(inner: Inner, dtype: DType, device: DeviceId) -> Self {
        DynTensor {
            inner,
            dtype,
            device,
        }
    }

    /// The rank/kind enum, the dtype and the device id.
    pub(crate) fn into_parts(self) -> (Inner, DType, DeviceId) {
        (self.inner, self.dtype, self.device)
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
        Ok(DynTensor {
            inner,
            dtype,
            device: self.device,
        })
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
        let (dtype, device) = (self.dtype, self.device);
        let inner = zip_rank!(OP, self.inner, other.inner, a, b => a.matmul(b)?);
        Ok(DynTensor {
            inner,
            dtype,
            device,
        })
    }

    /// Returns a copy of the tensor on `device`. Copies between backends go through host memory.
    ///
    /// Fails if the device is not available (see [`DynTensor::from_data`]) or its backend doesn't
    /// support the dtype.
    pub fn to_device(&self, device: DeviceId) -> Result<Self> {
        self.to_device_burn(&resolve(device)?)
    }

    /// [`DynTensor::to_device`] on a Burn device. The device must have a [`DeviceId`].
    pub(crate) fn to_device_burn(&self, device: &Device) -> Result<Self> {
        let id = id_of(device)?;
        check_device_supports(self.dtype, device, id)?;
        if self.dtype == DType::Bool && bool_store(&self.burn_device()) != bool_store(device) {
            // Burn 0.22.1 hands the source's bool storage to the target backend, and the CubeCL
            // backends panic on storage that isn't their own. Go through host memory ourselves.
            return DynTensor::from_data_burn(&self.to_data()?, device);
        }
        let dtype = self.dtype;
        let inner = map_rank!(&self.inner, t => t.to_device(device));
        Ok(DynTensor {
            inner,
            dtype,
            device: id,
        })
    }

    /// Checks dtype, rank and device of a binary operation's operands.
    pub(crate) fn check_binary(&self, op: &'static str, other: &Self) -> Result<()> {
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
        let (lhs, rhs) = (self.device, other.device);
        if lhs != rhs {
            return Err(TensorError::DeviceMismatch {
                op,
                lhs: lhs.to_string(),
                rhs: rhs.to_string(),
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
            .field("device", &self.device)
            .finish()
    }
}

impl Inner {
    pub(crate) fn dims(&self) -> Vec<usize> {
        with_rank!(self, t => t.dims())
    }
}

impl<const D: usize> Ranked<D> {
    pub(crate) fn dims(&self) -> Vec<usize> {
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

    pub(crate) fn device(&self) -> Device {
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
    pub(crate) fn reshape<const N: usize>(self, shape: &[usize]) -> Result<Ranked<N>> {
        let dims: [usize; N] = shape
            .try_into()
            .map_err(|_| TensorError::UnsupportedRank { rank: shape.len() })?;
        Ok(match self {
            Ranked::Float(t) => Ranked::Float(t.reshape(dims)),
            Ranked::Int(t) => Ranked::Int(t.reshape(dims)),
            Ranked::Bool(t) => Ranked::Bool(t.reshape(dims)),
        })
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
    pub(crate) fn kind_error(&self, op: &'static str) -> TensorError {
        let dtype = match self {
            Ranked::Float(_) => DType::F32,
            Ranked::Int(_) => DType::I64,
            Ranked::Bool(_) => DType::Bool,
        };
        TensorError::UnsupportedOperation { op, dtype }
    }
}

/// The Burn device for a public [`DeviceId`], or `DeviceUnavailable`. See `device/resolve.rs` for
/// why unknown ids never reach Burn.
fn resolve(id: DeviceId) -> Result<Device> {
    resolve_device(id).map_err(|reason| TensorError::DeviceUnavailable { device: id, reason })
}

/// The [`DeviceId`] of a Burn device. Devices from [`resolve_device`] always have one; a Burn
/// device of a kind Polypous doesn't use (for example Burn's default device) is an error.
fn id_of(device: &Device) -> Result<DeviceId> {
    device_id_of(device).ok_or_else(|| TensorError::Backend {
        message: format!("{device:?} is not a device Polypous can address"),
    })
}

/// Fails if `device`'s backend can't store or compute `dtype`. `bool` is always supported: Burn
/// stores it in whatever form the device has.
pub(crate) fn check_device_supports(dtype: DType, device: &Device, id: DeviceId) -> Result<()> {
    if dtype == DType::Bool || device.supports_dtype(dtype.to_burn()) {
        Ok(())
    } else {
        Err(TensorError::UnsupportedDType {
            dtype,
            device: id.to_string(),
        })
    }
}

/// How `device` stores bool tensors (1-byte native, `u8` or `u32`).
pub(crate) fn bool_store(device: &Device) -> BoolStore {
    device.settings().bool_dtype
}

/// Encodes host bools (one byte, 0 or 1) in the given Burn bool storage, in host byte order.
fn encode_bool(bytes: &[u8], store: BoolStore) -> Vec<u8> {
    match store {
        BoolStore::Native | BoolStore::U8 => bytes.to_vec(),
        BoolStore::U32 => bytes
            .iter()
            .flat_map(|&b| u32::from(b).to_ne_bytes())
            .collect(),
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
