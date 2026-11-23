"""The Polypous random generator and initial values, from the spec's formulas.

A Python implementation of docs/specs/graph-ir.md §3.1.12 (SplitMix64 as a counter-based hash,
`u24`, `U`, `N`, dropout masks) and §2.4.1 (initial values of every `init` kind, in `f64` rounded
once to `f32`), written from the spec, independently of the Rust code in
crates/polypous-ir/src/rng.rs and crates/polypous-train/src/init.rs. The reference fixtures of
testing-strategy.md §3.8 ("Initial values") and §3.10 (dropout masks of training-mode graphs) come
from here.

Standard library only (Python 3.12+), so the unit tests run in CI without PyTorch. `math.log` and
`math.cos` are the platform's, as in Rust (§3.1.12, "Exactness").
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
TWO_POW_M24 = 2.0**-24
TWO_POW_M53 = 2.0**-53
DROPOUT_PREFIX = "dropout:"
INIT_PREFIX = "init:"


def mix(z: int) -> int:
    """SplitMix64's finalizer, wrapping `u64` arithmetic."""
    z &= MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def fnv1a(label: str) -> int:
    """64-bit FNV-1a of the label's bytes (labels are ASCII)."""
    h = 0xCBF29CE484222325
    for b in label.encode("ascii"):
        h = ((h ^ b) * 0x00000100000001B3) & MASK64
    return h


def stream_key(seed: int, step: int, label: str) -> int:
    """`mix(mix(seed ^ GAMMA) ^ mix(step) ^ fnv1a(label))`."""
    return mix(mix(seed ^ GAMMA) ^ mix(step) ^ fnv1a(label))


def draw(key: int, i: int) -> int:
    """`mix(key + (i + 1) * GAMMA)`, wrapping."""
    return mix((key + (i + 1) * GAMMA) & MASK64)


def u24(z: int) -> float:
    """`(z >> 40) * 2^-24`, exact."""
    return float(z >> 40) * TWO_POW_M24


def u53(z: int) -> float:
    """`(z >> 11) * 2^-53`, exact (the spec's `U(i)` for `z = draw(key, i)`)."""
    return float(z >> 11) * TWO_POW_M53


def normal(key: int, i: int) -> float:
    """`N(i)`: Box–Muller, one value from draws `2i` and `2i + 1`."""
    v1 = float((draw(key, 2 * i) >> 11) + 1) * TWO_POW_M53
    v2 = u53(draw(key, 2 * i + 1))
    return math.sqrt(-2.0 * math.log(v1)) * math.cos(math.tau * v2)


def dropout_mask(seed: int, step: int, node_path: str, shape: Sequence[int], p: float) -> list[bool]:
    """The keep mask of `nn.dropout` in training mode (graph-ir.md §3.3), row-major: element `i` is
    kept if `u24(draw(key, i)) >= p`, `key = stream_key(seed, step, "dropout:" + node_path)`.
    `p = 0` keeps everything and draws nothing."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must be in [0, 1), got {p}")
    n = math.prod(shape)
    if p == 0.0:
        return [True] * n
    key = stream_key(seed, step, DROPOUT_PREFIX + node_path)
    return [u24(draw(key, i)) >= p for i in range(n)]


def to_f32(v: float) -> float:
    """`v` rounded once to `f32` (round to nearest, ties to even), as a Python float. `struct`'s
    `f` format rounds correctly and raises only if the rounded value overflows; then it is `±inf`
    (IEEE 754)."""
    try:
        return struct.unpack("<f", struct.pack("<f", v))[0]
    except OverflowError:
        return math.copysign(math.inf, v)


def fans_of(op: str, role: str, shape: Sequence[int]) -> tuple[int, int] | None:
    """`(fan_in, fan_out)` of graph-ir.md §2.4.1 ("Fans"), or None for a role without fans."""
    if role != "weight" or op not in ("nn.linear", "nn.embedding", "nn.conv1d", "nn.conv2d"):
        return None
    rf = math.prod(shape[2:])
    return shape[1] * rf, shape[0] * rf


FAN_KINDS = ("kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal")


def initial_values_f64(seed: int, name: str, shape: Sequence[int], init: dict, fans: tuple[int, int] | None) -> list[float]:
    """The values of graph-ir.md §2.4.1's table in `f64`, before rounding, in row-major order.
    `init` is the graph's `init` object (`{"kind": "uniform", "low": …, "high": …}`)."""
    kind = init["kind"]
    n = math.prod(shape)
    key = stream_key(seed, 0, INIT_PREFIX + name)
    if kind in FAN_KINDS:
        if fans is None:
            raise ValueError(f"{name}: init {kind} needs fans")
        fan_in, fan_out = fans
    if kind == "zeros":
        return [0.0] * n
    if kind == "ones":
        return [1.0] * n
    if kind == "constant":
        return [float(init["value"])] * n
    if kind == "uniform":
        low, high = float(init["low"]), float(init["high"])
    elif kind == "normal":
        mean, std = float(init["mean"]), float(init["std"])
    elif kind == "kaiming_uniform":
        a = float(init.get("a", 0.0))
        gain = math.sqrt(2.0 / (1.0 + a * a))
        std = gain / math.sqrt(float(fan_in))
        bound = math.sqrt(3.0) * std
        low, high = -bound, bound
        kind = "uniform"
    elif kind == "kaiming_normal":
        gain = math.sqrt(2.0 / (1.0 + 0.0))
        mean, std = 0.0, gain / math.sqrt(float(fan_in))
        kind = "normal"
    elif kind == "xavier_uniform":
        std = 1.0 * math.sqrt(2.0 / float(fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        low, high = -bound, bound
        kind = "uniform"
    elif kind == "xavier_normal":
        mean, std = 0.0, 1.0 * math.sqrt(2.0 / float(fan_in + fan_out))
        kind = "normal"
    else:
        raise ValueError(f"unknown init kind {kind!r}")
    if kind == "uniform":
        return [low + (high - low) * u53(draw(key, i)) for i in range(n)]
    return [mean + std * normal(key, i) for i in range(n)]


def initial_values(seed: int, name: str, shape: Sequence[int], init: dict, fans: tuple[int, int] | None) -> list[float]:
    """The `f32` initial values (graph-ir.md §2.4.1): `initial_values_f64` rounded once."""
    return [to_f32(v) for v in initial_values_f64(seed, name, shape, init, fans)]
