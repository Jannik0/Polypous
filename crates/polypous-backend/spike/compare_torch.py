"""PyTorch side of the autodiff spike (#71). Uses the reference environment of
docs/dev-environment.md (Python 3.14, torch 2.15.0+cpu).

    python crates/polypous-backend/spike/compare_torch.py q3        # writes spike/out/q3_torch.json
    python crates/polypous-backend/spike/compare_torch.py q2 FILE…  # compares q2_*.json from the Rust spike

Not a fixture generator: the spike's numbers only. Gradient fixtures are #74.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "out"


def val(v):
    return float(v) if not isinstance(v, str) else float(v.replace("NaN", "nan"))


def t(spec, dtype):
    x = torch.tensor([val(v) for v in spec["values"]], dtype=torch.float32)
    return x.reshape(spec["shape"]).to(dtype)


I = lambda v, s=None: torch.tensor(v, dtype=torch.int64) if s is None else torch.tensor(v, dtype=torch.int64).reshape(s)

# name -> function of the input tensors, mirroring q2_cases() in autodiff_spike_tests.rs.
CASES = {
    "add_broadcast": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul_broadcast": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: torch.pow(a, b),
    "pow_special": lambda a, b: torch.pow(a, b),
    "maximum_ties": torch.maximum,
    "minimum_ties": torch.minimum,
    "maximum_nan": torch.maximum,
    "neg": torch.neg,
    "abs": torch.abs,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "sin": torch.sin,
    "cos": torch.cos,
    "tanh": torch.tanh,
    "clamp_bounds": lambda x: torch.clamp(x, -1.0, 1.0),
    "clamp_min_nan": lambda x: torch.clamp(x, min=0.0),
    "where": lambda a, b: torch.where(a > 0, a, b),
    "reshape": lambda x: x.reshape(3, 2),
    "flatten": lambda x: torch.flatten(x, 1, 2),
    "permute": lambda x: x.permute(2, 0, 1),
    "transpose": lambda x: x.transpose(0, 1),
    "squeeze": lambda x: x.squeeze(1),
    "unsqueeze": lambda x: x.unsqueeze(1),
    "expand": lambda x: x.expand(2, 3),
    "concat": lambda a, b: torch.cat([a, b], 1),
    "split_second": lambda x: torch.split(x, [2, 3], 1)[1],
    "slice_step": lambda x: x[:, 1:6:2],
    "pad": lambda x: F.pad(x, (1, 2, 0, 1), value=0.5),
    "gather": lambda x: torch.gather(x, 1, I([[2, 0, 2], [1, 1, 0]])),
    "index_select": lambda x: torch.index_select(x, 0, I([2, 0, 2])),
    "full_like_mul": lambda x: x * torch.full_like(x, 2.5),
    "cast_f16_roundtrip": lambda x: x.to(torch.float16).to(torch.float32),
    "sum_dim": lambda x: x.sum(1),
    "mean_dim": lambda x: x.mean(0, keepdim=True),
    "max_all_ties": lambda x: torch.amax(x),
    "max_dim_ties": lambda x: torch.amax(x, 1),
    "min_dim_ties": lambda x: torch.amin(x, 1),
    "max_dim_nan": lambda x: torch.amax(x, 1),
    "cumsum": lambda x: torch.cumsum(x, 1),
    "matmul_broadcast": torch.matmul,
    "softmax": lambda x: torch.softmax(x, 1),
    "log_softmax": lambda x: torch.log_softmax(x, 1),
    "top_k": lambda x: torch.topk(x, 2, 1, True, True).values,
    "linear": F.linear,
    "embedding": lambda w: F.embedding(I([1, 3, 1]), w),
    "conv1d_groups_stride": lambda x, w, b: F.conv1d(x, w, b, 2, 1, 1, 2),
    "conv2d_groups_dilation": lambda x, w, b: F.conv2d(x, w, b, (1, 2), (1, 0), (2, 1), 2),
    "max_pool2d_ties": lambda x: F.max_pool2d(x, 2, 2),
    "max_pool2d_overlap_pad": lambda x: F.max_pool2d(x, 3, 2, 1),
    "avg_pool2d_include_pad": lambda x: F.avg_pool2d(x, 3, 2, 1, count_include_pad=True),
    "avg_pool2d_exclude_pad": lambda x: F.avg_pool2d(x, 3, 2, 1, count_include_pad=False),
    "adaptive_avg_pool2d": lambda x: F.adaptive_avg_pool2d(x, (3, 3)),
    "dropout_inference": lambda x: F.dropout(x, 0.5, training=False),
    "batch_norm_inference": lambda x, w, b: F.batch_norm(
        x, torch.tensor([0.1, -0.2, 0.3]).to(x.dtype), torch.tensor([0.5, 1.5, 2.0]).to(x.dtype), w, b, training=False, eps=1e-5
    ),
    "layer_norm": lambda x, w, b: F.layer_norm(x, (4,), w, b, 1e-5),
    "group_norm": lambda x, w, b: F.group_norm(x, 2, w, b, 1e-5),
    "rms_norm": lambda x, w: F.rms_norm(x, (4,), w, 1e-6),
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, 0.1),
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "sigmoid": torch.sigmoid,
    "softplus": lambda x: F.softplus(x, 1.0, 20.0),
    "softplus_beta2": lambda x: F.softplus(x, 2.0, 6.0),
    "mish": F.mish,
}


def close(a, b, rtol, atol):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * abs(b)


def q2(files):
    for f in files:
        print(f"== {f}")
        results = json.loads(Path(f).read_text())
        for r in results:
            name = r["name"]
            if "error" in r or "panic" in r:
                print(f"{name:<26} RUST ERROR {r.get('error') or r.get('panic')}")
                continue
            dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[r["dtype"]]
            rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-3)
            xs = [t(i, dtype).requires_grad_() for i in r["inputs"]]
            out = CASES[name](*xs)
            up = torch.tensor([val(v) for v in r["upstream"]], dtype=torch.float32).reshape(out.shape).to(dtype)
            (out * up).sum().backward()
            fwd_bad = [
                (i, a, float(b))
                for i, (a, b) in enumerate(zip(map(val, r["out"]), out.detach().float().flatten().tolist()))
                if not close(a, b, rtol, atol)
            ]
            notes = []
            if fwd_bad:
                notes.append(f"FORWARD differs at {fwd_bad[:3]}")
            for k, (x, g) in enumerate(zip(xs, r["grads"])):
                want = x.grad
                if g is None:
                    if want is not None and bool((want != 0).any()):
                        notes.append(f"input {k}: CUT (torch grad {want.float().flatten().tolist()})")
                    continue
                if want is None:
                    want = torch.zeros_like(x)
                got = [val(v) for v in g]
                want = want.float().flatten().tolist()
                bad = [(i, a, b) for i, (a, b) in enumerate(zip(got, want)) if not close(a, b, rtol, atol)]
                errs = [abs(a - b) for a, b in zip(got, want) if math.isfinite(a) and math.isfinite(b)]
                if bad:
                    notes.append(f"input {k}: DIFFERS at {len(bad)}/{len(got)}: " + "; ".join(f"[{i}] got {a:.6g} torch {b:.6g}" for i, a, b in bad[:4]))
                else:
                    notes.append(f"input {k}: ok (max abs {max(errs, default=0):.2e})")
            print(f"{name:<26} " + " | ".join(notes))


def q3():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(71)
    w0 = (torch.rand(3, 4, generator=g) * 2 - 1).float()
    b0 = (torch.rand(3, generator=g) * 2 - 1).float()
    x = (torch.rand(2, 4, generator=g) * 2 - 1).float()
    y = (torch.rand(2, 3, generator=g) * 2 - 1).float()
    configs = {
        "sgd_nesterov": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, dampening=0.0, nesterov=True, weight_decay=0.01, foreach=False),
        "sgd_momentum_dampening": lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, dampening=0.1, foreach=False),
        "adam": lambda p: torch.optim.Adam(p, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False),
        "adamw": lambda p: torch.optim.AdamW(p, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, foreach=False),
    }
    steps = {}
    for name, make in configs.items():
        w = w0.clone().requires_grad_()
        b = b0.clone().requires_grad_()
        opt = make([w, b])
        rows = []
        for _ in range(3):
            opt.zero_grad()
            loss = ((F.linear(x, w, b) - y) ** 2).mean()
            loss.backward()
            gw, gb = w.grad.clone(), b.grad.clone()
            opt.step()
            rows.append({"w": w.detach().flatten().tolist(), "b": b.detach().tolist(), "grad_w": gw.flatten().tolist(), "grad_b": gb.tolist()})
        steps[name] = rows
    OUT.mkdir(exist_ok=True)
    data = {"torch": torch.__version__, "w0": w0.flatten().tolist(), "b0": b0.tolist(), "x": x.flatten().tolist(), "y": y.flatten().tolist(), "steps": steps}
    (OUT / "q3_torch.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT / 'q3_torch.json'} (torch {torch.__version__})")


if __name__ == "__main__":
    if sys.argv[1] == "q3":
        q3()
    else:
        q2(sys.argv[2:])
