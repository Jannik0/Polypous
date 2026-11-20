"""PyTorch side of the autodiff spike (#71), for the optimizer comparison that #79 keeps. Uses
the reference environment of docs/dev-environment.md (Python 3.14, torch 2.15.0+cpu).

    python crates/polypous-backend/spike/compare_torch.py q3        # writes spike/out/q3_torch.json

Not a fixture generator: the spike's numbers only. The spike's op-gradient comparison (q2) was
replaced by the gradient fixtures (#74) and their harness (#75).
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "out"


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
    if sys.argv[1:] != ["q3"]:
        sys.exit("usage: compare_torch.py q3")
    q3()
