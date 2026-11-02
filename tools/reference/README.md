# Reference fixture generators

Python scripts that produce ground-truth fixtures from established implementations (PyTorch,
transformers, diffusers, gguf-py, librosa, llama.cpp). See [docs/testing-strategy.md](../../docs/testing-strategy.md).

Rules:
- Deterministic: fixed seeds, recorded library versions in the fixture's JSON sidecar.
- Output: `*.safetensors` + `*.json` into `fixtures/small/<area>/` (small) or a manifest entry (large).
- Python is a development dependency only. Nothing in Polypous runs Python.

Setup: [docs/dev-environment.md](../../docs/dev-environment.md#python-reference-environment).
The first generators and `requirements.txt` arrive in M1.
