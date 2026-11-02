# Polypous

**Polypous** is a Rust layer on top of [Burn](https://github.com/tracel-ai/burn) that makes it easy to
**build, train and run neural networks**, and to **run large pretrained models on ordinary consumer hardware**.

> Status: **pre-development.** This repository currently contains the project brief, architecture,
> specifications and decision records. No code has been written yet. See the [roadmap](docs/roadmap.md).

## What it is for

- **App developers** building ML tools for non-technical users: visual network builders, local chat,
  transcription and image generation apps.
- **"It just runs."** Polypous picks the hardware automatically: as many layers as fit on the GPU
  (CUDA, Metal or Vulkan), the next ones in CPU RAM, and the rest streamed from disk. It puts
  reliability first and speed second.
- **Models are data.** Networks are described as a serializable graph. Pretrained architecture
  families (Llama, Whisper, Stable Diffusion, …) come as data-only `.polyp` definition files, so new
  architectures can be added without changing Polypous itself.
- **Pretrained weights** load from `safetensors` and `GGUF`, with native kernels for common GGUF
  quantization formats.
- **Easy to wrap.** The public API follows rules that make bindings for other languages (for example
  a Node.js addon for Electron apps) straightforward.

## Documentation

Start with the [documentation index](docs/README.md). Most important:

| Document | Purpose |
|---|---|
| [Project brief](docs/project-brief.md) | Vision, users, goals, non-goals, v1 target |
| [Architecture](docs/architecture.md) | Crates, layers, data flow |
| [Roadmap](docs/roadmap.md) | Milestones and exit criteria |
| [Specifications](docs/specs/) | Graph IR, `.polyp` format, placement, quantization, pipelines, public API, training |
| [Decision records](docs/adr/) | Why things are the way they are |
| [CONTRIBUTING](CONTRIBUTING.md) / [CLAUDE.md](CLAUDE.md) | How humans and AI agents work on this repo |

## License

[MIT](LICENSE)
