# Contributing to Polypous

Thanks for your interest! Polypous is developed mainly by AI agents under the direction of a human
project owner. Human contributions follow the same process and rules, described in
**[CLAUDE.md](CLAUDE.md)** (hard rules, conventions, where things go).

## Quick start

1. Read the [project brief](docs/project-brief.md) and [architecture](docs/architecture.md).
2. Set up your machine: [docs/dev-environment.md](docs/dev-environment.md).
3. Pick a task (see [development workflow](docs/development-workflow.md)) and read its spec.
4. Work on a branch, add or update reference tests, keep the spec in sync.
5. Open a pull request using the checklist below.

## Pull request checklist

- [ ] The change does one thing. The PR description says what and why.
- [ ] Spec(s) updated, or "no spec change needed" stated.
- [ ] Tests added or updated. Numerical code is tested against reference fixtures.
- [ ] `cargo fmt`, `cargo clippy -D warnings`, `cargo test --workspace` pass.
- [ ] GPU tests: state which backends were run (CUDA / Vulkan / Metal) and on which machine, or that they were not run.
- [ ] No new dependency, or its license and need are stated.
- [ ] `CHANGELOG.md` updated for user-visible changes.

## Design changes

Significant design changes need an ADR (`docs/adr/`, status *Proposed*). Open it as its own PR
or as the first commit of a PR, and wait for the owner to accept it before building on it.

## Reporting problems

Bug reports are most useful with: Polypous version, OS, output of `polyp devices`, the model file
(name/source + quantization), the placement plan summary, and logs with `RUST_LOG=polypous=debug`.

## License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).
