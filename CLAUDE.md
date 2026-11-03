# CLAUDE.md — Guide for AI agents working on Polypous

Polypous is a Rust layer on top of Burn for building, training and running neural networks
reliably on consumer hardware. **Most of the code is written by AI agents**, and this file is your
entry point. Human contributors: see [CONTRIBUTING.md](CONTRIBUTING.md). The rules are the same.

## Read before you work

1. [docs/project-brief.md](docs/project-brief.md): priorities (correctness > stability > ease of use > portability > performance), goals, non-goals.
2. [docs/architecture.md](docs/architecture.md): crates and their boundaries.
3. The **spec** for the area you are touching, in [docs/specs/](docs/specs/).
4. Relevant **ADRs** in [docs/adr/](docs/adr/). Accepted ADRs are binding.
5. [docs/development-workflow.md](docs/development-workflow.md): how a task goes from start to done.

## Hard rules

- **Specs are the source of truth.** If you implement behavior that is not specified, specify it first
  (or in the same change). If code and spec disagree, fix one of them in the same change. Never leave them out of sync.
- **Don't reopen accepted ADRs** in code. To change a decision, write a `Proposed` ADR and stop for the owner's review.
- **Burn and CubeCL are used only in `polypous-backend` and `polypous-quant`** ([ADR-0007](docs/adr/0007-burn-version-pinning.md)).
  No Burn types in any other crate's API.
- **`polypous-ir`, `polypous-formats`, `polypous-polyp` stay pure Rust** with no GPU or Burn dependency.
- **`.polyp` files are data only** ([ADR-0006](docs/adr/0006-polyp-files-are-data-only.md)). Never add a way to run code from them.
- **Public API rules** in [docs/specs/public-api.md §2](docs/specs/public-api.md#2-wrapper-friendliness-rules-binding-for-all-public-items) apply to every public item of the `polypous` crate.
- **No panics on untrusted input.** Model files, `.polyp` files and graphs are untrusted. Check sizes
  before allocating. `unwrap()`/`expect()` only where an invariant is proven locally (comment why).
- **Numerical code needs reference tests** ([docs/testing-strategy.md](docs/testing-strategy.md)).
  "Looks right" is not done. Never loosen a tolerance to make a test pass without explaining why in the change description.
- **Never fake results.** If a GPU test could not run in your environment, say so explicitly. Do not
  claim it passed.
- **No new dependencies** without checking license compatibility (MIT-compatible only, [ADR-0010](docs/adr/0010-mit-license.md))
  and stating why it is needed.
- Commits and pushes only when the task or the owner asks for them. Work on a branch, not on `main`.

## Environment

- Reference machine: Windows 11 / Debian dual boot, RTX 3060 Laptop (6 GB, CUDA + Vulkan), Intel iGPU,
  16 GB RAM. **No Metal hardware**, so Metal code is compile-checked only. See [docs/hardware-targets.md](docs/hardware-targets.md).
- **Cloud sessions have no GPU** (4 vCPU, 16 GB RAM, Ubuntu). There you can run CPU tests and
  Vulkan on lavapipe only, and you must say so in the PR. See [docs/dev-environment.md](docs/dev-environment.md#cloud-environments-for-agents).
- Tasks are GitHub issues on `Jannik0/Polypous`. Read the issue for your task and reference it in the PR.
- **In cloud sessions, `gh` commands that use GraphQL fail** (`gh issue …`, `gh pr …`) with an HTTP 403.
  Use the REST API instead, for example `gh api repos/Jannik0/Polypous/issues/1 --jq .body`.
- Setup instructions: [docs/dev-environment.md](docs/dev-environment.md).

## Commands

> The workspace crates are skeletons for now. The GPU features, the CLI subcommands and the fixture
> script arrive with later milestones; until then those commands are the planned interface.

```bash
cargo fmt --all                                   # format
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                            # CPU tests, must pass everywhere
cargo test --workspace --features cuda,vulkan -- --include-ignored gpu   # GPU tests on the reference machine
cargo run -p polypous-cli -- devices              # list detected devices
cargo run -p polypous-cli -- validate polyps/llama.polyp
python tools/reference/fetch_fixtures.py          # download large fixtures (hash-verified)
```

## Conventions

- Rust edition and toolchain pinned in `rust-toolchain.toml`. Run `cargo fmt` and keep clippy clean.
- Errors: `thiserror` per crate, mapped to `polypous::Error` at the facade. Logging: `tracing`, never `println!` in library code.
- Naming: crates `polypous-*`. Op names in the IR are `opset.op_name` (`nn.linear`). JSON fields are `snake_case`.
- Public items get doc comments. Modules with non-obvious algorithms link to the spec section they implement.
- Tests live next to the code (`#[cfg(test)]`) for unit tests and in `crates/*/tests/` for reference/integration tests.
- GPU-only tests are `#[ignore = "gpu"]` and gated on features.
- Keep changes focused: one task, one branch, one PR. Update `CHANGELOG.md` under *Unreleased* for user-visible changes.
- **Attribution:** commits written by an agent end with exactly `Co-Authored-By: Claude <noreply@anthropic.com>`.
  Never mention the **model name** (no "Claude Opus …") or a **session link** (`Claude-Session`
  trailers, claude.ai/code URLs) in commits, PRs, issues or comments. `.claude/settings.json` sets
  Claude Code's attribution accordingly.
- **Email addresses:** the only personal address in the repository is the owner's GitHub no-reply
  address, `38440534+Jannik0@users.noreply.github.com`. Never add any other personal email address.

## Where things go

| You are adding… | Put it in… | Also update… |
|---|---|---|
| A new IR op | `polypous-ir` catalog + `polypous-backend` impl | `docs/specs/graph-ir.md` opset list, reference test |
| A new quant format | `polypous-quant` | `docs/specs/quantization.md`, bit-exact fixture |
| A new architecture family | `polyps/<name>.polyp` | Model test against reference; ops/components it needs first |
| A pipeline component | `polypous-pipelines` | `docs/specs/pipelines.md`, reference test |
| A design decision | `docs/adr/NNNN-*.md` (Proposed) | ADR index |
