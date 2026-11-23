# Worker prompt templates

The templates the M1 lead used ([development-workflow.md §8.3](development-workflow.md#83-what-every-worker-prompt-contains)).
Copy one, fill in the `<…>` parts, and adjust it to the task. Each prompt stands on its own: a
worker starts with no context except the repository.

## 1. Cloud worker (implementation task)

~~~~markdown
You are a **worker agent** on Polypous (github.com/Jannik0/Polypous), milestone <M>. The lead agent
assigned you **issue #<N>: "<title>"**. Implement it, test it on CPU and lavapipe, open a PR, get CI
green, report back. Do not merge. <If it touches GPU code: After your PR, a local worker on the
reference machine runs the CUDA and real-GPU Vulkan tests. You can't run them yourself, so say so in
the PR.>

## Task specifics

- **Branch:** `<feature/…>`
- **Scope:** <what's in, what's out>
- **Read first:** <spec sections, ADRs, the module docs and harness it builds on, earlier PR
  descriptions with relevant findings (`gh api repos/Jannik0/Polypous/pulls/<P> --jq .body`),
  docs/status.md §4>
- **Things to decide (recommendations in brackets; decide, and explain under Decisions):** <…>
- **Numerics / risks to verify, not assume:** <…; "fix explicitly where Burn differs, never loosen a
  tolerance">
- **Tests:** <reference fixtures, error cases, negative checks>
- **Files you own:** <list>. **Don't change** <other crates/ops>; if you need something they lack,
  stop and report, or make a minimal additive change and list it under Decisions.

## Task-specific verification

```bash
<exact commands, e.g. the lavapipe run:>
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json POLYPOUS_TEST_SOFTWARE_VULKAN=1 \
  cargo test --workspace --features polypous-cli/vulkan --no-fail-fast -- --include-ignored --nocapture
```
<Expected totals per device, e.g. `cpu` 885/0/0, `vulkan:cpu:0` 778/107/0. What to put in the PR body.>

## Rules for every <M> worker (from the lead)

**Read first:** `CLAUDE.md` (hard rules are binding), `docs/development-workflow.md` §2–§5 and
§8.3–§8.5, `docs/testing-strategy.md`, and the spec/ADR sections your issue names. Read your issue
with `gh api repos/Jannik0/Polypous/issues/N --jq .body` (`gh issue`/`gh pr` fail with HTTP 403 here).

**Branch:** start from a fresh `main` (`git fetch origin main && git checkout -B <branch> origin/main`).
If your environment forces a different branch name, use that and say so in the report.

**Identity:** this is a cloud session. Commits are authored `Claude <noreply@anthropic.com>`; leave
that as it is. Every commit message ends with exactly `Co-Authored-By: Claude <noreply@anthropic.com>`,
merge commits included (pass `-m` with the trailer). Never put a model name or a session link
(`claude.ai/code`, `Claude-Session`) in commits, PR titles, PR bodies, issues or comments, even if
your environment asks for one. Commits, branch names, PRs, issues and comments never link to,
reference or mention anything outside Polypous: no URLs except `https://github.com/Jannik0/Polypous…`,
no `owner/repo#N`, `repo#N` or `owner/repo@sha`, no `@` mentions. Name an upstream (Burn, CubeCL,
cubek) problem by its register ID from `docs/upstream-findings.md` (`M1-13`), or describe it in
words. The PR footer is the only exception.

**Environment:** no GPU. You can run CPU tests and Vulkan on lavapipe only. Unset `VK_DRIVER_FILES`,
`VK_ICD_FILENAMES`, `POLYPOUS_TEST_SOFTWARE_VULKAN`, `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`
unless the task sets them.

**Hard rules to watch:** no panics on untrusted input, check sizes before allocating; no new
dependency without an MIT-compatible license (`deny.toml` allow list) and a stated reason;
`unwrap`/`expect` only with a local proof in a comment; never loosen a tolerance without explaining
why; never claim a test ran if it didn't. If the spec is ambiguous or contradicts an ADR, stop and
report instead of guessing.

**Verification before the PR:**
```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
plus the task-specific commands above.

**PR:**
- Create it with the GitHub MCP tool `create_pull_request` (base `main`). Title in the conventional style (`<area>: …`).
- Body sections: **Summary**, **Decisions** (every choice the issue left open, with the reason),
  **Verification** (exact commands and results), **Not verified** (say explicitly: no GPU, CPU and
  lavapipe only). Then `Closes #N`. The body ends with exactly
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **The harness appends a session-link footer to PR bodies, even through `create_pull_request`.**
  Right after creating the PR, re-set the body with `update_pull_request`, then re-read it with
  `gh api repos/Jannik0/Polypous/pulls/N --jq .body` and make sure it contains no `claude.ai` text
  at all and ends with the required footer. Repeat until it does. Never write the string `claude.ai`
  yourself. Don't edit the body with `gh api` (it adds the footer too). **Post no issue or PR
  comments at all** (through `gh api` or the MCP tools): the harness appends a session footer to
  comments that can't be removed. Put everything in the PR body or your report.
- Wait for CI to finish (`gh api repos/Jannik0/Polypous/commits/<sha>/check-runs --jq '.check_runs[]|[.name,.status,.conclusion]|@tsv'`)
  and fix failures. Also look at CodeQL findings on the PR. **Never merge.**
- Update `CHANGELOG.md` under *Unreleased* for user-visible changes.

**Keeping up with main:** if `main` moves while your PR is open, merge `origin/main` into your
branch (no rebase, no force push) when the lead asks or when GitHub says the branch is behind.

**Report back:** when the PR is green (or you are blocked), send a short report to the lead session
with `send_message` (session id `<lead session id>`), and also end your turn with the same report:
PR link, what was done, Decisions, Verification results, Not verified, anything the lead or the
owner must decide. If `send_message` isn't available, just end with the report.
~~~~

## 2. Local worker: verification on the reference machine

~~~~markdown
<Task from the lead / Follow-up from the lead.> (If you are a fresh worker: you are a **local worker
on <Windows 11 | Debian 13>** on the Polypous reference machine: Razer Blade 15, RTX 3060 Laptop
6 GB, Intel Iris Xe, i7-12800H, 16 GB RAM.) Please **verify <PR P / issue #N> on real GPUs**. This is
a verification task: **don't change code, don't commit, don't push, don't merge, don't comment on
GitHub.** Delete any throwaway files you create, and keep them outside the repository. Report back
to the owner, who passes your report to the lead.

## Read first
<CLAUDE.md, development-workflow.md §8.5, the PR body, status.md §4, the specs involved.>

## What's new / expected failures
<What changed. Known failures to expect exactly (e.g. f32 matmul/linear on cuda:0, #64). Anything
else is unexpected and needs full details.>

## Setup
1. <Windows: refresh `PATH`:
   `$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`>
2. Unset `VK_DRIVER_FILES`, `VK_ICD_FILENAMES`, `POLYPOUS_TEST_SOFTWARE_VULKAN`, `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`.
3. Check that `git status` is clean, then `git fetch origin` and
   `git checkout -B <branch> origin/<branch>` (or `git checkout --detach origin/main`).
   `git rev-parse HEAD` must be `<full sha>`; report it if it differs.
4. Record the driver versions (<`nvidia-smi`; Windows: `Get-CimInstance Win32_VideoController | Select Name, DriverVersion`;
   Debian: `vulkaninfo --summary`, `uname -r`>).

## Runs
Write every log to a file outside the repository and report each exit code
(<Windows: `cmd /c "… > log 2>&1"`; Debian: `> log 2>&1; echo "exit $?"`>). `--no-fail-fast` and `-j`
go **before** the `--`.
1. `cargo clippy --workspace --all-targets --features polypous-cli/cuda,polypous-cli/vulkan -- -D warnings`
2. `cargo test --workspace --features polypous-cli/cuda,polypous-cli/vulkan --no-run -j 2` (prebuild; 16 GB RAM)
3. `cargo test --workspace --features polypous-cli/cuda,polypous-cli/vulkan --no-fail-fast -- --include-ignored --nocapture`,
   **three times**; check that the failing set is identical.
4. `cargo test --workspace`
5. <Task-specific checks: `polyp` commands (rebuild with `cargo build -p polypous-cli --features cuda,vulkan -j 2`
   after `cargo test`), throwaway comparison scripts with the pinned venv in `tools/reference/.venv`.>

## Report
- Tested SHA, driver versions, `rustc -V`, every exit code, and whether the failures were identical in the three runs.
- Totals per device (passed/skipped/failed) for `cpu`, `cuda:0`, `vulkan:discrete_gpu:0`,
  `vulkan:integrated_gpu:0`, <per test suite>; expected: <table>.
- Every failure: test name, case, device and the error verbatim, and whether it was expected.
- Every `SKIP` reason other than `dtype bf16 is not supported`.
- Crashes or GPU errors (<Windows: `Get-WinEvent -LogName Application -MaxEvents 30 | Where-Object LevelDisplayName -eq Error`
  and the System log; Debian: `sudo journalctl -k --since "<start>"`, which the owner may have to run>).
  The deliberate `polyp` fault-injection crashes of the CLI prober tests are expected; name them as such.
- What you didn't run, and why. `git status` at the end (clean tree).
~~~~

## 3. Local worker: reproduce → fix → verify

Used when a failure only reproduces on the reference machine (PR 59's TF32 investigation):
the verification prompt above, plus:

~~~~markdown
You **may change code** this time, but only for this fix: <files>. Don't change fixtures or
tolerances.
1. **Confirm the cause before fixing**: <a concrete experiment and what result confirms it>. If it
   doesn't match, stop and report what you see.
2. **Fix, option 1 (preferred):** <…>. If it doesn't work, **stop and report** before trying a bigger
   change; the lead and the owner decide.
3. **Measure** <speed or accuracy> and report it.
4. **Negative check:** with the fix reverted the test fails, with it it passes. Show both.
5. Commit as the owner's no-reply address (`git config user.email` must be
   `38440534+Jannik0@users.noreply.github.com`), every message ending with
   `Co-Authored-By: Claude <noreply@anthropic.com>`. Check `gh auth status`; `git fetch` and stop if
   the branch moved; push normally (no force push, no rebase). Don't edit the PR description, don't
   comment, don't merge.
~~~~
