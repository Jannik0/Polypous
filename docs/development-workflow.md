# Development Workflow

How work moves from idea to merged code in a project built mostly by AI agents.

## 1. Roles

- **Owner (human):** sets priorities, accepts ADRs, reviews and merges PRs, runs or approves
  tests on hardware agents cannot access.
- **Agents:** take tasks, write specs/ADRs (as *Proposed*), code and tests. They report results honestly, including what was not verified.

## 2. Units of work

Work is split into **tasks** small enough for one agent session and one PR: typically one op family,
one component, one spec section, one format reader. A task description contains:

```markdown
### Task: <imperative title>
- Milestone: M1
- Crates: polypous-ir
- Spec: docs/specs/graph-ir.md §2.6 (repeat blocks)
- Depends on: <tasks that must be merged first>
- Done when:
  - <observable criterion, ideally a test>
  - <...>
- Out of scope: <what not to do>
```

Tasks are tracked as **GitHub issues** at `Jannik0/Polypous`, grouped by milestone (`M0`, `M1`, …)
and labeled `task`. An agent starting a task reads its issue first and links the PR to it (`Closes #N`).
Tasks that touch different crates can run in parallel. Tasks that touch the same crate run one after another.

## 3. Lifecycle of a task

1. **Understand:** read the spec and ADRs, and look at existing code in the crate.
2. **Clarify the spec:** if the spec is ambiguous or missing what the task needs, update it first.
   If that needs a design decision, write a *Proposed* ADR and stop for the owner.
3. **Tests first, where practical:** add reference fixtures and failing tests that encode the spec.
4. **Implement** in the smallest reasonable steps.
5. **Verify:** fmt, clippy, CPU tests. GPU tests where available. State which ran.
6. **Document:** spec in sync, doc comments, `CHANGELOG.md`.
7. **PR** with the checklist from [CONTRIBUTING.md](../CONTRIBUTING.md).

## 4. Definition of done

- Every "done when" criterion is shown by a test or a documented manual check.
- CI is green.
- Spec, code and tests agree.
- No `TODO` without a linked task.

## 5. Branches and commits

- `main` is always green. Work happens on `feature/<short-name>`, `fix/<short-name>` or `docs/<short-name>` branches.
- Conventional-style commit subjects: `ir: add repeat block validation`, `quant: Q4_K dequant CPU kernel`.
  The prefix is the crate short name, or `docs`, `ci`, `tools`.
- Squash-merge PRs (enforced by the `main` ruleset). The squash commit message is the PR title and
  description. GitHub adds `Co-authored-by: Claude <noreply@anthropic.com>` for agent-authored
  commits; keep it.
- Attribution names Claude, never the model, and never links a session (see [CLAUDE.md](../CLAUDE.md#conventions)).
  Before merging, check the PR title and description for a model name or a `claude.ai/code` link and
  remove it, because the description becomes the squash commit message on `main`.
  The **PR title and description** CI job ([`pr-text.yml`](../.github/workflows/pr-text.yml),
  patterns in [`tools/ci/check_pr_text.py`](../tools/ci/check_pr_text.py)) fails a PR whose title or
  description contains `claude.ai/code`, `Claude-Session` or a model name with a version
  (`Opus`/`Sonnet`/`Haiku`/`Fable` + number). If the description mentions "Claude Code", it must also
  end with exactly `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. It re-runs when
  the description is edited, so fix the text instead of re-running the job.

## 6. Parallel agents

- Several agents may work at once, each in its own git worktree and branch.
- Coordination happens through task dependencies, not shared files. If two tasks need the same file,
  run them one after the other.
- Cross-cutting refactors (renaming a core type) are announced as their own task, and other tasks wait.

## 7. When an agent is stuck

Stop and report instead of guessing when:
- the spec contradicts an ADR or itself
- a reference implementation disagrees with the spec
- a test needs hardware that is not available (report it, don't skip it silently)
- a tolerance would need loosening

## 8. Lead agent and worker agents

A milestone is run by one **lead agent** that plans, delegates, reviews and verifies, and by
**worker agents** that implement one task each. This is how M0 ran; the notes below are what worked.

### 8.1 Roles

- **Owner:** accepts ADRs, approves every merge, and passes prompts and reports between the lead and
  local workers. A cloud lead can't start or message sessions on the owner's machines. The owner
  also does whatever needs a person: reboots, driver updates, admin rights.
- **Lead:**
  - Turns the roadmap into issues on a milestone (task template in §2).
  - Decides where each task runs, writes the worker prompts, and reviews PRs.
  - Gets reference-machine results before merging backend, kernel or probing changes.
  - Merges after the owner says so, and keeps [status.md](status.md) current.
  - Implements only small fixes itself.
- **Workers:** one task, one branch, one PR, as in §3–§5. They report what they verified and what
  they couldn't, and never merge.

### 8.2 Where a task runs

| Worker | Use it for | Can't do |
|---|---|---|
| **Cloud session** (Ubuntu, 4 vCPU, no GPU; [dev-environment.md](dev-environment.md#cloud-environments-for-agents)) | Pure-Rust crates (IR, formats, parsers, planner), CPU kernels, CPU reference tests, docs, CI | CUDA, real-GPU Vulkan, driver behavior. Lavapipe only |
| **Local Windows session** on the reference machine | CUDA and Vulkan on Windows, Windows-only code (registry, DLLs), checks that need a reboot or a driver update (the owner performs those) | Linux driver behavior |
| **Local Debian session** on the reference machine | Linux driver behavior, the real lavapipe crash (opt-in), Vulkan loader issues, hiding libraries in a private namespace (`unshare --user --map-root-user --mount` plus overlay whiteouts, no root) | Windows specifics |

When a bug only reproduces on the reference machine, give the whole **reproduce → fix → verify**
loop to that machine's local worker instead of a cloud worker that would have to fix blind. #26 (the
Vulkan enumeration race), #29 (the prober) and #32 (the Debian findings) went this way.

### 8.3 What every worker prompt contains

- **Task and sources:**
  - The issue number. Read it with `gh api repos/Jannik0/Polypous/issues/N --jq .body`, because in cloud sessions `gh issue`/`gh pr` fail with HTTP 403.
  - What to read first: CLAUDE.md, the spec sections, the ADRs.
  - The branch name, starting from a fresh `main`.
- **Identity:**
  - In local sessions, `git config user.email` must be the owner's no-reply address (CLAUDE.md).
    Cloud sessions commit as `Claude <noreply@anthropic.com>` (signed, set by the cloud
    environment); leave that as it is.
  - Check `gh auth status`. If gh isn't logged in, commit but don't push, and report the branch state.
- **Environment:**
  - Unset `VK_DRIVER_FILES`, `VK_ICD_FILENAMES`, `POLYPOUS_TEST_SOFTWARE_VULKAN`, `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` unless the task sets them.
  - Rebuild `polyp` with `cargo build -p polypous-cli --features cuda,vulkan` after any `cargo test`: the test build at `target/debug/polyp` contains prober fault injection.
  - Use a fresh `--probe-cache-dir` for cold runs.
- **Done when:**
  - Observable criteria with the exact expected output.
  - For every bug fix, a **negative check**: revert only the fix, show that the new test (and the real-hardware symptom) fails, then restore.
- **Scope:** what's out of scope, and which files to stay out of when other workers run in parallel.
- **Verification:** the exact commands. Cloud workers state that they ran CPU and lavapipe only.
- **PR rules:**
  - "Decisions", "Verification" and "Not verified" sections.
  - `Closes #N`, or `Part of #N` when the issue stays open.
  - Wait for green CI. Don't merge.
  - The attribution rules of CLAUDE.md. After opening the PR, re-read its body through the API.
- **The report back:** the PR link, results, decisions, and what wasn't verified.

### 8.4 Reviewing and merging

1. **Checks:**
   - All required checks are green (six; see [status.md §2](status.md#2-what-exists-after-m0)), and Build (macOS, Metal) too when it ran on the PR.
   - Every commit ends with `Co-Authored-By: Claude <noreply@anthropic.com>` and has no session link or model name.
   - Commit authors:
     - Commits from local sessions use the owner's no-reply address.
     - Commits from cloud sessions are authored `Claude <noreply@anthropic.com>`. That's expected: it isn't a personal address, and the squash merge replaces these commits on `main`.
     - Any other address is a finding.
   - The PR title and description have no session link and no model name (the PR-text job enforces this).
2. **Review focus:**
   - Untrusted input is bounded: files, prober output, registry, sysfs.
   - No panics across the API.
   - No state that differs between `Engine`s (public-api.md §2 rule 9). A process-wide hardware cache is acceptable when documented.
   - No Burn types outside `polypous-backend`/`polypous-quant`.
   - Spec, code, tests and `CHANGELOG.md` agree.
   - Tests can fail: negative checks were shown.
3. **Reference machine:** for backend, kernel and probing changes, run the GPU tests and the relevant real-hardware checks on Windows, and on Debian when Linux behavior is involved. Record the results in the PR description.
4. **Merge** after the owner's go:
   - The ruleset allows squash merges only.
   - Merge through the REST API: `PUT repos/Jannik0/Polypous/pulls/N/merge` with:
     - `merge_method: squash`
     - `commit_title: "<title> (#N)"`
     - a `commit_message` that summarizes the change and its verification, ending with the co-author line
     - `sha`: the head commit you verified
   - The squash commit on `main` is authored by the account that merges, which is the owner's,
     also from cloud sessions (they act as the owner on GitHub). With the owner's GitHub setting
     "Keep my email addresses private" on, it carries the no-reply address.
   - **Check the squash commit right after merging:** `git log -1 --format='%an <%ae>%n%B' origin/main`
     after a fetch. It must show the owner's no-reply address, the title and message you sent, the
     co-author line, and no session link or model name. A commit on `main` can't be fixed afterwards
     (no force pushes), so stop and tell the owner if anything is off.
   - **First merge by a cloud lead:** it has not been verified that the cloud harness leaves a
     merge call's `commit_message` alone (it does rewrite PR descriptions written through `gh api`,
     §8.5). A cloud lead's first merge is therefore a deliberate test on a documentation-only PR,
     checked as above before any other merge. If the harness changed the message, the owner merges
     from then on with GitHub's "Squash and merge" button, which uses the PR title and description
     that CI has already checked.
5. **A branch behind `main`:** merge `main` into the branch, resolve conflicts (usually `CHANGELOG.md`), re-verify and push. The ruleset forbids force pushes.
6. **After the merge:**
   - Make sure the issue closed.
   - Check that the branch was deleted. The repository deletes merged branches automatically, but not always.
   - Update [status.md](status.md) when a milestone item or known limitation changed.

### 8.5 Tool quirks seen in M0

- **Cloud sessions:**
  - The harness appends a session-link footer to PR descriptions written through `gh api`. The PR-text job fails on it. Edit the description (or write it with the GitHub tool) until only the required footer remains.
  - It appends a `claude.ai/code` footer to issue bodies written through `gh api` too (seen in M1), and re-appends it on every `gh api` edit. Write and edit issues with the GitHub MCP tool (`issue_write`), which leaves the body alone, and re-read the body afterwards. A PR created with the MCP tool (`create_pull_request`) still got the session-link footer; an edit with `update_pull_request` removed it.
  - The app's **Auto-fix** CI monitor can only be turned on for a PR the app knows the session opened. Otherwise the lead checks CI itself.
  - On GitHub, cloud sessions act as the owner's account: their PRs show as opened by the owner,
    and their commits as authored by `Claude <noreply@anthropic.com>`. Commit messages from cloud
    sessions in M0 had no session-link trailer.
- **Safety checks:** auto-mode safety checks can block destructive git operations (deleting a remote branch) and file deletions inside scripts. Don't work around them; ask the owner.
- **Windows PowerShell:**
  - Refresh `PATH` before `cargo`: `$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`.
  - Piping `cargo` into `Select-Object -First N` cuts the pipeline short and distorts the exit code. Write logs to a file instead.
- **`target/`:** debug test binaries with `cuda` or `vulkan` are about 1.2 GB each.
