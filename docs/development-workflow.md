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
- Squash-merge PRs.

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
