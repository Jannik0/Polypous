# CI helper scripts

Used by the workflows in [`.github/workflows/`](../../.github/workflows/). See
[docs/testing-strategy.md §5](../../docs/testing-strategy.md#5-ci). Python 3, standard library only,
so they run on every runner without setup. The first two read `cargo metadata --format-version 1`.

| Script | Purpose |
|---|---|
| `check_dependency_boundary.py` | Fails if a crate other than `polypous-backend`/`polypous-quant` depends on `burn*`/`cubecl*` ([ADR-0007](../../docs/adr/0007-burn-version-pinning.md)), or if `polypous-ir`/`-formats`/`-polyp` have a GPU dependency (`burn*`, `cubecl*`, `wgpu*`, `cudarc*`, `ash`, `vulkan*`, `metal*`), directly or transitively. |
| `backend_feature.py FEATURE` | Prints `available=…` and `args=--features polypous-backend/FEATURE` for `$GITHUB_OUTPUT`, so jobs can use a backend feature once it exists. |
| `check_pr_text.py` | Reads `PR_TITLE`/`PR_BODY` and fails on a Claude Code session link, a `Claude-Session` trailer or a model name, or on an agent-authored description without the fixed footer ([development-workflow §5](../../docs/development-workflow.md#5-branches-and-commits)). Used by `pr-text.yml`. |

```bash
python3 tools/ci/check_dependency_boundary.py                       # check the workspace
python3 -m unittest discover -s tools/ci -p 'test_*.py' -v          # test the check
```

The tests run the check against `fixtures/workspace-clean.json` (trimmed `cargo metadata` of the
skeleton workspace) with deliberate violations injected.
