# Test fixtures

- `small/`: committed fixtures (< 1 MB each, < 50 MB total), organized by area (`ops/`, `quant/`, `tokenizers/`, …).
- `manifest.json`: large fixtures and test models (URL, SHA-256, size, license), downloaded into
  `cache/` (git-ignored) by `tools/reference/fetch_fixtures.py`.

See [docs/testing-strategy.md](../docs/testing-strategy.md#3-reference-fixtures).
