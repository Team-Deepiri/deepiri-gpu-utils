# CodeQL Setup for deepiri-gpu-utils

This folder contains the CodeQL configuration for security scanning in this repository.

## What each file does

- `.github/workflows/codeql.yml`
  - Defines when scans run and how GitHub Actions executes CodeQL.
- `.github/codeql/codeql-config.yml`
  - Defines analysis scoping rules. In this repository, it currently focuses on `paths-ignore` to skip generated/runtime artifacts.

## CodeQL workflow breakdown (`.github/workflows/codeql.yml`)

### `name: CodeQL`

The display name in the Actions tab.

### `on.pull_request.branches` and `on.push.branches`

```yaml
on:
  pull_request:
    branches: [main, dev]
  push:
    branches: [main, dev]
```

Runs scans when PRs target `main` or `dev`, and when commits are pushed to `main` or `dev`.

### `permissions`

```yaml
permissions:
  actions: read
  contents: read
  security-events: write
```

Uses least-privilege permissions. `security-events: write` is required so CodeQL can upload findings.

### Language setup (current)

```yaml
with:
  languages: python
```

This workflow currently runs analysis for Python.

### Checkout step

```yaml
with:
  fetch-depth: 0
```

- `fetch-depth: 0` keeps full git history (safe default for analysis and troubleshooting).

### Initialize CodeQL

```yaml
uses: github/codeql-action/init@v3
with:
  config-file: ./.github/codeql/codeql-config.yml
```

Starts the CodeQL engine and loads `.github/codeql/codeql-config.yml`.

### Analyze

```yaml
uses: github/codeql-action/analyze@v3
```

Executes queries and uploads results to GitHub Security.

## Config breakdown (`.github/codeql/codeql-config.yml`)

### Current structure

The current config intentionally defines `paths-ignore` only.

```yaml
paths-ignore:
  - '**/__pycache__/**'
  - '**/.pytest_cache/**'
  - '**/.mypy_cache/**'
  - '**/.ruff_cache/**'
  - '**/.venv/**'
  - '**/venv/**'
  - '**/dist/**'
  - '**/build/**'
  - '**/coverage/**'
  - '**/htmlcov/**'
  - '**/*.pyc'
```

These exclusions reduce noise and runtime by skipping common Python caches, virtual environments, build outputs, and coverage artifacts.

## Best practices

1. Keep trigger scope intentional.
   Use branch filters (`main`, `dev`) to control cost and noise.
2. Keep language list explicit.
   Only include languages with meaningful source code.
3. Add `paths` only when you need stricter include scoping.
   This repository currently relies on defaults plus `paths-ignore`.
4. Exclude generated/vendor artifacts.
   Keep caches, dependencies, build outputs, logs, and minified files in `paths-ignore` as needed.
5. Pin to stable major action versions.
   `@v3` is the current stable major for CodeQL actions used here.
6. Review alerts regularly.
   Triage high/critical findings first and suppress only with documented reasoning.

## Maintenance examples

Keeping this updated as code and language coverage evolve is important. Here are common maintenance changes.

### Keep language scope aligned with this repository

This workflow currently analyzes Python only:

```yaml
with:
  languages: python
```

Only change this value when this repository adds production code in another supported language.

### Add explicit include paths (optional)

If you want stricter scan scope later, add existing directories such as `src` and `tests`:

```yaml
paths:
  - src
  - tests
```

### Exclude another generated folder

Add another glob to `paths-ignore`, for example:

```yaml
- '**/generated/**'
```