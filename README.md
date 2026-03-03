# agentic-python-dependency

PLLM-inspired agentic dependency resolver for Python snippets and projects.

The system follows the white paper flow:
- extract likely third-party packages from Python source
- retrieve version candidates from PyPI
- infer package pins with Gemma3 via Ollama
- validate the result in Docker
- retry with error-aware repair prompts when the failure is dependency-related

## Stack

- Python 3.12
- LangChain plus LangGraph
- Ollama with `gemma3:4b` for extraction
- Ollama with `gemma3:12b` for versioning, repair, and adjudication
- Host Ollama plus host Docker
- Gistable benchmark support pinned to `665d39a2bd82543d5196555f0801ef8fd4a3ee48`

## Local install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg
```

## Local runtime

Run Ollama directly on the host machine:

```bash
ollama pull gemma3:4b
ollama pull gemma3:12b
```

Make sure Docker is available on the host:

```bash
docker version
ollama list
```

If you want to use a specific execution tradeoff, pick a preset:

```bash
apd --preset performance smoke --jobs 1
apd --preset optimized smoke --jobs 1
apd --preset balanced smoke --jobs 1
apd --preset accuracy smoke --jobs 1
```

## Quick start

If you just cloned the repo and want the easiest commands, use these:

```bash
apd doctor
apd smoke --jobs 1
apd full --jobs 1
apd solve --path /path/to/python/repo
```

- `apd doctor` checks Docker, Ollama, required models, and dataset readiness.
- `apd smoke` runs the beginner-friendly smoke benchmark flow.
- `apd full` runs the full benchmark.
- `apd solve` runs dependency resolution for a local repo.

## CLI

```bash
apd benchmark fetch-gistable --ref 665d39a2bd82543d5196555f0801ef8fd4a3ee48
apd benchmark make-subsets
apd benchmark run --subset smoke30
apd --preset optimized benchmark run --subset smoke30
apd case run --case-id 000769db6848429c9b3eac30361d9140
apd project solve --path /path/to/python/repo
apd report summarize --run-id <run-id>
apd report modules --run-id <run-id> --grouping canonical --top 50
```

## One-step benchmark commands

Run the prepared benchmark segment in one command:

```bash
apd benchmark segment --jobs 2
apd smoke --jobs 2
```

Run the full benchmark in one command:

```bash
apd benchmark full --jobs 2
apd full --jobs 2
```

`benchmark segment` automatically fetches Gistable, regenerates `smoke30`, and runs it.
`benchmark full` automatically fetches Gistable and runs all valid cases.

## Presets

Four presets are available:

- `performance`
- `optimized`
- `balanced`
- `accuracy`

Examples:

```bash
apd --preset performance benchmark full --jobs 1
apd --preset optimized benchmark segment --jobs 2
apd --preset balanced project solve --path /path/to/python/repo
apd --preset accuracy benchmark run --subset smoke30
```

You can also override only the prompt profile:

```bash
apd --preset accuracy --prompt-profile paper benchmark run --subset smoke30
```

## Module reporting

Canonical module-family reporting is now the default:

```bash
apd report modules --run-id <run-id>
```

To inspect the ungrouped raw module buckets:

```bash
apd report modules --run-id <run-id> --grouping raw --top 50
```

## Tests

```bash
ruff check .
pytest
```

## Benchmark artifacts

Benchmark runs are written to `artifacts/runs/<run-id>/`.

Each case stores:
- prompts
- model outputs
- generated requirements
- generated Dockerfile
- build and run logs
- final result metadata
