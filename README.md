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

## Official PyEGo resolver setup

If you run with `--resolver pyego`, use a dedicated Python `3.11` environment for PyEGo:

```bash
python3.11 -m venv .venv-pyego
source .venv-pyego/bin/activate
python -m pip install --upgrade pip
python -m pip install -r external/PyEGo/requirements.txt
export APDR_PYEGO_PYTHON="$(pwd)/.venv-pyego/bin/python"
```

PyEGo depends on `typed_ast.ast27`, which is not supported on Python `3.12+`.
Official PyEGo also requires a running Neo4j instance with PyKG loaded and
`external/PyEGo/config.py` configured (for example `NEO4J_URI = "bolt://localhost:7687"`).
For Aura, set `NEO4J_URI` with `bolt+s://...` (not `neo4j+s://`), and set
`NEO4J_USERNAME` / `NEO4J_DATABASE` in the same config file when needed.

## Local runtime

Run Ollama directly on the host machine:

```bash
ollama pull gemma3:4b
ollama pull gemma3:12b
ollama pull qwen3.5:9b
ollama pull gpt-oss:20b
```

Make sure Docker is available on the host:

```bash
docker version
ollama list
```

If you want to use a specific execution tradeoff, pick a preset:

```bash
apdr --preset performance smoke --jobs 1
apdr --preset efficient smoke --jobs 1
apdr --preset optimized smoke --jobs 1
apdr --preset balanced smoke --jobs 1
apdr --preset thorough smoke --jobs 1
apdr --preset accuracy smoke --jobs 1
apdr --preset experimental smoke --jobs 1
apdr --preset research smoke --jobs 1
```

Model bundles are also selectable:

```bash
apdr --model-profile gemma-moe smoke --jobs 1
apdr --model-profile gemma-moe-lite smoke --jobs 1
apdr --model-profile qwen35-9b smoke --jobs 1
apdr --model-profile qwen35-moe-lite smoke --jobs 1
apdr --model-profile gpt-oss-20b smoke --jobs 1
```

Runtime behavior is selectable too:

```bash
apdr --no-moe smoke --jobs 1
apdr --no-rag smoke --jobs 1
apdr --no-langchain smoke --jobs 1
apdr --model-profile custom --extraction-model gemma3:1b --runner-model qwen3.5:4b smoke --jobs 1
apdr --extraction-model gemma3:1b --runner-model gemma3:4b --version-model qwen3.5:9b --repair-model qwen3.5:9b --adjudication-model gpt-oss:20b smoke --jobs 1
```

## Quick start

If you just cloned the repo and want the easiest commands, use these:

```bash
apdr ui
apdr doctor
apdr smoke --jobs 1
apdr full --jobs 1
apdr solve --path /path/to/python/repo
```

- `apdr ui` launches a prompt-toolkit powered terminal control center for common benchmark, report, and project commands, including a live benchmark dashboard with preset, progress, active cases, elapsed time, and result counts.
- `apdr doctor` checks Docker, Ollama, required models, and dataset readiness.
- `apdr smoke` runs the beginner-friendly smoke benchmark flow.
- `apdr full` runs the full benchmark.
- `apdr solve` runs dependency resolution for a local repo.
- For official PyEGo local setup, open `apdr ui` and use:
  `Configure -> Official setup -> Setup local PyEGo Neo4j (recommended)`.

## CLI

```bash
apdr benchmark fetch-gistable --ref 665d39a2bd82543d5196555f0801ef8fd4a3ee48
apdr benchmark make-subsets
apdr benchmark save-competition-filter
apdr benchmark run --subset smoke30
apdr --benchmark-source competition-run benchmark run --subset smoke30
apdr --benchmark-source competition-run --competition-csv /abs/path/pyego_results.csv --competition-csv /abs/path/summary-all-runs.csv benchmark run --subset smoke30
apdr --preset optimized benchmark run --subset smoke30
apdr --fresh-run --model-profile gpt-oss-20b benchmark run --subset smoke30
apdr case run --case-id 000769db6848429c9b3eac30361d9140
apdr project solve --path /path/to/python/repo
apdr report summarize --run-id <run-id>
apdr report modules --run-id <run-id> --grouping canonical --top 50
apdr report trace --run-id <run-id> --case-id <case-id>
```

## One-step benchmark commands

Run the prepared benchmark segment in one command:

```bash
apdr benchmark segment --jobs 2
apdr smoke --jobs 2
```

Run the full benchmark in one command:

```bash
apdr benchmark full --jobs 2
apdr full --jobs 2
```

`benchmark segment` automatically fetches Gistable, regenerates `smoke30`, and runs it.
`benchmark full` automatically fetches Gistable and runs all valid cases.

## Presets

Eight presets are available:

- `performance`
- `efficient`
- `optimized`
- `balanced`
- `thorough`
- `accuracy`
- `experimental`
- `research`

Examples:

```bash
apdr --preset performance benchmark full --jobs 1
apdr --preset efficient benchmark segment --jobs 2
apdr --preset optimized benchmark segment --jobs 2
apdr --preset balanced project solve --path /path/to/python/repo
apdr --preset thorough benchmark run --subset smoke30
apdr --preset accuracy benchmark run --subset smoke30
apdr --preset experimental benchmark run --subset smoke30
apdr --preset research benchmark run --subset smoke30
```

`research` keeps the advanced multi-plan workflow that was previously in `experimental`.
The new `experimental` preset is a fixed accuracy-style profile with extra small-model guardrails.

Research-only bundle/feature controls are available with `--preset research`:

```bash
apdr --preset research --research-bundle baseline benchmark run --subset smoke30
apdr --preset research --research-bundle enhanced benchmark run --subset smoke30
apdr --preset research --research-bundle full benchmark run --subset smoke30
apdr --preset research --research-feature dynamic_imports --no-research-feature repair_memory smoke --jobs 1
```

You can also override only the prompt profile:

```bash
apdr --preset accuracy --prompt-profile paper benchmark run --subset smoke30
```

For completely fresh benchmarking without resume state or LLM cache reuse:

```bash
apdr --fresh-run benchmark run --subset smoke30
```

To disable only the LLM cache while keeping normal run IDs and artifact behavior:

```bash
apdr --no-llm-cache benchmark run --subset smoke30
```

To override individual model names instead of a bundle:

```bash
apdr --extraction-model gemma3:4b --reasoning-model qwen3.5:9b smoke --jobs 1
```

You can also disable or mix individual runtime features:

```bash
apdr --no-moe --runner-model qwen3.5:9b benchmark run --subset smoke30
apdr --no-rag --no-langchain --extraction-model gemma3:1b --runner-model gemma3:4b smoke --jobs 1
apdr --extraction-model qwen3.5:0.8b --runner-model qwen3.5:4b --version-model qwen3.5:9b --repair-model gpt-oss:20b benchmark segment --jobs 2
```

## Module reporting

Canonical module-family reporting is now the default:

```bash
apdr report modules --run-id <run-id>
```

To inspect the ungrouped raw module buckets:

```bash
apdr report modules --run-id <run-id> --grouping raw --top 50
```

`competition-run` uses `all-gists` cases but filters to gist IDs found in official-result CSV files.
APDR reads CSV IDs from:
- `--competition-csv` (repeatable)
- `APDR_COMPETITION_RESULT_CSVS` (comma-separated absolute paths)
- default detected files:
  - `~/Downloads/pyego_results.csv`
  - `~/Downloads/summary-all-runs.csv`

APDR also supports a repo-tracked fallback filter file for cross-machine reproducibility:
- default path: `competition/competition-case-ids.txt`
- override with: `--competition-filter-file` or `APDR_COMPETITION_CASE_IDS_FILE`

To refresh the repo filter file from your current CSV inputs:

```bash
apdr benchmark save-competition-filter
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

If you run with `--trace-llm`, you can inspect prompts and model responses with:

```bash
apdr report trace --run-id <run-id>
apdr report trace --run-id <run-id> --case-id <case-id> --tail 200
```
