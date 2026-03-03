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
- Docker Compose with an `app`, `ollama`, and `executor-dind` service
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

## CLI

```bash
apd benchmark fetch-gistable --ref 665d39a2bd82543d5196555f0801ef8fd4a3ee48
apd benchmark make-subsets
apd benchmark run --subset smoke30
apd case run --case-id 000769db6848429c9b3eac30361d9140
apd project solve --path /path/to/python/repo
apd report summarize --run-id <run-id>
```

## Docker Compose

Start the full runtime:

```bash
docker compose up --build
```

Then pull the required models in the Ollama container:

```bash
docker compose exec ollama ollama pull gemma3:4b
docker compose exec ollama ollama pull gemma3:12b
```

Run a smoke benchmark from the app container:

```bash
docker compose exec app apd benchmark fetch-gistable
docker compose exec app apd benchmark make-subsets
docker compose exec app apd benchmark run --subset smoke30
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
