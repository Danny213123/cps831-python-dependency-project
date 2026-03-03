# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- PLLM-inspired Python dependency resolver with a LangGraph workflow for extraction, version inference, repair, execution, and result finalization.
- Ollama-backed LangChain model routing using `gemma3:4b` for extraction and `gemma3:12b` for versioning, repair, and adjudication.
- CLI entrypoint `apd` with benchmark, case, project, and reporting commands.
- Gistable benchmark support pinned to the paper dataset commit, including deterministic `smoke30` subset generation.
- Docker-based execution pipeline for benchmark cases and general project validation, with Docker Compose support for `app`, `ollama`, and `executor-dind`.
- PyPI metadata caching with raw JSON storage and indexed release filtering.
- Prompt templates matching the paper's three prompt stages.
- Failure analysis reporting, module-level success-rate tables, summary artifacts, and benchmark leaderboard output.
- LLM trace logging to artifact files and on-disk LLM response caching for repeated benchmark runs.
- Unit, integration, and benchmark test coverage for import extraction, graph routing, Docker execution, version filtering, and reporting.

### Changed

- Replaced the initial starter scaffold with a functional `agentic_python_dependency` package and benchmark-oriented project structure.
- Improved benchmark Dockerfile patching to remove original Python package install lines and inject a controlled dependency installation sequence.
- Added Python 2 compatibility heuristics for release selection, including stricter cutoff rules and stable-release preference.
- Tightened repair handling so follow-up attempts stay constrained to originally inferred direct dependencies.
- Added runtime-profile heuristics for CLI, service, GUI, import-smoke, and raw `sys.argv` benchmark cases.
- Switched command output to file-first artifacts instead of dumping JSON to stdout.
- Added benchmark timing fields for total wall-clock completion time and human-readable duration.
- Added parallel benchmark workers via `apd benchmark run --jobs N`.
- Reordered benchmark Docker build layers so dependency installation can reuse cache before copying snippet source files.

### Fixed

- Fixed host runtime defaults for Ollama and Docker so local execution works outside Compose.
- Fixed SSL certificate handling for Gistable fetches and PyPI metadata downloads on macOS environments with incomplete trust stores.
- Fixed PyPI 404 handling so unresolved package names do not abort a benchmark run.
- Fixed import extraction for Python 2 snippets by falling back to syntax-tolerant parsing when `ast.parse()` fails.
- Fixed Docker build and run timeouts so they are recorded as failed attempts instead of crashing the workflow.
- Fixed LangGraph recursion-limit failures for legitimate retry paths.
- Fixed malformed model-output handling for fenced Markdown, prose responses, and placeholder versions such as `package==<version>`.
- Fixed repair-loop stalls by early-stopping unusable retries and adjudicating malformed repair output instead of crashing.
- Fixed repeated false-positive dependency additions such as stdlib modules, packaging tools, and unrelated repair suggestions.
- Fixed benchmark noise from upstream LangChain debug deprecation warnings by suppressing the warning in the CLI.
