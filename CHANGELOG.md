# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- PLLM-inspired Python dependency resolver with a LangGraph workflow for extraction, version inference, repair, execution, and result finalization.
- Ollama-backed LangChain model routing using `gemma3:4b` for extraction and `gemma3:12b` for versioning, repair, and adjudication.
- Six user-selectable execution presets spanning `performance` through `accuracy`, including `efficient` and `thorough` intermediate tradeoff levels.
- Prompt-profile support with `paper`, `optimized-lite`, `optimized`, and `optimized-strict` prompt sets.
- CLI entrypoint `apd` with benchmark, case, project, and reporting commands.
- One-step benchmark wrapper commands for running the prepared segment and full benchmark without manually chaining setup steps.
- Newcomer-friendly top-level commands `apd doctor`, `apd smoke`, `apd full`, and `apd solve`.
- Interactive terminal command `apd ui` for guided benchmark, reporting, and local-project workflows.
- Case timeline tracking with per-case and per-attempt UTC timestamps, plus `apd report timeline` artifacts for run-level timing analysis.
- Gistable benchmark support pinned to the paper dataset commit, including deterministic `smoke30` subset generation.
- Docker-based execution pipeline for benchmark cases and general project validation using the host Docker daemon.
- PyPI metadata caching with raw JSON storage and indexed release filtering.
- System-package bootstrap rules for native benchmark dependencies such as `pygame`.
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
- Added automatic `smoke30` regeneration before segment runs and explicit `apd benchmark segment` / `apd benchmark full` entrypoints.
- Reordered benchmark Docker build layers so dependency installation can reuse cache before copying snippet source files.
- Reduced repeated benchmark overhead by reusing prepared execution contexts and caching LLM responses on disk.
- Added a simpler first-run CLI surface and help text so new users can avoid the nested benchmark subcommands.
- Added a newcomer-oriented health check flow with `apd doctor` and beginner-friendly top-level commands.
- Added a live stdout benchmark progress bar with completed-case counts, percent complete, and elapsed time for serial and parallel runs.
- Integrated the benchmark runner with the terminal UI so smoke/full runs can render a live dashboard with preset, prompt profile, active cases, success/failure counts, elapsed time, and final artifact paths.
- Replaced the dry hand-rolled terminal UI with a prompt-toolkit powered command center that uses styled dialogs and a fullscreen benchmark dashboard.
- Added canonical module-family reporting with raw fallback mode for paper-style comparison tables.
- Added preset-aware routing for prompt usage, deterministic version selection, repair behavior, and compatibility handling.
- Added fresh-run and no-LLM-cache execution options plus bundled model selection for `gemma-moe`, `qwen35-9b`, and `gpt-oss-20b`, with per-stage extraction/reasoning model overrides.
- Added dependency-reason, prompt-profile, preset, candidate-provenance, and compatibility-policy metadata to run artifacts and summaries.
- Added timeline artifacts (`timeline.json`, `timeline.csv`, `timeline.md`) and exposed timeline viewing in the interactive terminal UI.
- Switched documentation away from Compose sidecars to host Ollama plus host Docker usage for local GPU testing.

### Fixed

- Fixed host runtime defaults for Ollama and Docker so local execution works outside Compose.
- Fixed SSL certificate handling for Gistable fetches and PyPI metadata downloads on macOS environments with incomplete trust stores.
- Fixed PyPI 404 handling so unresolved package names do not abort a benchmark run.
- Fixed Windows PyPI metadata-store failures by sanitizing reserved cache filenames, locking cache writes, and degrading DuckDB indexing to a non-fatal best-effort path.
- Fixed import extraction for Python 2 snippets by falling back to syntax-tolerant parsing when `ast.parse()` fails.
- Fixed Docker build and run timeouts so they are recorded as failed attempts instead of crashing the workflow.
- Fixed LangGraph recursion-limit failures for legitimate retry paths.
- Fixed malformed model-output handling for fenced Markdown, prose responses, and placeholder versions such as `package==<version>`.
- Fixed repair-loop stalls by early-stopping unusable retries and adjudicating malformed repair output instead of crashing.
- Fixed repeated false-positive dependency additions such as stdlib modules, packaging tools, and unrelated repair suggestions.
- Fixed benchmark noise from upstream LangChain debug deprecation warnings by suppressing the warning in the CLI.
- Fixed native package classification for missing system prerequisites such as SDL during `pygame` builds.
- Fixed local-module/API mismatches such as `compress` being treated as generic runtime errors instead of non-PyPI dependency issues.
- Fixed overly permissive package extraction by tracking candidate provenance and rejecting more non-package prompt output before PyPI lookup.
- Fixed long PyPI cache filenames by truncating and hashing cache keys safely.
- Fixed Windows Docker subprocess handling when `stdout` or `stderr` is `None`.
- Fixed the terminal UI dependency path by declaring `prompt_toolkit` as a runtime dependency for `apd ui`.
- Added a trace-view CLI path so prompt/response logs written by `--trace-llm` can be inspected directly without manually opening artifact files.
- Fixed `apd ui` benchmark shutdown so `Ctrl+C` requests a clean cooperative stop instead of being swallowed by the fullscreen dashboard.
