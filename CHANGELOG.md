# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0] - 2026-03-08

### Added

- Added a device-hosted web dashboard and API for benchmark runs, including a Vite frontend, network-accessible run/case views, and per-case attempt drilldowns with activity timelines and artifact links.
- Added per-attempt post-failure LLM analysis so failed benchmark attempts now persist a short model-written explanation in `result.json` and display it directly in the web attempt view.
- Added terminal/web command-center parity improvements including a web launch option in the CLI UI, command-center metadata on the web homepage, and version display in benchmark run info.

### Changed

- Redesigned the web dashboard into a premium terminal-inspired interface with tabbed navigation, collapsible activity sections, scrollable completed-case views, and improved live-run readability for longer benchmarks.
- Updated benchmark progress and attempt rendering so PLLM comparison status, cache/build-skip metrics, timing fields, and attempt-level context stay visible across resume flows and web refreshes.
- Improved prompt plumbing so new optional research variables and post-attempt analysis prompts render safely through both workflow-side formatting and direct prompt-runner template invocation.

### Fixed

- Fixed shutdown and resume UX regressions by restoring completed-case tables on resume, handling `Ctrl-C`/signal exits cleanly, and preventing missing prompt variables from crashing active runs.
- Fixed benchmark validation regressions in Python 2 and legacy-stack cases, including safer import-statement handling, better deferred fallback behavior, and more reliable repair prompt context.
- Fixed dashboard rendering gaps around PLLM match semantics, run metadata restoration, and web attempt detail presentation so current benchmark state remains inspectable while runs are in progress.

## [2.1.0] - 2026-03-08

### Added

- Added persistent Docker environment reuse, build-skip accounting, and run-level timing breakdowns for Docker build, Docker run, and LLM time so benchmark performance is measurable and repeated environments are reused across attempts.
- Added richer repair guidance and context, including conflict notes, validator-aware runtime-profile repair rules, cross-package `requires_dist` compatibility instructions, and duplicate-plan avoidance in repair prompts.
- Added benchmark UI/runtime visibility improvements including displayed app version, live cache/build-skip totals, and stronger prompt-rendering fallbacks in the shared prompt runner.

### Changed

- Reworked benchmark Docker image generation to be cache-friendly: dependency installation now happens before workspace content, BuildKit pip cache mounts are used, and generated environments are mounted at runtime instead of copied into each image.
- Improved model-led planning context for legacy stacks by surfacing better source compatibility evidence for TensorFlow, Keras, Gym, and related coupled dependency families instead of relying on narrow deterministic steering.
- Expanded repair and candidate planning context so the model can reason about family-level compatibility, validation-strategy changes, and conflict notes directly from prompt inputs.

### Fixed

- Fixed multiple benchmark regressions caused by Python-2-incompatible import-statement validation helpers, missing optional prompt variables in live prompt rendering, and prompt-template crashes after adding new repair fields.
- Fixed repeated false negatives in legacy TensorFlow/Keras/Gym and Python 2 cases by improving deferred fallback behavior, safer validation profiles, repair delta handling, and compatibility-hint propagation.
- Fixed resume/runtime UX regressions around restored tables, PLLM match display, runtime-version display, and prompt compatibility between workflow-side rendering and direct prompt-runner rendering.

## [2.0.0] - 2026-03-08

### Added

- Added model-selectable runtime profiles for benchmark validation, including import-spec probing for hardware-sensitive ML stacks and richer validation context passed into research planning and repair prompts.
- Added PLLM baseline integration to the benchmark UX, including vendored competition CSV support, live dashboard match status, run-table PASS/FAIL parity columns, and richer reporting slices for classifier origin, root-cause buckets, and deferred Python fallback usage.
- Added package-metadata parsing/caching, loadout persistence, and expanded dashboard resume hydration so restarted benchmark sessions preserve prior completed-case rows and runtime settings.

### Changed

- Reworked research-mode planning so the APDR model owns candidate selection and repair direction: compatibility heuristics now flow into prompt context instead of silently hard-steering version order, and Python fallback triggers a fresh re-plan rather than deterministic pin reconstruction.
- Separated build-log and run-log failure classification, widened retry routing to prefer fallback/repair unless a failure is categorically impossible, and allowed repair plans to merge partial dependency deltas instead of requiring complete replacement plans.
- Updated benchmark baselines and comparison logic to treat PLLM pass/fail parity as the primary match signal for competition runs, even when exact failure labels differ.

### Fixed

- Fixed multiple benchmark false negatives caused by eager Python 2 downgrades, duplicate repair retries, shell-unsafe bootstrap requirements, prompt-template crashes, misclassified TensorFlow/Keras/Gym compatibility errors, and stale resume-time runtime-config checks.
- Fixed dashboard regressions where resumed runs lost completed-case tables or crashed on missing prompt variables, and improved live benchmarking output so PLLM comparison status and restored rows stay visible after resume.
- Fixed several build/bootstrap failure paths by surfacing concrete retry hints for missing system headers, Cython compatibility, typing on Python 2, and related native-build traps instead of collapsing them into `UnknownError`.

## [1.1.6] - 2026-03-05

### Added

- Added a research alias-resolution prompt stage (`resolve_aliases.txt`) that asks the model to map unresolved import names to probable PyPI package names before candidate-plan generation.
- Added `alias-resolutions.json` case artifacts with unresolved-before/after snapshots plus resolved and rejected alias mappings.

### Changed

- Wired the research graph/fallback flow to execute `resolve_aliases` between PyPI metadata retrieval and version-specific metadata retrieval.
- Updated research RAG context to include `unresolved_packages` so downstream planning prompts see unresolved import evidence.
- Updated `pypi-evidence.json` writing to include alias-resolution outcomes when applicable.

### Fixed

- Fixed unresolved imports being silently dropped in research runs by validating LLM-proposed import-to-package aliases against PyPI and reinserting valid mappings into planning inputs.

## [1.1.5] - 2026-03-05

### Changed

- Broadened experimental/research version-candidate windows for older Python targets so precheck and planning can consider older compatible releases instead of only recent versions.
- Updated constraint-pack selection in experimental/research flows to use target-Python-aware top-k limits.

### Fixed

- Fixed Python-target reconciliation to preserve Python 2 compatibility signals (syntax and Python-2-only stdlib imports) even when no benchmark Dockerfile target is available.
- Fixed structured-output fallback behavior that was forcing `target_python` to benchmark defaults (often `3.12`) in non-docker benchmark sources.
- Fixed aggressive `ConstraintConflictError` precheck behavior by treating conflict notes as advisory and only failing precheck on invalid Python-constraint intersection.

## [1.1.4] - 2026-03-05

### Changed

- Reduced benchmark startup overhead by reusing the already-fetched dataset object between run preparation and execution instead of fetching twice per invocation.
- Optimized `competition-run` startup selection to iterate directly over filtered competition case IDs and check existence, rather than scanning all `all-gists` directories first.

### Fixed

- Fixed slow startup behavior on large Windows repositories by removing redundant filesystem scans in competition filter resolution and case selection.

## [1.1.3] - 2026-03-04

### Added

- Added a repo-tracked competition gist filter file at `competition/competition-case-ids.txt` so `competition-run` selections are reproducible across machines after `git pull`.
- Added `apdr benchmark save-competition-filter` to build/sync the repo filter file from configured official CSV sources.
- Added `--competition-filter-file` CLI override and `APDR_COMPETITION_CASE_IDS_FILE` support for custom filter-file paths.
- Added dataset tests for competition filter fallback and sync behavior.

### Changed

- Updated `competition-run` case selection to:
  - prefer CSV-derived IDs when available,
  - auto-sync those IDs into the filter file,
  - fall back to the repo filter file when CSVs are missing.
- Updated README competition-run docs with repo filter workflow for cross-system use.

## [1.1.2] - 2026-03-04

### Changed

- Updated benchmark summary generation to use active run metadata defaults when a run has no completed case artifacts, so preset/profile/resolver fields remain accurate in `summary.json`.
- Updated the standalone summary command to read run-state defaults, keeping regenerated summaries consistent with the original run configuration.

### Fixed

- Fixed a zero-case benchmark bug where runs could exit with status `0` and appear completed even when no cases were selected.
- Fixed zero-case runs being mislabeled as `optimized` by default in `summary.json`; summaries now preserve the selected preset (including `research`) from run settings/state.
- Added explicit empty-run handling and persisted `run-state` status `empty` with a clear error marker to avoid silent no-op benchmark runs.

## [1.1.1] - 2026-03-04

### Added

- Per-run runtime comparison artifacts that join APDR outcomes with official CSV fields during execution:
  - `run-vs-csv.csv`
  - `run-vs-csv.md`
- Per-run gist match exports for quick parity checks:
  - `gistid-matches.csv` (`gistid,matches`)
  - `gistid-matches-detailed.csv` (`matches_passed` and `matches_official_result` plus mapping context)

### Changed

- Extended benchmark execution to update comparison and match artifacts incrementally as each case completes, including resumed runs with pre-existing case results.
- Improved PyEGo local Neo4j bootstrap flow in UI with stronger automatic setup sequencing and clearer diagnostics in failure dialogs.

### Fixed

- Fixed Apple Silicon local Neo4j setup by automatically retrying with `--platform linux/amd64` when the Neo4j image has no native arm64 manifest.
- Fixed local Neo4j setup reliability with longer pull/load timeouts and explicit timeout diagnostics.
- Fixed Neo4j bootstrap command compatibility by resolving `neo4j-admin` from known container paths when it is not on `PATH`.
- Fixed Neo4j 3.5 database load failures by creating `/data/databases/graph.db` before import in the setup container flow.
- Fixed repeated PyEGo run failures from unreachable Neo4j by adding in-flow UI remediation prompts that can auto-run local Neo4j setup and re-validate runtime requirements.

## [1.1.0] - 2026-03-04

### Added

- APDR UI home-level `Loadouts` menu with save/load/delete support for reusable resolver, preset, model, runtime, and research settings.
- Persistent loadout storage under `data/loadouts/*.json` with sanitized names and compatibility-safe restore behavior.
- Automatic official resolver dependency bootstrap in UI runs: `pip install -r requirements.txt` for `pyego` and `readpye` when a Python 3.11 interpreter is available, with per-interpreter/per-requirements cache markers under `data/runtime_bootstrap`.

### Changed

- UI resolver switching now auto-detects and assigns a Python 3.11 interpreter for `pyego` (prefers `.venv-pyego`, then `python3.11`/`python311` on PATH).
- UI runtime validation now blocks `pyego`/`readpye` runs earlier with actionable setup details when official baseline prerequisites are not satisfied.
- Doctor and benchmark preflight now include stricter PyEGo runtime checks tied to interpreter compatibility and `typed_ast.ast27` availability.

### Fixed

- Prevented full-run `pyego` failure cascades caused by missing `typed_ast` by failing fast before scheduling benchmark cases.
- Reduced repeated official-baseline setup failures by caching successful bootstrap installs and reusing them across UI sessions.

## [1.0.0] - 2026-03-04

### Added

- Benchmark source selection across CLI and UI, including global `--benchmark-source` and `APDR_BENCHMARK_CASE_SOURCE` support for `all-gists` vs `dockerized-gists`.
- Benchmark source metadata in run-state/progress artifacts and per-case results (`case_source`) to keep resume/report behavior aligned with the executed corpus.
- Experimental v2 bundle/feature layer for the `experimental` preset, with `baseline`, `enhanced`, and `full` bundle modes plus per-feature overrides for dynamic aliases, transitive conflicts, smart repair routing, multipass inference, repair memory, Python constraint intersection, version negotiation, repair feedback, and dynamic imports.
- Workspace-local experimental helpers for package metadata extraction, repo-derived alias discovery, dynamic import detection, retry-policy routing, constraint-pack generation, candidate-bundle generation, and repair feedback memory.
- New research-rag prompt templates for package inference, package cross-validation, candidate plan generation, repair memory-aware repair planning, and version negotiation.
- Additional experimental artifacts including repo alias maps, top-level module maps, constraint packs, conflict notes, Python constraints, package candidate cross-validation, repair-memory summaries, strategy histories, candidate bundles, version negotiation results, dynamic imports, and error-routing reports.
- Experimental APDR-only preset with a hybrid-RAG workflow that gathers repo evidence, builds structured retrieval context, generates ranked candidate dependency plans, tries those plans before repair, and records repo/PyPI/RAG/plan artifacts for each case.
- Experimental prompt profile `research-rag` with strict JSON outputs for package inference, candidate-plan generation, and repair planning.
- Repo-evidence, RAG-context, and structured-output helper modules to support evidence-driven experimental runs without hardcoded package-specific final answers.
- Resolver selection support with `apdr`, `pyego`, and `readpye` runtime modes, including UI selection and artifact metadata so baseline-style runs can be compared from the same interface.
- Automatic per-run case results exports in `results.csv`, `results.md`, and `results.json`, including case number, modules, success/failure, attempts, dependencies, and timing fields.
- Persistent benchmark run-state artifacts in `run-state.json` and `run-state.md`, capturing live progress, elapsed time, active cases, last completed case, and resumable run metadata.
- Interactive resume-run selection in `apdr ui`, using saved `run-state.json` metadata to list resumable benchmarks and restart them with the original run ID, target, and saved progress context.
- Interactive failed-case replay selection in `apdr ui`, allowing users to pick a previous run with failures and rerun only those failed benchmark cases with the current runtime settings.
- PLLM-inspired Python dependency resolver with a LangGraph workflow for extraction, version inference, repair, execution, and result finalization.
- Ollama-backed LangChain model routing using `gemma3:4b` for extraction and `gemma3:12b` for versioning, repair, and adjudication.
- Six user-selectable execution presets spanning `performance` through `accuracy`, including `efficient` and `thorough` intermediate tradeoff levels.
- Prompt-profile support with `paper`, `optimized-lite`, `optimized`, and `optimized-strict` prompt sets.
- Runtime controls for toggling MoE routing, RAG, and LangChain independently, plus stage-specific model selection for extractor, runner, version, repair, and adjudication roles.
- CLI entrypoint `apdr` with benchmark, case, project, and reporting commands.
- One-step benchmark wrapper commands for running the prepared segment and full benchmark without manually chaining setup steps.
- Newcomer-friendly top-level commands `apdr doctor`, `apdr smoke`, `apdr full`, and `apdr solve`.
- Interactive terminal command `apdr ui` for guided benchmark, reporting, and local-project workflows.
- Case timeline tracking with per-case and per-attempt UTC timestamps, plus `apdr report timeline` artifacts for run-level timing analysis.
- Gistable benchmark support pinned to the paper dataset commit, including deterministic `smoke30` subset generation.
- Docker-based execution pipeline for benchmark cases and general project validation using the host Docker daemon.
- PyPI metadata caching with raw JSON storage and indexed release filtering.
- System-package bootstrap rules for native benchmark dependencies such as `pygame`.
- Prompt templates matching the paper's three prompt stages.
- Failure analysis reporting, module-level success-rate tables, summary artifacts, and benchmark leaderboard output.
- LLM trace logging to artifact files and on-disk LLM response caching for repeated benchmark runs.
- Unit, integration, and benchmark test coverage for import extraction, graph routing, Docker execution, version filtering, and reporting.

### Changed

- Switched Gistable benchmark execution default from `dockerized-gists` to `all-gists`, while retaining an explicit source switch for dockerized-only replay.
- Simplified the interactive command center layout to grouped `Run`, `Reports`, and `Configure` flows so the UI fits smaller terminal screens.
- Extended benchmark case loading to resolve snippets from the selected source and generate an ephemeral Dockerfile automatically when a case has no Dockerfile.
- Removed the `target_python` hint from experimental Prompt A and multipass package-inference prompts so APDR must infer the required Python version from code and evidence instead of inheriting the benchmark Dockerfile version as prompt input.
- Aligned Prompt A with the paper-style modules-plus-Python-version flow by having initial package inference return both dependency modules and an inferred Python version, recording benchmark vs inferred Python version metadata in run artifacts, and rewriting benchmark Docker base images to the inferred Python version for APDR-driven execution.
- Extended the experimental workflow so enhanced/full runs can perform repo-derived alias discovery, multipass package inference, version-specific metadata retrieval, constraint prechecks, feedback-memory loading, candidate-bundle generation, and version negotiation before execution.
- Extended the interactive UI and CLI to configure experimental bundles/features explicitly and preserve that configuration in run state, dashboards, and resumed runs.
- Extended reporting and result exports with experimental bundle, feature, retry-severity, strategy-type, conflict-precheck, and related experimental accuracy metrics.
- Replaced the initial starter scaffold with a functional `agentic_python_dependency` package and benchmark-oriented project structure.
- Improved benchmark Dockerfile patching to remove original Python package install lines and inject a controlled dependency installation sequence.
- Added Python 2 compatibility heuristics for release selection, including stricter cutoff rules and stable-release preference.
- Tightened repair handling so follow-up attempts stay constrained to originally inferred direct dependencies.
- Added runtime-profile heuristics for CLI, service, GUI, import-smoke, and raw `sys.argv` benchmark cases.
- Switched command output to file-first artifacts instead of dumping JSON to stdout.
- Added benchmark timing fields for total wall-clock completion time and human-readable duration.
- Added parallel benchmark workers via `apdr benchmark run --jobs N`.
- Added automatic `smoke30` regeneration before segment runs and explicit `apdr benchmark segment` / `apdr benchmark full` entrypoints.
- Reordered benchmark Docker build layers so dependency installation can reuse cache before copying snippet source files.
- Reduced repeated benchmark overhead by reusing prepared execution contexts and caching LLM responses on disk.
- Added a simpler first-run CLI surface and help text so new users can avoid the nested benchmark subcommands.
- Added a newcomer-oriented health check flow with `apdr doctor` and beginner-friendly top-level commands.
- Added a live stdout benchmark progress bar with completed-case counts, percent complete, and elapsed time for serial and parallel runs.
- Integrated the benchmark runner with the terminal UI so smoke/full runs can render a live dashboard with preset, prompt profile, active cases, success/failure counts, elapsed time, and final artifact paths.
- Extended the benchmark dashboard in `apdr ui` to show live success rate, seconds-per-case throughput, and estimated time to completion during active runs.
- Restored benchmark dashboards from saved run state when resuming with the same run ID, including previously accumulated elapsed time.
- Replaced the dry hand-rolled terminal UI with a prompt-toolkit powered command center that uses styled dialogs and a fullscreen benchmark dashboard.
- Added canonical module-family reporting with raw fallback mode for paper-style comparison tables.
- Added preset-aware routing for prompt usage, deterministic version selection, repair behavior, and compatibility handling.
- Added fresh-run and no-LLM-cache execution options plus bundled model selection for `gemma-moe`, `qwen35-9b`, and `gpt-oss-20b`, with per-stage extraction/reasoning model overrides.
- Added dependency-reason, prompt-profile, preset, candidate-provenance, and compatibility-policy metadata to run artifacts and summaries.
- Added experimental summary and result metadata including RAG mode, structured-prompting status, candidate-plan counts, selected candidate rank, repair-cycle counts, and structured-prompt failure counts.
- Added resolver metadata to doctor output, benchmark dashboards, result artifacts, and summary artifacts.
- Added timeline artifacts (`timeline.json`, `timeline.csv`, `timeline.md`) and exposed timeline viewing in the interactive terminal UI.
- Added a paper-compatible module report mode that builds top-module tables from the hard subset in `all-gists` so the reported module families line up more closely with the paper.
- Made paper-compatible module success rates coverage-aware for preview and partial runs, so APDR percentages reflect the cases actually executed while still preserving the full cohort sizes in the report.
- Parallelized module report generation across snippet reads and import extraction to improve throughput on large benchmark cohorts.
- Updated the interactive module report flow so `apdr ui` renders the generated markdown table instead of dumping raw captured command output.
- Updated the interactive reporting flows so summary, failures, module report, and timeline views list existing run directories and let users select a run instead of manually typing a run ID.
- Updated the CLI and terminal UI to validate that the `experimental` preset can only be used with the `apdr` resolver, and to preserve resolver/preset context when resuming runs from saved state.
- Updated paper-compatible markdown tables to prefer modules covered by the current run when displaying the top rows for preview and partial runs.
- Corrected the paper-compatible module-report cohort to use the repair benchmark set (`initial-eval = ImportError`) so APDR overlap and success-rate calculations match the benchmark runs.
- Switched documentation away from Compose sidecars to host Ollama plus host Docker usage for local GPU testing.

### Fixed

- Fixed module-report snippet lookup for mixed run corpora by resolving snippet paths from recorded case source metadata instead of assuming dockerized-only paths.
- Fixed gistable load/evidence/runtime paths to handle cases where Dockerfiles are absent, preventing all-gists cases from failing before execution.
- Fixed benchmark Python-version reconciliation so APDR keeps the benchmark Dockerfile Python when Prompt A infers a conflicting Python 3 version for code that is not even Python-3-syntax-compatible, preventing obvious Python 2 vs 3 syntax breakage during execution.
- Fixed Windows subprocess decoding in Docker execution and official-baseline wrappers by capturing byte output and decoding with UTF-8 replacement instead of relying on CP1252 text decoding, preventing `_readerthread` `UnicodeDecodeError` crashes during runs.
- Fixed research-rag prompt template rendering by escaping literal JSON braces in the new structured prompt files, preventing `str.format` crashes such as `KeyError: '\n  "packages"'` during experimental package inference.
- Fixed experimental classification routing so legacy experimental-baseline behavior does not silently inherit the new smart-repair retry policy unless that feature is explicitly enabled.
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
- Fixed `readpye` unpinned dependency plans being dropped during normalization by preserving already selected resolver-produced dependencies.
- Fixed paused or interrupted benchmark runs losing their saved progress context by persisting live run-state updates throughout execution and preserving partial progress on resume.
- Fixed benchmark noise from upstream LangChain debug deprecation warnings by suppressing the warning in the CLI.
- Fixed native package classification for missing system prerequisites such as SDL during `pygame` builds.
- Fixed local-module/API mismatches such as `compress` being treated as generic runtime errors instead of non-PyPI dependency issues.
- Fixed overly permissive package extraction by tracking candidate provenance and rejecting more non-package prompt output before PyPI lookup.
- Fixed long PyPI cache filenames by truncating and hashing cache keys safely.
- Fixed Windows Docker subprocess handling when `stdout` or `stderr` is `None`.
- Fixed `apdr ui` module reporting so paper-compatible reports no longer crash on unreadable `all-gists` snippets on Windows; unreadable cases are skipped and surfaced in the generated report instead.
- Fixed paper-compatible preview reports showing `0.00` success rates across the board by using covered-case denominators instead of the entire paper cohort when the run only covers a subset of cases.
- Fixed the terminal UI dependency path by declaring `prompt_toolkit` as a runtime dependency for `apdr ui`.
- Fixed module-report warning noise so `SyntaxWarning` output from benchmark snippets no longer floods the UI dialog when generating reports.
- Added a trace-view CLI path so prompt/response logs written by `--trace-llm` can be inspected directly without manually opening artifact files.
- Fixed `apdr ui` benchmark shutdown so `Ctrl+C` requests a clean cooperative stop instead of being swallowed by the fullscreen dashboard.
