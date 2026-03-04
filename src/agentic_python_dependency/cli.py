from __future__ import annotations

import argparse
import contextlib
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import shutil
import sys
import time
import threading
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from agentic_python_dependency.benchmark.gistable import GistableDataset
from agentic_python_dependency.benchmark.subsets import build_smoke30
from agentic_python_dependency.config import MODEL_PROFILE_DEFAULTS, Settings
from agentic_python_dependency.presets import (
    RESEARCH_BUNDLE_DEFAULTS,
    RESEARCH_FEATURES,
    PRESET_CONFIGS,
)
from agentic_python_dependency.graph import ResolutionWorkflow
from agentic_python_dependency.reporting import analyze_failures, build_module_success_table, build_timeline_view, summarize_run
from agentic_python_dependency.terminal_ui import launch_terminal_ui
from agentic_python_dependency.tools.official_baselines import validate_pyego_runtime

RUNTIME_COMPARISON_COLUMNS = [
    "case_number",
    "case_id",
    "run_result",
    "run_success",
    "run_final_error_category",
    "run_attempts",
    "run_wall_clock_seconds",
    "run_started_at",
    "run_finished_at",
    "official_in_csv",
    "official_result",
    "official_passed",
    "official_duration",
    "official_python_modules",
    "official_file",
    "official_csv_sources",
]

RUNTIME_COMPARISON_MD_COLUMNS = [
    "case_number",
    "case_id",
    "run_result",
    "run_final_error_category",
    "run_wall_clock_seconds",
    "official_in_csv",
    "official_result",
    "official_passed",
    "official_duration",
    "official_csv_sources",
]

GIST_MATCH_COLUMNS = ["gistid", "matches"]
GIST_MATCH_DETAILED_COLUMNS = [
    "gistid",
    "matches_passed",
    "matches_official_result",
    "run_official_result",
    "official_result",
    "run_success",
    "official_passed",
    "run_final_error_category",
]


def _notify_path(label: str, path: Path) -> None:
    print(f"{label}: {path}", file=sys.stderr)


def format_model_summary(settings: Settings) -> str:
    stage_models = settings.active_stage_models()
    routing = "moe" if settings.use_moe else "single"
    backend = "langchain" if settings.use_langchain else "direct"
    rag = "rag" if settings.use_rag else "no-rag"
    return (
        f"{settings.model_profile} [{routing}/{backend}/{rag}] "
        f"extract={stage_models['extract']} "
        f"runner={stage_models['runner']} "
        f"version={stage_models['version']} "
        f"repair={stage_models['repair']} "
        f"adjudicate={stage_models['adjudicate']}"
    )


def summary_defaults_from_settings(settings: Settings) -> dict[str, object]:
    stage_models = settings.active_stage_models()
    return {
        "resolver": settings.resolver,
        "preset": settings.preset,
        "prompt_profile": settings.prompt_profile,
        "model_profile": settings.model_profile,
        "use_moe": settings.use_moe,
        "use_rag": settings.use_rag,
        "use_langchain": settings.use_langchain,
        "rag_mode": settings.rag_mode,
        "structured_prompting": settings.structured_prompting,
        "research_bundle": settings.research_bundle,
        "research_features": list(settings.research_features),
        "extraction_model": stage_models["extract"],
        "runner_model": stage_models["runner"],
        "version_model": stage_models["version"],
        "repair_model": stage_models["repair"],
        "adjudication_model": stage_models["adjudicate"],
    }


class BenchmarkObserver(Protocol):
    def start(
        self,
        *,
        run_id: str,
        total: int,
        completed: int,
        successes: int,
        failures: int,
        resolver: str,
        preset: str,
        prompt_profile: str,
        model_summary: str,
        research_bundle: str = "baseline",
        research_features: tuple[str, ...] = (),
        benchmark_source: str = "all-gists",
        jobs: int,
        target: str,
        artifacts_dir: Path,
        elapsed_seconds: float = 0.0,
    ) -> None: ...

    def case_started(self, case_id: str) -> None: ...

    def advance(self, result: dict[str, object]) -> None: ...

    def stop_requested(self) -> bool: ...

    def finish(self, *, summary_path: Path, warnings_path: Path | None, status: str = "completed") -> None: ...


@contextlib.contextmanager
def redirect_runtime_warnings(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_showwarning = warnings.showwarning
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        def _showwarning(
            message: warnings.WarningMessage | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file=None,
            line: str | None = None,
        ) -> None:
            handle.write(warnings.formatwarning(message, category, filename, lineno, line))
            handle.flush()

        warnings.showwarning = _showwarning
        with contextlib.redirect_stderr(handle):
            try:
                yield output_path
            finally:
                warnings.showwarning = original_showwarning


def load_run_state(run_dir: Path) -> dict[str, object]:
    state_path = run_dir / "run-state.json"
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_run_state(run_dir: Path, payload: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run-state.json"
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    active_cases = payload.get("current_cases", [])
    if not isinstance(active_cases, list):
        active_cases = []
    lines = [
        "# Run Status",
        "",
        f"- Run ID: `{payload.get('run_id', run_dir.name)}`",
        f"- Status: `{payload.get('status', 'unknown')}`",
        f"- Resolver: `{payload.get('resolver', 'apdr')}`",
        f"- Preset: `{payload.get('preset', 'optimized')}`",
        f"- Benchmark source: `{payload.get('benchmark_source', 'all-gists')}`",
        f"- Prompt profile: `{payload.get('prompt_profile', 'optimized')}`",
        f"- Research bundle: `{payload.get('research_bundle', 'baseline')}`",
        f"- Research features: `{', '.join(payload.get('research_features', [])) if isinstance(payload.get('research_features', []), list) and payload.get('research_features', []) else 'none'}`",
        f"- Jobs: `{payload.get('jobs', 1)}`",
        f"- Target: `{payload.get('target', 'benchmark')}`",
        f"- Progress: `{payload.get('completed', 0)}/{payload.get('total', 0)}`",
        f"- Successes: `{payload.get('successes', 0)}`",
        f"- Failures: `{payload.get('failures', 0)}`",
        f"- Elapsed: `{format_elapsed(float(payload.get('elapsed_seconds', 0.0) or 0.0))}`",
        f"- Started at: `{payload.get('started_at', '') or 'unknown'}`",
        f"- Last updated: `{payload.get('last_updated_at', '') or 'unknown'}`",
        f"- Last case: `{payload.get('last_case_id', '') or 'none'}`",
        f"- Last status: `{payload.get('last_status', '') or 'none'}`",
        "",
    ]
    if active_cases:
        lines.append("## Active Cases")
        lines.append("")
        for case_id in active_cases:
            lines.append(f"- `{case_id}`")
        lines.append("")
    summary_path = payload.get("summary_path")
    warnings_path = payload.get("warnings_path")
    if summary_path or warnings_path:
        lines.append("## Artifacts")
        lines.append("")
        if summary_path:
            lines.append(f"- Summary: `{summary_path}`")
        if warnings_path:
            lines.append(f"- Warnings: `{warnings_path}`")
        lines.append("")
    error_text = str(payload.get("last_error", "") or "").strip()
    if error_text:
        lines.extend(["## Last Error", "", error_text, ""])
    (run_dir / "run-state.md").write_text("\n".join(lines), encoding="utf-8")


class PersistentBenchmarkObserver:
    def __init__(self, inner: BenchmarkObserver, run_dir: Path, restored_state: dict[str, object] | None = None):
        self.inner = inner
        self.run_dir = run_dir
        self.restored_state = restored_state or {}
        self.session_started_at = time.monotonic()
        self.prior_elapsed_seconds = float(self.restored_state.get("elapsed_seconds", 0.0) or 0.0)
        self.started_at = str(self.restored_state.get("started_at", "") or datetime.now(timezone.utc).isoformat())
        self.payload: dict[str, object] = {
            "run_id": self.restored_state.get("run_id", run_dir.name),
            "status": "pending",
            "resolver": self.restored_state.get("resolver", "apdr"),
            "preset": self.restored_state.get("preset", "optimized"),
            "benchmark_source": self.restored_state.get("benchmark_source", "all-gists"),
            "prompt_profile": self.restored_state.get("prompt_profile", "optimized"),
            "research_bundle": self.restored_state.get("research_bundle", "baseline"),
            "research_features": self.restored_state.get("research_features", []),
            "jobs": int(self.restored_state.get("jobs", 1) or 1),
            "target": self.restored_state.get("target", "benchmark"),
            "total": int(self.restored_state.get("total", 0) or 0),
            "completed": int(self.restored_state.get("completed", 0) or 0),
            "successes": int(self.restored_state.get("successes", 0) or 0),
            "failures": int(self.restored_state.get("failures", 0) or 0),
            "elapsed_seconds": self.prior_elapsed_seconds,
            "started_at": self.started_at,
            "last_updated_at": "",
            "current_cases": [],
            "last_case_id": str(self.restored_state.get("last_case_id", "") or ""),
            "last_status": str(self.restored_state.get("last_status", "") or ""),
            "summary_path": self.restored_state.get("summary_path"),
            "warnings_path": self.restored_state.get("warnings_path"),
            "stop_requested": False,
        }

    def _elapsed_seconds(self) -> float:
        return self.prior_elapsed_seconds + (time.monotonic() - self.session_started_at)

    def _persist(self, *, status: str | None = None, last_error: str | None = None) -> None:
        if status is not None:
            self.payload["status"] = status
        if last_error is not None:
            self.payload["last_error"] = last_error
        self.payload["elapsed_seconds"] = self._elapsed_seconds()
        self.payload["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        write_run_state(self.run_dir, self.payload)

    def start(
        self,
        *,
        run_id: str,
        total: int,
        completed: int,
        successes: int,
        failures: int,
        resolver: str,
        preset: str,
        prompt_profile: str,
        model_summary: str,
        research_bundle: str = "baseline",
        research_features: tuple[str, ...] = (),
        benchmark_source: str = "all-gists",
        jobs: int,
        target: str,
        artifacts_dir: Path,
    ) -> None:
        self.payload.update(
            {
                "run_id": run_id,
                "status": "running",
                "resolver": resolver,
                "preset": preset,
                "benchmark_source": benchmark_source,
                "prompt_profile": prompt_profile,
                "research_bundle": research_bundle,
                "research_features": list(research_features),
                "model_summary": model_summary,
                "jobs": jobs,
                "target": target,
                "artifacts_dir": str(artifacts_dir),
                "total": total,
                "completed": completed,
                "successes": successes,
                "failures": failures,
                "current_cases": [],
                "stop_requested": False,
            }
        )
        self._persist(status="running")
        self.inner.start(
            run_id=run_id,
            total=total,
            completed=completed,
            successes=successes,
            failures=failures,
            resolver=resolver,
            preset=preset,
            prompt_profile=prompt_profile,
            model_summary=model_summary,
            research_bundle=research_bundle,
            research_features=research_features,
            benchmark_source=benchmark_source,
            jobs=jobs,
            target=target,
            artifacts_dir=artifacts_dir,
            elapsed_seconds=self.prior_elapsed_seconds,
        )

    def case_started(self, case_id: str) -> None:
        current_cases = list(self.payload.get("current_cases", []))
        if case_id not in current_cases:
            current_cases.append(case_id)
        self.payload["current_cases"] = current_cases
        self._persist()
        self.inner.case_started(case_id)

    def advance(self, result: dict[str, object]) -> None:
        case_id = str(result.get("case_id", ""))
        current_cases = [item for item in list(self.payload.get("current_cases", [])) if item != case_id]
        success = bool(result.get("success", False))
        self.payload["current_cases"] = current_cases
        self.payload["completed"] = int(self.payload.get("completed", 0) or 0) + 1
        self.payload["successes"] = int(self.payload.get("successes", 0) or 0) + int(success)
        self.payload["failures"] = int(self.payload.get("failures", 0) or 0) + int(not success)
        self.payload["last_case_id"] = case_id
        self.payload["last_status"] = "success" if success else str(result.get("final_error_category", "failure"))
        self._persist()
        self.inner.advance(result)

    def stop_requested(self) -> bool:
        should_stop = self.inner.stop_requested()
        if should_stop and not bool(self.payload.get("stop_requested", False)):
            self.payload["stop_requested"] = True
            self._persist(status="stopping")
        return should_stop

    def finish(self, *, summary_path: Path, warnings_path: Path | None, status: str = "completed") -> None:
        self.payload["current_cases"] = []
        self.payload["summary_path"] = str(summary_path)
        self.payload["warnings_path"] = str(warnings_path) if warnings_path is not None else ""
        self._persist(status=status)
        self.inner.finish(summary_path=summary_path, warnings_path=warnings_path, status=status)

    def abort(self, exc: BaseException) -> None:
        self.payload["current_cases"] = []
        self._persist(status="interrupted", last_error=f"{type(exc).__name__}: {exc}")


def format_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("#" * width) + "]"
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class BenchmarkProgress:
    def __init__(self, run_id: str, total: int, completed: int = 0, refresh_interval: float = 1.0):
        self.run_id = run_id
        self.total = total
        self.completed = completed
        self.successes = 0
        self.failures = 0
        self.current_case_id = ""
        self.last_case_id = ""
        self.last_status = ""
        self.resolver = "apdr"
        self.preset = "optimized"
        self.prompt_profile = "optimized"
        self.research_bundle = "baseline"
        self.research_features: tuple[str, ...] = ()
        self.benchmark_source = "all-gists"
        self.model_summary = "gemma-moe: gemma3:4b / gemma3:12b"
        self.jobs = 1
        self.target = "benchmark"
        self.active_cases = 0
        self.artifacts_dir = Path(".")
        self.started_at = time.monotonic()
        self._lock = threading.RLock()
        self._isatty = sys.stdout.isatty()
        self._refresh_interval = refresh_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cancel_requested = False

    def _line(self) -> str:
        bar = format_progress_bar(self.completed, self.total)
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        status_bits = [f"ok {self.successes}", f"fail {self.failures}"]
        if self.current_case_id:
            status_bits.append(f"current {self.current_case_id}")
        elif self.last_case_id:
            status_bits.append(f"last {self.last_case_id}:{self.last_status}")
        return (
            f"Benchmark {self.run_id} {bar} "
            f"{self.completed}/{self.total} {percent:5.1f}% "
            f"resolver {self.resolver} preset {self.preset}/{self.research_bundle} {' '.join(status_bits)} "
            f"source {self.benchmark_source} "
            f"elapsed {format_elapsed(time.monotonic() - self.started_at)}"
        )

    def render(self) -> None:
        with self._lock:
            line = self._line()
            if self._isatty:
                print(f"\r{line}", end="", file=sys.stdout, flush=True)
            else:
                print(line, file=sys.stdout, flush=True)

    def start(
        self,
        *,
        run_id: str,
        total: int,
        completed: int,
        successes: int,
        failures: int,
        resolver: str,
        preset: str,
        prompt_profile: str,
        model_summary: str,
        research_bundle: str = "baseline",
        research_features: tuple[str, ...] = (),
        benchmark_source: str = "all-gists",
        jobs: int,
        target: str,
        artifacts_dir: Path,
        elapsed_seconds: float = 0.0,
    ) -> None:
        self.run_id = run_id
        self.total = total
        self.completed = completed
        self.successes = successes
        self.failures = failures
        self.resolver = resolver
        self.preset = preset
        self.prompt_profile = prompt_profile
        self.research_bundle = research_bundle
        self.research_features = tuple(research_features)
        self.benchmark_source = benchmark_source
        self.model_summary = model_summary
        self.jobs = jobs
        self.target = target
        self.artifacts_dir = artifacts_dir
        self.started_at = time.monotonic() - elapsed_seconds
        self.render()
        if not self._isatty or self._thread is not None:
            return

        def _refresh_loop() -> None:
            while not self._stop_event.wait(self._refresh_interval):
                self.render()

        self._thread = threading.Thread(target=_refresh_loop, name="benchmark-progress", daemon=True)
        self._thread.start()

    def case_started(self, case_id: str) -> None:
        with self._lock:
            self.current_case_id = case_id
            self.active_cases += 1
            self.render()

    def advance(self, result: dict[str, object]) -> None:
        success = bool(result.get("success", False))
        case_id = str(result.get("case_id", ""))
        with self._lock:
            self.completed = min(self.total, self.completed + 1)
            self.successes += int(success)
            self.failures += int(not success)
            self.last_case_id = case_id
            self.last_status = "ok" if success else str(result.get("final_error_category", "fail"))
            if self.current_case_id == case_id:
                self.current_case_id = ""
            self.active_cases = max(0, self.active_cases - 1)
            self.render()

    def request_stop(self) -> None:
        with self._lock:
            self._cancel_requested = True
            self.render()

    def stop_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def _finish_render(self, status: str) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._refresh_interval + 0.1)
        with self._lock:
            if status == "completed":
                self.completed = self.total
            line = self._line()
            if self._isatty:
                print(f"\r{line}", end="", file=sys.stdout, flush=True)
                print(file=sys.stdout, flush=True)
            else:
                print(line, file=sys.stdout, flush=True)

    def finish(self, *, summary_path: Path, warnings_path: Path | None, status: str = "completed") -> None:
        self._finish_render(status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apdr",
        description="Agentic Python dependency resolver.",
        epilog=(
            "Beginner-friendly commands:\n"
            "  apdr doctor\n"
            "  apdr smoke --jobs 1\n"
            "  apdr full --jobs 1\n"
            "  apdr solve --path /path/to/repo\n\n"
            "Advanced commands:\n"
            "  apdr benchmark segment --jobs 2\n"
            "  apdr benchmark full --jobs 2\n"
            "  apdr report summarize --run-id <run_id>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trace-llm",
        action="store_true",
        help="Write prompts and model outputs to llm-trace.log files during execution.",
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Start from a clean run directory and bypass the on-disk LLM cache for this invocation.",
    )
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        help="Disable reading and writing the on-disk LLM cache for this invocation.",
    )
    parser.add_argument(
        "--resolver",
        choices=["apdr", "pyego", "readpye"],
        default=None,
        help="Select the resolver strategy: full APDR loop, PyEGo-style baseline, or ReadPyE-style baseline.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS),
        default=None,
        help="Select the execution preset that controls speed vs. accuracy behavior.",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["paper", "optimized-lite", "optimized", "optimized-strict", "research-rag"],
        default=None,
        help="Override the prompt profile independently of the selected preset.",
    )
    parser.add_argument(
        "--research-bundle",
        choices=sorted(RESEARCH_BUNDLE_DEFAULTS),
        default=None,
        help="Select the research feature bundle when --preset research is active.",
    )
    parser.add_argument(
        "--research-feature",
        action="append",
        choices=list(RESEARCH_FEATURES),
        default=[],
        help="Enable an additional research feature. Repeatable.",
    )
    parser.add_argument(
        "--no-research-feature",
        action="append",
        choices=list(RESEARCH_FEATURES),
        default=[],
        help="Disable a research feature from the selected bundle. Repeatable.",
    )
    parser.add_argument(
        "--model-profile",
        choices=sorted(MODEL_PROFILE_DEFAULTS),
        default=None,
        help="Select the Ollama model bundle to use for extraction and reasoning.",
    )
    parser.add_argument(
        "--moe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable stage-specific model routing.",
    )
    parser.add_argument(
        "--rag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable retrieval-augmented version selection with PyPI metadata.",
    )
    parser.add_argument(
        "--langchain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the LangChain-backed Ollama client.",
    )
    parser.add_argument(
        "--extraction-model",
        default=None,
        help="Override the extraction-stage Ollama model name.",
    )
    parser.add_argument(
        "--runner-model",
        default=None,
        help="Override the shared runner/reasoning model name.",
    )
    parser.add_argument(
        "--reasoning-model",
        default=None,
        help="Alias for --runner-model.",
    )
    parser.add_argument(
        "--version-model",
        default=None,
        help="Override the version-selection model name.",
    )
    parser.add_argument(
        "--repair-model",
        default=None,
        help="Override the repair-stage model name.",
    )
    parser.add_argument(
        "--adjudication-model",
        default=None,
        help="Override the adjudication/cleanup model name.",
    )
    parser.add_argument(
        "--benchmark-source",
        choices=["all-gists", "dockerized-gists", "competition-run"],
        default=None,
        help=(
            "Select which Gistable case collection to execute. "
            "`competition-run` uses all-gists filtered to gist ids found in official result CSVs."
        ),
    )
    parser.add_argument(
        "--competition-csv",
        action="append",
        default=[],
        help=(
            "Additional CSV path containing official competition gist ids "
            "(repeatable, merged with APDR_COMPETITION_RESULT_CSVS/defaults)."
        ),
    )
    parser.add_argument(
        "--competition-filter-file",
        default=None,
        help=(
            "Path to a repo-tracked competition gist-id filter file. "
            "When --benchmark-source competition-run is used, APDR falls back to this file if CSVs are unavailable."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check Docker, Ollama, models, and dataset readiness.")
    doctor.add_argument("--ref", default=None)

    smoke = subparsers.add_parser("smoke", help="Run the beginner-friendly smoke benchmark flow.")
    smoke.add_argument("--ref", default=None)
    smoke.add_argument("--run-id", default=None)
    smoke.add_argument("--jobs", type=int, default=1)

    full_easy = subparsers.add_parser("full", help="Run the full benchmark flow.")
    full_easy.add_argument("--ref", default=None)
    full_easy.add_argument("--run-id", default=None)
    full_easy.add_argument("--jobs", type=int, default=1)

    solve_easy = subparsers.add_parser("solve", help="Resolve dependencies for a local Python project.")
    solve_easy.add_argument("--path", required=True)
    solve_easy.add_argument("--validation-command", default=None)
    solve_easy.add_argument("--run-id", default=None)

    subparsers.add_parser("ui", help="Launch the interactive terminal UI.")

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)

    fetch = benchmark_sub.add_parser("fetch-gistable")
    fetch.add_argument("--ref", default=None)

    make_subsets = benchmark_sub.add_parser("make-subsets")
    make_subsets.add_argument("--ref", default=None)

    save_comp_filter = benchmark_sub.add_parser("save-competition-filter")
    save_comp_filter.add_argument("--ref", default=None)

    segment = benchmark_sub.add_parser("segment")
    segment.add_argument("--subset", default="smoke30")
    segment.add_argument("--ref", default=None)
    segment.add_argument("--run-id", default=None)
    segment.add_argument("--jobs", type=int, default=1)

    full_run = benchmark_sub.add_parser("full")
    full_run.add_argument("--ref", default=None)
    full_run.add_argument("--run-id", default=None)
    full_run.add_argument("--jobs", type=int, default=1)

    run = benchmark_sub.add_parser("run")
    run_group = run.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--subset")
    run_group.add_argument("--full", action="store_true")
    run.add_argument("--ref", default=None)
    run.add_argument("--run-id", default=None)
    run.add_argument("--jobs", type=int, default=1)

    case = subparsers.add_parser("case")
    case_sub = case.add_subparsers(dest="case_command", required=True)
    case_run = case_sub.add_parser("run")
    case_run.add_argument("--case-id", required=True)
    case_run.add_argument("--ref", default=None)
    case_run.add_argument("--run-id", default=None)

    project = subparsers.add_parser("project")
    project_sub = project.add_subparsers(dest="project_command", required=True)
    solve = project_sub.add_parser("solve")
    solve.add_argument("--path", required=True)
    solve.add_argument("--validation-command", default=None)
    solve.add_argument("--run-id", default=None)

    report = subparsers.add_parser("report")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    summarize = report_sub.add_parser("summarize")
    summarize.add_argument("--run-id", required=True)
    failures = report_sub.add_parser("failures")
    failures.add_argument("--run-id", required=True)
    failures.add_argument("--category", default=None)
    failures.add_argument("--limit", type=int, default=10)
    modules = report_sub.add_parser("modules")
    modules.add_argument("--run-id", required=True)
    modules.add_argument("--top", type=int, default=15)
    modules.add_argument("--ref", default=None)
    modules.add_argument("--grouping", choices=["canonical", "raw"], default="canonical")
    modules.add_argument(
        "--paper-compatible",
        action="store_true",
        help="Build the module table from the paper-style hard subset in all-gists (initial-eval=ImportError).",
    )
    trace = report_sub.add_parser("trace")
    trace.add_argument("--run-id", required=True)
    trace.add_argument("--case-id", default=None)
    trace.add_argument("--tail", type=int, default=0, help="Show only the last N lines of the trace log.")
    timeline = report_sub.add_parser("timeline")
    timeline.add_argument("--run-id", required=True)

    return parser


def ensure_smoke_subset(
    settings: Settings,
    ref: str | None,
    subset_name: str = "smoke30",
    *,
    notify: bool = True,
) -> Path:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)
    if subset_name == "smoke30":
        smoke = build_smoke30(dataset, ref)
        subset_path = dataset.save_subset(subset_name, smoke, ref)
        if notify:
            _notify_path("Subset written", subset_path)
        return subset_path

    subset_path = dataset.dataset_root(ref) / "subsets" / f"{subset_name}.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset not found: {subset_name}")
    return subset_path


def _official_lookup_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def load_official_csv_lookup(settings: Settings) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for csv_path in settings.competition_result_csvs:
        if not csv_path.exists():
            continue
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    case_ids = GistableDataset._extract_case_ids_from_row({key: str(value or "") for key, value in row.items()})
                    if not case_ids:
                        continue
                    for case_id in case_ids:
                        entry = lookup.setdefault(
                            case_id,
                            {
                                "official_result": "",
                                "official_passed": "",
                                "official_duration": "",
                                "official_python_modules": "",
                                "official_file": "",
                                "_sources": set(),
                            },
                        )
                        entry["official_result"] = entry["official_result"] or _official_lookup_value(row, "result")
                        entry["official_passed"] = entry["official_passed"] or _official_lookup_value(row, "passed")
                        entry["official_duration"] = entry["official_duration"] or _official_lookup_value(row, "duration")
                        entry["official_python_modules"] = entry["official_python_modules"] or _official_lookup_value(
                            row,
                            "python_modules",
                            "modules",
                        )
                        entry["official_file"] = entry["official_file"] or _official_lookup_value(row, "file")
                        entry["_sources"].add(csv_path.name)
        except OSError:
            continue

    normalized: dict[str, dict[str, str]] = {}
    for case_id, payload in lookup.items():
        sources = payload.get("_sources", set())
        normalized[case_id] = {
            "official_result": str(payload.get("official_result", "")),
            "official_passed": str(payload.get("official_passed", "")),
            "official_duration": str(payload.get("official_duration", "")),
            "official_python_modules": str(payload.get("official_python_modules", "")),
            "official_file": str(payload.get("official_file", "")),
            "official_csv_sources": ";".join(sorted(str(source) for source in sources)),
        }
    return normalized


def build_runtime_comparison_row(
    result: dict[str, object],
    official_lookup: dict[str, dict[str, str]],
    *,
    case_number: int,
) -> dict[str, object]:
    case_id = str(result.get("case_id", "") or "")
    success = bool(result.get("success", False))
    official = official_lookup.get(case_id, {})
    return {
        "case_number": case_number,
        "case_id": case_id,
        "run_result": "success" if success else "failure",
        "run_success": success,
        "run_final_error_category": str(
            result.get("final_error_category", "Success" if success else "UnknownError") or ""
        ),
        "run_attempts": int(result.get("attempts", 0) or 0),
        "run_wall_clock_seconds": float(result.get("wall_clock_seconds", 0.0) or 0.0),
        "run_started_at": str(result.get("started_at", "") or ""),
        "run_finished_at": str(result.get("finished_at", "") or ""),
        "official_in_csv": bool(official),
        "official_result": official.get("official_result", ""),
        "official_passed": official.get("official_passed", ""),
        "official_duration": official.get("official_duration", ""),
        "official_python_modules": official.get("official_python_modules", ""),
        "official_file": official.get("official_file", ""),
        "official_csv_sources": official.get("official_csv_sources", ""),
    }


def _comparison_markdown_cell(value: object) -> str:
    rendered = str(value)
    return rendered.replace("|", "\\|")


def _parse_official_passed_flag(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"true", "t", "yes", "y", "pass", "passed"}:
        return True
    if text in {"false", "f", "no", "n", "fail", "failed"}:
        return False
    try:
        return float(text) > 0
    except ValueError:
        return None


def _normalize_result_label(value: object) -> str:
    return str(value or "").strip().lower()


def _run_result_label_for_official(row: dict[str, object]) -> str:
    if bool(row.get("run_success", False)):
        return "otherpass"
    category = _normalize_result_label(row.get("run_final_error_category", ""))
    if category in {"modulenotfounderror", "modulenotfound"}:
        return "modulenotfound"
    if category in {
        "importerror",
        "syntaxerror",
        "nameerror",
        "typeerror",
        "attributeerror",
    }:
        return category
    return "otherfailure"


def gist_match_row_from_runtime_row(row: dict[str, object]) -> dict[str, object]:
    run_success = bool(row.get("run_success", False))
    official_passed = _parse_official_passed_flag(row.get("official_passed", ""))
    if official_passed is None:
        matches: str | bool = ""
    else:
        matches = run_success == official_passed
    return {"gistid": str(row.get("case_id", "") or ""), "matches": matches}


def gist_match_detailed_row_from_runtime_row(row: dict[str, object]) -> dict[str, object]:
    run_success = bool(row.get("run_success", False))
    official_passed = _parse_official_passed_flag(row.get("official_passed", ""))
    if official_passed is None:
        matches_passed: str | bool = ""
    else:
        matches_passed = run_success == official_passed

    run_official_result = _run_result_label_for_official(row)
    official_result = _normalize_result_label(row.get("official_result", ""))
    matches_official_result: str | bool = ""
    if official_result:
        matches_official_result = run_official_result == official_result

    return {
        "gistid": str(row.get("case_id", "") or ""),
        "matches_passed": matches_passed,
        "matches_official_result": matches_official_result,
        "run_official_result": run_official_result,
        "official_result": str(row.get("official_result", "") or ""),
        "run_success": run_success,
        "official_passed": str(row.get("official_passed", "") or ""),
        "run_final_error_category": str(row.get("run_final_error_category", "") or ""),
    }


def _write_runtime_comparison_tables(
    csv_path: Path,
    markdown_path: Path,
    rows: list[dict[str, object]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(column.replace("_", " ") for column in RUNTIME_COMPARISON_MD_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in RUNTIME_COMPARISON_MD_COLUMNS) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(_comparison_markdown_cell(row.get(column, "")) for column in RUNTIME_COMPARISON_MD_COLUMNS) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_runtime_comparison_row(
    csv_path: Path,
    markdown_path: Path,
    row: dict[str, object],
) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COMPARISON_COLUMNS)
        writer.writerow(row)
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "| "
            + " | ".join(_comparison_markdown_cell(row.get(column, "")) for column in RUNTIME_COMPARISON_MD_COLUMNS)
            + " |\n"
        )


def _write_gist_match_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GIST_MATCH_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(gist_match_row_from_runtime_row(row))


def _write_gist_match_detailed_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GIST_MATCH_DETAILED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(gist_match_detailed_row_from_runtime_row(row))


def _append_gist_match_row(path: Path, runtime_row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GIST_MATCH_COLUMNS)
        writer.writerow(gist_match_row_from_runtime_row(runtime_row))


def _append_gist_match_detailed_row(path: Path, runtime_row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GIST_MATCH_DETAILED_COLUMNS)
        writer.writerow(gist_match_detailed_row_from_runtime_row(runtime_row))


def load_existing_case_results(run_dir: Path, case_ids: list[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case_id in case_ids:
        result_path = run_dir / case_id / "result.json"
        if not result_path.exists():
            continue
        try:
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return results


def load_all_run_results(run_dir: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for result_path in sorted(run_dir.glob("*/result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            results.append(payload)
    return results


def failed_case_ids_for_run(run_dir: Path) -> list[str]:
    failed_case_ids: list[str] = []
    for result in load_all_run_results(run_dir):
        case_id = str(result.get("case_id", "") or "")
        success = bool(result.get("success", False))
        if case_id and not success:
            failed_case_ids.append(case_id)
    return failed_case_ids


def resolve_trace_path(settings: Settings, run_id: str, case_id: str | None = None) -> Path:
    run_dir = settings.artifacts_dir / run_id
    return (run_dir / case_id / "llm-trace.log") if case_id else (run_dir / "llm-trace.log")


def collect_doctor_report(settings: Settings, ref: str | None = None) -> dict[str, object]:
    dataset = GistableDataset(settings)
    dataset_root = dataset.dataset_root(ref)
    marker = dataset_root / ".fetch-complete"
    checks: list[dict[str, str]] = []

    def add_check(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    python_detail = sys.executable
    add_check("python", "ok", python_detail)

    docker_path = shutil.which("docker")
    add_check("docker_cli", "ok" if docker_path else "missing", docker_path or "docker not found on PATH")

    pyego_entrypoint = settings.pyego_root / "PyEGo.py"
    add_check(
        "pyego_repo",
        "ok" if pyego_entrypoint.exists() else "warning",
        str(pyego_entrypoint) if pyego_entrypoint.exists() else f"missing: {pyego_entrypoint}",
    )
    if settings.resolver == "pyego":
        pyego_runtime_ok, pyego_runtime_detail = validate_pyego_runtime(settings)
        add_check("pyego_runtime", "ok" if pyego_runtime_ok else "warning", pyego_runtime_detail)
    else:
        add_check("pyego_runtime", "ok", f"resolver={settings.resolver}; not required")

    readpye_entrypoint = settings.readpye_root / "run.py"
    add_check(
        "readpye_repo",
        "ok" if readpye_entrypoint.exists() else "warning",
        str(readpye_entrypoint) if readpye_entrypoint.exists() else f"missing: {readpye_entrypoint}",
    )
    if settings.resolver == "readpye":
        add_check(
            "readpye_langdir",
            "ok" if settings.readpye_language_dir else "warning",
            settings.readpye_language_dir or "APDR_READPYE_LANGDIR not configured",
        )

    ollama_path = shutil.which("ollama")
    add_check("ollama_cli", "ok" if ollama_path else "missing", ollama_path or "ollama not found on PATH")

    ollama_models: list[str] = []
    try:
        with urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ollama_models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        add_check("ollama_server", "ok", settings.ollama_base_url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        add_check("ollama_server", "warning", f"{settings.ollama_base_url} unavailable: {exc}")

    seen_models: set[str] = set()
    for model_name in settings.active_stage_models().values():
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        status = "ok" if model_name in ollama_models else "missing"
        detail = "installed" if status == "ok" else "pull this model before running benchmarks"
        add_check(f"model:{model_name}", status, detail)

    dataset_status = "ok" if marker.exists() else "warning"
    dataset_detail = str(dataset_root) if marker.exists() else "benchmark dataset not fetched yet"
    add_check("gistable_dataset", dataset_status, dataset_detail)

    overall_status = "ok" if all(check["status"] == "ok" for check in checks) else "warning"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "resolver": settings.resolver,
        "preset": settings.preset,
        "benchmark_source": settings.benchmark_case_source,
        "prompt_profile": settings.prompt_profile,
        "research_bundle": settings.research_bundle,
        "research_features": list(settings.research_features),
        "model_profile": settings.model_profile,
        "use_moe": settings.use_moe,
        "use_rag": settings.use_rag,
        "use_langchain": settings.use_langchain,
        "checks": checks,
    }


def doctor_command(settings: Settings, ref: str | None) -> int:
    report = collect_doctor_report(settings, ref)
    artifact_path = settings.project_root / "artifacts" / "doctor-latest.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] preset: {settings.preset}")
    print(f"[INFO] resolver: {settings.resolver}")
    print(f"[INFO] prompt_profile: {settings.prompt_profile}")
    print(f"[INFO] benchmark_source: {settings.benchmark_case_source}")
    print(f"[INFO] research_bundle: {settings.research_bundle}")
    print(f"[INFO] research_features: {', '.join(settings.research_features) or 'none'}")
    print(f"[INFO] model_profile: {settings.model_profile}")
    print(f"[INFO] moe: {'enabled' if settings.use_moe else 'disabled'}")
    print(f"[INFO] rag: {'enabled' if settings.use_rag else 'disabled'}")
    print(f"[INFO] rag_mode: {settings.rag_mode}")
    print(f"[INFO] structured_prompting: {'enabled' if settings.structured_prompting else 'disabled'}")
    print(f"[INFO] langchain: {'enabled' if settings.use_langchain else 'disabled'}")
    print(f"[INFO] extraction_model: {settings.extraction_model}")
    print(f"[INFO] runner_model: {settings.reasoning_model}")
    print(f"[INFO] version_model: {settings.version_model}")
    print(f"[INFO] repair_model: {settings.repair_model}")
    print(f"[INFO] adjudication_model: {settings.adjudication_model}")
    print(f"[INFO] llm_cache: {'disabled' if settings.disable_llm_cache else 'enabled'}")
    print(f"[INFO] pyego_root: {settings.pyego_root}")
    print(f"[INFO] readpye_root: {settings.readpye_root}")
    if settings.readpye_language_dir:
        print(f"[INFO] readpye_language_dir: {settings.readpye_language_dir}")
    for check in report["checks"]:
        status = check["status"].upper()
        print(f"[{status}] {check['name']}: {check['detail']}")
    _notify_path("Doctor report written", artifact_path)
    return 0


def run_benchmark(
    settings: Settings,
    ref: str | None,
    subset: str | None,
    full: bool,
    run_id: str | None,
    jobs: int = 1,
    observer: BenchmarkObserver | None = None,
    *,
    notify_paths: bool = True,
    fresh_run: bool = False,
) -> int:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)
    if subset:
        case_ids = dataset.load_subset(subset, ref)
    else:
        case_ids = dataset.valid_case_ids(ref, case_source=settings.benchmark_case_source)
    return run_case_batch(
        settings,
        ref,
        case_ids,
        run_id,
        jobs=jobs,
        observer=observer,
        notify_paths=notify_paths,
        fresh_run=fresh_run,
        target_label=subset or ("full" if full else "benchmark"),
    )


def run_case_batch(
    settings: Settings,
    ref: str | None,
    case_ids: list[str],
    run_id: str | None,
    *,
    jobs: int = 1,
    observer: BenchmarkObserver | None = None,
    notify_paths: bool = True,
    fresh_run: bool = False,
    target_label: str = "benchmark",
) -> int:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)

    active_run_id = run_id or uuid4().hex[:12]
    run_dir = settings.artifacts_dir / active_run_id
    if fresh_run and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if settings.resolver == "pyego":
        pyego_runtime_ok, pyego_runtime_detail = validate_pyego_runtime(settings)
        if not pyego_runtime_ok:
            message = f"PyEGo preflight failed: {pyego_runtime_detail}"
            warnings_path = run_dir / "warnings.log"
            warnings_path.parent.mkdir(parents=True, exist_ok=True)
            warnings_path.write_text(f"{message}\n", encoding="utf-8")
            print(f"[ERROR] {message}", file=sys.stderr)
            _notify_path("Warnings log written", warnings_path)
            return 2

    restored_state = load_run_state(run_dir)
    started_at = time.monotonic()
    restored_elapsed_seconds = float(restored_state.get("elapsed_seconds", 0.0) or 0.0)
    warnings_path = run_dir / "warnings.log"
    summary_defaults = summary_defaults_from_settings(settings)

    completed_case_ids = [case_id for case_id in case_ids if (run_dir / case_id / "result.json").exists()]
    completed_case_id_set = set(completed_case_ids)
    pending_case_ids = [case_id for case_id in case_ids if case_id not in completed_case_id_set]
    completed_results = load_existing_case_results(run_dir, completed_case_ids)
    official_lookup = load_official_csv_lookup(settings)
    runtime_table_csv_path = run_dir / "run-vs-csv.csv"
    runtime_table_md_path = run_dir / "run-vs-csv.md"
    gist_match_csv_path = run_dir / "gistid-matches.csv"
    gist_match_detailed_csv_path = run_dir / "gistid-matches-detailed.csv"
    runtime_rows = [
        build_runtime_comparison_row(result, official_lookup, case_number=index)
        for index, result in enumerate(completed_results, start=1)
    ]
    _write_runtime_comparison_tables(runtime_table_csv_path, runtime_table_md_path, runtime_rows)
    _write_gist_match_table(gist_match_csv_path, runtime_rows)
    _write_gist_match_detailed_table(gist_match_detailed_csv_path, runtime_rows)
    completed_successes = sum(1 for result in completed_results if bool(result.get("success", False)))
    completed_failures = len(completed_results) - completed_successes
    if not case_ids:
        summary = summarize_run(
            run_dir,
            total_elapsed_seconds=restored_elapsed_seconds + (time.monotonic() - started_at),
            run_defaults=summary_defaults,
        )
        write_run_state(
            run_dir,
            {
                "run_id": active_run_id,
                "status": "empty",
                "resolver": settings.resolver,
                "preset": settings.preset,
                "benchmark_source": settings.benchmark_case_source,
                "prompt_profile": settings.prompt_profile,
                "research_bundle": settings.research_bundle,
                "research_features": list(settings.research_features),
                "jobs": jobs,
                "target": target_label,
                "total": 0,
                "completed": 0,
                "successes": 0,
                "failures": 0,
                "elapsed_seconds": summary.total_wall_clock_time,
                "started_at": str(restored_state.get("started_at", "") or datetime.now(timezone.utc).isoformat()),
                "last_updated_at": datetime.now(timezone.utc).isoformat(),
                "current_cases": [],
                "last_case_id": "",
                "last_status": "no_cases_selected",
                "summary_path": str(run_dir / "summary.json"),
                "warnings_path": "",
                "stop_requested": False,
                "last_error": "No benchmark cases were selected for this run.",
            },
        )
        if notify_paths:
            _notify_path("Summary written", run_dir / "summary.json")
            _notify_path("Run-vs-CSV table written", run_dir / "run-vs-csv.csv")
            _notify_path("Gist match table written", run_dir / "gistid-matches.csv")
            _notify_path("Gist detailed match table written", run_dir / "gistid-matches-detailed.csv")
        print("[ERROR] no benchmark cases were selected for this run", file=sys.stderr)
        return 2

    inner_observer = observer or BenchmarkProgress(active_run_id, len(case_ids), completed=len(completed_case_ids))
    progress_observer = PersistentBenchmarkObserver(inner_observer, run_dir, restored_state)
    progress_observer.start(
        run_id=active_run_id,
        total=len(case_ids),
        completed=len(completed_case_ids),
        successes=completed_successes,
        failures=completed_failures,
        resolver=settings.resolver,
        preset=settings.preset,
        prompt_profile=settings.prompt_profile,
        model_summary=format_model_summary(settings),
        research_bundle=settings.research_bundle,
        research_features=settings.research_features,
        benchmark_source=settings.benchmark_case_source,
        jobs=jobs,
        target=target_label,
        artifacts_dir=run_dir,
    )

    def process_case(case_id: str) -> dict[str, object]:
        case = dataset.load_case(case_id, ref, case_source=settings.benchmark_case_source)
        workflow = ResolutionWorkflow(settings)
        state = workflow.initial_state_for_case(case, run_id=active_run_id)
        final_state = workflow.run(state)
        return dict(final_state["final_result"])

    try:
        with redirect_runtime_warnings(warnings_path):
            if jobs <= 1:
                for case_id in pending_case_ids:
                    if progress_observer.stop_requested():
                        break
                    progress_observer.case_started(case_id)
                    result = process_case(case_id)
                    progress_observer.advance(result)
                    row = build_runtime_comparison_row(
                        result,
                        official_lookup,
                        case_number=len(runtime_rows) + 1,
                    )
                    runtime_rows.append(row)
                    _append_runtime_comparison_row(runtime_table_csv_path, runtime_table_md_path, row)
                    _append_gist_match_row(gist_match_csv_path, row)
                    _append_gist_match_detailed_row(gist_match_detailed_csv_path, row)
            else:
                with ThreadPoolExecutor(max_workers=jobs) as executor:
                    pending_iterator = iter(pending_case_ids)
                    futures: dict[object, str] = {}
                    for _ in range(min(jobs, len(pending_case_ids))):
                        if progress_observer.stop_requested():
                            break
                        case_id = next(pending_iterator, None)
                        if case_id is None:
                            break
                        progress_observer.case_started(case_id)
                        futures[executor.submit(process_case, case_id)] = case_id
                    while futures:
                        future = next(as_completed(futures))
                        case_id = futures.pop(future)
                        result = future.result()
                        progress_observer.advance(result)
                        row = build_runtime_comparison_row(
                            result,
                            official_lookup,
                            case_number=len(runtime_rows) + 1,
                        )
                        runtime_rows.append(row)
                        _append_runtime_comparison_row(runtime_table_csv_path, runtime_table_md_path, row)
                        _append_gist_match_row(gist_match_csv_path, row)
                        _append_gist_match_detailed_row(gist_match_detailed_csv_path, row)
                        if progress_observer.stop_requested():
                            continue
                        next_case_id = next(pending_iterator, None)
                        if next_case_id is not None:
                            progress_observer.case_started(next_case_id)
                            futures[executor.submit(process_case, next_case_id)] = next_case_id
    except BaseException as exc:
        progress_observer.abort(exc)
        raise

    summary = summarize_run(
        run_dir,
        total_elapsed_seconds=restored_elapsed_seconds + (time.monotonic() - started_at),
        run_defaults=summary_defaults,
    )
    non_empty_warnings_path = warnings_path if warnings_path.exists() and warnings_path.stat().st_size > 0 else None
    completed_total = len([case_id for case_id in case_ids if (run_dir / case_id / "result.json").exists()])
    final_status = "completed" if completed_total >= len(case_ids) else "paused"
    progress_observer.finish(
        summary_path=run_dir / "summary.json",
        warnings_path=non_empty_warnings_path,
        status=final_status,
    )
    if notify_paths:
        _notify_path("Summary written", run_dir / "summary.json")
        _notify_path("Run-vs-CSV table written", run_dir / "run-vs-csv.csv")
        _notify_path("Gist match table written", run_dir / "gistid-matches.csv")
        _notify_path("Gist detailed match table written", run_dir / "gistid-matches-detailed.csv")
        if non_empty_warnings_path is not None:
            _notify_path("Warnings written", warnings_path)
        if settings.trace_llm:
            _notify_path("LLM trace written", run_dir / "llm-trace.log")
    return 0


def run_failed_cases(
    settings: Settings,
    source_run_id: str,
    ref: str | None,
    run_id: str | None,
    *,
    jobs: int = 1,
    observer: BenchmarkObserver | None = None,
    notify_paths: bool = True,
    fresh_run: bool = False,
) -> int:
    source_run_dir = settings.artifacts_dir / source_run_id
    failed_case_ids = failed_case_ids_for_run(source_run_dir)
    if not failed_case_ids:
        print(f"[INFO] no failed cases found in run {source_run_id}")
        return 0
    return run_case_batch(
        settings,
        ref,
        failed_case_ids,
        run_id,
        jobs=jobs,
        observer=observer,
        notify_paths=notify_paths,
        fresh_run=fresh_run,
        target_label=f"failed:{source_run_id}",
    )


def run_case(settings: Settings, case_id: str, ref: str | None, run_id: str | None, fresh_run: bool = False) -> int:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)
    if fresh_run and run_id:
        case_run_dir = settings.artifacts_dir / run_id
        if case_run_dir.exists():
            shutil.rmtree(case_run_dir)
    case = dataset.load_case(case_id, ref, case_source=settings.benchmark_case_source)
    workflow = ResolutionWorkflow(settings)
    state = workflow.initial_state_for_case(case, run_id=run_id)
    final_state = workflow.run(state)
    _notify_path("Result written", Path(final_state["artifact_dir"]) / "result.json")
    if settings.trace_llm:
        _notify_path("LLM trace written", Path(final_state["artifact_dir"]) / "llm-trace.log")
    return 0


def run_project(
    settings: Settings,
    project_path: str,
    validation_command: str | None,
    run_id: str | None,
    fresh_run: bool = False,
) -> int:
    if fresh_run and run_id:
        project_run_dir = settings.artifacts_dir / run_id
        if project_run_dir.exists():
            shutil.rmtree(project_run_dir)
    workflow = ResolutionWorkflow(settings)
    state = workflow.initial_state_for_project(Path(project_path).resolve(), validation_command, run_id=run_id)
    final_state = workflow.run(state)
    _notify_path("Result written", Path(final_state["artifact_dir"]) / "result.json")
    if settings.trace_llm:
        _notify_path("LLM trace written", Path(final_state["artifact_dir"]) / "llm-trace.log")
    return 0


def summarize_command(settings: Settings, run_id: str) -> int:
    summary = summarize_run(
        settings.artifacts_dir / run_id,
        run_defaults=load_run_state(settings.artifacts_dir / run_id),
    )
    _notify_path("Summary written", settings.artifacts_dir / run_id / "summary.json")
    return 0


def failures_command(settings: Settings, run_id: str, category: str | None, limit: int) -> int:
    analysis = analyze_failures(settings.artifacts_dir / run_id, limit=limit, category=category)
    output_name = "failure-analysis.json" if not category else f"failure-analysis-{category}.json"
    _notify_path("Failure analysis written", settings.artifacts_dir / run_id / output_name)
    return 0


def modules_command(
    settings: Settings,
    run_id: str,
    top: int,
    ref: str | None,
    grouping: str,
    paper_compatible: bool = False,
) -> int:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)
    build_module_success_table(
        settings.artifacts_dir / run_id,
        dataset,
        ref=ref,
        top_n=top,
        grouping=grouping,
        paper_compatible=paper_compatible,
    )
    suffix_parts: list[str] = []
    if paper_compatible:
        suffix_parts.append("paper")
    if grouping == "raw":
        suffix_parts.append("raw")
    suffix = "" if not suffix_parts else "-" + "-".join(suffix_parts)
    _notify_path("Module success table written", settings.artifacts_dir / run_id / f"module-success{suffix}.md")
    return 0


def timeline_command(settings: Settings, run_id: str) -> int:
    build_timeline_view(settings.artifacts_dir / run_id)
    _notify_path("Timeline written", settings.artifacts_dir / run_id / "timeline.md")
    return 0


def trace_command(settings: Settings, run_id: str, case_id: str | None, tail: int) -> int:
    trace_path = resolve_trace_path(settings, run_id, case_id)
    if not trace_path.exists():
        print(
            f"Trace log not found at {trace_path}. Run the command with --trace-llm first.",
            file=sys.stderr,
        )
        return 1
    contents = trace_path.read_text(encoding="utf-8")
    if tail > 0:
        lines = contents.splitlines()
        contents = "\n".join(lines[-tail:]) + ("\n" if lines else "")
    print(contents, end="" if contents.endswith("\n") or not contents else "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings(
        "ignore",
        message="Importing debug from langchain root module is no longer supported.*",
        category=UserWarning,
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env(
        resolver_override=args.resolver,
        preset_override=args.preset,
        prompt_profile_override=args.prompt_profile,
        model_profile_override=args.model_profile,
        use_moe_override=args.moe,
        use_rag_override=args.rag,
        use_langchain_override=args.langchain,
        extraction_model_override=args.extraction_model,
        runner_model_override=args.runner_model or args.reasoning_model,
        reasoning_model_override=args.reasoning_model,
        version_model_override=args.version_model,
        repair_model_override=args.repair_model,
        adjudication_model_override=args.adjudication_model,
        disable_llm_cache_override=args.no_llm_cache or args.fresh_run,
        research_bundle_override=args.research_bundle,
        research_feature_overrides=args.research_feature,
        research_feature_disable_overrides=args.no_research_feature,
        benchmark_case_source_override=args.benchmark_source,
        competition_result_csvs_override=args.competition_csv,
        competition_case_ids_file_override=args.competition_filter_file,
    )
    if args.trace_llm:
        settings.trace_llm = True
    if settings.preset in {"research", "experimental"} and settings.resolver != "apdr":
        parser.error("The research and experimental presets are only supported with the apdr resolver.")
    if (
        args.research_bundle is not None
        or args.research_feature
        or args.no_research_feature
    ) and settings.preset != "research":
        parser.error("Research bundles and features are only supported with --preset research.")

    if args.command == "doctor":
        return doctor_command(settings, args.ref)

    if args.command == "smoke":
        ensure_smoke_subset(settings, args.ref, "smoke30")
        return run_benchmark(settings, args.ref, "smoke30", False, args.run_id, args.jobs, fresh_run=args.fresh_run)

    if args.command == "full":
        return run_benchmark(settings, args.ref, None, True, args.run_id, args.jobs, fresh_run=args.fresh_run)

    if args.command == "solve":
        return run_project(settings, args.path, args.validation_command, args.run_id, fresh_run=args.fresh_run)

    if args.command == "ui":
        return launch_terminal_ui(
            settings,
            doctor_command=doctor_command,
            run_benchmark=run_benchmark,
            run_failed_cases=run_failed_cases,
            run_project=run_project,
            summarize_command=summarize_command,
            failures_command=failures_command,
            modules_command=modules_command,
            timeline_command=timeline_command,
            ensure_smoke_subset=ensure_smoke_subset,
        )

    if args.command == "benchmark":
        dataset = GistableDataset(settings)
        if args.benchmark_command == "fetch-gistable":
            _notify_path("Benchmark dataset ready", dataset.fetch(args.ref))
            return 0
        if args.benchmark_command == "make-subsets":
            ensure_smoke_subset(settings, args.ref, "smoke30")
            return 0
        if args.benchmark_command == "save-competition-filter":
            _notify_path("Benchmark dataset ready", dataset.fetch(args.ref))
            filter_path, case_count = dataset.save_competition_case_ids_file(args.ref)
            if case_count <= 0:
                print(
                    "[ERROR] no competition gist ids found from configured CSV sources; filter file was not updated",
                    file=sys.stderr,
                )
                return 2
            _notify_path("Competition filter written", filter_path)
            print(f"[INFO] competition filter case count: {case_count}", file=sys.stderr)
            return 0
        if args.benchmark_command == "segment":
            ensure_smoke_subset(settings, args.ref, args.subset)
            return run_benchmark(
                settings,
                args.ref,
                args.subset,
                False,
                args.run_id,
                args.jobs,
                fresh_run=args.fresh_run,
            )
        if args.benchmark_command == "full":
            _notify_path("Benchmark dataset ready", dataset.fetch(args.ref))
            return run_benchmark(settings, args.ref, None, True, args.run_id, args.jobs, fresh_run=args.fresh_run)
        if args.benchmark_command == "run":
            if args.subset == "smoke30":
                ensure_smoke_subset(settings, args.ref, "smoke30")
            return run_benchmark(
                settings,
                args.ref,
                args.subset,
                args.full,
                args.run_id,
                args.jobs,
                fresh_run=args.fresh_run,
            )

    if args.command == "case" and args.case_command == "run":
        return run_case(settings, args.case_id, args.ref, args.run_id, fresh_run=args.fresh_run)

    if args.command == "project" and args.project_command == "solve":
        return run_project(settings, args.path, args.validation_command, args.run_id, fresh_run=args.fresh_run)

    if args.command == "report" and args.report_command == "summarize":
        return summarize_command(settings, args.run_id)
    if args.command == "report" and args.report_command == "failures":
        return failures_command(settings, args.run_id, args.category, args.limit)
    if args.command == "report" and args.report_command == "modules":
        return modules_command(settings, args.run_id, args.top, args.ref, args.grouping, args.paper_compatible)
    if args.command == "report" and args.report_command == "timeline":
        return timeline_command(settings, args.run_id)
    if args.command == "report" and args.report_command == "trace":
        return trace_command(settings, args.run_id, args.case_id, args.tail)

    parser.error("Unsupported command.")
    return 2
