from __future__ import annotations

import argparse
import contextlib
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
from uuid import uuid4

from agentic_python_dependency.benchmark.gistable import GistableDataset
from agentic_python_dependency.benchmark.subsets import build_smoke30
from agentic_python_dependency.config import Settings
from agentic_python_dependency.graph import ResolutionWorkflow
from agentic_python_dependency.reporting import analyze_failures, build_module_success_table, summarize_run


def _notify_path(label: str, path: Path) -> None:
    print(f"{label}: {path}", file=sys.stderr)


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
        self.started_at = time.monotonic()
        self._lock = threading.RLock()
        self._isatty = sys.stdout.isatty()
        self._refresh_interval = refresh_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _line(self) -> str:
        bar = format_progress_bar(self.completed, self.total)
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        return (
            f"Benchmark {self.run_id} {bar} "
            f"{self.completed}/{self.total} {percent:5.1f}% "
            f"elapsed {format_elapsed(time.monotonic() - self.started_at)}"
        )

    def render(self) -> None:
        with self._lock:
            line = self._line()
            if self._isatty:
                print(f"\r{line}", end="", file=sys.stdout, flush=True)
            else:
                print(line, file=sys.stdout, flush=True)

    def start(self) -> None:
        if not self._isatty or self._thread is not None:
            return

        def _refresh_loop() -> None:
            while not self._stop_event.wait(self._refresh_interval):
                self.render()

        self._thread = threading.Thread(target=_refresh_loop, name="benchmark-progress", daemon=True)
        self._thread.start()

    def advance(self, count: int = 1) -> None:
        with self._lock:
            self.completed = min(self.total, self.completed + count)
            self.render()

    def finish(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._refresh_interval + 0.1)
        with self._lock:
            self.completed = self.total
            line = self._line()
            if self._isatty:
                print(f"\r{line}", end="", file=sys.stdout, flush=True)
                print(file=sys.stdout, flush=True)
            else:
                print(line, file=sys.stdout, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apd",
        description="Agentic Python dependency resolver.",
        epilog=(
            "Beginner-friendly commands:\n"
            "  apd doctor\n"
            "  apd smoke --jobs 1\n"
            "  apd full --jobs 1\n"
            "  apd solve --path /path/to/repo\n\n"
            "Advanced commands:\n"
            "  apd benchmark segment --jobs 2\n"
            "  apd benchmark full --jobs 2\n"
            "  apd report summarize --run-id <run_id>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trace-llm",
        action="store_true",
        help="Write prompts and model outputs to llm-trace.log files during execution.",
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

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)

    fetch = benchmark_sub.add_parser("fetch-gistable")
    fetch.add_argument("--ref", default=None)

    make_subsets = benchmark_sub.add_parser("make-subsets")
    make_subsets.add_argument("--ref", default=None)

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

    return parser


def ensure_smoke_subset(settings: Settings, ref: str | None, subset_name: str = "smoke30") -> Path:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)
    if subset_name == "smoke30":
        smoke = build_smoke30(dataset, ref)
        subset_path = dataset.save_subset(subset_name, smoke, ref)
        _notify_path("Subset written", subset_path)
        return subset_path

    subset_path = dataset.dataset_root(ref) / "subsets" / f"{subset_name}.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset not found: {subset_name}")
    return subset_path


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

    for model_name in (settings.extraction_model, settings.reasoning_model):
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
        "checks": checks,
    }


def doctor_command(settings: Settings, ref: str | None) -> int:
    report = collect_doctor_report(settings, ref)
    artifact_path = settings.project_root / "artifacts" / "doctor-latest.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
) -> int:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)
    if subset:
        case_ids = dataset.load_subset(subset, ref)
    else:
        case_ids = dataset.valid_case_ids(ref)

    active_run_id = run_id or uuid4().hex[:12]
    run_dir = settings.artifacts_dir / active_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    warnings_path = run_dir / "warnings.log"

    completed_case_ids = [case_id for case_id in case_ids if (run_dir / case_id / "result.json").exists()]
    completed_case_id_set = set(completed_case_ids)
    pending_case_ids = [case_id for case_id in case_ids if case_id not in completed_case_id_set]
    progress = BenchmarkProgress(active_run_id, len(case_ids), completed=len(completed_case_ids))
    progress.render()
    progress.start()

    def process_case(case_id: str) -> None:
        case = dataset.load_case(case_id, ref)
        workflow = ResolutionWorkflow(settings)
        state = workflow.initial_state_for_case(case, run_id=active_run_id)
        workflow.run(state)

    with redirect_runtime_warnings(warnings_path):
        if jobs <= 1:
            for case_id in pending_case_ids:
                process_case(case_id)
                progress.advance()
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {executor.submit(process_case, case_id): case_id for case_id in pending_case_ids}
                for future in as_completed(futures):
                    future.result()
                    progress.advance()

    progress.finish()

    summary = summarize_run(run_dir, total_elapsed_seconds=time.monotonic() - started_at)
    _notify_path("Summary written", run_dir / "summary.json")
    if warnings_path.exists() and warnings_path.stat().st_size > 0:
        _notify_path("Warnings written", warnings_path)
    if settings.trace_llm:
        _notify_path("LLM trace written", run_dir / "llm-trace.log")
    return 0


def run_case(settings: Settings, case_id: str, ref: str | None, run_id: str | None) -> int:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)
    case = dataset.load_case(case_id, ref)
    workflow = ResolutionWorkflow(settings)
    state = workflow.initial_state_for_case(case, run_id=run_id)
    final_state = workflow.run(state)
    _notify_path("Result written", Path(final_state["artifact_dir"]) / "result.json")
    if settings.trace_llm:
        _notify_path("LLM trace written", Path(final_state["artifact_dir"]) / "llm-trace.log")
    return 0


def run_project(settings: Settings, project_path: str, validation_command: str | None, run_id: str | None) -> int:
    workflow = ResolutionWorkflow(settings)
    state = workflow.initial_state_for_project(Path(project_path).resolve(), validation_command, run_id=run_id)
    final_state = workflow.run(state)
    _notify_path("Result written", Path(final_state["artifact_dir"]) / "result.json")
    if settings.trace_llm:
        _notify_path("LLM trace written", Path(final_state["artifact_dir"]) / "llm-trace.log")
    return 0


def summarize_command(settings: Settings, run_id: str) -> int:
    summary = summarize_run(settings.artifacts_dir / run_id)
    _notify_path("Summary written", settings.artifacts_dir / run_id / "summary.json")
    return 0


def failures_command(settings: Settings, run_id: str, category: str | None, limit: int) -> int:
    analysis = analyze_failures(settings.artifacts_dir / run_id, limit=limit, category=category)
    output_name = "failure-analysis.json" if not category else f"failure-analysis-{category}.json"
    _notify_path("Failure analysis written", settings.artifacts_dir / run_id / output_name)
    return 0


def modules_command(settings: Settings, run_id: str, top: int, ref: str | None) -> int:
    dataset = GistableDataset(settings)
    dataset.fetch(ref)
    build_module_success_table(settings.artifacts_dir / run_id, dataset, ref=ref, top_n=top)
    _notify_path("Module success table written", settings.artifacts_dir / run_id / "module-success.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings(
        "ignore",
        message="Importing debug from langchain root module is no longer supported.*",
        category=UserWarning,
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.trace_llm:
        settings.trace_llm = True

    if args.command == "doctor":
        return doctor_command(settings, args.ref)

    if args.command == "smoke":
        ensure_smoke_subset(settings, args.ref, "smoke30")
        return run_benchmark(settings, args.ref, "smoke30", False, args.run_id, args.jobs)

    if args.command == "full":
        return run_benchmark(settings, args.ref, None, True, args.run_id, args.jobs)

    if args.command == "solve":
        return run_project(settings, args.path, args.validation_command, args.run_id)

    if args.command == "benchmark":
        dataset = GistableDataset(settings)
        if args.benchmark_command == "fetch-gistable":
            _notify_path("Benchmark dataset ready", dataset.fetch(args.ref))
            return 0
        if args.benchmark_command == "make-subsets":
            ensure_smoke_subset(settings, args.ref, "smoke30")
            return 0
        if args.benchmark_command == "segment":
            ensure_smoke_subset(settings, args.ref, args.subset)
            return run_benchmark(settings, args.ref, args.subset, False, args.run_id, args.jobs)
        if args.benchmark_command == "full":
            _notify_path("Benchmark dataset ready", dataset.fetch(args.ref))
            return run_benchmark(settings, args.ref, None, True, args.run_id, args.jobs)
        if args.benchmark_command == "run":
            if args.subset == "smoke30":
                ensure_smoke_subset(settings, args.ref, "smoke30")
            return run_benchmark(settings, args.ref, args.subset, args.full, args.run_id, args.jobs)

    if args.command == "case" and args.case_command == "run":
        return run_case(settings, args.case_id, args.ref, args.run_id)

    if args.command == "project" and args.project_command == "solve":
        return run_project(settings, args.path, args.validation_command, args.run_id)

    if args.command == "report" and args.report_command == "summarize":
        return summarize_command(settings, args.run_id)
    if args.command == "report" and args.report_command == "failures":
        return failures_command(settings, args.run_id, args.category, args.limit)
    if args.command == "report" and args.report_command == "modules":
        return modules_command(settings, args.run_id, args.top, args.ref)

    parser.error("Unsupported command.")
    return 2
