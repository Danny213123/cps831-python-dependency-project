from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apd", description="Agentic Python dependency resolver.")
    parser.add_argument(
        "--trace-llm",
        action="store_true",
        help="Write prompts and model outputs to llm-trace.log files during execution.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)

    fetch = benchmark_sub.add_parser("fetch-gistable")
    fetch.add_argument("--ref", default=None)

    make_subsets = benchmark_sub.add_parser("make-subsets")
    make_subsets.add_argument("--ref", default=None)

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

    pending_case_ids = [case_id for case_id in case_ids if not (run_dir / case_id / "result.json").exists()]

    def process_case(case_id: str) -> None:
        case = dataset.load_case(case_id, ref)
        workflow = ResolutionWorkflow(settings)
        state = workflow.initial_state_for_case(case, run_id=active_run_id)
        workflow.run(state)

    if jobs <= 1:
        for case_id in pending_case_ids:
            process_case(case_id)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(process_case, case_id): case_id for case_id in pending_case_ids}
            for future in as_completed(futures):
                future.result()

    summary = summarize_run(run_dir, total_elapsed_seconds=time.monotonic() - started_at)
    _notify_path("Summary written", run_dir / "summary.json")
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

    if args.command == "benchmark":
        dataset = GistableDataset(settings)
        if args.benchmark_command == "fetch-gistable":
            _notify_path("Benchmark dataset ready", dataset.fetch(args.ref))
            return 0
        if args.benchmark_command == "make-subsets":
            dataset.fetch(args.ref)
            smoke = build_smoke30(dataset, args.ref)
            subset_path = dataset.save_subset("smoke30", smoke, args.ref)
            _notify_path("Subset written", subset_path)
            return 0
        if args.benchmark_command == "run":
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
