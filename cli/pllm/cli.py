from __future__ import annotations

import argparse
from typing import Sequence

from cli.pllm.benchmark_data import (
    BenchmarkSource,
    breakdown_summary,
    rebuild_competition_filter,
    resolve_snippet_path,
)
from cli.pllm.core import RunConfig, doctor_passed, format_doctor_report, run_doctor, stream_executor


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        checks = run_doctor(base_url=args.base)
        print(format_doctor_report(checks))
        return 0 if doctor_passed(checks) else 1

    if args.command == "benchmark":
        if args.benchmark_command == "breakdown":
            print(breakdown_summary(active_source=args.source))
            return 0
        if args.benchmark_command == "rebuild-filter":
            path, matched, csv_total = rebuild_competition_filter()
            print(f"Filter written: {path}")
            print(f"CSV id count: {csv_total}")
            print(f"Matched all-gists ids: {matched}")
            return 0 if matched > 0 else 1
        return 1

    if args.command == "ui":
        from cli.pllm.terminal_ui import launch_terminal_ui

        return launch_terminal_ui(
            default_model=args.model,
            default_base=args.base,
            default_loop=args.loop,
            default_range=args.range,
            default_file=args.file,
            default_benchmark_source=args.benchmark_source,
        )

    resolved_file = args.file
    if args.case_id:
        snippet_path = resolve_snippet_path(args.case_id, source=args.benchmark_source)
        if snippet_path is None:
            print(
                f"Unable to find case id {args.case_id} in source {args.benchmark_source}. "
                "Check dataset files and case id."
            )
            return 1
        resolved_file = str(snippet_path)

    if not resolved_file:
        print("Either --file or --case-id is required for run.")
        return 2

    config = RunConfig(
        file=resolved_file,
        model=args.model,
        base=args.base,
        temp=args.temp,
        loop=args.loop,
        search_range=args.range,
        rag=args.rag,
        verbose=args.verbose,
    )

    if args.dashboard:
        from cli.pllm.terminal_ui import run_config_with_dashboard

        return_code, stats = run_config_with_dashboard(config)
    else:
        return_code, stats = stream_executor(config, line_callback=lambda line, _: print(line))

    print(
        f"Exit code {return_code}. "
        f"Elapsed {stats.elapsed_seconds:.1f}s. "
        f"Build ok/fail: {stats.build_successes}/{stats.build_failures}. "
        f"Lines: {stats.lines}."
    )
    return return_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pllm",
        description="PLLM command center and test_executor wrapper",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run tools/pllm/test_executor.py")
    _add_shared_runtime_args(run_parser, require_file=False)
    run_parser.add_argument(
        "--case-id",
        default="",
        help="Gistable case id to resolve from benchmark data",
    )
    run_parser.add_argument(
        "--benchmark-source",
        choices=_benchmark_source_choices(),
        default="all-gists",
        help="Benchmark source used when --case-id is provided",
    )
    run_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show live dashboard while streaming output",
    )

    ui_parser = subparsers.add_parser("ui", help="Launch interactive terminal UI")
    _add_shared_runtime_args(ui_parser, require_file=False)
    ui_parser.add_argument(
        "--benchmark-source",
        choices=_benchmark_source_choices(),
        default="all-gists",
        help="Default benchmark source for UI case runs",
    )

    doctor_parser = subparsers.add_parser("doctor", help="Run local environment checks")
    doctor_parser.add_argument(
        "--base",
        default="http://localhost:11434",
        help="Ollama base URL to validate",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark data helpers")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)
    breakdown_parser = benchmark_subparsers.add_parser("breakdown", help="Show dataset and filter breakdowns")
    breakdown_parser.add_argument(
        "--source",
        choices=_benchmark_source_choices(),
        default="all-gists",
        help="Active source used in summary output",
    )
    benchmark_subparsers.add_parser("rebuild-filter", help="Rebuild competition filter from project CSVs")

    return parser


def _add_shared_runtime_args(parser: argparse.ArgumentParser, *, require_file: bool) -> None:
    if require_file:
        parser.add_argument(
            "--file",
            required=True,
            help="Absolute path to snippet file",
        )
    else:
        parser.add_argument(
            "--file",
            default="",
            help="Optional default snippet path for UI",
        )

    parser.add_argument("--model", default="gemma2", help="Ollama model name")
    parser.add_argument("--base", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--temp", type=float, default=0.7, help="Model temperature")
    parser.add_argument("--loop", type=int, default=10, help="Maximum loop attempts")
    parser.add_argument("--range", type=int, default=0, help="Python version search range")
    parser.add_argument("--verbose", action="store_true", help="Verbose output from test executor")
    parser.add_argument(
        "--no-rag",
        dest="rag",
        action="store_false",
        help="Disable initial RAG import extraction",
    )
    parser.set_defaults(rag=True)


def _benchmark_source_choices() -> tuple[BenchmarkSource, BenchmarkSource, BenchmarkSource]:
    return ("all-gists", "dockerized-gists", "competition-run")


if __name__ == "__main__":
    raise SystemExit(main())
