from agentic_python_dependency.cli import build_parser


def test_benchmark_run_parser_accepts_jobs_flag() -> None:
    parser = build_parser()

    args = parser.parse_args(["benchmark", "run", "--subset", "smoke30", "--jobs", "3"])

    assert args.command == "benchmark"
    assert args.benchmark_command == "run"
    assert args.jobs == 3
