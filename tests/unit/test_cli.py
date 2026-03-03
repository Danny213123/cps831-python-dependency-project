from agentic_python_dependency.cli import build_parser


def test_benchmark_run_parser_accepts_jobs_flag() -> None:
    parser = build_parser()

    args = parser.parse_args(["benchmark", "run", "--subset", "smoke30", "--jobs", "3"])

    assert args.command == "benchmark"
    assert args.benchmark_command == "run"
    assert args.jobs == 3


def test_benchmark_segment_parser_accepts_subset_and_jobs() -> None:
    parser = build_parser()

    args = parser.parse_args(["benchmark", "segment", "--subset", "smoke30", "--jobs", "2"])

    assert args.command == "benchmark"
    assert args.benchmark_command == "segment"
    assert args.subset == "smoke30"
    assert args.jobs == 2


def test_benchmark_full_parser_accepts_jobs() -> None:
    parser = build_parser()

    args = parser.parse_args(["benchmark", "full", "--jobs", "4"])

    assert args.command == "benchmark"
    assert args.benchmark_command == "full"
    assert args.jobs == 4
