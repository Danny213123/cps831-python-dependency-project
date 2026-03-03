from pathlib import Path

from agentic_python_dependency.cli import build_parser, collect_doctor_report
from agentic_python_dependency.config import Settings


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


def test_top_level_smoke_parser_accepts_jobs() -> None:
    parser = build_parser()

    args = parser.parse_args(["smoke", "--jobs", "2"])

    assert args.command == "smoke"
    assert args.jobs == 2


def test_top_level_full_parser_accepts_jobs() -> None:
    parser = build_parser()

    args = parser.parse_args(["full", "--jobs", "3"])

    assert args.command == "full"
    assert args.jobs == 3


def test_top_level_solve_parser_accepts_path() -> None:
    parser = build_parser()

    args = parser.parse_args(["solve", "--path", "/tmp/example"])

    assert args.command == "solve"
    assert args.path == "/tmp/example"


def test_top_level_doctor_parser_accepts_ref() -> None:
    parser = build_parser()

    args = parser.parse_args(["doctor", "--ref", "abc123"])

    assert args.command == "doctor"
    assert args.ref == "abc123"


def test_collect_doctor_report_marks_missing_tools_and_dataset(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path)

    monkeypatch.setattr("agentic_python_dependency.cli.shutil.which", lambda _: None)

    def fake_urlopen(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("agentic_python_dependency.cli.urllib.request.urlopen", fake_urlopen)

    report = collect_doctor_report(settings)

    assert report["overall_status"] == "warning"
    names = {check["name"]: check for check in report["checks"]}
    assert names["docker_cli"]["status"] == "missing"
    assert names["ollama_server"]["status"] == "warning"
    assert names["gistable_dataset"]["status"] == "warning"
