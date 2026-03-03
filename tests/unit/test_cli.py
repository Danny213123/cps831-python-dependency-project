import time
import warnings
from pathlib import Path

from agentic_python_dependency.cli import (
    BenchmarkProgress,
    build_parser,
    collect_doctor_report,
    format_elapsed,
    format_progress_bar,
    redirect_runtime_warnings,
)
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


def test_format_progress_bar_renders_partial_progress() -> None:
    assert format_progress_bar(3, 4, width=8) == "[######--]"


def test_format_elapsed_formats_hms() -> None:
    assert format_elapsed(3661.9) == "01:01:01"


def test_benchmark_progress_line_contains_run_id_and_counts(monkeypatch) -> None:
    progress = BenchmarkProgress("run123", total=10, completed=4)
    monkeypatch.setattr(progress, "started_at", progress.started_at - 65)

    line = progress._line()

    assert "Benchmark run123" in line
    assert "4/10" in line
    assert "40.0%" in line
    assert "elapsed 00:01:05" in line


def test_benchmark_progress_refresh_thread_starts_and_stops(monkeypatch) -> None:
    progress = BenchmarkProgress("run123", total=10, refresh_interval=0.01)
    monkeypatch.setattr(progress, "_isatty", True)

    progress.start()
    time.sleep(0.03)
    progress.finish()

    assert progress._thread is not None
    assert not progress._thread.is_alive()


def test_redirect_runtime_warnings_writes_warning_to_file(tmp_path: Path) -> None:
    warning_path = tmp_path / "warnings.log"

    with redirect_runtime_warnings(warning_path):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(r"invalid escape sequence '\s'", SyntaxWarning)

    contents = warning_path.read_text(encoding="utf-8")
    assert "SyntaxWarning" in contents
    assert "invalid escape sequence" in contents
