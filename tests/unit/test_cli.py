import time
import warnings
from pathlib import Path
import pytest

from agentic_python_dependency.cli import (
    BenchmarkProgress,
    PersistentBenchmarkObserver,
    build_parser,
    collect_doctor_report,
    format_elapsed,
    format_progress_bar,
    load_run_state,
    redirect_runtime_warnings,
    resolve_trace_path,
    main,
)
from agentic_python_dependency.config import Settings


def test_benchmark_run_parser_accepts_jobs_flag() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--preset",
            "thorough",
            "--resolver",
            "readpye",
            "--model-profile",
            "gpt-oss-20b",
            "--fresh-run",
            "--benchmark-source",
            "competition-run",
            "--competition-csv",
            "/tmp/official-results.csv",
            "benchmark",
            "run",
            "--subset",
            "smoke30",
            "--jobs",
            "3",
        ]
    )

    assert args.command == "benchmark"
    assert args.benchmark_command == "run"
    assert args.jobs == 3
    assert args.preset == "thorough"
    assert args.resolver == "readpye"
    assert args.model_profile == "gpt-oss-20b"
    assert args.fresh_run is True
    assert args.benchmark_source == "competition-run"
    assert args.competition_csv == ["/tmp/official-results.csv"]


def test_benchmark_run_parser_accepts_new_moe_model_profiles() -> None:
    parser = build_parser()

    gemma_args = parser.parse_args(["--model-profile", "gemma-moe-lite", "smoke"])
    qwen_args = parser.parse_args(["--model-profile", "qwen35-moe-lite", "smoke"])

    assert gemma_args.model_profile == "gemma-moe-lite"
    assert qwen_args.model_profile == "qwen35-moe-lite"


def test_benchmark_run_parser_accepts_runtime_controls() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--no-moe",
            "--no-rag",
            "--no-langchain",
            "--extraction-model",
            "extract:model",
            "--runner-model",
            "runner:model",
            "--version-model",
            "version:model",
            "--repair-model",
            "repair:model",
            "--adjudication-model",
            "adj:model",
            "smoke",
        ]
    )

    assert args.moe is False
    assert args.rag is False
    assert args.langchain is False
    assert args.extraction_model == "extract:model"
    assert args.runner_model == "runner:model"
    assert args.version_model == "version:model"
    assert args.repair_model == "repair:model"
    assert args.adjudication_model == "adj:model"


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


def test_top_level_ui_parser_is_available() -> None:
    parser = build_parser()

    args = parser.parse_args(["ui"])

    assert args.command == "ui"


def test_parser_program_name_is_apdr() -> None:
    parser = build_parser()

    assert parser.prog == "apdr"


def test_report_modules_parser_accepts_grouping() -> None:
    parser = build_parser()

    args = parser.parse_args(["report", "modules", "--run-id", "run123", "--grouping", "raw"])

    assert args.command == "report"
    assert args.report_command == "modules"
    assert args.grouping == "raw"


def test_report_modules_parser_accepts_paper_compatible() -> None:
    parser = build_parser()

    args = parser.parse_args(["report", "modules", "--run-id", "run123", "--paper-compatible"])

    assert args.command == "report"
    assert args.report_command == "modules"
    assert args.paper_compatible is True


def test_report_trace_parser_accepts_case_and_tail() -> None:
    parser = build_parser()

    args = parser.parse_args(["report", "trace", "--run-id", "run123", "--case-id", "case1", "--tail", "20"])

    assert args.command == "report"
    assert args.report_command == "trace"
    assert args.case_id == "case1"
    assert args.tail == 20


def test_report_timeline_parser_accepts_run_id() -> None:
    parser = build_parser()

    args = parser.parse_args(["report", "timeline", "--run-id", "run123"])

    assert args.command == "report"
    assert args.report_command == "timeline"
    assert args.run_id == "run123"


def test_parser_accepts_experimental_preset_and_prompt_profile() -> None:
    parser = build_parser()

    args = parser.parse_args(["--preset", "experimental", "--prompt-profile", "research-rag", "smoke"])

    assert args.preset == "experimental"
    assert args.prompt_profile == "research-rag"


def test_parser_accepts_research_bundle_and_feature_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--preset",
            "research",
            "--research-bundle",
            "enhanced",
            "--research-feature",
            "dynamic_imports",
            "--no-research-feature",
            "repair_memory",
            "smoke",
        ]
    )

    assert args.preset == "research"
    assert args.research_bundle == "enhanced"
    assert args.research_feature == ["dynamic_imports"]
    assert args.no_research_feature == ["repair_memory"]


def test_main_rejects_experimental_with_non_apdr_resolver(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["--preset", "experimental", "--resolver", "pyego", "doctor"])

    assert excinfo.value.code == 2


def test_main_rejects_research_with_non_apdr_resolver(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["--preset", "research", "--resolver", "readpye", "doctor"])

    assert excinfo.value.code == 2


def test_main_rejects_research_feature_flags_without_research_preset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["--preset", "optimized", "--research-bundle", "enhanced", "doctor"])

    assert excinfo.value.code == 2


def test_collect_doctor_report_marks_missing_tools_and_dataset(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path)

    monkeypatch.setattr("agentic_python_dependency.cli.shutil.which", lambda _: None)

    def fake_urlopen(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("agentic_python_dependency.cli.urllib.request.urlopen", fake_urlopen)

    report = collect_doctor_report(settings)

    assert report["overall_status"] == "warning"
    assert report["resolver"] == "apdr"
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
    assert "ok 0" in line
    assert "fail 0" in line
    assert "elapsed 00:01:05" in line


def test_benchmark_progress_refresh_thread_starts_and_stops(monkeypatch) -> None:
    progress = BenchmarkProgress("run123", total=10, refresh_interval=0.01)
    monkeypatch.setattr(progress, "_isatty", True)

    progress.start(
        run_id="run123",
        total=10,
        completed=0,
        successes=0,
        failures=0,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe: gemma3:4b / gemma3:12b",
        jobs=1,
        target="smoke30",
        artifacts_dir=Path("/tmp/run123"),
    )
    time.sleep(0.03)
    progress.finish(summary_path=Path("/tmp/run123/summary.json"), warnings_path=None)

    assert progress._thread is not None
    assert not progress._thread.is_alive()


def test_benchmark_progress_tracks_case_results() -> None:
    progress = BenchmarkProgress("run123", total=3, completed=1)
    progress.start(
        run_id="run123",
        total=3,
        completed=1,
        successes=1,
        failures=0,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe: gemma3:4b / gemma3:12b",
        jobs=1,
        target="smoke30",
        artifacts_dir=Path("/tmp/run123"),
    )

    progress.case_started("case-2")
    progress.advance({"case_id": "case-2", "success": False, "final_error_category": "TimeoutError"})

    assert progress.completed == 2
    assert progress.successes == 1
    assert progress.failures == 1
    assert progress.last_case_id == "case-2"
    assert progress.last_status == "TimeoutError"


def test_benchmark_progress_can_request_stop() -> None:
    progress = BenchmarkProgress("run123", total=3)

    assert progress.stop_requested() is False

    progress.request_stop()

    assert progress.stop_requested() is True


def test_persistent_benchmark_observer_writes_run_state_files(tmp_path: Path) -> None:
    inner = BenchmarkProgress("run123", total=5)
    observer = PersistentBenchmarkObserver(inner, tmp_path / "run123")

    observer.start(
        run_id="run123",
        total=5,
        completed=1,
        successes=1,
        failures=0,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe",
        jobs=1,
        target="smoke30",
        artifacts_dir=tmp_path / "run123",
    )
    observer.case_started("case-2")
    observer.advance({"case_id": "case-2", "success": True, "final_error_category": "Success"})
    observer.finish(summary_path=tmp_path / "run123" / "summary.json", warnings_path=None, status="paused")

    payload = load_run_state(tmp_path / "run123")
    assert payload["status"] == "paused"
    assert payload["completed"] == 2
    assert payload["successes"] == 2
    assert (tmp_path / "run123" / "run-state.md").exists()


def test_persistent_benchmark_observer_restores_elapsed_seconds(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class RecordingObserver(BenchmarkProgress):
        def start(self, **kwargs):  # type: ignore[override]
            captured.update(kwargs)

        def case_started(self, case_id: str) -> None:
            return None

        def advance(self, result: dict[str, object]) -> None:
            return None

        def finish(self, *, summary_path: Path, warnings_path: Path | None, status: str = "completed") -> None:
            return None

        def stop_requested(self) -> bool:
            return False

    observer = PersistentBenchmarkObserver(
        RecordingObserver("run123", total=5),
        tmp_path / "run123",
        {"elapsed_seconds": 42.0, "started_at": "2026-03-03T00:00:00+00:00"},
    )

    observer.start(
        run_id="run123",
        total=5,
        completed=2,
        successes=1,
        failures=1,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe",
        jobs=1,
        target="smoke30",
        artifacts_dir=tmp_path / "run123",
    )

    assert captured["elapsed_seconds"] == 42.0


def test_redirect_runtime_warnings_writes_warning_to_file(tmp_path: Path) -> None:
    warning_path = tmp_path / "warnings.log"

    with redirect_runtime_warnings(warning_path):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(r"invalid escape sequence '\s'", SyntaxWarning)

    contents = warning_path.read_text(encoding="utf-8")
    assert "SyntaxWarning" in contents
    assert "invalid escape sequence" in contents


def test_resolve_trace_path_supports_run_and_case_scope(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)

    assert resolve_trace_path(settings, "run1") == settings.artifacts_dir / "run1" / "llm-trace.log"
    assert resolve_trace_path(settings, "run1", "case1") == settings.artifacts_dir / "run1" / "case1" / "llm-trace.log"
