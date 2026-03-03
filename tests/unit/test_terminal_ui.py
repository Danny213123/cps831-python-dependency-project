from pathlib import Path

from agentic_python_dependency.config import Settings
from agentic_python_dependency.terminal_ui import TerminalBenchmarkDashboard, TerminalUI


def make_settings(tmp_path: Path) -> Settings:
    return Settings.from_env(project_root=tmp_path)


def test_terminal_ui_can_exit_immediately(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    exit_code = ui.run()

    assert exit_code == 0
    assert any("Agentic Python Dependency" in line for line in outputs)
    assert any("Exiting APD UI." in line for line in outputs)


def test_terminal_ui_can_switch_preset(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["p", "6", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert settings.preset == "accuracy"
    assert settings.prompt_profile == "optimized-strict"
    assert settings.max_attempts == 5


def test_terminal_ui_can_switch_model_bundle(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["m", "2", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert settings.model_profile == "qwen35-9b"
    assert settings.extraction_model == "qwen3.5:9b"
    assert settings.reasoning_model == "qwen3.5:9b"


def test_terminal_ui_can_toggle_fresh_run(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["f", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert ui._fresh_run is True


def test_terminal_ui_smoke_run_uses_dashboard(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    benchmark_calls: list[dict[str, object]] = []
    inputs = iter(["2", "1", "", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    def fake_run_benchmark(*args, **kwargs):
        benchmark_calls.append(kwargs)
        return 0

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=fake_run_benchmark,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert benchmark_calls
    assert benchmark_calls[0]["notify_paths"] is False
    assert isinstance(benchmark_calls[0]["observer"], TerminalBenchmarkDashboard)


def test_terminal_ui_can_run_timeline_view(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    timeline_calls: list[tuple[object, ...]] = []
    inputs = iter(["l", "run123", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        timeline_command=lambda *args, **kwargs: timeline_calls.append(args) or 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert timeline_calls
    assert timeline_calls[0][1] == "run123"


def test_terminal_ui_module_report_can_choose_paper_compatible(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    module_calls: list[tuple[object, ...]] = []
    inputs = iter(["7", "run123", "15", "canonical", "paper-compatible", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: module_calls.append(args) or 0,
        timeline_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert module_calls
    assert module_calls[0][-1] is True


def test_terminal_ui_surfaces_captured_command_errors_without_crashing(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["7", "run123", "15", "canonical", "paper-compatible", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        timeline_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    exit_code = ui.run()

    assert exit_code == 0
    assert any("RuntimeError: boom" in line for line in outputs)


def test_terminal_benchmark_dashboard_tracks_state(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    dashboard.start(
        run_id="run123",
        total=5,
        completed=1,
        successes=1,
        failures=0,
        preset="balanced",
        prompt_profile="optimized-strict",
        model_summary="gemma-moe: gemma3:4b / gemma3:12b",
        jobs=2,
        target="smoke30",
        artifacts_dir=Path("/tmp/run123"),
    )
    dashboard.case_started("case-2")
    dashboard.case_started("case-3")
    dashboard.advance({"case_id": "case-2", "success": True})
    dashboard.finish(summary_path=Path("/tmp/run123/summary.json"), warnings_path=None)

    assert dashboard.completed == 5
    assert dashboard.successes == 2
    assert dashboard.failures == 0
    assert dashboard.last_case_id == "case-2"
    assert dashboard.current_cases == []


def test_terminal_benchmark_dashboard_can_request_stop(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    assert dashboard.stop_requested() is False

    dashboard.request_stop()

    assert dashboard.stop_requested() is True
