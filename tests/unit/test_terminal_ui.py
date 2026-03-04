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
        run_failed_cases=lambda *args, **kwargs: 0,
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
    assert any("APDR Command Center" in line for line in outputs)
    assert any("Exiting APDR UI." in line for line in outputs)


def test_terminal_ui_can_switch_preset(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["4", "2", "6", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
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


def test_terminal_ui_switching_to_experimental_forces_apdr_resolver(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.resolver = "pyego"
    outputs: list[str] = []
    inputs = iter(["4", "2", "8", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert settings.preset == "experimental"
    assert settings.resolver == "apdr"


def test_terminal_ui_can_switch_resolver(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["4", "1", "2", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert settings.resolver == "pyego"


def test_terminal_ui_switching_resolver_from_experimental_restores_supported_preset(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.preset = "experimental"
    settings.prompt_profile = "optimized-strict"
    outputs: list[str] = []
    inputs = iter(["4", "1", "2", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert settings.resolver == "pyego"
    assert settings.preset == "accuracy"


def test_terminal_ui_can_switch_model_bundle(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["4", "3", "3", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
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
    inputs = iter(["4", "7", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
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


def test_terminal_ui_can_configure_runtime_controls(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["4", "5", "1", "", "4", "5", "6", "custom:version", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert settings.use_moe is False
    assert settings.version_model == "custom:version"
    assert settings.model_profile == "custom"


def test_terminal_ui_blocks_research_controls_when_preset_is_not_research(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["4", "4", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert any("Research controls are only available when the preset is research." in line for line in outputs)


def test_terminal_ui_smoke_run_uses_dashboard(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    benchmark_calls: list[dict[str, object]] = []
    inputs = iter(["2", "1", "1", "", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    def fake_run_benchmark(*args, **kwargs):
        benchmark_calls.append(kwargs)
        return 0

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=fake_run_benchmark,
        run_failed_cases=lambda *args, **kwargs: 0,
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


def test_terminal_ui_can_resume_saved_benchmark_run(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    benchmark_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    inputs = iter(["2", "3", "1", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    run_dir = settings.artifacts_dir / "run123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-state.json").write_text(
        (
            '{"run_id":"run123","status":"paused","target":"smoke30","jobs":2,'
            '"completed":3,"total":30}'
        ),
        encoding="utf-8",
    )

    def fake_run_benchmark(*args, **kwargs):
        benchmark_calls.append((args, kwargs))
        return 0

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=fake_run_benchmark,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        timeline_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert benchmark_calls
    args, kwargs = benchmark_calls[0]
    assert args[2] == "smoke30"
    assert args[3] is False
    assert args[4] == "run123"
    assert args[5] == 2
    assert kwargs["fresh_run"] is False
    assert isinstance(kwargs["observer"], TerminalBenchmarkDashboard)


def test_terminal_ui_reports_when_no_resumable_runs_exist(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["2", "3", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        timeline_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    exit_code = ui.run()

    assert exit_code == 0
    assert any("No resumable benchmark runs found." in line for line in outputs)


def test_terminal_ui_can_retry_failed_cases_from_prior_run(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    failed_case_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    inputs = iter(["2", "4", "1", "2", "", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    run_dir = settings.artifacts_dir / "run123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-state.json").write_text(
        '{"run_id":"run123","status":"completed","target":"full","jobs":1,"completed":2,"total":2}',
        encoding="utf-8",
    )
    failed_case_dir = run_dir / "case-a"
    failed_case_dir.mkdir(parents=True, exist_ok=True)
    (failed_case_dir / "result.json").write_text('{"case_id":"case-a","success":false}', encoding="utf-8")
    passed_case_dir = run_dir / "case-b"
    passed_case_dir.mkdir(parents=True, exist_ok=True)
    (passed_case_dir / "result.json").write_text('{"case_id":"case-b","success":true}', encoding="utf-8")

    def fake_run_failed_cases(*args, **kwargs):
        failed_case_calls.append((args, kwargs))
        return 0

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=fake_run_failed_cases,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        timeline_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert failed_case_calls
    args, kwargs = failed_case_calls[0]
    assert args[1] == "run123"
    assert args[2] is None
    assert args[3] is None
    assert kwargs["jobs"] == 2
    assert kwargs["notify_paths"] is False
    assert kwargs["fresh_run"] is False
    assert isinstance(kwargs["observer"], TerminalBenchmarkDashboard)


def test_terminal_ui_reports_when_no_failed_case_runs_exist(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["2", "4", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        timeline_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    exit_code = ui.run()

    assert exit_code == 0
    assert any("No prior runs with failed cases were found." in line for line in outputs)


def test_terminal_ui_can_run_timeline_view(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    timeline_calls: list[tuple[object, ...]] = []
    inputs = iter(["3", "4", "1", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    (settings.artifacts_dir / "run123").mkdir(parents=True, exist_ok=True)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
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
    inputs = iter(["3", "3", "1", "15", "canonical", "paper-compatible", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    run_dir = settings.artifacts_dir / "run123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "module-success-paper.md").write_text("# Module Success Table\n", encoding="utf-8")

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
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
    assert any("Module Success Table" in line for line in outputs)


def test_terminal_ui_surfaces_captured_command_errors_without_crashing(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["3", "3", "1", "15", "canonical", "paper-compatible", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    (settings.artifacts_dir / "run123").mkdir(parents=True, exist_ok=True)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
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


def test_terminal_ui_can_select_run_for_summary(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    summary_calls: list[tuple[object, ...]] = []
    inputs = iter(["3", "1", "1", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    (settings.artifacts_dir / "run123").mkdir(parents=True, exist_ok=True)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
        run_project=lambda *args, **kwargs: 0,
        summarize_command=lambda *args, **kwargs: summary_calls.append(args) or 0,
        failures_command=lambda *args, **kwargs: 0,
        modules_command=lambda *args, **kwargs: 0,
        ensure_smoke_subset=lambda *args, **kwargs: tmp_path,
        output=outputs.append,
        input_fn=lambda prompt: next(inputs),
    )

    ui.run()

    assert summary_calls
    assert summary_calls[0][1] == "run123"


def test_terminal_ui_reports_when_no_runs_exist(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["3", "1", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: 0,
        run_failed_cases=lambda *args, **kwargs: 0,
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
    assert any("No run directories found" in line for line in outputs)


def test_terminal_benchmark_dashboard_tracks_state(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    dashboard.start(
        run_id="run123",
        total=5,
        completed=1,
        successes=1,
        failures=0,
        resolver="apdr",
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


def test_terminal_benchmark_dashboard_reports_rate_speed_and_eta(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    dashboard.start(
        run_id="run123",
        total=10,
        completed=4,
        successes=3,
        failures=1,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe-lite",
        jobs=1,
        target="smoke30",
        artifacts_dir=Path("/tmp/run123"),
        elapsed_seconds=40.0,
    )
    dashboard._stop_event.set()
    if dashboard._thread is not None:
        dashboard._thread.join(timeout=0.2)

    assert dashboard._seconds_per_completed_case(40.0) == 10.0
    assert dashboard._eta_seconds(40.0) == 60.0
    rendered = "".join(fragment for _, fragment in dashboard._formatted_text())
    assert "Success rate: 75.0%" in rendered
    assert "Speed: 10.0s/case" in rendered
    assert "ETA: 00:01:00" in rendered


def test_terminal_benchmark_dashboard_can_request_stop(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    assert dashboard.stop_requested() is False

    dashboard.request_stop()

    assert dashboard.stop_requested() is True
