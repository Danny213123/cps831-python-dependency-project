from pathlib import Path
import subprocess

from agentic_python_dependency.config import Settings
from agentic_python_dependency.router import OllamaInvocationStats, OllamaStatsTracker
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


def test_terminal_ui_switching_to_research_applies_full_preset_runtime_config(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["4", "2", "7", "", "8"])

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

    assert settings.preset == "research"
    assert settings.prompt_profile == "research-rag"
    assert settings.max_attempts == 15
    assert settings.rag_mode == "hybrid"
    assert settings.structured_prompting is True
    assert settings.candidate_plan_count == 3
    assert settings.allow_candidate_fallback_before_repair is True
    assert settings.repair_cycle_limit == 2
    assert settings.repo_evidence_enabled is True


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


def test_terminal_ui_switching_to_pyego_autodetects_python311(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["4", "1", "2", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(
        TerminalUI,
        "_discover_python311_for_pyego",
        lambda self: ("/tmp/pyego311/python", (3, 11, 9)),
    )

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
    assert settings.pyego_python == "/tmp/pyego311/python"


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


def test_terminal_ui_blocks_pyego_run_when_runtime_is_invalid(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.resolver = "pyego"
    outputs: list[str] = []
    benchmark_calls: list[dict[str, object]] = []
    inputs = iter(["2", "1", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(
        "agentic_python_dependency.terminal_ui.validate_pyego_runtime",
        lambda _: (False, "typed_ast missing"),
    )

    ui = TerminalUI(
        settings=settings,
        doctor_command=lambda *args, **kwargs: 0,
        run_benchmark=lambda *args, **kwargs: benchmark_calls.append(kwargs) or 0,
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

    assert not benchmark_calls
    assert any("PyEGo runtime check failed." in line for line in outputs)


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
    inputs = iter(["4", "8", "", "8"])

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


def test_terminal_ui_can_open_official_setup_from_config_menu(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    inputs = iter(["4", "6", "1", "", "8"])
    calls: list[str] = []

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(TerminalUI, "_setup_local_pyego_neo4j", lambda self: calls.append("called"))

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

    assert calls == ["called"]


def test_terminal_ui_rewrites_pyego_config_for_local_neo4j(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.pyego_root = tmp_path / "external" / "PyEGo"
    settings.pyego_root.mkdir(parents=True, exist_ok=True)
    config_path = settings.pyego_root / "config.py"
    config_path.write_text(
        'NEO4J_URI = "neo4j+s://instance.databases.neo4j.io"\n'
        'NEO4J_PWD = "secret"\n',
        encoding="utf-8",
    )

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
    )

    rewritten = ui._rewrite_pyego_local_neo4j_config()
    updated = rewritten.read_text(encoding="utf-8")

    assert 'NEO4J_URI = "bolt://localhost:7687"' in updated
    assert "NEO4J_PWD = None" in updated
    assert "NEO4J_USERNAME = None" in updated
    assert "NEO4J_DATABASE = None" in updated


def test_terminal_ui_detects_docker_arm_manifest_errors(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
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
    )

    assert ui._is_docker_platform_manifest_error(
        "no matching manifest for linux/arm64/v8 in the manifest list entries"
    )
    assert ui._is_docker_platform_manifest_error("no match for platform in manifest: not found")
    assert not ui._is_docker_platform_manifest_error("connection refused")


def test_terminal_ui_builds_pykg_dump_from_split_parts(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.pyego_root = tmp_path / "external" / "PyEGo"
    pykg_dir = settings.pyego_root / "PyKG"
    pykg_dir.mkdir(parents=True, exist_ok=True)
    (pykg_dir / "PyKG.dump.aa").write_bytes(b"hello ")
    (pykg_dir / "PyKG.dump.ab").write_bytes(b"world")

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
    )

    dump_path = ui._ensure_pykg_dump_file()

    assert dump_path.name == "PyKG.dump"
    assert dump_path.read_bytes() == b"hello world"


def test_terminal_ui_can_save_and_load_loadout(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.preset = "accuracy"
    settings.use_rag = False
    outputs: list[str] = []
    save_inputs = iter(["5", "1", "nightly", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    save_ui = TerminalUI(
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
        input_fn=lambda prompt: next(save_inputs),
    )
    save_ui.run()

    settings.preset = "optimized"
    settings.use_rag = True

    load_inputs = iter(["5", "2", "1", "", "8"])
    load_ui = TerminalUI(
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
        input_fn=lambda prompt: next(load_inputs),
    )
    load_ui.run()

    assert settings.preset == "accuracy"
    assert settings.use_rag is False
    assert (settings.data_dir / "loadouts" / "nightly.json").exists()


def test_terminal_ui_can_delete_loadout(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    loadouts_dir = settings.data_dir / "loadouts"
    loadouts_dir.mkdir(parents=True, exist_ok=True)
    (loadouts_dir / "temp.json").write_text("{}", encoding="utf-8")

    outputs: list[str] = []
    inputs = iter(["5", "3", "1", "", "8"])

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

    assert not (loadouts_dir / "temp.json").exists()


def test_terminal_ui_bootstraps_pyego_requirements_once_on_python311(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.pyego_root = tmp_path / "external" / "PyEGo"
    settings.pyego_root.mkdir(parents=True, exist_ok=True)
    (settings.pyego_root / "requirements.txt").write_text("typed-ast>=1.4.1\n", encoding="utf-8")

    outputs: list[str] = []
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
        input_fn=lambda prompt: "",
    )

    monkeypatch.setattr(
        TerminalUI,
        "_ensure_python311_for_resolver",
        lambda self, resolver: ("/tmp/python311", (3, 11, 8)),
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("agentic_python_dependency.terminal_ui.subprocess.run", fake_run)

    first_ok, first_detail = ui._maybe_auto_install_official_requirements("pyego")
    second_ok, second_detail = ui._maybe_auto_install_official_requirements("pyego")

    assert first_ok is True
    assert "Auto-installed pyego requirements" in first_detail
    assert second_ok is True
    assert "already bootstrapped" in second_detail
    assert len(calls) == 1
    assert calls[0][:4] == ["/tmp/python311", "-m", "pip", "install"]


def test_terminal_ui_skips_bootstrap_when_python_is_not_311(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.readpye_root = tmp_path / "external" / "ReadPyE"
    settings.readpye_root.mkdir(parents=True, exist_ok=True)
    (settings.readpye_root / "requirements.txt").write_text("neo4j==4.4.5\n", encoding="utf-8")

    outputs: list[str] = []
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
        input_fn=lambda prompt: "",
    )

    monkeypatch.setattr(
        TerminalUI,
        "_ensure_python311_for_resolver",
        lambda self, resolver: ("/tmp/python313", (3, 13, 2)),
    )
    monkeypatch.setattr(
        "agentic_python_dependency.terminal_ui.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pip install should not run")),
    )

    ok, detail = ui._maybe_auto_install_official_requirements("readpye")

    assert ok is True
    assert "Skipped auto-install" in detail


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


def test_terminal_ui_resume_research_run_restores_full_preset_runtime_config(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    benchmark_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    inputs = iter(["2", "3", "1", "", "8"])

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    run_dir = settings.artifacts_dir / "run123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-state.json").write_text(
        (
            '{"run_id":"run123","status":"paused","target":"full","jobs":1,'
            '"completed":3,"total":30,"preset":"research","prompt_profile":"research-rag"}'
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
    assert settings.preset == "research"
    assert settings.prompt_profile == "research-rag"
    assert settings.rag_mode == "hybrid"
    assert settings.structured_prompting is True
    assert settings.allow_candidate_fallback_before_repair is True
    assert settings.repair_cycle_limit == 2


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
    assert dashboard.recent_case_results
    assert dashboard.recent_case_results[0]["case_id"] == "case-2"
    assert dashboard.recent_case_results[0]["success"] is True


def test_terminal_benchmark_dashboard_preloads_completed_results_on_start(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    dashboard.start(
        run_id="run123",
        total=5,
        completed=2,
        successes=1,
        failures=1,
        resolver="apdr",
        preset="research",
        prompt_profile="research-rag",
        model_summary="mistral-nemo-12b",
        jobs=1,
        target="full",
        artifacts_dir=Path("/tmp/run123"),
        completed_results=[
            {
                "case_id": "case-newer",
                "success": False,
                "attempts": 3,
                "target_python": "2.7.18",
                "wall_clock_seconds": 12.5,
                "final_error_category": "ImportError",
                "result_matches_csv": "PASS",
                "dependencies": ["rx==1.2.4", "twisted==19.10.0"],
            },
            {
                "case_id": "case-older",
                "success": True,
                "attempts": 1,
                "target_python": "3.12",
                "wall_clock_seconds": 3.2,
                "result_matches_csv": "FAIL",
                "dependencies": ["requests==2.32.3"],
            },
        ],
    )
    dashboard._stop_event.set()
    if dashboard._thread is not None:
        dashboard._thread.join(timeout=0.2)

    assert [row["case_id"] for row in dashboard.recent_case_results[:2]] == ["case-newer", "case-older"]
    assert dashboard.recent_case_results[0]["pllm_match"] == "MATCH"
    assert dashboard.recent_case_results[1]["pllm_match"] == "MISS"
    rendered = "".join(fragment for _, fragment in dashboard._results_table_formatted_text())
    assert "case-newer" in rendered
    assert "case-older" in rendered
    assert "MATCH" in rendered
    assert "MISS" in rendered


def test_terminal_benchmark_dashboard_reports_rate_speed_and_eta(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)
    stats = OllamaStatsTracker()
    stats.record(
        OllamaInvocationStats(
            stage="repair",
            model="gemma3:12b",
            prompt_eval_count=20,
            prompt_eval_duration_ns=100_000_000,
            eval_count=45,
            eval_duration_ns=900_000_000,
        )
    )

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
        ollama_stats=stats,
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
    assert "1 calls, out 45 tok @ 50.0 tok/s" in rendered
    assert "repair / gemma3:12b @ 50.0 tok/s" in rendered


def test_terminal_benchmark_dashboard_renders_recent_case_table(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    dashboard.start(
        run_id="run123",
        total=4,
        completed=0,
        successes=0,
        failures=0,
        resolver="apdr",
        preset="research",
        prompt_profile="research-rag",
        model_summary="mistral-nemo-12b",
        jobs=1,
        target="full",
        artifacts_dir=Path("/tmp/run123"),
    )
    dashboard.advance(
        {
            "case_id": "case-pass",
            "success": True,
            "attempts": 1,
            "target_python": "3.12",
            "wall_clock_seconds": 5.2,
            "result_matches_csv": "PASS",
            "dependencies": ["requests==2.32.3"],
        }
    )
    dashboard.advance(
        {
            "case_id": "case-fail",
            "success": False,
            "attempts": 2,
            "target_python": "3.8",
            "wall_clock_seconds": 11.4,
            "final_error_category": "NativeBuildError",
            "result_matches_csv": "FAIL",
            "dependencies": ["scrapy==2.9.0", "Twisted==25.5.0"],
        }
    )
    dashboard._stop_event.set()
    if dashboard._thread is not None:
        dashboard._thread.join(timeout=0.2)

    rendered = "".join(fragment for _, fragment in dashboard._results_table_formatted_text())

    assert "STAT" in rendered
    assert "PLLM" in rendered
    assert "DEPENDENCIES" in rendered
    assert "FAIL" in rendered
    assert "PASS" in rendered
    assert "MATCH" in rendered
    assert "MISS" in rendered
    assert rendered.index("case-fail") < rendered.index("case-pass")
    assert "NativeBuildError" in rendered


def test_terminal_benchmark_dashboard_renders_case_activity(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    dashboard.start(
        run_id="run123",
        total=4,
        completed=0,
        successes=0,
        failures=0,
        resolver="apdr",
        preset="research",
        prompt_profile="research-rag",
        model_summary="mistral-nemo-12b",
        jobs=1,
        target="full",
        artifacts_dir=Path("/tmp/run123"),
    )
    dashboard.case_started("case-1")
    dashboard.case_event("case-1", attempt=1, kind="docker_build_start", detail="Starting docker build.")

    rendered = "".join(fragment for _, fragment in dashboard._formatted_text())

    assert "Active cases" in rendered
    assert "docker_build_start" in rendered
    assert "Starting docker build." in rendered
    assert "Recent activity" in rendered


def test_terminal_benchmark_dashboard_can_request_stop(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)

    assert dashboard.stop_requested() is False

    dashboard.request_stop()

    assert dashboard.stop_requested() is True


def test_terminal_benchmark_dashboard_can_request_hard_stop(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    dashboard = TerminalBenchmarkDashboard(refresh_interval=0.01)
    invoked: list[str] = []
    dashboard.set_hard_exit_callback(lambda: invoked.append("exit"))

    dashboard.request_hard_stop()

    assert dashboard.stop_requested() is True
    assert dashboard.hard_stop_requested() is True
    assert invoked == ["exit"]
