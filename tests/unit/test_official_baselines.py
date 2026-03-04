from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agentic_python_dependency.config import Settings
from agentic_python_dependency.tools.official_baselines import (
    _write_process_logs,
    parse_dockerfile_plan,
    parse_pyego_dependency_json,
    validate_pyego_runtime,
)


def test_parse_pyego_dependency_json_extracts_python_packages_and_system_libs() -> None:
    plan = parse_pyego_dependency_json(
        {
            "python_version": "2.7",
            "system_lib": [
                {"name": "libjpeg-dev", "version": "latest", "install_method": "apt"},
                {"name": "brew-only", "version": "latest", "install_method": "brew"},
            ],
            "python_packages": [
                {"name": "PyYAML", "version": "6.0.2", "install_method": "pip"},
                {"name": "requests", "version": "latest", "install_method": "pip"},
            ],
            "message": "",
        }
    )

    assert plan.target_python == "2.7"
    assert [dependency.pin() for dependency in plan.dependencies] == ["PyYAML==6.0.2", "requests"]
    assert plan.system_packages == ["libjpeg-dev"]
    assert plan.implementation == "official"


def test_parse_dockerfile_plan_extracts_dependencies_and_system_packages() -> None:
    plan = parse_dockerfile_plan(
        "\n".join(
            [
                "FROM python:2.7.18",
                "RUN apt-get update && apt-get install -y --no-install-recommends libjpeg-dev pkg-config && rm -rf /var/lib/apt/lists/*",
                "RUN pip install twisted==20.3 cryptography==3.3.2",
                "",
            ]
        )
    )

    assert plan.target_python == "2.7.18"
    assert [dependency.pin() for dependency in plan.dependencies] == [
        "cryptography==3.3.2",
        "twisted==20.3",
    ]
    assert plan.system_packages == ["libjpeg-dev", "pkg-config"]


def test_write_process_logs_decodes_non_utf8_bytes(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(["tool"], 1, stdout=b"out\x81\n", stderr=b"err\x81\n")

    _write_process_logs(tmp_path, "baseline", completed)

    assert "out" in (tmp_path / "baseline.stdout.log").read_text(encoding="utf-8")
    assert "err" in (tmp_path / "baseline.stderr.log").read_text(encoding="utf-8")


def _settings_with_pyego(tmp_path: Path) -> Settings:
    settings = Settings.from_env(project_root=tmp_path)
    pyego_root = tmp_path / "external" / "PyEGo"
    pyego_root.mkdir(parents=True)
    (pyego_root / "PyEGo.py").write_text("print('ok')\n", encoding="utf-8")
    (pyego_root / "config.py").write_text(
        'NEO4J_URI = "bolt://localhost:7687"\nNEO4J_PWD = None\n',
        encoding="utf-8",
    )
    settings.pyego_root = pyego_root
    settings.pyego_python = "python"
    return settings


def test_validate_pyego_runtime_rejects_python_above_311(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings_with_pyego(tmp_path)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=b"3.13.2\n", stderr=b"")

    monkeypatch.setattr("agentic_python_dependency.tools.official_baselines.subprocess.run", fake_run)
    ok, detail = validate_pyego_runtime(settings)

    assert not ok
    assert "APDR_PYEGO_PYTHON <= 3.11" in detail


def test_validate_pyego_runtime_reports_missing_typed_ast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings_with_pyego(tmp_path)
    calls: list[int] = []

    def fake_run(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args[0], 0, stdout=b"3.11.10\n", stderr=b"")
        return subprocess.CompletedProcess(
            args[0],
            1,
            stdout=b"",
            stderr=b"ModuleNotFoundError: No module named 'typed_ast'\n",
        )

    monkeypatch.setattr("agentic_python_dependency.tools.official_baselines.subprocess.run", fake_run)
    ok, detail = validate_pyego_runtime(settings)

    assert not ok
    assert "missing typed_ast.ast27" in detail
    assert "pip install -r" in detail


def test_validate_pyego_runtime_succeeds_when_requirements_are_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings_with_pyego(tmp_path)
    calls: list[int] = []

    def fake_run(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args[0], 0, stdout=b"3.11.12\n", stderr=b"")
        return subprocess.CompletedProcess(args[0], 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr("agentic_python_dependency.tools.official_baselines.subprocess.run", fake_run)
    monkeypatch.setattr(
        "agentic_python_dependency.tools.official_baselines._probe_pyego_neo4j_connection",
        lambda *args, **kwargs: (True, "ok"),
    )
    ok, detail = validate_pyego_runtime(settings)

    assert ok
    assert "ready" in detail


def test_validate_pyego_runtime_reports_missing_neo4j_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings_with_pyego(tmp_path)
    (settings.pyego_root / "config.py").write_text(
        'NEO4J_URI = "YOUR NEO4J URI"\nNEO4J_PWD = "YOUR NEO4J PASSWORD"\n',
        encoding="utf-8",
    )
    calls: list[int] = []

    def fake_run(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args[0], 0, stdout=b"3.11.10\n", stderr=b"")
        return subprocess.CompletedProcess(args[0], 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr("agentic_python_dependency.tools.official_baselines.subprocess.run", fake_run)
    ok, detail = validate_pyego_runtime(settings)

    assert not ok
    assert "Neo4j configuration is missing" in detail


def test_validate_pyego_runtime_reports_neo4j_connection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings_with_pyego(tmp_path)
    calls: list[int] = []

    def fake_run(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args[0], 0, stdout=b"3.11.10\n", stderr=b"")
        return subprocess.CompletedProcess(args[0], 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr("agentic_python_dependency.tools.official_baselines.subprocess.run", fake_run)
    monkeypatch.setattr(
        "agentic_python_dependency.tools.official_baselines._probe_pyego_neo4j_connection",
        lambda *args, **kwargs: (False, "Cannot open connection to ConnectionProfile('bolt://localhost:7687')"),
    )
    ok, detail = validate_pyego_runtime(settings)

    assert not ok
    assert "cannot reach Neo4j" in detail


def test_validate_pyego_runtime_normalizes_neo4j_scheme_and_passes_username_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings_with_pyego(tmp_path)
    (settings.pyego_root / "config.py").write_text(
        (
            'NEO4J_URI = "neo4j+s://instance.databases.neo4j.io"\n'
            'NEO4J_PWD = "secret"\n'
            'NEO4J_USERNAME = "instance"\n'
            'NEO4J_DATABASE = "instance"\n'
        ),
        encoding="utf-8",
    )
    calls: list[int] = []
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args[0], 0, stdout=b"3.11.10\n", stderr=b"")
        return subprocess.CompletedProcess(args[0], 0, stdout=b"ok\n", stderr=b"")

    def fake_probe(python_executable, *, uri, password, username, database):
        captured["python_executable"] = python_executable
        captured["uri"] = uri
        captured["password"] = password
        captured["username"] = username
        captured["database"] = database
        return True, "ok"

    monkeypatch.setattr("agentic_python_dependency.tools.official_baselines.subprocess.run", fake_run)
    monkeypatch.setattr(
        "agentic_python_dependency.tools.official_baselines._probe_pyego_neo4j_connection",
        fake_probe,
    )

    ok, _ = validate_pyego_runtime(settings)

    assert ok
    assert captured["uri"] == "bolt+s://instance.databases.neo4j.io"
    assert captured["username"] == "instance"
    assert captured["database"] == "instance"
