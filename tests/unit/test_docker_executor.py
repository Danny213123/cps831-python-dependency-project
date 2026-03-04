import subprocess
from pathlib import Path

from agentic_python_dependency.config import Settings
from agentic_python_dependency.tools.docker_executor import DockerExecutor, PreparedExecutionContext


def test_patch_dockerfile_inserts_dependency_install_before_cmd() -> None:
    dockerfile = "\n".join(
        [
            "FROM python:3.12-slim",
            "WORKDIR /app",
            'CMD ["python", "snippet.py"]',
            "",
        ]
    )

    patched = DockerExecutor.patch_dockerfile(dockerfile)

    assert "COPY requirements.generated.txt /tmp/requirements.generated.txt" in patched
    assert patched.index("RUN pip install --no-cache-dir -r /tmp/requirements.generated.txt") < patched.index(
        'CMD ["python", "snippet.py"]'
    )


def test_patch_dockerfile_rewrites_benchmark_pip_installs_for_python2() -> None:
    dockerfile = "\n".join(
        [
            "FROM python:2.7.13",
            'RUN ["pip", "install", "emoji"]',
            "RUN pip install requests",
            'ADD snippet.py snippet.py',
            'CMD ["python", "snippet.py"]',
            "",
        ]
    )

    patched = DockerExecutor.patch_dockerfile(
        dockerfile,
        rewrite_python_installs=True,
        target_python="2.7.13",
        bootstrap_pins=["numpy==1.16.6"],
    )

    assert 'RUN ["pip", "install", "emoji"]' not in patched
    assert "RUN pip install requests" not in patched
    assert "RUN pip install --no-cache-dir --upgrade 'pip<21' 'setuptools<45' 'wheel<0.35'" in patched
    assert "RUN pip install --no-cache-dir numpy==1.16.6" in patched
    assert patched.index("RUN pip install --no-cache-dir --upgrade 'pip<21' 'setuptools<45' 'wheel<0.35'") < patched.index(
        'ADD snippet.py snippet.py'
    )


def test_patch_dockerfile_rewrites_python_base_image_when_target_changes() -> None:
    dockerfile = "\n".join(
        [
            "FROM python:2.7.13-slim",
            "WORKDIR /app",
            'CMD ["python", "snippet.py"]',
            "",
        ]
    )

    patched = DockerExecutor.patch_dockerfile(
        dockerfile,
        rewrite_base_python=True,
        target_python="3.11",
    )

    assert "FROM python:3.11-slim" in patched
    assert "FROM python:2.7.13-slim" not in patched


def test_patch_dockerfile_injects_system_packages_for_pygame() -> None:
    dockerfile = "\n".join(
        [
            "FROM python:2.7.13",
            'ADD snippet.py snippet.py',
            'CMD ["python", "snippet.py"]',
            "",
        ]
    )

    patched = DockerExecutor.patch_dockerfile(
        dockerfile,
        rewrite_python_installs=True,
        target_python="2.7.13",
        system_packages=["libsdl1.2-dev", "pkg-config"],
    )

    assert "apt-get update && apt-get install -y --no-install-recommends libsdl1.2-dev pkg-config" in patched
    assert patched.index("apt-get update && apt-get install -y --no-install-recommends libsdl1.2-dev pkg-config") < patched.index(
        'ADD snippet.py snippet.py'
    )


def test_execute_converts_run_timeout_into_failed_result(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    settings.run_timeout_seconds = 60
    executor = DockerExecutor(settings)
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-1",
        validation_command=None,
        artifact_dir=tmp_path,
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(command, 0, stdout="built\n", stderr="")
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(
                command,
                timeout=60,
                output="partial stdout\n",
                stderr="partial stderr\n",
            )
        if command[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "image", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(context)

    assert result.build_succeeded is True
    assert result.run_succeeded is False
    assert result.exit_code == 124
    assert "docker run timed out after 60 seconds." in result.run_log
    assert ["docker", "rm", "-f", "pllm-test-case-1-run"] in calls
    assert ["docker", "image", "rm", "-f", "pllm-test-case-1"] in calls


def test_execute_adds_headless_runtime_environment(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    executor = DockerExecutor(settings)
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-2",
        validation_command="python snippet.py --help",
        artifact_dir=tmp_path,
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        if command[:3] == ["docker", "image", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(context)

    assert result.run_succeeded is True
    run_command = next(command for command in calls if command[:2] == ["docker", "run"])
    assert "-e" in run_command
    assert "MPLBACKEND=Agg" in run_command
    assert "SDL_VIDEODRIVER=dummy" in run_command
    assert "QT_QPA_PLATFORM=offscreen" in run_command


def test_execute_handles_none_stdout_on_windows_subprocess(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    executor = DockerExecutor(settings)
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-3",
        validation_command=None,
        artifact_dir=tmp_path,
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(command, 0, stdout=None, stderr="built\n")
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout=None, stderr="ran\n")
        if command[:3] == ["docker", "image", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(context)

    assert result.build_succeeded is True
    assert result.run_succeeded is True
    assert result.build_log == "built\n"
    assert result.run_log == "ran\n"


def test_execute_decodes_non_utf8_bytes_with_replacement(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    executor = DockerExecutor(settings)
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-4",
        validation_command=None,
        artifact_dir=tmp_path,
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"ok\x81\n", stderr=b"")
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"warn\x81\n")
        if command[:3] == ["docker", "image", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(context)

    assert result.build_succeeded is True
    assert result.run_succeeded is True
    assert "ok" in result.build_log
    assert "warn" in result.run_log
