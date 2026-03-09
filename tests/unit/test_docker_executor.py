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

    assert patched.startswith("# syntax=docker/dockerfile:1.7\n")
    assert "COPY requirements.generated.txt /tmp/requirements.generated.txt" in patched
    assert "RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.generated.txt" in patched
    assert patched.index("RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.generated.txt") < patched.index(
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
    assert "RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade 'pip<21' 'setuptools<45' 'wheel<0.35'" in patched
    assert "RUN --mount=type=cache,target=/root/.cache/pip pip install numpy==1.16.6" in patched
    assert patched.index("RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade 'pip<21' 'setuptools<45' 'wheel<0.35'") < patched.index(
        'ADD snippet.py snippet.py'
    )


def test_patch_dockerfile_shell_quotes_bootstrap_specifiers() -> None:
    dockerfile = "\n".join(
        [
            "FROM python:2.7.18-slim",
            'ADD snippet.py snippet.py',
            'CMD ["python", "snippet.py"]',
            "",
        ]
    )

    patched = DockerExecutor.patch_dockerfile(
        dockerfile,
        rewrite_python_installs=True,
        target_python="2.7.18",
        bootstrap_pins=["numpy==1.16.6", "Cython<3"],
    )

    assert "RUN --mount=type=cache,target=/root/.cache/pip pip install 'Cython<3' numpy==1.16.6" in patched


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


def test_patch_dockerfile_rewrites_legacy_python_apt_sources_before_system_packages() -> None:
    dockerfile = "\n".join(
        [
            "FROM python:2.7.18-slim",
            'ADD snippet.py snippet.py',
            'CMD ["python", "snippet.py"]',
            "",
        ]
    )

    patched = DockerExecutor.patch_dockerfile(
        dockerfile,
        rewrite_python_installs=True,
        target_python="2.7.18",
        system_packages=["gfortran", "libopenblas-dev"],
    )

    assert "archive.debian.org/debian" in patched
    assert "archive.debian.org/debian-security" in patched
    assert "/buster-updates/d" in patched
    assert "Acquire::Check-Valid-Until" in patched
    assert patched.index("archive.debian.org/debian") < patched.index(
        "apt-get update && apt-get install -y --no-install-recommends gfortran libopenblas-dev"
    )


def test_patch_dockerfile_normalizes_hdf5_serial_layout_when_libhdf5_is_injected() -> None:
    dockerfile = "\n".join(
        [
            "FROM python:2.7.18-slim",
            'ADD snippet.py snippet.py',
            'CMD ["python", "snippet.py"]',
            "",
        ]
    )

    patched = DockerExecutor.patch_dockerfile(
        dockerfile,
        rewrite_python_installs=True,
        target_python="2.7.18",
        system_packages=["libhdf5-dev"],
    )

    assert "apt-get update && apt-get install -y --no-install-recommends libhdf5-dev" in patched
    assert "find /usr/lib -type d -path '*/hdf5/serial'" in patched
    assert 'ln -sf "$HDF5_SERIAL_LIBDIR/$lib" "/usr/local/lib/$lib"' in patched
    assert "ln -sfn /usr/include/hdf5/serial /usr/local/include/hdf5" in patched
    assert patched.index("apt-get update && apt-get install -y --no-install-recommends libhdf5-dev") < patched.index(
        "find /usr/lib -type d -path '*/hdf5/serial'"
    )


def test_prepare_benchmark_context_generates_cacheable_runtime_mounted_dockerfile(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    executor = DockerExecutor(settings)
    case_root = tmp_path / "case"
    case_root.mkdir()
    snippet_path = case_root / "snippet.py"
    snippet_path.write_text("print('ok')\n", encoding="utf-8")
    from agentic_python_dependency.state import BenchmarkCase, ResolvedDependency

    context = executor.prepare_benchmark_context(
        BenchmarkCase(case_id="case", root_dir=case_root, snippet_path=snippet_path, dockerfile_path=None),
        [ResolvedDependency(name="requests", version="2.32.3")],
        tmp_path / "artifact",
        "attempt-image",
    )

    dockerfile = context.dockerfile_path.read_text(encoding="utf-8")
    assert context.mount_workspace is True
    assert context.environment_cache_key
    assert context.image_tag.startswith("apdr-env-")
    assert "COPY . /workspace" not in dockerfile
    assert "COPY requirements.generated.txt /tmp/requirements.generated.txt" in dockerfile


def test_environment_cache_key_ignores_validation_command_and_changes_with_environment_inputs(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    executor = DockerExecutor(settings)

    key_one = executor._environment_cache_key(
        mode="benchmark-generated",
        target_python="3.12",
        docker_platform="linux/amd64",
        dependencies=["requests==2.32.3"],
        bootstrap_pins=["wheel==0.45.1"],
        system_packages=["pkg-config"],
        dockerfile_recipe="FROM python:3.12-slim\n",
    )
    key_two = executor._environment_cache_key(
        mode="benchmark-generated",
        target_python="3.12",
        docker_platform="linux/amd64",
        dependencies=["requests==2.32.3"],
        bootstrap_pins=["wheel==0.45.1"],
        system_packages=["pkg-config"],
        dockerfile_recipe="FROM python:3.12-slim\n",
    )
    key_three = executor._environment_cache_key(
        mode="benchmark-generated",
        target_python="3.12",
        docker_platform="linux/amd64",
        dependencies=["requests==2.31.0"],
        bootstrap_pins=["wheel==0.45.1"],
        system_packages=["pkg-config"],
        dockerfile_recipe="FROM python:3.12-slim\n",
    )

    assert key_one == key_two
    assert key_one != key_three


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
    assert ["docker", "rm", "-f", "pllm-test-case-1-attempt-0-run"] in calls
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


def test_execute_mounts_workspace_for_cacheable_generated_context(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    settings.keep_images = True
    executor = DockerExecutor(settings)
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="apdr-env-test",
        validation_command="python snippet.py",
        artifact_dir=tmp_path,
        mount_workspace=True,
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(context)

    assert result.run_succeeded is True
    run_command = next(command for command in calls if command[:2] == ["docker", "run"])
    assert "-v" in run_command
    assert f"{tmp_path.resolve()}:/workspace" in run_command


def test_execute_reuses_cached_image_and_skips_build(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    executor = DockerExecutor(settings)
    executor._store_cache_entry("cache-key", "apdr-env-test", mode="generated")
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="apdr-env-test",
        validation_command="python snippet.py",
        artifact_dir=tmp_path,
        environment_cache_key="cache-key",
        mount_workspace=True,
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        if command[:3] == ["docker", "image", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(context)

    assert result.build_succeeded is True
    assert result.build_skipped is True
    assert result.image_cache_hit is True
    assert not any(command[:2] == ["docker", "build"] for command in calls)
    assert ["docker", "image", "rm", "-f", "apdr-env-test"] in calls
    assert executor._load_cache_manifest()["entries"] == {}


def test_execute_honors_requested_docker_platform(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    executor = DockerExecutor(settings)
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-platform",
        validation_command=None,
        artifact_dir=tmp_path,
        docker_platform="linux/amd64",
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

    executor.execute(context)

    build_command = next(command for command in calls if command[:2] == ["docker", "build"])
    run_command = next(command for command in calls if command[:2] == ["docker", "run"])
    assert build_command[:4] == ["docker", "build", "--platform", "linux/amd64"]
    assert run_command[:7] == ["docker", "run", "--rm", "--name", "pllm-test-case-platform-attempt-0-run", "--platform", "linux/amd64"]


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


def test_python2_scientific_builds_get_extended_build_timeout(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    settings.build_timeout_seconds = 300
    executor = DockerExecutor(settings)
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-5",
        validation_command=None,
        artifact_dir=tmp_path,
        system_packages=["gfortran"],
        target_python="2.7.18",
        dependencies=["numpy==1.16.6", "pandas==0.24.2", "scipy==1.2.2"],
    )

    assert executor._build_timeout_seconds(context) == 900


def test_execute_uses_extended_build_timeout_for_python2_scientific_builds(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    settings.build_timeout_seconds = 300
    executor = DockerExecutor(settings)
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-6",
        validation_command=None,
        artifact_dir=tmp_path,
        system_packages=["gfortran"],
        target_python="2.7.18",
        dependencies=["numpy==1.16.6", "pandas==0.24.2", "scipy==1.2.2"],
    )
    timeouts: list[int] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["docker", "build"]:
            timeouts.append(int(kwargs["timeout"]))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        if command[:3] == ["docker", "image", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    executor.execute(context)

    assert timeouts == [900]


def test_execute_emits_activity_events(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    activity_events: list[tuple[str, int, str, str]] = []
    executor = DockerExecutor(
        settings,
        activity_callback=lambda case_id, attempt, kind, detail: activity_events.append((case_id, attempt, kind, detail)),
    )
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-7",
        validation_command="python snippet.py",
        artifact_dir=tmp_path,
        case_id="case-7",
        attempt_number=2,
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(command, 0, stdout="built\n", stderr="")
        if command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")
        if command[:3] == ["docker", "image", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    executor.execute(context)

    assert [event[2] for event in activity_events] == [
        "docker_build_start",
        "docker_build_finish",
        "docker_run_start",
        "docker_run_finish",
        "docker_image_cleanup",
    ]
    assert all(event[0] == "case-7" for event in activity_events)
    assert all(event[1] == 2 for event in activity_events)


def test_execute_persists_build_log_and_emits_failure_excerpt(monkeypatch, tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    activity_events: list[tuple[str, int, str, str]] = []
    executor = DockerExecutor(
        settings,
        activity_callback=lambda case_id, attempt, kind, detail: activity_events.append((case_id, attempt, kind, detail)),
    )
    context = PreparedExecutionContext(
        context_dir=tmp_path,
        dockerfile_path=tmp_path / "Dockerfile.generated",
        image_tag="pllm-test-case-8",
        validation_command=None,
        artifact_dir=tmp_path,
        case_id="case-8",
        attempt_number=1,
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="Step 1/2\n",
                stderr="ERROR: failed to solve: docker.io/library/python:3.12-slim: not found\n",
            )
        if command[:3] == ["docker", "image", "rm"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(context)

    assert result.build_succeeded is False
    assert (tmp_path / "build.log").read_text(encoding="utf-8").startswith("Step 1/2")
    assert any(
        kind == "docker_build_finish" and "failed to solve" in detail
        for _, _, kind, detail in activity_events
    )
