from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parents[2]
PLLM_DIR = ROOT_DIR / "tools" / "pllm"
EXECUTOR_SCRIPT = PLLM_DIR / "test_executor.py"

LineCallback = Callable[[str, "RunStats"], None]


@dataclass
class RunConfig:
    file: str
    model: str = "gemma2"
    base: str = "http://localhost:11434"
    temp: float = 0.7
    loop: int = 10
    search_range: int = 0
    rag: bool = True
    verbose: bool = False


@dataclass
class RunStats:
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    lines: int = 0
    build_successes: int = 0
    build_failures: int = 0
    completed_processes: int = 0
    last_line: str = ""
    python_versions_line: str = ""

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    details: str
    required: bool = True


def build_executor_command(config: RunConfig, python_executable: str | None = None) -> list[str]:
    python_cmd = python_executable or _default_python_command()
    return [
        python_cmd,
        str(EXECUTOR_SCRIPT),
        "-f",
        config.file,
        "-b",
        config.base,
        "-m",
        config.model,
        "-t",
        str(config.temp),
        "-l",
        str(config.loop),
        "-r",
        str(config.search_range),
        "-ra",
        "true" if config.rag else "false",
        *([] if not config.verbose else ["-v"]),
    ]


def stream_executor(
    config: RunConfig,
    line_callback: LineCallback | None = None,
    *,
    stop_event: threading.Event | None = None,
    python_executable: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, RunStats]:
    stats = RunStats()
    command = build_executor_command(config, python_executable=python_executable)

    process = subprocess.Popen(
        command,
        cwd=str(PLLM_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, **(env or {})},
    )

    def stop_watcher() -> None:
        if stop_event is None:
            return
        while process.poll() is None:
            if stop_event.is_set():
                process.terminate()
                return
            time.sleep(0.2)

    watcher = threading.Thread(target=stop_watcher, name="pllm-stop-watcher", daemon=True)
    watcher.start()

    return_code = 1
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            _update_stats(stats, line)
            if line_callback is not None:
                line_callback(line, stats)
            else:
                print(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return_code = 130
    finally:
        stats.finished_at = time.monotonic()
    return return_code, stats


def run_doctor(base_url: str = "http://localhost:11434") -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []

    checks.append(
        DoctorCheck(
            name="PLLM executor path",
            ok=EXECUTOR_SCRIPT.exists(),
            details=str(EXECUTOR_SCRIPT),
            required=True,
        )
    )

    python_ok = (3, 11) <= tuple(_python_version_tuple())
    checks.append(
        DoctorCheck(
            name="Python version",
            ok=python_ok,
            details=f"Detected Python {'.'.join(map(str, _python_version_tuple()))}",
            required=True,
        )
    )

    docker_path = shutil.which("docker")
    checks.append(
        DoctorCheck(
            name="Docker CLI",
            ok=docker_path is not None,
            details=docker_path or "docker not found on PATH",
            required=True,
        )
    )
    if docker_path:
        docker_ok, docker_msg = _run_command(["docker", "version", "--format", "{{.Server.Version}}"])
        checks.append(
            DoctorCheck(
                name="Docker daemon",
                ok=docker_ok,
                details=docker_msg,
                required=True,
            )
        )

    ollama_path = shutil.which("ollama")
    checks.append(
        DoctorCheck(
            name="Ollama CLI",
            ok=ollama_path is not None,
            details=ollama_path or "ollama not found on PATH",
            required=False,
        )
    )

    api_ok, api_msg = _check_ollama_api(base_url)
    checks.append(
        DoctorCheck(
            name="Ollama API",
            ok=api_ok,
            details=api_msg,
            required=True,
        )
    )

    prompt_toolkit_ok = _module_importable("prompt_toolkit")
    checks.append(
        DoctorCheck(
            name="prompt_toolkit (UI)",
            ok=prompt_toolkit_ok,
            details="installed" if prompt_toolkit_ok else "missing - install prompt-toolkit",
            required=False,
        )
    )

    dataset_archive = ROOT_DIR / "hard-gists.tar.gz"
    checks.append(
        DoctorCheck(
            name="Dataset archive",
            ok=dataset_archive.exists(),
            details=str(dataset_archive),
            required=False,
        )
    )

    benchmark_root = ROOT_DIR / "data" / "benchmarks" / "gistable" / "665d39a2bd82543d5196555f0801ef8fd4a3ee48"
    checks.append(
        DoctorCheck(
            name="Benchmark root",
            ok=benchmark_root.exists(),
            details=str(benchmark_root),
            required=False,
        )
    )
    all_gists = benchmark_root / "all-gists"
    checks.append(
        DoctorCheck(
            name="Benchmark all-gists",
            ok=all_gists.exists(),
            details=str(all_gists),
            required=False,
        )
    )
    dockerized_gists = benchmark_root / "dockerized-gists"
    checks.append(
        DoctorCheck(
            name="Benchmark dockerized-gists",
            ok=dockerized_gists.exists(),
            details=str(dockerized_gists),
            required=False,
        )
    )
    competition_filter = ROOT_DIR / "competition" / "competition-case-ids.txt"
    checks.append(
        DoctorCheck(
            name="Competition filter file",
            ok=competition_filter.exists(),
            details=str(competition_filter),
            required=False,
        )
    )

    return checks


def list_tools() -> list[str]:
    tools_root = ROOT_DIR / "tools"
    if not tools_root.exists():
        return []
    tools = []
    for path in tools_root.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            tools.append(path.name)
    tools.sort()
    return tools


def fetch_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    models = _fetch_ollama_models_from_api(base_url)
    if models:
        return models
    return _fetch_ollama_models_from_cli()


def doctor_passed(checks: list[DoctorCheck]) -> bool:
    return all(check.ok for check in checks if check.required)


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    lines = []
    for check in checks:
        level = "REQ" if check.required else "OPT"
        status = "OK" if check.ok else "FAIL"
        lines.append(f"[{status}] [{level}] {check.name}: {check.details}")
    return "\n".join(lines)


def _update_stats(stats: RunStats, line: str) -> None:
    stripped = line.strip()
    stats.lines += 1
    stats.last_line = stripped

    if "docker build complete!" in stripped:
        stats.build_successes += 1
    elif "docker build failed!" in stripped:
        stats.build_failures += 1
    elif "Processing completed without the timeout" in stripped:
        stats.completed_processes += 1
    elif stripped.startswith("[") and stripped.endswith("]") and "." in stripped:
        stats.python_versions_line = stripped


def _python_version_tuple() -> tuple[int, int, int]:
    return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def _default_python_command() -> str:
    if sys.executable:
        return sys.executable
    if shutil.which("python3"):
        return "python3"
    return "python"


def _run_command(command: list[str], timeout: int = 8) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive wrapper
        return False, str(exc)

    output = (completed.stdout or completed.stderr or "").strip()
    if not output:
        output = f"exit code {completed.returncode}"
    return completed.returncode == 0, output


def _check_ollama_api(base_url: str) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", [])
        model_count = len(models) if isinstance(models, list) else 0
        return True, f"reachable at {url} ({model_count} models visible)"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"{exc}"


def _fetch_ollama_models_from_api(base_url: str) -> list[str]:
    url = base_url.rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("models", [])
        if not isinstance(models, list):
            return []
        names = []
        for item in models:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    names.append(name)
        return sorted(set(names))
    except Exception:
        return []


def _fetch_ollama_models_from_cli() -> list[str]:
    ollama_path = shutil.which("ollama")
    if ollama_path is None:
        return []
    ok, output = _run_command([ollama_path, "list"])
    if not ok:
        return []
    names = []
    for idx, line in enumerate(output.splitlines()):
        if idx == 0 and "NAME" in line and "SIZE" in line:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        name = stripped.split()[0].strip()
        if name:
            names.append(name)
    return sorted(set(names))


def _module_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False
