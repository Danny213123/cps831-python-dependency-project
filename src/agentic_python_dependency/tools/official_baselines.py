from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_python_dependency.config import Settings
from agentic_python_dependency.state import ResolvedDependency


class OfficialBaselineError(RuntimeError):
    pass


@dataclass(slots=True)
class OfficialBaselinePlan:
    target_python: str | None
    dependencies: list[ResolvedDependency]
    system_packages: list[str]
    implementation: str
    raw_payload: dict[str, Any]


def parse_pyego_dependency_json(payload: dict[str, Any]) -> OfficialBaselinePlan:
    dependencies: list[ResolvedDependency] = []
    for package in payload.get("python_packages", []):
        name = str(package.get("name", "")).strip()
        if not name:
            continue
        version = str(package.get("version", "")).strip()
        dependencies.append(ResolvedDependency(name=name, version="" if version in {"", "latest"} else version))

    system_packages: list[str] = []
    for package in payload.get("system_lib", []):
        if str(package.get("install_method", "")).strip().lower() != "apt":
            continue
        name = str(package.get("name", "")).strip()
        if name and name not in system_packages:
            system_packages.append(name)

    target_python = str(payload.get("python_version", "")).strip() or None
    return OfficialBaselinePlan(
        target_python=target_python,
        dependencies=sorted(dependencies, key=lambda dependency: dependency.name.lower()),
        system_packages=system_packages,
        implementation="official",
        raw_payload=payload,
    )


def parse_dockerfile_plan(dockerfile_text: str) -> OfficialBaselinePlan:
    target_python: str | None = None
    dependencies: list[ResolvedDependency] = []
    dependency_index: dict[str, ResolvedDependency] = {}
    system_packages: list[str] = []

    for line in dockerfile_text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("from python:") and target_python is None:
            target_python = stripped.split(":", 1)[1].split()[0].replace("-slim", "")
            continue
        if not lowered.startswith("run "):
            continue

        if "apt-get install" in lowered:
            install_segment = lowered.split("apt-get install", 1)[1]
            install_segment = install_segment.split("&&", 1)[0]
            tokens = shlex.split(install_segment)
            skip_next = False
            for token in tokens:
                if skip_next:
                    skip_next = False
                    continue
                if token in {"-y", "--yes", "--no-install-recommends", "install"}:
                    continue
                if token in {"-o", "--option"}:
                    skip_next = True
                    continue
                package = token.strip()
                if package and package not in system_packages:
                    system_packages.append(package)
            continue

        pip_match = re.search(r"\bpip(?:3)?\s+install\s+(.+)", stripped, re.IGNORECASE)
        if not pip_match:
            continue
        token_stream = shlex.split(pip_match.group(1))
        for token in token_stream:
            if token.startswith("-") or token.startswith("/tmp/requirements"):
                continue
            name, version = (token.split("==", 1) + [""])[:2] if "==" in token else (token, "")
            dependency_index[name.lower()] = ResolvedDependency(name=name, version=version)

    dependencies = sorted(dependency_index.values(), key=lambda dependency: dependency.name.lower())
    return OfficialBaselinePlan(
        target_python=target_python,
        dependencies=dependencies,
        system_packages=system_packages,
        implementation="official",
        raw_payload={
            "dockerfile": dockerfile_text,
            "target_python": target_python,
            "system_packages": system_packages,
            "dependencies": [dependency.pin() for dependency in dependencies],
        },
    )


def _decode_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _write_process_logs(artifact_dir: Path, stem: str, completed: subprocess.CompletedProcess[bytes]) -> None:
    (artifact_dir / f"{stem}.stdout.log").write_text(_decode_output(completed.stdout), encoding="utf-8")
    (artifact_dir / f"{stem}.stderr.log").write_text(_decode_output(completed.stderr), encoding="utf-8")


def run_pyego(settings: Settings, program_root: Path, artifact_dir: Path) -> OfficialBaselinePlan:
    script = settings.pyego_root / "PyEGo.py"
    if not script.exists():
        raise OfficialBaselineError(f"PyEGo entrypoint not found at {script}")

    output_path = artifact_dir / "pyego.dependency.json"
    command = [
        settings.pyego_python,
        str(script),
        "-t",
        "json",
        "-p",
        str(output_path),
        "-r",
        str(program_root),
    ]
    completed = subprocess.run(
        command,
        cwd=settings.pyego_root,
        capture_output=True,
        check=False,
    )
    _write_process_logs(artifact_dir, "pyego", completed)
    if completed.returncode != 0:
        raise OfficialBaselineError((_decode_output(completed.stderr) or _decode_output(completed.stdout) or "PyEGo failed").strip())
    if not output_path.exists():
        raise OfficialBaselineError("PyEGo did not write dependency json output")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OfficialBaselineError("PyEGo dependency json payload is not an object")
    return parse_pyego_dependency_json(payload)


def run_readpye(settings: Settings, program_path: Path, artifact_dir: Path) -> OfficialBaselinePlan:
    script = settings.readpye_root / "run.py"
    if not script.exists():
        raise OfficialBaselineError(f"ReadPyE entrypoint not found at {script}")
    if not settings.readpye_language_dir:
        raise OfficialBaselineError("ReadPyE language directory is not configured")

    output_path = artifact_dir / "readpye.Dockerfile"
    command = [
        settings.readpye_python,
        str(script),
        "-l",
        settings.readpye_language_dir,
        "-p",
        str(program_path),
        "-o",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=settings.readpye_root,
        capture_output=True,
        check=False,
    )
    _write_process_logs(artifact_dir, "readpye", completed)
    if completed.returncode != 0:
        raise OfficialBaselineError((_decode_output(completed.stderr) or _decode_output(completed.stdout) or "ReadPyE failed").strip())
    if not output_path.exists():
        raise OfficialBaselineError("ReadPyE did not write Dockerfile output")
    return parse_dockerfile_plan(output_path.read_text(encoding="utf-8"))
