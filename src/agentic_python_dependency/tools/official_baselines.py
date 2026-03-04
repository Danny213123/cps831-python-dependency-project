from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentic_python_dependency.config import Settings
from agentic_python_dependency.state import ResolvedDependency


class OfficialBaselineError(RuntimeError):
    pass


def _probe_python_version(python_executable: str) -> tuple[int, int, int]:
    completed = subprocess.run(
        [
            python_executable,
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        details = (_decode_output(completed.stderr) or _decode_output(completed.stdout) or "version probe failed").strip()
        raise OfficialBaselineError(
            f"Failed to run APDR_PYEGO_PYTHON interpreter '{python_executable}' to probe version: {details}"
        )
    version_text = _decode_output(completed.stdout).strip()
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_text)
    if not match:
        raise OfficialBaselineError(
            f"Unable to parse APDR_PYEGO_PYTHON version output from '{python_executable}': {version_text or '<empty>'}"
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _read_pyego_config_values(config_path: Path) -> tuple[str | None, str | None, str | None, str | None]:
    if not config_path.exists():
        return None, None, None, None
    try:
        source = config_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None, None, None, None

    uri: str | None = None
    password: str | None = None
    username: str | None = None
    database: str | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in {"NEO4J_URI", "NEO4J_PWD", "NEO4J_USERNAME", "NEO4J_DATABASE"}:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if target.id == "NEO4J_URI":
                if value is not None:
                    uri = str(value).strip()
            elif target.id == "NEO4J_PWD":
                if value is not None:
                    password = str(value).strip()
            elif target.id == "NEO4J_USERNAME":
                if value is not None:
                    username = str(value).strip()
            elif target.id == "NEO4J_DATABASE":
                if value is not None:
                    database = str(value).strip()
    return uri, password, username, database


def _normalize_py2neo_uri(uri: str) -> str:
    if uri.startswith("neo4j+s://"):
        return "bolt+s://" + uri[len("neo4j+s://"):]
    if uri.startswith("neo4j://"):
        return "bolt://" + uri[len("neo4j://"):]
    return uri


def _probe_pyego_neo4j_connection(
    python_executable: str,
    *,
    uri: str,
    password: str | None,
    username: str | None,
    database: str | None,
) -> tuple[bool, str]:
    probe_code = (
        "import os\n"
        "from py2neo import Graph\n"
        "uri = os.environ['APDR_PYEGO_NEO4J_URI']\n"
        "password = os.environ.get('APDR_PYEGO_NEO4J_PWD')\n"
        "username = os.environ.get('APDR_PYEGO_NEO4J_USERNAME')\n"
        "database = os.environ.get('APDR_PYEGO_NEO4J_DATABASE')\n"
        "if password == '':\n"
        "    password = None\n"
        "kwargs = {}\n"
        "if username:\n"
        "    kwargs['user'] = username\n"
        "if password is not None:\n"
        "    kwargs['password'] = password\n"
        "if database:\n"
        "    kwargs['name'] = database\n"
        "graph = Graph(uri=uri, **kwargs)\n"
        "graph.run('RETURN 1').evaluate()\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["APDR_PYEGO_NEO4J_URI"] = uri
    env["APDR_PYEGO_NEO4J_PWD"] = password or ""
    env["APDR_PYEGO_NEO4J_USERNAME"] = username or ""
    env["APDR_PYEGO_NEO4J_DATABASE"] = database or ""
    completed = subprocess.run(
        [python_executable, "-c", probe_code],
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
    )
    if completed.returncode == 0:
        return True, f"neo4j reachable at {uri}"
    details = (_decode_output(completed.stderr) or _decode_output(completed.stdout) or "probe failed").strip()
    return False, details


def validate_pyego_runtime(settings: Settings) -> tuple[bool, str]:
    script = settings.pyego_root / "PyEGo.py"
    if not script.exists():
        return False, f"PyEGo entrypoint not found at {script}"

    try:
        major, minor, patch = _probe_python_version(settings.pyego_python)
    except OfficialBaselineError as exc:
        return False, str(exc)

    if (major, minor) > (3, 11):
        return (
            False,
            (
                "PyEGo requires APDR_PYEGO_PYTHON <= 3.11 because it depends on typed_ast.ast27; "
                f"current interpreter is {settings.pyego_python} ({major}.{minor}.{patch}). "
                "Create a Python 3.11 environment and point APDR_PYEGO_PYTHON to it."
            ),
        )

    typed_ast_probe = subprocess.run(
        [
            settings.pyego_python,
            "-c",
            "import typed_ast; from typed_ast import ast27; print('ok')",
        ],
        capture_output=True,
        check=False,
    )
    if typed_ast_probe.returncode != 0:
        details = (_decode_output(typed_ast_probe.stderr) or _decode_output(typed_ast_probe.stdout) or "import failed").strip()
        requirements_path = settings.pyego_root / "requirements.txt"
        return (
            False,
            (
                "PyEGo runtime is missing typed_ast.ast27. "
                f"Interpreter: {settings.pyego_python} ({major}.{minor}.{patch}). "
                f"Install dependencies with: {settings.pyego_python} -m pip install -r {requirements_path}. "
                f"Original error: {details}"
            ),
        )

    config_path = settings.pyego_root / "config.py"
    neo4j_uri, neo4j_password, neo4j_username, neo4j_database = _read_pyego_config_values(config_path)
    if not neo4j_uri or "YOUR NEO4J URI" in neo4j_uri:
        return (
            False,
            (
                "PyEGo Neo4j configuration is missing. "
                f"Set NEO4J_URI in {config_path} (example: bolt://localhost:7687) and start Neo4j."
            ),
        )

    neo4j_uri = _normalize_py2neo_uri(neo4j_uri)
    parsed = urlparse(neo4j_uri)
    if not parsed.scheme:
        neo4j_uri = f"bolt://{neo4j_uri}"
    neo4j_ok, neo4j_detail = _probe_pyego_neo4j_connection(
        settings.pyego_python,
        uri=neo4j_uri,
        password=neo4j_password,
        username=neo4j_username,
        database=neo4j_database,
    )
    if not neo4j_ok:
        return (
            False,
            (
                "PyEGo cannot reach Neo4j. "
                f"Configured URI: {neo4j_uri}. "
                f"Details: {neo4j_detail}. "
                "Start Neo4j, load PyKG, and verify credentials in external/PyEGo/config.py."
            ),
        )

    return True, f"ready ({settings.pyego_python} on Python {major}.{minor}.{patch})"


def ensure_pyego_runtime(settings: Settings) -> None:
    ok, detail = validate_pyego_runtime(settings)
    if not ok:
        raise OfficialBaselineError(detail)


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
    ensure_pyego_runtime(settings)
    script = settings.pyego_root / "PyEGo.py"

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
