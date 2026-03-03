from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_python_dependency.config import Settings
from agentic_python_dependency.state import BenchmarkCase, ProjectTarget, ResolvedDependency


@dataclass(slots=True)
class PreparedExecutionContext:
    context_dir: Path
    dockerfile_path: Path
    image_tag: str
    validation_command: str | None
    artifact_dir: Path


@dataclass(slots=True)
class DockerExecutionResult:
    build_succeeded: bool
    run_succeeded: bool
    exit_code: int | None
    build_log: str
    run_log: str
    image_tag: str
    wall_clock_seconds: float


class DockerExecutor:
    SYSTEM_PACKAGE_MAP = {
        "pygame": [
            "libsdl1.2-dev",
            "libsdl-image1.2-dev",
            "libsdl-mixer1.2-dev",
            "libsdl-ttf2.0-dev",
            "libsmpeg-dev",
            "libportmidi-dev",
            "libavformat-dev",
            "libswscale-dev",
            "libjpeg-dev",
            "libfreetype6-dev",
            "pkg-config",
        ],
    }

    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _decode_output(output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output

    def _format_timeout_log(
        self,
        operation: str,
        timeout_seconds: int,
        stdout: str | bytes | None,
        stderr: str | bytes | None,
    ) -> str:
        parts = [f"{operation} timed out after {timeout_seconds} seconds."]
        rendered_stdout = self._decode_output(stdout).strip()
        rendered_stderr = self._decode_output(stderr).strip()
        if rendered_stdout:
            parts.append(rendered_stdout)
        if rendered_stderr:
            parts.append(rendered_stderr)
        return "\n\n".join(parts).strip() + "\n"

    @staticmethod
    def render_requirements(dependencies: list[ResolvedDependency]) -> str:
        return "\n".join(dep.pin() for dep in dependencies) + ("\n" if dependencies else "")

    @staticmethod
    def bootstrap_pins(dependencies: list[ResolvedDependency]) -> list[str]:
        preferred = ("setuptools", "wheel", "cython", "numpy")
        dependency_map = {dependency.name.lower(): dependency.pin() for dependency in dependencies}
        return [dependency_map[name] for name in preferred if name in dependency_map]

    @classmethod
    def system_packages(cls, dependencies: list[ResolvedDependency]) -> list[str]:
        packages: list[str] = []
        seen: set[str] = set()
        for dependency in dependencies:
            for package in cls.SYSTEM_PACKAGE_MAP.get(dependency.name.lower(), []):
                if package not in seen:
                    seen.add(package)
                    packages.append(package)
        return packages

    @staticmethod
    def _is_python_install_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped.upper().startswith("RUN "):
            return False
        lowered = stripped.lower()
        if any(package_manager in lowered for package_manager in ("apt-get", "apk add", "yum install", "dnf install")):
            return False
        return bool(
            re.search(r"\bpip(?:3)?\s+install\b", lowered)
            or re.search(r'["\']pip(?:3)?["\']\s*,\s*["\']install["\']', lowered)
        )

    @staticmethod
    def patch_dockerfile(
        dockerfile_text: str,
        requirements_file: str = "requirements.generated.txt",
        *,
        rewrite_python_installs: bool = False,
        target_python: str = "3.12",
        bootstrap_pins: list[str] | None = None,
        system_packages: list[str] | None = None,
    ) -> str:
        lines = dockerfile_text.splitlines()
        rewritten_lines: list[str] = []
        for line in lines:
            if rewrite_python_installs and DockerExecutor._is_python_install_line(line):
                continue
            rewritten_lines.append(line)

        injection: list[str] = []
        if system_packages:
            injection.append(
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                + " ".join(system_packages)
                + " && rm -rf /var/lib/apt/lists/*"
            )
        if target_python.startswith("2"):
            injection.append("RUN pip install --no-cache-dir --upgrade 'pip<21' 'setuptools<45' 'wheel<0.35'")
        if bootstrap_pins:
            injection.append(f"RUN pip install --no-cache-dir {' '.join(bootstrap_pins)}")
        injection.extend(
            [
                f"COPY {requirements_file} /tmp/{requirements_file}",
                f"RUN pip install --no-cache-dir -r /tmp/{requirements_file}",
            ]
        )

        if rewrite_python_installs:
            for index, line in enumerate(rewritten_lines):
                stripped = line.strip()
                upper = stripped.upper()
                if upper.startswith("ADD ") or (
                    upper.startswith("COPY ") and requirements_file not in stripped and "/tmp/" not in stripped
                ):
                    patched = rewritten_lines[:index] + injection + rewritten_lines[index:]
                    return "\n".join(patched) + "\n"

        for index, line in enumerate(rewritten_lines):
            upper = line.strip().upper()
            if upper.startswith("CMD ") or upper.startswith("ENTRYPOINT "):
                patched = rewritten_lines[:index] + injection + rewritten_lines[index:]
                return "\n".join(patched) + "\n"
        return "\n".join(rewritten_lines + injection) + "\n"

    def _copy_project_tree(self, source_root: Path, destination_root: Path) -> None:
        ignore = shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build")
        for child in source_root.iterdir():
            target = destination_root / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True, ignore=ignore)
            else:
                shutil.copy2(child, target)

    def prepare_benchmark_context(
        self,
        case: BenchmarkCase,
        dependencies: list[ResolvedDependency],
        artifact_dir: Path,
        image_tag: str,
        target_python: str = "3.12",
        validation_command: str | None = None,
    ) -> PreparedExecutionContext:
        context_dir = artifact_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        for child in case.root_dir.iterdir():
            target = context_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
        requirements_path = context_dir / "requirements.generated.txt"
        dockerfile_path = context_dir / "Dockerfile.generated"
        requirements_path.write_text(self.render_requirements(dependencies), encoding="utf-8")
        dockerfile_text = case.dockerfile_path.read_text(encoding="utf-8")
        dockerfile_path.write_text(
            self.patch_dockerfile(
                dockerfile_text,
                rewrite_python_installs=True,
                target_python=target_python,
                bootstrap_pins=self.bootstrap_pins(dependencies),
                system_packages=self.system_packages(dependencies),
            ),
            encoding="utf-8",
        )
        return PreparedExecutionContext(context_dir, dockerfile_path, image_tag, validation_command, artifact_dir)

    def prepare_project_context(
        self,
        target: ProjectTarget,
        dependencies: list[ResolvedDependency],
        artifact_dir: Path,
        image_tag: str,
    ) -> PreparedExecutionContext:
        context_dir = artifact_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        self._copy_project_tree(target.root_dir, context_dir)
        requirements_path = context_dir / "requirements.generated.txt"
        dockerfile_path = context_dir / "Dockerfile.generated"
        requirements_path.write_text(self.render_requirements(dependencies), encoding="utf-8")
        if target.dockerfile_path and target.dockerfile_path.exists():
            dockerfile_text = target.dockerfile_path.read_text(encoding="utf-8")
            dockerfile_path.write_text(
                self.patch_dockerfile(
                    dockerfile_text,
                    bootstrap_pins=self.bootstrap_pins(dependencies),
                    system_packages=self.system_packages(dependencies),
                ),
                encoding="utf-8",
            )
        else:
            dockerfile_path.write_text(
                "\n".join(
                    [
                        "FROM python:3.12-slim",
                        "WORKDIR /workspace",
                        "COPY . /workspace",
                        "COPY requirements.generated.txt /tmp/requirements.generated.txt",
                        "RUN pip install --no-cache-dir -r /tmp/requirements.generated.txt",
                        'CMD ["python", "-m", "compileall", "."]',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        return PreparedExecutionContext(
            context_dir=context_dir,
            dockerfile_path=dockerfile_path,
            image_tag=image_tag,
            validation_command=target.validation_command,
            artifact_dir=artifact_dir,
        )

    def execute(self, context: PreparedExecutionContext) -> DockerExecutionResult:
        env = os.environ.copy()
        if self.settings.docker_host:
            env["DOCKER_HOST"] = self.settings.docker_host
        else:
            env.pop("DOCKER_HOST", None)

        build_cmd = [
            "docker",
            "build",
            "-f",
            str(context.dockerfile_path),
            "-t",
            context.image_tag,
            str(context.context_dir),
        ]
        build_started = time.monotonic()
        try:
            build = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.build_timeout_seconds,
                env=env,
                check=False,
            )
            build_log = build.stdout + build.stderr
            build_succeeded = build.returncode == 0
            exit_code = build.returncode
        except subprocess.TimeoutExpired as exc:
            build = None
            build_log = self._format_timeout_log(
                "docker build",
                self.settings.build_timeout_seconds,
                exc.stdout,
                exc.stderr,
            )
            build_succeeded = False
            exit_code = 124
        run_log = ""
        run_success = False

        if build_succeeded:
            container_name = f"{context.image_tag}-run"
            run_cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "-e",
                "MPLBACKEND=Agg",
                "-e",
                "SDL_VIDEODRIVER=dummy",
                "-e",
                "QT_QPA_PLATFORM=offscreen",
                "--memory",
                self.settings.memory_limit,
                "--cpus",
                self.settings.cpu_limit,
                context.image_tag,
            ]
            if context.validation_command:
                run_cmd.extend(["sh", "-lc", context.validation_command])
            try:
                run = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.settings.run_timeout_seconds,
                    env=env,
                    check=False,
                )
                run_log = run.stdout + run.stderr
                exit_code = run.returncode
                run_success = run.returncode == 0
            except subprocess.TimeoutExpired as exc:
                run_log = self._format_timeout_log(
                    "docker run",
                    self.settings.run_timeout_seconds,
                    exc.stdout,
                    exc.stderr,
                )
                exit_code = 124
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=False,
                )

        if not self.settings.keep_images:
            subprocess.run(
                ["docker", "image", "rm", "-f", context.image_tag],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

        return DockerExecutionResult(
            build_succeeded=build_succeeded,
            run_succeeded=run_success,
            exit_code=exit_code,
            build_log=build_log,
            run_log=run_log,
            image_tag=context.image_tag,
            wall_clock_seconds=time.monotonic() - build_started,
        )


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
