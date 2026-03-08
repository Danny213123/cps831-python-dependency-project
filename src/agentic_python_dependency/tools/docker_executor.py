from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agentic_python_dependency.config import Settings
from agentic_python_dependency.state import BenchmarkCase, ProjectTarget, ResolvedDependency


@dataclass(slots=True)
class PreparedExecutionContext:
    context_dir: Path
    dockerfile_path: Path
    image_tag: str
    validation_command: str | None
    artifact_dir: Path
    docker_platform: str = ""
    system_packages: list[str] = field(default_factory=list)
    bootstrap_pins: list[str] = field(default_factory=list)
    target_python: str = "3.12"
    dependencies: list[str] = field(default_factory=list)
    case_id: str = ""
    attempt_number: int = 0
    environment_cache_key: str = ""
    mount_workspace: bool = False


@dataclass(slots=True)
class DockerExecutionResult:
    build_succeeded: bool
    run_succeeded: bool
    exit_code: int | None
    build_log: str
    run_log: str
    image_tag: str
    wall_clock_seconds: float
    build_wall_clock_seconds: float = 0.0
    run_wall_clock_seconds: float = 0.0
    build_skipped: bool = False
    image_cache_hit: bool = False


class DockerExecutor:
    _BUILDKIT_SYNTAX = "# syntax=docker/dockerfile:1.7"
    _CACHE_MANIFEST_NAME = "docker-image-cache.json"
    _CACHE_MANIFEST_VERSION = 1
    _CACHE_LOCK = threading.RLock()
    LONG_BUILD_PACKAGES = {"numpy", "pandas", "scipy", "pymc3", "theano", "h5py"}
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

    def __init__(
        self,
        settings: Settings,
        activity_callback: Callable[[str, int, str, str], None] | None = None,
    ):
        self.settings = settings
        self.activity_callback = activity_callback
        self._cache_manifest_path = self.settings.data_dir / "loadouts" / self._CACHE_MANIFEST_NAME

    def _emit_activity(self, context: PreparedExecutionContext, *, kind: str, detail: str) -> None:
        if self.activity_callback is None or not context.case_id:
            return
        self.activity_callback(
            context.case_id,
            attempt=context.attempt_number,
            kind=kind,
            detail=detail,
        )

    @staticmethod
    def _activity_excerpt(log_text: str, *, limit: int = 160) -> str:
        lines = [line.strip() for line in log_text.splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if any(token in lowered for token in ("error", "failed", "no matching", "denied", "not found", "requires")):
                return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."
        if not lines:
            return ""
        line = lines[-1]
        return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."

    @staticmethod
    def _write_attempt_log(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _decode_output(output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output

    @staticmethod
    def _stable_lines(items: list[str] | None) -> list[str]:
        if not items:
            return []
        deduped: dict[str, str] = {}
        for item in items:
            candidate = str(item or "").strip()
            if not candidate:
                continue
            deduped[candidate.lower()] = candidate
        return [deduped[key] for key in sorted(deduped)]

    @classmethod
    def _ensure_buildkit_syntax_header(cls, dockerfile_text: str) -> str:
        lines = dockerfile_text.splitlines()
        if lines and lines[0].strip().lower().startswith("# syntax="):
            return dockerfile_text if dockerfile_text.endswith("\n") else dockerfile_text + "\n"
        rendered = "\n".join([cls._BUILDKIT_SYNTAX, *lines]).rstrip() + "\n"
        return rendered

    @staticmethod
    def _cache_mount_pip(command: str) -> str:
        return f"RUN --mount=type=cache,target=/root/.cache/pip {command}"

    @classmethod
    def _environment_cache_key(
        cls,
        *,
        mode: str,
        target_python: str,
        docker_platform: str,
        dependencies: list[str],
        bootstrap_pins: list[str],
        system_packages: list[str],
        dockerfile_recipe: str,
    ) -> str:
        payload = json.dumps(
            {
                "mode": mode,
                "target_python": target_python,
                "docker_platform": docker_platform,
                "dependencies": cls._stable_lines(dependencies),
                "bootstrap_pins": cls._stable_lines(bootstrap_pins),
                "system_packages": cls._stable_lines(system_packages),
                "dockerfile_recipe": dockerfile_recipe,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _image_tag_for_cache_key(cache_key: str) -> str:
        return f"apdr-env-{cache_key[:24]}"

    def _load_cache_manifest(self) -> dict[str, object]:
        with self._CACHE_LOCK:
            if not self._cache_manifest_path.exists():
                return {"version": self._CACHE_MANIFEST_VERSION, "entries": {}}
            try:
                payload = json.loads(self._cache_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"version": self._CACHE_MANIFEST_VERSION, "entries": {}}
            if not isinstance(payload, dict):
                return {"version": self._CACHE_MANIFEST_VERSION, "entries": {}}
            entries = payload.get("entries", {})
            if not isinstance(entries, dict):
                entries = {}
            return {
                "version": self._CACHE_MANIFEST_VERSION,
                "entries": entries,
            }

    def _store_cache_entry(self, cache_key: str, image_tag: str, *, mode: str) -> None:
        with self._CACHE_LOCK:
            payload = self._load_cache_manifest()
            entries = dict(payload.get("entries", {}))
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            entries[cache_key] = {
                "image_tag": image_tag,
                "mode": mode,
                "last_used_at": now,
            }
            payload["entries"] = entries
            self._cache_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._cache_manifest_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            temp_path.replace(self._cache_manifest_path)

    @staticmethod
    def _docker_image_exists(image_tag: str, env: dict[str, str]) -> bool:
        probe = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True,
            env=env,
            check=False,
        )
        return probe.returncode == 0

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

    def _build_timeout_seconds(self, context: PreparedExecutionContext) -> int:
        timeout_seconds = self.settings.build_timeout_seconds
        dependency_names = {
            dependency.split("==", 1)[0].strip().lower()
            for dependency in context.dependencies
            if dependency.strip()
        }
        if context.target_python.startswith("2") and dependency_names & self.LONG_BUILD_PACKAGES:
            return max(timeout_seconds, timeout_seconds * 3)
        if context.system_packages and dependency_names & {"numpy", "pandas", "scipy", "h5py"}:
            return max(timeout_seconds, timeout_seconds * 2)
        return timeout_seconds

    @staticmethod
    def render_requirements(dependencies: list[ResolvedDependency]) -> str:
        rendered = sorted((dep.pin() for dep in dependencies), key=str.lower)
        return "\n".join(rendered) + ("\n" if rendered else "")

    @staticmethod
    def _normalize_requirement_name(requirement: str) -> str:
        candidate = requirement.strip()
        if not candidate:
            return ""
        for separator in ("[", "<", ">", "=", "!", "~"):
            if separator in candidate:
                candidate = candidate.split(separator, 1)[0]
        return candidate.strip().replace("-", "_").lower()

    @classmethod
    def bootstrap_pins(
        cls,
        dependencies: list[ResolvedDependency],
        *,
        target_python: str = "3.12",
        extra_bootstrap_pins: list[str] | None = None,
    ) -> list[str]:
        preferred = ("setuptools", "wheel", "cython", "numpy")
        dependency_map = {dependency.name.lower(): dependency.pin() for dependency in dependencies}
        pins = [dependency_map[name] for name in preferred if name in dependency_map]
        dependency_names = {dependency.name.lower() for dependency in dependencies}
        if target_python.startswith("2") and dependency_names & cls.LONG_BUILD_PACKAGES and "cython" not in dependency_map:
            pins.append("Cython<3")
        for requirement in extra_bootstrap_pins or []:
            normalized_requirement = cls._normalize_requirement_name(requirement)
            if not normalized_requirement:
                continue
            if any(
                cls._normalize_requirement_name(existing) == normalized_requirement
                for existing in pins
            ):
                continue
            pins.append(requirement)
        return cls._stable_lines(pins)

    @classmethod
    def system_packages(cls, dependencies: list[ResolvedDependency]) -> list[str]:
        packages: list[str] = []
        seen: set[str] = set()
        for dependency in dependencies:
            for package in cls.SYSTEM_PACKAGE_MAP.get(dependency.name.lower(), []):
                if package not in seen:
                    seen.add(package)
                    packages.append(package)
        return cls._stable_lines(packages)

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
    def _legacy_python_apt_bootstrap(target_python: str) -> list[str]:
        if not target_python.startswith("2"):
            return []
        return [
            "RUN if [ -f /etc/apt/sources.list ]; then "
            "sed -ri 's|deb.debian.org/debian|archive.debian.org/debian|g; "
            "s|security.debian.org/debian-security|archive.debian.org/debian-security|g; "
            "/buster-updates/d' /etc/apt/sources.list; fi",
            "RUN printf 'Acquire::Check-Valid-Until \"false\";\\n"
            "Acquire::AllowInsecureRepositories \"true\";\\n' > /etc/apt/apt.conf.d/99apdr-archive",
        ]

    @staticmethod
    def _system_package_post_install_lines(system_packages: list[str] | None = None) -> list[str]:
        if not system_packages:
            return []
        normalized = {package.strip().lower() for package in system_packages if package.strip()}
        if "libhdf5-dev" not in normalized and "libhdf5-serial-dev" not in normalized:
            return []
        return [
            "RUN HDF5_SERIAL_LIBDIR=\"$(find /usr/lib -type d -path '*/hdf5/serial' | head -n 1)\"; "
            "if [ -n \"$HDF5_SERIAL_LIBDIR\" ]; then "
            "for lib in libhdf5.so libhdf5_hl.so libhdf5_cpp.so; do "
            "if [ -e \"$HDF5_SERIAL_LIBDIR/$lib\" ]; then ln -sf \"$HDF5_SERIAL_LIBDIR/$lib\" \"/usr/local/lib/$lib\"; fi; "
            "done; "
            "fi; "
            "if [ -d /usr/include/hdf5/serial ]; then mkdir -p /usr/local/include && ln -sfn /usr/include/hdf5/serial /usr/local/include/hdf5; fi; "
            "ldconfig || true",
        ]

    @classmethod
    def _dependency_injection_lines(
        cls,
        requirements_file: str,
        *,
        target_python: str,
        bootstrap_pins: list[str] | None = None,
        system_packages: list[str] | None = None,
    ) -> list[str]:
        injection: list[str] = []
        if system_packages:
            injection.extend(cls._legacy_python_apt_bootstrap(target_python))
            injection.append(
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                + " ".join(cls._stable_lines(system_packages))
                + " && rm -rf /var/lib/apt/lists/*"
            )
            injection.extend(cls._system_package_post_install_lines(system_packages))
        if target_python.startswith("2"):
            injection.append(
                cls._cache_mount_pip("pip install --upgrade 'pip<21' 'setuptools<45' 'wheel<0.35'")
            )
        if bootstrap_pins:
            rendered_pins = " ".join(shlex.quote(pin) for pin in cls._stable_lines(bootstrap_pins))
            injection.append(cls._cache_mount_pip(f"pip install {rendered_pins}"))
        injection.extend(
            [
                f"COPY {requirements_file} /tmp/{requirements_file}",
                cls._cache_mount_pip(f"pip install -r /tmp/{requirements_file}"),
            ]
        )
        return injection

    @staticmethod
    def patch_dockerfile(
        dockerfile_text: str,
        requirements_file: str = "requirements.generated.txt",
        *,
        rewrite_python_installs: bool = False,
        rewrite_base_python: bool = False,
        target_python: str = "3.12",
        bootstrap_pins: list[str] | None = None,
        system_packages: list[str] | None = None,
    ) -> str:
        dockerfile_text = DockerExecutor._ensure_buildkit_syntax_header(dockerfile_text)
        lines = dockerfile_text.splitlines()
        rewritten_lines: list[str] = []
        for line in lines:
            if rewrite_python_installs and DockerExecutor._is_python_install_line(line):
                continue
            if rewrite_base_python and line.strip().lower().startswith("from python:"):
                suffix = ""
                lowered = line.strip().lower()
                if "-slim" in lowered:
                    suffix = "-slim"
                line = f"FROM python:{target_python}{suffix}"
                rewrite_base_python = False
            rewritten_lines.append(line)

        injection = DockerExecutor._dependency_injection_lines(
            requirements_file,
            target_python=target_python,
            bootstrap_pins=bootstrap_pins,
            system_packages=system_packages,
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

    @classmethod
    def _render_generated_dockerfile(
        cls,
        *,
        target_python: str,
        default_command: str,
        bootstrap_pins: list[str],
        system_packages: list[str],
    ) -> str:
        injection = cls._dependency_injection_lines(
            "requirements.generated.txt",
            target_python=target_python,
            bootstrap_pins=bootstrap_pins,
            system_packages=system_packages,
        )
        return cls._ensure_buildkit_syntax_header(
            "\n".join(
                [
                    f"FROM python:{target_python}-slim",
                    "WORKDIR /workspace",
                    *injection,
                    default_command,
                    "",
                ]
            )
        )

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
        extra_system_packages: list[str] | None = None,
        extra_bootstrap_pins: list[str] | None = None,
        case_id: str = "",
        attempt_number: int = 0,
    ) -> PreparedExecutionContext:
        context_dir = artifact_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        if case.root_dir.exists():
            for child in case.root_dir.iterdir():
                target = context_dir / child.name
                if child.is_dir():
                    shutil.copytree(child, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, target)
        elif case.snippet_path.exists():
            shutil.copy2(case.snippet_path, context_dir / "snippet.py")
        requirements_path = context_dir / "requirements.generated.txt"
        dockerfile_path = context_dir / "Dockerfile.generated"
        requirements_path.write_text(self.render_requirements(dependencies), encoding="utf-8")
        merged_system_packages = self.system_packages(dependencies)
        for package in extra_system_packages or []:
            if package not in merged_system_packages:
                merged_system_packages.append(package)
        merged_system_packages = self._stable_lines(merged_system_packages)
        bootstrap_pins = self.bootstrap_pins(
            dependencies,
            target_python=target_python,
            extra_bootstrap_pins=extra_bootstrap_pins,
        )
        if case.dockerfile_path is not None and case.dockerfile_path.exists():
            dockerfile_text = case.dockerfile_path.read_text(encoding="utf-8")
            generated_dockerfile = self.patch_dockerfile(
                dockerfile_text,
                rewrite_python_installs=True,
                rewrite_base_python=True,
                target_python=target_python,
                bootstrap_pins=bootstrap_pins,
                system_packages=merged_system_packages,
            )
            dockerfile_path.write_text(
                generated_dockerfile,
                encoding="utf-8",
            )
            environment_cache_key = ""
            image_ref = image_tag
            mount_workspace = False
        else:
            generated_dockerfile = self._render_generated_dockerfile(
                target_python=target_python,
                default_command='CMD ["python", "snippet.py"]',
                bootstrap_pins=bootstrap_pins,
                system_packages=merged_system_packages,
            )
            dockerfile_path.write_text(generated_dockerfile, encoding="utf-8")
            environment_cache_key = self._environment_cache_key(
                mode="benchmark-generated",
                target_python=target_python,
                docker_platform=self.settings.benchmark_platform,
                dependencies=[dependency.pin() for dependency in dependencies],
                bootstrap_pins=bootstrap_pins,
                system_packages=merged_system_packages,
                dockerfile_recipe=generated_dockerfile,
            )
            image_ref = self._image_tag_for_cache_key(environment_cache_key)
            mount_workspace = True
        return PreparedExecutionContext(
            context_dir=context_dir,
            dockerfile_path=dockerfile_path,
            image_tag=image_ref,
            validation_command=validation_command,
            artifact_dir=artifact_dir,
            docker_platform=self.settings.benchmark_platform,
            system_packages=merged_system_packages,
            bootstrap_pins=bootstrap_pins,
            target_python=target_python,
            dependencies=[dependency.pin() for dependency in dependencies],
            case_id=case_id,
            attempt_number=attempt_number,
            environment_cache_key=environment_cache_key,
            mount_workspace=mount_workspace,
        )

    def prepare_project_context(
        self,
        target: ProjectTarget,
        dependencies: list[ResolvedDependency],
        artifact_dir: Path,
        image_tag: str,
        extra_system_packages: list[str] | None = None,
        extra_bootstrap_pins: list[str] | None = None,
        case_id: str = "",
        attempt_number: int = 0,
    ) -> PreparedExecutionContext:
        context_dir = artifact_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        self._copy_project_tree(target.root_dir, context_dir)
        merged_system_packages = self.system_packages(dependencies)
        for package in extra_system_packages or []:
            if package not in merged_system_packages:
                merged_system_packages.append(package)
        merged_system_packages = self._stable_lines(merged_system_packages)
        bootstrap_pins = self.bootstrap_pins(
            dependencies,
            target_python="3.12",
            extra_bootstrap_pins=extra_bootstrap_pins,
        )
        requirements_path = context_dir / "requirements.generated.txt"
        dockerfile_path = context_dir / "Dockerfile.generated"
        requirements_path.write_text(self.render_requirements(dependencies), encoding="utf-8")
        if target.dockerfile_path and target.dockerfile_path.exists():
            dockerfile_text = target.dockerfile_path.read_text(encoding="utf-8")
            generated_dockerfile = self.patch_dockerfile(
                dockerfile_text,
                bootstrap_pins=bootstrap_pins,
                system_packages=merged_system_packages,
            )
            dockerfile_path.write_text(
                generated_dockerfile,
                encoding="utf-8",
            )
            environment_cache_key = ""
            image_ref = image_tag
            mount_workspace = False
        else:
            generated_dockerfile = self._render_generated_dockerfile(
                target_python="3.12",
                default_command='CMD ["python", "-m", "compileall", "."]',
                bootstrap_pins=bootstrap_pins,
                system_packages=merged_system_packages,
            )
            dockerfile_path.write_text(generated_dockerfile, encoding="utf-8")
            environment_cache_key = self._environment_cache_key(
                mode="project-generated",
                target_python="3.12",
                docker_platform="",
                dependencies=[dependency.pin() for dependency in dependencies],
                bootstrap_pins=bootstrap_pins,
                system_packages=merged_system_packages,
                dockerfile_recipe=generated_dockerfile,
            )
            image_ref = self._image_tag_for_cache_key(environment_cache_key)
            mount_workspace = True
        return PreparedExecutionContext(
            context_dir=context_dir,
            dockerfile_path=dockerfile_path,
            image_tag=image_ref,
            validation_command=target.validation_command,
            artifact_dir=artifact_dir,
            system_packages=merged_system_packages,
            bootstrap_pins=bootstrap_pins,
            dependencies=[dependency.pin() for dependency in dependencies],
            case_id=case_id,
            attempt_number=attempt_number,
            environment_cache_key=environment_cache_key,
            mount_workspace=mount_workspace,
        )

    def execute(self, context: PreparedExecutionContext) -> DockerExecutionResult:
        env = os.environ.copy()
        env.setdefault("DOCKER_BUILDKIT", "1")
        if self.settings.docker_host:
            env["DOCKER_HOST"] = self.settings.docker_host
        else:
            env.pop("DOCKER_HOST", None)

        build_log = ""
        build_succeeded = False
        build_skipped = False
        image_cache_hit = False
        build_exit_code: int | None = None
        build_elapsed = 0.0
        build_cmd = [
            "docker",
            "build",
        ]
        if context.docker_platform:
            build_cmd.extend(["--platform", context.docker_platform])
        build_cmd.extend(
            [
                "-f",
                str(context.dockerfile_path),
                "-t",
                context.image_tag,
                str(context.context_dir),
            ]
        )
        build_timeout_seconds = self._build_timeout_seconds(context)
        build_started = time.monotonic()
        self._emit_activity(
            context,
            kind="docker_build_start",
            detail=f"Starting docker build for attempt {context.attempt_number} ({context.image_tag}).",
        )
        if context.environment_cache_key and self._docker_image_exists(context.image_tag, env):
            build_skipped = True
            image_cache_hit = True
            build_succeeded = True
            build_exit_code = 0
            build_elapsed = time.monotonic() - build_started
            build_log = (
                f"docker build skipped; reused cached image {context.image_tag} "
                f"for environment {context.environment_cache_key}.\n"
            )
            self._write_attempt_log(context.artifact_dir / "build.log", build_log)
            self._store_cache_entry(
                context.environment_cache_key,
                context.image_tag,
                mode="generated",
            )
            self._emit_activity(
                context,
                kind="docker_build_finish",
                detail=(
                    f"Docker build skipped via environment cache in {time.monotonic() - build_started:.1f}s."
                ),
            )
        else:
            try:
                build = subprocess.run(
                    build_cmd,
                    capture_output=True,
                    timeout=build_timeout_seconds,
                    env=env,
                    check=False,
                )
                build_log = self._decode_output(build.stdout) + self._decode_output(build.stderr)
                build_succeeded = build.returncode == 0
                build_exit_code = build.returncode
                build_elapsed = time.monotonic() - build_started
                self._write_attempt_log(context.artifact_dir / "build.log", build_log)
                failure_excerpt = self._activity_excerpt(build_log)
                if build_succeeded and context.environment_cache_key:
                    self._store_cache_entry(
                        context.environment_cache_key,
                        context.image_tag,
                        mode="generated",
                    )
                self._emit_activity(
                    context,
                    kind="docker_build_finish",
                    detail=(
                        f"Docker build {'succeeded' if build_succeeded else 'failed'} "
                        f"in {time.monotonic() - build_started:.1f}s."
                        + (f" {failure_excerpt}" if failure_excerpt and not build_succeeded else "")
                    ),
                )
            except subprocess.TimeoutExpired as exc:
                build_log = self._format_timeout_log(
                    "docker build",
                    build_timeout_seconds,
                    exc.stdout,
                    exc.stderr,
                )
                build_succeeded = False
                build_exit_code = 124
                build_elapsed = time.monotonic() - build_started
                self._write_attempt_log(context.artifact_dir / "build.log", build_log)
                self._emit_activity(
                    context,
                    kind="docker_build_finish",
                    detail=f"Docker build timed out after {build_timeout_seconds}s.",
                )
        run_log = ""
        run_success = False
        run_elapsed = 0.0

        if build_succeeded:
            container_suffix = context.case_id or f"attempt-{context.attempt_number}"
            container_suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "-", container_suffix)[:48]
            container_name = f"{context.image_tag}-{container_suffix}-run"
            run_cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
            ]
            if context.docker_platform:
                run_cmd.extend(["--platform", context.docker_platform])
            if context.mount_workspace:
                run_cmd.extend(["-v", f"{context.context_dir.resolve()}:/workspace"])
            run_cmd.extend(
                [
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
            )
            if context.validation_command:
                run_cmd.extend(["sh", "-lc", context.validation_command])
            run_started = time.monotonic()
            self._emit_activity(
                context,
                kind="docker_run_start",
                detail=f"Starting container validation for attempt {context.attempt_number}.",
            )
            try:
                run = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    timeout=self.settings.run_timeout_seconds,
                    env=env,
                    check=False,
                )
                run_log = self._decode_output(run.stdout) + self._decode_output(run.stderr)
                build_exit_code = run.returncode
                run_success = run.returncode == 0
                run_elapsed = time.monotonic() - run_started
                self._write_attempt_log(context.artifact_dir / "run.log", run_log)
                failure_excerpt = self._activity_excerpt(run_log)
                self._emit_activity(
                    context,
                    kind="docker_run_finish",
                    detail=(
                        f"Container run {'succeeded' if run_success else 'failed'} "
                        f"in {time.monotonic() - run_started:.1f}s."
                        + (f" {failure_excerpt}" if failure_excerpt and not run_success else "")
                    ),
                )
            except subprocess.TimeoutExpired as exc:
                run_log = self._format_timeout_log(
                    "docker run",
                    self.settings.run_timeout_seconds,
                    exc.stdout,
                    exc.stderr,
                )
                build_exit_code = 124
                run_elapsed = time.monotonic() - run_started
                self._write_attempt_log(context.artifact_dir / "run.log", run_log)
                self._emit_activity(
                    context,
                    kind="docker_run_finish",
                    detail=f"Container run timed out after {self.settings.run_timeout_seconds}s.",
                )
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    env=env,
                    check=False,
                )

        if not self.settings.keep_images and not context.environment_cache_key:
            subprocess.run(
                ["docker", "image", "rm", "-f", context.image_tag],
                capture_output=True,
                env=env,
                check=False,
            )

        return DockerExecutionResult(
            build_succeeded=build_succeeded,
            run_succeeded=run_success,
            exit_code=build_exit_code,
            build_log=build_log,
            run_log=run_log,
            image_tag=context.image_tag,
            wall_clock_seconds=build_elapsed + run_elapsed,
            build_wall_clock_seconds=build_elapsed,
            run_wall_clock_seconds=run_elapsed,
            build_skipped=build_skipped,
            image_cache_hit=image_cache_hit,
        )


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
