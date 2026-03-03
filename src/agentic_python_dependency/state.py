from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict


Mode = Literal["gistable", "project"]


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    root_dir: Path
    snippet_path: Path
    dockerfile_path: Path
    initial_eval: str = ""
    final_eval: str = ""
    source_url: str = ""


@dataclass(slots=True)
class ProjectTarget:
    root_dir: Path
    validation_command: str
    python_files: list[Path] = field(default_factory=list)
    dockerfile_path: Path | None = None


@dataclass(slots=True)
class PackageCandidate:
    name: str
    source: str


@dataclass(slots=True)
class PackageVersionOptions:
    package: str
    versions: list[str]
    requires_python: dict[str, str] = field(default_factory=dict)
    upload_time: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ResolvedDependency:
    name: str
    version: str

    def pin(self) -> str:
        return f"{self.name}=={self.version}"


@dataclass(slots=True)
class AttemptRecord:
    attempt_number: int
    dependencies: list[str]
    image_tag: str
    build_succeeded: bool
    run_succeeded: bool
    exit_code: int | None
    error_category: str
    error_details: str
    validation_command: str | None
    wall_clock_seconds: float
    artifact_dir: str


@dataclass(slots=True)
class ExecutionOutcome:
    success: bool
    category: str
    message: str
    build_succeeded: bool
    run_succeeded: bool
    exit_code: int | None = None
    build_log: str = ""
    run_log: str = ""
    image_tag: str = ""
    dependency_retryable: bool = False


@dataclass(slots=True)
class BenchmarkSummary:
    run_id: str
    total_cases: int
    successes: int
    failures: int
    success_rate: float
    initial_import_errors: int
    final_import_errors: int
    mean_attempts_to_success: float
    mean_wall_clock_time: float
    total_wall_clock_time: float = 0.0
    total_wall_clock_human: str = "00:00:00"
    transitions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResolutionState(TypedDict, total=False):
    run_id: str
    mode: Mode
    case_id: str
    benchmark_case: BenchmarkCase
    project_target: ProjectTarget
    source_files: dict[str, str]
    extracted_imports: list[str]
    inferred_packages: list[str]
    unresolved_packages: list[str]
    version_options: list[PackageVersionOptions]
    selected_dependencies: list[ResolvedDependency]
    repaired_dependency_lines: list[str]
    attempt_records: list[AttemptRecord]
    prompt_history: dict[str, Any]
    model_outputs: dict[str, Any]
    artifact_dir: str
    current_attempt_dir: str
    current_attempt: int
    repair_stall_count: int
    current_validation_command: str
    current_runtime_profile: str
    generated_dockerfile: str
    generated_requirements: str
    prepared_execution_context: Any
    last_execution: ExecutionOutcome
    last_error_category: str
    last_error_details: str
    stop_reason: str
    final_result: dict[str, Any]
    target_python: str
