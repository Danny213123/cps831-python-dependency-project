from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from agentic_python_dependency.presets import (
    ExperimentalBundleName,
    ExperimentalFeatureName,
    GroupingMode,
    PresetName,
    PromptProfile,
)
from agentic_python_dependency.config import ResolverName


Mode = Literal["gistable", "project"]


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    root_dir: Path
    snippet_path: Path
    dockerfile_path: Path | None = None
    case_source: str = "all-gists"
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
class InferenceCandidate:
    package: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    reason: str = ""
    accepted: bool = False


@dataclass(slots=True)
class CandidateDependency:
    name: str
    version: str

    def pin(self) -> str:
        return f"{self.name}=={self.version}" if self.version else self.name


@dataclass(slots=True)
class CandidatePlan:
    rank: int
    reason: str
    dependencies: list[CandidateDependency] = field(default_factory=list)


@dataclass(slots=True)
class PackageVersionOptions:
    package: str
    versions: list[str]
    requires_python: dict[str, str] = field(default_factory=dict)
    upload_time: dict[str, str] = field(default_factory=dict)
    policy_notes: list[str] = field(default_factory=list)
    requires_dist: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class ConflictNote:
    package: str
    related_package: str
    kind: str
    reason: str
    severity: str = "warning"


@dataclass(slots=True)
class ConstraintPack:
    target_python: str
    candidate_versions: dict[str, list[str]] = field(default_factory=dict)
    requires_python: dict[str, dict[str, str]] = field(default_factory=dict)
    requires_dist: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    conflict_notes: list[ConflictNote] = field(default_factory=list)
    package_conflict_matrix: dict[str, list[str]] = field(default_factory=dict)
    python_intersection: list[str] = field(default_factory=list)
    python_intersection_valid: bool = True
    conflict_precheck_failed: bool = False


@dataclass(slots=True)
class RepairStrategyRecord:
    strategy_type: str
    delta_from_previous: list[str] = field(default_factory=list)
    failure_category: str = ""
    failure_signature: str = ""
    result: str = ""


@dataclass(slots=True)
class RepairMemorySummary:
    recent_strategies: list[str] = field(default_factory=list)
    blocked_strategy_families: list[str] = field(default_factory=list)
    orthogonal_hints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetryDecision:
    category: str
    severity: str
    repair_allowed: bool
    candidate_fallback_allowed: bool
    repair_retry_budget: int
    native_retry_budget: int
    reason: str = ""


@dataclass(slots=True)
class ResolvedDependency:
    name: str
    version: str

    def pin(self) -> str:
        return f"{self.name}=={self.version}" if self.version else self.name


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
    started_at: str = ""
    finished_at: str = ""


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
    retry_severity: str = "terminal"


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
    resolver: str = "apd"
    preset: str = "optimized"
    prompt_profile: str = "optimized"
    model_profile: str = "gemma-moe"
    use_moe: bool = True
    use_rag: bool = True
    use_langchain: bool = True
    rag_mode: str = "pypi"
    structured_prompting: bool = False
    extraction_model: str = "gemma3:4b"
    runner_model: str = "gemma3:12b"
    version_model: str = "gemma3:12b"
    repair_model: str = "gemma3:12b"
    adjudication_model: str = "gemma3:12b"
    total_wall_clock_time: float = 0.0
    total_wall_clock_human: str = "00:00:00"
    transitions: dict[str, int] = field(default_factory=dict)
    dependency_reason_counts: dict[str, int] = field(default_factory=dict)
    experimental_case_count: int = 0
    candidate_plan_attempts: int = 0
    average_candidate_rank_selected: float = 0.0
    repair_cycle_count: int = 0
    structured_prompt_failures: int = 0
    experimental_bundle: str = "baseline"
    experimental_features: list[str] = field(default_factory=list)
    conflict_precheck_failures: int = 0
    python_constraint_blocked_cases: int = 0
    dynamic_alias_hit_rate: float = 0.0
    repair_memory_hit_rate: float = 0.0
    repeated_strategy_avoidance_count: int = 0
    version_negotiation_case_count: int = 0
    dynamic_import_case_count: int = 0
    feedback_memory_case_count: int = 0

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
    case_started_at: str
    case_finished_at: str
    last_execution: ExecutionOutcome
    last_error_category: str
    last_error_details: str
    stop_reason: str
    final_result: dict[str, Any]
    target_python: str
    benchmark_target_python: str
    inferred_target_python: str
    python_version_source: str
    resolver: ResolverName
    preset: PresetName
    prompt_profile: PromptProfile
    experimental_bundle: ExperimentalBundleName
    experimental_features: tuple[ExperimentalFeatureName, ...]
    dependency_reason: str
    candidate_provenance: dict[str, str]
    repair_outcome: str
    applied_compatibility_policy: dict[str, list[str]]
    version_selection_source: str
    inference_candidates: list[InferenceCandidate]
    repo_alias_candidates: dict[str, list[str]]
    dynamic_import_candidates: list[str]
    constraint_pack: ConstraintPack | None
    repair_memory_summary: RepairMemorySummary | None
    retry_decision: RetryDecision | None
    strategy_history: list[RepairStrategyRecord]
    feedback_memory_hits: int
    version_conflict_notes: list[ConflictNote]
    python_constraint_intersection: list[str]
    top_level_module_map: dict[str, list[str]]
    system_dependencies: list[str]
    resolver_implementation: str
    repo_evidence: dict[str, Any]
    pypi_evidence: dict[str, Any]
    rag_context: dict[str, Any]
    candidate_plans: list[CandidatePlan]
    remaining_candidate_plans: list[CandidatePlan]
    selected_candidate_plan: CandidatePlan | None
    selected_candidate_rank: int | None
    repair_cycle_count: int
    structured_outputs: dict[str, Any]
    experimental_path: bool
    structured_prompt_failures: int
