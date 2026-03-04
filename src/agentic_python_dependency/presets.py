from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PresetName = Literal["performance", "efficient", "optimized", "balanced", "thorough", "accuracy", "experimental"]
PromptProfile = Literal["paper", "optimized-lite", "optimized", "optimized-strict", "experimental-rag"]
GroupingMode = Literal["canonical", "raw"]
CompatibilityPolicyMode = Literal["essential", "curated", "full"]
VersionPromptMode = Literal["high_risk_only", "efficient", "optimized", "balanced", "thorough", "accuracy"]
RagMode = Literal["disabled", "pypi", "hybrid"]


COMPATIBILITY_SENSITIVE_PACKAGES = {
    "django",
    "sqlalchemy",
    "fabric",
    "boto3",
    "pygame",
    "matplotlib",
    "numpy",
    "pandas",
    "tensorflow",
    "lxml",
}


@dataclass(frozen=True, slots=True)
class PresetConfig:
    name: PresetName
    prompt_profile: PromptProfile
    max_attempts: int
    compatibility_policy: CompatibilityPolicyMode
    version_prompt_mode: VersionPromptMode
    allow_adjudication: bool
    allow_alias_retry: bool
    extract_llm_for_project_frameworks: bool
    accuracy_extract_cleanup: bool
    reporting_grouping: GroupingMode = "canonical"
    rag_mode: RagMode = "pypi"
    structured_prompting: bool = False
    candidate_plan_count: int = 1
    allow_candidate_fallback_before_repair: bool = False
    repair_cycle_limit: int = 0
    repo_evidence_enabled: bool = False


PRESET_CONFIGS: dict[PresetName, PresetConfig] = {
    "performance": PresetConfig(
        name="performance",
        prompt_profile="optimized-lite",
        max_attempts=2,
        compatibility_policy="essential",
        version_prompt_mode="high_risk_only",
        allow_adjudication=False,
        allow_alias_retry=False,
        extract_llm_for_project_frameworks=False,
        accuracy_extract_cleanup=False,
    ),
    "efficient": PresetConfig(
        name="efficient",
        prompt_profile="optimized-lite",
        max_attempts=3,
        compatibility_policy="curated",
        version_prompt_mode="efficient",
        allow_adjudication=True,
        allow_alias_retry=False,
        extract_llm_for_project_frameworks=False,
        accuracy_extract_cleanup=False,
    ),
    "optimized": PresetConfig(
        name="optimized",
        prompt_profile="optimized",
        max_attempts=3,
        compatibility_policy="curated",
        version_prompt_mode="optimized",
        allow_adjudication=True,
        allow_alias_retry=False,
        extract_llm_for_project_frameworks=False,
        accuracy_extract_cleanup=False,
    ),
    "balanced": PresetConfig(
        name="balanced",
        prompt_profile="optimized-strict",
        max_attempts=4,
        compatibility_policy="full",
        version_prompt_mode="balanced",
        allow_adjudication=True,
        allow_alias_retry=True,
        extract_llm_for_project_frameworks=True,
        accuracy_extract_cleanup=False,
    ),
    "thorough": PresetConfig(
        name="thorough",
        prompt_profile="optimized-strict",
        max_attempts=4,
        compatibility_policy="full",
        version_prompt_mode="thorough",
        allow_adjudication=True,
        allow_alias_retry=True,
        extract_llm_for_project_frameworks=True,
        accuracy_extract_cleanup=False,
    ),
    "accuracy": PresetConfig(
        name="accuracy",
        prompt_profile="optimized-strict",
        max_attempts=5,
        compatibility_policy="full",
        version_prompt_mode="accuracy",
        allow_adjudication=True,
        allow_alias_retry=True,
        extract_llm_for_project_frameworks=True,
        accuracy_extract_cleanup=True,
    ),
    "experimental": PresetConfig(
        name="experimental",
        prompt_profile="experimental-rag",
        max_attempts=6,
        compatibility_policy="full",
        version_prompt_mode="accuracy",
        allow_adjudication=True,
        allow_alias_retry=True,
        extract_llm_for_project_frameworks=True,
        accuracy_extract_cleanup=True,
        rag_mode="hybrid",
        structured_prompting=True,
        candidate_plan_count=3,
        allow_candidate_fallback_before_repair=True,
        repair_cycle_limit=2,
        repo_evidence_enabled=True,
    ),
}


def normalize_preset(value: str | None) -> PresetName:
    if not value:
        return "optimized"
    normalized = value.strip().lower()
    if normalized not in PRESET_CONFIGS:
        raise ValueError(f"Unsupported preset: {value}")
    return normalized  # type: ignore[return-value]


def normalize_prompt_profile(value: str | None) -> PromptProfile | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in {"paper", "optimized-lite", "optimized", "optimized-strict", "experimental-rag"}:
        raise ValueError(f"Unsupported prompt profile: {value}")
    return normalized  # type: ignore[return-value]


def get_preset_config(preset: str | None) -> PresetConfig:
    return PRESET_CONFIGS[normalize_preset(preset)]
