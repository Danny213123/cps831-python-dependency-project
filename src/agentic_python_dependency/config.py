from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentic_python_dependency.presets import (
    GroupingMode,
    PresetName,
    PromptProfile,
    RagMode,
    ResearchBundleName,
    ResearchFeatureName,
    get_preset_config,
    normalize_preset,
    normalize_prompt_profile,
    normalize_research_bundle,
    resolve_research_features,
)

ResolverName = Literal["apdr", "pyego", "readpye"]
BenchmarkCaseSource = Literal["all-gists", "dockerized-gists", "competition-run"]

ModelProfileName = Literal[
    "gemma-moe",
    "gemma-moe-lite",
    "qwen35-9b",
    "qwen35-moe-lite",
    "gpt-oss-20b",
    "mistral-nemo-12b",
    "custom",
]

MODEL_PROFILE_DEFAULTS: dict[ModelProfileName, tuple[str, str]] = {
    "gemma-moe": ("gemma3:4b", "gemma3:12b"),
    "gemma-moe-lite": ("gemma3:1b", "gemma3:4b"),
    "qwen35-9b": ("qwen3.5:9b", "qwen3.5:9b"),
    "qwen35-moe-lite": ("qwen3.5:0.8b", "qwen3.5:4b"),
    "gpt-oss-20b": ("gpt-oss:20b", "gpt-oss:20b"),
    "mistral-nemo-12b": ("mistral-nemo:12b", "mistral-nemo:12b"),
    "custom": ("gemma3:4b", "gemma3:12b"),
}

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def normalize_model_profile(value: str | None) -> ModelProfileName:
    if not value:
        return "gemma-moe"
    normalized = value.strip().lower()
    if normalized not in MODEL_PROFILE_DEFAULTS:
        raise ValueError(f"Unsupported model profile: {value}")
    return normalized  # type: ignore[return-value]


def normalize_resolver(value: str | None) -> ResolverName:
    if not value:
        return "apdr"
    normalized = value.strip().lower()
    if normalized not in {"apdr", "pyego", "readpye"}:
        raise ValueError(f"Unsupported resolver: {value}")
    return normalized  # type: ignore[return-value]


def normalize_benchmark_case_source(value: str | None) -> BenchmarkCaseSource:
    if not value:
        return "all-gists"
    normalized = value.strip().lower()
    if normalized not in {"all-gists", "dockerized-gists", "competition-run"}:
        raise ValueError(f"Unsupported benchmark case source: {value}")
    return normalized  # type: ignore[return-value]


def parse_path_list(value: str) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw in value.split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        paths.append(Path(candidate).expanduser().resolve())
    return tuple(paths)


def default_competition_result_csvs(project_root: Path) -> tuple[Path, ...]:
    candidates = [
        project_root / "data" / "benchmarks" / "gistable" / "competition" / "hard-gists-l10-r1-10-final.csv",
        Path.home() / "Downloads" / "hard-gists-l10-r1-10-final.csv",
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            deduped.append(resolved)
    return tuple(deduped)


def parse_optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


@dataclass(slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    benchmark_dir: Path
    pypi_cache_dir: Path
    llm_cache_dir: Path
    package_metadata_dir: Path
    workspace_memory_dir: Path
    prompts_dir: Path
    external_tools_dir: Path = Path(".")
    pyego_root: Path = Path(".")
    readpye_root: Path = Path(".")
    pyego_python: str = sys.executable
    readpye_python: str = sys.executable
    readpye_language_dir: str = ""
    competition_result_csvs: tuple[Path, ...] = ()
    competition_case_ids_file: Path = Path("competition/competition-case-ids.txt")
    ollama_base_url: str = "http://127.0.0.1:11434"
    docker_host: str = ""
    benchmark_platform: str = "linux/amd64"
    resolver: ResolverName = "apdr"
    model_profile: ModelProfileName = "gemma-moe"
    use_moe: bool = True
    use_rag: bool = True
    use_langchain: bool = True
    extraction_model: str = "gemma3:4b"
    reasoning_model: str = "gemma3:12b"
    version_model: str = "gemma3:12b"
    repair_model: str = "gemma3:12b"
    adjudication_model: str = "gemma3:12b"
    temperature: float = 0.0
    num_ctx: int = 8192
    max_attempts: int = 4
    build_timeout_seconds: int = 300
    run_timeout_seconds: int = 60
    memory_limit: str = "2g"
    cpu_limit: str = "2"
    keep_images: bool = False
    trace_llm: bool = False
    disable_llm_cache: bool = False
    benchmark_ref: str = "665d39a2bd82543d5196555f0801ef8fd4a3ee48"
    benchmark_case_source: BenchmarkCaseSource = "all-gists"
    preset: PresetName = "optimized"
    prompt_profile: PromptProfile = "optimized"
    default_module_grouping: GroupingMode = "canonical"
    rag_mode: RagMode = "pypi"
    structured_prompting: bool = False
    candidate_plan_count: int = 1
    allow_candidate_fallback_before_repair: bool = False
    repair_cycle_limit: int = 0
    repo_evidence_enabled: bool = False
    research_bundle: ResearchBundleName = "baseline"
    research_features: tuple[ResearchFeatureName, ...] = ()

    @classmethod
    def from_env(
        cls,
        project_root: Path | None = None,
        *,
        resolver_override: str | None = None,
        preset_override: str | None = None,
        prompt_profile_override: str | None = None,
        model_profile_override: str | None = None,
        use_moe_override: bool | None = None,
        use_rag_override: bool | None = None,
        use_langchain_override: bool | None = None,
        extraction_model_override: str | None = None,
        runner_model_override: str | None = None,
        reasoning_model_override: str | None = None,
        version_model_override: str | None = None,
        repair_model_override: str | None = None,
        adjudication_model_override: str | None = None,
        disable_llm_cache_override: bool | None = None,
        research_bundle_override: str | None = None,
        research_feature_overrides: list[str] | None = None,
        research_feature_disable_overrides: list[str] | None = None,
        benchmark_case_source_override: str | None = None,
        competition_result_csvs_override: list[str] | None = None,
        competition_case_ids_file_override: str | None = None,
    ) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        data_dir = root / "data"
        artifacts_dir = root / "artifacts" / "runs"
        benchmark_dir = data_dir / "benchmarks"
        pypi_cache_dir = data_dir / "pypi_cache"
        llm_cache_dir = data_dir / "llm_cache"
        package_metadata_dir = data_dir / "package_metadata"
        workspace_memory_dir = data_dir / "research_memory"
        prompts_dir = root / "src" / "agentic_python_dependency" / "prompts"
        external_tools_dir = root / "external"
        pyego_root = Path(os.getenv("APDR_PYEGO_ROOT", str(external_tools_dir / "PyEGo"))).resolve()
        readpye_root = Path(os.getenv("APDR_READPYE_ROOT", str(external_tools_dir / "ReadPyE"))).resolve()
        pyego_python = os.getenv("APDR_PYEGO_PYTHON", sys.executable)
        readpye_python = os.getenv("APDR_READPYE_PYTHON", sys.executable)
        readpye_language_dir = os.getenv("APDR_READPYE_LANGDIR", "")
        resolver = normalize_resolver(resolver_override or os.getenv("APDR_RESOLVER"))
        benchmark_case_source = normalize_benchmark_case_source(
            benchmark_case_source_override or os.getenv("APDR_BENCHMARK_CASE_SOURCE")
        )
        competition_result_csvs = (
            tuple(Path(value).expanduser().resolve() for value in competition_result_csvs_override)
            if competition_result_csvs_override
            else parse_path_list(os.getenv("APDR_COMPETITION_RESULT_CSVS", ""))
        )
        if not competition_result_csvs:
            competition_result_csvs = default_competition_result_csvs(root)
        competition_case_ids_file = Path(
            competition_case_ids_file_override
            or os.getenv("APDR_COMPETITION_CASE_IDS_FILE")
            or str(root / "competition" / "competition-case-ids.txt")
        ).expanduser().resolve()
        preset = normalize_preset(preset_override or os.getenv("APDR_PRESET"))
        preset_config = get_preset_config(preset)
        prompt_profile = normalize_prompt_profile(prompt_profile_override or os.getenv("APDR_PROMPT_PROFILE"))
        research_bundle = normalize_research_bundle(
            research_bundle_override or os.getenv("APDR_RESEARCH_BUNDLE") or preset_config.research_bundle
        )
        env_enabled_features = [
            value.strip()
            for value in os.getenv("APDR_RESEARCH_FEATURES", "").split(",")
            if value.strip()
        ]
        env_disabled_features = [
            value.strip()
            for value in os.getenv("APDR_NO_RESEARCH_FEATURES", "").split(",")
            if value.strip()
        ]
        research_features = resolve_research_features(
            research_bundle,
            enabled=[*(research_feature_overrides or []), *env_enabled_features],
            disabled=[*(research_feature_disable_overrides or []), *env_disabled_features],
        )
        env_disable_llm_cache = os.getenv("APDR_DISABLE_LLM_CACHE", "").lower() in TRUE_VALUES
        env_use_moe = parse_optional_bool(os.getenv("APDR_USE_MOE", ""))
        env_use_rag = parse_optional_bool(os.getenv("APDR_USE_RAG", ""))
        env_use_langchain = parse_optional_bool(os.getenv("APDR_USE_LANGCHAIN", ""))
        selected_model_profile = normalize_model_profile(model_profile_override or os.getenv("APDR_MODEL_PROFILE"))
        default_extraction_model, default_reasoning_model = MODEL_PROFILE_DEFAULTS[selected_model_profile]
        extraction_model = extraction_model_override or os.getenv("APDR_EXTRACTION_MODEL") or default_extraction_model
        reasoning_model = (
            runner_model_override
            or reasoning_model_override
            or os.getenv("APDR_RUNNER_MODEL")
            or os.getenv("APDR_REASONING_MODEL")
            or default_reasoning_model
        )
        version_model = version_model_override or os.getenv("APDR_VERSION_MODEL") or reasoning_model
        repair_model = repair_model_override or os.getenv("APDR_REPAIR_MODEL") or reasoning_model
        adjudication_model = adjudication_model_override or os.getenv("APDR_ADJUDICATION_MODEL") or reasoning_model
        use_moe = True if env_use_moe is None else env_use_moe
        if use_moe_override is not None:
            use_moe = use_moe_override
        use_rag = True if env_use_rag is None else env_use_rag
        if use_rag_override is not None:
            use_rag = use_rag_override
        use_langchain = True if env_use_langchain is None else env_use_langchain
        if use_langchain_override is not None:
            use_langchain = use_langchain_override
        effective_model_profile = (
            "custom"
            if extraction_model != default_extraction_model
            or reasoning_model != default_reasoning_model
            or version_model != reasoning_model
            or repair_model != reasoning_model
            or adjudication_model != reasoning_model
            else selected_model_profile
        )
        settings = cls(
            project_root=root,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            benchmark_dir=benchmark_dir,
            pypi_cache_dir=pypi_cache_dir,
            llm_cache_dir=llm_cache_dir,
            package_metadata_dir=package_metadata_dir,
            workspace_memory_dir=workspace_memory_dir,
            prompts_dir=prompts_dir,
            external_tools_dir=external_tools_dir,
            pyego_root=pyego_root,
            readpye_root=readpye_root,
            pyego_python=pyego_python,
            readpye_python=readpye_python,
            readpye_language_dir=readpye_language_dir,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            docker_host=os.getenv("DOCKER_HOST", ""),
            benchmark_platform=(os.getenv("APDR_BENCHMARK_PLATFORM", "linux/amd64") or "").strip(),
            resolver=resolver,
            benchmark_case_source=benchmark_case_source,
            competition_result_csvs=competition_result_csvs,
            competition_case_ids_file=competition_case_ids_file,
            model_profile=effective_model_profile,
            use_moe=use_moe,
            use_rag=use_rag,
            use_langchain=use_langchain,
            extraction_model=extraction_model,
            reasoning_model=reasoning_model,
            version_model=version_model,
            repair_model=repair_model,
            adjudication_model=adjudication_model,
            keep_images=os.getenv("APDR_KEEP_IMAGES", "").lower() in {"1", "true", "yes"},
            trace_llm=os.getenv("APDR_TRACE_LLM", "").lower() in {"1", "true", "yes"},
            disable_llm_cache=env_disable_llm_cache if disable_llm_cache_override is None else disable_llm_cache_override,
            preset=preset,
            prompt_profile=prompt_profile or preset_config.prompt_profile,
            default_module_grouping=preset_config.reporting_grouping,
            max_attempts=preset_config.max_attempts,
            rag_mode=preset_config.rag_mode,
            structured_prompting=preset_config.structured_prompting,
            candidate_plan_count=preset_config.candidate_plan_count,
            allow_candidate_fallback_before_repair=preset_config.allow_candidate_fallback_before_repair,
            repair_cycle_limit=preset_config.repair_cycle_limit,
            repo_evidence_enabled=preset_config.repo_evidence_enabled,
            research_bundle=research_bundle,
            research_features=research_features if preset == "research" else (),
        )
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.artifacts_dir,
            self.benchmark_dir,
            self.pypi_cache_dir,
            self.pypi_cache_dir / "raw",
            self.llm_cache_dir,
            self.package_metadata_dir,
            self.package_metadata_dir / "raw",
            self.package_metadata_dir / "parsed",
            self.workspace_memory_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def prompt_template_dir(self) -> Path:
        candidate = self.prompts_dir / self.prompt_profile
        return candidate if candidate.exists() else self.prompts_dir

    def stage_model(self, stage: str) -> str:
        if not self.use_moe:
            return self.reasoning_model
        if stage == "extract":
            return self.extraction_model
        if stage == "version":
            return self.version_model
        if stage == "repair":
            return self.repair_model
        if stage == "adjudicate":
            return self.adjudication_model
        return self.reasoning_model

    def active_stage_models(self) -> dict[str, str]:
        return {
            "extract": self.stage_model("extract"),
            "runner": self.reasoning_model,
            "version": self.stage_model("version"),
            "repair": self.stage_model("repair"),
            "adjudicate": self.stage_model("adjudicate"),
        }

    def research_feature_enabled(self, feature: ResearchFeatureName | str) -> bool:
        return feature in self.research_features

    def effective_runtime_config(self) -> dict[str, object]:
        stage_models = self.active_stage_models()
        return {
            "effective_model_profile": self.model_profile,
            "effective_rag_mode": self.rag_mode,
            "effective_structured_prompting": self.structured_prompting,
            "effective_repair_cycle_limit": self.repair_cycle_limit,
            "effective_candidate_fallback_before_repair": self.allow_candidate_fallback_before_repair,
            "model_profile": self.model_profile,
            "use_moe": self.use_moe,
            "use_rag": self.use_rag,
            "use_langchain": self.use_langchain,
            "extraction_model": stage_models["extract"],
            "runner_model": stage_models["runner"],
            "version_model": stage_models["version"],
            "repair_model": stage_models["repair"],
            "adjudication_model": stage_models["adjudicate"],
            "rag_mode": self.rag_mode,
            "structured_prompting": self.structured_prompting,
            "candidate_plan_count": self.candidate_plan_count,
            "allow_candidate_fallback_before_repair": self.allow_candidate_fallback_before_repair,
            "repair_cycle_limit": self.repair_cycle_limit,
            "repo_evidence_enabled": self.repo_evidence_enabled,
            "benchmark_platform": self.benchmark_platform,
            "research_bundle": self.research_bundle,
            "research_features": list(self.research_features),
        }

    def apply_runtime_config(self, payload: Mapping[str, object]) -> None:
        def _get(*keys: str) -> object | None:
            for key in keys:
                if key in payload:
                    return payload[key]
            return None

        model_profile = _get("effective_model_profile", "model_profile")
        if isinstance(model_profile, str) and model_profile in MODEL_PROFILE_DEFAULTS:
            self.model_profile = model_profile

        for key in ("use_moe", "use_rag", "use_langchain", "structured_prompting", "repo_evidence_enabled"):
            value = _get(key)
            if isinstance(value, bool):
                setattr(self, key, value)

        rag_mode = _get("effective_rag_mode", "rag_mode")
        if isinstance(rag_mode, str) and rag_mode:
            self.rag_mode = rag_mode

        candidate_plan_count = _get("candidate_plan_count")
        if isinstance(candidate_plan_count, int):
            self.candidate_plan_count = candidate_plan_count

        repair_cycle_limit = _get("effective_repair_cycle_limit", "repair_cycle_limit")
        if isinstance(repair_cycle_limit, int):
            self.repair_cycle_limit = repair_cycle_limit

        candidate_fallback = _get(
            "effective_candidate_fallback_before_repair",
            "allow_candidate_fallback_before_repair",
        )
        if isinstance(candidate_fallback, bool):
            self.allow_candidate_fallback_before_repair = candidate_fallback

        benchmark_platform = _get("benchmark_platform")
        if isinstance(benchmark_platform, str):
            self.benchmark_platform = benchmark_platform.strip()

        for key, attr in (
            ("extraction_model", "extraction_model"),
            ("runner_model", "reasoning_model"),
            ("reasoning_model", "reasoning_model"),
            ("version_model", "version_model"),
            ("repair_model", "repair_model"),
            ("adjudication_model", "adjudication_model"),
        ):
            value = _get(key)
            if isinstance(value, str) and value.strip():
                setattr(self, attr, value.strip())

        research_bundle = _get("research_bundle")
        if isinstance(research_bundle, str) and research_bundle:
            self.research_bundle = research_bundle

        research_features = _get("research_features")
        if isinstance(research_features, (list, tuple)):
            self.research_features = tuple(
                str(feature) for feature in research_features if isinstance(feature, str)
            )

    def runtime_config_mismatches(self, payload: Mapping[str, object]) -> list[str]:
        current = self.effective_runtime_config()
        mismatches: list[str] = []
        for key, current_value in current.items():
            if key not in payload:
                continue
            saved_value = payload[key]
            if isinstance(current_value, list):
                if not isinstance(saved_value, (list, tuple)) or list(saved_value) != current_value:
                    mismatches.append(f"{key}: saved={saved_value!r} current={current_value!r}")
                continue
            if isinstance(current_value, bool):
                if bool(saved_value) != current_value:
                    mismatches.append(f"{key}: saved={saved_value!r} current={current_value!r}")
                continue
            if isinstance(current_value, int):
                try:
                    normalized_saved = int(saved_value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    mismatches.append(f"{key}: saved={saved_value!r} current={current_value!r}")
                    continue
                if normalized_saved != current_value:
                    mismatches.append(f"{key}: saved={saved_value!r} current={current_value!r}")
                continue
            if str(saved_value) != str(current_value):
                mismatches.append(f"{key}: saved={saved_value!r} current={current_value!r}")
        return mismatches

    def validate_benchmark_runtime(self) -> list[str]:
        errors: list[str] = []
        if self.preset == "research" and self.research_bundle == "full":
            if not self.use_rag:
                errors.append("research/full requires RAG to remain enabled")
            if self.rag_mode != "hybrid":
                errors.append("research/full requires rag_mode=hybrid")
            if not self.structured_prompting:
                errors.append("research/full requires structured_prompting=true")
            if self.repair_cycle_limit <= 0:
                errors.append("research/full requires repair_cycle_limit>0")
            if not self.allow_candidate_fallback_before_repair:
                errors.append("research/full requires candidate fallback before repair")
        return errors
