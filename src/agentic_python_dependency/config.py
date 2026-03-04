from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentic_python_dependency.presets import (
    GroupingMode,
    PresetName,
    PromptProfile,
    get_preset_config,
    normalize_preset,
    normalize_prompt_profile,
)

ModelProfileName = Literal[
    "gemma-moe",
    "gemma-moe-lite",
    "qwen35-9b",
    "qwen35-moe-lite",
    "gpt-oss-20b",
    "custom",
]

MODEL_PROFILE_DEFAULTS: dict[ModelProfileName, tuple[str, str]] = {
    "gemma-moe": ("gemma3:4b", "gemma3:12b"),
    "gemma-moe-lite": ("gemma3:1b", "gemma3:4b"),
    "qwen35-9b": ("qwen3.5:9b", "qwen3.5:9b"),
    "qwen35-moe-lite": ("qwen3.5:0.8b", "qwen3.5:4b"),
    "gpt-oss-20b": ("gpt-oss:20b", "gpt-oss:20b"),
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
    prompts_dir: Path
    ollama_base_url: str = "http://127.0.0.1:11434"
    docker_host: str = ""
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
    preset: PresetName = "optimized"
    prompt_profile: PromptProfile = "optimized"
    default_module_grouping: GroupingMode = "canonical"

    @classmethod
    def from_env(
        cls,
        project_root: Path | None = None,
        *,
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
    ) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        data_dir = root / "data"
        artifacts_dir = root / "artifacts" / "runs"
        benchmark_dir = data_dir / "benchmarks"
        pypi_cache_dir = data_dir / "pypi_cache"
        llm_cache_dir = data_dir / "llm_cache"
        prompts_dir = root / "src" / "agentic_python_dependency" / "prompts"
        preset = normalize_preset(preset_override or os.getenv("APD_PRESET"))
        preset_config = get_preset_config(preset)
        prompt_profile = normalize_prompt_profile(prompt_profile_override or os.getenv("APD_PROMPT_PROFILE"))
        env_disable_llm_cache = os.getenv("APD_DISABLE_LLM_CACHE", "").lower() in TRUE_VALUES
        env_use_moe = parse_optional_bool(os.getenv("APD_USE_MOE", ""))
        env_use_rag = parse_optional_bool(os.getenv("APD_USE_RAG", ""))
        env_use_langchain = parse_optional_bool(os.getenv("APD_USE_LANGCHAIN", ""))
        selected_model_profile = normalize_model_profile(model_profile_override or os.getenv("APD_MODEL_PROFILE"))
        default_extraction_model, default_reasoning_model = MODEL_PROFILE_DEFAULTS[selected_model_profile]
        extraction_model = extraction_model_override or os.getenv("APD_EXTRACTION_MODEL") or default_extraction_model
        reasoning_model = (
            runner_model_override
            or reasoning_model_override
            or os.getenv("APD_RUNNER_MODEL")
            or os.getenv("APD_REASONING_MODEL")
            or default_reasoning_model
        )
        version_model = version_model_override or os.getenv("APD_VERSION_MODEL") or reasoning_model
        repair_model = repair_model_override or os.getenv("APD_REPAIR_MODEL") or reasoning_model
        adjudication_model = adjudication_model_override or os.getenv("APD_ADJUDICATION_MODEL") or reasoning_model
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
            prompts_dir=prompts_dir,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            docker_host=os.getenv("DOCKER_HOST", ""),
            model_profile=effective_model_profile,
            use_moe=use_moe,
            use_rag=use_rag,
            use_langchain=use_langchain,
            extraction_model=extraction_model,
            reasoning_model=reasoning_model,
            version_model=version_model,
            repair_model=repair_model,
            adjudication_model=adjudication_model,
            keep_images=os.getenv("APD_KEEP_IMAGES", "").lower() in {"1", "true", "yes"},
            trace_llm=os.getenv("APD_TRACE_LLM", "").lower() in {"1", "true", "yes"},
            disable_llm_cache=env_disable_llm_cache if disable_llm_cache_override is None else disable_llm_cache_override,
            preset=preset,
            prompt_profile=prompt_profile or preset_config.prompt_profile,
            default_module_grouping=preset_config.reporting_grouping,
            max_attempts=preset_config.max_attempts,
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
