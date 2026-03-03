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

ModelProfileName = Literal["gemma-moe", "qwen35-9b", "gpt-oss-20b", "custom"]

MODEL_PROFILE_DEFAULTS: dict[ModelProfileName, tuple[str, str]] = {
    "gemma-moe": ("gemma3:4b", "gemma3:12b"),
    "qwen35-9b": ("qwen3.5:9b", "qwen3.5:9b"),
    "gpt-oss-20b": ("gpt-oss:20b", "gpt-oss:20b"),
    "custom": ("gemma3:4b", "gemma3:12b"),
}


def normalize_model_profile(value: str | None) -> ModelProfileName:
    if not value:
        return "gemma-moe"
    normalized = value.strip().lower()
    if normalized not in MODEL_PROFILE_DEFAULTS:
        raise ValueError(f"Unsupported model profile: {value}")
    return normalized  # type: ignore[return-value]


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
    extraction_model: str = "gemma3:4b"
    reasoning_model: str = "gemma3:12b"
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
        extraction_model_override: str | None = None,
        reasoning_model_override: str | None = None,
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
        env_disable_llm_cache = os.getenv("APD_DISABLE_LLM_CACHE", "").lower() in {"1", "true", "yes"}
        selected_model_profile = normalize_model_profile(model_profile_override or os.getenv("APD_MODEL_PROFILE"))
        default_extraction_model, default_reasoning_model = MODEL_PROFILE_DEFAULTS[selected_model_profile]
        extraction_model = extraction_model_override or os.getenv("APD_EXTRACTION_MODEL") or default_extraction_model
        reasoning_model = reasoning_model_override or os.getenv("APD_REASONING_MODEL") or default_reasoning_model
        effective_model_profile = (
            "custom"
            if extraction_model != default_extraction_model or reasoning_model != default_reasoning_model
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
            extraction_model=extraction_model,
            reasoning_model=reasoning_model,
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
