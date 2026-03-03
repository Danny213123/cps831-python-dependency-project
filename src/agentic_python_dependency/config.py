from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agentic_python_dependency.presets import (
    GroupingMode,
    PresetName,
    PromptProfile,
    get_preset_config,
    normalize_preset,
    normalize_prompt_profile,
)


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
            extraction_model=os.getenv("APD_EXTRACTION_MODEL", "gemma3:4b"),
            reasoning_model=os.getenv("APD_REASONING_MODEL", "gemma3:12b"),
            keep_images=os.getenv("APD_KEEP_IMAGES", "").lower() in {"1", "true", "yes"},
            trace_llm=os.getenv("APD_TRACE_LLM", "").lower() in {"1", "true", "yes"},
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
