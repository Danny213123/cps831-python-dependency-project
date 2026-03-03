from pathlib import Path

from agentic_python_dependency.config import Settings


def test_settings_from_env_uses_preset_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="performance")

    assert settings.preset == "performance"
    assert settings.prompt_profile == "optimized-lite"
    assert settings.max_attempts == 2


def test_settings_from_env_supports_efficient_preset(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="efficient")

    assert settings.preset == "efficient"
    assert settings.prompt_profile == "optimized-lite"
    assert settings.max_attempts == 3


def test_settings_from_env_allows_prompt_profile_override(tmp_path: Path) -> None:
    settings = Settings.from_env(
        project_root=tmp_path,
        preset_override="accuracy",
        prompt_profile_override="paper",
    )

    assert settings.preset == "accuracy"
    assert settings.prompt_profile == "paper"
    assert settings.max_attempts == 5


def test_settings_from_env_supports_thorough_preset(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="thorough")

    assert settings.preset == "thorough"
    assert settings.prompt_profile == "optimized-strict"
    assert settings.max_attempts == 4


def test_settings_from_env_supports_model_profile_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, model_profile_override="qwen35-9b")

    assert settings.model_profile == "qwen35-9b"
    assert settings.extraction_model == "qwen3.5:9b"
    assert settings.reasoning_model == "qwen3.5:9b"


def test_settings_from_env_marks_custom_model_overrides(tmp_path: Path) -> None:
    settings = Settings.from_env(
        project_root=tmp_path,
        model_profile_override="gemma-moe",
        reasoning_model_override="gpt-oss:20b",
    )

    assert settings.model_profile == "custom"
    assert settings.extraction_model == "gemma3:4b"
    assert settings.reasoning_model == "gpt-oss:20b"


def test_settings_from_env_allows_disabling_llm_cache(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, disable_llm_cache_override=True)

    assert settings.disable_llm_cache is True


def test_prompt_template_dir_prefers_profile_subdirectory(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="optimized")
    prompts_root = tmp_path / "prompts"
    settings.prompts_dir = prompts_root
    (prompts_root / "optimized").mkdir(parents=True)

    assert settings.prompt_template_dir == prompts_root / "optimized"


def test_prompt_template_dir_falls_back_to_root_when_profile_missing(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="optimized")
    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir(parents=True)
    settings.prompts_dir = prompts_root

    assert settings.prompt_template_dir == prompts_root
