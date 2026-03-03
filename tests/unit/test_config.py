from pathlib import Path

from agentic_python_dependency.config import Settings


def test_settings_from_env_uses_preset_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="performance")

    assert settings.preset == "performance"
    assert settings.prompt_profile == "optimized-lite"
    assert settings.max_attempts == 2


def test_settings_from_env_allows_prompt_profile_override(tmp_path: Path) -> None:
    settings = Settings.from_env(
        project_root=tmp_path,
        preset_override="accuracy",
        prompt_profile_override="paper",
    )

    assert settings.preset == "accuracy"
    assert settings.prompt_profile == "paper"
    assert settings.max_attempts == 5


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
