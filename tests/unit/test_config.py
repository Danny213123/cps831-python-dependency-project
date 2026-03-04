from pathlib import Path

from agentic_python_dependency.config import Settings


def test_settings_from_env_uses_preset_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="performance")

    assert settings.preset == "performance"
    assert settings.prompt_profile == "optimized-lite"
    assert settings.max_attempts == 2
    assert settings.resolver == "apdr"


def test_settings_from_env_supports_resolver_override(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, resolver_override="pyego")

    assert settings.resolver == "pyego"


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


def test_settings_from_env_supports_research_preset(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")

    assert settings.preset == "research"
    assert settings.prompt_profile == "research-rag"
    assert settings.max_attempts == 6
    assert settings.rag_mode == "hybrid"
    assert settings.structured_prompting is True
    assert settings.candidate_plan_count == 3
    assert settings.repo_evidence_enabled is True


def test_settings_from_env_supports_experimental_preset(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="experimental")

    assert settings.preset == "experimental"
    assert settings.prompt_profile == "optimized-strict"
    assert settings.max_attempts == 5
    assert settings.rag_mode == "pypi"
    assert settings.structured_prompting is False
    assert settings.candidate_plan_count == 1
    assert settings.repo_evidence_enabled is False


def test_settings_from_env_supports_model_profile_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, model_profile_override="qwen35-9b")

    assert settings.model_profile == "qwen35-9b"
    assert settings.extraction_model == "qwen3.5:9b"
    assert settings.reasoning_model == "qwen3.5:9b"


def test_settings_from_env_supports_gemma_moe_lite_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, model_profile_override="gemma-moe-lite")

    assert settings.model_profile == "gemma-moe-lite"
    assert settings.extraction_model == "gemma3:1b"
    assert settings.reasoning_model == "gemma3:4b"


def test_settings_from_env_supports_qwen35_moe_lite_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, model_profile_override="qwen35-moe-lite")

    assert settings.model_profile == "qwen35-moe-lite"
    assert settings.extraction_model == "qwen3.5:0.8b"
    assert settings.reasoning_model == "qwen3.5:4b"


def test_settings_from_env_marks_custom_model_overrides(tmp_path: Path) -> None:
    settings = Settings.from_env(
        project_root=tmp_path,
        model_profile_override="gemma-moe",
        reasoning_model_override="gpt-oss:20b",
    )

    assert settings.model_profile == "custom"
    assert settings.extraction_model == "gemma3:4b"
    assert settings.reasoning_model == "gpt-oss:20b"


def test_settings_from_env_supports_runtime_toggle_overrides(tmp_path: Path) -> None:
    settings = Settings.from_env(
        project_root=tmp_path,
        use_moe_override=False,
        use_rag_override=False,
        use_langchain_override=False,
        version_model_override="model:version",
        repair_model_override="model:repair",
        adjudication_model_override="model:adjudicate",
    )

    assert settings.use_moe is False
    assert settings.use_rag is False
    assert settings.use_langchain is False
    assert settings.stage_model("extract") == settings.reasoning_model
    assert settings.stage_model("version") == settings.reasoning_model
    assert settings.version_model == "model:version"
    assert settings.repair_model == "model:repair"
    assert settings.adjudication_model == "model:adjudicate"


def test_settings_from_env_supports_false_runtime_env_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APDR_USE_MOE", "false")
    monkeypatch.setenv("APDR_USE_RAG", "0")
    monkeypatch.setenv("APDR_USE_LANGCHAIN", "no")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.use_moe is False
    assert settings.use_rag is False
    assert settings.use_langchain is False


def test_settings_from_env_supports_competition_run_benchmark_source(tmp_path: Path) -> None:
    settings = Settings.from_env(
        project_root=tmp_path,
        benchmark_case_source_override="competition-run",
    )

    assert settings.benchmark_case_source == "competition-run"


def test_settings_from_env_supports_competition_csv_overrides(tmp_path: Path) -> None:
    csv_path = tmp_path / "official.csv"
    csv_path.write_text("name\nabc123\n", encoding="utf-8")

    settings = Settings.from_env(
        project_root=tmp_path,
        competition_result_csvs_override=[str(csv_path)],
    )

    assert settings.competition_result_csvs == (csv_path.resolve(),)


def test_settings_from_env_supports_competition_filter_file_override(tmp_path: Path) -> None:
    filter_path = tmp_path / "competition" / "competition-case-ids.txt"

    settings = Settings.from_env(
        project_root=tmp_path,
        competition_case_ids_file_override=str(filter_path),
    )

    assert settings.competition_case_ids_file == filter_path.resolve()


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
