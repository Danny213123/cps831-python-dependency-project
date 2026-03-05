from __future__ import annotations

from pathlib import Path

from agentic_python_dependency.config import Settings
from agentic_python_dependency.graph import ResolutionWorkflow
from agentic_python_dependency.presets import resolve_research_features
from agentic_python_dependency.state import PackageVersionOptions
from agentic_python_dependency.tools.constraint_pack import build_constraint_pack, generate_candidate_bundles
from agentic_python_dependency.tools.dynamic_imports import collect_dynamic_import_candidates
from agentic_python_dependency.tools.repo_aliases import build_repo_alias_candidates
from agentic_python_dependency.tools.repair_feedback import append_feedback_event, summarize_feedback_memory
from agentic_python_dependency.tools.retry_policy import classify_retry_decision


class FakePyPIStore:
    def get_version_options(self, package: str, target_python: str, *, preset: str = "optimized") -> PackageVersionOptions:
        if package == "opencv-contrib-python":
            return PackageVersionOptions(package=package, versions=["4.10.0"])
        raise FileNotFoundError(package)

    @staticmethod
    def release_files(package: str, version: str) -> list[dict[str, str]]:
        return [{"url": f"https://example.invalid/{package}/{version}.whl"}]


class FakePackageMetadataStore:
    @staticmethod
    def parse_release_metadata(package: str, version: str, *, release_files: list[dict[str, str]] | None = None) -> dict[str, object]:
        assert release_files
        return {
            "package": package,
            "version": version,
            "top_level_modules": ["cv2"],
            "requires_dist": ["numpy>=1.21"],
            "source": "wheel",
        }


def test_settings_from_env_resolves_research_bundle_and_feature_overrides(tmp_path: Path) -> None:
    settings = Settings.from_env(
        project_root=tmp_path,
        preset_override="research",
        research_bundle_override="enhanced",
        research_feature_overrides=["dynamic_imports"],
        research_feature_disable_overrides=["repair_memory"],
    )

    assert settings.research_bundle == "enhanced"
    assert "dynamic_aliases" in settings.research_features
    assert "dynamic_imports" in settings.research_features
    assert "repair_memory" not in settings.research_features


def test_resolve_research_features_respects_bundle_order_and_overrides() -> None:
    features = resolve_research_features(
        "baseline",
        enabled=["repair_memory", "dynamic_imports"],
        disabled=["repair_memory"],
    )

    assert features == ("dynamic_imports",)


def test_collect_dynamic_import_candidates_detects_resolved_ambiguous_and_entry_points(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        """
[project]
name = "example"
version = "0.1.0"

[project.scripts]
demo = "package.cli:main"
""".strip(),
        encoding="utf-8",
    )

    payload = collect_dynamic_import_candidates(
        {
            "example.py": (
                "import importlib\n"
                "module = importlib.import_module('yaml')\n"
                "mod = __import__(name)\n"
            )
        },
        project_root=project_root,
    )

    assert "yaml" in payload["resolved"]
    assert "package" in payload["resolved"]
    assert "__import__" in payload["ambiguous"]


def test_build_repo_alias_candidates_uses_declared_package_metadata() -> None:
    alias_map, top_level_map = build_repo_alias_candidates(
        {"declared_packages": ["opencv-contrib-python"]},
        target_python="3.12",
        pypi_store=FakePyPIStore(),
        package_metadata_store=FakePackageMetadataStore(),
        preset="research",
    )

    assert alias_map == {"cv2": ["opencv-contrib-python"]}
    assert top_level_map == {"opencv-contrib-python": ["cv2"]}


def test_build_constraint_pack_detects_python_intersection_and_pairwise_conflicts() -> None:
    pack = build_constraint_pack(
        [
            PackageVersionOptions(
                package="alpha",
                versions=["1.0.0"],
                requires_python={"1.0.0": ">=4"},
                requires_dist={"1.0.0": ["beta<1"]},
            ),
            PackageVersionOptions(
                package="beta",
                versions=["2.0.0"],
                requires_python={"2.0.0": ">=3.8"},
                requires_dist={"2.0.0": []},
            ),
        ],
        target_python="3.12",
    )

    assert pack.python_intersection_valid is False
    assert pack.conflict_precheck_failed is True
    assert any(note.kind == "requires_python" for note in pack.conflict_notes)
    assert any(note.kind == "requires_dist" for note in pack.conflict_notes)
    assert generate_candidate_bundles(pack)


def test_build_constraint_pack_keeps_requires_dist_conflicts_non_terminal_when_python_intersection_is_valid() -> None:
    pack = build_constraint_pack(
        [
            PackageVersionOptions(
                package="alpha",
                versions=["1.0.0"],
                requires_python={"1.0.0": ">=3.8"},
                requires_dist={"1.0.0": ["beta<1"]},
            ),
            PackageVersionOptions(
                package="beta",
                versions=["2.0.0"],
                requires_python={"2.0.0": ">=3.8"},
                requires_dist={"2.0.0": []},
            ),
        ],
        target_python="3.12",
    )

    assert pack.python_intersection_valid is True
    assert pack.conflict_precheck_failed is False
    assert any(note.kind == "requires_dist" for note in pack.conflict_notes)


def test_classify_retry_decision_limits_native_build_retries_once_system_packages_were_injected() -> None:
    first = classify_retry_decision("NativeBuildError", system_packages_injected=False, native_retry_used=0)
    second = classify_retry_decision("NativeBuildError", system_packages_injected=True, native_retry_used=1)

    assert first.severity == "limited_retryable"
    assert first.candidate_fallback_allowed is True
    assert first.repair_allowed is True
    assert second.candidate_fallback_allowed is False
    assert second.repair_allowed is False


def test_feedback_memory_summary_aggregates_workspace_local_history(tmp_path: Path) -> None:
    memory_dir = tmp_path / "experimental_memory"
    append_feedback_event(
        memory_dir,
        {
            "error_category": "ImportError",
            "target_python": "3.12",
            "strategy_type": "downgrade",
            "success": True,
        },
    )
    append_feedback_event(
        memory_dir,
        {
            "error_category": "ImportError",
            "target_python": "3.12",
            "strategy_type": "downgrade",
            "success": False,
        },
    )

    summary = summarize_feedback_memory(memory_dir)

    assert summary["count"] == 2
    assert summary["entries"][0]["strategy_type"] == "downgrade"
    assert summary["entries"][0]["successes"] == 1
    assert summary["entries"][0]["failures"] == 1


def test_research_prompt_templates_render_without_format_key_errors(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"
    workflow = ResolutionWorkflow(settings)

    rendered = {
        "package_inference": workflow._format_prompt(
            "package_inference.txt",
            raw_file="import yaml\n",
            extracted_imports="yaml",
            repo_evidence="{}",
        ),
        "package_cross_validate": workflow._format_prompt(
            "package_cross_validate.txt",
            target_python="3.12",
            enabled_features="multipass_inference",
            candidate_payload="[]",
            repo_evidence="{}",
        ),
        "candidate_plans_v2": workflow._format_prompt(
            "candidate_plans_v2.txt",
            target_python="3.12",
            allowed_packages="PyYAML",
            rag_context="{}",
            max_plan_count=3,
        ),
        "repair_attempt_v2": workflow._format_prompt(
            "repair_attempt_v2.txt",
            target_python="3.12",
            allowed_packages="PyYAML",
            previous_plan="PyYAML==6.0.2",
            attempted_plans="",
            error_details="ModuleNotFoundError",
            repair_memory="{}",
            feedback_summary="{}",
            rag_context="{}",
        ),
        "version_negotiation": workflow._format_prompt(
            "version_negotiation.txt",
            target_python="3.12",
            candidate_bundles="[]",
            conflict_notes="[]",
            repo_evidence="{}",
        ),
    }

    assert all(rendered.values())
    assert "Target Python:" not in rendered["package_inference"]
