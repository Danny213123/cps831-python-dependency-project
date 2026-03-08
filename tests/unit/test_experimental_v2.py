from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from agentic_python_dependency.config import Settings
from agentic_python_dependency.graph import (
    ResolutionWorkflow,
    parse_package_inference_output,
    route_after_research_classification,
)
from agentic_python_dependency.presets import resolve_research_features
from agentic_python_dependency.state import AttemptRecord, BenchmarkCase, CandidateDependency, CandidatePlan, ConflictNote, ExecutionOutcome, PackageVersionOptions, ResolvedDependency
from agentic_python_dependency.tools.constraint_pack import build_constraint_pack, generate_candidate_bundles
from agentic_python_dependency.tools.dynamic_imports import collect_dynamic_import_candidates
from agentic_python_dependency.tools.repo_aliases import build_repo_alias_candidates
from agentic_python_dependency.tools.repair_feedback import append_feedback_event, summarize_feedback_memory
from agentic_python_dependency.tools.retry_policy import classify_retry_decision
from agentic_python_dependency.tools.rag_context import summarize_rag_context
from agentic_python_dependency.tools.structured_outputs import (
    StructuredOutputError,
    parse_alias_resolution_payload,
    parse_candidate_plan_payload,
    parse_repair_plan_payload,
    parse_source_compatibility_payload,
)


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


class FakeAliasPromptRunner:
    @staticmethod
    def stage_model(stage: str) -> str:
        return stage

    @staticmethod
    def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
        assert stage == "extract"
        assert template == "resolve_aliases.txt"
        assert "memcache" in variables["unresolved_packages"]
        return '{"aliases":[{"import_name":"memcache","pypi_package":"python-memcached"}]}'


class FakeAliasPyPIStore:
    @staticmethod
    def get_version_options(
        package: str,
        target_python: str,
        *,
        limit: int = 20,
        preset: str = "optimized",
    ) -> PackageVersionOptions:
        assert target_python
        assert limit >= 1
        assert preset
        if package == "python-memcached":
            return PackageVersionOptions(package="python-memcached", versions=["1.62"])
        raise FileNotFoundError(package)


class TargetAwarePyPIStore:
    def __init__(self, versions_by_target: dict[tuple[str, str], list[str]]):
        self.versions_by_target = versions_by_target

    def get_version_options(
        self,
        package: str,
        target_python: str,
        *,
        limit: int = 20,
        preset: str = "optimized",
    ) -> PackageVersionOptions:
        assert limit >= 1
        assert preset
        versions = self.versions_by_target.get((package, target_python))
        if versions is None:
            raise FileNotFoundError(package)
        return PackageVersionOptions(package=package, versions=versions[:limit])


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
    assert not generate_candidate_bundles(pack)


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


def test_generate_candidate_bundles_preserves_semantic_version_priority() -> None:
    pack = build_constraint_pack(
        [
            PackageVersionOptions(
                package="scrapy",
                versions=["2.11.2", "2.10.1", "2.9.0"],
                requires_python={
                    "2.11.2": ">=3.8",
                    "2.10.1": ">=3.8",
                    "2.9.0": ">=3.8",
                },
            ),
            PackageVersionOptions(
                package="sip",
                versions=["6.15.1", "6.8.6", "6.8.5"],
                requires_python={
                    "6.15.1": ">=3.8",
                    "6.8.6": ">=3.8",
                    "6.8.5": ">=3.8",
                },
            ),
        ],
        target_python="3.12",
    )

    bundles = generate_candidate_bundles(pack, beam_width=4)

    assert [(dep.name, dep.version) for dep in bundles[0]] == [
        ("scrapy", "2.11.2"),
        ("sip", "6.15.1"),
    ]
    assert [(dep.name, dep.version) for dep in bundles[1]] == [
        ("scrapy", "2.11.2"),
        ("sip", "6.8.6"),
    ]
    assert [(dep.name, dep.version) for dep in bundles[2]] == [
        ("scrapy", "2.10.1"),
        ("sip", "6.15.1"),
    ]


def test_generate_candidate_bundles_filters_requires_dist_incompatible_combinations() -> None:
    pack = build_constraint_pack(
        [
            PackageVersionOptions(
                package="alpha",
                versions=["2.0.0", "1.0.0"],
                requires_python={
                    "2.0.0": ">=3.8",
                    "1.0.0": ">=3.8",
                },
                requires_dist={
                    "2.0.0": ["beta>=2"],
                    "1.0.0": ["beta<2"],
                },
            ),
            PackageVersionOptions(
                package="beta",
                versions=["2.0.0", "1.0.0"],
                requires_python={
                    "2.0.0": ">=3.8",
                    "1.0.0": ">=3.8",
                },
            ),
        ],
        target_python="3.12",
    )

    bundles = generate_candidate_bundles(pack, beam_width=8)

    assert [[(dep.name, dep.version) for dep in bundle] for bundle in bundles] == [
        [("alpha", "2.0.0"), ("beta", "2.0.0")],
        [("alpha", "1.0.0"), ("beta", "1.0.0")],
    ]


def test_generate_candidate_bundles_avoids_lexicographic_beam_pruning() -> None:
    keras_versions = [
        "2.15.0",
        "2.13.1",
        "2.12.0",
        "2.11.0",
        "2.10.0",
        "2.9.0",
        "2.8.0",
        "2.7.0",
        "2.6.0",
        "2.5.0",
        "2.4.3",
        "2.4.2",
        "2.4.1",
    ]
    numpy_versions = [
        "1.25.0",
        "1.24.3",
        "1.21.9",
        "1.21.8",
        "1.21.7",
        "1.21.6",
        "1.21.5",
        "1.21.4",
        "1.21.3",
        "1.21.2",
        "1.21.1",
        "1.21.0",
        "1.20.9",
    ]
    pack = build_constraint_pack(
        [
            PackageVersionOptions(
                package="gym",
                versions=["0.26.2"],
                requires_python={"0.26.2": ">=3.8"},
            ),
            PackageVersionOptions(
                package="keras",
                versions=keras_versions,
                requires_python={version: ">=3.8" for version in keras_versions},
            ),
            PackageVersionOptions(
                package="numpy",
                versions=numpy_versions,
                requires_python={version: ">=3.8" for version in numpy_versions},
            ),
            PackageVersionOptions(
                package="tensorflow",
                versions=["2.13.1"],
                requires_python={"2.13.1": ">=3.8"},
                requires_dist={"2.13.1": ["keras<2.14,>=2.13.1", "numpy<=1.24.3,>=1.22"]},
            ),
        ],
        target_python="3.12",
        top_k=20,
    )

    bundles = generate_candidate_bundles(pack, beam_width=12, max_bundles=10)

    assert [[(dep.name, dep.version) for dep in bundle] for bundle in bundles] == [
        [
            ("gym", "0.26.2"),
            ("keras", "2.13.1"),
            ("numpy", "1.24.3"),
            ("tensorflow", "2.13.1"),
        ]
    ]


def test_generate_candidate_bundles_prioritizes_source_compatible_versions_when_requested() -> None:
    pack = build_constraint_pack(
        [
            PackageVersionOptions(
                package="gym",
                versions=["0.26.2", "0.26.1", "0.25.2"],
                requires_python={version: ">=3.8" for version in ["0.26.2", "0.26.1", "0.25.2"]},
            ),
            PackageVersionOptions(
                package="keras",
                versions=["2.15.0", "2.13.1", "2.12.0", "2.11.0", "2.10.0", "2.9.0", "2.8.0", "2.7.0", "2.6.0", "2.4.3"],
                requires_python={version: ">=3.8" for version in ["2.15.0", "2.13.1", "2.12.0", "2.11.0", "2.10.0", "2.9.0", "2.8.0", "2.7.0", "2.6.0", "2.4.3"]},
            ),
            PackageVersionOptions(
                package="numpy",
                versions=["1.24.3", "1.24.2", "1.24.1", "1.24.0", "1.23.5", "1.23.4", "1.23.3", "1.23.2", "1.23.1", "1.23.0", "1.22.4", "1.19.5"],
                requires_python={version: ">=3.8" for version in ["1.24.3", "1.24.2", "1.24.1", "1.24.0", "1.23.5", "1.23.4", "1.23.3", "1.23.2", "1.23.1", "1.23.0", "1.22.4", "1.19.5"]},
            ),
            PackageVersionOptions(
                package="tensorflow",
                versions=["2.13.1", "2.13.0", "2.12.1", "2.12.0", "2.11.0", "2.10.1", "2.4.4"],
                requires_python={version: ">=3.8" for version in ["2.13.1", "2.13.0", "2.12.1", "2.12.0", "2.11.0", "2.10.1", "2.4.4"]},
                requires_dist={
                    "2.13.1": ["keras<2.14,>=2.13.1", "numpy<=1.24.3,>=1.22"],
                    "2.13.0": ["keras<2.14,>=2.13.1", "numpy<=1.24.3,>=1.22"],
                    "2.12.1": ["keras<2.13,>=2.12.0", "numpy<1.24,>=1.22"],
                    "2.12.0": ["keras<2.13,>=2.12.0", "numpy<1.24,>=1.22"],
                    "2.11.0": ["keras<2.12,>=2.11.0", "numpy<1.24,>=1.20"],
                    "2.10.1": ["keras<2.11,>=2.10.0", "numpy<1.24,>=1.20"],
                    "2.4.4": ["keras<=2.4.3", "numpy<=1.19.5,>=1.19.2"],
                },
            ),
        ],
        target_python="3.8",
        top_k=20,
    )

    bundles = generate_candidate_bundles(
        pack,
        beam_width=12,
        max_bundles=3,
        version_cap_per_package=12,
        preferred_versions_by_package={
            "gym": ["0.25.2"],
            "keras": ["2.4.3"],
            "numpy": ["1.19.5"],
            "tensorflow": ["2.4.4"],
        },
    )

    assert [(dep.name, dep.version) for dep in bundles[0]] == [
        ("gym", "0.25.2"),
        ("keras", "2.4.3"),
        ("numpy", "1.19.5"),
        ("tensorflow", "2.4.4"),
    ]


def test_generate_candidate_bundles_surface_legacy_keras_family_before_version_cap_when_model_hints_it(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(settings, pypi_store=FakePyPIStore())
    case_root = tmp_path / "case-legacy-keras-bundles"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import gym\nimport keras\nimport numpy\nimport tensorflow\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-legacy-keras-bundles", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["target_python"] = "3.8"
    state["version_options"] = [
        PackageVersionOptions(
            package="gym",
            versions=["0.26.2", "0.26.1", "0.26.0", "0.25.2"],
            requires_python={version: ">=3.8" for version in ["0.26.2", "0.26.1", "0.26.0", "0.25.2"]},
        ),
        PackageVersionOptions(
            package="keras",
            versions=["2.15.0", "2.13.1", "2.12.0", "2.11.0", "2.10.0", "2.9.0", "2.4.3"],
            requires_python={version: ">=3.8" for version in ["2.15.0", "2.13.1", "2.12.0", "2.11.0", "2.10.0", "2.9.0", "2.4.3"]},
        ),
        PackageVersionOptions(
            package="numpy",
            versions=["1.24.4", "1.24.3", "1.24.2", "1.24.1", "1.24.0", "1.23.5", "1.19.5"],
            requires_python={version: ">=3.8" for version in ["1.24.4", "1.24.3", "1.24.2", "1.24.1", "1.24.0", "1.23.5", "1.19.5"]},
        ),
        PackageVersionOptions(
            package="tensorflow",
            versions=["2.13.1", "2.13.0", "2.12.1", "2.12.0", "2.11.1", "2.10.1", "2.4.4"],
            requires_python={version: ">=3.8" for version in ["2.13.1", "2.13.0", "2.12.1", "2.12.0", "2.11.1", "2.10.1", "2.4.4"]},
            requires_dist={
                "2.13.1": ["keras<2.14,>=2.13.1", "numpy<=1.24.3,>=1.22"],
                "2.13.0": ["keras<2.14,>=2.13.1", "numpy<=1.24.3,>=1.22"],
                "2.12.1": ["keras<2.13,>=2.12.0", "numpy<1.24,>=1.22"],
                "2.12.0": ["keras<2.13,>=2.12.0", "numpy<1.24,>=1.22"],
                "2.11.1": ["keras<2.12,>=2.11.0", "numpy<1.24,>=1.20"],
                "2.10.1": ["keras<2.11,>=2.10.0", "numpy<1.24,>=1.20"],
                "2.4.4": ["keras<=2.4.3", "numpy<=1.19.5,>=1.19.2"],
            },
        ),
    ]
    pack = build_constraint_pack(state["version_options"], target_python="3.8", top_k=None)

    modern_only = generate_candidate_bundles(pack, beam_width=12, max_bundles=4, version_cap_per_package=6)
    assert all(
        [("tensorflow", "2.4.4"), ("numpy", "1.19.5")] != [(dep.name, dep.version) for dep in bundle if dep.name in {"tensorflow", "numpy"}]
        for bundle in modern_only
    )

    state["llm_source_compatibility_hints"] = [
        {"package": "gym", "preferred_specifier": "<0.26.0", "reason": "legacy_gym_api"},
        {"package": "keras", "preferred_specifier": "<=2.4.3", "reason": "legacy_standalone_keras_api"},
        {"package": "numpy", "preferred_specifier": "<=1.19.5", "reason": "legacy_keras_tensorflow_numpy_api"},
        {"package": "tensorflow", "preferred_specifier": "<=2.4.4", "reason": "legacy_standalone_keras_api"},
    ]
    hinted = generate_candidate_bundles(
        pack,
        beam_width=12,
        max_bundles=4,
        version_cap_per_package=6,
        preferred_versions_by_package=workflow._preferred_bundle_versions(state),
    )

    assert [(dep.name, dep.version) for dep in hinted[0]] == [
        ("gym", "0.25.2"),
        ("keras", "2.4.3"),
        ("numpy", "1.19.5"),
        ("tensorflow", "2.4.4"),
    ]


def test_retrieve_pypi_metadata_uses_benchmark_platform_override(tmp_path: Path) -> None:
    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def get_version_options(
            self,
            package: str,
            target_python: str,
            *,
            limit: int = 20,
            preset: str = "optimized",
            platform: str | None = None,
        ) -> PackageVersionOptions:
            self.calls.append(
                {
                    "package": package,
                    "target_python": target_python,
                    "limit": limit,
                    "preset": preset,
                    "platform": platform,
                }
            )
            return PackageVersionOptions(package=package, versions=["2.4.4"])

    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(settings, pypi_store=RecordingStore())
    case_root = tmp_path / "case"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import tensorflow\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-platform", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["inferred_packages"] = ["tensorflow"]
    state["target_python"] = "3.8"

    workflow.retrieve_pypi_metadata(state)

    assert workflow.pypi_store.calls[0]["platform"] == "linux/amd64"


def test_apply_source_version_preferences_annotates_llm_source_compatibility_hints_in_research(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(settings, pypi_store=FakePyPIStore())
    case_root = tmp_path / "case"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import tensorflow as tf\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-legacy", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["inferred_packages"] = ["gym", "keras", "numpy", "tensorflow"]
    state["llm_source_compatibility_hints"] = [
        {"package": "gym", "preferred_specifier": "<0.26", "reason": "legacy_gym_api"},
        {"package": "keras", "preferred_specifier": "<=2.4.3", "reason": "legacy_keras_api"},
        {"package": "numpy", "preferred_specifier": "<=1.19.5", "reason": "legacy_tensorflow_numpy"},
        {"package": "tensorflow", "preferred_specifier": "<=2.4.4", "reason": "legacy_keras_tensorflow_api"},
    ]
    state["version_options"] = [
        PackageVersionOptions(package="gym", versions=["0.26.2", "0.26.1", "0.25.2"]),
        PackageVersionOptions(package="keras", versions=["2.15.0", "2.13.1", "2.4.3", "2.4.2"]),
        PackageVersionOptions(package="numpy", versions=["1.24.4", "1.24.3", "1.19.5", "1.19.4"]),
        PackageVersionOptions(package="tensorflow", versions=["2.13.1", "2.10.1", "2.4.4", "2.4.3"]),
    ]

    workflow._apply_source_version_preferences(state)

    versions_by_package = {option.package: option.versions for option in state["version_options"]}
    assert versions_by_package["gym"] == ["0.26.2", "0.26.1", "0.25.2"]
    assert versions_by_package["keras"] == ["2.15.0", "2.13.1", "2.4.3", "2.4.2"]
    assert versions_by_package["numpy"] == ["1.24.4", "1.24.3", "1.19.5", "1.19.4"]
    assert versions_by_package["tensorflow"] == ["2.13.1", "2.10.1", "2.4.4", "2.4.3"]
    assert "source_compat_legacy_keras_api:<=2.4.3" in state["applied_compatibility_policy"]["keras"]
    assert "source_compat_legacy_gym_api:<0.26" in state["applied_compatibility_policy"]["gym"]


def test_retrieve_pypi_metadata_asks_model_for_source_compatibility_hints(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "version"
            assert template == "source_compatibility.txt"
            assert "tensorflow" in variables["allowed_packages"]
            assert "import_specs" in variables["validation_options"]
            assert "tensorflow_models_module" in variables["source_signals"]
            assert "tensorflow_app_flags" in variables["source_signals"]
            assert "python2_xrange" in variables["source_signals"]
            return (
                '{"default_runtime_profile":"import_specs","compatibility_hints":['
                '{"package":"tensorflow","preferred_specifier":"<2.0.0","reason":"uses tensorflow.contrib"},'
                '{"package":"gensim","preferred_specifier":">=0.13.3,<4.0.0","reason":"uses index2word"}]}'
            )

        @staticmethod
        def invoke_text(stage: str, prompt_text: str) -> str:
            raise AssertionError("Adjudication should not be needed in this test")

    class PyPIStore:
        @staticmethod
        def get_version_options(
            package: str,
            target_python: str,
            *,
            limit: int = 20,
            preset: str = "optimized",
        ) -> PackageVersionOptions:
            assert target_python == "3.7"
            assert limit >= 1
            assert preset
            if package == "gensim":
                return PackageVersionOptions(package="gensim", versions=["4.2.0", "3.8.3", "3.7.3"])
            if package == "tensorflow":
                return PackageVersionOptions(package="tensorflow", versions=["2.11.0", "2.10.1", "1.15.5", "1.14.0"])
            raise FileNotFoundError(package)

    workflow = ResolutionWorkflow(settings, prompt_runner=PromptRunner(), pypi_store=PyPIStore())
    case_root = tmp_path / "case-tf1"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text(
        "\n".join(
                [
                    "# required tensorflow 0.12",
                    "from gensim.models import Word2Vec",
                    "import tensorflow as tf",
                    "from tensorflow.models.embedding import gen_word2vec as word2vec",
                    "from tensorflow.contrib.tensorboard.plugins import projector",
                    "flags = tf.app.flags",
                    "model = Word2Vec.load('YOUR-MODEL')",
                    "for word in model.wv.index2word[:10000]:",
                    "    print(word)",
                    "for _ in xrange(10):",
                    "    pass",
                    "sess = tf.InteractiveSession()",
                ]
            ),
            encoding="utf-8",
        )
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-tf1", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["inferred_packages"] = ["gensim", "tensorflow"]
    state["target_python"] = "3.7"
    state["validation_options"] = [
        {"profile": "docker_cmd", "command": "", "reason": "exact"},
        {"profile": "import_specs", "command": "python - <<'PY'\nprint('ok')\nPY", "reason": "safe"},
    ]
    state["default_validation_profile"] = "docker_cmd"
    state["current_runtime_profile"] = "docker_cmd"
    state["current_validation_command"] = ""

    workflow.retrieve_pypi_metadata(state)

    assert state["default_validation_profile"] == "import_specs"
    assert state["current_runtime_profile"] == "import_specs"
    assert "source_compat_uses tensorflow.contrib:<2.0.0" in state["applied_compatibility_policy"]["tensorflow"]
    assert "source_compat_uses index2word:>=0.13.3,<4.0.0" in state["applied_compatibility_policy"]["gensim"]
    assert state["llm_source_compatibility_hints"] == [
        {"package": "tensorflow", "preferred_specifier": "<2.0.0", "reason": "uses tensorflow.contrib"},
        {"package": "gensim", "preferred_specifier": ">=0.13.3,<4.0.0", "reason": "uses index2word"},
    ]
    assert state["structured_outputs"]["source_compatibility"]["default_runtime_profile"] == "import_specs"


def test_source_signal_summary_detects_legacy_keras_and_gym_signals(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(settings, pypi_store=FakePyPIStore())
    case_root = tmp_path / "case-legacy-keras-signals"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text(
        "\n".join(
            [
                "import gym",
                "from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation",
                "from keras.models import Sequential, Model",
                "import keras.backend as K",
                "import tensorflow as tf",
                "state = env.reset()",
                "new_state, reward, done, _ = env.step(action)",
                "model = Model(input=state_in, output=out)",
            ]
        ),
        encoding="utf-8",
    )
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-legacy-keras-signals", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}

    rendered = workflow._source_signal_summary(state, limit=5000)

    assert "keras_layers_merge_import" in rendered
    assert "standalone_keras_api" in rendered
    assert "keras_legacy_model_input" in rendered
    assert "keras_legacy_model_output" in rendered
    assert "gym_legacy_step_signature" in rendered
    assert "gym_legacy_reset_signature" in rendered


def test_retrieve_pypi_metadata_asks_model_for_legacy_keras_gym_family_hints(
    tmp_path: Path,
) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "version"
            assert template == "source_compatibility.txt"
            assert "keras_layers_merge_import" in variables["source_signals"]
            assert "standalone_keras_api" in variables["source_signals"]
            assert "keras_legacy_model_input" in variables["source_signals"]
            assert "keras_legacy_model_output" in variables["source_signals"]
            assert "gym_legacy_step_signature" in variables["source_signals"]
            assert "gym_legacy_reset_signature" in variables["source_signals"]
            return (
                '{"compatibility_hints":['
                '{"package":"gym","preferred_specifier":"<0.26.0","reason":"legacy gym api"},'
                '{"package":"keras","preferred_specifier":"<=2.4.3","reason":"legacy standalone keras api"},'
                '{"package":"numpy","preferred_specifier":"<=1.19.5","reason":"legacy keras tensorflow numpy api"},'
                '{"package":"tensorflow","preferred_specifier":"<=2.4.4","reason":"legacy standalone keras api"}]}'
            )

        @staticmethod
        def invoke_text(stage: str, prompt_text: str) -> str:
            raise AssertionError("Adjudication should not be needed in this test")

    class PyPIStore:
        @staticmethod
        def get_version_options(
            package: str,
            target_python: str,
            *,
            limit: int = 20,
            preset: str = "optimized",
        ) -> PackageVersionOptions:
            assert target_python == "3.8"
            assert limit >= 1
            assert preset
            if package == "gym":
                return PackageVersionOptions(package="gym", versions=["0.26.2", "0.26.1", "0.25.2"])
            if package == "keras":
                return PackageVersionOptions(package="keras", versions=["2.15.0", "2.13.1", "2.4.3", "2.4.2"])
            if package == "numpy":
                return PackageVersionOptions(package="numpy", versions=["1.24.4", "1.24.3", "1.19.5", "1.19.4"])
            if package == "tensorflow":
                return PackageVersionOptions(package="tensorflow", versions=["2.13.1", "2.10.1", "2.4.4", "2.4.3"])
            raise FileNotFoundError(package)

    workflow = ResolutionWorkflow(settings, prompt_runner=PromptRunner(), pypi_store=PyPIStore())
    case_root = tmp_path / "case-legacy-keras-hints"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text(
        "\n".join(
            [
                "import gym",
                "from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation",
                "from keras.models import Sequential, Model",
                "import keras.backend as K",
                "import tensorflow as tf",
                "state = env.reset()",
                "new_state, reward, done, _ = env.step(action)",
                "model = Model(input=state_in, output=out)",
            ]
        ),
        encoding="utf-8",
    )
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-legacy-keras-hints", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-legacy-keras-hints")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["inferred_packages"] = ["gym", "keras", "numpy", "tensorflow"]
    state["target_python"] = "3.8"
    state["validation_options"] = [
        {"profile": "docker_cmd", "command": "", "reason": "exact"},
        {"profile": "import_specs", "command": "python - <<'PY'\nprint('ok')\nPY", "reason": "safe"},
    ]
    state["default_validation_profile"] = "docker_cmd"
    state["current_runtime_profile"] = "docker_cmd"
    state["current_validation_command"] = ""

    workflow.retrieve_pypi_metadata(state)

    assert state["llm_source_compatibility_hints"] == [
        {"package": "gym", "preferred_specifier": "<0.26.0", "reason": "legacy gym api"},
        {"package": "keras", "preferred_specifier": "<=2.4.3", "reason": "legacy standalone keras api"},
        {"package": "numpy", "preferred_specifier": "<=1.19.5", "reason": "legacy keras tensorflow numpy api"},
        {"package": "tensorflow", "preferred_specifier": "<=2.4.4", "reason": "legacy standalone keras api"},
    ]
    assert workflow._preferred_bundle_versions(state) == {
        "gym": ["0.25.2"],
        "keras": ["2.4.3", "2.4.2"],
        "numpy": ["1.19.5", "1.19.4"],
        "tensorflow": ["2.4.4", "2.4.3"],
    }
    rendered = workflow._planning_version_space_summary(state, limit=5000)
    assert rendered.index('"0.25.2"') < rendered.index('"0.26.2"')
    assert rendered.index('"2.4.3"') < rendered.index('"2.15.0"')
    assert rendered.index('"1.19.5"') < rendered.index('"1.24.4"')


def test_planning_version_space_summary_surfaces_model_selected_preferred_versions_first(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(settings, pypi_store=FakePyPIStore())
    case_root = tmp_path / "case-summary"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text(
        "\n".join(
            [
                "# required tensorflow 0.12",
                "from gensim.models import Word2Vec",
                "import tensorflow as tf",
                "from tensorflow.contrib.tensorboard.plugins import projector",
                "for word in model.wv.index2word[:10000]:",
                "    print(word)",
            ]
        ),
        encoding="utf-8",
    )
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-summary", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["inferred_packages"] = ["gensim", "tensorflow"]
    state["llm_source_compatibility_hints"] = [
        {"package": "gensim", "preferred_specifier": ">=0.13.3,<4.0.0", "reason": "uses index2word"},
        {"package": "tensorflow", "preferred_specifier": "<2.0.0", "reason": "uses tensorflow.contrib"},
    ]
    state["version_options"] = [
        PackageVersionOptions(package="gensim", versions=["4.2.0", "4.1.2", "3.8.3", "3.7.3"]),
        PackageVersionOptions(package="tensorflow", versions=["2.11.0", "2.10.1", "1.15.5", "1.14.0"]),
    ]

    rendered = workflow._planning_version_space_summary(state, limit=5000)

    assert rendered.index('"gensim"') < rendered.index('"tensorflow"')
    assert rendered.index('"3.8.3"') < rendered.index('"4.2.0"')
    assert rendered.index('"1.15.5"') < rendered.index('"2.11.0"')


def test_parse_source_compatibility_payload_is_strict() -> None:
    runtime_profile, hints = parse_source_compatibility_payload(
        (
            '{"default_runtime_profile":"import_specs","compatibility_hints":['
            '{"package":"tensorflow","preferred_specifier":"<2.0.0","reason":"legacy tf1"}]}'
        ),
        allowed_packages={"tensorflow"},
        allowed_runtime_profiles={"docker_cmd", "import_specs"},
    )

    assert runtime_profile == "import_specs"
    assert hints == [
        {"package": "tensorflow", "preferred_specifier": "<2.0.0", "reason": "legacy tf1"}
    ]

    try:
        parse_source_compatibility_payload(
            '{"compatibility_hints":[{"package":"tensorflow","preferred_specifier":"not-a-spec"}]}',
            allowed_packages={"tensorflow"},
            allowed_runtime_profiles={"import_specs"},
        )
    except StructuredOutputError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Invalid preferred_specifier should raise StructuredOutputError.")


def test_research_prompts_emphasize_coherent_legacy_families(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"
    workflow = ResolutionWorkflow(settings, pypi_store=FakePyPIStore())
    source_hints = (
        '[{"package":"gym","preferred_specifier":"<0.26.0","reason":"legacy gym api"},'
        '{"package":"keras","preferred_specifier":"<=2.4.3","reason":"legacy standalone keras api"},'
        '{"package":"numpy","preferred_specifier":"<=1.19.5","reason":"legacy keras tensorflow numpy api"},'
        '{"package":"tensorflow","preferred_specifier":"<=2.4.4","reason":"legacy standalone keras api"}]'
    )

    candidate_prompt = workflow._format_prompt(
        "candidate_plans_v2.txt",
        target_python="3.8",
        allowed_packages="gym\nkeras\nnumpy\ntensorflow",
        version_space='{"packages":[]}',
        rag_context="{}",
        validation_options='[{"profile":"docker_cmd","command":"","reason":"exact"}]',
        default_validation_profile="docker_cmd",
        candidate_bundle_hints='{"generated_bundles":[],"negotiated_bundles":[]}',
        conflict_notes="[]",
        source_compatibility_hints=source_hints,
        max_plan_count=3,
    )
    repair_prompt = workflow._format_prompt(
        "repair_attempt_v2.txt",
        target_python="3.8",
        allowed_packages="gym\nkeras\nnumpy\ntensorflow",
        version_space='{"packages":[]}',
        validation_options='[{"profile":"docker_cmd","command":"","reason":"exact"}]',
        default_validation_profile="docker_cmd",
        conflict_notes='[{"package":"tensorflow","related_package":"keras","kind":"requires_dist","reason":"tensorflow constrains keras","severity":"warning"}]',
        source_compatibility_hints=source_hints,
        previous_plan="gym==0.26.2\nkeras==2.15.0\nnumpy==1.24.4\ntensorflow==2.13.1",
        attempted_plans="gym==0.26.2, keras==2.15.0, numpy==1.24.4, tensorflow==2.13.1",
        error_details="tensorflow 2.13.1 depends on keras<2.14 and >=2.13.1",
        repair_memory='{"entries":[]}',
        feedback_summary='{"entries":[]}',
        rag_context="{}",
        max_plan_count=2,
    )

    assert "keep related packages internally coherent" in candidate_prompt
    assert "legacy standalone Keras / legacy Gym snippets" in candidate_prompt
    assert source_hints in candidate_prompt
    assert "revise the coupled family together" in repair_prompt
    assert "family-level repair" in repair_prompt
    assert "prefer a `runtime_profile` change over dependency churn" in repair_prompt
    assert "normalized dependency set plus `runtime_profile` is identical to an attempted plan" in repair_prompt
    assert "Conflict notes:" in repair_prompt
    assert "tensorflow 2.13.1 depends on keras<2.14 and >=2.13.1" in repair_prompt
    assert "tensorflow constrains keras" in repair_prompt


def test_repair_prompt_research_passes_conflict_notes_to_prompt_runner(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "repair"
            assert template in {"repair_attempt.txt", "repair_attempt_v2.txt"}
            assert "tensorflow requires numpy<2.2.0" in variables["conflict_notes"]
            return '{"repair_applicable":false,"plans":[]}'

    workflow = ResolutionWorkflow(settings, prompt_runner=PromptRunner())
    case_root = tmp_path / "case-repair-conflict-notes"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import tensorflow as tf\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-repair-conflict-notes", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-repair-conflict-notes")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["target_python"] = "3.8"
    state["current_attempt"] = 2
    state["inferred_packages"] = ["numpy", "tensorflow"]
    state["version_options"] = [
        PackageVersionOptions(package="numpy", versions=["2.4.1", "2.1.3"]),
        PackageVersionOptions(
            package="tensorflow",
            versions=["2.19.1"],
            requires_dist={"2.19.1": ["numpy<2.2.0"]},
        ),
    ]
    state["selected_dependencies"] = [
        ResolvedDependency(name="numpy", version="2.4.1"),
        ResolvedDependency(name="tensorflow", version="2.19.1"),
    ]
    state["validation_options"] = [{"profile": "import_specs", "reason": "safe module probe"}]
    state["default_validation_profile"] = "import_specs"
    state["version_conflict_notes"] = [
        ConflictNote(
            package="tensorflow",
            related_package="numpy",
            kind="requires_dist",
            reason="tensorflow requires numpy<2.2.0",
            severity="warning",
        )
    ]
    state["last_error_details"] = "ResolutionImpossible: tensorflow 2.19.1 depends on numpy<2.2.0"

    updated = workflow.repair_prompt_c_research(state)

    assert updated["repair_model_concluded_impossible"] is True
    assert updated["repair_plan_unavailable_reason"] == "model_not_applicable"


def test_retrieve_version_specific_metadata_filters_versions_using_enriched_requires_python(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.research_features = ("transitive_conflicts",)
    workflow = ResolutionWorkflow(settings, pypi_store=FakePyPIStore())

    class MetadataStore:
        @staticmethod
        def parse_release_metadata(
            package: str,
            version: str,
            *,
            release_files: list[dict[str, str]] | None = None,
        ) -> dict[str, object]:
            assert package == "pymc3"
            assert release_files
            requires_python = {
                "3.7": ">=3.5.4",
                "3.6": ">=2.7",
                "3.5": ">=2.7",
            }[version]
            return {
                "package": package,
                "version": version,
                "top_level_modules": ["pymc3"],
                "requires_dist": ["numpy>=1.13.0"],
                "requires_python": requires_python,
                "source": "wheel",
            }

    workflow.package_metadata_store = MetadataStore()
    case_root = tmp_path / "case"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import pymc3\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-enriched-python", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["target_python"] = "2.7.18"
    state["version_options"] = [
        PackageVersionOptions(
            package="pymc3",
            versions=["3.7", "3.6", "3.5"],
            requires_python={"3.7": "", "3.6": ">=2.7", "3.5": ">=2.7"},
            platform_notes={"3.7": ["source_only_native_risk"], "3.6": ["wheel_available"], "3.5": ["wheel_available"]},
            requires_dist={"3.7": [], "3.6": [], "3.5": []},
        )
    ]

    updated = workflow.retrieve_version_specific_metadata(state)

    assert updated["version_options"][0].versions == ["3.6", "3.5"]
    assert updated["version_options"][0].requires_python == {"3.6": ">=2.7", "3.5": ">=2.7"}
    assert any(
        note["version"] == "3.7"
        for note in updated["pypi_evidence"]["metadata_enrichment"][0]["dropped_versions"]
    )
    assert any("pymc3 3.7: dropped after metadata Requires-Python >=3.5.4" in note for note in updated["platform_compatibility_notes"])


def test_activate_deferred_python_fallback_rebuilds_with_metadata_filtered_versions(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.research_features = ("transitive_conflicts",)

    class DeferredFallbackPyPIStore:
        @staticmethod
        def get_version_options(
            package: str,
            target_python: str,
            *,
            limit: int = 20,
            preset: str = "optimized",
        ) -> PackageVersionOptions:
            assert target_python == "2.7.18"
            assert limit >= 1
            assert preset
            if package != "pymc3":
                raise FileNotFoundError(package)
            return PackageVersionOptions(
                package="pymc3",
                versions=["3.7", "3.6", "3.5"],
                requires_python={"3.7": "", "3.6": ">=2.7", "3.5": ">=2.7"},
                requires_dist={"3.7": [], "3.6": [], "3.5": []},
            )

        @staticmethod
        def release_files(package: str, version: str) -> list[dict[str, str]]:
            assert package == "pymc3"
            return [{"url": f"https://example.invalid/{package}/{version}.whl"}]

    class MetadataStore:
        @staticmethod
        def parse_release_metadata(
            package: str,
            version: str,
            *,
            release_files: list[dict[str, str]] | None = None,
        ) -> dict[str, object]:
            assert package == "pymc3"
            assert release_files
            return {
                "package": package,
                "version": version,
                "top_level_modules": ["pymc3"],
                "requires_dist": ["numpy>=1.13.0"],
                "requires_python": {
                    "3.7": ">=3.5.4",
                    "3.6": ">=2.7",
                    "3.5": ">=2.7",
                }[version],
                "source": "wheel",
            }

    workflow = ResolutionWorkflow(settings, pypi_store=DeferredFallbackPyPIStore())
    workflow.package_metadata_store = MetadataStore()
    case_root = tmp_path / "case-deferred-fallback-filter"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import pymc3\nprint 'hello world'\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(
            case_id="case-deferred-fallback-filter",
            root_dir=case_root,
            snippet_path=snippet,
            case_source="all-gists",
        )
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-deferred-fallback-filter")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["inferred_packages"] = ["pymc3"]
    state["selected_dependencies"] = [ResolvedDependency(name="pymc3", version="3.7")]
    state["target_python"] = "3.12"
    state["deferred_target_python"] = "2.7.18"

    workflow._activate_deferred_python_fallback(state)

    assert state["target_python"] == "2.7.18"
    assert state["python_version_source"] == "deferred_python_fallback"
    assert state["python_fallback_used"] is True
    assert state["pending_python_fallback"] is True
    assert state["selected_dependencies"] == []
    assert state["generated_requirements"] == "# pending model replan after deferred python fallback\n"


def test_version_negotiation_retains_deterministic_fallback_plans(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"
    settings.research_features = ("version_negotiation",)

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "version"
            assert template == "version_negotiation.txt"
            return (
                '{"selected_bundles":['
                '{"rank":1,"reason":"best bundle","dependencies":['
                '{"name":"redis","version":"5.0.0"},'
                '{"name":"sqlalchemy","version":"2.0.0"}'
                "]}]}"
            )

    activity_events: list[tuple[str, int, str, str]] = []
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=PromptRunner(),
        activity_callback=lambda case_id, attempt, kind, detail: activity_events.append((case_id, attempt, kind, detail)),
    )
    case_root = tmp_path / "case-negotiation"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import redis\nimport sqlalchemy\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-negotiation", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["target_python"] = "3.12"
    state["inferred_packages"] = ["redis", "sqlalchemy"]
    state["version_options"] = [
        PackageVersionOptions(package="redis", versions=["5.0.0", "4.0.0"]),
        PackageVersionOptions(package="sqlalchemy", versions=["2.0.0", "1.4.52"]),
    ]
    state["structured_outputs"] = {
        "candidate_bundles": [
            {
                "rank": 1,
                "dependencies": [
                    {"name": "redis", "version": "5.0.0", "platform_notes": ["wheel_available"]},
                    {"name": "sqlalchemy", "version": "2.0.0", "platform_notes": ["wheel_available"]},
                ],
            },
            {
                "rank": 2,
                "dependencies": [
                    {"name": "redis", "version": "4.0.0", "platform_notes": ["wheel_available"]},
                    {"name": "sqlalchemy", "version": "2.0.0", "platform_notes": ["wheel_available"]},
                ],
            },
            {
                "rank": 3,
                "dependencies": [
                    {"name": "redis", "version": "5.0.0", "platform_notes": ["wheel_available"]},
                    {"name": "sqlalchemy", "version": "1.4.52", "platform_notes": ["wheel_available"]},
                ],
            },
        ]
    }

    updated = workflow.negotiate_version_bundles(state)

    assert updated["candidate_plan_strategy"] == "llm+fallback-augmented"
    assert len(updated["candidate_plans"]) == 3
    assert [dependency.version for dependency in updated["candidate_plans"][0].dependencies] == ["5.0.0", "2.0.0"]
    assert [dependency.version for dependency in updated["candidate_plans"][1].dependencies] == ["4.0.0", "2.0.0"]
    assert updated["structured_outputs"]["version_negotiation"]["candidate_plan_strategy"] == "llm+fallback-augmented"


def test_repair_prompt_receives_install_time_python_mismatch_details(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"
    error_message = "ERROR: Package 'pymc3' requires a different Python: 2.7.18 not in '>=3.5.4'"

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "repair"
            assert error_message in variables["error_details"]
            return (
                '{"repair_applicable":true,"plans":['
                '{"rank":1,"reason":"downgrade to a Python 2 compatible release","dependencies":['
                '{"name":"pymc3","version":"3.6.1"}'
                "]}]}"
            )

    activity_events: list[tuple[str, int, str, str]] = []
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=PromptRunner(),
        activity_callback=lambda case_id, attempt, kind, detail: activity_events.append((case_id, attempt, kind, detail)),
    )
    case_root = tmp_path / "case-repair"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import pymc3\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-repair", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["target_python"] = "2.7.18"
    state["current_attempt"] = 2
    state["inferred_packages"] = ["pymc3"]
    state["version_options"] = [
        PackageVersionOptions(
            package="pymc3",
            versions=["3.6.1"],
            requires_python={"3.6.1": ">=2.7"},
        )
    ]
    state["selected_dependencies"] = [ResolvedDependency(name="pymc3", version="3.7")]
    state["last_error_details"] = error_message

    updated = workflow.repair_prompt_c_research(state)

    assert updated["candidate_plan_strategy"] == "llm-selected"
    assert updated["candidate_plans"][0].dependencies[0].version == "3.6.1"
    assert updated["model_outputs"]["repair"]
    event_kinds = [kind for _, _, kind, _ in activity_events]
    assert "repair_cycle_started" in event_kinds
    assert "repair_plan_ready" in event_kinds


def test_repair_prompt_research_discards_already_attempted_plans(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "repair"
            return (
                '{"repair_applicable":true,"plans":['
                '{"rank":1,"reason":"repeat prior downgrade","dependencies":['
                '{"name":"rx","version":"1.6.1"},'
                '{"name":"twisted","version":"19.10.0"}'
                "]}]}"
            )

    workflow = ResolutionWorkflow(settings, prompt_runner=PromptRunner())
    case_root = tmp_path / "case-duplicate-repair"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import rx\nimport twisted\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-duplicate-repair", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-duplicate-repair")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["target_python"] = "2.7.18"
    state["current_attempt"] = 3
    state["inferred_packages"] = ["rx", "twisted"]
    state["version_options"] = [
        PackageVersionOptions(package="rx", versions=["1.6.1"]),
        PackageVersionOptions(package="twisted", versions=["19.10.0"]),
    ]
    state["selected_dependencies"] = [
        ResolvedDependency(name="rx", version="1.6.1"),
        ResolvedDependency(name="twisted", version="19.10.0"),
    ]
    state["attempt_records"] = [
        AttemptRecord(
            attempt_number=2,
            dependencies=["rx==1.6.1", "twisted==19.10.0"],
            image_tag="img-2",
            build_succeeded=True,
            run_succeeded=False,
            exit_code=1,
            error_category="ImportError",
            error_details="cannot import name Disposable",
            validation_command="python snippet.py",
            wall_clock_seconds=1.0,
            artifact_dir=str(case_root / "attempt_02"),
        )
    ]
    state["last_error_details"] = "ImportError: cannot import name Disposable"

    updated = workflow.repair_prompt_c_research(state)

    assert updated["candidate_plans"] == []
    assert updated["remaining_candidate_plans"] == []
    assert updated["repair_model_concluded_impossible"] is False
    assert updated["repair_plan_unavailable_reason"] == "no_novel_plans"


def test_classify_retry_decision_limits_native_build_retries_once_system_packages_were_injected() -> None:
    first = classify_retry_decision("NativeBuildError", system_packages_injected=False, native_retry_used=0)
    second = classify_retry_decision("NativeBuildError", system_packages_injected=True, native_retry_used=1)

    assert first.severity == "limited_retryable"
    assert first.candidate_fallback_allowed is True
    assert first.repair_allowed is True
    assert second.candidate_fallback_allowed is False
    assert second.repair_allowed is False


def test_classify_retry_decision_treats_build_timeout_as_limited_retryable() -> None:
    decision = classify_retry_decision("BuildTimeoutError")

    assert decision.severity == "limited_retryable"
    assert decision.candidate_fallback_allowed is True
    assert decision.repair_allowed is True


def test_classify_retry_decision_routes_system_dependency_without_hints_to_repair_and_fallback() -> None:
    decision = classify_retry_decision("SystemDependencyError", has_system_package_hints=False)

    assert decision.severity == "repair_retryable"
    assert decision.candidate_fallback_allowed is True
    assert decision.repair_allowed is True


def test_classify_retry_decision_defaults_unknown_errors_to_retryable() -> None:
    decision = classify_retry_decision("UnknownError")

    assert decision.severity == "repair_retryable"
    assert decision.candidate_fallback_allowed is True
    assert decision.repair_allowed is True


def test_classify_outcome_uses_build_log_guided_followup_after_native_retry_is_exhausted(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.research_features = ("smart_repair_routing",)
    activity_events: list[tuple[str, int, str, str]] = []
    workflow = ResolutionWorkflow(
        settings,
        activity_callback=lambda case_id, attempt, kind, detail: activity_events.append((case_id, attempt, kind, detail)),
    )

    case_root = tmp_path / "case-native-followup"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import pymc3\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-native-followup", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    dependencies = [
        "numpy==1.16.6",
        "pandas==0.24.2",
        "pymc3==3.6",
        "scipy==1.2.2",
        "theano==1.0.4",
    ]
    state["current_attempt"] = 3
    state["selected_dependencies"] = [ResolvedDependency(name=item.split("==", 1)[0], version=item.split("==", 1)[1]) for item in dependencies]
    state["remaining_candidate_plans"] = [
        CandidatePlan(
            rank=2,
            reason="fallback",
            dependencies=[
                CandidateDependency(name="numpy", version="1.16.6"),
                CandidateDependency(name="pandas", version="0.24.2"),
                CandidateDependency(name="pymc3", version="3.6"),
                CandidateDependency(name="scipy", version="1.2.2"),
                CandidateDependency(name="theano", version="1.0.3"),
            ],
        )
    ]
    state["system_packages_attempted"] = [
        "gfortran",
        "libopenblas-dev",
        "liblapack-dev",
        "build-essential",
        "gcc",
        "g++",
        "libhdf5-dev",
    ]
    state["prepared_execution_context"] = SimpleNamespace(system_packages=list(state["system_packages_attempted"]))
    state["attempt_records"] = [
        AttemptRecord(
            attempt_number=2,
            dependencies=list(dependencies),
            image_tag="img-2",
            build_succeeded=False,
            run_succeeded=False,
            exit_code=1,
            error_category="NativeBuildError",
            error_details="previous failure",
            validation_command=None,
            wall_clock_seconds=10.0,
            artifact_dir=str(case_root / "attempt_02"),
        ),
        AttemptRecord(
            attempt_number=3,
            dependencies=list(dependencies),
            image_tag="img-3",
            build_succeeded=False,
            run_succeeded=False,
            exit_code=1,
            error_category="NativeBuildError",
            error_details="current failure",
            validation_command=None,
            wall_clock_seconds=10.0,
            artifact_dir=str(case_root / "attempt_03"),
        ),
    ]
    state["last_execution"] = ExecutionOutcome(
        success=False,
        category="NativeBuildError",
        message="",
        build_succeeded=False,
        run_succeeded=False,
        exit_code=1,
        build_log=(
            "ERROR: Failed building wheel for h5py\n"
            "error: libhdf5.so: cannot open shared object file: No such file or directory"
        ),
        run_log="",
        image_tag="img-3",
    )

    updated = workflow.classify_outcome(state)

    assert updated["retry_decision"].reason == "build-log-guided-followup"
    assert updated["retry_decision"].candidate_fallback_allowed is True
    assert updated["retry_decision"].repair_allowed is True
    assert updated["strategy_history"][-1].strategy_type == "same_plan_retry"
    assert route_after_research_classification(updated, settings) == "select_next_candidate_plan"
    event_kinds = [kind for _, _, kind, _ in activity_events]
    assert "attempt_classified" in event_kinds
    assert "candidate_fallback_planned" in event_kinds


def test_classify_outcome_uses_native_retry_for_new_hdf5_hint(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.research_features = ("smart_repair_routing",)
    activity_events: list[tuple[str, int, str, str]] = []
    workflow = ResolutionWorkflow(
        settings,
        activity_callback=lambda case_id, attempt, kind, detail: activity_events.append((case_id, attempt, kind, detail)),
    )

    case_root = tmp_path / "case-hdf5-native-retry"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import pymc3\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-hdf5-native-retry", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-hdf5")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    dependencies = [
        "numpy==1.16.6",
        "pandas==0.24.2",
        "pymc3==3.6",
        "scipy==1.2.2",
        "theano==1.0.2",
    ]
    state["current_attempt"] = 5
    state["selected_dependencies"] = [ResolvedDependency(name=item.split("==", 1)[0], version=item.split("==", 1)[1]) for item in dependencies]
    state["system_packages_attempted"] = [
        "gfortran",
        "libopenblas-dev",
        "liblapack-dev",
        "build-essential",
        "gcc",
        "g++",
    ]
    state["prepared_execution_context"] = SimpleNamespace(system_packages=list(state["system_packages_attempted"]))
    state["attempt_records"] = [
        AttemptRecord(
            attempt_number=5,
            dependencies=list(dependencies),
            image_tag="img-5",
            build_succeeded=False,
            run_succeeded=False,
            exit_code=1,
            error_category="NativeBuildError",
            error_details="current failure",
            validation_command=None,
            wall_clock_seconds=10.0,
            artifact_dir=str(case_root / "attempt_05"),
        ),
    ]
    state["last_execution"] = ExecutionOutcome(
        success=False,
        category="NativeBuildError",
        message="",
        build_succeeded=False,
        run_succeeded=False,
        exit_code=1,
        build_log=(
            "ERROR: Failed building wheel for h5py\n"
            "error: libhdf5.so: cannot open shared object file: No such file or directory"
        ),
        run_log="",
        image_tag="img-5",
    )

    updated = workflow.classify_outcome(state)

    assert updated["pending_native_retry"] is True
    assert "libhdf5-dev" in updated["system_packages_attempted"]
    assert "libhdf5-dev" in updated["system_dependencies"]
    assert updated["retry_decision"].reason == "deterministic-native-system-retry"
    assert route_after_research_classification(updated, settings) == "retry_current_plan"
    event_kinds = [kind for _, _, kind, _ in activity_events]
    assert "attempt_classified" in event_kinds
    assert "native_retry_planned" in event_kinds


def test_classify_outcome_uses_bootstrap_retry_for_missing_typing_backport(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.research_features = ("smart_repair_routing",)
    workflow = ResolutionWorkflow(settings)

    case_root = tmp_path / "case-typing-bootstrap"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import twisted\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-typing-bootstrap", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-typing")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["current_attempt"] = 1
    state["selected_dependencies"] = [ResolvedDependency(name="twisted", version="19.10.0")]
    state["attempt_records"] = [
        AttemptRecord(
            attempt_number=1,
            dependencies=["twisted==19.10.0"],
            image_tag="img-1",
            build_succeeded=False,
            run_succeeded=False,
            exit_code=1,
            error_category="ExecutionFailed",
            error_details="",
            validation_command=None,
            wall_clock_seconds=1.0,
            artifact_dir=str(case_root / "attempt_01"),
        )
    ]
    state["last_execution"] = ExecutionOutcome(
        success=False,
        category="ExecutionFailed",
        message="Execution failed.",
        build_succeeded=False,
        run_succeeded=False,
        exit_code=1,
        build_log="ImportError: No module named typing",
        run_log="",
        image_tag="img-1",
    )

    updated = workflow.classify_outcome(state)

    assert updated["pending_native_retry"] is True
    assert updated["bootstrap_dependencies"] == ["typing==3.10.0.0"]
    assert updated["bootstrap_packages_attempted"] == ["typing==3.10.0.0"]
    assert route_after_research_classification(updated, settings) == "retry_current_plan"


def test_workflow_prefers_benchmark_python_when_selected_dependencies_are_directly_compatible(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(
        settings,
        pypi_store=TargetAwarePyPIStore(
            {
                ("scrapy", "3.12"): ["2.11.2", "2.10.1", "2.9.0"],
            }
        ),
    )
    state = workflow.initial_state_for_case(
        BenchmarkCase(
            case_id="case-benchmark-target-preference",
            root_dir=tmp_path,
            snippet_path=tmp_path / "snippet.py",
            dockerfile_path=tmp_path / "Dockerfile",
        )
    )
    state["source_files"] = {"snippet.py": "import scrapy\n"}
    state["extracted_imports"] = ["scrapy"]
    state["selected_dependencies"] = [workflow._deterministic_dependencies([PackageVersionOptions(package="scrapy", versions=["2.11.2"])])[0]]
    state["benchmark_target_python"] = "3.12"
    state["target_python"] = "3.8"
    state["python_version_source"] = "llm_prompt_a"

    workflow._maybe_prefer_benchmark_target_python(state)

    assert state["target_python"] == "3.12"
    assert state["python_version_source"] == "benchmark_dockerfile_preferred_compatible"


def test_extract_imports_records_deferred_python_fallback_for_gist_py2_syntax(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(settings)
    case_root = tmp_path / "case-py2-guardrail"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    dockerfile = case_root / "Dockerfile"
    snippet.write_text("import requests\nprint 'hello world'\n", encoding="utf-8")
    dockerfile.write_text("", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(
            case_id="case-py2-guardrail",
            root_dir=case_root,
            snippet_path=snippet,
            dockerfile_path=dockerfile,
            case_source="competition-run",
            initial_eval="ImportError",
        )
    )

    state = workflow.load_target(state)
    state = workflow.extract_imports(state)

    assert state["benchmark_target_python"] == "3.12"
    assert state["target_python"] == "3.12"
    assert state["deferred_target_python"] == "2.7.18"
    assert state["python_version_source"] == "benchmark_default"


def test_classify_outcome_activates_deferred_python_fallback_after_runtime_syntax_error(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(
        settings,
        pypi_store=TargetAwarePyPIStore({("requests", "2.7.18"): ["2.20.1"]}),
    )
    case_root = tmp_path / "case-py2-fallback"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import requests\nprint 'hello world'\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-py2-fallback", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-py2-fallback")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["extracted_imports"] = ["requests"]
    state["inferred_packages"] = ["requests"]
    state["current_attempt"] = 1
    state["benchmark_target_python"] = "3.12"
    state["target_python"] = "3.12"
    state["deferred_target_python"] = "2.7.18"
    state["selected_dependencies"] = [ResolvedDependency(name="requests", version="2.32.3")]
    state["attempt_records"] = [
        AttemptRecord(
            attempt_number=1,
            dependencies=["requests==2.32.3"],
            image_tag="img-1",
            build_succeeded=True,
            run_succeeded=False,
            exit_code=1,
            error_category="ExecutionFailed",
            error_details="",
            validation_command="python snippet.py",
            wall_clock_seconds=1.0,
            artifact_dir=str(case_root / "attempt_01"),
        )
    ]
    state["last_execution"] = ExecutionOutcome(
        success=False,
        category="ExecutionFailed",
        message="Execution failed.",
        build_succeeded=True,
        run_succeeded=False,
        exit_code=1,
        build_log="",
        run_log="SyntaxError: invalid syntax",
        image_tag="img-1",
    )

    updated = workflow.classify_outcome(state)

    assert updated["pending_python_fallback"] is True
    assert updated["target_python"] == "2.7.18"
    assert updated["python_version_source"] == "deferred_python_fallback"
    assert updated["python_fallback_used"] is True
    assert updated["selected_dependencies"] == []
    assert updated["classifier_origin"] == "run"
    assert route_after_research_classification(updated, settings) == "replan_after_python_fallback"


def test_classify_outcome_reserves_last_attempt_for_deferred_python_fallback_after_build_only_failures(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.max_attempts = 3
    workflow = ResolutionWorkflow(settings)
    case_root = tmp_path / "case-py2-build-fallback"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import pymc3\nprint 'hello world'\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-py2-build-fallback", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-py2-build-fallback")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["target_python"] = "3.8"
    state["deferred_target_python"] = "2.7.18"
    state["current_attempt"] = 2
    state["selected_dependencies"] = [
        ResolvedDependency(name="numpy", version="1.24.4"),
        ResolvedDependency(name="pymc3", version="3.11.5"),
        ResolvedDependency(name="scipy", version="1.10.0"),
    ]
    state["attempt_records"] = [
        AttemptRecord(
            attempt_number=1,
            dependencies=["numpy==1.24.4", "pymc3==3.11.6", "scipy==1.10.1"],
            image_tag="img-1",
            build_succeeded=False,
            run_succeeded=False,
            exit_code=1,
            error_category="ResolutionError",
            error_details="ResolutionImpossible",
            validation_command="python snippet.py",
            wall_clock_seconds=1.0,
            artifact_dir=str(case_root / "attempt_01"),
        ),
        AttemptRecord(
            attempt_number=2,
            dependencies=["numpy==1.24.4", "pymc3==3.11.5", "scipy==1.10.0"],
            image_tag="img-2",
            build_succeeded=False,
            run_succeeded=False,
            exit_code=1,
            error_category="ExecutionFailed",
            error_details="",
            validation_command="python snippet.py",
            wall_clock_seconds=1.0,
            artifact_dir=str(case_root / "attempt_02"),
        ),
    ]
    state["last_execution"] = ExecutionOutcome(
        success=False,
        category="ExecutionFailed",
        message="Execution failed.",
        build_succeeded=False,
        run_succeeded=False,
        exit_code=1,
        build_log="ERROR: ResolutionImpossible",
        run_log="",
        image_tag="img-2",
    )

    updated = workflow.classify_outcome(state)

    assert updated["pending_python_fallback"] is True
    assert updated["target_python"] == "2.7.18"
    assert updated["python_fallback_used"] is True
    assert updated["retry_decision"].reason == "reserved-deferred-python-fallback"
    assert route_after_research_classification(updated, settings) == "replan_after_python_fallback"


def test_replan_after_python_fallback_uses_model_selected_plan_and_runtime_profile(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"
    settings.research_features = ()

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "version"
            assert template == "candidate_plans.txt"
            assert "2.20.1" in variables["version_space"]
            return (
                '{"plans":[{"rank":1,"reason":"py2 compatible requests line","runtime_profile":"import_statements",'
                '"dependencies":[{"name":"requests","version":"2.20.1"}]}]}'
            )

        @staticmethod
        def invoke_text(stage: str, prompt_text: str) -> str:
            raise AssertionError("Adjudication should not be needed in this test")

    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=PromptRunner(),
        pypi_store=TargetAwarePyPIStore({("requests", "2.7.18"): ["2.20.1", "2.18.0"]}),
    )
    case_root = tmp_path / "case-py2-replan"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import requests\nprint 'hello world'\n", encoding="utf-8")
    state = workflow.initial_state_for_case(
        BenchmarkCase(case_id="case-py2-replan", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    )
    state["artifact_dir"] = str(tmp_path / "artifacts-py2-replan")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["extracted_imports"] = ["requests"]
    state["inferred_packages"] = ["requests"]
    state["target_python"] = "2.7.18"
    state["pending_python_fallback"] = True
    state["validation_options"] = [
        {"profile": "snippet_exec", "command": "python snippet.py", "reason": "default"},
        {"profile": "import_statements", "command": "python - <<'PY'\nprint('imports-ok')\nPY", "reason": "safe"},
    ]
    state["default_validation_profile"] = "snippet_exec"
    state["current_runtime_profile"] = "snippet_exec"
    state["current_validation_command"] = "python snippet.py"

    updated = workflow.replan_after_python_fallback(state)

    assert updated["pending_python_fallback"] is False
    assert [dependency.pin() for dependency in updated["selected_dependencies"]] == ["requests==2.20.1"]
    assert updated["selected_candidate_plan"].runtime_profile == "import_statements"
    assert updated["current_runtime_profile"] == "import_statements"


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
            version_space="{}",
            rag_context="{}",
            validation_options="[]",
            default_validation_profile="docker_cmd",
            candidate_bundle_hints="{}",
            conflict_notes="[]",
            source_compatibility_hints="[]",
            max_plan_count=3,
        ),
        "repair_attempt_v2": workflow._format_prompt(
            "repair_attempt_v2.txt",
            target_python="3.12",
            allowed_packages="PyYAML",
            version_space="{}",
            previous_plan="PyYAML==6.0.2",
            attempted_plans="",
            error_details="ModuleNotFoundError",
            repair_memory="{}",
            feedback_summary="{}",
            rag_context="{}",
            validation_options="[]",
            default_validation_profile="docker_cmd",
            conflict_notes="[]",
            source_compatibility_hints="[]",
        ),
        "version_negotiation": workflow._format_prompt(
            "version_negotiation.txt",
            target_python="3.12",
            candidate_bundles="[]",
            conflict_notes="[]",
            source_compatibility_hints="[]",
            repo_evidence="{}",
        ),
        "source_compatibility": workflow._format_prompt(
            "source_compatibility.txt",
            target_python="3.12",
            allowed_packages="tensorflow\ngensim",
            version_space="{}",
            validation_options="[]",
            default_validation_profile="docker_cmd",
            source_signals="[]",
            raw_file="import tensorflow as tf\n",
        ),
        "resolve_aliases": workflow._format_prompt(
            "resolve_aliases.txt",
            unresolved_packages="memcache\nyaml",
            raw_file="import memcache\nimport yaml\n",
        ),
    }

    assert all(rendered.values())
    assert "Target Python:" not in rendered["package_inference"]


def test_format_prompt_backfills_optional_research_variables(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"
    workflow = ResolutionWorkflow(settings)

    rendered = workflow._format_prompt(
        "candidate_plans_v2.txt",
        target_python="3.12",
        allowed_packages="PyYAML",
        version_space="{}",
        rag_context="{}",
        validation_options="[]",
        default_validation_profile="docker_cmd",
        candidate_bundle_hints="{}",
        source_compatibility_hints="[]",
        max_plan_count=3,
    )

    assert "Conflict notes:" in rendered


def test_parse_alias_resolution_payload_is_strict() -> None:
    parsed = parse_alias_resolution_payload(
        '{"aliases":[{"import_name":"memcache","pypi_package":"python-memcached"}]}'
    )
    assert parsed == [{"import_name": "memcache", "pypi_package": "python-memcached"}]

    try:
        parse_alias_resolution_payload('{"aliases":[{"import_name":"memcache"}]}')
    except StructuredOutputError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Missing pypi_package should raise StructuredOutputError.")


def test_resolve_aliases_converts_unresolved_imports_to_validated_packages(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"
    workflow = ResolutionWorkflow(settings, prompt_runner=FakeAliasPromptRunner(), pypi_store=FakeAliasPyPIStore())

    case_root = tmp_path / "case"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import memcache\n", encoding="utf-8")
    case = BenchmarkCase(case_id="case-1", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    state = workflow.initial_state_for_case(case, run_id="run-1")
    state["source_files"] = {"snippet.py": "import memcache\n"}
    state["target_python"] = "3.12"
    state["inferred_packages"] = ["memcache"]
    state["candidate_provenance"] = {"memcache": "ast"}
    state["version_options"] = []
    state["unresolved_packages"] = ["memcache"]
    state["structured_outputs"] = {}

    updated = workflow.resolve_aliases(state)

    assert updated["unresolved_packages"] == []
    assert "python-memcached" in updated["inferred_packages"]
    assert "memcache" not in updated["inferred_packages"]
    assert updated["version_options"][0].package == "python-memcached"
    assert updated["candidate_provenance"]["python-memcached"] == "alias"


def test_research_extract_backfills_known_runtime_aliases_omitted_by_llm(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "extract"
            assert template == "initial_imports.txt"
            assert "memcache" in variables["extracted_imports"]
            return (
                '{"packages":['
                '{"package":"redis","confidence":1.0,"source":"import","evidence":["import redis"]}'
                '],"python_version":"3.12"}'
            )

    workflow = ResolutionWorkflow(settings, prompt_runner=PromptRunner())

    case_root = tmp_path / "case"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import memcache\nimport redis\n", encoding="utf-8")
    case = BenchmarkCase(case_id="case-1", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    state = workflow.initial_state_for_case(case, run_id="run-1")
    state["source_files"] = {"snippet.py": "import memcache\nimport redis\n"}
    state["extracted_imports"] = ["memcache", "redis"]
    state["repo_evidence"] = {}

    updated = workflow._infer_packages_prompt_a_research(state, state["extracted_imports"])

    assert updated["inferred_packages"] == ["python-memcached", "redis"]
    assert updated["candidate_provenance"]["python-memcached"] == "alias"


def test_parse_package_inference_output_extracts_python_version_from_markdown_wrapped_json() -> None:
    packages, python_version = parse_package_inference_output(
        """
Based on the code:

```json
{
  "packages": [
    {"package": "tensorflow"},
    {"package": "numpy"}
  ],
  "python_version": "3.7"
}
```
""".strip()
    )

    assert packages == ["tensorflow", "numpy"]
    assert python_version == "3.7"


def test_research_extract_preserves_python_version_from_markdown_wrapped_json(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "extract"
            assert template == "initial_imports.txt"
            return """
Legacy snippet analysis:

```json
{
  "packages": [
    {"package": "tensorflow", "confidence": 1.0, "source": "code", "evidence": ["import tensorflow as tf"]},
    {"package": "numpy", "confidence": 1.0, "source": "code", "evidence": ["import numpy as np"]}
  ],
  "python_version": "3.7"
}
```
""".strip()

        @staticmethod
        def invoke_text(stage: str, prompt_text: str) -> str:
            assert stage == "adjudicate"
            return (
                '{"packages":['
                '{"package":"tensorflow","confidence":1.0,"source":"code","evidence":["import tensorflow as tf"]},'
                '{"package":"numpy","confidence":1.0,"source":"code","evidence":["import numpy as np"]}'
                '],"python_version":"3.7"}'
            )

    workflow = ResolutionWorkflow(settings, prompt_runner=PromptRunner())

    case_root = tmp_path / "case-py37"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import tensorflow as tf\nimport numpy as np\n", encoding="utf-8")
    case = BenchmarkCase(case_id="case-py37", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    state = workflow.initial_state_for_case(case, run_id="run-1")
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["extracted_imports"] = ["tensorflow", "numpy"]
    state["repo_evidence"] = {}
    state["benchmark_target_python"] = "3.12"
    state["target_python"] = "3.12"
    state["python_version_source"] = "benchmark_default"

    updated = workflow._infer_packages_prompt_a_research(state, state["extracted_imports"])

    assert updated["inferred_target_python"] == "3.7"
    assert updated["target_python"] == "3.7"
    assert updated["python_version_source"] == "llm_prompt_a"


def test_finalize_result_handles_unsupported_imports_without_execution(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    workflow = ResolutionWorkflow(settings)

    case_root = tmp_path / "case"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("from PyQt4 import QtGui\nimport maya.OpenMayaUI as mui\n", encoding="utf-8")
    case = BenchmarkCase(case_id="case-unsupported", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    state = workflow.initial_state_for_case(case, run_id="run-1")
    state["artifact_dir"] = str(tmp_path / "artifacts")
    Path(state["artifact_dir"]).mkdir(parents=True, exist_ok=True)
    state["source_files"] = {"snippet.py": snippet.read_text(encoding="utf-8")}
    state["unsupported_imports"] = ["PyQt4", "maya"]
    state["selected_dependencies"] = []
    state["candidate_plans"] = [CandidatePlan(rank=1, reason="imports require unsupported external runtime packages", dependencies=[])]
    state["selected_candidate_plan"] = state["candidate_plans"][0]
    state["dependency_reason"] = "unsupported_imports"
    state["repair_skipped_reason"] = "unsupported_imports_only"

    final_state = workflow.finalize_result(state)

    assert final_state["final_result"]["success"] is False
    assert final_state["final_result"]["final_error_category"] == "UnsupportedImportError"
    assert final_state["final_result"]["dependency_reason"] == "unsupported_imports"
    assert final_state["final_result"]["repair_skipped_reason"] == "unsupported_imports_only"


def test_parse_candidate_plan_payload_rejects_missing_required_packages() -> None:
    raw_output = (
        '{"plans":[{"rank":1,"reason":"short reason","dependencies":'
        '[{"name":"redis","version":"5.0.0"}]}]}'
    )

    try:
        parse_candidate_plan_payload(
            raw_output,
            allowed_packages={"redis", "sqlalchemy"},
            allowed_versions={"redis": {"5.0.0"}, "sqlalchemy": {"2.0.0"}},
            required_packages={"redis", "sqlalchemy"},
        )
    except StructuredOutputError as exc:
        assert "missing required packages" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Missing required packages should raise StructuredOutputError.")


def test_parse_candidate_plan_payload_rejects_duplicate_normalized_packages() -> None:
    raw_output = (
        '{"plans":[{"rank":1,"reason":"duplicate package","dependencies":'
        '[{"name":"GitPython","version":"3.1.18"},{"name":"gitpython","version":"3.1.18"}]}]}'
    )

    try:
        parse_candidate_plan_payload(
            raw_output,
            allowed_packages={"gitpython"},
            allowed_versions={"gitpython": {"3.1.18"}},
            required_packages={"gitpython"},
        )
    except StructuredOutputError as exc:
        assert "duplicates another package after normalization" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Duplicate normalized packages should raise StructuredOutputError.")


def test_parse_candidate_plan_payload_accepts_runtime_profile() -> None:
    raw_output = (
        '{"plans":[{"rank":1,"reason":"safer validation","runtime_profile":"import_statements","dependencies":'
        '[{"name":"tensorflow","version":"2.12.1"}]}]}'
    )

    plans = parse_candidate_plan_payload(
        raw_output,
        allowed_packages={"tensorflow"},
        allowed_versions={"tensorflow": {"2.12.1"}},
        required_packages={"tensorflow"},
        allowed_runtime_profiles={"docker_cmd", "import_statements"},
    )

    assert plans[0].runtime_profile == "import_statements"


def test_parse_repair_plan_payload_merges_partial_repairs_and_skips_invalid_plans() -> None:
    raw_output = (
        '{"repair_applicable":true,"plans":['
        '{"rank":1,"reason":"downgrade tensorflow","dependencies":[{"name":"tensorflow","version":"2.12.0"}]},'
        '{"rank":2,"reason":"invalid helper","dependencies":[{"name":"python","version":"3.7"}]},'
        '{"rank":3,"reason":"switch validation only","runtime_profile":"import_smoke","dependencies":[]}'
        "]}"
    )

    repair_applicable, plans = parse_repair_plan_payload(
        raw_output,
        allowed_packages={"gym", "keras", "numpy", "tensorflow"},
        allowed_versions={
            "gym": {"0.25.2"},
            "keras": {"2.11.0"},
            "numpy": {"1.24.1"},
            "tensorflow": {"2.13.1", "2.12.0"},
        },
        required_packages={"gym", "keras", "numpy", "tensorflow"},
        allowed_runtime_profiles={"import_smoke", "import_statements"},
        previous_plan=[
            CandidateDependency(name="gym", version="0.25.2"),
            CandidateDependency(name="keras", version="2.11.0"),
            CandidateDependency(name="numpy", version="1.24.1"),
            CandidateDependency(name="tensorflow", version="2.13.1"),
        ],
    )

    assert repair_applicable is True
    assert len(plans) == 2
    assert [dependency.pin() for dependency in plans[0].dependencies] == [
        "gym==0.25.2",
        "keras==2.11.0",
        "numpy==1.24.1",
        "tensorflow==2.12.0",
    ]
    assert plans[1].runtime_profile == "import_smoke"


def test_build_constraint_pack_uses_full_version_space_for_python_intersection() -> None:
    pack = build_constraint_pack(
        [
            PackageVersionOptions(
                package="tensorflow",
                versions=["2.12.1", "2.11.0", "1.15.5"],
                requires_python={
                    "2.12.1": ">=3.8",
                    "2.11.0": ">=3.8",
                    "1.15.5": ">=2.7",
                },
            )
        ],
        target_python="2.7.18",
    )

    assert pack.python_intersection_valid is True
    assert pack.candidate_versions["tensorflow"] == ["2.12.1", "2.11.0", "1.15.5"]


def test_summarize_rag_context_keeps_high_signal_fields_within_limit() -> None:
    summary = summarize_rag_context(
        {
            "target_python": "2.7.18",
            "research_bundle": "enhanced",
            "research_features": ["transitive_conflicts", "python_constraint_intersection"],
            "imports": ["numpy", "tensorflow", "Image"],
            "inferred_packages": ["numpy", "tensorflow"],
            "unresolved_packages": ["Image"],
            "repo_evidence": {
                "mode": "gistable",
                "dockerfile_summary": "FROM python:2.7.18-slim",
                "source_summary": "import tensorflow as tf",
            },
            "pypi_evidence": {
                "packages": [
                    {
                        "package": "tensorflow",
                        "versions": ["0.12.0"],
                        "requires_dist": {"0.12.0": ["keras>=3.10.0", "numpy>=1.26.0"]},
                    }
                ],
                "alias_resolution": {"resolved_aliases": [{"import_name": "memcache", "pypi_package": "python-memcached"}]},
            },
            "version_summaries": [
                {
                    "package": "tensorflow",
                    "versions": ["0.12.0", "0.11.0"],
                    "requires_python": {"0.12.0": ">=2.7"},
                    "requires_dist": {"0.12.0": ["keras>=3.10.0", "numpy>=1.26.0"]},
                    "policy_notes": [],
                }
            ],
            "version_conflict_notes": [
                {
                    "package": "tensorflow",
                    "related_package": "numpy",
                    "kind": "requires_dist",
                    "reason": "tensorflow has conflicting numpy constraints",
                    "severity": "warning",
                }
            ],
        },
        limit=700,
    )

    assert len(summary) <= 700
    assert '"package_versions"' in summary
    assert '"alias_resolution"' in summary
    assert '"version_conflict_notes"' in summary
    assert "tensorflow" in summary
    assert "keras>=3.10.0" in summary
    assert '"pypi_evidence"' not in summary


def test_generate_candidate_plans_falls_back_when_llm_omits_required_packages(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"

    class PromptRunner:
        @staticmethod
        def stage_model(stage: str) -> str:
            return stage

        @staticmethod
        def invoke_template(stage: str, template: str, variables: dict[str, str]) -> str:
            assert stage == "version"
            assert template in {"candidate_plans.txt", "candidate_plans_v2.txt"}
            return (
                '{"plans":[{"rank":1,"reason":"bad partial plan","dependencies":'
                '[{"name":"redis","version":"5.0.0"}]}]}'
            )

        @staticmethod
        def invoke_text(stage: str, prompt_text: str) -> str:
            assert stage == "adjudicate"
            return (
                '{"plans":[{"rank":1,"reason":"bad partial plan","dependencies":'
                '[{"name":"redis","version":"5.0.0"}]}]}'
            )

    workflow = ResolutionWorkflow(settings, prompt_runner=PromptRunner())

    case_root = tmp_path / "case"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import redis\nimport sqlalchemy\n", encoding="utf-8")
    case = BenchmarkCase(case_id="case-1", root_dir=case_root, snippet_path=snippet, case_source="all-gists")
    state = workflow.initial_state_for_case(case, run_id="run-1")
    state["source_files"] = {"snippet.py": "import redis\nimport sqlalchemy\n"}
    state["target_python"] = "3.12"
    state["current_attempt"] = 0
    state["inferred_packages"] = ["redis", "sqlalchemy"]
    state["version_options"] = [
        PackageVersionOptions(package="redis", versions=["5.0.0", "4.0.0"]),
        PackageVersionOptions(package="sqlalchemy", versions=["2.0.0", "1.4.52"]),
    ]
    state["rag_context"] = {
        "target_python": "3.12",
        "imports": ["redis", "sqlalchemy"],
        "inferred_packages": ["redis", "sqlalchemy"],
        "version_summaries": [
            {"package": "redis", "versions": ["5.0.0", "4.0.0"], "requires_dist": {}, "policy_notes": []},
            {"package": "sqlalchemy", "versions": ["2.0.0", "1.4.52"], "requires_dist": {}, "policy_notes": []},
        ],
    }

    updated = workflow.generate_candidate_plans(state)

    assert [dependency.name for dependency in updated["candidate_plans"][0].dependencies] == ["redis", "sqlalchemy"]
    assert updated["structured_prompt_failures"] == 1
