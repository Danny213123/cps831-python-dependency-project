from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_python_dependency.config import Settings
from agentic_python_dependency.graph import ResolutionWorkflow
from agentic_python_dependency.state import BenchmarkCase, PackageVersionOptions, ResolvedDependency
from agentic_python_dependency.tools.docker_executor import DockerExecutionResult
from agentic_python_dependency.tools.official_baselines import OfficialBaselinePlan


@dataclass
class PreparedContext:
    context_dir: Path
    dockerfile_path: Path
    image_tag: str
    validation_command: str | None
    artifact_dir: Path


class FakePromptRunner:
    def __init__(self, scripted: dict[str, list[str]]):
        self.scripted = scripted

    @staticmethod
    def stage_model(stage: str) -> str:
        return stage

    def invoke_template(self, stage: str, template_name: str, variables: dict[str, str]) -> str:
        return self.scripted[stage].pop(0)

    def invoke_text(self, stage: str, prompt_text: str) -> str:
        return self.scripted[stage].pop(0)


class FakePyPIStore:
    def __init__(self, options: dict[str, list[str] | PackageVersionOptions]):
        self.options = options

    def get_version_options(
        self,
        package: str,
        target_python: str,
        limit: int = 20,
        *,
        preset: str = "optimized",
    ) -> PackageVersionOptions:
        if package not in self.options:
            raise FileNotFoundError(package)
        configured = self.options[package]
        if isinstance(configured, PackageVersionOptions):
            return configured
        return PackageVersionOptions(package=package, versions=configured)

    @staticmethod
    def format_prompt_block(options: list[PackageVersionOptions]) -> str:
        return "\n".join(f"{option.package}: {', '.join(option.versions)}" for option in options)

    @staticmethod
    def release_files(package: str, version: str) -> list[dict[str, str]]:
        return [{"url": f"https://example.invalid/{package}/{version}.whl"}]


class FakeDockerExecutor:
    def __init__(self, results: list[DockerExecutionResult]):
        self.results = results

    def _prepared(self, artifact_dir: Path, image_tag: str, validation_command: str | None) -> PreparedContext:
        context_dir = artifact_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path = context_dir / "Dockerfile.generated"
        dockerfile_path.write_text("FROM python:3.12-slim\n", encoding="utf-8")
        (context_dir / "requirements.generated.txt").write_text("", encoding="utf-8")
        return PreparedContext(context_dir, dockerfile_path, image_tag, validation_command, artifact_dir)

    def prepare_benchmark_context(
        self,
        case,
        dependencies,
        artifact_dir: Path,
        image_tag: str,
        target_python: str = "3.12",
        validation_command: str | None = None,
        extra_system_packages: list[str] | None = None,
        extra_bootstrap_pins: list[str] | None = None,
        case_id: str = "",
        attempt_number: int = 0,
    ) -> PreparedContext:
        return self._prepared(artifact_dir, image_tag, validation_command)

    def prepare_project_context(
        self,
        target,
        dependencies,
        artifact_dir: Path,
        image_tag: str,
        extra_system_packages: list[str] | None = None,
        extra_bootstrap_pins: list[str] | None = None,
        case_id: str = "",
        attempt_number: int = 0,
    ) -> PreparedContext:
        return self._prepared(artifact_dir, image_tag, target.validation_command)

    def execute(self, context: PreparedContext) -> DockerExecutionResult:
        return self.results.pop(0)


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings.from_env(project_root=tmp_path)
    settings.prompts_dir = Path(__file__).resolve().parents[2] / "src" / "agentic_python_dependency" / "prompts"
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return settings


def write_project(tmp_path: Path, source: str) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "example.py").write_text(source, encoding="utf-8")
    return project_root


def write_benchmark_case(tmp_path: Path, case_id: str, snippet: str, dockerfile: str, initial_eval: str = "ImportError") -> BenchmarkCase:
    case_root = tmp_path / case_id
    case_root.mkdir()
    snippet_path = case_root / "snippet.py"
    dockerfile_path = case_root / "Dockerfile"
    snippet_path.write_text(snippet, encoding="utf-8")
    dockerfile_path.write_text(dockerfile, encoding="utf-8")
    return BenchmarkCase(
        case_id=case_id,
        root_dir=case_root,
        snippet_path=snippet_path,
        dockerfile_path=dockerfile_path,
        initial_eval=initial_eval,
    )


def test_workflow_repairs_retryable_dependency_failure(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import yaml\nprint('ok')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["yaml"],
                "version": ["PyYAML==0.0.0"],
                "repair": ["PyYAML==6.0.2"],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=False,
                    exit_code=1,
                    build_log="",
                    run_log="ModuleNotFoundError: No module named 'yaml'",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                ),
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-2",
                    wall_clock_seconds=0.1,
                ),
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is True
    assert final_state["final_result"]["attempts"] == 2


def test_workflow_adjudicates_malformed_repair_output(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import yaml\nprint('ok')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["yaml"],
                "version": ["PyYAML==0.0.0"],
                "repair": ["The error messages indicate a few problems:\nPyYAML==6.0.2"],
                "adjudicate": ["PyYAML==6.0.2"],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=False,
                    exit_code=1,
                    build_log="",
                    run_log="ModuleNotFoundError: No module named 'yaml'",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                ),
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-2",
                    wall_clock_seconds=0.1,
                ),
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is True
    assert final_state["final_result"]["dependencies"] == ["PyYAML==6.0.2"]


def test_workflow_prompt_a_can_infer_target_python(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "# prompt-driven python version inference\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ['{"modules":["PyYAML"],"python_version":"3.11"}'],
                "version": ["PyYAML==6.0.2"],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["target_python"] == "3.11"
    assert final_state["inferred_target_python"] == "3.11"
    assert final_state["python_version_source"] == "llm_prompt_a"
    assert final_state["final_result"]["target_python"] == "3.11"
    assert final_state["final_result"]["inferred_target_python"] == "3.11"


def test_benchmark_prompt_a_python_guardrail_keeps_python2_for_python2_only_syntax(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    case = write_benchmark_case(
        tmp_path,
        "case_guardrail",
        "print 'python2 only'\n",
        "FROM python:2.7.13\nCMD [\"python\", \"snippet.py\"]\n",
    )
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ['{"modules":["requests"],"python_version":"3.11"}'],
                "version": ["requests==2.31.0"],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"requests": ["2.31.0"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-guardrail",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_case(case))

    assert final_state["benchmark_target_python"] == "2.7.13"
    assert final_state["inferred_target_python"] == "3.11"
    assert final_state["target_python"] == "2.7.13"
    assert final_state["python_version_source"] == "benchmark_dockerfile_syntax_guardrail"


def test_workflow_filters_unrelated_repair_dependencies(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import yaml\nprint('ok')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["yaml"],
                "version": ["PyYAML==0.0.0"],
                "repair": ["PyYAML==6.0.2\npip==26.0.1"],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=False,
                    exit_code=1,
                    build_log="",
                    run_log="ModuleNotFoundError: No module named 'yaml'",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                ),
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-2",
                    wall_clock_seconds=0.1,
                ),
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is True
    assert final_state["final_result"]["dependencies"] == ["PyYAML==6.0.2"]


def test_workflow_stops_repair_loop_after_repeated_unusable_repair_output(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import yaml\nprint('ok')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["yaml"],
                "version": ["PyYAML==0.0.0"],
                "repair": [
                    "The package appears incompatible with Python 2.7.",
                    "The package appears incompatible with Python 2.7.",
                ],
                "adjudicate": ["```", "```"],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=False,
                    exit_code=1,
                    build_log="",
                    run_log="ModuleNotFoundError: No module named 'yaml'",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                ),
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=False,
                    exit_code=1,
                    build_log="",
                    run_log="ModuleNotFoundError: No module named 'yaml'",
                    image_tag="img-2",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is False
    assert final_state["final_result"]["attempts"] == 2
    assert final_state["final_result"]["stop_reason"] == "RepairOutputStalled"


def test_workflow_skips_version_prompt_when_no_compatible_versions_exist(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import yaml\n")
    prompt_runner = FakePromptRunner({"extract": ["yaml"], "version": [], "repair": [], "adjudicate": []})
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=prompt_runner,
        pypi_store=FakePyPIStore({"PyYAML": []}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["unresolved_packages"] == ["PyYAML"]
    assert final_state["final_result"]["dependencies"] == []


def test_workflow_uses_single_version_fast_path(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import yaml\n")
    prompt_runner = FakePromptRunner({"extract": ["yaml"], "version": [], "repair": [], "adjudicate": []})
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=prompt_runner,
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["dependencies"] == ["PyYAML==6.0.2"]


def test_workflow_marks_skipped_prompt_b_for_stdlib_only_project(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import argparse\nimport os\nprint('ok')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["argparse\nos"],
                "version": [],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["prompt_history"]["prompt_b"].startswith("# skipped:")
    assert final_state["generated_requirements"] == "# no inferred third-party dependencies\n"
    assert final_state["final_result"]["success"] is True


def test_workflow_uses_deterministic_version_selector_for_performance_preset(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.preset = "performance"
    settings.prompt_profile = "optimized-lite"
    settings.max_attempts = 2
    project_root = write_project(tmp_path, "import requests\nprint('ok')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["requests"],
                "version": [],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"requests": ["2.32.3", "2.31.0"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["dependency_reason"] == "deterministic_version_selector"
    assert final_state["version_selection_source"] == "deterministic_version_selector"
    assert final_state["final_result"]["dependencies"] == ["requests==2.32.3"]


def test_workflow_pyego_requires_official_baseline_when_unavailable(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.resolver = "pyego"
    project_root = write_project(tmp_path, "import yaml\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner({"extract": [], "version": [], "repair": [], "adjudicate": []}),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["version_selection_source"] == "pyego_official_required"
    assert final_state["dependency_reason"] == "official_baseline_unavailable"
    assert final_state["resolver_implementation"] == "official-required"
    assert final_state["last_execution"].category == "OfficialBaselineUnavailable"
    assert final_state["final_result"]["resolver"] == "pyego"
    assert final_state["final_result"]["resolver_implementation"] == "official-required"
    assert final_state["final_result"]["final_error_category"] == "OfficialBaselineUnavailable"
    assert final_state["final_result"]["dependencies"] == []
    assert final_state["final_result"]["attempts"] == 0


def test_workflow_pyego_prefers_official_baseline_when_available(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.resolver = "pyego"
    project_root = write_project(tmp_path, "import yaml\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner({"extract": [], "version": [], "repair": [], "adjudicate": []}),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    def fake_official(_: ResolutionWorkflow, state: dict[str, Any]) -> tuple[OfficialBaselinePlan | None, str]:
        return (
            OfficialBaselinePlan(
                target_python="3.11",
                dependencies=[ResolvedDependency(name="PyYAML", version="6.0.1")],
                system_packages=["libyaml-dev"],
                implementation="official",
                raw_payload={"source": "test"},
            ),
            "pyego_official",
        )

    monkeypatch.setattr(ResolutionWorkflow, "_run_official_baseline", fake_official)
    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["version_selection_source"] == "pyego_official"
    assert final_state["resolver_implementation"] == "official"
    assert final_state["final_result"]["resolver_implementation"] == "official"
    assert final_state["final_result"]["system_dependencies"] == ["libyaml-dev"]
    assert final_state["final_result"]["dependencies"] == ["PyYAML==6.0.1"]


def test_workflow_readpye_uses_static_imports_and_unpinned_packages(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.resolver = "readpye"
    project_root = write_project(tmp_path, "import yaml\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner({"extract": [], "version": [], "repair": [], "adjudicate": []}),
        pypi_store=FakePyPIStore({}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["version_selection_source"] == "readpye_unpinned"
    assert final_state["dependency_reason"] == "readpye_unpinned"
    assert final_state["final_result"]["resolver"] == "readpye"
    assert final_state["final_result"]["dependencies"] == ["PyYAML"]


def test_workflow_readpye_prefers_official_baseline_when_available(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.resolver = "readpye"
    project_root = write_project(tmp_path, "import yaml\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner({"extract": [], "version": [], "repair": [], "adjudicate": []}),
        pypi_store=FakePyPIStore({}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    def fake_official(_: ResolutionWorkflow, state: dict[str, Any]) -> tuple[OfficialBaselinePlan | None, str]:
        return (
            OfficialBaselinePlan(
                target_python="2.7.18",
                dependencies=[ResolvedDependency(name="PyYAML", version="")],
                system_packages=[],
                implementation="official",
                raw_payload={"source": "test"},
            ),
            "readpye_official",
        )

    monkeypatch.setattr(ResolutionWorkflow, "_run_official_baseline", fake_official)
    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["version_selection_source"] == "readpye_official"
    assert final_state["resolver_implementation"] == "official"
    assert final_state["final_result"]["resolver_implementation"] == "official"
    assert final_state["final_result"]["dependencies"] == ["PyYAML"]


def test_workflow_pyego_does_not_enter_repair_loop(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.resolver = "pyego"
    project_root = write_project(tmp_path, "import yaml\nprint('ok')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner({"extract": [], "version": [], "repair": [], "adjudicate": []}),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=False,
                    exit_code=1,
                    build_log="",
                    run_log="ModuleNotFoundError: No module named 'yaml'",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is False
    assert final_state["final_result"]["attempts"] == 0
    assert final_state["final_result"]["final_error_category"] == "OfficialBaselineUnavailable"


def test_research_workflow_tries_ranked_candidate_plans_before_repair(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.preset = "research"
    settings.prompt_profile = "research-rag"
    settings.max_attempts = 6
    settings.rag_mode = "hybrid"
    settings.structured_prompting = True
    settings.candidate_plan_count = 3
    settings.allow_candidate_fallback_before_repair = True
    settings.repair_cycle_limit = 2
    settings.repo_evidence_enabled = True
    project_root = write_project(tmp_path, "import yaml\nprint('ok')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": [
                    '{"packages":[{"package":"PyYAML","confidence":0.98,"source":"alias","evidence":["import yaml"]}]}'
                ],
                "version": [
                    '{"plans":['
                    '{"rank":1,"reason":"older candidate","dependencies":[{"name":"PyYAML","version":"6.0.1"}]},'
                    '{"rank":2,"reason":"newer candidate","dependencies":[{"name":"PyYAML","version":"6.0.2"}]}'
                    ']}'
                ],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2", "6.0.1"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=False,
                    exit_code=1,
                    build_log="",
                    run_log="ModuleNotFoundError: No module named 'yaml'",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                ),
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="success",
                    image_tag="img-2",
                    wall_clock_seconds=0.1,
                ),
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is True
    assert final_state["final_result"]["attempts"] == 2
    assert final_state["final_result"]["selected_candidate_rank"] == 2
    assert final_state["final_result"]["research_path"] is True
    assert final_state["repair_cycle_count"] == 0
    assert (Path(final_state["artifact_dir"]) / "repo-evidence.json").exists()
    assert (Path(final_state["artifact_dir"]) / "candidate-plans.json").exists()


def test_research_enhanced_stops_before_execution_when_python_constraints_are_impossible(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.preset = "research"
    settings.prompt_profile = "research-rag"
    settings.research_bundle = "enhanced"
    settings.research_features = ("python_constraint_intersection",)
    project_root = write_project(tmp_path, "import badpkg\n")
    docker_executor = FakeDockerExecutor([])
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": [
                    '{"packages":[{"package":"badpkg","confidence":0.99,"source":"llm","evidence":["import badpkg"]}]}'
                ],
                "version": [],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore(
            {
                "badpkg": PackageVersionOptions(
                    package="badpkg",
                    versions=["1.0.0"],
                    requires_python={"1.0.0": ">=4"},
                )
            }
        ),
        docker_executor=docker_executor,
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is False
    assert final_state["final_result"]["attempts"] == 0
    assert final_state["final_result"]["final_error_category"] == "ConstraintConflictError"
    assert final_state["final_result"]["conflict_precheck_failed"] is True
    assert final_state["final_result"]["python_constraint_intersection"] == ["badpkg:"]


def test_workflow_stops_on_syntax_error(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "print('hello')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner({"extract": [""], "version": [""], "repair": [], "adjudicate": []}),
        pypi_store=FakePyPIStore({}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=False,
                    exit_code=1,
                    build_log="",
                    run_log="SyntaxError: invalid syntax",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is False
    assert final_state["final_result"]["attempts"] == 1
    assert final_state["final_result"]["final_error_category"] == "SyntaxError"


def test_workflow_uses_adjudication_for_malformed_version_output(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import yaml\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["yaml"],
                "version": ["Use PyYAML version 6.0.2"],
                "repair": [],
                "adjudicate": ["PyYAML==6.0.2"],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is True
    assert final_state["final_result"]["dependencies"] == ["PyYAML==6.0.2"]


def test_workflow_accepts_fenced_adjudication_output(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import yaml\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["yaml"],
                "version": ["Use PyYAML version 6.0.2"],
                "repair": [],
                "adjudicate": ["```text\nPyYAML==6.0.2\n```"],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["final_result"]["success"] is True
    assert final_state["final_result"]["dependencies"] == ["PyYAML==6.0.2"]


def test_workflow_skips_unresolved_pypi_packages(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import requests\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["requests\nnot-a-real-package"],
                "version": ["requests==2.32.3"],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"requests": ["2.32.3"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))

    assert final_state["unresolved_packages"] == []
    assert final_state["final_result"]["dependencies"] == ["requests==2.32.3"]


def test_workflow_writes_llm_trace_log(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.trace_llm = True
    project_root = write_project(tmp_path, "import importlib\nmodule = importlib.import_module('yaml')\n")
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["yaml"],
                "version": ["PyYAML==6.0.2"],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
    )

    final_state = workflow.run(workflow.initial_state_for_project(project_root))
    trace_log = Path(final_state["artifact_dir"]) / "llm-trace.log"

    assert trace_log.exists()
    content = trace_log.read_text(encoding="utf-8")
    assert "--- PROMPT ---" in content
    assert "--- RESPONSE ---" in content


def test_workflow_emits_activity_events(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project_root = write_project(tmp_path, "import importlib\nmodule = importlib.import_module('yaml')\n")
    activity_events: list[tuple[str, int, str, str]] = []
    workflow = ResolutionWorkflow(
        settings,
        prompt_runner=FakePromptRunner(
            {
                "extract": ["yaml"],
                "version": ["PyYAML==6.0.2"],
                "repair": [],
                "adjudicate": [],
            }
        ),
        pypi_store=FakePyPIStore({"PyYAML": ["6.0.2"]}),
        docker_executor=FakeDockerExecutor(
            [
                DockerExecutionResult(
                    build_succeeded=True,
                    run_succeeded=True,
                    exit_code=0,
                    build_log="",
                    run_log="",
                    image_tag="img-1",
                    wall_clock_seconds=0.1,
                )
            ]
        ),
        activity_callback=lambda case_id, attempt, kind, detail: activity_events.append((case_id, attempt, kind, detail)),
    )

    state = workflow.initial_state_for_project(project_root)
    final_state = workflow.run(state)

    event_kinds = [kind for _, _, kind, _ in activity_events]
    assert final_state["final_result"]["success"] is True
    assert "llm_prompt_sent" in event_kinds
    assert "llm_response_received" in event_kinds
    assert "docker_context_prepared" in event_kinds
    assert "candidate_execution_started" in event_kinds
