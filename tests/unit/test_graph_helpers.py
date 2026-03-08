from pathlib import Path

from agentic_python_dependency.config import Settings
from agentic_python_dependency.graph import (
    ResolutionWorkflow,
    _has_python2_only_imports,
    filter_allowed_dependencies,
    infer_benchmark_validation_profile,
    infer_graph_recursion_limit,
    infer_validation_command,
    parse_dependency_lines,
    reconcile_inferred_target_python,
    route_after_normalize,
    route_after_classification,
    route_after_execute,
    route_after_research_classification,
    route_after_research_repair,
    snap_to_valid_docker_tag,
)
from agentic_python_dependency.state import CandidatePlan, ExecutionOutcome, ResolutionState, ResolvedDependency
from agentic_python_dependency.tools.retry_policy import classify_retry_decision


def test_parse_dependency_lines_is_strict() -> None:
    dependencies = parse_dependency_lines("PyYAML==6.0.2\nrequests==2.32.3\n")

    assert [dependency.pin() for dependency in dependencies] == ["PyYAML==6.0.2", "requests==2.32.3"]


def test_parse_dependency_lines_rejects_malformed_output() -> None:
    try:
        parse_dependency_lines("PyYAML 6.0.2")
    except ValueError as exc:
        assert "Malformed" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Malformed output should raise ValueError.")


def test_parse_dependency_lines_rejects_placeholder_versions() -> None:
    try:
        parse_dependency_lines("setuptools==<version>")
    except ValueError as exc:
        assert "Malformed" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Placeholder versions should raise ValueError.")


def test_parse_dependency_lines_ignores_markdown_fences() -> None:
    dependencies = parse_dependency_lines("```text\nPyYAML==6.0.2\nrequests==2.32.3\n```")

    assert [dependency.pin() for dependency in dependencies] == ["PyYAML==6.0.2", "requests==2.32.3"]


def test_filter_allowed_dependencies_rejects_unrelated_repairs() -> None:
    filtered = filter_allowed_dependencies(
        [
            ResolvedDependency(name="html5lib", version="1.1"),
            ResolvedDependency(name="xml", version="2.7.0"),
            ResolvedDependency(name="pip", version="26.0.1"),
        ],
        ["html5lib"],
    )

    assert [dependency.pin() for dependency in filtered] == ["html5lib==1.1"]


def test_route_helpers() -> None:
    success_state = ResolutionState(last_execution=ExecutionOutcome(True, "Success", "", True, True))
    failure_state = ResolutionState(
        current_attempt=1,
        last_execution=ExecutionOutcome(
            False,
            "ImportError",
            "",
            True,
            False,
            dependency_retryable=True,
        ),
    )

    assert route_after_execute(success_state) == "finalize_result"
    assert route_after_execute(failure_state) == "classify_outcome"
    assert route_after_classification(failure_state, max_attempts=4) == "repair_prompt_c"
    failure_state["pending_native_retry"] = True
    assert route_after_classification(failure_state, max_attempts=4) == "retry_current_plan"
    assert route_after_normalize(ResolutionState()) == "materialize_execution_context"
    assert route_after_normalize(ResolutionState(stop_reason="RepairOutputStalled")) == "finalize_result"


def test_infer_validation_command_prefers_pytest(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()

    assert infer_validation_command(tmp_path) == "pytest -q"


def test_snap_to_valid_docker_tag_accepts_exact_and_major_minor() -> None:
    assert snap_to_valid_docker_tag("3.12") == "3.12"
    assert snap_to_valid_docker_tag("3.12.9") == "3.12.9"
    assert snap_to_valid_docker_tag("3.12.7") == "3.12"
    assert snap_to_valid_docker_tag("3.79045.1") is None


def test_python2_only_import_signal_helper_detects_py2_stdlib_roots() -> None:
    assert _has_python2_only_imports(["sys", "BaseHTTPServer"]) is True
    assert _has_python2_only_imports(["sys", "json"]) is False


def test_reconcile_inferred_target_python_falls_back_on_invalid_version() -> None:
    version, source = reconcile_inferred_target_python(
        "2.5",
        benchmark_target_python="2.7.13",
        source_text="import requests\n",
    )

    assert version == "2.7.13"
    assert source == "benchmark_dockerfile_invalid_version"


def test_reconcile_inferred_target_python_uses_python2_import_signal() -> None:
    version, source = reconcile_inferred_target_python(
        "3.12",
        benchmark_target_python="2.7.13",
        source_text="import BaseHTTPServer\n",
        extracted_imports=["BaseHTTPServer"],
    )

    assert version == "2.7.13"
    assert source == "python2_import_signal"


def test_reconcile_inferred_target_python_uses_py2_syntax_guardrail_without_benchmark_py2() -> None:
    version, source = reconcile_inferred_target_python(
        "3.7",
        benchmark_target_python="3.12",
        source_text="print 'hello world'\n",
    )

    assert version == "3.7"
    assert source == "llm_prompt_a"


def test_reconcile_inferred_target_python_uses_py2_import_signal_without_benchmark_py2() -> None:
    version, source = reconcile_inferred_target_python(
        "3.7",
        benchmark_target_python="3.12",
        source_text="import BaseHTTPServer\n",
        extracted_imports=["BaseHTTPServer"],
    )

    assert version == "2.7.18"
    assert source == "python2_import_signal_default"


def test_infer_graph_recursion_limit_scales_with_attempts() -> None:
    assert infer_graph_recursion_limit(1) >= 32
    assert infer_graph_recursion_limit(4) > 25


def test_infer_benchmark_validation_profile_uses_help_for_cli_cases() -> None:
    profile, command = infer_benchmark_validation_profile("import argparse\nparser = argparse.ArgumentParser()\n", [])

    assert profile == "cli_help"
    assert "snippet.py --help" in command


def test_infer_benchmark_validation_profile_uses_stubbed_args_for_raw_sys_argv_cases() -> None:
    profile, command = infer_benchmark_validation_profile("import sys\nprint(sys.argv[1])\nprint(sys.argv[2])\n", [])

    assert profile == "argv_stub"
    assert "sys.argv = ['snippet.py', '/tmp/apdr-arg1', '/tmp/apdr-arg2']" in command


def test_infer_benchmark_validation_profile_uses_import_smoke_for_service_cases() -> None:
    profile, command = infer_benchmark_validation_profile("from flask import Flask\napp = Flask(__name__)\napp.run()\n", ["flask"])

    assert profile == "service_import"
    assert "runpy.run_path" in command


def test_infer_benchmark_validation_profile_prefers_import_smoke_for_main_guard_scripts() -> None:
    profile, command = infer_benchmark_validation_profile("if __name__ == '__main__':\n    print('hello')\n", [])

    assert profile == "main_guard_import"
    assert "runpy.run_path('snippet.py', run_name='not_main')" in command


def test_route_after_research_classification_repairs_package_compatibility_when_no_fallbacks_remain(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    state = ResolutionState(
        current_attempt=1,
        repair_cycle_count=0,
        remaining_candidate_plans=[],
        pending_native_retry=False,
        retry_decision=classify_retry_decision("PackageCompatibilityError"),
        last_execution=ExecutionOutcome(
            False,
            "PackageCompatibilityError",
            "Package 'pymc3' requires a different Python: 2.7.18 not in '>=3.5.4'",
            False,
            False,
            dependency_retryable=True,
        ),
    )

    assert route_after_research_classification(state, settings) == "repair_prompt_c_research"


def test_route_after_research_classification_uses_candidate_fallback_after_build_timeout(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    state = ResolutionState(
        current_attempt=1,
        repair_cycle_count=0,
        remaining_candidate_plans=[CandidatePlan(rank=2, reason="fallback", dependencies=[])],
        pending_native_retry=False,
        retry_decision=classify_retry_decision("BuildTimeoutError"),
        last_execution=ExecutionOutcome(
            False,
            "BuildTimeoutError",
            "docker build timed out after 300 seconds.",
            False,
            False,
            dependency_retryable=True,
        ),
    )

    assert route_after_research_classification(state, settings) == "select_next_candidate_plan"


def test_route_after_research_classification_replans_after_python_fallback(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    state = ResolutionState(
        current_attempt=1,
        repair_cycle_count=0,
        remaining_candidate_plans=[],
        pending_python_fallback=True,
        retry_decision=classify_retry_decision("SyntaxError"),
        last_execution=ExecutionOutcome(
            False,
            "SyntaxError",
            "SyntaxError: invalid syntax",
            True,
            False,
            dependency_retryable=True,
        ),
    )

    assert route_after_research_classification(state, settings) == "replan_after_python_fallback"


def test_route_after_research_repair_retries_until_model_marks_impossible(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    state = ResolutionState(
        repair_cycle_count=1,
        repair_model_concluded_impossible=False,
        candidate_plans=[],
    )

    assert route_after_research_repair(state, settings) == "repair_prompt_c_research"


def test_route_after_research_repair_finalizes_when_model_marks_impossible(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    state = ResolutionState(
        repair_cycle_count=1,
        repair_model_concluded_impossible=True,
        candidate_plans=[],
    )

    assert route_after_research_repair(state, settings) == "finalize_result"


def test_infer_benchmark_validation_profile_prefers_import_smoke_for_data_scripts() -> None:
    profile, command = infer_benchmark_validation_profile("import pandas as pd\ndata = pd.read_csv('data.csv')\n", ["pandas"])

    assert profile == "import_smoke"
    assert "__import__('pandas')" in command


def test_infer_benchmark_validation_profile_uses_import_specs_for_tensorflow_ml_loops() -> None:
    source = "\n".join(
        [
            "import gym",
            "from keras.layers import Dense, merge",
            "import tensorflow as tf",
            "class network:",
            "    def run(self):",
            "        return 1",
            "for eps in range(NUM_EPS):",
            "    print(eps)",
        ]
    )

    profile, command = infer_benchmark_validation_profile(source, ["gym", "keras", "tensorflow"])

    assert profile == "import_specs"
    assert "importlib.util as importlib_util" in command
    assert '"tensorflow"' in command
    assert "find_spec(name)" in command
    assert "__import__(" not in command


def test_infer_benchmark_validation_profile_uses_import_statements_for_non_tensorflow_ml_loops() -> None:
    source = "\n".join(
        [
            "import gym",
            "from keras.layers import Dense, merge",
            "for eps in range(NUM_EPS):",
            "    print(eps)",
        ]
    )

    profile, command = infer_benchmark_validation_profile(source, ["gym", "keras"])

    assert profile == "import_statements"
    assert "ast.parse(source, filename='snippet.py')" in command
    assert "ast.ImportFrom" in command


def test_infer_benchmark_validation_profile_uses_headless_imports_for_gui_cases() -> None:
    profile, command = infer_benchmark_validation_profile("import pygame\npygame.display.set_mode((100, 100))\n", ["pygame"])

    assert profile == "headless_imports"
    assert "__import__('pygame')" in command


def test_native_retry_hints_detect_lxml_headers() -> None:
    hints, system_packages, bootstrap_pins = ResolutionWorkflow._native_retry_hints(
        "Error: Please make sure the libxml2 and libxslt development packages are installed."
    )

    assert "lxml_headers" in hints
    assert "libxml2-dev" in system_packages
    assert "libxslt1-dev" in system_packages
    assert "zlib1g-dev" in system_packages
    assert bootstrap_pins == []
