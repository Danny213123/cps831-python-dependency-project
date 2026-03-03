from pathlib import Path

from agentic_python_dependency.graph import (
    filter_allowed_dependencies,
    infer_benchmark_validation_profile,
    infer_graph_recursion_limit,
    infer_validation_command,
    parse_dependency_lines,
    route_after_normalize,
    route_after_classification,
    route_after_execute,
)
from agentic_python_dependency.state import ExecutionOutcome, ResolutionState, ResolvedDependency


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
    assert route_after_normalize(ResolutionState()) == "materialize_execution_context"
    assert route_after_normalize(ResolutionState(stop_reason="RepairOutputStalled")) == "finalize_result"


def test_infer_validation_command_prefers_pytest(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()

    assert infer_validation_command(tmp_path) == "pytest -q"


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
    assert "sys.argv = ['snippet.py', '/tmp/apd-arg1', '/tmp/apd-arg2']" in command


def test_infer_benchmark_validation_profile_uses_import_smoke_for_service_cases() -> None:
    profile, command = infer_benchmark_validation_profile("from flask import Flask\napp = Flask(__name__)\napp.run()\n", ["flask"])

    assert profile == "service_import"
    assert "runpy.run_path" in command


def test_infer_benchmark_validation_profile_prefers_import_smoke_for_main_guard_scripts() -> None:
    profile, command = infer_benchmark_validation_profile("if __name__ == '__main__':\n    print('hello')\n", [])

    assert profile == "main_guard_import"
    assert "runpy.run_path('snippet.py', run_name='not_main')" in command


def test_infer_benchmark_validation_profile_prefers_import_smoke_for_data_scripts() -> None:
    profile, command = infer_benchmark_validation_profile("import pandas as pd\ndata = pd.read_csv('data.csv')\n", ["pandas"])

    assert profile == "import_smoke"
    assert "__import__('pandas')" in command


def test_infer_benchmark_validation_profile_uses_headless_imports_for_gui_cases() -> None:
    profile, command = infer_benchmark_validation_profile("import pygame\npygame.display.set_mode((100, 100))\n", ["pygame"])

    assert profile == "headless_imports"
    assert "__import__('pygame')" in command
