from __future__ import annotations

import ast
import inspect
import json
import re
import shutil
from string import Formatter
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from agentic_python_dependency.benchmark.gistable import GistableDataset
from agentic_python_dependency.config import Settings
from agentic_python_dependency.presets import (
    COMPATIBILITY_SENSITIVE_PACKAGES,
    ResearchFeatureName,
    get_preset_config,
)
from agentic_python_dependency.router import OllamaPromptRunner
from agentic_python_dependency.state import (
    AttemptRecord,
    BenchmarkCase,
    CandidateDependency,
    CandidatePlan,
    ConflictNote,
    ExecutionOutcome,
    InferenceCandidate,
    ProjectTarget,
    PackageVersionOptions,
    RepairMemorySummary,
    RepairStrategyRecord,
    RetryDecision,
    ResolutionState,
    ResolvedDependency,
)
from agentic_python_dependency.tools.constraint_pack import (
    build_constraint_pack,
    constraint_pack_to_dict,
    generate_candidate_bundles,
)
from agentic_python_dependency.tools.docker_executor import DockerExecutor
from agentic_python_dependency.tools.dynamic_imports import collect_dynamic_import_candidates
from agentic_python_dependency.tools.error_classifier import classify_error
from agentic_python_dependency.tools.import_extractor import (
    PY2_STDLIB_EXTRAS,
    STDLIB_MODULES,
    discover_python_files,
    extract_import_roots_from_code,
    filter_third_party_imports,
    is_ambiguous_import,
    is_trap_package_name,
    is_unsupported_import,
    looks_like_package_name,
    load_python_sources,
    normalize_candidate_packages,
    normalize_candidate_packages_with_sources,
    runtime_package_alias,
)
from agentic_python_dependency.tools.official_baselines import (
    OfficialBaselineError,
    OfficialBaselinePlan,
    run_pyego,
    run_readpye,
)
from agentic_python_dependency.tools.package_metadata import PackageMetadataStore
from agentic_python_dependency.tools.pypi_store import PyPIMetadataStore
from agentic_python_dependency.tools.rag_context import build_research_rag_context, summarize_rag_context
from agentic_python_dependency.tools.repair_feedback import append_feedback_event, summarize_feedback_memory
from agentic_python_dependency.tools.repo_aliases import build_repo_alias_candidates
from agentic_python_dependency.tools.repo_evidence import build_repo_evidence
from agentic_python_dependency.tools.retry_policy import classify_retry_decision
from agentic_python_dependency.tools.structured_outputs import (
    StructuredOutputError,
    parse_alias_resolution_payload,
    parse_candidate_plan_payload,
    parse_cross_validation_payload,
    parse_experimental_package_payload,
    parse_repair_plan_payload,
    parse_source_compatibility_payload,
    parse_version_negotiation_payload,
)


def normalize_package_name(value: str) -> str:
    return value.strip().replace("-", "_").lower()


def filter_allowed_dependencies(
    dependencies: list[ResolvedDependency], allowed_packages: list[str] | None
) -> list[ResolvedDependency]:
    if not allowed_packages:
        return dependencies
    allowed = {normalize_package_name(package) for package in allowed_packages}
    filtered = [dependency for dependency in dependencies if normalize_package_name(dependency.name) in allowed]
    deduped: dict[str, ResolvedDependency] = {}
    for dependency in filtered:
        deduped[normalize_package_name(dependency.name)] = dependency
    return sorted(deduped.values(), key=lambda dependency: dependency.name.lower())


def parse_dependency_lines(raw_output: str) -> list[ResolvedDependency]:
    dependencies: dict[str, ResolvedDependency] = {}
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "```" or (line.startswith("```") and "==" not in line):
            continue
        if "==" not in line or line.count("==") != 1:
            raise ValueError(f"Malformed dependency line: {line}")
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version or " " in name or " " in version:
            raise ValueError(f"Malformed dependency line: {line}")
        if any(marker in version for marker in "<>[](){}"):
            raise ValueError(f"Malformed dependency line: {line}")
        try:
            Version(version)
        except InvalidVersion:
            raise ValueError(f"Malformed dependency line: {line}")
        dependencies[name.lower()] = ResolvedDependency(name=name, version=version)
    return sorted(dependencies.values(), key=lambda dependency: dependency.name.lower())


PYTHON_VERSION_RE = re.compile(r"\b((?:2|3)\.\d+(?:\.\d+)?)\b")


def normalize_python_version_hint(value: str) -> str | None:
    candidate = value.strip()
    match = PYTHON_VERSION_RE.search(candidate)
    if not match:
        return None
    version = match.group(1)
    parts = version.split(".")
    if len(parts) == 2:
        return version
    if len(parts) == 3:
        return version
    return None


def _extract_json_object_text(raw_output: str) -> str | None:
    candidates: list[str] = []
    stripped = raw_output.strip()
    if stripped:
        candidates.append(stripped)
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw_output, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())
    first_brace = raw_output.find("{")
    last_brace = raw_output.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(raw_output[first_brace : last_brace + 1].strip())
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return candidate
    return None


def parse_package_inference_output(raw_output: str) -> tuple[list[str], str | None]:
    raw_output = raw_output.strip()
    if not raw_output:
        return [], None

    json_candidate = _extract_json_object_text(raw_output)
    try:
        payload = json.loads(json_candidate if json_candidate is not None else raw_output)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        version = normalize_python_version_hint(str(payload.get("python_version", "")).strip())
        packages_field = None
        for key in ("packages", "modules", "python_modules"):
            if key in payload:
                packages_field = payload[key]
                break
        packages: list[str] = []
        if isinstance(packages_field, list):
            for item in packages_field:
                if isinstance(item, dict):
                    package = str(item.get("package", "") or item.get("name", "")).strip()
                else:
                    package = str(item).strip()
                if package:
                    packages.append(package)
        elif isinstance(packages_field, str):
            packages.extend(part.strip() for part in packages_field.split(",") if part.strip())
        return packages, version

    packages: list[str] = []
    python_version: str | None = None
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if "python" in lowered and "version" in lowered:
            inferred = normalize_python_version_hint(line)
            if inferred:
                python_version = inferred
            continue
        if re.fullmatch(r"(?:2|3)\.\d+(?:\.\d+)?", line):
            python_version = line
            continue
        if ":" in line:
            label, _, value = line.partition(":")
            if "python" in label.lower() and "version" in label.lower():
                inferred = normalize_python_version_hint(value)
                if inferred:
                    python_version = inferred
                continue
            if label.strip().lower() in {"modules", "packages"}:
                packages.extend(part.strip() for part in value.split(",") if part.strip())
                continue
        packages.append(line)
    return packages, python_version


def _is_python3_syntax_compatible(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


VALID_DOCKER_PYTHON_TAGS = {
    "2.7.13",
    "2.7.14",
    "2.7.15",
    "2.7.16",
    "2.7.17",
    "2.7.18",
    "3.5",
    "3.5.10",
    "3.6",
    "3.6.15",
    "3.7",
    "3.7.17",
    "3.8",
    "3.8.20",
    "3.9",
    "3.9.21",
    "3.10",
    "3.10.16",
    "3.11",
    "3.11.12",
    "3.12",
    "3.12.9",
    "3.13",
    "3.13.2",
}


def snap_to_valid_docker_tag(version: str) -> str | None:
    if version in VALID_DOCKER_PYTHON_TAGS:
        return version
    parts = version.split(".")
    if len(parts) >= 2:
        major_minor = f"{parts[0]}.{parts[1]}"
        if major_minor in VALID_DOCKER_PYTHON_TAGS:
            return major_minor
    return None


def _has_python2_only_imports(extracted_imports: list[str]) -> bool:
    return bool(set(extracted_imports) & PY2_STDLIB_EXTRAS)


def reconcile_inferred_target_python(
    inferred_version: str | None,
    *,
    benchmark_target_python: str,
    source_text: str,
    extracted_imports: list[str] | None = None,
) -> tuple[str | None, str | None]:
    if not inferred_version:
        return None, None
    snapped = snap_to_valid_docker_tag(inferred_version)
    if snapped is None:
        if benchmark_target_python:
            return benchmark_target_python, "benchmark_dockerfile_invalid_version"
        return "3.12", "default_fallback_invalid_version"
    inferred_version = snapped
    inferred_major = inferred_version.split(".", 1)[0]
    benchmark_major = benchmark_target_python.split(".", 1)[0] if benchmark_target_python else ""
    if inferred_major == "3" and extracted_imports and _has_python2_only_imports(extracted_imports):
        if benchmark_major == "2" and benchmark_target_python:
            return benchmark_target_python, "python2_import_signal"
        return "2.7.18", "python2_import_signal_default"
    if inferred_major == "3" and not _is_python3_syntax_compatible(source_text):
        if benchmark_major == "2" and benchmark_target_python:
            return benchmark_target_python, "benchmark_dockerfile_syntax_guardrail"
        return inferred_version, "llm_prompt_a"
    if inferred_major == benchmark_major:
        return inferred_version, "llm_prompt_a"
    return inferred_version, "llm_prompt_a"


def route_after_execute(state: ResolutionState) -> str:
    return "finalize_result" if state["last_execution"].success else "classify_outcome"


def route_after_normalize(state: ResolutionState) -> str:
    return (
        "finalize_result"
        if state.get("stop_reason") in {"RepairOutputStalled", "OfficialPyEGoUnavailable"}
        else "materialize_execution_context"
    )


def route_after_classification(state: ResolutionState, max_attempts: int) -> str:
    execution = state["last_execution"]
    if state.get("pending_python_fallback") and state["current_attempt"] < max_attempts:
        return "retry_current_plan"
    if state.get("pending_native_retry") and state["current_attempt"] < max_attempts:
        return "retry_current_plan"
    if execution.dependency_retryable and state["current_attempt"] < max_attempts:
        return "repair_prompt_c"
    return "finalize_result"


def route_after_research_plan_selection(state: ResolutionState) -> str:
    selected_plan = state.get("selected_candidate_plan")
    if not selected_plan:
        return "finalize_result"
    if not selected_plan.dependencies and state.get("repair_skipped_reason"):
        return "finalize_result"
    return "materialize_execution_context"


def route_after_research_classification(state: ResolutionState, settings: Settings) -> str:
    decision = state.get("retry_decision")
    execution = state["last_execution"]
    if state["current_attempt"] >= settings.max_attempts:
        return "finalize_result"
    if state.get("pending_python_fallback"):
        return "replan_after_python_fallback"
    if state.get("pending_runtime_profile_retry"):
        return "runtime_profile_repair_research"
    if state.get("pending_native_retry"):
        return "retry_current_plan"
    if decision is None:
        if not execution.dependency_retryable:
            return "finalize_result"
        if state.get("remaining_candidate_plans"):
            return "select_next_candidate_plan"
        if state.get("repair_cycle_count", 0) < settings.repair_cycle_limit:
            return "repair_prompt_c_research"
        return "finalize_result"
    if decision.candidate_fallback_allowed and settings.allow_candidate_fallback_before_repair and state.get("remaining_candidate_plans"):
        return "select_next_candidate_plan"
    repair_budget = settings.repair_cycle_limit
    if decision.repair_retry_budget:
        repair_budget = min(repair_budget, decision.repair_retry_budget)
    if decision.repair_allowed and state.get("repair_cycle_count", 0) < repair_budget:
        return "repair_prompt_c_research"
    return "finalize_result"


def route_after_research_repair(state: ResolutionState, settings: Settings) -> str:
    if state.get("pending_python_fallback"):
        return "replan_after_python_fallback"
    if state.get("candidate_plans"):
        return "select_next_candidate_plan"
    if state.get("repair_model_concluded_impossible"):
        return "finalize_result"
    if state.get("repair_cycle_count", 0) < settings.repair_cycle_limit:
        return "repair_prompt_c_research"
    return "finalize_result"


def route_after_constraint_precheck(state: ResolutionState) -> str:
    pack = state.get("constraint_pack")
    if pack is not None and pack.conflict_precheck_failed:
        return "finalize_result"
    return "load_feedback_memory_summary"


def infer_validation_command(project_root: Path) -> str:
    if (project_root / "tests").exists() or (project_root / "pytest.ini").exists():
        return "pytest -q"

    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        if config.get("tool", {}).get("pytest"):
            return "pytest -q"
        scripts = config.get("project", {}).get("scripts", {})
        if scripts:
            command = next(iter(scripts))
            return f"{command} --help"

    src_dir = project_root / "src"
    if src_dir.exists():
        for candidate in src_dir.rglob("__main__.py"):
            package = ".".join(candidate.relative_to(src_dir).with_suffix("").parts[:-1])
            if package:
                return f"python -m {package}"
    return "python -m compileall ."


def build_import_smoke_command(import_roots: list[str]) -> str:
    imports_clause = ""
    if import_roots:
        imports_clause = "\n".join(f"__import__({root!r})" for root in import_roots)
    else:
        imports_clause = "import py_compile\npy_compile.compile('snippet.py', doraise=True)"
    return (
        "python - <<'PY'\n"
        "import os\n"
        "for key, value in [('MPLBACKEND', 'Agg'), ('SDL_VIDEODRIVER', 'dummy'), ('QT_QPA_PLATFORM', 'offscreen')]:\n"
        "    os.environ.setdefault(key, value)\n"
        f"{imports_clause}\n"
        "print('imports-ok')\n"
        "PY"
    )


def build_import_spec_probe_command(import_roots: list[str]) -> str:
    modules = sorted({root for root in import_roots if root})
    if not modules:
        return (
            "python - <<'PY'\n"
            "import py_compile\n"
            "py_compile.compile('snippet.py', doraise=True)\n"
            "print('import-specs-ok')\n"
            "PY"
        )
    return (
        "python - <<'PY'\n"
        "import importlib.util as importlib_util\n"
        "modules = "
        f"{json.dumps(modules)}\n"
        "missing = [name for name in modules if importlib_util.find_spec(name) is None]\n"
        "if missing:\n"
        "    raise ImportError('Missing module specs: ' + ', '.join(missing))\n"
        "print('import-specs-ok')\n"
        "PY"
    )


def build_snippet_import_command() -> str:
    return (
        "python - <<'PY'\n"
        "import os\n"
        "import sys\n"
        "import runpy\n"
        "for key, value in [('MPLBACKEND', 'Agg'), ('SDL_VIDEODRIVER', 'dummy'), ('QT_QPA_PLATFORM', 'offscreen')]:\n"
        "    os.environ.setdefault(key, value)\n"
        "sys.argv = ['snippet.py']\n"
        "runpy.run_path('snippet.py', run_name='not_main')\n"
        "print('import-ok')\n"
        "PY"
    )


def build_import_statements_command() -> str:
    return (
        "python - <<'PY'\n"
        "import ast\n"
        "import io\n"
        "import os\n"
        "import sys\n"
        "for key, value in [('MPLBACKEND', 'Agg'), ('SDL_VIDEODRIVER', 'dummy'), ('QT_QPA_PLATFORM', 'offscreen')]:\n"
        "    os.environ.setdefault(key, value)\n"
        "if sys.version_info[0] < 3:\n"
        "    source = open('snippet.py', 'rb').read()\n"
        "    tree = compile(source, 'snippet.py', 'exec', ast.PyCF_ONLY_AST)\n"
        "    module = compile('', 'snippet.py', 'exec', ast.PyCF_ONLY_AST)\n"
        "else:\n"
        "    source = io.open('snippet.py', 'r', encoding='utf-8').read()\n"
        "    tree = ast.parse(source, filename='snippet.py')\n"
        "    module = ast.parse('', filename='snippet.py')\n"
        "module.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]\n"
        "if hasattr(module, 'type_ignores'):\n"
        "    module.type_ignores = []\n"
        "namespace = {}\n"
        "exec(compile(module, 'snippet.py', 'exec'), namespace, namespace)\n"
        "print('imports-ok')\n"
        "PY"
    )


def _has_top_level_ml_training_loop(source_code: str, extracted_imports: list[str]) -> bool:
    ml_imports = {"tensorflow", "keras", "torch", "gym", "gymnasium"}
    normalized_imports = {normalize_package_name(item) for item in extracted_imports}
    if not normalized_imports & ml_imports:
        return False
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            return True
    return False


def build_snippet_stub_argv_command(max_index: int) -> str:
    args = ["'snippet.py'"]
    for index in range(1, max_index + 1):
        args.append(f"'/tmp/apdr-arg{index}'")
    return (
        "python - <<'PY'\n"
        "import os\n"
        "import sys\n"
        "import runpy\n"
        "for key, value in [('MPLBACKEND', 'Agg'), ('SDL_VIDEODRIVER', 'dummy'), ('QT_QPA_PLATFORM', 'offscreen')]:\n"
        "    os.environ.setdefault(key, value)\n"
        "for path, payload in [('/tmp/apdr-arg1', b'apdr'), ('/tmp/apdr-arg2', b'apdr')]:\n"
        "    with open(path, 'wb') as handle:\n"
        "        handle.write(payload)\n"
        f"sys.argv = [{', '.join(args)}]\n"
        "runpy.run_path('snippet.py', run_name='__main__')\n"
        "print('argv-smoke-ok')\n"
        "PY"
    )


def build_benchmark_validation_options(source_code: str, extracted_imports: list[str]) -> tuple[list[dict[str, str]], str]:
    lowered = source_code.lower()
    options: list[dict[str, str]] = []
    normalized_imports = {normalize_package_name(item) for item in extracted_imports}
    hardware_sensitive_ml_imports = {"tensorflow", "torch"}

    def add_option(profile: str, command: str, reason: str) -> None:
        if any(option["profile"] == profile for option in options):
            return
        options.append({"profile": profile, "command": command, "reason": reason})

    default_profile = "docker_cmd"
    default_command = ""
    if any(token in lowered for token in ("flask(", "uvicorn.run", "serve_forever(", "app.run(")):
        default_profile = "service_import"
        default_command = build_snippet_import_command()
        add_option("service_import", default_command, "service entrypoints should be import-validated")
    elif any(token in lowered for token in ("argparse", "optparse", "click.command", "sys.argv")):
        if "argparse" not in lowered and "optparse" not in lowered and "click.command" not in lowered and "sys.argv" in lowered:
            max_index = 0
            for match in re.findall(r"sys\.argv\[(\d+)\]", source_code):
                max_index = max(max_index, int(match))
            if max_index:
                default_profile = "argv_stub"
                default_command = build_snippet_stub_argv_command(max_index)
                add_option("argv_stub", default_command, "raw sys.argv access needs stubbed positional args")
            else:
                default_profile = "cli_help"
                default_command = (
                    "python snippet.py --help >/tmp/apdr-help.txt 2>&1 || "
                    "python snippet.py -h >/tmp/apdr-help.txt 2>&1 || "
                    f"{build_snippet_import_command()}"
                )
                add_option("cli_help", default_command, "CLI scripts should prefer help-mode validation")
        else:
            default_profile = "cli_help"
            default_command = (
                "python snippet.py --help >/tmp/apdr-help.txt 2>&1 || "
                "python snippet.py -h >/tmp/apdr-help.txt 2>&1 || "
                f"{build_snippet_import_command()}"
            )
            add_option("cli_help", default_command, "CLI scripts should prefer help-mode validation")
    import_only_tokens = (
        "get_ipython(",
        "db.model",
        "eventregistry(",
        "boto3.client(",
        "read_csv(",
        ".parse(",
        "open(",
        "django.conf.settings",
        "models.model",
    )
    if "__name__ == '__main__'" in source_code or '__name__ == "__main__"' in source_code:
        default_profile = "main_guard_import"
        default_command = build_snippet_import_command()
        add_option("main_guard_import", default_command, "main-guarded scripts should avoid full execution")
    elif _has_top_level_ml_training_loop(source_code, extracted_imports):
        default_profile = "import_statements"
        default_command = build_import_statements_command()
        add_option("import_statements", default_command, "long-running ML scripts should validate import statements only")
    elif any(token in lowered for token in import_only_tokens):
        default_profile = "import_smoke"
        default_command = build_import_smoke_command(extracted_imports)
        add_option("import_smoke", default_command, "framework or data scripts should use import-only validation")
    elif any(
        token in lowered
        for token in ("tkinter", "tkinter.", "turtle", "pygame", "mainloop(", "plt.show(", "pyqt", "wx.")
    ):
        default_profile = "headless_imports"
        default_command = build_import_smoke_command(extracted_imports)
        add_option("headless_imports", default_command, "GUI imports should use headless import validation")

    add_option(default_profile, default_command, "default heuristic validation profile")
    if extracted_imports:
        add_option("import_smoke", build_import_smoke_command(extracted_imports), "top-level imports only")
        if normalized_imports & hardware_sensitive_ml_imports:
            add_option("import_specs", build_import_spec_probe_command(extracted_imports), "check module specs without importing hardware-sensitive packages")
    add_option("import_statements", build_import_statements_command(), "execute only import statements from the snippet")
    add_option("docker_cmd", "", "run the snippet exactly as the benchmark Docker command would")
    return options, default_profile


def infer_benchmark_validation_profile(source_code: str, extracted_imports: list[str]) -> tuple[str, str]:
    options, default_profile = build_benchmark_validation_options(source_code, extracted_imports)
    for option in options:
        if option["profile"] == default_profile:
            return option["profile"], option["command"]
    return ("docker_cmd", "")


def detect_target_python_from_dockerfile(dockerfile_text: str) -> str:
    for line in dockerfile_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("from python:"):
            tag = stripped.split(":", 1)[1].split()[0]
            return tag.replace("-slim", "")
    return "3.12"


def infer_graph_recursion_limit(max_attempts: int) -> int:
    # Initial pass uses 8 nodes before routing. Each retryable failed attempt adds
    # roughly 8 more node executions (classify, repair, retrieve, infer, normalize,
    # materialize, execute, classify/finalize path). Give the graph headroom above
    # the theoretical path length so legitimate retry loops do not trip LangGraph.
    return max(32, 16 + (max_attempts * 12))


class ResolutionWorkflow:
    _OPTIONAL_PROMPT_VARIABLE_DEFAULTS: dict[str, Any] = {
        "candidate_bundle_hints": "{}",
        "conflict_notes": "[]",
        "source_compatibility_hints": "[]",
        "source_signals": "[]",
        "repeated_missing_symbol_failures": "[]",
        "symbol_compatibility_repair_candidates": "[]",
        "validation_options": "[]",
        "default_validation_profile": "",
        "repair_memory": "{}",
        "feedback_summary": "{}",
        "max_plan_count": 1,
    }
    _DEFAULT_ATTEMPT_FAILURE_ANALYSIS_TEMPLATE = """You are writing a short post-attempt failure analysis for a Python dependency resolution benchmark.

Use only the supplied evidence. Do not invent facts that are not present in the logs or classification output.
Return plain text only, maximum 4 short lines.
Explain:
- whether the failure happened during build or runtime,
- the most likely immediate cause,
- whether Python version, selected dependencies, or validation profile appear to be the issue,
- and whether any part of the evidence is inconclusive.
Do not propose a repair plan.

Attempt number:
{attempt_number}

Target Python:
{target_python}

Runtime profile:
{runtime_profile}

Validation command:
{validation_command}

Dependencies:
{dependencies}

Classifier origin:
{classifier_origin}

Classified category:
{error_category}

Classified details:
{error_details}

Source compatibility hints:
{source_compatibility_hints}

Build log excerpt:
{build_log_excerpt}

Run log excerpt:
{run_log_excerpt}
"""
    _DEFAULT_RUNTIME_PROFILE_REPAIR_TEMPLATE = """You are deciding whether a failed benchmark validation should be repaired by changing the runtime profile instead of changing dependencies.

Return strict JSON only using this schema:
{{
  "repair_applicable": true,
  "plans": [
    {{
      "rank": 1,
      "reason": "short reason",
      "runtime_profile": "import_specs",
      "dependencies": []
    }}
  ]
}}

Rules:
- Use only allowed runtime profiles and allowed package versions already present in the supplied evidence.
- Prefer keeping the dependency plan unchanged when the build succeeded and the run failed in the validation wrapper.
- If the run log suggests a hardware-sensitive binary import failure such as `Illegal instruction`, AVX, or CPU-feature mismatch, prefer a safer non-importing validation profile such as `import_specs` when available.
- You may return a runtime-profile-only repair with an empty `dependencies` list; omitted packages will be preserved from the previous plan.
- Only change dependencies if the run failure strongly indicates that the installed package versions themselves are still wrong.
- When source compatibility hints are present, do not move away from those version families unless the supplied evidence clearly requires it.
- No markdown, no commentary, no text outside JSON.

Target Python:
{target_python}

Current runtime profile:
{current_runtime_profile}

Allowed packages:
{allowed_packages}

Version space:
{version_space}

Allowed validation profiles:
{validation_options}

Default validation profile:
{default_validation_profile}

Source compatibility hints:
{source_compatibility_hints}

Conflict notes:
{conflict_notes}

Previous plan:
{previous_plan}

Attempted plans:
{attempted_plans}

Error details:
{error_details}

Build log excerpt:
{build_log_excerpt}

Run log excerpt:
{run_log_excerpt}
"""

    def __init__(
        self,
        settings: Settings,
        prompt_runner: OllamaPromptRunner | None = None,
        pypi_store: PyPIMetadataStore | None = None,
        docker_executor: DockerExecutor | None = None,
        activity_callback: Callable[[str, int, str, str], None] | None = None,
    ):
        self.settings = settings
        self.preset_config = get_preset_config(settings.preset)
        self.prompt_runner = prompt_runner or OllamaPromptRunner(settings, settings.prompt_template_dir)
        self.pypi_store = pypi_store or PyPIMetadataStore(settings.pypi_cache_dir)
        self.package_metadata_store = PackageMetadataStore(settings.package_metadata_dir)
        self.docker_executor = docker_executor or DockerExecutor(settings)
        self.activity_callback = activity_callback
        setattr(self.docker_executor, "activity_callback", activity_callback)
        self.dataset = GistableDataset(settings)

    def _prompt_template(self, name: str) -> str:
        return (self.settings.prompt_template_dir / name).read_text(encoding="utf-8")

    def _format_prompt(self, name: str, **variables: Any) -> str:
        template = self._prompt_template(name)
        formatter = Formatter()
        resolved_variables = dict(variables)
        for _literal, field_name, _format_spec, _conversion in formatter.parse(template):
            if not field_name or field_name in resolved_variables:
                continue
            if field_name in self._OPTIONAL_PROMPT_VARIABLE_DEFAULTS:
                resolved_variables[field_name] = self._OPTIONAL_PROMPT_VARIABLE_DEFAULTS[field_name]
            else:
                resolved_variables[field_name] = ""
        return template.format(**resolved_variables)

    def _trace_log_paths(self, state: ResolutionState) -> list[Path]:
        if "artifact_dir" not in state:
            return []
        case_dir = Path(state["artifact_dir"])
        return [case_dir / "llm-trace.log", case_dir.parent / "llm-trace.log"]

    def _stage_model_name(self, stage: str) -> str:
        if hasattr(self.prompt_runner, "stage_model"):
            return self.prompt_runner.stage_model(stage)
        return stage

    def _official_baseline_target(self, state: ResolutionState) -> Path:
        if state["mode"] == "gistable":
            if self.settings.resolver == "readpye":
                return state["benchmark_case"].snippet_path
            return state["benchmark_case"].root_dir
        if self.settings.resolver == "readpye" and state["project_target"].python_files:
            return state["project_target"].python_files[0]
        return state["project_target"].root_dir

    def _run_official_baseline(self, state: ResolutionState) -> tuple[OfficialBaselinePlan | None, str]:
        artifact_dir = Path(state["artifact_dir"])
        target = self._official_baseline_target(state)
        if self.settings.resolver == "pyego":
            script = self.settings.pyego_root / "PyEGo.py"
            if not script.exists():
                return None, ""
            return run_pyego(self.settings, target, artifact_dir), "pyego_official"
        if self.settings.resolver == "readpye":
            script = self.settings.readpye_root / "run.py"
            if not script.exists():
                return None, ""
            return run_readpye(self.settings, target, artifact_dir), "readpye_official"
        return None, ""

    def _apply_official_baseline_plan(
        self,
        state: ResolutionState,
        plan: OfficialBaselinePlan,
        source: str,
    ) -> ResolutionState:
        selected = sorted(plan.dependencies, key=lambda dependency: dependency.name.lower())
        state["selected_dependencies"] = selected
        state["system_dependencies"] = list(plan.system_packages)
        if plan.target_python:
            state["target_python"] = plan.target_python
        state["resolver_implementation"] = plan.implementation
        state["generated_requirements"] = DockerExecutor.render_requirements(selected)
        state["dependency_reason"] = "official_baseline"
        state["version_selection_source"] = source
        state["prompt_history"]["prompt_b"] = (
            f"# skipped: using official {self.settings.resolver} baseline output"
        )
        state["model_outputs"]["version"].append(
            {
                "attempt": state["current_attempt"],
                "output": "\n".join(dependency.pin() for dependency in selected),
                "source": source,
                "system_packages": list(plan.system_packages),
                "target_python": plan.target_python,
            }
        )
        state.setdefault("structured_outputs", {})["official_baseline"] = plan.raw_payload
        return state

    def _mark_official_pyego_unavailable(self, state: ResolutionState, error: str) -> ResolutionState:
        message = f"Official PyEGo is required for resolver=pyego but is unavailable: {error}"
        state["resolver_implementation"] = "official-required"
        state["selected_dependencies"] = []
        state["system_dependencies"] = []
        state["dependency_reason"] = "official_baseline_unavailable"
        state["version_selection_source"] = "pyego_official_required"
        state["last_execution"] = ExecutionOutcome(
            success=False,
            category="OfficialBaselineUnavailable",
            message=message,
            build_succeeded=False,
            run_succeeded=False,
            dependency_retryable=False,
            retry_severity="terminal",
        )
        state["last_error_category"] = "OfficialBaselineUnavailable"
        state["last_error_details"] = message
        state["stop_reason"] = "OfficialPyEGoUnavailable"
        state["prompt_history"]["prompt_b"] = f"# official PyEGo required but unavailable: {error}"
        state["model_outputs"]["version"].append(
            {
                "attempt": state["current_attempt"],
                "output": "",
                "source": "pyego_official_required",
                "error": error,
            }
        )
        return state

    def _emit_trace(self, state: ResolutionState, message: str) -> None:
        if not self.settings.trace_llm:
            return
        for path in self._trace_log_paths(state):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(message)
                if not message.endswith("\n"):
                    handle.write("\n")

    def _emit_activity(self, state: ResolutionState, *, kind: str, detail: str) -> None:
        if self.activity_callback is None:
            return
        case_id = str(state.get("case_id", "") or "").strip()
        if not case_id:
            return
        attempt = int(state.get("current_attempt", 0) or 0)
        self.activity_callback(case_id, attempt=attempt, kind=kind, detail=detail)

    def _trace_request(self, state: ResolutionState, stage: str, prompt_text: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        model = self._stage_model_name(stage)
        attempt = state.get("current_attempt", 0)
        self._emit_activity(
            state,
            kind="llm_prompt_sent",
            detail=f"Sending {stage} prompt to {model}.",
        )
        message = (
            f"[{timestamp}] case={state.get('case_id', '')} attempt={attempt} stage={stage} model={model}\n"
            "--- PROMPT ---\n"
            f"{prompt_text.rstrip()}\n"
            "===\n"
        )
        self._emit_trace(state, message)

    def _trace_response(self, state: ResolutionState, stage: str, response_text: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        model = self._stage_model_name(stage)
        attempt = state.get("current_attempt", 0)
        self._emit_activity(
            state,
            kind="llm_response_received",
            detail=f"Received {stage} response from {model}.",
        )
        message = (
            f"[{timestamp}] case={state.get('case_id', '')} attempt={attempt} stage={stage} model={model}\n"
            "--- RESPONSE ---\n"
            f"{response_text.rstrip()}\n"
            "===\n"
        )
        self._emit_trace(state, message)

    def _attempt_failure_analysis_template(self) -> str:
        candidates = (
            self.settings.prompt_template_dir / "attempt_failure_analysis.txt",
            self.settings.prompts_dir / "research-rag" / "attempt_failure_analysis.txt",
        )
        for path in candidates:
            if path.exists():
                return path.read_text(encoding="utf-8")
        return self._DEFAULT_ATTEMPT_FAILURE_ANALYSIS_TEMPLATE

    @staticmethod
    def _trim_text_block(text: str, *, max_lines: int = 30, max_chars: int = 1800) -> str:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "-"
        excerpt = "\n".join(lines[-max_lines:])
        if len(excerpt) <= max_chars:
            return excerpt
        return "...\n" + excerpt[-(max_chars - 4) :].lstrip()

    def _render_attempt_failure_analysis_prompt(
        self,
        state: ResolutionState,
        *,
        attempt_number: int,
        target_python: str,
        runtime_profile: str,
        validation_command: str,
        dependencies: list[str],
        classifier_origin: str,
        error_category: str,
        error_details: str,
        build_log: str,
        run_log: str,
    ) -> str:
        template = self._attempt_failure_analysis_template()
        source_compatibility_hints = json.dumps(
            list(state.get("llm_source_compatibility_hints", [])),
            indent=2,
        )[:1600]
        return template.format(
            attempt_number=attempt_number,
            target_python=target_python or "-",
            runtime_profile=runtime_profile or "-",
            validation_command=validation_command or "-",
            dependencies="\n".join(dependencies) if dependencies else "-",
            classifier_origin=classifier_origin or "-",
            error_category=error_category or "-",
            error_details=self._trim_text_block(error_details, max_lines=12, max_chars=1400),
            source_compatibility_hints=source_compatibility_hints or "[]",
            build_log_excerpt=self._trim_text_block(build_log),
            run_log_excerpt=self._trim_text_block(run_log),
        )

    def _record_attempt_failure_analysis(
        self,
        state: ResolutionState,
        classified: ExecutionOutcome,
    ) -> None:
        if (
            classified.success
            or not state.get("attempt_records")
            or not state.get("attempt_failure_analysis_enabled", False)
        ):
            return
        latest_attempt = state["attempt_records"][-1]
        prompt_text = self._render_attempt_failure_analysis_prompt(
            state,
            attempt_number=latest_attempt.attempt_number,
            target_python=str(state.get("target_python", "") or ""),
            runtime_profile=str(state.get("current_runtime_profile", "") or ""),
            validation_command=str(latest_attempt.validation_command or ""),
            dependencies=list(latest_attempt.dependencies),
            classifier_origin=str(classified.classifier_origin or ""),
            error_category=str(classified.category or ""),
            error_details=str(classified.message or ""),
            build_log=str(classified.build_log or ""),
            run_log=str(classified.run_log or ""),
        )
        analysis_model = self._stage_model_name("analysis")
        try:
            self._trace_request(state, "analysis", prompt_text)
            analysis = self.prompt_runner.invoke_text("analysis", prompt_text).strip()
            self._trace_response(state, "analysis", analysis)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            latest_attempt.llm_failure_analysis = ""
            latest_attempt.llm_failure_analysis_model = analysis_model
            self._emit_activity(
                state,
                kind="failure_analysis_skipped",
                detail=f"Attempt failure analysis was skipped after {exc.__class__.__name__}.",
            )
            return
        latest_attempt.llm_failure_analysis = analysis
        latest_attempt.llm_failure_analysis_model = analysis_model
        state.setdefault("model_outputs", {}).setdefault("attempt_analysis", []).append(
            {
                "attempt": latest_attempt.attempt_number,
                "model": analysis_model,
                "output": analysis,
            }
        )
        self._emit_activity(
            state,
            kind="failure_analysis_recorded",
            detail=f"Recorded LLM failure analysis for attempt {latest_attempt.attempt_number}.",
        )

    @staticmethod
    def _dynamic_import_signals(source_text: str) -> bool:
        lowered = source_text.lower()
        return any(
            token in lowered
            for token in (
                "importlib",
                "__import__(",
                "pkg_resources",
                "entry_points(",
                "find_spec(",
                "plugin",
            )
        )

    def _source_signal_summary(self, state: ResolutionState, *, limit: int = 1800) -> str:
        source_text = "\n".join(state.get("source_files", {}).values())
        lowered = source_text.lower()
        signals: list[dict[str, str]] = []

        def add_signal(signal: str, evidence: str) -> None:
            signals.append({"signal": signal, "evidence": evidence})

        token_signals = (
            ("required tensorflow 0.", "legacy_tensorflow_comment", "required tensorflow 0.x"),
            ("tensorflow.models.", "tensorflow_models_module", "tensorflow.models.* import"),
            ("tensorflow.contrib", "tensorflow_contrib_api", "tensorflow.contrib usage"),
            ("tf.contrib", "tensorflow_contrib_alias_api", "tf.contrib usage"),
            ("tf.app.flags", "tensorflow_app_flags", "tf.app.flags usage"),
            ("tf.flags", "tensorflow_flags_alias", "tf.flags usage"),
            ("tf.merge_all_summaries", "tensorflow_merge_all_summaries", "tf.merge_all_summaries usage"),
            ("tf.train.summarywriter", "tensorflow_summary_writer", "tf.train.SummaryWriter usage"),
            ("tf.summary.filewriter", "tensorflow_filewriter", "tf.summary.FileWriter usage"),
            ("interactivesession(", "tensorflow_interactive_session", "tf.InteractiveSession usage"),
            ("index2word", "gensim_index2word", "gensim index2word usage"),
            ("word2vec.load(", "gensim_word2vec_load", "Word2Vec.load usage"),
        )
        for token, signal, evidence in token_signals:
            if token in lowered:
                add_signal(signal, evidence)
        has_tensorflow_import = bool(
            re.search(r"^\s*(?:from\s+tensorflow\b|import\s+tensorflow\b)", source_text, re.MULTILINE)
        )
        if re.search(r"\bxrange\s*\(", source_text):
            add_signal("python2_xrange", "xrange() usage")
        has_keras_layers_merge = bool(
            re.search(r"^\s*from\s+keras\.layers\s+import\b.*\bmerge\b", source_text, re.MULTILINE)
        )
        if has_keras_layers_merge:
            add_signal("keras_layers_merge_import", "from keras.layers import ... merge")
        has_standalone_keras = bool(
            re.search(r"^\s*(?:from\s+keras(?:\.|\b)|import\s+keras\b)", source_text, re.MULTILINE)
        )
        if has_standalone_keras:
            add_signal("standalone_keras_api", "imports standalone keras package")
        has_legacy_model_input = bool(re.search(r"\bmodel\s*\([^)]*\binput\s*=", source_text, re.IGNORECASE))
        if has_legacy_model_input:
            add_signal("keras_legacy_model_input", "Model(input=...) usage")
        has_legacy_model_output = bool(re.search(r"\bmodel\s*\([^)]*\boutput\s*=", source_text, re.IGNORECASE))
        if has_legacy_model_output:
            add_signal("keras_legacy_model_output", "Model(output=...) usage")
        has_legacy_gym_step = bool(
            re.search(
            r"^\s*[^=\n,]+,\s*[^=\n,]+,\s*[^=\n,]+,\s*[^=\n,]+\s*=\s*.*\.step\(",
            source_text,
            re.MULTILINE,
            )
        )
        if has_legacy_gym_step:
            add_signal("gym_legacy_step_signature", "four-value env.step(...) unpack")
        has_legacy_gym_reset = bool(
            re.search(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*.*\.reset\(\)\s*$", source_text, re.MULTILINE)
        )
        if has_legacy_gym_reset:
            add_signal("gym_legacy_reset_signature", "single-value env.reset() assignment")
        if has_standalone_keras and (has_keras_layers_merge or has_legacy_model_input or has_legacy_model_output):
            add_signal(
                "legacy_standalone_keras_family_candidate",
                "standalone keras with legacy Model()/merge API usage",
            )
        if has_tensorflow_import and has_standalone_keras and (
            has_keras_layers_merge or has_legacy_model_input or has_legacy_model_output
        ):
            add_signal(
                "legacy_standalone_keras_tensorflow_family_candidate",
                "tensorflow import paired with standalone legacy keras API usage",
            )
        if has_standalone_keras and (has_legacy_gym_step or has_legacy_gym_reset):
            add_signal(
                "legacy_standalone_keras_gym_family_candidate",
                "standalone keras paired with legacy gym reset/step usage",
            )
        rendered = json.dumps(signals, indent=2)
        return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."

    def _benchmark_resolution_platform(self, state: ResolutionState) -> str:
        if state.get("mode") != "gistable":
            return ""
        return str(self.settings.benchmark_platform or "").strip()

    def _get_version_options_for_state(
        self,
        state: ResolutionState,
        package: str,
        target_python: str,
        *,
        limit: int,
    ) -> PackageVersionOptions:
        kwargs: dict[str, Any] = {}
        try:
            parameters = inspect.signature(self.pypi_store.get_version_options).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "limit" in parameters:
            kwargs["limit"] = limit
        if "preset" in parameters:
            kwargs["preset"] = self.settings.preset
        platform_override = self._benchmark_resolution_platform(state)
        if platform_override and "platform" in parameters:
            kwargs["platform"] = platform_override
        return self.pypi_store.get_version_options(package, target_python, **kwargs)

    def _should_infer_source_compatibility_hints(self, state: ResolutionState) -> bool:
        if not self._is_research() or not self._uses_full_apd():
            return False
        if state.get("mode") != "gistable":
            return False
        if not state.get("version_options") or not state.get("validation_options"):
            return False
        inferred = {normalize_package_name(package) for package in state.get("inferred_packages", [])}
        return bool(inferred & {"tensorflow", "torch", "keras", "gensim", "gym"})

    def _infer_source_compatibility_hints(self, state: ResolutionState) -> None:
        state["llm_source_compatibility_hints"] = []
        state.setdefault("structured_outputs", {})["source_compatibility"] = {
            "default_runtime_profile": "",
            "compatibility_hints": [],
        }
        if not self._should_infer_source_compatibility_hints(state):
            return
        allowed_packages = self._research_allowed_packages(state)
        if not allowed_packages:
            return
        allowed_runtime_profiles = {
            option.get("profile", "")
            for option in state.get("validation_options", [])
            if option.get("profile")
        }
        source_context = "\n\n".join(
            f"# file: {file_name}\n{code}" for file_name, code in state.get("source_files", {}).items()
        )[:7000]
        prompt_variables = {
            "target_python": state.get("target_python", ""),
            "allowed_packages": "\n".join(sorted(allowed_packages)),
            "version_space": self._planning_version_space_summary(state),
            "validation_options": self._validation_options_summary(state),
            "default_validation_profile": state.get("default_validation_profile", ""),
            "source_signals": self._source_signal_summary(state),
            "raw_file": source_context,
        }
        prompt_text = self._format_prompt("source_compatibility.txt", **prompt_variables)
        self._trace_request(state, "version", prompt_text)
        raw_output = self.prompt_runner.invoke_template("version", "source_compatibility.txt", prompt_variables)
        self._trace_response(state, "version", raw_output)
        state["model_outputs"]["version"].append(
            {
                "attempt": state["current_attempt"],
                "output": raw_output,
                "source": "source_compatibility",
            }
        )
        state["structured_outputs"]["source_compatibility_raw"] = raw_output

        runtime_profile = ""
        hints: list[dict[str, str]] = []
        try:
            runtime_profile, hints = parse_source_compatibility_payload(
                raw_output,
                allowed_packages=allowed_packages,
                allowed_runtime_profiles=allowed_runtime_profiles,
            )
        except StructuredOutputError:
            cleaned_output = self._adjudicate_json(
                state,
                "source_compatibility",
                raw_output,
                '{"default_runtime_profile":"import_specs","compatibility_hints":[{"package":"tensorflow","preferred_specifier":"<2.0.0","reason":"legacy tf1 api"}]}',
            )
            try:
                runtime_profile, hints = parse_source_compatibility_payload(
                    cleaned_output,
                    allowed_packages=allowed_packages,
                    allowed_runtime_profiles=allowed_runtime_profiles,
                )
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                runtime_profile = ""
                hints = []

        state["llm_source_compatibility_hints"] = hints
        payload = {
            "default_runtime_profile": runtime_profile,
            "compatibility_hints": hints,
        }
        state["structured_outputs"]["source_compatibility"] = payload
        self._write_json_artifact(state, "source-compatibility.json", payload)
        if runtime_profile:
            state["default_validation_profile"] = runtime_profile
            self._apply_validation_profile(state, runtime_profile)

    def _source_version_preferences(self, state: ResolutionState) -> dict[str, tuple[str, str]]:
        preferences: dict[str, tuple[str, str]] = {}
        for item in state.get("llm_source_compatibility_hints", []):
            package = normalize_package_name(str(item.get("package", "")))
            preferred_specifier = str(item.get("preferred_specifier", "")).strip()
            reason = str(item.get("reason", "")).strip() or "llm_source_compatibility"
            if not package or not preferred_specifier:
                continue
            preferences[package] = (preferred_specifier, reason)
        return preferences

    def _apply_source_version_preferences(self, state: ResolutionState) -> None:
        preferences = self._source_version_preferences(state)
        if not preferences:
            return
        updated_options: list[PackageVersionOptions] = []
        changed = False
        for option in state.get("version_options", []):
            normalized_package = normalize_package_name(option.package)
            preference = preferences.get(normalized_package)
            if preference is None or not option.versions:
                updated_options.append(option)
                continue
            specifier_text, reason = preference
            try:
                specifier = SpecifierSet(specifier_text)
            except InvalidSpecifier:
                updated_options.append(option)
                continue
            preferred_versions: list[str] = []
            fallback_versions: list[str] = []
            for version in option.versions:
                try:
                    parsed = Version(version)
                except InvalidVersion:
                    fallback_versions.append(version)
                    continue
                if parsed in specifier:
                    preferred_versions.append(version)
                else:
                    fallback_versions.append(version)
            if not preferred_versions:
                updated_options.append(option)
                continue
            reordered_versions = option.versions if self._is_research() else preferred_versions + fallback_versions
            policy_notes = list(option.policy_notes)
            policy_note = f"source_compat_{reason}:{specifier_text}"
            if policy_note not in policy_notes:
                policy_notes.append(policy_note)
            changed |= reordered_versions != option.versions or policy_notes != option.policy_notes
            updated_options.append(
                PackageVersionOptions(
                    package=option.package,
                    versions=reordered_versions,
                    requires_python=dict(option.requires_python),
                    upload_time=dict(option.upload_time),
                    policy_notes=policy_notes,
                    platform_notes={version: list(notes) for version, notes in option.platform_notes.items()},
                    requires_dist={version: list(entries) for version, entries in option.requires_dist.items()},
                )
            )
        if not changed:
            return
        state["version_options"] = updated_options
        state["applied_compatibility_policy"] = {
            option.package: option.policy_notes for option in updated_options if option.policy_notes
        }

    def _should_use_extract_llm(self, state: ResolutionState, extracted_imports: list[str]) -> bool:
        if not extracted_imports:
            return True
        combined_source = "\n".join(state.get("source_files", {}).values())
        if self._dynamic_import_signals(combined_source):
            return True
        if (
            state["mode"] == "project"
            and self.preset_config.extract_llm_for_project_frameworks
            and len(state.get("source_files", {})) > 1
        ):
            return True
        return False

    def _deterministic_dependencies(self, options: list[PackageVersionOptions]) -> list[ResolvedDependency]:
        return [
            ResolvedDependency(name=option.package, version=option.versions[0])
            for option in options
            if option.versions
        ]

    def _should_use_version_llm(self, options: list[PackageVersionOptions]) -> bool:
        multi_version_options = [option for option in options if len(option.versions) > 1]
        if not multi_version_options:
            return False
        mode = self.preset_config.version_prompt_mode
        if mode == "accuracy":
            return True
        risky = any(normalize_package_name(option.package) in COMPATIBILITY_SENSITIVE_PACKAGES for option in options)
        if mode == "high_risk_only":
            return risky
        if mode == "efficient":
            return risky or len(multi_version_options) >= 3
        if mode == "optimized":
            return risky or len(multi_version_options) >= 2
        if mode == "balanced":
            return risky or len(options) >= 3
        if mode == "thorough":
            return risky or len(multi_version_options) >= 1
        return False

    def _repair_allowed_packages(self, state: ResolutionState) -> str:
        allowed_packages = sorted(state.get("inferred_packages", []), key=str.lower)
        return "\n".join(allowed_packages)

    @staticmethod
    def _record_bad_initial_candidate(
        state: ResolutionState,
        *,
        package: str,
        reason: str,
        source: str,
    ) -> None:
        entries = list(state.get("bad_initial_candidates", []))
        candidate = {"package": package, "reason": reason, "source": source}
        if candidate not in entries:
            entries.append(candidate)
        state["bad_initial_candidates"] = entries

    def _candidate_block_reason(
        self,
        state: ResolutionState,
        package: str,
        sources: list[str] | set[str] | tuple[str, ...],
    ) -> str | None:
        normalized_package = normalize_package_name(package)
        source_set = {str(source) for source in sources}
        repo_declared = {
            normalize_package_name(package_name)
            for package_name in state.get("repo_evidence", {}).get("declared_packages", [])
            if isinstance(package_name, str) and package_name.strip()
        }
        repo_alias_targets = {
            normalize_package_name(candidate)
            for packages in state.get("repo_alias_candidates", {}).values()
            for candidate in packages
            if isinstance(candidate, str) and candidate.strip()
        }
        if is_unsupported_import(package):
            return "unsupported_external_runtime"
        if is_trap_package_name(package) and normalized_package not in repo_declared and normalized_package not in repo_alias_targets:
            return "trap_package_without_repo_evidence"
        if (
            is_ambiguous_import(package)
            and normalized_package not in repo_declared
            and normalized_package not in repo_alias_targets
            and "repo_declared" not in source_set
            and "repo_alias" not in source_set
        ):
            return "ambiguous_import_without_repo_evidence"
        return None

    def _sanitize_candidate_provenance(
        self,
        state: ResolutionState,
        provenance: dict[str, str],
    ) -> dict[str, str]:
        filtered: dict[str, str] = {}
        unsupported = set(state.get("unsupported_imports", []))
        ambiguous = set(state.get("ambiguous_imports", []))
        for package, source in provenance.items():
            reason = self._candidate_block_reason(state, package, [source])
            if reason is not None:
                if is_unsupported_import(package):
                    unsupported.add(package)
                if is_ambiguous_import(package) or is_trap_package_name(package):
                    ambiguous.add(package)
                self._record_bad_initial_candidate(state, package=package, reason=reason, source=source)
                continue
            filtered[package] = source
        state["unsupported_imports"] = sorted(unsupported)
        state["ambiguous_imports"] = sorted(ambiguous)
        return filtered

    def _set_inferred_packages_from_provenance(
        self,
        state: ResolutionState,
        provenance: dict[str, str],
    ) -> ResolutionState:
        filtered = self._sanitize_candidate_provenance(state, provenance)
        state["inferred_packages"] = sorted(filtered, key=str.lower)
        state["candidate_provenance"] = filtered
        if state["inferred_packages"]:
            state["repair_skipped_reason"] = ""
        elif state.get("unsupported_imports") and not state.get("ambiguous_imports"):
            state["repair_skipped_reason"] = "unsupported_imports_only"
        elif state.get("unsupported_imports") or state.get("ambiguous_imports"):
            state["repair_skipped_reason"] = "no_supported_candidates"
        else:
            state["repair_skipped_reason"] = ""
        return state

    def _maybe_alias_retry(self, state: ResolutionState) -> list[str] | None:
        if not self.preset_config.allow_alias_retry:
            return None
        match = re.search(r"No module named ['\"]?([A-Za-z0-9_\.]+)['\"]?", state.get("last_error_details", ""))
        if not match:
            return None
        missing_module = match.group(1).split(".", 1)[0]
        alias = runtime_package_alias(missing_module)
        if not alias:
            return None
        normalized_alias = normalize_package_name(alias)
        normalized_inferred = {normalize_package_name(package) for package in state.get("inferred_packages", [])}
        selected_dependencies = list(state.get("selected_dependencies", []))
        if normalized_alias not in normalized_inferred and not any(
            is_trap_package_name(dependency.name) for dependency in selected_dependencies
        ):
            return None
        try:
            option = self._get_version_options_for_state(
                state,
                alias,
                state["target_python"],
                limit=self._version_option_limit(state.get("target_python", "3.12")),
            )
        except FileNotFoundError:
            return None
        if not option.versions:
            return None
        state["repair_outcome"] = "alias_retry"
        repaired_dependencies = [
            dependency.pin()
            for dependency in selected_dependencies
            if normalize_package_name(dependency.name) != normalize_package_name(missing_module)
            and not is_trap_package_name(dependency.name)
            and normalize_package_name(dependency.name) != normalized_alias
        ]
        repaired_dependencies.append(f"{alias}=={option.versions[0]}")
        return sorted(set(repaired_dependencies), key=str.lower)

    def _candidate_provenance_from(self, packages: list[str], extracted_imports: list[str]) -> dict[str, str]:
        return normalize_candidate_packages_with_sources(packages, extracted_imports)

    def _backfill_runtime_alias_imports(self, packages: list[str], extracted_imports: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for package in packages:
            if not package:
                continue
            normalized = normalize_package_name(package)
            if normalized in seen:
                continue
            merged.append(package)
            seen.add(normalized)
            alias = runtime_package_alias(package)
            if alias:
                seen.add(normalize_package_name(alias))
        for import_name in extracted_imports:
            alias = runtime_package_alias(import_name)
            if not alias:
                continue
            normalized_import = normalize_package_name(import_name)
            normalized_alias = normalize_package_name(alias)
            if normalized_import in seen or normalized_alias in seen:
                continue
            merged.append(import_name)
            seen.add(normalized_import)
            seen.add(normalized_alias)
        return merged

    def _uses_full_apd(self) -> bool:
        return self.settings.resolver == "apdr"

    def _resolver_uses_rag(self) -> bool:
        return self.settings.use_rag and self.settings.resolver != "readpye"

    def _is_research(self) -> bool:
        return self.settings.preset == "research"

    def _is_experimental(self) -> bool:
        return self.settings.preset == "experimental"

    def _research_feature_enabled(self, feature: ResearchFeatureName) -> bool:
        return self._is_research() and self.settings.research_feature_enabled(feature)

    @staticmethod
    def _write_json_artifact(state: ResolutionState, filename: str, payload: Any) -> None:
        artifact_dir = state.get("artifact_dir")
        if not artifact_dir:
            return
        Path(artifact_dir, filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _candidate_plan_payload(plans: list[CandidatePlan]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for plan in plans:
            item = {
                "rank": plan.rank,
                "reason": plan.reason,
                "dependencies": [{"name": dep.name, "version": dep.version} for dep in plan.dependencies],
            }
            if plan.runtime_profile:
                item["runtime_profile"] = plan.runtime_profile
            payload.append(item)
        return payload

    @staticmethod
    def _default_validation_option(state: ResolutionState) -> dict[str, str]:
        options = state.get("validation_options", [])
        default_profile = str(state.get("default_validation_profile", "") or "").strip()
        for option in options:
            if option.get("profile") == default_profile:
                return option
        if options:
            return options[0]
        return {"profile": state.get("current_runtime_profile", "docker_cmd"), "command": state.get("current_validation_command", ""), "reason": "current"}

    @staticmethod
    def _runtime_profile_option(state: ResolutionState, profile: str) -> dict[str, str] | None:
        for option in state.get("validation_options", []):
            if option.get("profile") == profile:
                return option
        return None

    @staticmethod
    def _is_runtime_profile_hardware_failure(error_details: str) -> bool:
        return bool(
            re.search(
                r"Illegal instruction|compiled to use AVX instructions|AVX instructions, but these aren't available",
                error_details,
                re.IGNORECASE,
            )
        )

    def _should_prompt_runtime_profile_repair(
        self,
        state: ResolutionState,
        execution: ExecutionOutcome,
        classified: ExecutionOutcome,
    ) -> bool:
        if not self._is_research() or not self._uses_full_apd():
            return False
        if state["current_attempt"] >= self.settings.max_attempts:
            return False
        if not execution.build_succeeded or execution.run_succeeded:
            return False
        if not state.get("selected_dependencies"):
            return False
        if str(state.get("current_runtime_profile", "") or "").strip() == "import_specs":
            return False
        if self._runtime_profile_option(state, "import_specs") is None:
            return False
        normalized_dependencies = {
            normalize_package_name(dependency.name) for dependency in state.get("selected_dependencies", [])
        }
        if not normalized_dependencies & {"tensorflow", "torch"}:
            return False
        return self._is_runtime_profile_hardware_failure(classified.message or execution.run_log or "")

    def _apply_validation_profile(self, state: ResolutionState, runtime_profile: str | None = None) -> None:
        if state.get("mode") != "gistable":
            return
        selected_profile = str(runtime_profile or "").strip()
        selected_option: dict[str, str] | None = None
        if selected_profile:
            for option in state.get("validation_options", []):
                if option.get("profile") == selected_profile:
                    selected_option = option
                    break
        if selected_option is None:
            selected_option = self._default_validation_option(state)
        state["current_runtime_profile"] = str(selected_option.get("profile", "docker_cmd") or "docker_cmd")
        state["current_validation_command"] = str(selected_option.get("command", "") or "")

    def _validation_options_summary(self, state: ResolutionState, *, limit: int = 2500) -> str:
        payload = [
            {
                "profile": option.get("profile", ""),
                "reason": option.get("reason", ""),
                "command": option.get("command", ""),
            }
            for option in state.get("validation_options", [])
        ]
        rendered = json.dumps(payload, indent=2)
        return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."

    def _planning_version_space_summary(self, state: ResolutionState, *, limit: int = 5000) -> str:
        preferred_versions_by_package = self._preferred_bundle_versions(state)
        package_limit = 16
        for version_limit in (24, 16, 12, 8, 6, 4, 2):
            payload = {
                "packages": [
                    {
                        "package": option.package,
                        "versions": (
                            [
                                *preferred_versions_by_package.get(option.package, []),
                                *[
                                    version
                                    for version in option.versions
                                    if version not in set(preferred_versions_by_package.get(option.package, []))
                                ],
                            ]
                        )[:version_limit],
                        "policy_notes": option.policy_notes[:4],
                        "requires_python": {
                            version: option.requires_python.get(version, "")
                            for version in (
                                [
                                    *preferred_versions_by_package.get(option.package, []),
                                    *[
                                        candidate_version
                                        for candidate_version in option.versions
                                        if candidate_version
                                        not in set(preferred_versions_by_package.get(option.package, []))
                                    ],
                                ]
                            )[: min(version_limit, 12)]
                            if option.requires_python.get(version, "")
                        },
                        "platform_notes": {
                            version: option.platform_notes.get(version, [])[:4]
                            for version in (
                                [
                                    *preferred_versions_by_package.get(option.package, []),
                                    *[
                                        candidate_version
                                        for candidate_version in option.versions
                                        if candidate_version
                                        not in set(preferred_versions_by_package.get(option.package, []))
                                    ],
                                ]
                            )[: min(version_limit, 8)]
                            if option.platform_notes.get(version, [])
                        },
                        "requires_dist": {
                            version: option.requires_dist.get(version, [])[:6]
                            for version in (
                                [
                                    *preferred_versions_by_package.get(option.package, []),
                                    *[
                                        candidate_version
                                        for candidate_version in option.versions
                                        if candidate_version
                                        not in set(preferred_versions_by_package.get(option.package, []))
                                    ],
                                ]
                            )[: min(version_limit, 6)]
                            if option.requires_dist.get(version, [])
                        },
                    }
                    for option in state.get("version_options", [])[:package_limit]
                ],
                "unresolved_packages": state.get("unresolved_packages", [])[:8],
            }
            rendered = json.dumps(payload, indent=2)
            if len(rendered) <= limit:
                return rendered
        minimal = json.dumps(
            {
                "packages": [
                    {"package": option.package, "versions": option.versions[:2]}
                    for option in state.get("version_options", [])[:8]
                ]
            },
            indent=2,
        )
        return minimal if len(minimal) <= limit else minimal[: limit - 3] + "..."

    def _candidate_bundle_hint_summary(self, state: ResolutionState, *, limit: int = 3500) -> str:
        payload = {
            "generated_bundles": state.get("structured_outputs", {}).get("candidate_bundles", [])[:8],
            "negotiated_bundles": state.get("structured_outputs", {}).get("version_negotiation", {}).get(
                "retained_candidate_plans",
                [],
            )[:6],
        }
        rendered = json.dumps(payload, indent=2)
        return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."

    def _conflict_note_summary(self, state: ResolutionState, *, limit: int = 2000) -> str:
        payload = [
            {
                "package": note.package,
                "related_package": note.related_package,
                "kind": note.kind,
                "reason": note.reason,
                "severity": note.severity,
            }
            for note in state.get("version_conflict_notes", [])[:16]
        ]
        rendered = json.dumps(payload, indent=2)
        return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."

    def _source_compatibility_hint_summary(self, state: ResolutionState, *, limit: int = 1600) -> str:
        preferences = self._source_version_preferences(state)
        if not preferences:
            return "[]"
        payload = [
            {
                "package": package,
                "preferred_specifier": specifier,
                "reason": reason,
            }
            for package, (specifier, reason) in sorted(preferences.items())
        ]
        rendered = json.dumps(payload, indent=2)
        return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."

    @staticmethod
    def _missing_symbol_signature(error_details: str) -> tuple[str, str] | None:
        patterns = [
            (r"cannot import name ['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", "cannot_import_name"),
            (r"has no attribute ['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", "missing_attribute"),
        ]
        for pattern, kind in patterns:
            match = re.search(pattern, error_details, re.IGNORECASE)
            if match:
                return kind, match.group(1)
        return None

    def _repeated_missing_symbol_failures(self, state: ResolutionState) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for attempt in state.get("attempt_records", []):
            if not attempt.error_details:
                continue
            signature = self._missing_symbol_signature(attempt.error_details)
            if signature is None:
                continue
            kind, symbol = signature
            entry = grouped.setdefault(
                signature,
                {
                    "kind": kind,
                    "symbol": symbol,
                    "attempts": [],
                    "runtime_profiles": [],
                    "dependency_versions": {},
                    "example_error": self._activity_excerpt(attempt.error_details, limit=200),
                },
            )
            if attempt.attempt_number not in entry["attempts"]:
                entry["attempts"].append(attempt.attempt_number)
            runtime_profile = str(attempt.runtime_profile or "").strip()
            if runtime_profile and runtime_profile not in entry["runtime_profiles"]:
                entry["runtime_profiles"].append(runtime_profile)
            for dependency in attempt.dependencies:
                if "==" not in dependency:
                    continue
                name, version = dependency.split("==", 1)
                normalized_name = normalize_package_name(name)
                versions = entry["dependency_versions"].setdefault(normalized_name, [])
                if version not in versions:
                    versions.append(version)
        return [
            entry
            for entry in grouped.values()
            if len(entry["attempts"]) >= 2
        ]

    def _repeated_missing_symbol_summary(self, state: ResolutionState, *, limit: int = 2000) -> str:
        payload = self._repeated_missing_symbol_failures(state)
        if not payload:
            return "[]"
        rendered = json.dumps(payload, indent=2)
        return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."

    @staticmethod
    def _version_family(version: str) -> str:
        parts = version.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
        return version

    def _symbol_compatibility_repair_candidates_summary(
        self,
        state: ResolutionState,
        *,
        limit: int = 2000,
    ) -> str:
        repeated_failures = self._repeated_missing_symbol_failures(state)
        if not repeated_failures:
            return "[]"
        version_options = {
            normalize_package_name(option.package): list(option.versions)
            for option in state.get("version_options", [])
        }
        payload: list[dict[str, Any]] = []
        for failure in repeated_failures:
            package_entries: list[dict[str, Any]] = []
            dependency_versions = failure.get("dependency_versions", {})
            if not isinstance(dependency_versions, dict):
                continue
            for package_name, tried_versions in dependency_versions.items():
                if not isinstance(tried_versions, list):
                    continue
                available_versions = version_options.get(package_name, [])
                if not available_versions:
                    continue
                tried_families = {
                    self._version_family(str(version))
                    for version in tried_versions
                    if str(version).strip()
                }
                untried_families: dict[str, list[str]] = {}
                for version in available_versions:
                    family = self._version_family(version)
                    if family in tried_families:
                        continue
                    untried_families.setdefault(family, []).append(version)
                if not untried_families:
                    continue
                package_entries.append(
                    {
                        "package": package_name,
                        "tried_versions": tried_versions,
                        "tried_families": sorted(tried_families),
                        "untried_families": untried_families,
                    }
                )
            if package_entries:
                payload.append(
                    {
                        "kind": failure.get("kind", ""),
                        "symbol": failure.get("symbol", ""),
                        "packages": package_entries,
                    }
                )
        if not payload:
            return "[]"
        rendered = json.dumps(payload, indent=2)
        return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."

    def _preferred_bundle_versions(self, state: ResolutionState) -> dict[str, list[str]]:
        preferences = self._source_version_preferences(state)
        if not preferences:
            return {}
        preferred_versions: dict[str, list[str]] = {}
        for option in state.get("version_options", []):
            normalized_package = normalize_package_name(option.package)
            preference = preferences.get(normalized_package)
            if preference is None:
                continue
            specifier_text, _reason = preference
            try:
                specifier = SpecifierSet(specifier_text)
            except InvalidSpecifier:
                continue
            matching_versions: list[str] = []
            for version in option.versions:
                try:
                    if Version(version) in specifier:
                        matching_versions.append(version)
                except InvalidVersion:
                    continue
            if not matching_versions and option.versions:
                tail_window = min(6, len(option.versions))
                matching_versions = list(option.versions[-tail_window:])
            if matching_versions:
                preferred_versions[option.package] = matching_versions
        return preferred_versions

    @staticmethod
    def _candidate_plan_signature(plan: CandidatePlan) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (
            str(plan.runtime_profile or "").strip(),
            tuple(
                sorted((normalize_package_name(dependency.name), dependency.version) for dependency in plan.dependencies)
            ),
        )

    @staticmethod
    def _attempt_record_signature(attempt: AttemptRecord) -> tuple[str, tuple[tuple[str, str], ...]]:
        signature: list[tuple[str, str]] = []
        for dependency in attempt.dependencies:
            if "==" not in dependency:
                continue
            name, version = dependency.split("==", 1)
            signature.append((normalize_package_name(name), version))
        return (str(attempt.runtime_profile or "").strip(), tuple(sorted(signature)))

    def _filter_novel_candidate_plans(
        self,
        state: ResolutionState,
        plans: list[CandidatePlan],
    ) -> list[CandidatePlan]:
        attempted_signatures = {
            self._attempt_record_signature(attempt)
            for attempt in state.get("attempt_records", [])
            if attempt.dependencies
        }
        retained: list[CandidatePlan] = []
        seen_signatures: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for plan in plans:
            signature = self._candidate_plan_signature(plan)
            if not signature or signature in attempted_signatures or signature in seen_signatures:
                continue
            retained.append(plan)
            seen_signatures.add(signature)
        return self._rebuild_ranked_plans(retained)

    @staticmethod
    def _requires_python_allows_target(specifier: str, target_python: str) -> bool:
        candidate = str(specifier or "").strip()
        if not candidate:
            return True
        try:
            return Version(target_python) in SpecifierSet(candidate)
        except (InvalidSpecifier, InvalidVersion):
            return True

    @staticmethod
    def _rebuild_ranked_plans(plans: list[CandidatePlan]) -> list[CandidatePlan]:
        return [
            CandidatePlan(
                rank=index + 1,
                reason=plan.reason,
                dependencies=list(plan.dependencies),
                runtime_profile=plan.runtime_profile,
            )
            for index, plan in enumerate(plans)
        ]

    def _augment_ranked_candidate_plans(
        self,
        ranked_plans: list[CandidatePlan],
        *,
        bundles: list[dict[str, Any]],
        max_plan_count: int,
    ) -> tuple[list[CandidatePlan], str]:
        retained: list[CandidatePlan] = []
        seen_signatures: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

        for plan in ranked_plans:
            signature = self._candidate_plan_signature(plan)
            if signature in seen_signatures:
                continue
            retained.append(plan)
            seen_signatures.add(signature)

        llm_plan_count = len(retained)
        for bundle in bundles:
            if len(retained) >= max_plan_count:
                break
            dependencies_payload = bundle.get("dependencies", [])
            if not isinstance(dependencies_payload, list):
                continue
            dependencies: list[CandidateDependency] = []
            for dependency in dependencies_payload:
                if not isinstance(dependency, dict):
                    dependencies = []
                    break
                name = str(dependency.get("name", "")).strip()
                version = str(dependency.get("version", "")).strip()
                if not name or not version:
                    dependencies = []
                    break
                dependencies.append(CandidateDependency(name=name, version=version))
            if not dependencies:
                continue
            fallback_plan = CandidatePlan(
                rank=0,
                reason="deterministic fallback conflict-free bundle",
                dependencies=dependencies,
            )
            signature = self._candidate_plan_signature(fallback_plan)
            if signature in seen_signatures:
                continue
            retained.append(fallback_plan)
            seen_signatures.add(signature)

        if llm_plan_count and len(retained) > llm_plan_count:
            strategy = "llm+fallback-augmented"
        elif llm_plan_count:
            strategy = "llm-selected"
        else:
            strategy = "deterministic-fallback"
        return self._rebuild_ranked_plans(retained[:max_plan_count]), strategy

    @staticmethod
    def _allowed_versions_map(options: list[PackageVersionOptions]) -> dict[str, set[str]]:
        return {normalize_package_name(option.package): set(option.versions) for option in options}

    @staticmethod
    def _build_pypi_evidence_payload(
        options: list[PackageVersionOptions], unresolved_packages: list[str]
    ) -> dict[str, Any]:
        return {
            "packages": [
                {
                    "package": option.package,
                    "versions": option.versions,
                    "requires_python": option.requires_python,
                    "requires_dist": option.requires_dist,
                    "policy_notes": option.policy_notes,
                    "platform_notes": option.platform_notes,
                }
                for option in options
            ],
            "unresolved_packages": unresolved_packages,
        }

    def _research_allowed_packages(self, state: ResolutionState) -> set[str]:
        return {normalize_package_name(package) for package in state.get("inferred_packages", [])}

    def _research_bundle_name(self) -> str:
        return self.settings.research_bundle if self._is_research() else "baseline"

    def _version_option_limit(self, target_python: str) -> int:
        try:
            target = Version(target_python)
        except InvalidVersion:
            return 20
        if self._is_research():
            if target <= Version("3.8"):
                return 80
            if target <= Version("3.10"):
                return 50
            return 30
        if self._is_experimental():
            if target <= Version("3.8"):
                return 60
            return 30
        if self.settings.preset in {"accuracy", "thorough"} and target <= Version("3.8"):
            return 40
        return 20

    def _constraint_top_k(self, target_python: str) -> int:
        try:
            target = Version(target_python)
        except InvalidVersion:
            return 5
        if self._is_research():
            if target <= Version("3.8"):
                return 20
            if target <= Version("3.10"):
                return 12
            return 8
        if self._is_experimental():
            return 12 if target <= Version("3.8") else 8
        return 5

    @staticmethod
    def _strategy_type_from(previous: list[str], current: list[str]) -> str:
        if not previous:
            return "fallback_candidate"
        if previous == current:
            return "same_plan_retry"
        previous_names = {item.split("==", 1)[0] for item in previous}
        current_names = {item.split("==", 1)[0] for item in current}
        if current_names < previous_names:
            return "drop_package"
        if current_names > previous_names:
            return "pin_all"
        previous_versions = {item.split("==", 1)[0]: item.split("==", 1)[1] for item in previous if "==" in item}
        current_versions = {item.split("==", 1)[0]: item.split("==", 1)[1] for item in current if "==" in item}
        if previous_versions and current_versions and previous_versions.keys() == current_versions.keys():
            lowered = False
            raised = False
            for name, version in current_versions.items():
                previous_version = previous_versions.get(name)
                if not previous_version:
                    continue
                try:
                    current_parsed = Version(version)
                    previous_parsed = Version(previous_version)
                except InvalidVersion:
                    continue
                lowered |= current_parsed < previous_parsed
                raised |= current_parsed > previous_parsed
            if lowered and not raised:
                return "downgrade"
            if raised and not lowered:
                return "upgrade"
        return "fallback_candidate"

    def _feedback_memory_summary(self, state: ResolutionState) -> dict[str, Any]:
        summary = summarize_feedback_memory(self.settings.workspace_memory_dir)
        state["feedback_memory_hits"] = len(summary.get("entries", []))
        self._write_json_artifact(state, "repair-feedback-context.json", summary)
        return summary

    def _adjudicate_json(self, state: ResolutionState, stage: str, raw_output: str, schema_hint: str) -> str:
        cleanup_prompt = (
            "Rewrite the following content as strict JSON only. "
            f"Use this schema: {schema_hint}. "
            "Do not include markdown or commentary.\n\n"
            f"{raw_output}"
        )
        self._trace_request(state, "adjudicate", cleanup_prompt)
        cleaned_output = self.prompt_runner.invoke_text("adjudicate", cleanup_prompt)
        self._trace_response(state, "adjudicate", cleaned_output)
        state["model_outputs"]["adjudicate"].append({"stage": stage, "output": cleaned_output})
        return cleaned_output

    def _prepare_selected_dependencies(self, state: ResolutionState) -> None:
        dependencies = sorted(state.get("selected_dependencies", []), key=lambda dep: dep.name.lower())
        state["selected_dependencies"] = dependencies
        state["generated_requirements"] = (
            "\n".join(dependency.pin() for dependency in dependencies) + "\n"
            if dependencies
            else "# no inferred third-party dependencies\n"
        )

    @staticmethod
    def _activity_excerpt(error_details: str, *, limit: int = 120) -> str:
        for line in error_details.splitlines():
            stripped = line.strip()
            if stripped and not stripped.lower().startswith("failure focus:"):
                return stripped if len(stripped) <= limit else stripped[: limit - 3].rstrip() + "..."
        return ""

    @staticmethod
    def _summarize_dependency_pins(
        dependencies: list[ResolvedDependency] | list[CandidateDependency],
        *,
        limit: int = 3,
    ) -> str:
        pins = [dependency.pin() for dependency in dependencies]
        if not pins:
            return "-"
        if len(pins) <= limit:
            return ", ".join(pins)
        return f"{', '.join(pins[:limit])}, +{len(pins) - limit} more"

    def _emit_post_classification_activity(
        self,
        state: ResolutionState,
        *,
        category: str,
        next_step: str,
        suggested_packages: list[str] | None = None,
    ) -> None:
        if next_step == "retry_current_plan":
            if state.get("pending_python_fallback"):
                self._emit_activity(
                    state,
                    kind="python_fallback_planned",
                    detail=(
                        "Retrying with deferred Python fallback "
                        f"{state.get('target_python', '')} after {category}."
                    ),
                )
                return
            package_summary = ", ".join(suggested_packages or state.get("system_dependencies", []))
            detail = (
                f"Retrying current plan after {category}"
                + (f" with bootstrap/system hints: {package_summary}." if package_summary else ".")
            )
            self._emit_activity(state, kind="native_retry_planned", detail=detail)
            return
        if next_step == "select_next_candidate_plan":
            remaining = list(state.get("remaining_candidate_plans", []))
            next_rank = remaining[0].rank if remaining else "next"
            self._emit_activity(
                state,
                kind="candidate_fallback_planned",
                detail=f"Trying fallback candidate plan rank {next_rank} next ({len(remaining)} plan(s) remaining).",
            )
            return
        if next_step in {"repair_prompt_c", "repair_prompt_c_research"}:
            self._emit_activity(
                state,
                kind="repair_planned",
                detail=f"Routing to repair after {category} using the latest build/run logs.",
            )
            return
        if next_step == "finalize_result":
            self._emit_activity(
                state,
                kind="case_finalizing",
                detail=f"No further retries scheduled after {category}.",
            )

    @staticmethod
    def _failure_focus(error_details: str) -> str:
        patterns = (
            r"No module named ['\"]?([A-Za-z0-9_\.]+)['\"]?",
            r"cannot import name ['\"]?([A-Za-z0-9_\.]+)['\"]?",
            r"module ['\"]([A-Za-z0-9_\.]+)['\"] has no attribute ['\"]([A-Za-z0-9_\.]+)['\"]",
            r"(libxcb\.so\.1|libexempi|Exempi library not found|Unknown compiler\(s\)|setuptools\.build_meta|pg_config|zlib|typing|Cython\.Build)",
        )
        for pattern in patterns:
            match = re.search(pattern, error_details, re.IGNORECASE)
            if not match:
                continue
            groups = [group for group in match.groups() if group]
            if groups:
                return " / ".join(groups[:2])
            return match.group(0)
        return ""

    @staticmethod
    def _native_retry_hints(error_details: str) -> tuple[list[str], list[str], list[str]]:
        rules = [
            (
                re.compile(
                    r"Unknown compiler\(s\)|error: command ['\"](?:gcc|g\+\+)['\"] failed|"
                    r"unable to execute ['\"](?:gcc|g\+\+)['\"]|Python\.h: No such file",
                    re.IGNORECASE,
                ),
                ["compiler_toolchain"],
                ["build-essential", "gcc", "g++"],
                [],
            ),
            (
                re.compile(r"gfortran|openblas|lapack|blas|numpy\.distutils", re.IGNORECASE),
                ["scientific_native_stack"],
                ["gfortran", "libopenblas-dev", "liblapack-dev"],
                [],
            ),
            (
                re.compile(r"gobject-introspection|pygobject|libgirepository|pkg-config|cairo", re.IGNORECASE),
                ["gtk_gobject_build"],
                ["libgirepository1.0-dev", "libcairo2-dev", "pkg-config"],
                [],
            ),
            (
                re.compile(r"libxcb\.so\.1|qt|xcb", re.IGNORECASE),
                ["qt_x11_runtime"],
                ["libxcb1"],
                [],
            ),
            (
                re.compile(r"Exempi library not found|libexempi", re.IGNORECASE),
                ["xmp_exempi_runtime"],
                ["libexempi-dev"],
                [],
            ),
            (
                re.compile(r"libhdf5(?:\.so)?|h5py", re.IGNORECASE),
                ["hdf5_native_runtime"],
                ["libhdf5-dev"],
                [],
            ),
            (
                re.compile(r"pg_config executable not found", re.IGNORECASE),
                ["postgres_build_headers"],
                ["libpq-dev"],
                [],
            ),
            (
                re.compile(r"The headers or library files could not be found for zlib", re.IGNORECASE),
                ["zlib_headers"],
                ["zlib1g-dev"],
                [],
            ),
            (
                re.compile(
                    r"Please make sure the libxml2 and libxslt development packages are installed|"
                    r"lxml.*(?:libxml2|libxslt)",
                    re.IGNORECASE,
                ),
                ["lxml_headers"],
                ["libxml2-dev", "libxslt1-dev", "zlib1g-dev"],
                [],
            ),
            (
                re.compile(r"No module named ['\"]?typing['\"]?", re.IGNORECASE),
                ["typing_backport"],
                [],
                ["typing==3.10.0.0"],
            ),
            (
                re.compile(
                    r"No module named ['\"]?Cython\.Build['\"]?|"
                    r"Cython\.Compiler\.Errors\.CompileError.*(?:h5py|gevent|_conv\.pyx|corecext\.pyx)",
                    re.IGNORECASE | re.DOTALL,
                ),
                ["legacy_cython_build"],
                [],
                ["Cython<3"],
            ),
        ]
        hints: list[str] = []
        system_packages: list[str] = []
        bootstrap_pins: list[str] = []
        for pattern, rule_hints, rule_packages, rule_bootstrap_pins in rules:
            if not pattern.search(error_details):
                continue
            for hint in rule_hints:
                if hint not in hints:
                    hints.append(hint)
            for package in rule_packages:
                if package not in system_packages:
                    system_packages.append(package)
            for pin in rule_bootstrap_pins:
                if pin not in bootstrap_pins:
                    bootstrap_pins.append(pin)
        return hints, system_packages, bootstrap_pins

    @staticmethod
    def _requires_python2_runtime_retry(category: str, error_details: str) -> bool:
        if category == "SyntaxError":
            return True
        if category not in {"ImportError", "ModuleNotFoundError", "PackageCompatibilityError"}:
            return False
        if re.search(r"Missing parentheses in call to ['\"]print['\"]", error_details, re.IGNORECASE):
            return True
        for module in sorted(PY2_STDLIB_EXTRAS, key=len, reverse=True):
            if re.search(
                rf"No module named ['\"]?{re.escape(module)}['\"]?",
                error_details,
                re.IGNORECASE,
            ):
                return True
        return False

    def _should_reserve_last_attempt_for_deferred_python_fallback(
        self,
        state: ResolutionState,
        execution: ExecutionOutcome,
    ) -> bool:
        if not self._uses_full_apd() or state.get("mode") != "gistable":
            return False
        if execution.build_succeeded or execution.run_succeeded:
            return False
        if not state.get("deferred_target_python") or state.get("python_fallback_used"):
            return False
        current_target = str(state.get("target_python", "") or "").strip()
        if not current_target.startswith("3"):
            return False
        current_attempt = int(state.get("current_attempt", 0) or 0)
        if current_attempt >= self.settings.max_attempts:
            return False
        return current_attempt >= max(1, self.settings.max_attempts - 1)

    @staticmethod
    def _root_cause_bucket(category: str, error_details: str) -> str:
        lowered = error_details.lower()
        rules = [
            ("tensorflow_api_drift", r"tensorflow\.(?:examples|contrib)|tensorflow['\"] has no attribute ['\"](?:placeholder|flags)"),
            ("pg_config", r"pg_config executable not found"),
            ("zlib_headers", r"headers or library files could not be found for zlib"),
            ("lxml_headers", r"libxml2 and libxslt development packages are installed|lxml.*(?:libxml2|libxslt)"),
            ("typing_backport", r"no module named ['\"]typing['\"]"),
            ("legacy_cython_build", r"no module named ['\"]cython\.build['\"]|cython\.compiler\.errors\.compileerror"),
            ("service_connection", r"serverselectiontimeouterror|connection refused"),
            ("shared_library", r"cannot open shared object file|libxcb\.so\.1|libexempi"),
        ]
        for bucket, pattern in rules:
            if re.search(pattern, lowered, re.IGNORECASE):
                return bucket
        return normalize_package_name(category or "unknown")

    def _refresh_deferred_python_fallback(self, state: ResolutionState) -> None:
        if state.get("mode") != "gistable":
            state["deferred_target_python"] = ""
            return
        if state.get("python_fallback_used"):
            if not state.get("deferred_target_python"):
                state["deferred_target_python"] = "2.7.18"
            return
        current_target = str(state.get("target_python", "") or "").strip()
        if not current_target.startswith("3"):
            state["deferred_target_python"] = ""
            return
        extracted_imports = state.get("extracted_imports", [])
        if _has_python2_only_imports(extracted_imports):
            state["deferred_target_python"] = ""
            return
        source_text = "\n".join(state.get("source_files", {}).values())
        state["deferred_target_python"] = "2.7.18" if not _is_python3_syntax_compatible(source_text) else ""

    def _filter_version_options_for_target_python(self, state: ResolutionState) -> None:
        if self._is_research():
            self.retrieve_version_specific_metadata(state)

        target_python = state.get("target_python", "3.12")
        filtered_options: list[PackageVersionOptions] = []
        unresolved = list(state.get("unresolved_packages", []))
        platform_notes: set[str] = set(state.get("platform_compatibility_notes", []))

        for option in state.get("version_options", []):
            retained_versions: list[str] = []
            retained_requires_python: dict[str, str] = {}
            retained_upload_time: dict[str, str] = {}
            retained_platform_notes: dict[str, list[str]] = {}
            retained_requires_dist: dict[str, list[str]] = {}

            for version in option.versions:
                requires_python = str(option.requires_python.get(version, "") or "").strip()
                if requires_python and not self._requires_python_allows_target(requires_python, target_python):
                    platform_notes.add(
                        f"{option.package} {version}: dropped after recorded Requires-Python {requires_python}"
                    )
                    continue
                retained_versions.append(version)
                retained_requires_python[version] = requires_python
                retained_upload_time[version] = option.upload_time.get(version, "")
                retained_platform_notes[version] = list(option.platform_notes.get(version, []))
                retained_requires_dist[version] = list(option.requires_dist.get(version, []))

            if not retained_versions:
                if option.package not in unresolved:
                    unresolved.append(option.package)
                continue

            filtered_options.append(
                PackageVersionOptions(
                    package=option.package,
                    versions=retained_versions,
                    requires_python=retained_requires_python,
                    upload_time=retained_upload_time,
                    policy_notes=option.policy_notes,
                    platform_notes=retained_platform_notes,
                    requires_dist=retained_requires_dist,
                )
            )

        state["version_options"] = filtered_options
        state["unresolved_packages"] = unresolved
        state["platform_compatibility_notes"] = sorted(platform_notes)
        state["applied_compatibility_policy"] = {
            option.package: option.policy_notes for option in filtered_options if option.policy_notes
        }
        self._apply_source_version_preferences(state)

    def _rebuild_selected_dependencies_for_target_python(
        self,
        state: ResolutionState,
        *,
        package_names: list[str] | None = None,
    ) -> None:
        packages = [
            package
            for package in (package_names or [dependency.name for dependency in state.get("selected_dependencies", [])])
            if package
        ] or list(state.get("inferred_packages", []))
        version_limit = self._version_option_limit(state.get("target_python", "3.12"))
        options: list[PackageVersionOptions] = []
        unresolved: list[str] = []
        platform_notes: set[str] = set()
        for package in packages:
            if not looks_like_package_name(package):
                unresolved.append(package)
                continue
            try:
                option = self._get_version_options_for_state(
                    state,
                    package,
                    state["target_python"],
                    limit=version_limit,
                )
            except FileNotFoundError:
                unresolved.append(package)
                continue
            if not option.versions:
                unresolved.append(package)
                continue
            for version, notes in option.platform_notes.items():
                for note in notes:
                    platform_notes.add(f"{option.package} {version}: {note}")
            options.append(option)
        state["version_options"] = options
        state["unresolved_packages"] = unresolved
        state["platform_compatibility_notes"] = sorted(platform_notes)
        self._filter_version_options_for_target_python(state)
        options = list(state.get("version_options", []))
        unresolved = list(state.get("unresolved_packages", []))
        selected = self._deterministic_dependencies(options)
        state["selected_dependencies"] = sorted(selected, key=lambda dependency: dependency.name.lower())
        self._prepare_selected_dependencies(state)
        state["dependency_reason"] = "deferred_python_fallback"
        state["version_selection_source"] = "deferred_python_fallback"
        if self._is_research():
            pypi_evidence = self._build_pypi_evidence_payload(options, unresolved)
            existing_pypi_evidence = state.get("pypi_evidence", {})
            existing_alias_resolution = existing_pypi_evidence.get("alias_resolution")
            if existing_alias_resolution:
                pypi_evidence["alias_resolution"] = existing_alias_resolution
            metadata_enrichment = existing_pypi_evidence.get("metadata_enrichment")
            if metadata_enrichment:
                pypi_evidence["metadata_enrichment"] = metadata_enrichment
            state["pypi_evidence"] = pypi_evidence
            self._write_json_artifact(state, "pypi-evidence.json", pypi_evidence)
            plan = CandidatePlan(
                rank=1,
                reason="deterministic dependency refresh after deferred Python fallback",
                dependencies=[
                    CandidateDependency(name=dependency.name, version=dependency.version)
                    for dependency in state.get("selected_dependencies", [])
                ],
            )
            state["candidate_plans"] = [plan]
            state["remaining_candidate_plans"] = []
            state["selected_candidate_plan"] = plan
            state["selected_candidate_rank"] = 1
            state["candidate_plan_strategy"] = "deterministic-fallback"

    def _activate_deferred_python_fallback(self, state: ResolutionState) -> None:
        deferred_target = str(state.get("deferred_target_python", "") or "").strip()
        if not deferred_target:
            return
        state["target_python"] = deferred_target
        state["python_version_source"] = "deferred_python_fallback"
        state["python_fallback_used"] = True
        state["pending_python_fallback"] = True
        state["pending_native_retry"] = False
        state["system_dependencies"] = []
        state["system_dependency_hints"] = []
        state["system_packages_attempted"] = []
        state["bootstrap_dependencies"] = []
        state["bootstrap_packages_attempted"] = []
        state["version_options"] = []
        state["unresolved_packages"] = []
        state["platform_compatibility_notes"] = []
        state["constraint_pack"] = None
        state["version_conflict_notes"] = []
        state["python_constraint_intersection"] = []
        state["candidate_plans"] = []
        state["remaining_candidate_plans"] = []
        state["selected_candidate_plan"] = None
        state["selected_candidate_rank"] = None
        state["selected_dependencies"] = []
        state["generated_requirements"] = "# pending model replan after deferred python fallback\n"
        self._emit_activity(
            state,
            kind="python_fallback_activated",
            detail=f"Switching to deferred Python fallback {deferred_target} and restarting model planning.",
        )

    def _should_activate_deferred_python_fallback_after_repair(self, state: ResolutionState) -> bool:
        if not self._uses_full_apd() or state.get("mode") != "gistable":
            return False
        if state.get("python_fallback_used") or state.get("pending_python_fallback"):
            return False
        deferred_target = str(state.get("deferred_target_python", "") or "").strip()
        if not deferred_target:
            return False
        current_target = str(state.get("target_python", "") or "").strip()
        if not current_target.startswith("3"):
            return False
        current_attempt = int(state.get("current_attempt", 0) or 0)
        return current_attempt < self.settings.max_attempts

    def _apply_python_signal_guardrails(self, state: ResolutionState) -> None:
        self._refresh_deferred_python_fallback(state)
        if state.get("mode") != "gistable":
            return
        current_target = str(state.get("target_python", "") or "").strip()
        if not current_target.startswith("3"):
            return
        source_text = "\n".join(state.get("source_files", {}).values())
        resolved_python_version, version_source = reconcile_inferred_target_python(
            current_target,
            benchmark_target_python=state.get("benchmark_target_python", ""),
            source_text=source_text,
            extracted_imports=state.get("extracted_imports", []),
        )
        if version_source not in {
            "benchmark_dockerfile_syntax_guardrail",
            "python2_import_signal",
            "python2_import_signal_default",
        }:
            return
        if resolved_python_version:
            state["target_python"] = resolved_python_version
        state["deferred_target_python"] = ""
        state["python_version_source"] = version_source

    def _maybe_prefer_benchmark_target_python(self, state: ResolutionState) -> None:
        if state.get("mode") != "gistable":
            return
        if state.get("python_version_source") != "llm_prompt_a":
            return
        benchmark_target = str(state.get("benchmark_target_python", "") or "").strip()
        current_target = str(state.get("target_python", "") or "").strip()
        selected_dependencies = list(state.get("selected_dependencies", []))
        if not benchmark_target or not current_target or not selected_dependencies:
            return
        if benchmark_target == current_target:
            return
        benchmark_major = benchmark_target.split(".", 1)[0]
        current_major = current_target.split(".", 1)[0]
        if benchmark_major != current_major or benchmark_major != "3":
            return
        combined_source = "\n".join(state.get("source_files", {}).values())
        if not _is_python3_syntax_compatible(combined_source):
            return
        if _has_python2_only_imports(state.get("extracted_imports", [])):
            return

        option_limit = max(self._version_option_limit(benchmark_target), 120)
        for dependency in selected_dependencies:
            try:
                compatible = self._get_version_options_for_state(
                    state,
                    dependency.name,
                    benchmark_target,
                    limit=option_limit,
                )
            except FileNotFoundError:
                return
            if dependency.version not in set(compatible.versions):
                return

        state["target_python"] = benchmark_target
        state["python_version_source"] = "benchmark_dockerfile_preferred_compatible"

    def _default_research_plans(self, state: ResolutionState) -> list[CandidatePlan]:
        options = state.get("version_options", [])
        if not state.get("inferred_packages"):
            if state.get("unsupported_imports"):
                state["dependency_reason"] = "unsupported_imports"
                return [
                    CandidatePlan(
                        rank=1,
                        reason="imports require unsupported external runtime packages",
                        dependencies=[],
                    )
                ]
            if state.get("ambiguous_imports") or state.get("bad_initial_candidates"):
                state["dependency_reason"] = "no_supported_candidates"
                return [
                    CandidatePlan(
                        rank=1,
                        reason="no safe supported PyPI candidates after guarded resolution",
                        dependencies=[],
                    )
                ]
            state["dependency_reason"] = "stdlib_only"
            return [CandidatePlan(rank=1, reason="no third-party dependencies detected", dependencies=[])]
        if not options:
            state["dependency_reason"] = "no_compatible_versions"
            return [CandidatePlan(rank=1, reason="no compatible versions available", dependencies=[])]
        if all(len(option.versions) == 1 for option in options):
            state["dependency_reason"] = "single_version_fast_path"
            return [
                CandidatePlan(
                    rank=1,
                    reason="single compatible version per package",
                    dependencies=[
                        CandidateDependency(name=option.package, version=option.versions[0])
                        for option in options
                        if option.versions
                    ],
                )
            ]
        state["dependency_reason"] = "deterministic_version_selector"
        return [
            CandidatePlan(
                rank=1,
                reason="deterministic newest compatible stable versions",
                dependencies=[
                    CandidateDependency(name=option.package, version=option.versions[0])
                    for option in options
                    if option.versions
                ],
            )
        ]

    def replan_after_python_fallback(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research() or not state.get("pending_python_fallback"):
            return state
        state["pending_python_fallback"] = False
        state["pending_native_retry"] = False
        state["repaired_dependency_lines"] = []
        state["repair_outcome"] = ""
        state["candidate_plan_strategy"] = ""
        self._emit_activity(
            state,
            kind="python_fallback_replanning",
            detail=f"Replanning dependencies for Python {state.get('target_python', '')}.",
        )
        state = self.retrieve_pypi_metadata(state)
        state = self.resolve_aliases(state)
        state = self.retrieve_version_specific_metadata(state)
        state = self.build_constraint_pack(state)
        state = self.requires_python_intersection_check(state)
        if route_after_constraint_precheck(state) == "finalize_result":
            return state
        state = self.build_rag_context(state)
        state = self.generate_candidate_bundles(state)
        state = self.negotiate_version_bundles(state)
        state = self.generate_candidate_plans(state)
        state = self.select_next_candidate_plan(state)
        return state

    def initial_state_for_case(self, case: BenchmarkCase, run_id: str | None = None) -> ResolutionState:
        return ResolutionState(
            run_id=run_id or uuid4().hex[:12],
            mode="gistable",
            case_id=case.case_id,
            benchmark_case=case,
            case_started_at=datetime.now(timezone.utc).isoformat(),
            attempt_records=[],
            prompt_history={"prompt_a": [], "prompt_b": "", "prompt_c": []},
            model_outputs={"extract": [], "version": [], "repair": [], "adjudicate": [], "attempt_analysis": []},
            current_attempt=0,
            repair_stall_count=0,
            resolver=self.settings.resolver,
            preset=self.settings.preset,
            prompt_profile=self.settings.prompt_profile,
            research_bundle=self.settings.research_bundle,
            research_features=self.settings.research_features,
            dependency_reason="",
            candidate_provenance={},
            repair_outcome="",
            llm_source_compatibility_hints=[],
            applied_compatibility_policy={},
            version_selection_source="",
            candidate_plan_strategy="",
            inference_candidates=[],
            repo_alias_candidates={},
            dynamic_import_candidates=[],
            constraint_pack=None,
            repair_memory_summary=None,
            retry_decision=None,
            runtime_profile_retry_fallback_decision=None,
            strategy_history=[],
            feedback_memory_hits=0,
            version_conflict_notes=[],
            python_constraint_intersection=[],
            top_level_module_map={},
            system_dependencies=[],
            bootstrap_dependencies=[],
            unsupported_imports=[],
            ambiguous_imports=[],
            bad_initial_candidates=[],
            deferred_target_python="",
            python_fallback_used=False,
            pending_native_retry=False,
            pending_runtime_profile_retry=False,
            pending_python_fallback=False,
            classifier_origin="",
            validation_options=[],
            default_validation_profile="",
            system_dependency_hints=[],
            system_packages_attempted=[],
            bootstrap_packages_attempted=[],
            platform_compatibility_notes=[],
            repair_skipped_reason="",
            resolver_implementation="internal",
            repo_evidence={},
            pypi_evidence={},
            rag_context={},
            candidate_plans=[],
            remaining_candidate_plans=[],
            selected_candidate_plan=None,
            selected_candidate_rank=None,
            repair_cycle_count=0,
            repair_model_concluded_impossible=False,
            repair_plan_unavailable_reason="",
            structured_outputs={},
            research_path=self._is_research(),
            structured_prompt_failures=0,
            attempt_failure_analysis_enabled=False,
        )

    def initial_state_for_project(
        self, project_root: Path, validation_command: str | None = None, run_id: str | None = None
    ) -> ResolutionState:
        target = ProjectTarget(
            root_dir=project_root,
            validation_command=validation_command or infer_validation_command(project_root),
            python_files=discover_python_files(project_root),
            dockerfile_path=(project_root / "Dockerfile") if (project_root / "Dockerfile").exists() else None,
        )
        return ResolutionState(
            run_id=run_id or uuid4().hex[:12],
            mode="project",
            case_id=project_root.name,
            project_target=target,
            case_started_at=datetime.now(timezone.utc).isoformat(),
            attempt_records=[],
            prompt_history={"prompt_a": [], "prompt_b": "", "prompt_c": []},
            model_outputs={"extract": [], "version": [], "repair": [], "adjudicate": [], "attempt_analysis": []},
            current_attempt=0,
            repair_stall_count=0,
            resolver=self.settings.resolver,
            preset=self.settings.preset,
            prompt_profile=self.settings.prompt_profile,
            research_bundle=self.settings.research_bundle,
            research_features=self.settings.research_features,
            dependency_reason="",
            candidate_provenance={},
            repair_outcome="",
            llm_source_compatibility_hints=[],
            applied_compatibility_policy={},
            version_selection_source="",
            candidate_plan_strategy="",
            inference_candidates=[],
            repo_alias_candidates={},
            dynamic_import_candidates=[],
            constraint_pack=None,
            repair_memory_summary=None,
            retry_decision=None,
            runtime_profile_retry_fallback_decision=None,
            strategy_history=[],
            feedback_memory_hits=0,
            version_conflict_notes=[],
            python_constraint_intersection=[],
            top_level_module_map={},
            system_dependencies=[],
            bootstrap_dependencies=[],
            unsupported_imports=[],
            ambiguous_imports=[],
            bad_initial_candidates=[],
            deferred_target_python="",
            python_fallback_used=False,
            pending_native_retry=False,
            pending_runtime_profile_retry=False,
            pending_python_fallback=False,
            classifier_origin="",
            validation_options=[],
            default_validation_profile="",
            system_dependency_hints=[],
            system_packages_attempted=[],
            bootstrap_packages_attempted=[],
            platform_compatibility_notes=[],
            repair_skipped_reason="",
            resolver_implementation="internal",
            repo_evidence={},
            pypi_evidence={},
            rag_context={},
            candidate_plans=[],
            remaining_candidate_plans=[],
            selected_candidate_plan=None,
            selected_candidate_rank=None,
            repair_cycle_count=0,
            repair_model_concluded_impossible=False,
            repair_plan_unavailable_reason="",
            structured_outputs={},
            research_path=self._is_research(),
            structured_prompt_failures=0,
            attempt_failure_analysis_enabled=False,
        )

    def run(self, state: ResolutionState) -> ResolutionState:
        try:
            graph = self.build_research_graph() if self._is_research() else self.build_graph()
            return graph.invoke(
                state,
                config={"recursion_limit": infer_graph_recursion_limit(self.settings.max_attempts)},
            )
        except ImportError:  # pragma: no cover - fallback for environments without langgraph
            return self._run_research_fallback(state) if self._is_research() else self._run_fallback(state)

    def _run_fallback(self, state: ResolutionState) -> ResolutionState:
        current = state
        current = self.load_target(current)
        current = self.extract_imports(current)
        current = self.infer_packages_prompt_a(current)
        current = self.retrieve_pypi_metadata(current)
        current = self.infer_versions_prompt_b(current)
        current = self.normalize_dependency_plan(current)
        if route_after_normalize(current) == "finalize_result":
            return self.finalize_result(current)
        while True:
            current = self.materialize_execution_context(current)
            current = self.execute_candidate(current)
            if route_after_execute(current) == "finalize_result":
                return self.finalize_result(current)
            current = self.classify_outcome(current)
            next_step = route_after_classification(current, self.settings.max_attempts)
            if next_step == "retry_current_plan":
                continue
            if next_step == "finalize_result":
                return self.finalize_result(current)
            current = self.repair_prompt_c(current)
            current = self.retrieve_pypi_metadata(current)
            current = self.infer_versions_prompt_b(current)
            current = self.normalize_dependency_plan(current)
            if route_after_normalize(current) == "finalize_result":
                return self.finalize_result(current)

    def _run_research_fallback(self, state: ResolutionState) -> ResolutionState:
        current = state
        current = self.load_target(current)
        current = self.extract_imports(current)
        current = self.extract_dynamic_imports(current)
        current = self.gather_repo_evidence(current)
        current = self.build_dynamic_alias_candidates(current)
        current = self.infer_package_candidates(current)
        current = self.cross_validate_packages(current)
        current = self.retrieve_pypi_metadata(current)
        current = self.resolve_aliases(current)
        current = self.retrieve_version_specific_metadata(current)
        current = self.build_constraint_pack(current)
        current = self.requires_python_intersection_check(current)
        if route_after_constraint_precheck(current) == "finalize_result":
            return self.finalize_result(current)
        current = self.load_feedback_memory_summary(current)
        current = self.build_rag_context(current)
        current = self.generate_candidate_bundles(current)
        current = self.negotiate_version_bundles(current)
        current = self.generate_candidate_plans(current)
        current = self.select_next_candidate_plan(current)
        if not current.get("selected_candidate_plan") or (
            not current.get("selected_dependencies") and current.get("repair_skipped_reason")
        ):
            return self.finalize_result(current)
        while True:
            current = self.materialize_execution_context(current)
            current = self.execute_candidate(current)
            if route_after_execute(current) == "finalize_result":
                return self.finalize_result(current)
            current = self.classify_outcome(current)
            next_step = route_after_research_classification(current, self.settings)
            if next_step == "finalize_result":
                return self.finalize_result(current)
            if next_step == "retry_current_plan":
                continue
            if next_step == "runtime_profile_repair_research":
                current = self.runtime_profile_repair_research(current)
                post_runtime_repair = route_after_research_classification(current, self.settings)
                if post_runtime_repair == "finalize_result":
                    return self.finalize_result(current)
                if post_runtime_repair == "retry_current_plan":
                    continue
                if post_runtime_repair == "select_next_candidate_plan":
                    current = self.select_next_candidate_plan(current)
                    if not current.get("selected_candidate_plan"):
                        return self.finalize_result(current)
                    continue
                if post_runtime_repair == "replan_after_python_fallback":
                    current = self.replan_after_python_fallback(current)
                    if not current.get("selected_candidate_plan"):
                        return self.finalize_result(current)
                    continue
            if next_step == "replan_after_python_fallback":
                current = self.replan_after_python_fallback(current)
                if not current.get("selected_candidate_plan"):
                    return self.finalize_result(current)
                continue
            if next_step == "select_next_candidate_plan":
                current = self.select_next_candidate_plan(current)
                if not current.get("selected_candidate_plan"):
                    return self.finalize_result(current)
                continue
            current = self.build_repair_memory_summary(current)
            current = self.repair_prompt_c_research(current)
            repair_next_step = route_after_research_repair(current, self.settings)
            if repair_next_step == "finalize_result":
                return self.finalize_result(current)
            if repair_next_step == "repair_prompt_c_research":
                continue
            if repair_next_step == "replan_after_python_fallback":
                current = self.replan_after_python_fallback(current)
                if not current.get("selected_candidate_plan"):
                    return self.finalize_result(current)
                continue
            current = self.select_next_candidate_plan(current)
            if not current.get("selected_candidate_plan") or (
                not current.get("selected_dependencies") and current.get("repair_skipped_reason")
            ):
                return self.finalize_result(current)

    def build_graph(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(ResolutionState)
        graph.add_node("load_target", self.load_target)
        graph.add_node("extract_imports", self.extract_imports)
        graph.add_node("infer_packages_prompt_a", self.infer_packages_prompt_a)
        graph.add_node("retrieve_pypi_metadata", self.retrieve_pypi_metadata)
        graph.add_node("infer_versions_prompt_b", self.infer_versions_prompt_b)
        graph.add_node("normalize_dependency_plan", self.normalize_dependency_plan)
        graph.add_node("materialize_execution_context", self.materialize_execution_context)
        graph.add_node("execute_candidate", self.execute_candidate)
        graph.add_node("classify_outcome", self.classify_outcome)
        graph.add_node("repair_prompt_c", self.repair_prompt_c)
        graph.add_node("finalize_result", self.finalize_result)

        graph.add_edge(START, "load_target")
        graph.add_edge("load_target", "extract_imports")
        graph.add_edge("extract_imports", "infer_packages_prompt_a")
        graph.add_edge("infer_packages_prompt_a", "retrieve_pypi_metadata")
        graph.add_edge("retrieve_pypi_metadata", "infer_versions_prompt_b")
        graph.add_edge("infer_versions_prompt_b", "normalize_dependency_plan")
        graph.add_conditional_edges(
            "normalize_dependency_plan",
            route_after_normalize,
            {"materialize_execution_context": "materialize_execution_context", "finalize_result": "finalize_result"},
        )
        graph.add_edge("materialize_execution_context", "execute_candidate")
        graph.add_conditional_edges(
            "execute_candidate",
            route_after_execute,
            {"classify_outcome": "classify_outcome", "finalize_result": "finalize_result"},
        )
        graph.add_conditional_edges(
            "classify_outcome",
            lambda state: route_after_classification(state, self.settings.max_attempts),
            {
                "repair_prompt_c": "repair_prompt_c",
                "retry_current_plan": "materialize_execution_context",
                "finalize_result": "finalize_result",
            },
        )
        graph.add_edge("repair_prompt_c", "retrieve_pypi_metadata")
        graph.add_edge("finalize_result", END)
        return graph.compile()

    def build_research_graph(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(ResolutionState)
        graph.add_node("load_target", self.load_target)
        graph.add_node("extract_imports", self.extract_imports)
        graph.add_node("extract_dynamic_imports", self.extract_dynamic_imports)
        graph.add_node("gather_repo_evidence", self.gather_repo_evidence)
        graph.add_node("build_dynamic_alias_candidates", self.build_dynamic_alias_candidates)
        graph.add_node("infer_package_candidates", self.infer_package_candidates)
        graph.add_node("cross_validate_packages", self.cross_validate_packages)
        graph.add_node("retrieve_pypi_metadata", self.retrieve_pypi_metadata)
        graph.add_node("resolve_aliases", self.resolve_aliases)
        graph.add_node("retrieve_version_specific_metadata", self.retrieve_version_specific_metadata)
        graph.add_node("build_constraint_pack", self.build_constraint_pack)
        graph.add_node("requires_python_intersection_check", self.requires_python_intersection_check)
        graph.add_node("load_feedback_memory_summary", self.load_feedback_memory_summary)
        graph.add_node("build_rag_context", self.build_rag_context)
        graph.add_node("generate_candidate_bundles", self.generate_candidate_bundles)
        graph.add_node("negotiate_version_bundles", self.negotiate_version_bundles)
        graph.add_node("generate_candidate_plans", self.generate_candidate_plans)
        graph.add_node("select_next_candidate_plan", self.select_next_candidate_plan)
        graph.add_node("materialize_execution_context", self.materialize_execution_context)
        graph.add_node("execute_candidate", self.execute_candidate)
        graph.add_node("classify_outcome", self.classify_outcome)
        graph.add_node("runtime_profile_repair_research", self.runtime_profile_repair_research)
        graph.add_node("replan_after_python_fallback", self.replan_after_python_fallback)
        graph.add_node("build_repair_memory_summary", self.build_repair_memory_summary)
        graph.add_node("repair_prompt_c_research", self.repair_prompt_c_research)
        graph.add_node("finalize_result", self.finalize_result)

        graph.add_edge(START, "load_target")
        graph.add_edge("load_target", "extract_imports")
        graph.add_edge("extract_imports", "extract_dynamic_imports")
        graph.add_edge("extract_dynamic_imports", "gather_repo_evidence")
        graph.add_edge("gather_repo_evidence", "build_dynamic_alias_candidates")
        graph.add_edge("build_dynamic_alias_candidates", "infer_package_candidates")
        graph.add_edge("infer_package_candidates", "cross_validate_packages")
        graph.add_edge("cross_validate_packages", "retrieve_pypi_metadata")
        graph.add_edge("retrieve_pypi_metadata", "resolve_aliases")
        graph.add_edge("resolve_aliases", "retrieve_version_specific_metadata")
        graph.add_edge("retrieve_version_specific_metadata", "build_constraint_pack")
        graph.add_edge("build_constraint_pack", "requires_python_intersection_check")
        graph.add_conditional_edges(
            "requires_python_intersection_check",
            route_after_constraint_precheck,
            {"load_feedback_memory_summary": "load_feedback_memory_summary", "finalize_result": "finalize_result"},
        )
        graph.add_edge("load_feedback_memory_summary", "build_rag_context")
        graph.add_edge("build_rag_context", "generate_candidate_bundles")
        graph.add_edge("generate_candidate_bundles", "negotiate_version_bundles")
        graph.add_edge("negotiate_version_bundles", "generate_candidate_plans")
        graph.add_edge("generate_candidate_plans", "select_next_candidate_plan")
        graph.add_conditional_edges(
            "select_next_candidate_plan",
            route_after_research_plan_selection,
            {"materialize_execution_context": "materialize_execution_context", "finalize_result": "finalize_result"},
        )
        graph.add_edge("materialize_execution_context", "execute_candidate")
        graph.add_conditional_edges(
            "execute_candidate",
            route_after_execute,
            {"classify_outcome": "classify_outcome", "finalize_result": "finalize_result"},
        )
        graph.add_conditional_edges(
            "classify_outcome",
            lambda state: route_after_research_classification(state, self.settings),
            {
                "retry_current_plan": "materialize_execution_context",
                "runtime_profile_repair_research": "runtime_profile_repair_research",
                "replan_after_python_fallback": "replan_after_python_fallback",
                "select_next_candidate_plan": "select_next_candidate_plan",
                "repair_prompt_c_research": "build_repair_memory_summary",
                "finalize_result": "finalize_result",
            },
        )
        graph.add_conditional_edges(
            "runtime_profile_repair_research",
            lambda state: route_after_research_classification(state, self.settings),
            {
                "retry_current_plan": "materialize_execution_context",
                "runtime_profile_repair_research": "runtime_profile_repair_research",
                "replan_after_python_fallback": "replan_after_python_fallback",
                "select_next_candidate_plan": "select_next_candidate_plan",
                "repair_prompt_c_research": "build_repair_memory_summary",
                "finalize_result": "finalize_result",
            },
        )
        graph.add_conditional_edges(
            "replan_after_python_fallback",
            route_after_research_plan_selection,
            {"materialize_execution_context": "materialize_execution_context", "finalize_result": "finalize_result"},
        )
        graph.add_edge("build_repair_memory_summary", "repair_prompt_c_research")
        graph.add_conditional_edges(
            "repair_prompt_c_research",
            lambda state: route_after_research_repair(state, self.settings),
            {
                "repair_prompt_c_research": "build_repair_memory_summary",
                "replan_after_python_fallback": "replan_after_python_fallback",
                "select_next_candidate_plan": "select_next_candidate_plan",
                "finalize_result": "finalize_result",
            },
        )
        graph.add_edge("finalize_result", END)
        return graph.compile()

    def load_target(self, state: ResolutionState) -> ResolutionState:
        artifact_dir = self.settings.artifacts_dir / state["run_id"] / state["case_id"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        state["artifact_dir"] = str(artifact_dir)

        if state["mode"] == "gistable":
            case = state["benchmark_case"]
            source = case.snippet_path.read_text(encoding="utf-8")
            dockerfile_text = (
                case.dockerfile_path.read_text(encoding="utf-8")
                if case.dockerfile_path is not None and case.dockerfile_path.exists()
                else ""
            )
            state["source_files"] = {"snippet.py": source}
            state["current_validation_command"] = ""
            state["current_runtime_profile"] = "docker_cmd"
            benchmark_target_python = detect_target_python_from_dockerfile(dockerfile_text)
            state["benchmark_target_python"] = benchmark_target_python
            state["target_python"] = benchmark_target_python
            state["inferred_target_python"] = ""
            state["python_version_source"] = "benchmark_dockerfile" if dockerfile_text else "benchmark_default"
        else:
            target = state["project_target"]
            state["source_files"] = load_python_sources(target.root_dir)
            state["current_validation_command"] = target.validation_command
            state["current_runtime_profile"] = "project"
            state["target_python"] = "3.12"
            state["benchmark_target_python"] = ""
            state["inferred_target_python"] = ""
            state["python_version_source"] = "project_default"

        return state

    def extract_imports(self, state: ResolutionState) -> ResolutionState:
        extracted: set[str] = set()
        for code in state["source_files"].values():
            extracted.update(filter_third_party_imports(extract_import_roots_from_code(code)))
        state["extracted_imports"] = sorted(extracted)
        state["unsupported_imports"] = sorted({item for item in extracted if is_unsupported_import(item)})
        state["ambiguous_imports"] = sorted(
            {
                item
                for item in extracted
                if is_ambiguous_import(item) and not runtime_package_alias(item)
            }
        )
        if state["mode"] == "gistable":
            source = state["source_files"].get("snippet.py", "")
            validation_options, default_profile = build_benchmark_validation_options(source, state["extracted_imports"])
            if (
                default_profile == "docker_cmd"
                and state["benchmark_case"].case_source in {"all-gists", "competition-run"}
            ):
                default_profile = "snippet_exec"
                validation_options = [
                    option
                    for option in validation_options
                    if option.get("profile") != "docker_cmd"
                ]
                validation_options.insert(
                    0,
                    {
                        "profile": "snippet_exec",
                        "command": "python snippet.py",
                        "reason": "benchmark snippet execution default",
                    },
                )
            state["validation_options"] = validation_options
            state["default_validation_profile"] = default_profile
            self._apply_validation_profile(state, default_profile)
        self._apply_python_signal_guardrails(state)
        return state

    def extract_dynamic_imports(self, state: ResolutionState) -> ResolutionState:
        if not self._research_feature_enabled("dynamic_imports"):
            state["dynamic_import_candidates"] = []
            return state
        project_root = state["project_target"].root_dir if state["mode"] == "project" else None
        payload = collect_dynamic_import_candidates(state.get("source_files", {}), project_root=project_root)
        resolved = [
            candidate
            for candidate in payload.get("resolved", [])
            if looks_like_package_name(candidate)
        ]
        state["dynamic_import_candidates"] = resolved
        self._write_json_artifact(state, "dynamic-imports.json", payload)
        return state

    def gather_repo_evidence(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research() or not self.settings.repo_evidence_enabled:
            state["repo_evidence"] = {}
            return state
        evidence = build_repo_evidence(state)
        state["repo_evidence"] = evidence
        self._write_json_artifact(state, "repo-evidence.json", evidence)
        return state

    def build_dynamic_alias_candidates(self, state: ResolutionState) -> ResolutionState:
        if not self._research_feature_enabled("dynamic_aliases"):
            state["repo_alias_candidates"] = {}
            state["top_level_module_map"] = {}
            return state
        alias_map, top_level_module_map = build_repo_alias_candidates(
            state.get("repo_evidence", {}),
            target_python=state.get("target_python", "3.12"),
            pypi_store=self.pypi_store,
            package_metadata_store=self.package_metadata_store,
            preset=self.settings.preset,
        )
        state["repo_alias_candidates"] = alias_map
        state["top_level_module_map"] = top_level_module_map
        self._write_json_artifact(state, "repo-alias-map.json", alias_map)
        self._write_json_artifact(state, "top-level-module-map.json", top_level_module_map)
        return state

    def infer_package_candidates(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return self.infer_packages_prompt_a(state)

        if not self._research_feature_enabled("multipass_inference"):
            state = self.infer_packages_prompt_a(state)
            state["inference_candidates"] = [
                InferenceCandidate(
                    package=package,
                    confidence=1.0 if state.get("candidate_provenance", {}).get(package) != "llm" else 0.9,
                    sources=[state.get("candidate_provenance", {}).get(package, "llm")],
                    reason="baseline research inference",
                    accepted=True,
                )
                for package in state.get("inferred_packages", [])
            ]
            self._write_json_artifact(
                state,
                "package-candidates.json",
                [asdict(candidate) for candidate in state.get("inference_candidates", [])],
            )
            return state

        code = next(iter(state.get("source_files", {}).values()), "")
        repo_evidence_summary = json.dumps(state.get("repo_evidence", {}), indent=2)[:2000]
        prompt_text = self._format_prompt(
            "package_inference.txt",
            raw_file=code,
            extracted_imports="\n".join(state.get("extracted_imports", [])),
            repo_evidence=repo_evidence_summary,
        )
        self._trace_request(state, "extract", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "extract",
            "package_inference.txt",
            {
                "raw_file": code,
                "extracted_imports": "\n".join(state.get("extracted_imports", [])),
                "repo_evidence": repo_evidence_summary,
            },
        )
        self._trace_response(state, "extract", raw_output)
        state["prompt_history"]["prompt_a"] = [prompt_text]
        state["model_outputs"]["extract"] = [{"file": next(iter(state["source_files"].keys()), "snippet.py"), "output": raw_output}]
        state["structured_outputs"]["extract_raw"] = raw_output
        raw_version_output = raw_output
        extract_structured_failed = False
        try:
            llm_candidates = parse_experimental_package_payload(raw_output)
        except StructuredOutputError:
            cleaned_output = self._adjudicate_json(
                state,
                "package_inference",
                raw_output,
                '{"packages":[{"package":"PackageName","confidence":0.0,"source":"llm","evidence":["hint"]}]}',
            )
            try:
                llm_candidates = parse_experimental_package_payload(cleaned_output)
                raw_version_output = cleaned_output
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                llm_candidates = []
                extract_structured_failed = True
        _, inferred_python_version = parse_package_inference_output(raw_version_output)
        if inferred_python_version:
            resolved_python_version, version_source = reconcile_inferred_target_python(
                inferred_python_version,
                benchmark_target_python=state.get("benchmark_target_python", ""),
                source_text=code,
                extracted_imports=state.get("extracted_imports", []),
            )
            state["inferred_target_python"] = inferred_python_version
            if (
                extract_structured_failed
                and version_source == "llm_prompt_a"
                and (self._is_experimental() or self._is_research())
                and state.get("python_version_source") == "benchmark_dockerfile"
            ):
                benchmark_py = state.get("benchmark_target_python", "")
                if benchmark_py:
                    resolved_python_version = benchmark_py
                    version_source = "benchmark_dockerfile_structured_fallback"
            if resolved_python_version:
                state["target_python"] = resolved_python_version
            if version_source:
                state["python_version_source"] = version_source
            self._refresh_deferred_python_fallback(state)
        state["structured_outputs"]["extract"] = llm_candidates
        candidate_sources: dict[str, set[str]] = {}
        candidate_confidence: dict[str, float] = {}
        candidate_reason: dict[str, str] = {}

        for package in state.get("extracted_imports", []):
            normalized = runtime_package_alias(package) or package
            if looks_like_package_name(normalized):
                candidate_sources.setdefault(normalized, set()).add("ast")
        for package in state.get("dynamic_import_candidates", []):
            normalized = runtime_package_alias(package) or package
            if looks_like_package_name(normalized):
                candidate_sources.setdefault(normalized, set()).add("dynamic_import")
        for module, packages in state.get("repo_alias_candidates", {}).items():
            for package in packages:
                if looks_like_package_name(package):
                    candidate_sources.setdefault(package, set()).add("repo_alias")
                    candidate_reason[package] = f"repo alias from module {module}"
        for package in state.get("repo_evidence", {}).get("declared_packages", []):
            if looks_like_package_name(package):
                candidate_sources.setdefault(package, set()).add("repo_declared")
        for item in llm_candidates:
            package = str(item.get("package", "")).strip()
            if not looks_like_package_name(package):
                continue
            if package in STDLIB_MODULES:
                continue
            candidate_sources.setdefault(package, set()).add("llm")
            candidate_confidence[package] = max(candidate_confidence.get(package, 0.0), float(item.get("confidence", 0.0) or 0.0))
            if item.get("evidence"):
                candidate_reason[package] = "; ".join(str(entry) for entry in item.get("evidence", []))

        candidates: list[InferenceCandidate] = []
        for package, sources in sorted(candidate_sources.items(), key=lambda item: item[0].lower()):
            blocked_reason = self._candidate_block_reason(state, package, sorted(sources))
            if blocked_reason is not None:
                if is_unsupported_import(package):
                    state["unsupported_imports"] = sorted(set(state.get("unsupported_imports", [])) | {package})
                if is_ambiguous_import(package) or is_trap_package_name(package):
                    state["ambiguous_imports"] = sorted(set(state.get("ambiguous_imports", [])) | {package})
                self._record_bad_initial_candidate(
                    state,
                    package=package,
                    reason=blocked_reason,
                    source=",".join(sorted(sources)),
                )
                continue
            candidates.append(
                InferenceCandidate(
                    package=package,
                    confidence=candidate_confidence.get(package, 1.0 if "llm" not in sources else 0.75),
                    sources=sorted(sources),
                    reason=candidate_reason.get(package, ""),
                    accepted=False,
                )
            )
        state["inference_candidates"] = candidates
        self._write_json_artifact(state, "package-candidates.json", [asdict(candidate) for candidate in candidates])
        return state

    def cross_validate_packages(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        if not self._research_feature_enabled("multipass_inference"):
            return state
        candidates = state.get("inference_candidates", [])
        if not candidates:
            return state
        deterministic_accepted: list[InferenceCandidate] = []
        pending: list[InferenceCandidate] = []
        for candidate in candidates:
            non_llm_sources = [source for source in candidate.sources if source != "llm"]
            if len(non_llm_sources) >= 2:
                candidate.accepted = True
                deterministic_accepted.append(candidate)
            elif len(non_llm_sources) == 1 and candidate.confidence >= 0.85:
                candidate.accepted = True
                deterministic_accepted.append(candidate)
            else:
                pending.append(candidate)
        accepted = list(deterministic_accepted)
        rejected: list[dict[str, str]] = []
        if pending and self._uses_full_apd():
            candidates_json = json.dumps([asdict(candidate) for candidate in pending], indent=2)
            prompt_text = self._format_prompt(
                "package_cross_validate.txt",
                target_python=state.get("target_python", ""),
                candidate_payload=candidates_json,
                repo_evidence=json.dumps(state.get("repo_evidence", {}), indent=2)[:2500],
                enabled_features=", ".join(state.get("research_features", ())),
            )
            self._trace_request(state, "extract", prompt_text)
            raw_output = self.prompt_runner.invoke_template(
                "extract",
                "package_cross_validate.txt",
                {
                    "target_python": state.get("target_python", ""),
                    "candidate_payload": candidates_json,
                    "repo_evidence": json.dumps(state.get("repo_evidence", {}), indent=2)[:2500],
                    "enabled_features": ", ".join(state.get("research_features", ())),
                },
            )
            self._trace_response(state, "extract", raw_output)
            state["structured_outputs"]["package_cross_validate_raw"] = raw_output
            try:
                accepted_payload, rejected = parse_cross_validation_payload(raw_output)
            except StructuredOutputError:
                cleaned = self._adjudicate_json(
                    state,
                    "package_cross_validate",
                    raw_output,
                    '{"accepted_packages":[{"package":"name","confidence":0.9,"sources":["ast","llm"],"reason":"short"}],"rejected_packages":[{"package":"name","reason":"short"}]}',
                )
                try:
                    accepted_payload, rejected = parse_cross_validation_payload(cleaned)
                except StructuredOutputError:
                    state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                    accepted_payload, rejected = [], []
            accepted_map = {item["package"]: item for item in accepted_payload}
            for candidate in pending:
                if candidate.package in accepted_map:
                    candidate.accepted = True
                    candidate.confidence = accepted_map[candidate.package]["confidence"]
                    candidate.sources = accepted_map[candidate.package]["sources"]
                    candidate.reason = accepted_map[candidate.package]["reason"]
                    accepted.append(candidate)
        state["structured_outputs"]["package_cross_validation"] = {
            "accepted_packages": [asdict(candidate) for candidate in accepted],
            "rejected_packages": rejected,
        }
        self._write_json_artifact(state, "package-cross-validation.json", state["structured_outputs"]["package_cross_validation"])
        provenance = self._candidate_provenance_from([candidate.package for candidate in accepted], state.get("extracted_imports", []))
        for candidate in accepted:
            provenance[candidate.package] = candidate.sources[0] if candidate.sources else provenance.get(candidate.package, "llm")
        return self._set_inferred_packages_from_provenance(state, provenance)

    def build_rag_context(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        self._emit_activity(
            state,
            kind="planning_stage_started",
            detail="Building research RAG context.",
        )
        pypi_evidence = state.get("pypi_evidence", {})
        repo_evidence = state.get("repo_evidence", {})
        rag_context = build_research_rag_context(
            state,
            repo_evidence=repo_evidence,
            pypi_evidence=pypi_evidence,
        )
        state["rag_context"] = rag_context
        self._write_json_artifact(state, "rag-context.json", rag_context)
        self._emit_activity(
            state,
            kind="planning_stage_finished",
            detail="Research RAG context built.",
        )
        return state

    def infer_packages_prompt_a(self, state: ResolutionState) -> ResolutionState:
        extracted_imports = state.get("extracted_imports", [])
        if self._is_research():
            return self._infer_packages_prompt_a_research(state, extracted_imports)
        if not self._uses_full_apd():
            provenance = self._candidate_provenance_from(extracted_imports, extracted_imports)
            normalized = sorted(provenance, key=str.lower)
            state["prompt_history"]["prompt_a"] = []
            state["model_outputs"]["extract"] = [
                {
                    "file": next(iter(state["source_files"].keys()), "snippet.py"),
                    "output": "\n".join(normalized),
                    "source": f"{self.settings.resolver}_static_imports",
                }
            ]
            return self._set_inferred_packages_from_provenance(state, provenance)

        if (
            extracted_imports
            and not self._should_use_extract_llm(state, extracted_imports)
            and state.get("mode") != "gistable"
        ):
            provenance = self._candidate_provenance_from(extracted_imports, extracted_imports)
            normalized = sorted(provenance, key=str.lower)
            if normalized:
                state["prompt_history"]["prompt_a"] = []
                state["model_outputs"]["extract"] = [
                    {
                        "file": next(iter(state["source_files"].keys()), "snippet.py"),
                        "output": "\n".join(normalized),
                        "source": "fast_path_imports",
                    }
                ]
                return self._set_inferred_packages_from_provenance(state, provenance)

        inferred: set[str] = set()
        prompt_texts: list[str] = []
        outputs: list[dict[str, str]] = []

        for file_name, code in state["source_files"].items():
            prompt_text = self._format_prompt("initial_imports.txt", raw_file=code)
            prompt_texts.append(prompt_text)
            self._trace_request(state, "extract", prompt_text)
            raw_output = self.prompt_runner.invoke_template("extract", "initial_imports.txt", {"raw_file": code})
            self._trace_response(state, "extract", raw_output)
            outputs.append({"file": file_name, "output": raw_output})
            packages, inferred_python_version = parse_package_inference_output(raw_output)
            extract_structured_failed = False
            if inferred_python_version:
                resolved_python_version, version_source = reconcile_inferred_target_python(
                    inferred_python_version,
                    benchmark_target_python=state.get("benchmark_target_python", ""),
                    source_text=code,
                    extracted_imports=state.get("extracted_imports", []),
                )
                state["inferred_target_python"] = inferred_python_version
                if (
                    extract_structured_failed
                    and version_source == "llm_prompt_a"
                    and (self._is_experimental() or self._is_research())
                    and state.get("python_version_source") == "benchmark_dockerfile"
                ):
                    benchmark_py = state.get("benchmark_target_python", "")
                    if benchmark_py:
                        resolved_python_version = benchmark_py
                        version_source = "benchmark_dockerfile_structured_fallback"
                if resolved_python_version:
                    state["target_python"] = resolved_python_version
                if version_source:
                    state["python_version_source"] = version_source
                self._refresh_deferred_python_fallback(state)
            for package in packages:
                if package:
                    inferred.add(package)

        state["prompt_history"]["prompt_a"] = prompt_texts
        state["model_outputs"]["extract"] = outputs
        if not inferred:
            inferred.update(extracted_imports)
        inferred_candidates = self._backfill_runtime_alias_imports(sorted(inferred), extracted_imports)
        provenance = self._candidate_provenance_from(inferred_candidates, extracted_imports)
        normalized = sorted(provenance, key=str.lower)
        if self.preset_config.accuracy_extract_cleanup and inferred and not normalized and extracted_imports:
            cleanup_prompt = (
                "Rewrite the following text as plain package names only, one per line, with no explanations.\n\n"
                + "\n".join(sorted(inferred))
            )
            self._trace_request(state, "adjudicate", cleanup_prompt)
            cleaned_output = self.prompt_runner.invoke_text("adjudicate", cleanup_prompt)
            self._trace_response(state, "adjudicate", cleaned_output)
            state["model_outputs"]["adjudicate"].append(cleaned_output)
            cleaned_candidates = [line.strip() for line in cleaned_output.splitlines() if line.strip()]
            provenance = self._candidate_provenance_from(cleaned_candidates, extracted_imports)
            normalized = sorted(provenance, key=str.lower)
        if not normalized and extracted_imports:
            provenance = self._candidate_provenance_from(extracted_imports, extracted_imports)
            normalized = sorted(provenance, key=str.lower)
        return self._set_inferred_packages_from_provenance(state, provenance)

    def _infer_packages_prompt_a_research(
        self, state: ResolutionState, extracted_imports: list[str]
    ) -> ResolutionState:
        if not self._uses_full_apd():
            provenance = self._candidate_provenance_from(extracted_imports, extracted_imports)
            return self._set_inferred_packages_from_provenance(state, provenance)

        code = next(iter(state["source_files"].values()), "")
        repo_evidence_summary = json.dumps(state.get("repo_evidence", {}), indent=2)[:2000]
        prompt_text = self._format_prompt(
            "initial_imports.txt",
            raw_file=code,
            extracted_imports="\n".join(extracted_imports),
            repo_evidence=repo_evidence_summary,
        )
        self._trace_request(state, "extract", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "extract",
            "initial_imports.txt",
            {
                "raw_file": code,
                "extracted_imports": "\n".join(extracted_imports),
                "repo_evidence": repo_evidence_summary,
            },
        )
        self._trace_response(state, "extract", raw_output)
        state["prompt_history"]["prompt_a"] = [prompt_text]
        state["model_outputs"]["extract"] = [{"file": next(iter(state["source_files"].keys()), "snippet.py"), "output": raw_output}]
        state["structured_outputs"]["extract_raw"] = raw_output

        raw_version_output = raw_output
        extract_structured_failed = False
        try:
            packages = parse_experimental_package_payload(raw_output)
        except StructuredOutputError:
            cleaned_output = self._adjudicate_json(
                state,
                "extract",
                raw_output,
                '{"packages":[{"package":"PackageName","confidence":0.0,"source":"llm","evidence":["hint"]}]}',
            )
            try:
                packages = parse_experimental_package_payload(cleaned_output)
                raw_version_output = cleaned_output
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                packages = []
                extract_structured_failed = True

        inferred = [entry["package"] for entry in packages]
        _, inferred_python_version = parse_package_inference_output(raw_version_output)
        state["structured_outputs"]["extract"] = packages
        if inferred_python_version:
            resolved_python_version, version_source = reconcile_inferred_target_python(
                inferred_python_version,
                benchmark_target_python=state.get("benchmark_target_python", ""),
                source_text=code,
                extracted_imports=state.get("extracted_imports", []),
            )
            state["inferred_target_python"] = inferred_python_version
            if (
                extract_structured_failed
                and version_source == "llm_prompt_a"
                and (self._is_experimental() or self._is_research())
                and state.get("python_version_source") == "benchmark_dockerfile"
            ):
                benchmark_py = state.get("benchmark_target_python", "")
                if benchmark_py:
                    resolved_python_version = benchmark_py
                    version_source = "benchmark_dockerfile_structured_fallback"
            if resolved_python_version:
                state["target_python"] = resolved_python_version
            if version_source:
                state["python_version_source"] = version_source
            self._refresh_deferred_python_fallback(state)
        if not inferred:
            inferred = extracted_imports
        inferred_candidates = self._backfill_runtime_alias_imports(inferred, extracted_imports)
        provenance = self._candidate_provenance_from(inferred_candidates, extracted_imports)
        for entry in packages:
            normalized_name = next(
                (package for package in provenance if normalize_package_name(package) == normalize_package_name(entry["package"])),
                None,
            )
            if normalized_name:
                provenance[normalized_name] = str(entry.get("source", provenance[normalized_name]) or provenance[normalized_name])
        return self._set_inferred_packages_from_provenance(state, provenance)

    def retrieve_pypi_metadata(self, state: ResolutionState) -> ResolutionState:
        allowed_packages = state.get("inferred_packages", [])
        if state.get("repaired_dependency_lines"):
            try:
                repaired = filter_allowed_dependencies(
                    parse_dependency_lines("\n".join(state["repaired_dependency_lines"])),
                    allowed_packages,
                )
            except ValueError:
                repaired = []
            packages = [dependency.name for dependency in repaired]
        else:
            packages = allowed_packages
        if not self._resolver_uses_rag():
            state["version_options"] = []
            state["unresolved_packages"] = []
            state["applied_compatibility_policy"] = {}
            state["pypi_evidence"] = {}
            return state
        options = []
        unresolved: list[str] = []
        platform_notes: set[str] = set(state.get("platform_compatibility_notes", []))
        version_limit = self._version_option_limit(state.get("target_python", "3.12"))
        for package in packages:
            if not package:
                continue
            if not looks_like_package_name(package):
                unresolved.append(package)
                continue
            try:
                option = self._get_version_options_for_state(
                    state,
                    package,
                    state["target_python"],
                    limit=version_limit,
                )
            except FileNotFoundError:
                unresolved.append(package)
                continue
            if not option.versions:
                unresolved.append(package)
                continue
            for version, notes in option.platform_notes.items():
                for note in notes:
                    platform_notes.add(f"{option.package} {version}: {note}")
            options.append(option)
        state["version_options"] = options
        state["unresolved_packages"] = unresolved
        state["platform_compatibility_notes"] = sorted(platform_notes)
        state["applied_compatibility_policy"] = {
            option.package: option.policy_notes for option in options if option.policy_notes
        }
        self._infer_source_compatibility_hints(state)
        self._apply_source_version_preferences(state)
        if self._is_research():
            pypi_evidence = self._build_pypi_evidence_payload(
                state.get("version_options", []),
                unresolved,
            )
            state["pypi_evidence"] = pypi_evidence
            self._write_json_artifact(state, "pypi-evidence.json", pypi_evidence)
        return state

    def resolve_aliases(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research() or not self._uses_full_apd():
            return state
        unresolved = [package for package in state.get("unresolved_packages", []) if looks_like_package_name(package)]
        if not unresolved:
            return state

        source_context = "\n\n".join(
            f"# file: {file_name}\n{code}" for file_name, code in state.get("source_files", {}).items()
        )[:6000]
        unresolved_text = "\n".join(sorted(set(unresolved), key=str.lower))
        prompt_text = self._format_prompt(
            "resolve_aliases.txt",
            unresolved_packages=unresolved_text,
            raw_file=source_context,
        )
        self._trace_request(state, "extract", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "extract",
            "resolve_aliases.txt",
            {
                "unresolved_packages": unresolved_text,
                "raw_file": source_context,
            },
        )
        self._trace_response(state, "extract", raw_output)
        state["structured_outputs"]["resolve_aliases_raw"] = raw_output
        try:
            aliases = parse_alias_resolution_payload(raw_output)
        except StructuredOutputError:
            cleaned = self._adjudicate_json(
                state,
                "resolve_aliases",
                raw_output,
                '{"aliases":[{"import_name":"yaml","pypi_package":"PyYAML"}]}',
            )
            try:
                aliases = parse_alias_resolution_payload(cleaned)
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                aliases = []

        options_by_normalized = {
            normalize_package_name(option.package): option for option in state.get("version_options", [])
        }
        inferred_by_normalized = {
            normalize_package_name(package): package for package in state.get("inferred_packages", [])
        }
        provenance = dict(state.get("candidate_provenance", {}))
        resolved_aliases: list[dict[str, str]] = []
        rejected_aliases: list[dict[str, str]] = []
        resolved_imports: set[str] = set()
        version_limit = self._version_option_limit(state.get("target_python", "3.12"))

        for alias in aliases:
            import_name = alias["import_name"].split(".", 1)[0].strip()
            package = alias["pypi_package"].strip()
            if not import_name or not looks_like_package_name(package):
                rejected_aliases.append(
                    {
                        "import_name": import_name,
                        "pypi_package": package,
                        "reason": "invalid_package_name",
                    }
                )
                continue
            if is_unsupported_import(import_name):
                rejected_aliases.append(
                    {
                        "import_name": import_name,
                        "pypi_package": package,
                        "reason": "unsupported_external_runtime",
                    }
                )
                state["unsupported_imports"] = sorted(set(state.get("unsupported_imports", [])) | {import_name})
                self._record_bad_initial_candidate(
                    state,
                    package=package,
                    reason="unsupported_external_runtime",
                    source="alias",
                )
                continue
            blocked_reason = self._candidate_block_reason(state, package, ["alias"])
            if blocked_reason is not None:
                rejected_aliases.append(
                    {
                        "import_name": import_name,
                        "pypi_package": package,
                        "reason": blocked_reason,
                    }
                )
                if is_ambiguous_import(package) or is_trap_package_name(package):
                    state["ambiguous_imports"] = sorted(set(state.get("ambiguous_imports", [])) | {package})
                self._record_bad_initial_candidate(
                    state,
                    package=package,
                    reason=blocked_reason,
                    source="alias",
                )
                continue
            try:
                option = self._get_version_options_for_state(
                    state,
                    package,
                    state["target_python"],
                    limit=version_limit,
                )
            except FileNotFoundError:
                rejected_aliases.append(
                    {
                        "import_name": import_name,
                        "pypi_package": package,
                        "reason": "not_found_on_pypi",
                    }
                )
                continue
            if not option.versions:
                rejected_aliases.append(
                    {
                        "import_name": import_name,
                        "pypi_package": package,
                        "reason": "no_compatible_versions",
                    }
                )
                continue
            normalized_package = normalize_package_name(option.package)
            normalized_import = normalize_package_name(import_name)
            if normalized_import != normalized_package:
                inferred_by_normalized.pop(normalized_import, None)
                for key in list(provenance):
                    if normalize_package_name(key) == normalized_import:
                        provenance.pop(key, None)
            if normalized_package not in options_by_normalized:
                options_by_normalized[normalized_package] = option
            canonical_name = inferred_by_normalized.get(normalized_package, option.package)
            inferred_by_normalized[normalized_package] = canonical_name
            provenance[canonical_name] = "alias"
            resolved_imports.add(normalized_import)
            resolved_aliases.append(
                {
                    "import_name": import_name,
                    "pypi_package": option.package,
                }
            )

        state["version_options"] = list(options_by_normalized.values())
        state["inferred_packages"] = sorted(inferred_by_normalized.values(), key=str.lower)
        state["candidate_provenance"] = provenance
        state["unresolved_packages"] = [
            package
            for package in state.get("unresolved_packages", [])
            if normalize_package_name(package) not in resolved_imports
        ]
        if self._is_research():
            pypi_evidence = self._build_pypi_evidence_payload(
                state.get("version_options", []),
                state.get("unresolved_packages", []),
            )
            pypi_evidence["alias_resolution"] = {
                "resolved_aliases": resolved_aliases,
                "rejected_aliases": rejected_aliases,
            }
            state["pypi_evidence"] = pypi_evidence
            self._write_json_artifact(state, "pypi-evidence.json", pypi_evidence)
        self._write_json_artifact(
            state,
            "alias-resolutions.json",
            {
                "unresolved_before": unresolved,
                "resolved_aliases": resolved_aliases,
                "rejected_aliases": rejected_aliases,
                "unresolved_after": state.get("unresolved_packages", []),
            },
        )
        return state

    def retrieve_version_specific_metadata(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        if not (
            self._research_feature_enabled("transitive_conflicts")
            or self._research_feature_enabled("version_negotiation")
            or self._research_feature_enabled("dynamic_aliases")
        ):
            return state
        self._emit_activity(
            state,
            kind="planning_stage_started",
            detail=(
                "Enriching release metadata for "
                f"{len(state.get('version_options', []))} package(s)."
            ),
        )
        updated_options: list[PackageVersionOptions] = []
        top_k = self._constraint_top_k(state.get("target_python", "3.12"))
        target_python = state.get("target_python", "3.12")
        platform_notes: set[str] = set(state.get("platform_compatibility_notes", []))
        metadata_enrichment: list[dict[str, Any]] = []
        for option in state.get("version_options", []):
            retained_versions: list[str] = []
            retained_requires_python: dict[str, str] = {}
            retained_requires_dist: dict[str, list[str]] = {}
            retained_upload_time: dict[str, str] = {}
            retained_platform_notes: dict[str, list[str]] = {}
            dropped_versions: list[dict[str, str]] = []
            metadata_cache: dict[str, dict[str, Any]] = {}

            def load_release_metadata(version: str) -> dict[str, Any]:
                cached = metadata_cache.get(version)
                if cached is not None:
                    return cached
                release_files = self.pypi_store.release_files(option.package, version)
                cached = self.package_metadata_store.parse_release_metadata(
                    option.package,
                    version,
                    release_files=release_files,
                )
                metadata_cache[version] = cached
                return cached

            trusted_version_cap = max(top_k, self.settings.candidate_plan_count * 4)
            for version in option.versions:
                existing_requires_python = str(option.requires_python.get(version, "") or "").strip()
                metadata: dict[str, Any] | None = None
                requires_python = existing_requires_python
                if not requires_python and len(retained_versions) < trusted_version_cap:
                    metadata = load_release_metadata(version)
                    requires_python = str(metadata.get("requires_python", "") or "").strip()
                if requires_python and not self._requires_python_allows_target(requires_python, target_python):
                    dropped_versions.append(
                        {
                            "version": version,
                            "reason": f"metadata_requires_python:{requires_python}",
                        }
                    )
                    platform_notes.add(
                        f"{option.package} {version}: dropped after metadata Requires-Python {requires_python}"
                    )
                    continue
                retained_versions.append(version)
                retained_requires_python[version] = requires_python
                retained_upload_time[version] = option.upload_time.get(version, "")
                retained_platform_notes[version] = list(option.platform_notes.get(version, []))
                if not requires_python and len(retained_versions) > trusted_version_cap:
                    retained_platform_notes[version].append("metadata_not_scanned")
                if metadata is None and len(retained_requires_dist) < top_k:
                    metadata = load_release_metadata(version)
                if metadata is not None:
                    retained_requires_dist[version] = [str(item) for item in metadata.get("requires_dist", [])]
                else:
                    retained_requires_dist[version] = list(option.requires_dist.get(version, []))

            metadata_enrichment.append(
                {
                    "package": option.package,
                    "retained_versions": list(retained_versions),
                    "dropped_versions": dropped_versions,
                }
            )
            updated_options.append(
                PackageVersionOptions(
                    package=option.package,
                    versions=retained_versions,
                    requires_python=retained_requires_python,
                    upload_time=retained_upload_time,
                    policy_notes=option.policy_notes,
                    platform_notes=retained_platform_notes,
                    requires_dist=retained_requires_dist,
                )
            )
        if updated_options:
            state["version_options"] = updated_options
            state["platform_compatibility_notes"] = sorted(platform_notes)
            state["applied_compatibility_policy"] = {
                option.package: option.policy_notes for option in updated_options if option.policy_notes
            }
            self._apply_source_version_preferences(state)
            pypi_evidence = self._build_pypi_evidence_payload(
                state.get("version_options", []),
                state.get("unresolved_packages", []),
            )
            existing_alias_resolution = state.get("pypi_evidence", {}).get("alias_resolution")
            if existing_alias_resolution:
                pypi_evidence["alias_resolution"] = existing_alias_resolution
            pypi_evidence["metadata_enrichment"] = metadata_enrichment
            state["pypi_evidence"] = pypi_evidence
            self._write_json_artifact(state, "pypi-evidence.json", pypi_evidence)
        self._emit_activity(
            state,
            kind="planning_stage_finished",
            detail=(
                "Release metadata enrichment finished for "
                f"{len(updated_options)} package(s)."
            ),
        )
        return state

    def build_constraint_pack(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        if not (
            self._research_feature_enabled("transitive_conflicts")
            or self._research_feature_enabled("python_constraint_intersection")
            or self._research_feature_enabled("version_negotiation")
        ):
            state["constraint_pack"] = None
            state["version_conflict_notes"] = []
            state["python_constraint_intersection"] = []
            return state
        self._emit_activity(
            state,
            kind="planning_stage_started",
            detail="Building constraint pack from version options.",
        )
        pack = build_constraint_pack(
            state.get("version_options", []),
            target_python=state.get("target_python", "3.12"),
            top_k=None,
        )
        state["constraint_pack"] = pack
        state["version_conflict_notes"] = list(pack.conflict_notes)
        state["python_constraint_intersection"] = list(pack.python_intersection)
        self._write_json_artifact(state, "constraint-pack.json", constraint_pack_to_dict(pack))
        self._write_json_artifact(
            state,
            "conflict-notes.json",
            [
                {
                    "package": note.package,
                    "related_package": note.related_package,
                    "kind": note.kind,
                    "reason": note.reason,
                    "severity": note.severity,
                }
                for note in pack.conflict_notes
            ],
        )
        self._emit_activity(
            state,
            kind="planning_stage_finished",
            detail=(
                "Constraint pack built with "
                f"{len(pack.conflict_notes)} conflict note(s)."
            ),
        )
        return state

    def requires_python_intersection_check(self, state: ResolutionState) -> ResolutionState:
        pack = state.get("constraint_pack")
        if pack is None or not self._research_feature_enabled("python_constraint_intersection"):
            return state
        self._emit_activity(
            state,
            kind="planning_stage_started",
            detail="Checking Python version intersection across candidate packages.",
        )
        self._write_json_artifact(
            state,
            "python-constraints.json",
            {
                "target_python": state.get("target_python", ""),
                "python_intersection": pack.python_intersection,
                "python_intersection_valid": pack.python_intersection_valid,
                "conflict_precheck_failed": pack.conflict_precheck_failed,
            },
        )
        if not pack.python_intersection_valid or pack.conflict_precheck_failed:
            state["dependency_reason"] = "no_compatible_versions"
            state["last_execution"] = ExecutionOutcome(
                success=False,
                category="ConstraintConflictError",
                message="Research constraint precheck blocked all candidate plans.",
                build_succeeded=False,
                run_succeeded=False,
                dependency_retryable=False,
                retry_severity="terminal",
            )
            state["retry_decision"] = classify_retry_decision("ConstraintConflictError")
        self._emit_activity(
            state,
            kind="planning_stage_finished",
            detail=(
                "Python constraint intersection check finished: "
                f"{'valid' if pack.python_intersection_valid and not pack.conflict_precheck_failed else 'blocked'}."
            ),
        )
        return state

    def load_feedback_memory_summary(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research() or not self._research_feature_enabled("repair_feedback_loop"):
            state["feedback_memory_hits"] = 0
            return state
        self._feedback_memory_summary(state)
        return state

    def infer_versions_prompt_b(self, state: ResolutionState) -> ResolutionState:
        if state.get("repaired_dependency_lines"):
            raw_output = "\n".join(state["repaired_dependency_lines"])
            state["model_outputs"]["version"].append({"attempt": state["current_attempt"], "output": raw_output, "source": "repair"})
            state["prompt_history"]["prompt_b"] = raw_output
            try:
                state["selected_dependencies"] = filter_allowed_dependencies(
                    parse_dependency_lines(raw_output),
                    state.get("inferred_packages", []),
                )
            except ValueError:
                state["selected_dependencies"] = []
            state["dependency_reason"] = "repair"
            state["version_selection_source"] = state.get("repair_outcome") or "repair"
            return state

        if self.settings.resolver in {"pyego", "readpye"}:
            try:
                official_plan, source = self._run_official_baseline(state)
            except OfficialBaselineError as exc:
                if self.settings.resolver == "pyego":
                    return self._mark_official_pyego_unavailable(state, str(exc))
                state["resolver_implementation"] = "internal-fallback"
                state["prompt_history"]["prompt_b"] = (
                    f"# official {self.settings.resolver} integration unavailable: {exc}; "
                    "using internal baseline approximation"
                )
                state["model_outputs"]["version"].append(
                    {
                        "attempt": state["current_attempt"],
                        "output": "",
                        "source": f"{self.settings.resolver}_official_unavailable",
                        "error": str(exc),
                    }
                )
            else:
                if official_plan is not None:
                    return self._apply_official_baseline_plan(state, official_plan, source)
                if self.settings.resolver == "pyego":
                    return self._mark_official_pyego_unavailable(
                        state,
                        f"PyEGo entrypoint not found at {self.settings.pyego_root / 'PyEGo.py'}",
                    )
                state["resolver_implementation"] = "internal-fallback"

        if self.settings.resolver == "readpye":
            if state.get("inferred_packages"):
                selected = [ResolvedDependency(name=package, version="") for package in state.get("inferred_packages", [])]
                state["prompt_history"]["prompt_b"] = (
                    "# skipped: ReadPyE baseline uses static import extraction with unpinned package installs"
                )
                state["model_outputs"]["version"].append(
                    {
                        "attempt": state["current_attempt"],
                        "output": "\n".join(dependency.pin() for dependency in selected),
                        "source": "readpye_unpinned",
                    }
                )
                state["selected_dependencies"] = sorted(selected, key=lambda dependency: dependency.name.lower())
                state["dependency_reason"] = "readpye_unpinned"
                state["version_selection_source"] = "readpye_unpinned"
                return state
            state["prompt_history"]["prompt_b"] = "# skipped: ReadPyE baseline inferred no third-party packages"
            state["model_outputs"]["version"].append(
                {"attempt": state["current_attempt"], "output": "", "source": "readpye_no_dependencies"}
            )
            state["selected_dependencies"] = []
            state["dependency_reason"] = "stdlib_only"
            state["version_selection_source"] = "readpye_no_dependencies"
            return state

        if not state.get("version_options"):
            if not self.settings.use_rag and state.get("inferred_packages"):
                selected = [ResolvedDependency(name=package, version="") for package in state.get("inferred_packages", [])]
                state["prompt_history"]["prompt_b"] = "# skipped: RAG disabled; using unpinned package install plan"
                state["model_outputs"]["version"].append(
                    {
                        "attempt": state["current_attempt"],
                        "output": "\n".join(dependency.pin() for dependency in selected),
                        "source": "rag_disabled",
                    }
                )
                state["selected_dependencies"] = sorted(selected, key=lambda dependency: dependency.name.lower())
                state["dependency_reason"] = "rag_disabled"
                state["version_selection_source"] = "rag_disabled"
                return state
            state["prompt_history"]["prompt_b"] = "# skipped: no compatible PyPI version options were inferred"
            state["model_outputs"]["version"].append(
                {"attempt": state["current_attempt"], "output": "", "source": "no_version_options"}
            )
            state["selected_dependencies"] = []
            state["dependency_reason"] = "stdlib_only" if not state.get("inferred_packages") else "no_compatible_versions"
            state["version_selection_source"] = "no_version_options"
            return state

        if all(len(option.versions) == 1 for option in state.get("version_options", [])):
            selected = self._deterministic_dependencies(state["version_options"])
            state["prompt_history"]["prompt_b"] = (
                "# skipped: single compatible version available for each inferred package"
            )
            state["model_outputs"]["version"].append(
                {
                    "attempt": state["current_attempt"],
                    "output": "\n".join(dependency.pin() for dependency in selected),
                    "source": "single_version_fast_path",
                }
            )
            state["selected_dependencies"] = sorted(selected, key=lambda dependency: dependency.name.lower())
            state["dependency_reason"] = "single_version_fast_path"
            state["version_selection_source"] = "single_version_fast_path"
            return state

        if not self._should_use_version_llm(state["version_options"]):
            selected = self._deterministic_dependencies(state["version_options"])
            state["prompt_history"]["prompt_b"] = "# skipped: deterministic version selector"
            state["model_outputs"]["version"].append(
                {
                    "attempt": state["current_attempt"],
                    "output": "\n".join(dependency.pin() for dependency in selected),
                    "source": "deterministic_version_selector",
                }
            )
            state["selected_dependencies"] = sorted(selected, key=lambda dependency: dependency.name.lower())
            state["dependency_reason"] = "deterministic_version_selector"
            state["version_selection_source"] = "deterministic_version_selector"
            return state

        package_versions = self.pypi_store.format_prompt_block(state.get("version_options", []))
        previous_versions = ", ".join(
            sorted(
                {
                    dependency
                    for attempt in state.get("attempt_records", [])
                    for dependency in attempt.dependencies
                    if dependency
                }
            )
        ) or "none"
        format_instructions = "package_name==version, one per line"
        prompt_text = self._format_prompt(
            "version_selection.txt",
            package_versions=package_versions,
            previous_versions=previous_versions,
            format_instructions=format_instructions,
        )
        self._trace_request(state, "version", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "version",
            "version_selection.txt",
            {
                "package_versions": package_versions,
                "previous_versions": previous_versions,
                "format_instructions": format_instructions,
            },
        )
        self._trace_response(state, "version", raw_output)
        state["prompt_history"]["prompt_b"] = prompt_text
        state["model_outputs"]["version"].append({"attempt": state["current_attempt"], "output": raw_output})
        try:
            state["selected_dependencies"] = filter_allowed_dependencies(
                parse_dependency_lines(raw_output),
                state.get("inferred_packages", []),
            )
        except ValueError:
            state["selected_dependencies"] = []
        state["dependency_reason"] = "llm_version_selection"
        state["version_selection_source"] = "llm_version_selection"
        return state

    def generate_candidate_bundles(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        if not self._research_feature_enabled("version_negotiation"):
            state["structured_outputs"]["candidate_bundles"] = []
            return state
        pack = state.get("constraint_pack")
        if pack is None:
            state["structured_outputs"]["candidate_bundles"] = []
            return state
        self._emit_activity(
            state,
            kind="planning_stage_started",
            detail="Generating candidate dependency bundles.",
        )
        bundles = generate_candidate_bundles(
            pack,
            version_cap_per_package=self._constraint_top_k(state.get("target_python", "3.12")),
            preferred_versions_by_package=self._preferred_bundle_versions(state),
        )
        platform_notes_by_package = {
            normalize_package_name(option.package): option.platform_notes
            for option in state.get("version_options", [])
        }
        payload = [
            {
                "rank": index + 1,
                "dependencies": [
                    {
                        "name": dependency.name,
                        "version": dependency.version,
                        "platform_notes": platform_notes_by_package.get(
                            normalize_package_name(dependency.name),
                            {},
                        ).get(dependency.version, []),
                    }
                    for dependency in bundle
                ],
            }
            for index, bundle in enumerate(bundles)
        ]
        if not payload and state.get("version_conflict_notes"):
            state["dependency_reason"] = "no_compatible_versions"
            state["repair_skipped_reason"] = "no_conflict_free_candidate_bundle"
        state["structured_outputs"]["candidate_bundles"] = payload
        self._write_json_artifact(state, "candidate-bundles.json", payload)
        self._emit_activity(
            state,
            kind="planning_stage_finished",
            detail=f"Generated {len(payload)} candidate bundle(s).",
        )
        return state

    def negotiate_version_bundles(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        if not self._research_feature_enabled("version_negotiation"):
            return state
        bundles = state.get("structured_outputs", {}).get("candidate_bundles", [])
        if not bundles:
            state["candidate_plans"] = []
            state["remaining_candidate_plans"] = []
            return state
        self._emit_activity(
            state,
            kind="planning_stage_started",
            detail=f"Negotiating across {len(bundles)} candidate bundle(s).",
        )
        allowed_versions = self._allowed_versions_map(state.get("version_options", []))
        allowed_packages = self._research_allowed_packages(state)
        required_packages = set(allowed_versions)
        source_compatibility_hints = self._source_compatibility_hint_summary(state)
        prompt_text = self._format_prompt(
            "version_negotiation.txt",
            target_python=state.get("target_python", ""),
            candidate_bundles=json.dumps(bundles[:12], indent=2),
            repo_evidence=json.dumps(state.get("repo_evidence", {}), indent=2)[:3000],
            conflict_notes=json.dumps(
                [
                    {
                        "package": note.package,
                        "related_package": note.related_package,
                        "reason": note.reason,
                    }
                    for note in state.get("version_conflict_notes", [])
                ],
                indent=2,
            )[:2500],
            source_compatibility_hints=source_compatibility_hints,
        )
        self._trace_request(state, "version", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "version",
            "version_negotiation.txt",
            {
                "target_python": state.get("target_python", ""),
                "candidate_bundles": json.dumps(bundles[:12], indent=2),
                "repo_evidence": json.dumps(state.get("repo_evidence", {}), indent=2)[:3000],
                "conflict_notes": json.dumps(
                    [
                        {
                            "package": note.package,
                            "related_package": note.related_package,
                            "reason": note.reason,
                        }
                        for note in state.get("version_conflict_notes", [])
                    ],
                    indent=2,
                )[:2500],
                "source_compatibility_hints": source_compatibility_hints,
            },
        )
        self._trace_response(state, "version", raw_output)
        state["structured_outputs"]["version_negotiation_raw"] = raw_output
        try:
            plans = parse_version_negotiation_payload(
                raw_output,
                allowed_packages=allowed_packages,
                allowed_versions=allowed_versions,
                required_packages=required_packages,
            )
        except StructuredOutputError:
            cleaned_output = self._adjudicate_json(
                state,
                "version_negotiation",
                raw_output,
                '{"selected_bundles":[{"rank":1,"reason":"short","dependencies":[{"name":"package","version":"1.0.0"}]}]}',
            )
            try:
                plans = parse_version_negotiation_payload(
                    cleaned_output,
                    allowed_packages=allowed_packages,
                    allowed_versions=allowed_versions,
                    required_packages=required_packages,
                )
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                plans = []
        if plans:
            retained_plans, strategy = self._augment_ranked_candidate_plans(
                plans,
                bundles=bundles,
                max_plan_count=self.settings.candidate_plan_count,
            )
            state["structured_outputs"]["version_negotiation"] = {
                "llm_selected_plans": self._candidate_plan_payload(plans),
                "retained_candidate_plans": self._candidate_plan_payload(retained_plans),
                "candidate_plan_strategy": strategy,
            }
            self._write_json_artifact(state, "version-negotiation.json", state["structured_outputs"]["version_negotiation"])
            state["candidate_plans"] = retained_plans
            state["remaining_candidate_plans"] = list(retained_plans)
            state["version_selection_source"] = "research_version_negotiation"
            state["candidate_plan_strategy"] = strategy
            state["dependency_reason"] = "llm_version_selection"
        else:
            state["structured_outputs"]["version_negotiation"] = {
                "llm_selected_plans": [],
                "retained_candidate_plans": [],
                "candidate_plan_strategy": "",
            }
            self._write_json_artifact(state, "version-negotiation.json", state["structured_outputs"]["version_negotiation"])
        self._emit_activity(
            state,
            kind="planning_stage_finished",
            detail=(
                "Version bundle negotiation finished with "
                f"{len(state.get('candidate_plans', []))} retained plan(s)."
            ),
        )
        return state

    def generate_candidate_plans(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        self._emit_activity(
            state,
            kind="planning_stage_started",
            detail="Generating candidate install plans.",
        )

        default_plans = self._default_research_plans(state)
        if (
            self._research_feature_enabled("version_negotiation")
            and not state.get("structured_outputs", {}).get("candidate_bundles")
            and state.get("dependency_reason") == "no_compatible_versions"
        ):
            default_plans = [
                CandidatePlan(
                    rank=1,
                    reason="no conflict-free candidate bundles remain for the target python",
                    dependencies=[],
                )
            ]
        options = state.get("version_options", [])
        if default_plans and (
            not options
            or all(len(option.versions) == 1 for option in options)
        ):
            if not options:
                state["version_selection_source"] = "research_default_no_versions"
            elif all(len(option.versions) == 1 for option in options):
                state["version_selection_source"] = "research_single_version_fast_path"
            else:
                state["version_selection_source"] = "research_deterministic_default"
            state["candidate_plans"] = default_plans
            state["remaining_candidate_plans"] = list(default_plans)
            state["structured_outputs"]["candidate_plans"] = self._candidate_plan_payload(default_plans)
            state["candidate_plan_strategy"] = "deterministic-fallback"
            self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(default_plans))
            self._emit_activity(
                state,
                kind="planning_stage_finished",
                detail=f"Generated {len(default_plans)} candidate plan(s).",
            )
            return state

        if not options:
            state["candidate_plans"] = default_plans
            state["remaining_candidate_plans"] = list(default_plans)
            state["candidate_plan_strategy"] = "deterministic-fallback" if default_plans else ""
            self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(default_plans))
            self._emit_activity(
                state,
                kind="planning_stage_finished",
                detail=f"Generated {len(default_plans)} candidate plan(s).",
            )
            return state

        allowed_packages = sorted(state.get("inferred_packages", []), key=str.lower)
        allowed_package_set = self._research_allowed_packages(state)
        allowed_versions = self._allowed_versions_map(options)
        required_packages = set(allowed_versions)
        rag_context_summary = summarize_rag_context(state.get("rag_context", {}), limit=6000)
        version_space_summary = self._planning_version_space_summary(state)
        validation_options_summary = self._validation_options_summary(state)
        candidate_bundle_hints = self._candidate_bundle_hint_summary(state)
        source_compatibility_hints = self._source_compatibility_hint_summary(state)
        conflict_notes_summary = self._conflict_note_summary(state)
        allowed_runtime_profiles = {option.get("profile", "") for option in state.get("validation_options", []) if option.get("profile")}
        candidate_template = (
            "candidate_plans_v2.txt"
            if self._research_feature_enabled("transitive_conflicts")
            or self._research_feature_enabled("multipass_inference")
            else "candidate_plans.txt"
        )
        prompt_text = self._format_prompt(
            candidate_template,
            target_python=state.get("target_python", ""),
            allowed_packages="\n".join(allowed_packages),
            version_space=version_space_summary,
            rag_context=rag_context_summary,
            validation_options=validation_options_summary,
            default_validation_profile=state.get("default_validation_profile", ""),
            candidate_bundle_hints=candidate_bundle_hints,
            conflict_notes=conflict_notes_summary,
            source_compatibility_hints=source_compatibility_hints,
            max_plan_count=self.settings.candidate_plan_count,
        )
        self._trace_request(state, "version", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "version",
            candidate_template,
            {
                "target_python": state.get("target_python", ""),
                "allowed_packages": "\n".join(allowed_packages),
                "version_space": version_space_summary,
                "rag_context": rag_context_summary,
                "validation_options": validation_options_summary,
                "default_validation_profile": state.get("default_validation_profile", ""),
                "candidate_bundle_hints": candidate_bundle_hints,
                "conflict_notes": conflict_notes_summary,
                "source_compatibility_hints": source_compatibility_hints,
                "max_plan_count": self.settings.candidate_plan_count,
            },
        )
        self._trace_response(state, "version", raw_output)
        state["prompt_history"]["prompt_b"] = prompt_text
        state["model_outputs"]["version"].append({"attempt": state["current_attempt"], "output": raw_output})
        state["structured_outputs"]["candidate_plans_raw"] = raw_output

        try:
            plans = parse_candidate_plan_payload(
                raw_output,
                allowed_packages=allowed_package_set,
                allowed_versions=allowed_versions,
                required_packages=required_packages,
                allowed_runtime_profiles=allowed_runtime_profiles or None,
            )
        except StructuredOutputError:
            cleaned_output = self._adjudicate_json(
                state,
                "candidate_plans",
                raw_output,
                '{"plans":[{"rank":1,"reason":"short reason","dependencies":[{"name":"package","version":"1.0.0"}]}]}',
            )
            try:
                plans = parse_candidate_plan_payload(
                    cleaned_output,
                    allowed_packages=allowed_package_set,
                    allowed_versions=allowed_versions,
                    required_packages=required_packages,
                    allowed_runtime_profiles=allowed_runtime_profiles or None,
                )
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                plans = default_plans

        if not plans:
            plans = default_plans

        state["candidate_plans"] = plans
        state["remaining_candidate_plans"] = list(plans)
        state["structured_outputs"]["candidate_plans"] = self._candidate_plan_payload(plans)
        state["candidate_plan_strategy"] = "llm-selected"
        self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(plans))
        if not state.get("dependency_reason"):
            state["dependency_reason"] = "llm_version_selection"
        state["version_selection_source"] = "research_candidate_plans"
        self._emit_activity(
            state,
            kind="planning_stage_finished",
            detail=f"Generated {len(plans)} candidate plan(s).",
        )
        return state

    def select_next_candidate_plan(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        remaining = list(state.get("remaining_candidate_plans", []))
        selected_plan = remaining.pop(0) if remaining else None
        state["remaining_candidate_plans"] = remaining
        state["selected_candidate_plan"] = selected_plan
        state["selected_candidate_rank"] = selected_plan.rank if selected_plan else None
        if selected_plan is None:
            state["selected_dependencies"] = []
            state["generated_requirements"] = "# no inferred third-party dependencies\n"
            return state
        self._emit_activity(
            state,
            kind="candidate_plan_selected",
            detail=(
                f"Selected candidate plan rank {selected_plan.rank}: "
                f"{self._summarize_dependency_pins(selected_plan.dependencies)}"
                + (
                    f" with runtime profile {selected_plan.runtime_profile}."
                    if selected_plan.runtime_profile
                    else ""
                )
            ),
        )
        self._apply_validation_profile(state, selected_plan.runtime_profile)
        state["selected_dependencies"] = [
            ResolvedDependency(name=dependency.name, version=dependency.version) for dependency in selected_plan.dependencies
        ]
        self._prepare_selected_dependencies(state)
        state["repair_outcome"] = "candidate_plan"
        self._write_json_artifact(
            state,
            "selected-plan.json",
            {
                "rank": selected_plan.rank,
                "reason": selected_plan.reason,
                "dependencies": [dependency.pin() for dependency in selected_plan.dependencies],
                "runtime_profile": selected_plan.runtime_profile,
            },
        )
        return state

    def runtime_profile_repair_research(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research() or not state.get("pending_runtime_profile_retry"):
            return state
        state["pending_runtime_profile_retry"] = False
        allowed_packages = sorted(state.get("inferred_packages", []), key=str.lower)
        allowed_package_set = self._research_allowed_packages(state)
        allowed_versions = self._allowed_versions_map(state.get("version_options", []))
        required_packages = set(allowed_versions)
        previous_plan = "\n".join(dep.pin() for dep in state.get("selected_dependencies", []))
        attempted_plans = "\n".join(
            ", ".join(attempt.dependencies)
            for attempt in state.get("attempt_records", [])
            if attempt.dependencies
        )
        validation_options_summary = self._validation_options_summary(state)
        source_compatibility_hints = self._source_compatibility_hint_summary(state)
        conflict_notes_summary = self._conflict_note_summary(state)
        allowed_runtime_profiles = {
            option.get("profile", "")
            for option in state.get("validation_options", [])
            if option.get("profile")
        }
        build_log_excerpt = execution_build_excerpt = str(state.get("last_execution").build_log if state.get("last_execution") else "")[:2500]
        run_log_excerpt = execution_run_excerpt = str(state.get("last_execution").run_log if state.get("last_execution") else "")[:2500]
        prompt_text = self._format_prompt(
            "runtime_profile_repair.txt",
            target_python=state.get("target_python", ""),
            current_runtime_profile=state.get("current_runtime_profile", ""),
            allowed_packages="\n".join(allowed_packages),
            version_space=self._planning_version_space_summary(state),
            validation_options=validation_options_summary,
            default_validation_profile=state.get("default_validation_profile", ""),
            source_compatibility_hints=source_compatibility_hints,
            conflict_notes=conflict_notes_summary,
            previous_plan=previous_plan,
            attempted_plans=attempted_plans,
            error_details=state.get("last_error_details", ""),
            build_log_excerpt=build_log_excerpt,
            run_log_excerpt=run_log_excerpt,
        )
        self._trace_request(state, "repair", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "repair",
            "runtime_profile_repair.txt",
            {
                "target_python": state.get("target_python", ""),
                "current_runtime_profile": state.get("current_runtime_profile", ""),
                "allowed_packages": "\n".join(allowed_packages),
                "version_space": self._planning_version_space_summary(state),
                "validation_options": validation_options_summary,
                "default_validation_profile": state.get("default_validation_profile", ""),
                "source_compatibility_hints": source_compatibility_hints,
                "conflict_notes": conflict_notes_summary,
                "previous_plan": previous_plan,
                "attempted_plans": attempted_plans,
                "error_details": state.get("last_error_details", ""),
                "build_log_excerpt": execution_build_excerpt,
                "run_log_excerpt": execution_run_excerpt,
            },
        )
        self._trace_response(state, "repair", raw_output)
        state["prompt_history"]["prompt_c"].append(prompt_text)
        state["model_outputs"]["repair"].append(
            {"attempt": state["current_attempt"], "output": raw_output, "source": "runtime_profile_repair"}
        )
        state.setdefault("structured_outputs", {})["runtime_profile_repair_raw"] = raw_output
        try:
            repair_applicable, plans = parse_repair_plan_payload(
                raw_output,
                allowed_packages=allowed_package_set,
                allowed_versions=allowed_versions,
                required_packages=required_packages,
                allowed_runtime_profiles=allowed_runtime_profiles or None,
                previous_plan=[
                    CandidateDependency(name=dependency.name, version=dependency.version)
                    for dependency in state.get("selected_dependencies", [])
                ],
            )
        except StructuredOutputError:
            cleaned_output = self._adjudicate_json(
                state,
                "runtime_profile_repair",
                raw_output,
                '{"repair_applicable":true,"plans":[{"rank":1,"reason":"switch validation profile","runtime_profile":"import_specs","dependencies":[]}]}',
            )
            try:
                repair_applicable, plans = parse_repair_plan_payload(
                    cleaned_output,
                    allowed_packages=allowed_package_set,
                    allowed_versions=allowed_versions,
                    required_packages=required_packages,
                    allowed_runtime_profiles=allowed_runtime_profiles or None,
                    previous_plan=[
                        CandidateDependency(name=dependency.name, version=dependency.version)
                        for dependency in state.get("selected_dependencies", [])
                    ],
                )
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                repair_applicable, plans = False, []
        if plans:
            plans = self._filter_novel_candidate_plans(state, plans)
        state["structured_outputs"]["runtime_profile_repair"] = {
            "repair_applicable": repair_applicable,
            "plans": self._candidate_plan_payload(plans),
        }
        if not repair_applicable or not plans:
            state["retry_decision"] = state.get("runtime_profile_retry_fallback_decision")
            state["runtime_profile_retry_fallback_decision"] = None
            state["candidate_plans"] = []
            state["remaining_candidate_plans"] = []
            self._emit_activity(
                state,
                kind="runtime_profile_repair_unavailable",
                detail="Runtime-profile repair did not yield a novel plan; continuing with normal retry routing.",
            )
            return state
        state["retry_decision"] = None
        state["runtime_profile_retry_fallback_decision"] = None
        state["candidate_plans"] = plans
        state["remaining_candidate_plans"] = list(plans)
        state["candidate_plan_strategy"] = "llm-runtime-profile-repair"
        self._emit_activity(
            state,
            kind="runtime_profile_repair_ready",
            detail=f"Runtime-profile repair proposed {len(plans)} plan(s); selecting rank {plans[0].rank} next.",
        )
        return state

    def repair_prompt_c_research(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        state["repair_cycle_count"] = state.get("repair_cycle_count", 0) + 1
        state["repair_model_concluded_impossible"] = False
        state["repair_plan_unavailable_reason"] = ""
        self._emit_activity(
            state,
            kind="repair_cycle_started",
            detail=f"Starting repair cycle {state['repair_cycle_count']}.",
        )
        allowed_packages = sorted(state.get("inferred_packages", []), key=str.lower)
        allowed_package_set = self._research_allowed_packages(state)
        allowed_versions = self._allowed_versions_map(state.get("version_options", []))
        required_packages = set(allowed_versions)
        previous_plan = "\n".join(dep.pin() for dep in state.get("selected_dependencies", []))
        attempted_plans = "\n".join(
            ", ".join(attempt.dependencies)
            for attempt in state.get("attempt_records", [])
            if attempt.dependencies
        )
        rag_context_summary = summarize_rag_context(state.get("rag_context", {}), limit=8000)
        version_space_summary = self._planning_version_space_summary(state)
        validation_options_summary = self._validation_options_summary(state)
        source_compatibility_hints = self._source_compatibility_hint_summary(state)
        conflict_notes_summary = self._conflict_note_summary(state)
        repeated_missing_symbol_failures = self._repeated_missing_symbol_summary(state)
        symbol_compatibility_repair_candidates = self._symbol_compatibility_repair_candidates_summary(state)
        allowed_runtime_profiles = {
            option.get("profile", "")
            for option in state.get("validation_options", [])
            if option.get("profile")
        }
        use_symbol_compatibility_repair = (
            repeated_missing_symbol_failures != "[]"
            and self._missing_symbol_signature(state.get("last_error_details", "")) is not None
        )
        if use_symbol_compatibility_repair:
            repair_template = "symbol_compatibility_repair.txt"
        else:
            repair_template = "repair_attempt_v2.txt" if self._research_feature_enabled("repair_memory") else "repair_attempt.txt"
        feedback_summary = (
            summarize_feedback_memory(self.settings.workspace_memory_dir)
            if self._research_feature_enabled("repair_feedback_loop")
            else {"entries": []}
        )
        prompt_text = self._format_prompt(
            repair_template,
            target_python=state.get("target_python", ""),
            allowed_packages="\n".join(allowed_packages),
            version_space=version_space_summary,
            previous_plan=previous_plan,
            attempted_plans=attempted_plans,
            error_details=state.get("last_error_details", ""),
            rag_context=rag_context_summary,
            validation_options=validation_options_summary,
            default_validation_profile=state.get("default_validation_profile", ""),
            source_compatibility_hints=source_compatibility_hints,
            conflict_notes=conflict_notes_summary,
            repeated_missing_symbol_failures=repeated_missing_symbol_failures,
            symbol_compatibility_repair_candidates=symbol_compatibility_repair_candidates,
            max_plan_count=min(2, self.settings.candidate_plan_count),
            repair_memory=json.dumps(asdict(state.get("repair_memory_summary") or RepairMemorySummary()), indent=2)[:2500],
            feedback_summary=json.dumps(feedback_summary, indent=2)[:2500],
        )
        self._trace_request(state, "repair", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "repair",
            repair_template,
            {
                "target_python": state.get("target_python", ""),
                "allowed_packages": "\n".join(allowed_packages),
                "version_space": version_space_summary,
                "previous_plan": previous_plan,
                "attempted_plans": attempted_plans,
                "error_details": state.get("last_error_details", ""),
                "rag_context": rag_context_summary,
                "validation_options": validation_options_summary,
                "default_validation_profile": state.get("default_validation_profile", ""),
                "source_compatibility_hints": source_compatibility_hints,
                "conflict_notes": conflict_notes_summary,
                "repeated_missing_symbol_failures": repeated_missing_symbol_failures,
                "symbol_compatibility_repair_candidates": symbol_compatibility_repair_candidates,
                "max_plan_count": min(2, self.settings.candidate_plan_count),
                "repair_memory": json.dumps(asdict(state.get("repair_memory_summary") or RepairMemorySummary()), indent=2)[:2500],
                "feedback_summary": json.dumps(feedback_summary, indent=2)[:2500],
            },
        )
        self._trace_response(state, "repair", raw_output)
        state["prompt_history"]["prompt_c"].append(prompt_text)
        state["model_outputs"]["repair"].append({"attempt": state["current_attempt"], "output": raw_output})
        state["structured_outputs"]["repair_raw"] = raw_output
        parser_failed = False
        try:
            repair_applicable, plans = parse_repair_plan_payload(
                raw_output,
                allowed_packages=allowed_package_set,
                allowed_versions=allowed_versions,
                required_packages=required_packages,
                allowed_runtime_profiles=allowed_runtime_profiles or None,
                previous_plan=[
                    CandidateDependency(name=dependency.name, version=dependency.version)
                    for dependency in state.get("selected_dependencies", [])
                ],
            )
        except StructuredOutputError:
            cleaned_output = self._adjudicate_json(
                state,
                "repair",
                raw_output,
                '{"repair_applicable":true,"plans":[{"rank":1,"reason":"short reason","dependencies":[{"name":"package","version":"1.0.0"}]}]}',
            )
            try:
                repair_applicable, plans = parse_repair_plan_payload(
                    cleaned_output,
                    allowed_packages=allowed_package_set,
                    allowed_versions=allowed_versions,
                    required_packages=required_packages,
                    allowed_runtime_profiles=allowed_runtime_profiles or None,
                    previous_plan=[
                        CandidateDependency(name=dependency.name, version=dependency.version)
                        for dependency in state.get("selected_dependencies", [])
                    ],
                )
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                parser_failed = True
                repair_applicable, plans = False, []

        if plans:
            plans = self._filter_novel_candidate_plans(state, plans)

        state["structured_outputs"]["repair"] = {
            "repair_applicable": repair_applicable,
            "plans": self._candidate_plan_payload(plans),
        }
        if not repair_applicable or not plans:
            state["repair_outcome"] = "repair_not_applicable"
            state["candidate_plans"] = []
            state["remaining_candidate_plans"] = []
            state["candidate_plan_strategy"] = ""
            if parser_failed:
                state["repair_model_concluded_impossible"] = False
                state["repair_plan_unavailable_reason"] = "repair_parse_failed"
                state.pop("stop_reason", None)
                detail = "Repair output was unusable; retrying repair if budget remains."
            elif not repair_applicable:
                state["repair_model_concluded_impossible"] = True
                state["repair_plan_unavailable_reason"] = "model_not_applicable"
                state["stop_reason"] = "ModelConcludedNoFix"
                detail = "Repair model concluded no dependency-only fix is applicable."
            else:
                state["repair_model_concluded_impossible"] = False
                state["repair_plan_unavailable_reason"] = "no_novel_plans"
                state.pop("stop_reason", None)
                detail = "Repair only proposed already-attempted plans; retrying repair if budget remains."
            if self._should_activate_deferred_python_fallback_after_repair(state):
                self._activate_deferred_python_fallback(state)
                state["repair_model_concluded_impossible"] = False
                state["repair_plan_unavailable_reason"] = "deferred_python_fallback"
                state.pop("stop_reason", None)
                detail = "Repair produced no viable Python 3 plan; switching to deferred Python fallback."
            self._emit_activity(
                state,
                kind="repair_plan_unavailable",
                detail=detail,
            )
            return state
        state["repair_outcome"] = "llm_repair"
        state["repair_model_concluded_impossible"] = False
        state["repair_plan_unavailable_reason"] = ""
        if state.get("stop_reason") == "ModelConcludedNoFix":
            state.pop("stop_reason", None)
        state["candidate_plans"] = plans
        state["remaining_candidate_plans"] = list(plans)
        state["candidate_plan_strategy"] = "llm-selected"
        self._emit_activity(
            state,
            kind="repair_plan_ready",
            detail=f"Repair proposed {len(plans)} fallback plan(s); selecting rank {plans[0].rank} next.",
        )
        self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(plans))
        return state

    def normalize_dependency_plan(self, state: ResolutionState) -> ResolutionState:
        if state.get("stop_reason") == "OfficialPyEGoUnavailable":
            state["selected_dependencies"] = []
            state["generated_requirements"] = "# official PyEGo unavailable\n"
            return state
        state.pop("stop_reason", None)
        if state.get("selected_dependencies"):
            previous_dependencies = (
                state.get("attempt_records", [])[-1].dependencies if state.get("attempt_records") else []
            )
            dependencies = sorted(state["selected_dependencies"], key=lambda dep: dep.name.lower())
            state["selected_dependencies"] = dependencies
            state["generated_requirements"] = "\n".join(dep.pin() for dep in dependencies) + (
                "\n" if dependencies else ""
            )
            if state.get("repaired_dependency_lines"):
                current_dependencies = [dependency.pin() for dependency in dependencies]
                if current_dependencies == previous_dependencies:
                    state["repair_stall_count"] = state.get("repair_stall_count", 0) + 1
                else:
                    state["repair_stall_count"] = 0
                if state["repair_stall_count"] >= 2:
                    state["stop_reason"] = "RepairOutputStalled"
            else:
                state["repair_stall_count"] = 0
            return state

        if not state.get("version_options") and not state.get("repaired_dependency_lines"):
            state["selected_dependencies"] = []
            state["generated_requirements"] = "# no inferred third-party dependencies\n"
            return state

        previous_dependencies = state.get("attempt_records", [])[-1].dependencies if state.get("attempt_records") else []

        last_output = state["model_outputs"]["version"][-1]["output"]
        cleanup_prompt = (
            "Rewrite the following text as newline-delimited package==version entries only.\n\n"
            f"{last_output}"
        )
        if not self.preset_config.allow_adjudication:
            state["selected_dependencies"] = []
            state["generated_requirements"] = "# no inferred third-party dependencies\n"
            return state
        self._trace_request(state, "adjudicate", cleanup_prompt)
        cleaned_output = self.prompt_runner.invoke_text("adjudicate", cleanup_prompt)
        self._trace_response(state, "adjudicate", cleaned_output)
        state["model_outputs"]["adjudicate"].append(cleaned_output)
        try:
            dependencies = filter_allowed_dependencies(
                parse_dependency_lines(cleaned_output),
                state.get("inferred_packages", []),
            )
        except ValueError:
            state["selected_dependencies"] = []
            state["generated_requirements"] = "# no inferred third-party dependencies\n"
            if state.get("repaired_dependency_lines"):
                state["repair_stall_count"] = state.get("repair_stall_count", 0) + 1
                if state["repair_stall_count"] >= 2:
                    state["stop_reason"] = "RepairOutputStalled"
            return state
        state["selected_dependencies"] = dependencies
        state["generated_requirements"] = (
            "\n".join(dep.pin() for dep in dependencies) + "\n"
            if dependencies
            else "# no inferred third-party dependencies\n"
        )
        if state.get("repaired_dependency_lines"):
            if not dependencies:
                state["repair_stall_count"] = state.get("repair_stall_count", 0) + 1
                if state["repair_stall_count"] >= 2:
                    state["stop_reason"] = "RepairOutputStalled"
                return state
            current_dependencies = [dependency.pin() for dependency in dependencies]
            if current_dependencies == previous_dependencies:
                state["repair_stall_count"] = state.get("repair_stall_count", 0) + 1
            else:
                state["repair_stall_count"] = 0
            if state["repair_stall_count"] >= 2:
                state["stop_reason"] = "RepairOutputStalled"
        else:
            state["repair_stall_count"] = 0
        return state

    def materialize_execution_context(self, state: ResolutionState) -> ResolutionState:
        state["current_attempt"] += 1
        state["pending_native_retry"] = False
        state["pending_runtime_profile_retry"] = False
        state["pending_python_fallback"] = False
        artifact_dir = Path(state["artifact_dir"])
        attempt_dir = artifact_dir / f"attempt_{state['current_attempt']:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        state["current_attempt_dir"] = str(attempt_dir)

        image_tag = f"pllm-{state['run_id']}-{state['case_id']}-{state['current_attempt']}"
        if state["mode"] == "gistable":
            self._maybe_prefer_benchmark_target_python(state)
            context = self.docker_executor.prepare_benchmark_context(
                state["benchmark_case"],
                state["selected_dependencies"],
                attempt_dir,
                image_tag,
                state["target_python"],
                state.get("current_validation_command") or None,
                extra_system_packages=state.get("system_dependencies", []),
                extra_bootstrap_pins=state.get("bootstrap_dependencies", []),
                case_id=state["case_id"],
                attempt_number=state["current_attempt"],
            )
            shutil.copy2(state["benchmark_case"].snippet_path, artifact_dir / "source.py")
        else:
            context = self.docker_executor.prepare_project_context(
                state["project_target"],
                state["selected_dependencies"],
                attempt_dir,
                image_tag,
                extra_system_packages=state.get("system_dependencies", []),
                extra_bootstrap_pins=state.get("bootstrap_dependencies", []),
                case_id=state["case_id"],
                attempt_number=state["current_attempt"],
            )
            first_source = next(iter(state["source_files"].values()), "")
            (artifact_dir / "source.py").write_text(first_source, encoding="utf-8")

        state["prepared_execution_context"] = context
        dockerfile_text = context.dockerfile_path.read_text(encoding="utf-8")
        state["generated_dockerfile"] = dockerfile_text
        (artifact_dir / "requirements.generated.txt").write_text(state["generated_requirements"], encoding="utf-8")
        (artifact_dir / "Dockerfile.generated").write_text(dockerfile_text, encoding="utf-8")
        (artifact_dir / "prompt_a.txt").write_text("\n\n".join(state["prompt_history"]["prompt_a"]), encoding="utf-8")
        (artifact_dir / "prompt_b.txt").write_text(state["prompt_history"]["prompt_b"], encoding="utf-8")
        for index, prompt_c in enumerate(state["prompt_history"]["prompt_c"], start=1):
            (artifact_dir / f"prompt_c_attempt_{index}.txt").write_text(prompt_c, encoding="utf-8")
        if self._is_research():
            self._write_json_artifact(state, "structured-outputs.json", state.get("structured_outputs", {}))
        self._emit_activity(
            state,
            kind="docker_context_prepared",
            detail=f"Prepared Docker context for attempt {state['current_attempt']}.",
        )
        return state

    def execute_candidate(self, state: ResolutionState) -> ResolutionState:
        attempt_dir = Path(state["current_attempt_dir"])
        context = state["prepared_execution_context"]
        attempt_started_at = datetime.now(timezone.utc).isoformat()
        self._emit_activity(
            state,
            kind="candidate_execution_started",
            detail=f"Starting execution attempt {state['current_attempt']}.",
        )

        result = self.docker_executor.execute(context)
        attempt_finished_at = datetime.now(timezone.utc).isoformat()
        (attempt_dir / "build.log").write_text(result.build_log, encoding="utf-8")
        (attempt_dir / "run.log").write_text(result.run_log, encoding="utf-8")

        success = result.build_succeeded and result.run_succeeded
        state["last_execution"] = ExecutionOutcome(
            success=success,
            category="Success" if success else "ExecutionFailed",
            message="Execution succeeded." if success else "Execution failed.",
            build_succeeded=result.build_succeeded,
            run_succeeded=result.run_succeeded,
            exit_code=result.exit_code,
            build_log=result.build_log,
            run_log=result.run_log,
            image_tag=result.image_tag,
            dependency_retryable=False,
        )
        state["attempt_records"].append(
            AttemptRecord(
                attempt_number=state["current_attempt"],
                dependencies=[dependency.pin() for dependency in state["selected_dependencies"]],
                image_tag=result.image_tag,
                build_succeeded=result.build_succeeded,
                run_succeeded=result.run_succeeded,
                exit_code=result.exit_code,
                error_category="Success" if success else "ExecutionFailed",
                error_details="",
                validation_command=state.get("current_validation_command") or None,
                wall_clock_seconds=result.wall_clock_seconds,
                artifact_dir=str(attempt_dir),
                runtime_profile=str(state.get("current_runtime_profile", "") or ""),
                started_at=attempt_started_at,
                finished_at=attempt_finished_at,
                build_wall_clock_seconds=result.build_wall_clock_seconds,
                run_wall_clock_seconds=result.run_wall_clock_seconds,
                build_skipped=result.build_skipped,
                image_cache_hit=result.image_cache_hit,
                environment_cache_key=getattr(context, "environment_cache_key", ""),
            )
        )
        return state

    def classify_outcome(self, state: ResolutionState) -> ResolutionState:
        execution = state["last_execution"]
        classified = classify_error(
            execution.build_log,
            execution.run_log,
            execution.exit_code,
            build_succeeded=execution.build_succeeded,
            run_succeeded=execution.run_succeeded,
        )
        state["pending_native_retry"] = False
        state["pending_runtime_profile_retry"] = False
        state["pending_python_fallback"] = False
        state["runtime_profile_retry_fallback_decision"] = None
        state["system_dependency_hints"] = list(state.get("system_dependency_hints", []))
        state["bootstrap_dependencies"] = list(state.get("bootstrap_dependencies", []))
        state["bootstrap_packages_attempted"] = list(state.get("bootstrap_packages_attempted", []))
        if not self._uses_full_apd():
            classified.dependency_retryable = False
        error_focus = self._failure_focus(classified.message)
        if error_focus:
            classified.message = f"{classified.message}\n\nFailure focus: {error_focus}"
        hints, suggested_packages, suggested_bootstrap_pins = self._native_retry_hints(classified.message)
        if self._research_feature_enabled("smart_repair_routing"):
            system_packages_injected = bool(
                getattr(state.get("prepared_execution_context"), "system_packages", [])
            ) if state.get("prepared_execution_context") is not None else False
            retry_decision = classify_retry_decision(
                classified.category,
                system_packages_injected=system_packages_injected,
                has_system_package_hints=bool(suggested_packages),
                native_retry_used=sum(
                    1 for attempt in state.get("attempt_records", []) if attempt.error_category == "NativeBuildError"
                ),
            )
            classified.dependency_retryable = retry_decision.repair_allowed or retry_decision.candidate_fallback_allowed
            classified.retry_severity = retry_decision.severity
        else:
            retry_decision = classify_retry_decision(classified.category) if self._uses_full_apd() else None
            if retry_decision is not None:
                classified.dependency_retryable = retry_decision.repair_allowed or retry_decision.candidate_fallback_allowed
                classified.retry_severity = retry_decision.severity
            elif classified.dependency_retryable:
                classified.retry_severity = "repair_retryable"
        if (
            self._uses_full_apd()
            and execution.build_succeeded
            and not execution.run_succeeded
            and state.get("deferred_target_python")
            and not state.get("python_fallback_used")
            and state["current_attempt"] < self.settings.max_attempts
            and self._requires_python2_runtime_retry(classified.category, classified.message)
        ):
            self._activate_deferred_python_fallback(state)
            classified.dependency_retryable = True
            classified.retry_severity = "limited_retryable"
            if retry_decision is not None:
                retry_decision.repair_allowed = False
                retry_decision.candidate_fallback_allowed = False
                retry_decision.reason = "deferred-python-fallback"
                retry_decision.native_retry_budget = 0
                retry_decision.repair_retry_budget = 0
        elif self._should_reserve_last_attempt_for_deferred_python_fallback(state, execution):
            self._activate_deferred_python_fallback(state)
            classified.dependency_retryable = True
            classified.retry_severity = "limited_retryable"
            if retry_decision is not None:
                retry_decision.repair_allowed = False
                retry_decision.candidate_fallback_allowed = False
                retry_decision.reason = "reserved-deferred-python-fallback"
                retry_decision.native_retry_budget = 0
                retry_decision.repair_retry_budget = 0
        elif self._should_prompt_runtime_profile_repair(state, execution, classified):
            state["pending_runtime_profile_retry"] = True
            classified.dependency_retryable = True
            classified.retry_severity = "limited_retryable"
            self._emit_activity(
                state,
                kind="runtime_profile_repair_planned",
                detail=(
                    "Build succeeded but runtime validation failed with a hardware-sensitive import signature; "
                    "asking the repair model whether to switch validation profile before changing dependencies."
                ),
            )
            if retry_decision is not None:
                state["runtime_profile_retry_fallback_decision"] = RetryDecision(**asdict(retry_decision))
                retry_decision.repair_allowed = False
                retry_decision.candidate_fallback_allowed = False
                retry_decision.reason = "runtime-profile-repair-first"
                retry_decision.native_retry_budget = 0
                retry_decision.repair_retry_budget = 0
        else:
            attempted_packages = list(state.get("system_packages_attempted", []))
            attempted_bootstrap_pins = list(state.get("bootstrap_packages_attempted", []))
            new_system_packages = [
                package for package in suggested_packages if package not in attempted_packages
            ]
            new_bootstrap_pins = [
                pin for pin in suggested_bootstrap_pins if pin not in attempted_bootstrap_pins
            ]
            if (
                self._uses_full_apd()
                and state.get("selected_dependencies")
                and (
                    classified.category in {"NativeBuildError", "SystemDependencyError"}
                    or (classified.category == "PackageCompatibilityError" and new_bootstrap_pins)
                )
                and (new_system_packages or new_bootstrap_pins)
                and state["current_attempt"] < self.settings.max_attempts
            ):
                merged_system_packages = list(state.get("system_dependencies", []))
                for package in new_system_packages:
                    if package not in merged_system_packages:
                        merged_system_packages.append(package)
                    attempted_packages.append(package)
                merged_bootstrap_pins = list(state.get("bootstrap_dependencies", []))
                for pin in new_bootstrap_pins:
                    if pin not in merged_bootstrap_pins:
                        merged_bootstrap_pins.append(pin)
                    attempted_bootstrap_pins.append(pin)
                state["system_dependencies"] = merged_system_packages
                state["bootstrap_dependencies"] = merged_bootstrap_pins
                state["system_packages_attempted"] = attempted_packages
                state["bootstrap_packages_attempted"] = attempted_bootstrap_pins
                state["system_dependency_hints"] = sorted(set(state.get("system_dependency_hints", [])) | set(hints))
                state["pending_native_retry"] = True
                classified.dependency_retryable = True
                if retry_decision is not None:
                    retry_decision.repair_allowed = False
                    retry_decision.candidate_fallback_allowed = False
                    retry_decision.reason = "deterministic-native-system-retry"
                    retry_decision.native_retry_budget = max(0, retry_decision.native_retry_budget - 1)
                classified.retry_severity = "limited_retryable"
            elif (
                retry_decision is not None
                and classified.category in {"NativeBuildError", "SystemDependencyError"}
                and self._uses_full_apd()
                and state.get("selected_dependencies")
            ):
                retry_decision.candidate_fallback_allowed = (
                    retry_decision.candidate_fallback_allowed or bool(state.get("remaining_candidate_plans"))
                )
                retry_decision.repair_allowed = True
                retry_decision.repair_retry_budget = max(retry_decision.repair_retry_budget, 1)
                if retry_decision.repair_allowed or retry_decision.candidate_fallback_allowed:
                    retry_decision.reason = "build-log-guided-followup"
                classified.dependency_retryable = (
                    retry_decision.repair_allowed or retry_decision.candidate_fallback_allowed
                )
                classified.retry_severity = retry_decision.severity
        classified.image_tag = execution.image_tag
        state["last_execution"] = classified
        state["retry_decision"] = retry_decision
        state["classifier_origin"] = classified.classifier_origin
        state["last_error_category"] = classified.category
        state["last_error_details"] = classified.message
        failure_excerpt = self._failure_focus(classified.message) or self._activity_excerpt(classified.message)
        self._emit_activity(
            state,
            kind="attempt_classified",
            detail=(
                f"Attempt {state['current_attempt']} classified as {classified.category}"
                + (f": {failure_excerpt}" if failure_excerpt else ".")
            ),
        )
        latest_attempt = state["attempt_records"][-1]
        latest_attempt.error_category = classified.category
        latest_attempt.error_details = classified.message
        self._record_attempt_failure_analysis(state, classified)
        previous_dependencies = state.get("attempt_records", [])[-2].dependencies if len(state.get("attempt_records", [])) > 1 else []
        current_dependencies = latest_attempt.dependencies
        strategy_record = RepairStrategyRecord(
            strategy_type=self._strategy_type_from(previous_dependencies, current_dependencies),
            delta_from_previous=sorted(set(current_dependencies) ^ set(previous_dependencies)),
            failure_category=classified.category,
            failure_signature=classified.message[:240],
            result="success" if classified.success else "failure",
        )
        state.setdefault("strategy_history", []).append(strategy_record)
        if self._is_research():
            self._write_json_artifact(
                state,
                "error-routing.json",
                {
                    "category": retry_decision.category if retry_decision is not None else classified.category,
                    "severity": retry_decision.severity if retry_decision is not None else classified.retry_severity,
                    "repair_allowed": retry_decision.repair_allowed if retry_decision is not None else classified.dependency_retryable,
                    "candidate_fallback_allowed": retry_decision.candidate_fallback_allowed if retry_decision is not None else bool(state.get("remaining_candidate_plans")),
                    "repair_retry_budget": retry_decision.repair_retry_budget if retry_decision is not None else self.settings.repair_cycle_limit,
                    "native_retry_budget": retry_decision.native_retry_budget if retry_decision is not None else 0,
                    "reason": retry_decision.reason if retry_decision is not None else "legacy-retry-routing",
                    "pending_native_retry": state.get("pending_native_retry", False),
                    "pending_runtime_profile_retry": state.get("pending_runtime_profile_retry", False),
                    "pending_python_fallback": state.get("pending_python_fallback", False),
                    "classifier_origin": classified.classifier_origin,
                    "system_dependency_hints": list(state.get("system_dependency_hints", [])),
                    "system_packages_attempted": list(state.get("system_packages_attempted", [])),
                    "bootstrap_dependencies": list(state.get("bootstrap_dependencies", [])),
                    "bootstrap_packages_attempted": list(state.get("bootstrap_packages_attempted", [])),
                },
            )
            self._write_json_artifact(
                state,
                "strategy-history.json",
                [asdict(record) for record in state.get("strategy_history", [])],
            )
            if self._research_feature_enabled("repair_feedback_loop"):
                append_feedback_event(
                    self.settings.workspace_memory_dir,
                    {
                        "run_id": state.get("run_id", ""),
                        "case_id": state.get("case_id", ""),
                        "target_python": state.get("target_python", ""),
                        "resolver": state.get("resolver", self.settings.resolver),
                        "preset": state.get("preset", self.settings.preset),
                        "enabled_features": list(state.get("research_features", ())),
                        "error_category": classified.category,
                        "strategy_type": strategy_record.strategy_type,
                        "dependency_fingerprint": "|".join(current_dependencies),
                        "package_family_fingerprint": "|".join(sorted(dep.split("==", 1)[0] for dep in current_dependencies)),
                        "selected_candidate_rank": state.get("selected_candidate_rank"),
                        "success": classified.success,
                        "wall_clock_seconds": latest_attempt.wall_clock_seconds,
                    },
                )
        next_step = (
            route_after_research_classification(state, self.settings)
            if self._is_research()
            else route_after_classification(state, self.settings.max_attempts)
        )
        self._emit_post_classification_activity(
            state,
            category=classified.category,
            next_step=next_step,
            suggested_packages=suggested_packages + suggested_bootstrap_pins,
        )
        return state

    def build_repair_memory_summary(self, state: ResolutionState) -> ResolutionState:
        if not self._is_research():
            return state
        if not self._research_feature_enabled("repair_memory"):
            state["repair_memory_summary"] = RepairMemorySummary()
            return state
        strategies = state.get("strategy_history", [])
        recent = [record.strategy_type for record in strategies[-5:]]
        blocked = sorted({strategy for strategy in recent if recent.count(strategy) > 1})
        orthogonal_hints = []
        if "downgrade" in blocked:
            orthogonal_hints.append("try_alias_or_drop_package")
        if "fallback_candidate" in blocked:
            orthogonal_hints.append("try_transitive_conflict_avoidance")
        summary = RepairMemorySummary(
            recent_strategies=recent,
            blocked_strategy_families=blocked,
            orthogonal_hints=orthogonal_hints,
        )
        state["repair_memory_summary"] = summary
        self._write_json_artifact(state, "repair-memory-summary.json", asdict(summary))
        return state

    def repair_prompt_c(self, state: ResolutionState) -> ResolutionState:
        alias_retry = self._maybe_alias_retry(state)
        if alias_retry:
            state["repaired_dependency_lines"] = alias_retry
            self._emit_activity(
                state,
                kind="repair_plan_ready",
                detail="Alias retry produced an updated dependency plan.",
            )
            return state
        self._emit_activity(
            state,
            kind="repair_cycle_started",
            detail=f"Starting repair cycle for attempt {state['current_attempt']}.",
        )
        previous_packages = "\n".join(dep.pin() for dep in state["selected_dependencies"])
        error_details = state["last_error_details"]
        prompt_text = self._format_prompt(
            "repair_attempt.txt",
            allowed_packages=self._repair_allowed_packages(state),
            previous_packages=previous_packages,
            error_details=error_details,
        )
        self._trace_request(state, "repair", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "repair",
            "repair_attempt.txt",
            {
                "allowed_packages": self._repair_allowed_packages(state),
                "previous_packages": previous_packages,
                "error_details": error_details,
            },
        )
        self._trace_response(state, "repair", raw_output)
        state["prompt_history"]["prompt_c"].append(prompt_text)
        state["model_outputs"]["repair"].append({"attempt": state["current_attempt"], "output": raw_output})
        repaired_lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        try:
            repaired_dependencies = filter_allowed_dependencies(
                parse_dependency_lines("\n".join(repaired_lines)),
                state.get("inferred_packages", []),
            )
            state["repaired_dependency_lines"] = [dependency.pin() for dependency in repaired_dependencies]
            state["repair_outcome"] = "llm_repair"
            self._emit_activity(
                state,
                kind="repair_plan_ready",
                detail=(
                    "Repair proposed dependencies: "
                    f"{', '.join(state['repaired_dependency_lines'][:3])}"
                    + (f", +{len(state['repaired_dependency_lines']) - 3} more" if len(state["repaired_dependency_lines"]) > 3 else "")
                ),
            )
        except ValueError:
            state["repaired_dependency_lines"] = repaired_lines
            state["repair_outcome"] = "repair_not_applicable"
            self._emit_activity(
                state,
                kind="repair_plan_unavailable",
                detail="Repair response was not a valid dependency plan.",
            )
        return state

    def _terminal_outcome_without_execution(self, state: ResolutionState) -> ExecutionOutcome:
        unsupported_imports = list(state.get("unsupported_imports", []))
        ambiguous_imports = list(state.get("ambiguous_imports", []))
        bad_candidates = list(state.get("bad_initial_candidates", []))
        if unsupported_imports:
            rendered = ", ".join(sorted(unsupported_imports))
            return ExecutionOutcome(
                success=False,
                category="UnsupportedImportError",
                message=f"Unsupported external-runtime imports: {rendered}",
                build_succeeded=False,
                run_succeeded=False,
                dependency_retryable=False,
                retry_severity="terminal",
            )
        if ambiguous_imports:
            rendered = ", ".join(sorted(ambiguous_imports))
            return ExecutionOutcome(
                success=False,
                category="AmbiguousImportError",
                message=f"Ambiguous imports lacked safe PyPI evidence: {rendered}",
                build_succeeded=False,
                run_succeeded=False,
                dependency_retryable=False,
                retry_severity="terminal",
            )
        if state.get("dependency_reason") == "no_compatible_versions":
            return ExecutionOutcome(
                success=False,
                category="ConstraintConflictError",
                message="No compatible dependency versions remained after guarded selection.",
                build_succeeded=False,
                run_succeeded=False,
                dependency_retryable=False,
                retry_severity="terminal",
            )
        if bad_candidates:
            details = "; ".join(
                f"{entry.get('package', '')}:{entry.get('reason', '')}"
                for entry in bad_candidates[:5]
                if isinstance(entry, dict)
            )
            return ExecutionOutcome(
                success=False,
                category="CandidateResolutionError",
                message=details or "No safe supported dependency candidates remained.",
                build_succeeded=False,
                run_succeeded=False,
                dependency_retryable=False,
                retry_severity="terminal",
            )
        return ExecutionOutcome(
            success=False,
            category="ResolutionError",
            message="Workflow finalized before candidate execution.",
            build_succeeded=False,
            run_succeeded=False,
            dependency_retryable=False,
            retry_severity="terminal",
        )

    def finalize_result(self, state: ResolutionState) -> ResolutionState:
        artifact_dir = Path(state["artifact_dir"])
        execution = state.get("last_execution")
        if execution is None:
            execution = self._terminal_outcome_without_execution(state)
            state["last_execution"] = execution
        case_finished_at = datetime.now(timezone.utc).isoformat()
        state["case_finished_at"] = case_finished_at
        if not state.get("repair_skipped_reason") and not state.get("selected_dependencies"):
            if state.get("unsupported_imports"):
                state["repair_skipped_reason"] = "unsupported_imports_only"
            elif state.get("ambiguous_imports"):
                state["repair_skipped_reason"] = "ambiguous_imports_unresolved"
            elif state.get("bad_initial_candidates"):
                state["repair_skipped_reason"] = "no_supported_candidates"
        if state["attempt_records"]:
            latest_attempt_dir = Path(state["attempt_records"][-1].artifact_dir)
            for name in ("build.log", "run.log"):
                source = latest_attempt_dir / name
                if source.exists():
                    shutil.copy2(source, artifact_dir / name)

        model_outputs = {
            key: [entry if isinstance(entry, str) else entry for entry in values]
            for key, values in state["model_outputs"].items()
        }
        (artifact_dir / "model_outputs.json").write_text(
            json.dumps(model_outputs, indent=2),
            encoding="utf-8",
        )

        total_wall_clock = sum(attempt.wall_clock_seconds for attempt in state.get("attempt_records", []))
        total_build_wall_clock = sum(attempt.build_wall_clock_seconds for attempt in state.get("attempt_records", []))
        total_run_wall_clock = sum(attempt.run_wall_clock_seconds for attempt in state.get("attempt_records", []))
        image_cache_hits = sum(1 for attempt in state.get("attempt_records", []) if attempt.image_cache_hit)
        build_skips = sum(1 for attempt in state.get("attempt_records", []) if attempt.build_skipped)
        llm_wall_clock_seconds = 0.0
        stats_snapshot = getattr(self.prompt_runner, "stats_snapshot", None)
        if callable(stats_snapshot):
            snapshot = stats_snapshot()
            llm_wall_clock_seconds = max(0.0, float(snapshot.total_duration_ns) / 1_000_000_000)
        runtime_config = self.settings.effective_runtime_config()
        result = {
            "run_id": state["run_id"],
            "case_id": state["case_id"],
            "mode": state["mode"],
            "case_source": state.get("benchmark_case").case_source if state.get("benchmark_case") else "",
            "resolver": state.get("resolver", self.settings.resolver),
            "resolver_implementation": state.get("resolver_implementation", "internal"),
            "preset": state.get("preset", self.settings.preset),
            "prompt_profile": state.get("prompt_profile", self.settings.prompt_profile),
            "research_bundle": state.get("research_bundle", "baseline"),
            "research_features": list(state.get("research_features", ())),
            "model_profile": self.settings.model_profile,
            "effective_model_profile": runtime_config.get("effective_model_profile", self.settings.model_profile),
            "use_moe": self.settings.use_moe,
            "use_rag": self.settings.use_rag,
            "use_langchain": self.settings.use_langchain,
            "rag_mode": self.settings.rag_mode,
            "effective_rag_mode": runtime_config.get("effective_rag_mode", self.settings.rag_mode),
            "structured_prompting": self.settings.structured_prompting,
            "effective_structured_prompting": runtime_config.get(
                "effective_structured_prompting",
                self.settings.structured_prompting,
            ),
            "effective_repair_cycle_limit": runtime_config.get(
                "effective_repair_cycle_limit",
                self.settings.repair_cycle_limit,
            ),
            "effective_candidate_fallback_before_repair": runtime_config.get(
                "effective_candidate_fallback_before_repair",
                self.settings.allow_candidate_fallback_before_repair,
            ),
            "extraction_model": self.settings.stage_model("extract"),
            "runner_model": self.settings.reasoning_model,
            "version_model": self.settings.stage_model("version"),
            "repair_model": self.settings.stage_model("repair"),
            "adjudication_model": self.settings.stage_model("adjudicate"),
            "success": execution.success,
            "attempts": state["current_attempt"],
            "final_error_category": execution.category,
            "initial_eval": state.get("benchmark_case").initial_eval if state.get("benchmark_case") else "",
            "benchmark_target_python": state.get("benchmark_target_python", ""),
            "inferred_target_python": state.get("inferred_target_python", ""),
            "python_version_source": state.get("python_version_source", ""),
            "deferred_target_python": state.get("deferred_target_python", ""),
            "python_fallback_used": bool(state.get("python_fallback_used", False)),
            "target_python": state.get("target_python", ""),
            "runtime_profile": state.get("current_runtime_profile", ""),
            "validation_command": state.get("current_validation_command", ""),
            "dependencies": [dependency.pin() for dependency in state.get("selected_dependencies", [])],
            "system_dependencies": list(state.get("system_dependencies", [])),
            "bootstrap_dependencies": list(state.get("bootstrap_dependencies", [])),
            "system_dependency_hints": list(state.get("system_dependency_hints", [])),
            "system_packages_attempted": list(state.get("system_packages_attempted", [])),
            "bootstrap_packages_attempted": list(state.get("bootstrap_packages_attempted", [])),
            "dependency_reason": state.get("dependency_reason", ""),
            "candidate_provenance": state.get("candidate_provenance", {}),
            "unsupported_imports": list(state.get("unsupported_imports", [])),
            "ambiguous_imports": list(state.get("ambiguous_imports", [])),
            "bad_initial_candidates": list(state.get("bad_initial_candidates", [])),
            "platform_compatibility_notes": list(state.get("platform_compatibility_notes", [])),
            "repair_outcome": state.get("repair_outcome", ""),
            "repair_skipped_reason": state.get("repair_skipped_reason", ""),
            "version_selection_source": state.get("version_selection_source", ""),
            "candidate_plan_strategy": state.get("candidate_plan_strategy", ""),
            "compatibility_policy": state.get("applied_compatibility_policy", {}),
            "retrieval_sources": ["pypi"] + (["repo_evidence"] if state.get("repo_evidence") else []),
            "candidate_plan_count": len(state.get("candidate_plans", [])),
            "selected_candidate_rank": state.get("selected_candidate_rank"),
            "selected_candidate_reason": state.get("selected_candidate_plan").reason if state.get("selected_candidate_plan") else "",
            "selected_candidate_runtime_profile": (
                state.get("selected_candidate_plan").runtime_profile if state.get("selected_candidate_plan") else ""
            ),
            "repair_cycle_count": state.get("repair_cycle_count", 0),
            "research_path": state.get("research_path", False),
            "structured_prompt_failures": state.get("structured_prompt_failures", 0),
            "conflict_precheck_failed": bool(getattr(state.get("constraint_pack"), "conflict_precheck_failed", False)),
            "python_constraint_intersection": list(state.get("python_constraint_intersection", [])),
            "dynamic_import_candidates": list(state.get("dynamic_import_candidates", [])),
            "repair_memory_hits": len(getattr(state.get("repair_memory_summary"), "recent_strategies", [])),
            "dynamic_alias_hits": sum(1 for source in state.get("candidate_provenance", {}).values() if source == "repo_alias"),
            "multipass_inference_used": self._research_feature_enabled("multipass_inference"),
            "version_negotiation_used": self._research_feature_enabled("version_negotiation"),
            "feedback_memory_used": self._research_feature_enabled("repair_feedback_loop"),
            "retry_severity": getattr(state.get("retry_decision"), "severity", execution.retry_severity),
            "classifier_origin": getattr(execution, "classifier_origin", state.get("classifier_origin", "")),
            "root_cause_bucket": self._root_cause_bucket(execution.category, execution.message),
            "strategy_type": state.get("strategy_history", [])[-1].strategy_type if state.get("strategy_history") else "",
            "wall_clock_seconds": total_wall_clock,
            "docker_build_seconds_total": total_build_wall_clock,
            "docker_run_seconds_total": total_run_wall_clock,
            "llm_wall_clock_seconds": llm_wall_clock_seconds,
            "image_cache_hits": image_cache_hits,
            "build_skips": build_skips,
            "started_at": state.get("case_started_at", ""),
            "finished_at": case_finished_at,
            "artifact_dir": str(artifact_dir),
            "stop_reason": state.get("stop_reason", execution.category),
            "attempt_records": [asdict(attempt) for attempt in state.get("attempt_records", [])],
        }
        state["final_result"] = result
        if self._is_research():
            self._write_json_artifact(state, "repo-evidence.json", state.get("repo_evidence", {}))
            self._write_json_artifact(state, "pypi-evidence.json", state.get("pypi_evidence", {}))
            self._write_json_artifact(state, "rag-context.json", state.get("rag_context", {}))
            self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(state.get("candidate_plans", [])))
            self._write_json_artifact(state, "structured-outputs.json", state.get("structured_outputs", {}))
            if state.get("constraint_pack") is not None:
                self._write_json_artifact(state, "constraint-pack.json", constraint_pack_to_dict(state["constraint_pack"]))
            if state.get("repair_memory_summary") is not None:
                self._write_json_artifact(state, "repair-memory-summary.json", asdict(state["repair_memory_summary"]))
            if state.get("dynamic_import_candidates"):
                self._write_json_artifact(state, "dynamic-imports.json", {"resolved": state.get("dynamic_import_candidates", [])})
            if state.get("strategy_history"):
                self._write_json_artifact(state, "strategy-history.json", [asdict(record) for record in state["strategy_history"]])
        (artifact_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return state
