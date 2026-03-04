from __future__ import annotations

import json
import re
import shutil
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from packaging.version import InvalidVersion, Version

from agentic_python_dependency.benchmark.gistable import GistableDataset
from agentic_python_dependency.config import Settings
from agentic_python_dependency.presets import (
    COMPATIBILITY_SENSITIVE_PACKAGES,
    ExperimentalFeatureName,
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
    discover_python_files,
    extract_import_roots_from_code,
    filter_third_party_imports,
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
from agentic_python_dependency.tools.rag_context import build_experimental_rag_context, summarize_rag_context
from agentic_python_dependency.tools.repair_feedback import append_feedback_event, summarize_feedback_memory
from agentic_python_dependency.tools.repo_aliases import build_repo_alias_candidates
from agentic_python_dependency.tools.repo_evidence import build_repo_evidence
from agentic_python_dependency.tools.retry_policy import classify_retry_decision
from agentic_python_dependency.tools.structured_outputs import (
    StructuredOutputError,
    parse_candidate_plan_payload,
    parse_cross_validation_payload,
    parse_experimental_package_payload,
    parse_repair_plan_payload,
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


def parse_package_inference_output(raw_output: str) -> tuple[list[str], str | None]:
    raw_output = raw_output.strip()
    if not raw_output:
        return [], None

    try:
        payload = json.loads(raw_output)
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
    if execution.dependency_retryable and state["current_attempt"] < max_attempts:
        return "repair_prompt_c"
    return "finalize_result"


def route_after_experimental_plan_selection(state: ResolutionState) -> str:
    return "materialize_execution_context" if state.get("selected_candidate_plan") else "finalize_result"


def route_after_experimental_classification(state: ResolutionState, settings: Settings) -> str:
    decision = state.get("retry_decision")
    execution = state["last_execution"]
    if state["current_attempt"] >= settings.max_attempts:
        return "finalize_result"
    if decision is None:
        if not execution.dependency_retryable:
            return "finalize_result"
        if state.get("remaining_candidate_plans"):
            return "select_next_candidate_plan"
        if state.get("repair_cycle_count", 0) < settings.repair_cycle_limit:
            return "repair_prompt_c_experimental"
        return "finalize_result"
    if decision.candidate_fallback_allowed and settings.allow_candidate_fallback_before_repair and state.get("remaining_candidate_plans"):
        return "select_next_candidate_plan"
    repair_budget = settings.repair_cycle_limit
    if decision.repair_retry_budget:
        repair_budget = min(repair_budget, decision.repair_retry_budget)
    if decision.repair_allowed and state.get("repair_cycle_count", 0) < repair_budget:
        return "repair_prompt_c_experimental"
    return "finalize_result"


def route_after_experimental_repair(state: ResolutionState) -> str:
    return "select_next_candidate_plan" if state.get("candidate_plans") else "finalize_result"


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


def build_snippet_stub_argv_command(max_index: int) -> str:
    args = ["'snippet.py'"]
    for index in range(1, max_index + 1):
        args.append(f"'/tmp/apd-arg{index}'")
    return (
        "python - <<'PY'\n"
        "import os\n"
        "import sys\n"
        "import runpy\n"
        "for key, value in [('MPLBACKEND', 'Agg'), ('SDL_VIDEODRIVER', 'dummy'), ('QT_QPA_PLATFORM', 'offscreen')]:\n"
        "    os.environ.setdefault(key, value)\n"
        "for path, payload in [('/tmp/apd-arg1', b'apd'), ('/tmp/apd-arg2', b'apd')]:\n"
        "    with open(path, 'wb') as handle:\n"
        "        handle.write(payload)\n"
        f"sys.argv = [{', '.join(args)}]\n"
        "runpy.run_path('snippet.py', run_name='__main__')\n"
        "print('argv-smoke-ok')\n"
        "PY"
    )


def infer_benchmark_validation_profile(source_code: str, extracted_imports: list[str]) -> tuple[str, str]:
    lowered = source_code.lower()
    if any(token in lowered for token in ("flask(", ".run(", "uvicorn.run", "serve_forever(", "app.run(")):
        return ("service_import", build_snippet_import_command())
    if any(token in lowered for token in ("argparse", "optparse", "click.command", "sys.argv")):
        if "argparse" not in lowered and "optparse" not in lowered and "click.command" not in lowered and "sys.argv" in lowered:
            max_index = 0
            for match in re.findall(r"sys\.argv\[(\d+)\]", source_code):
                max_index = max(max_index, int(match))
            if max_index:
                return ("argv_stub", build_snippet_stub_argv_command(max_index))
        return (
            "cli_help",
            "python snippet.py --help >/tmp/apd-help.txt 2>&1 || "
            "python snippet.py -h >/tmp/apd-help.txt 2>&1 || "
            f"{build_snippet_import_command()}",
        )
    import_only_tokens = (
        "get_ipython(",
        "db.model",
        "eventregistry(",
        "boto3.client(",
        "read_csv(",
        ".parse(",
        "open(",
    )
    if "__name__ == '__main__'" in source_code or '__name__ == "__main__"' in source_code:
        return ("main_guard_import", build_snippet_import_command())
    if any(token in lowered for token in import_only_tokens):
        return ("import_smoke", build_import_smoke_command(extracted_imports))
    if any(
        token in lowered
        for token in ("tkinter", "tkinter.", "turtle", "pygame", "mainloop(", "plt.show(", "pyqt", "wx.")
    ):
        return ("headless_imports", build_import_smoke_command(extracted_imports))
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
    def __init__(
        self,
        settings: Settings,
        prompt_runner: OllamaPromptRunner | None = None,
        pypi_store: PyPIMetadataStore | None = None,
        docker_executor: DockerExecutor | None = None,
    ):
        self.settings = settings
        self.preset_config = get_preset_config(settings.preset)
        self.prompt_runner = prompt_runner or OllamaPromptRunner(settings, settings.prompt_template_dir)
        self.pypi_store = pypi_store or PyPIMetadataStore(settings.pypi_cache_dir)
        self.package_metadata_store = PackageMetadataStore(settings.package_metadata_dir)
        self.docker_executor = docker_executor or DockerExecutor(settings)
        self.dataset = GistableDataset(settings)

    def _prompt_template(self, name: str) -> str:
        return (self.settings.prompt_template_dir / name).read_text(encoding="utf-8")

    def _format_prompt(self, name: str, **variables: Any) -> str:
        return self._prompt_template(name).format(**variables)

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

    def _trace_request(self, state: ResolutionState, stage: str, prompt_text: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        model = self._stage_model_name(stage)
        attempt = state.get("current_attempt", 0)
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
        message = (
            f"[{timestamp}] case={state.get('case_id', '')} attempt={attempt} stage={stage} model={model}\n"
            "--- RESPONSE ---\n"
            f"{response_text.rstrip()}\n"
            "===\n"
        )
        self._emit_trace(state, message)

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
        if normalize_package_name(alias) not in {normalize_package_name(package) for package in state.get("inferred_packages", [])}:
            return None
        try:
            option = self.pypi_store.get_version_options(
                alias,
                state["target_python"],
                preset=self.settings.preset,
            )
        except FileNotFoundError:
            return None
        if not option.versions:
            return None
        state["repair_outcome"] = "alias_retry"
        return [f"{alias}=={option.versions[0]}"]

    def _candidate_provenance_from(self, packages: list[str], extracted_imports: list[str]) -> dict[str, str]:
        return normalize_candidate_packages_with_sources(packages, extracted_imports)

    def _uses_full_apd(self) -> bool:
        return self.settings.resolver == "apd"

    def _resolver_uses_rag(self) -> bool:
        return self.settings.use_rag and self.settings.resolver != "readpye"

    def _is_experimental(self) -> bool:
        return self.settings.preset == "experimental"

    def _experimental_feature_enabled(self, feature: ExperimentalFeatureName) -> bool:
        return self._is_experimental() and self.settings.experimental_feature_enabled(feature)

    @staticmethod
    def _write_json_artifact(state: ResolutionState, filename: str, payload: Any) -> None:
        artifact_dir = state.get("artifact_dir")
        if not artifact_dir:
            return
        Path(artifact_dir, filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _candidate_plan_payload(plans: list[CandidatePlan]) -> list[dict[str, Any]]:
        return [
            {
                "rank": plan.rank,
                "reason": plan.reason,
                "dependencies": [{"name": dep.name, "version": dep.version} for dep in plan.dependencies],
            }
            for plan in plans
        ]

    @staticmethod
    def _allowed_versions_map(options: list[PackageVersionOptions]) -> dict[str, set[str]]:
        return {normalize_package_name(option.package): set(option.versions) for option in options}

    def _experimental_allowed_packages(self, state: ResolutionState) -> set[str]:
        return {normalize_package_name(package) for package in state.get("inferred_packages", [])}

    def _experimental_bundle_name(self) -> str:
        return self.settings.experimental_bundle if self._is_experimental() else "baseline"

    @staticmethod
    def _strategy_type_from(previous: list[str], current: list[str]) -> str:
        if not previous:
            return "fallback_candidate"
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

    def _default_experimental_plans(self, state: ResolutionState) -> list[CandidatePlan]:
        options = state.get("version_options", [])
        if not state.get("inferred_packages"):
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

    def initial_state_for_case(self, case: BenchmarkCase, run_id: str | None = None) -> ResolutionState:
        return ResolutionState(
            run_id=run_id or uuid4().hex[:12],
            mode="gistable",
            case_id=case.case_id,
            benchmark_case=case,
            case_started_at=datetime.now(timezone.utc).isoformat(),
            attempt_records=[],
            prompt_history={"prompt_a": [], "prompt_b": "", "prompt_c": []},
            model_outputs={"extract": [], "version": [], "repair": [], "adjudicate": []},
            current_attempt=0,
            repair_stall_count=0,
            resolver=self.settings.resolver,
            preset=self.settings.preset,
            prompt_profile=self.settings.prompt_profile,
            experimental_bundle=self.settings.experimental_bundle,
            experimental_features=self.settings.experimental_features,
            dependency_reason="",
            candidate_provenance={},
            repair_outcome="",
            applied_compatibility_policy={},
            version_selection_source="",
            inference_candidates=[],
            repo_alias_candidates={},
            dynamic_import_candidates=[],
            constraint_pack=None,
            repair_memory_summary=None,
            retry_decision=None,
            strategy_history=[],
            feedback_memory_hits=0,
            version_conflict_notes=[],
            python_constraint_intersection=[],
            top_level_module_map={},
            system_dependencies=[],
            resolver_implementation="internal",
            repo_evidence={},
            pypi_evidence={},
            rag_context={},
            candidate_plans=[],
            remaining_candidate_plans=[],
            selected_candidate_plan=None,
            selected_candidate_rank=None,
            repair_cycle_count=0,
            structured_outputs={},
            experimental_path=self._is_experimental(),
            structured_prompt_failures=0,
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
            model_outputs={"extract": [], "version": [], "repair": [], "adjudicate": []},
            current_attempt=0,
            repair_stall_count=0,
            resolver=self.settings.resolver,
            preset=self.settings.preset,
            prompt_profile=self.settings.prompt_profile,
            experimental_bundle=self.settings.experimental_bundle,
            experimental_features=self.settings.experimental_features,
            dependency_reason="",
            candidate_provenance={},
            repair_outcome="",
            applied_compatibility_policy={},
            version_selection_source="",
            inference_candidates=[],
            repo_alias_candidates={},
            dynamic_import_candidates=[],
            constraint_pack=None,
            repair_memory_summary=None,
            retry_decision=None,
            strategy_history=[],
            feedback_memory_hits=0,
            version_conflict_notes=[],
            python_constraint_intersection=[],
            top_level_module_map={},
            system_dependencies=[],
            resolver_implementation="internal",
            repo_evidence={},
            pypi_evidence={},
            rag_context={},
            candidate_plans=[],
            remaining_candidate_plans=[],
            selected_candidate_plan=None,
            selected_candidate_rank=None,
            repair_cycle_count=0,
            structured_outputs={},
            experimental_path=self._is_experimental(),
            structured_prompt_failures=0,
        )

    def run(self, state: ResolutionState) -> ResolutionState:
        try:
            graph = self.build_experimental_graph() if self._is_experimental() else self.build_graph()
            return graph.invoke(
                state,
                config={"recursion_limit": infer_graph_recursion_limit(self.settings.max_attempts)},
            )
        except ImportError:  # pragma: no cover - fallback for environments without langgraph
            return self._run_experimental_fallback(state) if self._is_experimental() else self._run_fallback(state)

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
            if route_after_classification(current, self.settings.max_attempts) == "finalize_result":
                return self.finalize_result(current)
            current = self.repair_prompt_c(current)
            current = self.retrieve_pypi_metadata(current)
            current = self.infer_versions_prompt_b(current)
            current = self.normalize_dependency_plan(current)
            if route_after_normalize(current) == "finalize_result":
                return self.finalize_result(current)

    def _run_experimental_fallback(self, state: ResolutionState) -> ResolutionState:
        current = state
        current = self.load_target(current)
        current = self.extract_imports(current)
        current = self.extract_dynamic_imports(current)
        current = self.gather_repo_evidence(current)
        current = self.build_dynamic_alias_candidates(current)
        current = self.infer_package_candidates(current)
        current = self.cross_validate_packages(current)
        current = self.retrieve_pypi_metadata(current)
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
        if not current.get("selected_candidate_plan"):
            return self.finalize_result(current)
        while True:
            current = self.materialize_execution_context(current)
            current = self.execute_candidate(current)
            if route_after_execute(current) == "finalize_result":
                return self.finalize_result(current)
            current = self.classify_outcome(current)
            next_step = route_after_experimental_classification(current, self.settings)
            if next_step == "finalize_result":
                return self.finalize_result(current)
            if next_step == "select_next_candidate_plan":
                current = self.select_next_candidate_plan(current)
                if not current.get("selected_candidate_plan"):
                    return self.finalize_result(current)
                continue
            current = self.build_repair_memory_summary(current)
            current = self.repair_prompt_c_experimental(current)
            if route_after_experimental_repair(current) == "finalize_result":
                return self.finalize_result(current)
            current = self.select_next_candidate_plan(current)
            if not current.get("selected_candidate_plan"):
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
            {"repair_prompt_c": "repair_prompt_c", "finalize_result": "finalize_result"},
        )
        graph.add_edge("repair_prompt_c", "retrieve_pypi_metadata")
        graph.add_edge("finalize_result", END)
        return graph.compile()

    def build_experimental_graph(self):
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
        graph.add_node("build_repair_memory_summary", self.build_repair_memory_summary)
        graph.add_node("repair_prompt_c_experimental", self.repair_prompt_c_experimental)
        graph.add_node("finalize_result", self.finalize_result)

        graph.add_edge(START, "load_target")
        graph.add_edge("load_target", "extract_imports")
        graph.add_edge("extract_imports", "extract_dynamic_imports")
        graph.add_edge("extract_dynamic_imports", "gather_repo_evidence")
        graph.add_edge("gather_repo_evidence", "build_dynamic_alias_candidates")
        graph.add_edge("build_dynamic_alias_candidates", "infer_package_candidates")
        graph.add_edge("infer_package_candidates", "cross_validate_packages")
        graph.add_edge("cross_validate_packages", "retrieve_pypi_metadata")
        graph.add_edge("retrieve_pypi_metadata", "retrieve_version_specific_metadata")
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
            route_after_experimental_plan_selection,
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
            lambda state: route_after_experimental_classification(state, self.settings),
            {
                "select_next_candidate_plan": "select_next_candidate_plan",
                "repair_prompt_c_experimental": "build_repair_memory_summary",
                "finalize_result": "finalize_result",
            },
        )
        graph.add_edge("build_repair_memory_summary", "repair_prompt_c_experimental")
        graph.add_conditional_edges(
            "repair_prompt_c_experimental",
            route_after_experimental_repair,
            {"select_next_candidate_plan": "select_next_candidate_plan", "finalize_result": "finalize_result"},
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
            dockerfile_text = case.dockerfile_path.read_text(encoding="utf-8")
            state["source_files"] = {"snippet.py": source}
            state["current_validation_command"] = ""
            state["current_runtime_profile"] = "docker_cmd"
            benchmark_target_python = detect_target_python_from_dockerfile(dockerfile_text)
            state["benchmark_target_python"] = benchmark_target_python
            state["target_python"] = benchmark_target_python
            state["inferred_target_python"] = ""
            state["python_version_source"] = "benchmark_dockerfile"
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
        if state["mode"] == "gistable":
            source = state["source_files"].get("snippet.py", "")
            profile, command = infer_benchmark_validation_profile(source, state["extracted_imports"])
            state["current_runtime_profile"] = profile
            state["current_validation_command"] = command
        return state

    def extract_dynamic_imports(self, state: ResolutionState) -> ResolutionState:
        if not self._experimental_feature_enabled("dynamic_imports"):
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
        if not self._is_experimental() or not self.settings.repo_evidence_enabled:
            state["repo_evidence"] = {}
            return state
        evidence = build_repo_evidence(state)
        state["repo_evidence"] = evidence
        self._write_json_artifact(state, "repo-evidence.json", evidence)
        return state

    def build_dynamic_alias_candidates(self, state: ResolutionState) -> ResolutionState:
        if not self._experimental_feature_enabled("dynamic_aliases"):
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
        if not self._is_experimental():
            return self.infer_packages_prompt_a(state)

        if not self._experimental_feature_enabled("multipass_inference"):
            state = self.infer_packages_prompt_a(state)
            state["inference_candidates"] = [
                InferenceCandidate(
                    package=package,
                    confidence=1.0 if state.get("candidate_provenance", {}).get(package) != "llm" else 0.9,
                    sources=[state.get("candidate_provenance", {}).get(package, "llm")],
                    reason="baseline experimental inference",
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
        _, inferred_python_version = parse_package_inference_output(raw_version_output)
        if inferred_python_version:
            state["inferred_target_python"] = inferred_python_version
            state["target_python"] = inferred_python_version
            state["python_version_source"] = "llm_prompt_a"
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
            candidate_sources.setdefault(package, set()).add("llm")
            candidate_confidence[package] = max(candidate_confidence.get(package, 0.0), float(item.get("confidence", 0.0) or 0.0))
            if item.get("evidence"):
                candidate_reason[package] = "; ".join(str(entry) for entry in item.get("evidence", []))

        candidates = [
            InferenceCandidate(
                package=package,
                confidence=candidate_confidence.get(package, 1.0 if "llm" not in sources else 0.75),
                sources=sorted(sources),
                reason=candidate_reason.get(package, ""),
                accepted=False,
            )
            for package, sources in sorted(candidate_sources.items(), key=lambda item: item[0].lower())
        ]
        state["inference_candidates"] = candidates
        self._write_json_artifact(state, "package-candidates.json", [asdict(candidate) for candidate in candidates])
        return state

    def cross_validate_packages(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
            return state
        if not self._experimental_feature_enabled("multipass_inference"):
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
                enabled_features=", ".join(state.get("experimental_features", ())),
            )
            self._trace_request(state, "extract", prompt_text)
            raw_output = self.prompt_runner.invoke_template(
                "extract",
                "package_cross_validate.txt",
                {
                    "target_python": state.get("target_python", ""),
                    "candidate_payload": candidates_json,
                    "repo_evidence": json.dumps(state.get("repo_evidence", {}), indent=2)[:2500],
                    "enabled_features": ", ".join(state.get("experimental_features", ())),
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
        state["inferred_packages"] = sorted(provenance, key=str.lower)
        state["candidate_provenance"] = provenance
        return state

    def build_rag_context(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
            return state
        pypi_evidence = state.get("pypi_evidence", {})
        repo_evidence = state.get("repo_evidence", {})
        rag_context = build_experimental_rag_context(
            state,
            repo_evidence=repo_evidence,
            pypi_evidence=pypi_evidence,
        )
        state["rag_context"] = rag_context
        self._write_json_artifact(state, "rag-context.json", rag_context)
        return state

    def infer_packages_prompt_a(self, state: ResolutionState) -> ResolutionState:
        extracted_imports = state.get("extracted_imports", [])
        if self._is_experimental():
            return self._infer_packages_prompt_a_experimental(state, extracted_imports)
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
            state["inferred_packages"] = normalized
            state["candidate_provenance"] = provenance
            return state

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
                state["inferred_packages"] = normalized
                state["candidate_provenance"] = provenance
                return state

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
            if inferred_python_version:
                state["inferred_target_python"] = inferred_python_version
                state["target_python"] = inferred_python_version
                state["python_version_source"] = "llm_prompt_a"
            for package in packages:
                if package:
                    inferred.add(package)

        state["prompt_history"]["prompt_a"] = prompt_texts
        state["model_outputs"]["extract"] = outputs
        if not inferred:
            inferred.update(extracted_imports)
        provenance = self._candidate_provenance_from(sorted(inferred), extracted_imports)
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
        state["inferred_packages"] = normalized
        state["candidate_provenance"] = provenance
        return state

    def _infer_packages_prompt_a_experimental(
        self, state: ResolutionState, extracted_imports: list[str]
    ) -> ResolutionState:
        if not self._uses_full_apd():
            provenance = self._candidate_provenance_from(extracted_imports, extracted_imports)
            state["inferred_packages"] = sorted(provenance, key=str.lower)
            state["candidate_provenance"] = provenance
            return state

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

        inferred = [entry["package"] for entry in packages]
        _, inferred_python_version = parse_package_inference_output(raw_version_output)
        state["structured_outputs"]["extract"] = packages
        if inferred_python_version:
            state["inferred_target_python"] = inferred_python_version
            state["target_python"] = inferred_python_version
            state["python_version_source"] = "llm_prompt_a"
        if not inferred:
            inferred = extracted_imports
        provenance = self._candidate_provenance_from(inferred, extracted_imports)
        for entry in packages:
            normalized_name = next(
                (package for package in provenance if normalize_package_name(package) == normalize_package_name(entry["package"])),
                None,
            )
            if normalized_name:
                provenance[normalized_name] = str(entry.get("source", provenance[normalized_name]) or provenance[normalized_name])
        state["inferred_packages"] = sorted(provenance, key=str.lower)
        state["candidate_provenance"] = provenance
        return state

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
        for package in packages:
            if not package:
                continue
            if not looks_like_package_name(package):
                unresolved.append(package)
                continue
            try:
                option = self.pypi_store.get_version_options(
                    package,
                    state["target_python"],
                    preset=self.settings.preset,
                )
            except FileNotFoundError:
                unresolved.append(package)
                continue
            if not option.versions:
                unresolved.append(package)
                continue
            options.append(option)
        state["version_options"] = options
        state["unresolved_packages"] = unresolved
        state["applied_compatibility_policy"] = {
            option.package: option.policy_notes for option in options if option.policy_notes
        }
        if self._is_experimental():
            pypi_evidence = {
                "packages": [
                    {
                        "package": option.package,
                        "versions": option.versions,
                        "requires_python": option.requires_python,
                        "requires_dist": option.requires_dist,
                        "policy_notes": option.policy_notes,
                    }
                    for option in options
                ],
                "unresolved_packages": unresolved,
            }
            state["pypi_evidence"] = pypi_evidence
            self._write_json_artifact(state, "pypi-evidence.json", pypi_evidence)
        return state

    def retrieve_version_specific_metadata(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
            return state
        if not (
            self._experimental_feature_enabled("transitive_conflicts")
            or self._experimental_feature_enabled("version_negotiation")
            or self._experimental_feature_enabled("dynamic_aliases")
        ):
            return state
        updated_options: list[PackageVersionOptions] = []
        for option in state.get("version_options", []):
            release_requires_dist: dict[str, list[str]] = {}
            for version in option.versions[:5]:
                release_files = self.pypi_store.release_files(option.package, version)
                metadata = self.package_metadata_store.parse_release_metadata(
                    option.package,
                    version,
                    release_files=release_files,
                )
                release_requires_dist[version] = [str(item) for item in metadata.get("requires_dist", [])]
            updated_options.append(
                PackageVersionOptions(
                    package=option.package,
                    versions=option.versions,
                    requires_python=option.requires_python,
                    upload_time=option.upload_time,
                    policy_notes=option.policy_notes,
                    requires_dist={**option.requires_dist, **release_requires_dist},
                )
            )
        if updated_options:
            state["version_options"] = updated_options
            if state.get("pypi_evidence", {}).get("packages"):
                for payload, option in zip(state["pypi_evidence"]["packages"], updated_options, strict=False):
                    payload["requires_dist"] = option.requires_dist
        return state

    def build_constraint_pack(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
            return state
        if not (
            self._experimental_feature_enabled("transitive_conflicts")
            or self._experimental_feature_enabled("python_constraint_intersection")
            or self._experimental_feature_enabled("version_negotiation")
        ):
            state["constraint_pack"] = None
            state["version_conflict_notes"] = []
            state["python_constraint_intersection"] = []
            return state
        pack = build_constraint_pack(
            state.get("version_options", []),
            target_python=state.get("target_python", "3.12"),
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
        return state

    def requires_python_intersection_check(self, state: ResolutionState) -> ResolutionState:
        pack = state.get("constraint_pack")
        if pack is None or not self._experimental_feature_enabled("python_constraint_intersection"):
            return state
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
                message="Experimental constraint precheck blocked all candidate plans.",
                build_succeeded=False,
                run_succeeded=False,
                dependency_retryable=False,
                retry_severity="terminal",
            )
            state["retry_decision"] = classify_retry_decision("ConstraintConflictError")
        return state

    def load_feedback_memory_summary(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental() or not self._experimental_feature_enabled("repair_feedback_loop"):
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
        if not self._is_experimental():
            return state
        if not self._experimental_feature_enabled("version_negotiation"):
            state["structured_outputs"]["candidate_bundles"] = []
            return state
        pack = state.get("constraint_pack")
        if pack is None:
            state["structured_outputs"]["candidate_bundles"] = []
            return state
        bundles = generate_candidate_bundles(pack)
        payload = [
            {
                "rank": index + 1,
                "dependencies": [{"name": dependency.name, "version": dependency.version} for dependency in bundle],
            }
            for index, bundle in enumerate(bundles)
        ]
        state["structured_outputs"]["candidate_bundles"] = payload
        self._write_json_artifact(state, "candidate-bundles.json", payload)
        return state

    def negotiate_version_bundles(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
            return state
        if not self._experimental_feature_enabled("version_negotiation"):
            return state
        bundles = state.get("structured_outputs", {}).get("candidate_bundles", [])
        if not bundles:
            state["candidate_plans"] = []
            state["remaining_candidate_plans"] = []
            return state
        allowed_versions = self._allowed_versions_map(state.get("version_options", []))
        allowed_packages = self._experimental_allowed_packages(state)
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
            },
        )
        self._trace_response(state, "version", raw_output)
        state["structured_outputs"]["version_negotiation_raw"] = raw_output
        try:
            plans = parse_version_negotiation_payload(
                raw_output,
                allowed_packages=allowed_packages,
                allowed_versions=allowed_versions,
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
                )
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                plans = []
        state["structured_outputs"]["version_negotiation"] = self._candidate_plan_payload(plans)
        self._write_json_artifact(state, "version-negotiation.json", state["structured_outputs"]["version_negotiation"])
        if plans:
            state["candidate_plans"] = plans
            state["remaining_candidate_plans"] = list(plans)
            state["version_selection_source"] = "experimental_version_negotiation"
            state["dependency_reason"] = "llm_version_selection"
        return state

    def generate_candidate_plans(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
            return state
        if self._experimental_feature_enabled("version_negotiation") and state.get("candidate_plans"):
            self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(state.get("candidate_plans", [])))
            return state

        default_plans = self._default_experimental_plans(state)
        options = state.get("version_options", [])
        if default_plans and (
            not options
            or all(len(option.versions) == 1 for option in options)
        ):
            if not options:
                state["version_selection_source"] = "experimental_default_no_versions"
            elif all(len(option.versions) == 1 for option in options):
                state["version_selection_source"] = "experimental_single_version_fast_path"
            else:
                state["version_selection_source"] = "experimental_deterministic_default"
            state["candidate_plans"] = default_plans
            state["remaining_candidate_plans"] = list(default_plans)
            state["structured_outputs"]["candidate_plans"] = self._candidate_plan_payload(default_plans)
            self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(default_plans))
            return state

        if not options:
            state["candidate_plans"] = default_plans
            state["remaining_candidate_plans"] = list(default_plans)
            self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(default_plans))
            return state

        allowed_packages = sorted(state.get("inferred_packages", []), key=str.lower)
        allowed_package_set = self._experimental_allowed_packages(state)
        allowed_versions = self._allowed_versions_map(options)
        rag_context_summary = summarize_rag_context(state.get("rag_context", {}), limit=6000)
        candidate_template = (
            "candidate_plans_v2.txt"
            if self._experimental_feature_enabled("transitive_conflicts")
            or self._experimental_feature_enabled("multipass_inference")
            else "candidate_plans.txt"
        )
        prompt_text = self._format_prompt(
            candidate_template,
            target_python=state.get("target_python", ""),
            allowed_packages="\n".join(allowed_packages),
            rag_context=rag_context_summary,
            max_plan_count=self.settings.candidate_plan_count,
        )
        self._trace_request(state, "version", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "version",
            candidate_template,
            {
                "target_python": state.get("target_python", ""),
                "allowed_packages": "\n".join(allowed_packages),
                "rag_context": rag_context_summary,
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
                )
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                plans = default_plans

        if not plans:
            plans = default_plans

        state["candidate_plans"] = plans
        state["remaining_candidate_plans"] = list(plans)
        state["structured_outputs"]["candidate_plans"] = self._candidate_plan_payload(plans)
        self._write_json_artifact(state, "candidate-plans.json", self._candidate_plan_payload(plans))
        if not state.get("dependency_reason"):
            state["dependency_reason"] = "llm_version_selection"
        state["version_selection_source"] = "experimental_candidate_plans"
        return state

    def select_next_candidate_plan(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
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
            },
        )
        return state

    def repair_prompt_c_experimental(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
            return state
        state["repair_cycle_count"] = state.get("repair_cycle_count", 0) + 1
        allowed_packages = sorted(state.get("inferred_packages", []), key=str.lower)
        allowed_package_set = self._experimental_allowed_packages(state)
        allowed_versions = self._allowed_versions_map(state.get("version_options", []))
        previous_plan = "\n".join(dep.pin() for dep in state.get("selected_dependencies", []))
        attempted_plans = "\n".join(
            ", ".join(attempt.dependencies)
            for attempt in state.get("attempt_records", [])
            if attempt.dependencies
        )
        rag_context_summary = summarize_rag_context(state.get("rag_context", {}), limit=8000)
        repair_template = "repair_attempt_v2.txt" if self._experimental_feature_enabled("repair_memory") else "repair_attempt.txt"
        feedback_summary = (
            summarize_feedback_memory(self.settings.workspace_memory_dir)
            if self._experimental_feature_enabled("repair_feedback_loop")
            else {"entries": []}
        )
        prompt_text = self._format_prompt(
            repair_template,
            target_python=state.get("target_python", ""),
            allowed_packages="\n".join(allowed_packages),
            previous_plan=previous_plan,
            attempted_plans=attempted_plans,
            error_details=state.get("last_error_details", ""),
            rag_context=rag_context_summary,
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
                "previous_plan": previous_plan,
                "attempted_plans": attempted_plans,
                "error_details": state.get("last_error_details", ""),
                "rag_context": rag_context_summary,
                "max_plan_count": min(2, self.settings.candidate_plan_count),
                "repair_memory": json.dumps(asdict(state.get("repair_memory_summary") or RepairMemorySummary()), indent=2)[:2500],
                "feedback_summary": json.dumps(feedback_summary, indent=2)[:2500],
            },
        )
        self._trace_response(state, "repair", raw_output)
        state["prompt_history"]["prompt_c"].append(prompt_text)
        state["model_outputs"]["repair"].append({"attempt": state["current_attempt"], "output": raw_output})
        state["structured_outputs"]["repair_raw"] = raw_output
        try:
            repair_applicable, plans = parse_repair_plan_payload(
                raw_output,
                allowed_packages=allowed_package_set,
                allowed_versions=allowed_versions,
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
                )
            except StructuredOutputError:
                state["structured_prompt_failures"] = state.get("structured_prompt_failures", 0) + 1
                repair_applicable, plans = False, []

        state["structured_outputs"]["repair"] = {
            "repair_applicable": repair_applicable,
            "plans": self._candidate_plan_payload(plans),
        }
        if not repair_applicable or not plans:
            state["repair_outcome"] = "repair_not_applicable"
            state["candidate_plans"] = []
            state["remaining_candidate_plans"] = []
            return state
        state["repair_outcome"] = "llm_repair"
        state["candidate_plans"] = plans
        state["remaining_candidate_plans"] = list(plans)
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
        artifact_dir = Path(state["artifact_dir"])
        attempt_dir = artifact_dir / f"attempt_{state['current_attempt']:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        state["current_attempt_dir"] = str(attempt_dir)

        image_tag = f"pllm-{state['run_id']}-{state['case_id']}-{state['current_attempt']}"
        if state["mode"] == "gistable":
            context = self.docker_executor.prepare_benchmark_context(
                state["benchmark_case"],
                state["selected_dependencies"],
                attempt_dir,
                image_tag,
                state["target_python"],
                state.get("current_validation_command") or None,
                extra_system_packages=state.get("system_dependencies", []),
            )
            shutil.copy2(state["benchmark_case"].snippet_path, artifact_dir / "source.py")
        else:
            context = self.docker_executor.prepare_project_context(
                state["project_target"],
                state["selected_dependencies"],
                attempt_dir,
                image_tag,
                extra_system_packages=state.get("system_dependencies", []),
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
        if self._is_experimental():
            self._write_json_artifact(state, "structured-outputs.json", state.get("structured_outputs", {}))
        return state

    def execute_candidate(self, state: ResolutionState) -> ResolutionState:
        attempt_dir = Path(state["current_attempt_dir"])
        context = state["prepared_execution_context"]
        attempt_started_at = datetime.now(timezone.utc).isoformat()

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
                started_at=attempt_started_at,
                finished_at=attempt_finished_at,
            )
        )
        return state

    def classify_outcome(self, state: ResolutionState) -> ResolutionState:
        execution = state["last_execution"]
        classified = classify_error(execution.build_log, execution.run_log, execution.exit_code)
        if not self._uses_full_apd():
            classified.dependency_retryable = False
        if self._experimental_feature_enabled("smart_repair_routing"):
            system_packages_injected = bool(
                getattr(state.get("prepared_execution_context"), "system_packages", [])
            ) if state.get("prepared_execution_context") is not None else False
            retry_decision = classify_retry_decision(
                classified.category,
                system_packages_injected=system_packages_injected,
                native_retry_used=sum(
                    1 for attempt in state.get("attempt_records", []) if attempt.error_category == "NativeBuildError"
                ),
            )
            classified.dependency_retryable = retry_decision.repair_allowed or retry_decision.candidate_fallback_allowed
            classified.retry_severity = retry_decision.severity
        else:
            retry_decision = None
            if classified.dependency_retryable:
                classified.retry_severity = "repair_retryable"
        classified.image_tag = execution.image_tag
        state["last_execution"] = classified
        state["retry_decision"] = retry_decision
        state["last_error_category"] = classified.category
        state["last_error_details"] = classified.message
        latest_attempt = state["attempt_records"][-1]
        latest_attempt.error_category = classified.category
        latest_attempt.error_details = classified.message
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
        if self._is_experimental():
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
                },
            )
            self._write_json_artifact(
                state,
                "strategy-history.json",
                [asdict(record) for record in state.get("strategy_history", [])],
            )
            if self._experimental_feature_enabled("repair_feedback_loop"):
                append_feedback_event(
                    self.settings.workspace_memory_dir,
                    {
                        "run_id": state.get("run_id", ""),
                        "case_id": state.get("case_id", ""),
                        "target_python": state.get("target_python", ""),
                        "resolver": state.get("resolver", self.settings.resolver),
                        "preset": state.get("preset", self.settings.preset),
                        "enabled_features": list(state.get("experimental_features", ())),
                        "error_category": classified.category,
                        "strategy_type": strategy_record.strategy_type,
                        "dependency_fingerprint": "|".join(current_dependencies),
                        "package_family_fingerprint": "|".join(sorted(dep.split("==", 1)[0] for dep in current_dependencies)),
                        "selected_candidate_rank": state.get("selected_candidate_rank"),
                        "success": classified.success,
                        "wall_clock_seconds": latest_attempt.wall_clock_seconds,
                    },
                )
        return state

    def build_repair_memory_summary(self, state: ResolutionState) -> ResolutionState:
        if not self._is_experimental():
            return state
        if not self._experimental_feature_enabled("repair_memory"):
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
            return state
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
        except ValueError:
            state["repaired_dependency_lines"] = repaired_lines
            state["repair_outcome"] = "repair_not_applicable"
        return state

    def finalize_result(self, state: ResolutionState) -> ResolutionState:
        artifact_dir = Path(state["artifact_dir"])
        execution = state["last_execution"]
        case_finished_at = datetime.now(timezone.utc).isoformat()
        state["case_finished_at"] = case_finished_at
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
        result = {
            "run_id": state["run_id"],
            "case_id": state["case_id"],
            "mode": state["mode"],
            "resolver": state.get("resolver", self.settings.resolver),
            "resolver_implementation": state.get("resolver_implementation", "internal"),
            "preset": state.get("preset", self.settings.preset),
            "prompt_profile": state.get("prompt_profile", self.settings.prompt_profile),
            "experimental_bundle": state.get("experimental_bundle", "baseline"),
            "experimental_features": list(state.get("experimental_features", ())),
            "model_profile": self.settings.model_profile,
            "use_moe": self.settings.use_moe,
            "use_rag": self.settings.use_rag,
            "use_langchain": self.settings.use_langchain,
            "rag_mode": self.settings.rag_mode,
            "structured_prompting": self.settings.structured_prompting,
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
            "target_python": state.get("target_python", ""),
            "dependencies": [dependency.pin() for dependency in state.get("selected_dependencies", [])],
            "system_dependencies": list(state.get("system_dependencies", [])),
            "dependency_reason": state.get("dependency_reason", ""),
            "candidate_provenance": state.get("candidate_provenance", {}),
            "repair_outcome": state.get("repair_outcome", ""),
            "version_selection_source": state.get("version_selection_source", ""),
            "compatibility_policy": state.get("applied_compatibility_policy", {}),
            "retrieval_sources": ["pypi"] + (["repo_evidence"] if state.get("repo_evidence") else []),
            "candidate_plan_count": len(state.get("candidate_plans", [])),
            "selected_candidate_rank": state.get("selected_candidate_rank"),
            "selected_candidate_reason": state.get("selected_candidate_plan").reason if state.get("selected_candidate_plan") else "",
            "repair_cycle_count": state.get("repair_cycle_count", 0),
            "experimental_path": state.get("experimental_path", False),
            "structured_prompt_failures": state.get("structured_prompt_failures", 0),
            "conflict_precheck_failed": bool(getattr(state.get("constraint_pack"), "conflict_precheck_failed", False)),
            "python_constraint_intersection": list(state.get("python_constraint_intersection", [])),
            "dynamic_import_candidates": list(state.get("dynamic_import_candidates", [])),
            "repair_memory_hits": len(getattr(state.get("repair_memory_summary"), "recent_strategies", [])),
            "dynamic_alias_hits": sum(1 for source in state.get("candidate_provenance", {}).values() if source == "repo_alias"),
            "multipass_inference_used": self._experimental_feature_enabled("multipass_inference"),
            "version_negotiation_used": self._experimental_feature_enabled("version_negotiation"),
            "feedback_memory_used": self._experimental_feature_enabled("repair_feedback_loop"),
            "retry_severity": getattr(state.get("retry_decision"), "severity", execution.retry_severity),
            "strategy_type": state.get("strategy_history", [])[-1].strategy_type if state.get("strategy_history") else "",
            "wall_clock_seconds": total_wall_clock,
            "started_at": state.get("case_started_at", ""),
            "finished_at": case_finished_at,
            "artifact_dir": str(artifact_dir),
            "stop_reason": state.get("stop_reason", execution.category),
            "attempt_records": [asdict(attempt) for attempt in state.get("attempt_records", [])],
        }
        state["final_result"] = result
        if self._is_experimental():
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
