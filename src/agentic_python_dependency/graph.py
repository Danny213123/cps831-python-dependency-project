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
from agentic_python_dependency.presets import COMPATIBILITY_SENSITIVE_PACKAGES, get_preset_config
from agentic_python_dependency.router import OllamaPromptRunner
from agentic_python_dependency.state import (
    AttemptRecord,
    BenchmarkCase,
    ExecutionOutcome,
    ProjectTarget,
    ResolutionState,
    ResolvedDependency,
)
from agentic_python_dependency.tools.docker_executor import DockerExecutor
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
from agentic_python_dependency.tools.pypi_store import PyPIMetadataStore


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


def route_after_execute(state: ResolutionState) -> str:
    return "finalize_result" if state["last_execution"].success else "classify_outcome"


def route_after_normalize(state: ResolutionState) -> str:
    return "finalize_result" if state.get("stop_reason") == "RepairOutputStalled" else "materialize_execution_context"


def route_after_classification(state: ResolutionState, max_attempts: int) -> str:
    execution = state["last_execution"]
    if execution.dependency_retryable and state["current_attempt"] < max_attempts:
        return "repair_prompt_c"
    return "finalize_result"


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
            preset=self.settings.preset,
            prompt_profile=self.settings.prompt_profile,
            dependency_reason="",
            candidate_provenance={},
            repair_outcome="",
            applied_compatibility_policy={},
            version_selection_source="",
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
            preset=self.settings.preset,
            prompt_profile=self.settings.prompt_profile,
            dependency_reason="",
            candidate_provenance={},
            repair_outcome="",
            applied_compatibility_policy={},
            version_selection_source="",
        )

    def run(self, state: ResolutionState) -> ResolutionState:
        try:
            graph = self.build_graph()
            return graph.invoke(
                state,
                config={"recursion_limit": infer_graph_recursion_limit(self.settings.max_attempts)},
            )
        except ImportError:  # pragma: no cover - fallback for environments without langgraph
            return self._run_fallback(state)

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
            state["target_python"] = detect_target_python_from_dockerfile(dockerfile_text)
        else:
            target = state["project_target"]
            state["source_files"] = load_python_sources(target.root_dir)
            state["current_validation_command"] = target.validation_command
            state["current_runtime_profile"] = "project"
            state["target_python"] = "3.12"

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

    def infer_packages_prompt_a(self, state: ResolutionState) -> ResolutionState:
        extracted_imports = state.get("extracted_imports", [])
        if extracted_imports and not self._should_use_extract_llm(state, extracted_imports):
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
            prompt_text = self._format_prompt("initial_imports.txt", code=code)
            prompt_texts.append(prompt_text)
            self._trace_request(state, "extract", prompt_text)
            raw_output = self.prompt_runner.invoke_template("extract", "initial_imports.txt", {"code": code})
            self._trace_response(state, "extract", raw_output)
            outputs.append({"file": file_name, "output": raw_output})
            for line in raw_output.splitlines():
                package = line.strip()
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

        if not state.get("version_options"):
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
        prompt_text = self._format_prompt("version_selection.txt", package_versions=package_versions)
        self._trace_request(state, "version", prompt_text)
        raw_output = self.prompt_runner.invoke_template(
            "version",
            "version_selection.txt",
            {"package_versions": package_versions},
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

    def normalize_dependency_plan(self, state: ResolutionState) -> ResolutionState:
        state.pop("stop_reason", None)
        if not state.get("version_options") and not state.get("repaired_dependency_lines"):
            state["selected_dependencies"] = []
            state["generated_requirements"] = "# no inferred third-party dependencies\n"
            return state

        previous_dependencies = state.get("attempt_records", [])[-1].dependencies if state.get("attempt_records") else []

        if state.get("selected_dependencies"):
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
            )
            shutil.copy2(state["benchmark_case"].snippet_path, artifact_dir / "source.py")
        else:
            context = self.docker_executor.prepare_project_context(
                state["project_target"],
                state["selected_dependencies"],
                attempt_dir,
                image_tag,
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
        classified.image_tag = execution.image_tag
        state["last_execution"] = classified
        state["last_error_category"] = classified.category
        state["last_error_details"] = classified.message
        latest_attempt = state["attempt_records"][-1]
        latest_attempt.error_category = classified.category
        latest_attempt.error_details = classified.message
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
            "preset": state.get("preset", self.settings.preset),
            "prompt_profile": state.get("prompt_profile", self.settings.prompt_profile),
            "success": execution.success,
            "attempts": state["current_attempt"],
            "final_error_category": execution.category,
            "initial_eval": state.get("benchmark_case").initial_eval if state.get("benchmark_case") else "",
            "dependencies": [dependency.pin() for dependency in state.get("selected_dependencies", [])],
            "dependency_reason": state.get("dependency_reason", ""),
            "candidate_provenance": state.get("candidate_provenance", {}),
            "repair_outcome": state.get("repair_outcome", ""),
            "version_selection_source": state.get("version_selection_source", ""),
            "compatibility_policy": state.get("applied_compatibility_policy", {}),
            "wall_clock_seconds": total_wall_clock,
            "started_at": state.get("case_started_at", ""),
            "finished_at": case_finished_at,
            "artifact_dir": str(artifact_dir),
            "stop_reason": state.get("stop_reason", execution.category),
            "attempt_records": [asdict(attempt) for attempt in state.get("attempt_records", [])],
        }
        state["final_result"] = result
        (artifact_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return state
