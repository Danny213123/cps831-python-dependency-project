from __future__ import annotations

import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agentic_python_dependency.config import Settings
from agentic_python_dependency.presets import PRESET_CONFIGS


ActionCallback = Callable[..., int]


def _format_progress_bar(completed: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[" + ("#" * width) + "]"
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class TerminalBenchmarkDashboard:
    refresh_interval: float = 0.2

    def __post_init__(self) -> None:
        self.run_id = ""
        self.total = 0
        self.completed = 0
        self.successes = 0
        self.failures = 0
        self.preset = "optimized"
        self.prompt_profile = "optimized"
        self.jobs = 1
        self.target = "benchmark"
        self.artifacts_dir = Path(".")
        self.current_cases: list[str] = []
        self.last_case_id = ""
        self.last_status = ""
        self.started_at = time.monotonic()
        self.summary_path: Path | None = None
        self.warnings_path: Path | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._isatty = sys.stdout.isatty()

    def start(
        self,
        *,
        run_id: str,
        total: int,
        completed: int,
        successes: int,
        failures: int,
        preset: str,
        prompt_profile: str,
        jobs: int,
        target: str,
        artifacts_dir: Path,
    ) -> None:
        self.run_id = run_id
        self.total = total
        self.completed = completed
        self.successes = successes
        self.failures = failures
        self.preset = preset
        self.prompt_profile = prompt_profile
        self.jobs = jobs
        self.target = target
        self.artifacts_dir = artifacts_dir
        self.started_at = time.monotonic()
        self._render()
        if not self._isatty or self._thread is not None:
            return

        def _refresh_loop() -> None:
            while not self._stop_event.wait(self.refresh_interval):
                self._render()

        self._thread = threading.Thread(target=_refresh_loop, name="apd-ui-benchmark", daemon=True)
        self._thread.start()

    def case_started(self, case_id: str) -> None:
        with self._lock:
            if case_id not in self.current_cases:
                self.current_cases.append(case_id)
            self._render()

    def advance(self, result: dict[str, object]) -> None:
        case_id = str(result.get("case_id", ""))
        success = bool(result.get("success", False))
        with self._lock:
            self.completed = min(self.total, self.completed + 1)
            self.successes += int(success)
            self.failures += int(not success)
            self.last_case_id = case_id
            self.last_status = "success" if success else str(result.get("final_error_category", "failure"))
            self.current_cases = [item for item in self.current_cases if item != case_id]
            self._render()

    def finish(self, *, summary_path: Path, warnings_path: Path | None) -> None:
        self.summary_path = summary_path
        self.warnings_path = warnings_path
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.refresh_interval + 0.1)
        with self._lock:
            self.completed = self.total
            self.current_cases.clear()
            self._render(final=True)

    def _lines(self, final: bool = False) -> list[str]:
        width = max(72, min(shutil.get_terminal_size((100, 24)).columns, 120))
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        lines = [
            "=" * width,
            "APD Benchmark Dashboard".center(width),
            "=" * width,
            (
                f"Run ID: {self.run_id}    Target: {self.target}    "
                f"Preset: {self.preset}    Prompt profile: {self.prompt_profile}"
            ),
            f"Jobs: {self.jobs}    Artifacts: {self.artifacts_dir}",
            "",
            f"Progress  {_format_progress_bar(self.completed, self.total)}  {self.completed}/{self.total} ({percent:5.1f}%)",
            (
                f"Successes: {self.successes}    Failures: {self.failures}    "
                f"Active: {len(self.current_cases)}/{self.jobs}    Elapsed: {_format_elapsed(time.monotonic() - self.started_at)}"
            ),
        ]
        if self.current_cases:
            lines.append("Current cases:")
            lines.extend(f"  - {case_id}" for case_id in self.current_cases[: min(5, len(self.current_cases))])
        elif self.last_case_id:
            lines.append(f"Last completed: {self.last_case_id} ({self.last_status})")
        if final:
            lines.extend(
                [
                    "",
                    f"Summary: {self.summary_path}" if self.summary_path is not None else "",
                    (
                        f"Warnings: {self.warnings_path}"
                        if self.warnings_path is not None
                        else "Warnings: none recorded"
                    ),
                ]
            )
        return [line for line in lines if line != "" or final]

    def _render(self, final: bool = False) -> None:
        with self._lock:
            lines = self._lines(final=final)
            if self._isatty:
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write("\n".join(lines))
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                print("\n".join(lines), file=sys.stdout, flush=True)


@dataclass
class TerminalUI:
    settings: Settings
    doctor_command: ActionCallback
    run_benchmark: ActionCallback
    run_project: ActionCallback
    summarize_command: ActionCallback
    failures_command: ActionCallback
    modules_command: ActionCallback
    ensure_smoke_subset: Callable[..., Path]
    output: Callable[[str], None] = print
    input_fn: Callable[[str], str] = input

    def run(self) -> int:
        while True:
            self._clear()
            self._print_header()
            self._print_menu()
            choice = self.input_fn("\nSelect an option: ").strip().lower()
            if choice in {"q", "quit", "8"}:
                self.output("\nExiting APD UI.")
                return 0
            if choice == "1":
                self._pause_after(self.doctor_command(self.settings, None))
            elif choice == "2":
                self._pause_after(self._run_smoke())
            elif choice == "3":
                self._pause_after(self._run_full())
            elif choice == "4":
                self._pause_after(self._run_project_solve())
            elif choice == "5":
                self._pause_after(self._run_summary())
            elif choice == "6":
                self._pause_after(self._run_failures())
            elif choice == "7":
                self._pause_after(self._run_modules())
            elif choice == "p":
                self._choose_preset()
            elif choice == "t":
                self.settings.trace_llm = not self.settings.trace_llm
            else:
                self.output("\nInvalid choice.")
                self._wait()

    def _run_smoke(self) -> int:
        jobs = self._prompt_int("Jobs", 1)
        run_id = self._prompt_optional("Run ID", "")
        self.ensure_smoke_subset(self.settings, None, "smoke30", notify=False)
        dashboard = TerminalBenchmarkDashboard()
        return self.run_benchmark(
            self.settings,
            None,
            "smoke30",
            False,
            run_id or None,
            jobs,
            observer=dashboard,
            notify_paths=False,
        )

    def _run_full(self) -> int:
        jobs = self._prompt_int("Jobs", 1)
        run_id = self._prompt_optional("Run ID", "")
        dashboard = TerminalBenchmarkDashboard()
        return self.run_benchmark(
            self.settings,
            None,
            None,
            True,
            run_id or None,
            jobs,
            observer=dashboard,
            notify_paths=False,
        )

    def _run_project_solve(self) -> int:
        project_path = self._prompt_required("Project path")
        validation = self._prompt_optional("Validation command override", "")
        run_id = self._prompt_optional("Run ID", "")
        return self.run_project(self.settings, project_path, validation or None, run_id or None)

    def _run_summary(self) -> int:
        run_id = self._prompt_required("Run ID")
        return self.summarize_command(self.settings, run_id)

    def _run_failures(self) -> int:
        run_id = self._prompt_required("Run ID")
        category = self._prompt_optional("Failure category filter", "")
        limit = self._prompt_int("Limit", 10)
        return self.failures_command(self.settings, run_id, category or None, limit)

    def _run_modules(self) -> int:
        run_id = self._prompt_required("Run ID")
        top = self._prompt_int("Top modules", 15)
        grouping = self._prompt_choice("Grouping", ["canonical", "raw"], self.settings.default_module_grouping)
        return self.modules_command(self.settings, run_id, top, None, grouping)

    def _choose_preset(self) -> None:
        options = list(PRESET_CONFIGS)
        self.output("\nAvailable presets:")
        for index, preset in enumerate(options, start=1):
            marker = "*" if preset == self.settings.preset else " "
            self.output(f"  {index}. [{marker}] {preset}")
        choice = self.input_fn("Choose preset number: ").strip()
        if not choice.isdigit():
            return
        index = int(choice) - 1
        if index < 0 or index >= len(options):
            return
        selected = options[index]
        preset_config = PRESET_CONFIGS[selected]
        self.settings.preset = selected
        self.settings.prompt_profile = preset_config.prompt_profile
        self.settings.max_attempts = preset_config.max_attempts
        self.settings.default_module_grouping = preset_config.reporting_grouping

    def _print_header(self) -> None:
        width = max(72, min(shutil.get_terminal_size((90, 20)).columns, 110))
        title = "Agentic Python Dependency"
        self.output("=" * width)
        self.output(title.center(width))
        self.output("=" * width)
        self.output(f"Preset: {self.settings.preset}")
        self.output(f"Prompt profile: {self.settings.prompt_profile}")
        self.output(f"Trace LLM: {'on' if self.settings.trace_llm else 'off'}")
        self.output(f"Ollama: {self.settings.ollama_base_url}")
        self.output(f"Artifacts: {self.settings.artifacts_dir}")

    def _print_menu(self) -> None:
        self.output("\nActions")
        self.output("  1. Doctor")
        self.output("  2. Smoke benchmark")
        self.output("  3. Full benchmark")
        self.output("  4. Solve local project")
        self.output("  5. Summarize run")
        self.output("  6. Failure report")
        self.output("  7. Module report")
        self.output("  P. Change preset")
        self.output("  T. Toggle LLM tracing")
        self.output("  8. Quit")

    def _prompt_required(self, label: str) -> str:
        while True:
            value = self.input_fn(f"{label}: ").strip()
            if value:
                return value
            self.output("Value required.")

    def _prompt_optional(self, label: str, default: str) -> str:
        value = self.input_fn(f"{label} [{default}]: ").strip()
        return value or default

    def _prompt_int(self, label: str, default: int) -> int:
        value = self.input_fn(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default

    def _prompt_choice(self, label: str, choices: list[str], default: str) -> str:
        value = self.input_fn(f"{label} ({'/'.join(choices)}) [{default}]: ").strip().lower()
        return value if value in choices else default

    def _pause_after(self, exit_code: int) -> None:
        self.output(f"\nCommand finished with exit code {exit_code}.")
        self._wait()

    def _wait(self) -> None:
        self.input_fn("Press Enter to continue...")

    def _clear(self) -> None:
        if sys.stdout.isatty():
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()


def launch_terminal_ui(
    settings: Settings,
    doctor_command: ActionCallback,
    run_benchmark: ActionCallback,
    run_project: ActionCallback,
    summarize_command: ActionCallback,
    failures_command: ActionCallback,
    modules_command: ActionCallback,
    ensure_smoke_subset: Callable[..., Path],
) -> int:
    ui = TerminalUI(
        settings=settings,
        doctor_command=doctor_command,
        run_benchmark=run_benchmark,
        run_project=run_project,
        summarize_command=summarize_command,
        failures_command=failures_command,
        modules_command=modules_command,
        ensure_smoke_subset=ensure_smoke_subset,
    )
    return ui.run()
