from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML, AnyFormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.shortcuts import button_dialog, input_dialog, message_dialog, radiolist_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Box, Frame

from agentic_python_dependency.config import MODEL_PROFILE_DEFAULTS, Settings
from agentic_python_dependency.presets import PRESET_CONFIGS


ActionCallback = Callable[..., int]

UI_STYLE = Style.from_dict(
    {
        "dialog": "bg:#0b1321 #f1f5f9",
        "dialog frame.label": "bg:#fb8500 #0b1321 bold",
        "dialog.body": "bg:#0b1321 #f1f5f9",
        "button": "bg:#1d3557 #f1f5f9",
        "button.focused": "bg:#2a9d8f #081c15 bold",
        "shadow": "bg:#000000",
    }
)

DASHBOARD_STYLE = Style.from_dict(
    {
        "frame.border": "#5f6c7b",
        "frame.label": "bg:#ee6c4d #08121c bold",
        "headline": "#fb8500 bold",
        "muted": "#94a3b8",
        "label": "#98c1d9 bold",
        "value": "#f1f5f9",
        "good": "#2ec4b6 bold",
        "bad": "#ef476f bold",
        "accent": "#ffd166 bold",
        "bar.complete": "bg:#2a9d8f #2a9d8f",
        "bar.remaining": "bg:#1f2937 #1f2937",
    }
)


def _format_progress_bar(completed: int, total: int, width: int = 36) -> str:
    if total <= 0:
        return "█" * width
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return ("█" * filled) + ("░" * (width - filled))


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
        self.resolver = "apd"
        self.preset = "optimized"
        self.prompt_profile = "optimized"
        self.model_summary = "gemma-moe: gemma3:4b / gemma3:12b"
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
        self._cancel_requested = False
        self._thread: threading.Thread | None = None
        self._app_thread: threading.Thread | None = None
        self._app: Application[None] | None = None
        self._isatty = sys.stdin.isatty() and sys.stdout.isatty()

    def start(
        self,
        *,
        run_id: str,
        total: int,
        completed: int,
        successes: int,
        failures: int,
        resolver: str,
        preset: str,
        prompt_profile: str,
        model_summary: str,
        jobs: int,
        target: str,
        artifacts_dir: Path,
        elapsed_seconds: float = 0.0,
    ) -> None:
        self.run_id = run_id
        self.total = total
        self.completed = completed
        self.successes = successes
        self.failures = failures
        self.resolver = resolver
        self.preset = preset
        self.prompt_profile = prompt_profile
        self.model_summary = model_summary
        self.jobs = jobs
        self.target = target
        self.artifacts_dir = artifacts_dir
        self.started_at = time.monotonic() - elapsed_seconds
        if self._isatty:
            self._start_prompt_toolkit_app()
        else:
            self._render_text()

        def _refresh_loop() -> None:
            while not self._stop_event.wait(self.refresh_interval):
                self._refresh()

        self._thread = threading.Thread(target=_refresh_loop, name="apd-ui-benchmark-refresh", daemon=True)
        self._thread.start()

    def case_started(self, case_id: str) -> None:
        with self._lock:
            if case_id not in self.current_cases:
                self.current_cases.append(case_id)
        self._refresh()

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
        self._refresh()

    def finish(self, *, summary_path: Path, warnings_path: Path | None, status: str = "completed") -> None:
        self.summary_path = summary_path
        self.warnings_path = warnings_path
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.refresh_interval + 0.1)
        with self._lock:
            if status == "completed":
                self.completed = self.total
            self.current_cases.clear()
        self._refresh()
        if self._app is not None:
            self._app.exit(result=None)
        if self._app_thread is not None:
            self._app_thread.join(timeout=1.0)
        if not self._isatty:
            self._render_text(final=True)

    def request_stop(self) -> None:
        with self._lock:
            self._cancel_requested = True
        self._refresh()

    def stop_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def _refresh(self) -> None:
        if self._isatty:
            if self._app is not None:
                self._app.invalidate()
        else:
            self._render_text()

    def _start_prompt_toolkit_app(self) -> None:
        bindings = KeyBindings()

        @bindings.add("c-c")
        def _request_stop(event) -> None:
            self.request_stop()
            event.app.invalidate()

        control = FormattedTextControl(self._formatted_text, focusable=False)
        body = Box(
            body=Frame(
                body=Window(content=control, always_hide_cursor=True, wrap_lines=False),
                title="APD Benchmark Dashboard",
                style="class:frame",
            ),
            padding=1,
        )
        self._app = Application(
            layout=Layout(HSplit([body])),
            full_screen=True,
            mouse_support=False,
            style=DASHBOARD_STYLE,
            key_bindings=bindings,
        )

        def _runner() -> None:
            assert self._app is not None
            self._app.run(set_exception_handler=False)

        self._app_thread = threading.Thread(target=_runner, name="apd-ui-benchmark-app", daemon=True)
        self._app_thread.start()

    def _formatted_text(self) -> AnyFormattedText:
        width = max(72, min(shutil.get_terminal_size((100, 24)).columns - 4, 116))
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        bar = _format_progress_bar(self.completed, self.total)
        fragments: list[tuple[str, str]] = [
            ("class:headline", "APD benchmark in progress\n"),
            ("class:muted", "Use Ctrl+C only if you intend to stop the benchmark process itself.\n\n"),
            ("class:label", "Run ID       "), ("class:value", f"{self.run_id}\n"),
            ("class:label", "Target       "), ("class:value", f"{self.target}\n"),
            ("class:label", "Resolver     "), ("class:value", f"{self.resolver}\n"),
            ("class:label", "Preset       "), ("class:value", f"{self.preset}\n"),
            ("class:label", "Prompt       "), ("class:value", f"{self.prompt_profile}\n"),
            ("class:label", "Models       "), ("class:value", f"{getattr(self, 'model_summary', 'default')}\n"),
            ("class:label", "Jobs         "), ("class:value", f"{self.jobs}\n"),
            ("class:label", "Artifacts    "), ("class:value", f"{self.artifacts_dir}\n\n"),
            ("class:label", "Progress     "), ("class:accent", f"{self.completed}/{self.total} ({percent:5.1f}%)\n"),
            ("class:bar.complete", bar[: int((percent / 100.0) * len(bar))]),
            ("class:bar.remaining", bar[int((percent / 100.0) * len(bar)):]),
            ("", "\n\n"),
            ("class:good", f"Successes: {self.successes}"),
            ("", "    "),
            ("class:bad", f"Failures: {self.failures}"),
            ("", "    "),
            ("class:accent", f"Elapsed: {_format_elapsed(time.monotonic() - self.started_at)}\n"),
        ]
        if self._cancel_requested:
            fragments.extend(
                [
                    ("class:bad", "\nStop requested\n"),
                    ("class:muted", "APD will stop scheduling new cases and exit after the active work finishes.\n"),
                ]
            )
        if self.current_cases:
            fragments.append(("class:label", "\nActive cases\n"))
            for case_id in self.current_cases[: min(6, len(self.current_cases))]:
                fragments.append(("class:value", f"  • {case_id}\n"))
        elif self.last_case_id:
            fragments.extend(
                [
                    ("class:label", "\nLast completed\n"),
                    ("class:value", f"  • {self.last_case_id} ({self.last_status})\n"),
                ]
            )
        if self.summary_path is not None:
            fragments.extend(
                [
                    ("class:label", "\nSummary\n"),
                    ("class:value", f"  • {self.summary_path}\n"),
                    ("class:label", "Warnings\n"),
                    ("class:value", f"  • {self.warnings_path or 'none recorded'}\n"),
                ]
            )
        trailing = max(0, width - 1)
        fragments.append(("", " " * trailing))
        return fragments

    def _render_text(self, final: bool = False) -> None:
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        lines = [
            "=" * 80,
            "APD Benchmark Dashboard",
            "=" * 80,
            f"Run ID: {self.run_id}",
            f"Target: {self.target}",
            f"Resolver: {self.resolver}",
            f"Preset: {self.preset}",
            f"Prompt profile: {self.prompt_profile}",
            f"Models: {getattr(self, 'model_summary', 'default')}",
            f"Jobs: {self.jobs}",
            f"Artifacts: {self.artifacts_dir}",
            "",
            f"Progress: {_format_progress_bar(self.completed, self.total)} {self.completed}/{self.total} ({percent:5.1f}%)",
            f"Successes: {self.successes}    Failures: {self.failures}    Elapsed: {_format_elapsed(time.monotonic() - self.started_at)}",
        ]
        if self._cancel_requested:
            lines.append("Stop requested: APD will stop after the current active cases finish.")
        if self.current_cases:
            lines.append("Active cases:")
            lines.extend(f"  - {case_id}" for case_id in self.current_cases[: min(6, len(self.current_cases))])
        elif self.last_case_id:
            lines.append(f"Last completed: {self.last_case_id} ({self.last_status})")
        if final:
            lines.extend(
                [
                    "",
                    f"Summary: {self.summary_path}",
                    f"Warnings: {self.warnings_path or 'none recorded'}",
                ]
            )
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
    timeline_command: ActionCallback | None = None
    output: Callable[[str], None] = print
    input_fn: Callable[[str], str] = input

    def __post_init__(self) -> None:
        self._use_prompt_toolkit = (
            self.input_fn is input and self.output is print and sys.stdin.isatty() and sys.stdout.isatty()
        )
        self._fresh_run = False

    def run(self) -> int:
        if self._use_prompt_toolkit:
            return self._run_prompt_toolkit()
        return self._run_basic()

    def _run_prompt_toolkit(self) -> int:
        while True:
            choice = button_dialog(
                title="APD Command Center",
                text=self._menu_dialog_text(),
                buttons=[
                    ("Doctor", "1"),
                    ("Smoke benchmark", "2"),
                    ("Full benchmark", "3"),
                    ("Resume benchmark", "u"),
                    ("Solve local project", "4"),
                    ("Summarize run", "5"),
                    ("Failure report", "6"),
                    ("Module report", "7"),
                    ("Timeline view", "l"),
                    ("Resolver", "v"),
                    ("Preset", "p"),
                    ("Models", "m"),
                    ("Runtime", "r"),
                    ("Fresh run", "f"),
                    ("Trace", "t"),
                    ("Quit", "8"),
                ],
                style=UI_STYLE,
            ).run()
            if choice in {None, "8"}:
                message_dialog(title="APD", text="Exiting APD UI.", style=UI_STYLE).run()
                return 0
            exit_code = self._dispatch_choice(choice)
            if choice in {"2", "3", "u"}:
                self._show_status_dialog(f"Command finished with exit code {exit_code}.")

    def _run_basic(self) -> int:
        while True:
            self._clear()
            self._print_header()
            self._print_menu()
            choice = self.input_fn("\nSelect an option: ").strip().lower()
            if choice in {"q", "quit", "8"}:
                self.output("\nExiting APD UI.")
                return 0
            exit_code = self._dispatch_choice(choice)
            if choice not in {"v", "p", "m", "r", "f", "t"}:
                self._pause_after(exit_code)

    def _dispatch_choice(self, choice: str | None) -> int:
        if choice == "1":
            return self._run_captured(self.doctor_command, self.settings, None)
        if choice == "2":
            return self._run_smoke()
        if choice == "3":
            return self._run_full()
        if choice == "u":
            return self._run_resume_benchmark()
        if choice == "4":
            return self._run_project_solve()
        if choice == "5":
            return self._run_summary()
        if choice == "6":
            return self._run_failures()
        if choice == "7":
            return self._run_modules()
        if choice == "l":
            return self._run_timeline()
        if choice == "v":
            self._choose_resolver()
            return 0
        if choice == "p":
            self._choose_preset()
            return 0
        if choice == "m":
            self._choose_model_profile()
            return 0
        if choice == "r":
            self._configure_runtime()
            return 0
        if choice == "f":
            self._fresh_run = not self._fresh_run
            self._show_status_dialog(f"Fresh run is now {'on' if self._fresh_run else 'off'}.")
            return 0
        if choice == "t":
            self.settings.trace_llm = not self.settings.trace_llm
            self._show_status_dialog(f"LLM tracing is now {'on' if self.settings.trace_llm else 'off'}.")
            return 0
        self._show_status_dialog("Invalid choice.")
        return 0

    def _run_smoke(self) -> int:
        if not self._validate_runtime_selection():
            return 1
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
            fresh_run=self._fresh_run,
        )

    def _run_full(self) -> int:
        if not self._validate_runtime_selection():
            return 1
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
            fresh_run=self._fresh_run,
        )

    def _run_resume_benchmark(self) -> int:
        run_entry = self._prompt_resume_run()
        if run_entry is None:
            return 1
        saved_resolver = str(run_entry.get("resolver", "") or "")
        if saved_resolver:
            self.settings.resolver = saved_resolver
        saved_preset = str(run_entry.get("preset", "") or "")
        if saved_preset in PRESET_CONFIGS:
            preset_config = PRESET_CONFIGS[saved_preset]
            self.settings.preset = saved_preset
            self.settings.prompt_profile = preset_config.prompt_profile
            self.settings.max_attempts = preset_config.max_attempts
            self.settings.default_module_grouping = preset_config.reporting_grouping
        if not self._validate_runtime_selection():
            return 1
        target = str(run_entry.get("target", "benchmark") or "benchmark")
        jobs = int(run_entry.get("jobs", 1) or 1)
        run_id = str(run_entry["run_id"])
        if target == "smoke30":
            self.ensure_smoke_subset(self.settings, None, "smoke30", notify=False)
        dashboard = TerminalBenchmarkDashboard()
        return self.run_benchmark(
            self.settings,
            None,
            None if target in {"full", "benchmark"} else target,
            target == "full",
            run_id,
            jobs,
            observer=dashboard,
            notify_paths=False,
            fresh_run=False,
        )

    def _run_project_solve(self) -> int:
        if not self._validate_runtime_selection():
            return 1
        project_path = self._prompt_required("Project path")
        validation = self._prompt_optional("Validation command override", "")
        run_id = self._prompt_optional("Run ID", "")
        return self._run_captured(
            self.run_project,
            self.settings,
            project_path,
            validation or None,
            run_id or None,
            self._fresh_run,
        )

    def _run_summary(self) -> int:
        run_id = self._prompt_run_id()
        if not run_id:
            return 1
        return self._run_captured(self.summarize_command, self.settings, run_id)

    def _run_failures(self) -> int:
        run_id = self._prompt_run_id()
        if not run_id:
            return 1
        category = self._prompt_optional("Failure category filter", "")
        limit = self._prompt_int("Limit", 10)
        return self._run_captured(self.failures_command, self.settings, run_id, category or None, limit)

    def _run_modules(self) -> int:
        run_id = self._prompt_run_id()
        if not run_id:
            return 1
        top = self._prompt_int("Top modules", 15)
        grouping = self._prompt_choice("Grouping", ["canonical", "raw"], self.settings.default_module_grouping)
        cohort = self._prompt_choice("Cohort", ["run", "paper-compatible"], "run")
        report_path = self._module_report_path(run_id, grouping, cohort == "paper-compatible")
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exit_code = self.modules_command(
                    self.settings,
                    run_id,
                    top,
                    None,
                    grouping,
                    cohort == "paper-compatible",
                )
        except Exception as exc:
            payload = buffer.getvalue().strip()
            message = f"{type(exc).__name__}: {exc}"
            if payload:
                message = f"{payload}\n\n{message}"
            self._show_status_dialog(message)
            return 1
        rendered = report_path.read_text(encoding="utf-8") if report_path.exists() else buffer.getvalue().strip()
        self._show_status_dialog(rendered or "Module report completed.")
        return exit_code

    def _run_timeline(self) -> int:
        if self.timeline_command is None:
            self._show_status_dialog("Timeline command is unavailable.")
            return 1
        run_id = self._prompt_run_id()
        if not run_id:
            return 1
        return self._run_captured(self.timeline_command, self.settings, run_id)

    def _run_captured(self, action: ActionCallback, *args) -> int:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exit_code = action(*args)
        except Exception as exc:
            payload = buffer.getvalue().strip()
            message = f"{type(exc).__name__}: {exc}"
            if payload:
                message = f"{payload}\n\n{message}"
            self._show_status_dialog(message)
            return 1
        payload = buffer.getvalue().strip()
        if payload:
            self._show_status_dialog(payload)
        return exit_code

    def _module_report_path(self, run_id: str, grouping: str, paper_compatible: bool) -> Path:
        suffix_parts: list[str] = []
        if paper_compatible:
            suffix_parts.append("paper")
        if grouping == "raw":
            suffix_parts.append("raw")
        suffix = "" if not suffix_parts else "-" + "-".join(suffix_parts)
        return self.settings.artifacts_dir / run_id / f"module-success{suffix}.md"

    def _available_run_ids(self) -> list[str]:
        if not self.settings.artifacts_dir.exists():
            return []
        run_dirs = [path for path in self.settings.artifacts_dir.iterdir() if path.is_dir()]
        run_dirs.sort(key=lambda path: (-path.stat().st_mtime, path.name))
        return [path.name for path in run_dirs]

    def _read_run_state(self, run_id: str) -> dict[str, object]:
        run_state_path = self.settings.artifacts_dir / run_id / "run-state.json"
        if not run_state_path.exists():
            return {}
        try:
            payload = json.loads(run_state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _available_run_entries(self, *, resumable_only: bool = False) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for run_id in self._available_run_ids():
            state = self._read_run_state(run_id)
            completed = int(state.get("completed", 0) or 0)
            total = int(state.get("total", 0) or 0)
            status = str(state.get("status", "unknown") or "unknown")
            resumable = status in {"running", "stopping", "paused", "interrupted"} or (total > 0 and completed < total)
            if resumable_only and not resumable:
                continue
            target = str(state.get("target", "benchmark") or "benchmark")
            jobs = int(state.get("jobs", 1) or 1)
            resolver = str(state.get("resolver", "apd") or "apd")
            preset = str(state.get("preset", "optimized") or "optimized")
            label = f"{run_id} [{status}] {completed}/{total} target={target} jobs={jobs} resolver={resolver} preset={preset}"
            entries.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "status": status,
                    "completed": completed,
                    "total": total,
                    "target": target,
                    "jobs": jobs,
                    "resolver": resolver,
                    "preset": preset,
                }
            )
        return entries

    def _validate_runtime_selection(self) -> bool:
        if self.settings.preset == "experimental" and self.settings.resolver != "apd":
            self._show_status_dialog("The experimental preset is only supported with the apd resolver.")
            return False
        return True

    def _prompt_run_id(self) -> str | None:
        run_entries = self._available_run_entries()
        if not run_entries:
            self._show_status_dialog(f"No run directories found in {self.settings.artifacts_dir}.")
            return None
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title="Select run",
                text="Choose a run directory from artifacts/runs.",
                values=[(entry["run_id"], str(entry["label"])) for entry in run_entries],
                default=str(run_entries[0]["run_id"]),
                style=UI_STYLE,
            ).run()
            return selected

        self.output("\nAvailable runs:")
        for index, entry in enumerate(run_entries, start=1):
            self.output(f"  {index}. {entry['label']}")
        choice = self.input_fn("Choose run number: ").strip()
        if not choice.isdigit():
            self._show_status_dialog("Invalid run selection.")
            return None
        index = int(choice) - 1
        if index < 0 or index >= len(run_entries):
            self._show_status_dialog("Invalid run selection.")
            return None
        return str(run_entries[index]["run_id"])

    def _prompt_resume_run(self) -> dict[str, object] | None:
        run_entries = self._available_run_entries(resumable_only=True)
        if not run_entries:
            self._show_status_dialog("No resumable benchmark runs found.")
            return None
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title="Resume benchmark",
                text="Choose a saved benchmark run to resume.",
                values=[(str(entry["run_id"]), str(entry["label"])) for entry in run_entries],
                default=str(run_entries[0]["run_id"]),
                style=UI_STYLE,
            ).run()
            if selected is None:
                return None
            return next((entry for entry in run_entries if entry["run_id"] == selected), None)

        self.output("\nResumable runs:")
        for index, entry in enumerate(run_entries, start=1):
            self.output(f"  {index}. {entry['label']}")
        choice = self.input_fn("Choose run number: ").strip()
        if not choice.isdigit():
            self._show_status_dialog("Invalid run selection.")
            return None
        index = int(choice) - 1
        if index < 0 or index >= len(run_entries):
            self._show_status_dialog("Invalid run selection.")
            return None
        return run_entries[index]

    def _choose_preset(self) -> None:
        options = [(preset, preset) for preset in PRESET_CONFIGS]
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title="Select preset",
                text="Choose the tradeoff profile for the next runs.",
                values=options,
                default=self.settings.preset,
                style=UI_STYLE,
            ).run()
            if selected is None:
                return
        else:
            self.output("\nAvailable presets:")
            for index, preset in enumerate(PRESET_CONFIGS, start=1):
                marker = "*" if preset == self.settings.preset else " "
                self.output(f"  {index}. [{marker}] {preset}")
            choice = self.input_fn("Choose preset number: ").strip()
            if not choice.isdigit():
                return
            index = int(choice) - 1
            if index < 0 or index >= len(PRESET_CONFIGS):
                return
            selected = list(PRESET_CONFIGS)[index]
        preset_config = PRESET_CONFIGS[selected]
        self.settings.preset = selected
        self.settings.prompt_profile = preset_config.prompt_profile
        self.settings.max_attempts = preset_config.max_attempts
        self.settings.default_module_grouping = preset_config.reporting_grouping
        if selected == "experimental" and self.settings.resolver != "apd":
            self.settings.resolver = "apd"
        self._show_status_dialog(f"Preset switched to {selected}.")

    def _choose_resolver(self) -> None:
        options = [("apd", "apd"), ("pyego", "pyego"), ("readpye", "readpye")]
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title="Select resolver",
                text="Choose the dependency-resolution strategy for the next runs.",
                values=options,
                default=self.settings.resolver,
                style=UI_STYLE,
            ).run()
            if selected is None:
                return
        else:
            self.output("\nAvailable resolvers:")
            visible_resolvers = [resolver for resolver, _ in options]
            for index, resolver in enumerate(visible_resolvers, start=1):
                marker = "*" if resolver == self.settings.resolver else " "
                self.output(f"  {index}. [{marker}] {resolver}")
            choice = self.input_fn("Choose resolver number: ").strip()
            if not choice.isdigit():
                return
            index = int(choice) - 1
            if index < 0 or index >= len(visible_resolvers):
                return
            selected = visible_resolvers[index]
        self.settings.resolver = selected
        if selected != "apd" and self.settings.preset == "experimental":
            self.settings.preset = "accuracy"
            preset_config = PRESET_CONFIGS["accuracy"]
            self.settings.prompt_profile = preset_config.prompt_profile
            self.settings.max_attempts = preset_config.max_attempts
            self.settings.default_module_grouping = preset_config.reporting_grouping
        self._show_status_dialog(f"Resolver switched to {selected}.")

    def _choose_model_profile(self) -> None:
        options = [(profile, profile) for profile in MODEL_PROFILE_DEFAULTS if profile != "custom"]
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title="Select model bundle",
                text="Choose the Ollama model bundle for the next runs.",
                values=options,
                default=self.settings.model_profile if self.settings.model_profile != "custom" else "gemma-moe",
                style=UI_STYLE,
            ).run()
            if selected is None:
                return
        else:
            self.output("\nAvailable model bundles:")
            visible_profiles = [profile for profile in MODEL_PROFILE_DEFAULTS if profile != "custom"]
            for index, profile in enumerate(visible_profiles, start=1):
                marker = "*" if profile == self.settings.model_profile else " "
                self.output(f"  {index}. [{marker}] {profile}")
            choice = self.input_fn("Choose model bundle number: ").strip()
            if not choice.isdigit():
                return
            index = int(choice) - 1
            if index < 0 or index >= len(visible_profiles):
                return
            selected = visible_profiles[index]
        extraction_model, reasoning_model = MODEL_PROFILE_DEFAULTS[selected]
        self.settings.model_profile = selected
        self.settings.extraction_model = extraction_model
        self.settings.reasoning_model = reasoning_model
        self.settings.version_model = reasoning_model
        self.settings.repair_model = reasoning_model
        self.settings.adjudication_model = reasoning_model
        self._show_status_dialog(
            f"Model bundle switched to {selected} ({extraction_model} / {reasoning_model})."
        )

    def _configure_runtime(self) -> None:
        if self._use_prompt_toolkit:
            while True:
                choice = button_dialog(
                    title="Runtime controls",
                    text=self._runtime_dialog_text(),
                    buttons=[
                        ("Toggle MoE", "moe"),
                        ("Toggle RAG", "rag"),
                        ("Toggle LangChain", "langchain"),
                        ("Extractor", "extract"),
                        ("Runner", "runner"),
                        ("Version", "version"),
                        ("Repair", "repair"),
                        ("Adjudicate", "adjudicate"),
                        ("Back", "back"),
                    ],
                    style=UI_STYLE,
                ).run()
                if choice in {None, "back"}:
                    return
                self._apply_runtime_choice(choice)
        else:
            self.output("\nRuntime controls")
            self.output("  1. Toggle MoE")
            self.output("  2. Toggle RAG")
            self.output("  3. Toggle LangChain")
            self.output("  4. Set extractor model")
            self.output("  5. Set runner model")
            self.output("  6. Set version model")
            self.output("  7. Set repair model")
            self.output("  8. Set adjudication model")
            choice = self.input_fn("Select runtime option: ").strip()
            mapping = {
                "1": "moe",
                "2": "rag",
                "3": "langchain",
                "4": "extract",
                "5": "runner",
                "6": "version",
                "7": "repair",
                "8": "adjudicate",
            }
            selected = mapping.get(choice)
            if selected:
                self._apply_runtime_choice(selected)

    def _apply_runtime_choice(self, choice: str) -> None:
        if choice == "moe":
            self.settings.use_moe = not self.settings.use_moe
            self._show_status_dialog(f"MoE is now {'on' if self.settings.use_moe else 'off'}.")
            return
        if choice == "rag":
            self.settings.use_rag = not self.settings.use_rag
            self._show_status_dialog(f"RAG is now {'on' if self.settings.use_rag else 'off'}.")
            return
        if choice == "langchain":
            self.settings.use_langchain = not self.settings.use_langchain
            self._show_status_dialog(f"LangChain is now {'on' if self.settings.use_langchain else 'off'}.")
            return
        if choice == "extract":
            self.settings.extraction_model = self._prompt_optional("Extractor model", self.settings.extraction_model)
        elif choice == "runner":
            self.settings.reasoning_model = self._prompt_optional("Runner model", self.settings.reasoning_model)
        elif choice == "version":
            self.settings.version_model = self._prompt_optional("Version model", self.settings.version_model)
        elif choice == "repair":
            self.settings.repair_model = self._prompt_optional("Repair model", self.settings.repair_model)
        elif choice == "adjudicate":
            self.settings.adjudication_model = self._prompt_optional(
                "Adjudication model",
                self.settings.adjudication_model,
            )
        else:
            return
        self.settings.model_profile = "custom"
        self._show_status_dialog(self._runtime_dialog_text().replace("\n", "\n"))

    def _runtime_dialog_text(self) -> str:
        return (
            f"MoE: {'on' if self.settings.use_moe else 'off'}\n"
            f"RAG: {'on' if self.settings.use_rag else 'off'}\n"
            f"LangChain: {'on' if self.settings.use_langchain else 'off'}\n\n"
            f"Extractor: {self.settings.extraction_model}\n"
            f"Runner: {self.settings.reasoning_model}\n"
            f"Version: {self.settings.version_model}\n"
            f"Repair: {self.settings.repair_model}\n"
            f"Adjudication: {self.settings.adjudication_model}"
        )

    def _menu_dialog_text(self) -> AnyFormattedText:
        return HTML(
            "<b><ansibrightyellow>APD Command Center</ansibrightyellow></b>\n"
            "<style fg='#98c1d9'>Run benchmarks, inspect reports, and solve local projects without memorizing subcommands.</style>\n\n"
            f"<b>Preset:</b> {self.settings.preset}\n"
            f"<b>Resolver:</b> {self.settings.resolver}\n"
            f"<b>Model bundle:</b> {self.settings.model_profile}\n"
            f"<b>MoE:</b> {'on' if self.settings.use_moe else 'off'}\n"
            f"<b>RAG:</b> {'on' if self.settings.use_rag else 'off'}\n"
            f"<b>LangChain:</b> {'on' if self.settings.use_langchain else 'off'}\n"
            f"<b>Models:</b> {self.settings.extraction_model} / {self.settings.reasoning_model}\n"
            f"<b>Version/Repair/Adj:</b> {self.settings.version_model} / {self.settings.repair_model} / {self.settings.adjudication_model}\n"
            f"<b>Prompt profile:</b> {self.settings.prompt_profile}\n"
            f"<b>Fresh run:</b> {'on' if self._fresh_run else 'off'}\n"
            f"<b>Trace LLM:</b> {'on' if self.settings.trace_llm else 'off'}\n"
            f"<b>Ollama:</b> {self.settings.ollama_base_url}\n"
            f"<b>Artifacts:</b> {self.settings.artifacts_dir}"
        )

    def _show_status_dialog(self, text: str) -> None:
        if self._use_prompt_toolkit:
            message_dialog(title="APD", text=text, style=UI_STYLE).run()
        else:
            self.output(f"\n{text}")

    def _prompt_required(self, label: str) -> str:
        while True:
            value = self._prompt_optional(label, "")
            if value:
                return value
            self._show_status_dialog("Value required.")

    def _prompt_optional(self, label: str, default: str) -> str:
        if self._use_prompt_toolkit:
            value = input_dialog(title="APD", text=f"{label} [{default}]", style=UI_STYLE).run()
            if value is None:
                return default
            return value.strip() or default
        value = self.input_fn(f"{label} [{default}]: ").strip()
        return value or default

    def _prompt_int(self, label: str, default: int) -> int:
        raw_value = self._prompt_optional(label, str(default))
        try:
            parsed = int(raw_value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default

    def _prompt_choice(self, label: str, choices: list[str], default: str) -> str:
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title="APD",
                text=label,
                values=[(choice, choice) for choice in choices],
                default=default,
                style=UI_STYLE,
            ).run()
            return selected or default
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

    def _print_header(self) -> None:
        width = max(72, min(shutil.get_terminal_size((90, 20)).columns, 110))
        title = "Agentic Python Dependency"
        self.output("=" * width)
        self.output(title.center(width))
        self.output("=" * width)
        self.output(f"Preset: {self.settings.preset}")
        self.output(f"Resolver: {self.settings.resolver}")
        self.output(f"Model bundle: {self.settings.model_profile}")
        self.output(f"MoE: {'on' if self.settings.use_moe else 'off'}")
        self.output(f"RAG: {'on' if self.settings.use_rag else 'off'}")
        self.output(f"LangChain: {'on' if self.settings.use_langchain else 'off'}")
        self.output(f"Models: {self.settings.extraction_model} / {self.settings.reasoning_model}")
        self.output(
            f"Version/Repair/Adjudication: {self.settings.version_model} / {self.settings.repair_model} / {self.settings.adjudication_model}"
        )
        self.output(f"Prompt profile: {self.settings.prompt_profile}")
        self.output(f"Fresh run: {'on' if self._fresh_run else 'off'}")
        self.output(f"Trace LLM: {'on' if self.settings.trace_llm else 'off'}")
        self.output(f"Ollama: {self.settings.ollama_base_url}")
        self.output(f"Artifacts: {self.settings.artifacts_dir}")

    def _print_menu(self) -> None:
        self.output("\nActions")
        self.output("  1. Doctor")
        self.output("  2. Smoke benchmark")
        self.output("  3. Full benchmark")
        self.output("  U. Resume benchmark")
        self.output("  4. Solve local project")
        self.output("  5. Summarize run")
        self.output("  6. Failure report")
        self.output("  7. Module report")
        self.output("  L. Timeline view")
        self.output("  V. Change resolver")
        self.output("  P. Change preset")
        self.output("  M. Change model bundle")
        self.output("  R. Runtime controls")
        self.output("  F. Toggle fresh run / no LLM cache")
        self.output("  T. Toggle LLM tracing")
        self.output("  8. Quit")


def launch_terminal_ui(
    settings: Settings,
    doctor_command: ActionCallback,
    run_benchmark: ActionCallback,
    run_project: ActionCallback,
    summarize_command: ActionCallback,
    failures_command: ActionCallback,
    modules_command: ActionCallback,
    timeline_command: ActionCallback,
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
        timeline_command=timeline_command,
        ensure_smoke_subset=ensure_smoke_subset,
    )
    return ui.run()
