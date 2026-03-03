from __future__ import annotations

import contextlib
import io
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

from agentic_python_dependency.config import Settings
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

    def finish(self, *, summary_path: Path, warnings_path: Path | None) -> None:
        self.summary_path = summary_path
        self.warnings_path = warnings_path
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.refresh_interval + 0.1)
        with self._lock:
            self.completed = self.total
            self.current_cases.clear()
        self._refresh()
        if self._app is not None:
            self._app.exit(result=None)
        if self._app_thread is not None:
            self._app_thread.join(timeout=1.0)
        if not self._isatty:
            self._render_text(final=True)

    def _refresh(self) -> None:
        if self._isatty:
            if self._app is not None:
                self._app.invalidate()
        else:
            self._render_text()

    def _start_prompt_toolkit_app(self) -> None:
        bindings = KeyBindings()

        @bindings.add("c-c")
        def _ignore_interrupt(event) -> None:
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
            ("class:label", "Preset       "), ("class:value", f"{self.preset}\n"),
            ("class:label", "Prompt       "), ("class:value", f"{self.prompt_profile}\n"),
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
            f"Preset: {self.preset}",
            f"Prompt profile: {self.prompt_profile}",
            f"Jobs: {self.jobs}",
            f"Artifacts: {self.artifacts_dir}",
            "",
            f"Progress: {_format_progress_bar(self.completed, self.total)} {self.completed}/{self.total} ({percent:5.1f}%)",
            f"Successes: {self.successes}    Failures: {self.failures}    Elapsed: {_format_elapsed(time.monotonic() - self.started_at)}",
        ]
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
    output: Callable[[str], None] = print
    input_fn: Callable[[str], str] = input

    def __post_init__(self) -> None:
        self._use_prompt_toolkit = (
            self.input_fn is input and self.output is print and sys.stdin.isatty() and sys.stdout.isatty()
        )

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
                    ("Solve local project", "4"),
                    ("Summarize run", "5"),
                    ("Failure report", "6"),
                    ("Module report", "7"),
                    ("Preset", "p"),
                    ("Trace", "t"),
                    ("Quit", "8"),
                ],
                style=UI_STYLE,
            ).run()
            if choice in {None, "8"}:
                message_dialog(title="APD", text="Exiting APD UI.", style=UI_STYLE).run()
                return 0
            exit_code = self._dispatch_choice(choice)
            if choice in {"2", "3"}:
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
            if choice not in {"p", "t"}:
                self._pause_after(exit_code)

    def _dispatch_choice(self, choice: str | None) -> int:
        if choice == "1":
            return self._run_captured(self.doctor_command, self.settings, None)
        if choice == "2":
            return self._run_smoke()
        if choice == "3":
            return self._run_full()
        if choice == "4":
            return self._run_project_solve()
        if choice == "5":
            return self._run_summary()
        if choice == "6":
            return self._run_failures()
        if choice == "7":
            return self._run_modules()
        if choice == "p":
            self._choose_preset()
            return 0
        if choice == "t":
            self.settings.trace_llm = not self.settings.trace_llm
            self._show_status_dialog(f"LLM tracing is now {'on' if self.settings.trace_llm else 'off'}.")
            return 0
        self._show_status_dialog("Invalid choice.")
        return 0

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
        return self._run_captured(self.run_project, self.settings, project_path, validation or None, run_id or None)

    def _run_summary(self) -> int:
        run_id = self._prompt_required("Run ID")
        return self._run_captured(self.summarize_command, self.settings, run_id)

    def _run_failures(self) -> int:
        run_id = self._prompt_required("Run ID")
        category = self._prompt_optional("Failure category filter", "")
        limit = self._prompt_int("Limit", 10)
        return self._run_captured(self.failures_command, self.settings, run_id, category or None, limit)

    def _run_modules(self) -> int:
        run_id = self._prompt_required("Run ID")
        top = self._prompt_int("Top modules", 15)
        grouping = self._prompt_choice("Grouping", ["canonical", "raw"], self.settings.default_module_grouping)
        return self._run_captured(self.modules_command, self.settings, run_id, top, None, grouping)

    def _run_captured(self, action: ActionCallback, *args) -> int:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exit_code = action(*args)
        payload = buffer.getvalue().strip()
        if payload:
            self._show_status_dialog(payload)
        return exit_code

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
        self._show_status_dialog(f"Preset switched to {selected}.")

    def _menu_dialog_text(self) -> AnyFormattedText:
        return HTML(
            "<b><ansibrightyellow>APD Command Center</ansibrightyellow></b>\n"
            "<style fg='#98c1d9'>Run benchmarks, inspect reports, and solve local projects without memorizing subcommands.</style>\n\n"
            f"<b>Preset:</b> {self.settings.preset}\n"
            f"<b>Prompt profile:</b> {self.settings.prompt_profile}\n"
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
