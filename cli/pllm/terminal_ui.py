from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from cli.pllm.benchmark_data import (
    BenchmarkSource,
    breakdown_summary,
    list_case_ids,
    rebuild_competition_filter,
    resolve_snippet_path,
)
from cli.pllm.benchmark_runner import BenchmarkSummary, run_benchmark
from cli.pllm.core import (
    ROOT_DIR,
    RunConfig,
    RunStats,
    fetch_ollama_models,
    format_doctor_report,
    list_tools,
    run_doctor,
    stream_executor,
)

PROMPT_TOOLKIT_AVAILABLE = True
try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import AnyFormattedText, HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.shortcuts import button_dialog, checkboxlist_dialog, input_dialog, message_dialog, radiolist_dialog
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Box, Frame
except Exception:
    PROMPT_TOOLKIT_AVAILABLE = False


if PROMPT_TOOLKIT_AVAILABLE:
    UI_STYLE = Style.from_dict(
        {
            "dialog": "bg:#0b1321 #f1f5f9",
            "dialog frame.label": "bg:#ffb703 #0b1321 bold",
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
            "headline": "#ffb703 bold",
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


@dataclass
class UIOptions:
    tool: str
    model: str
    base: str
    temp: float
    loop: int
    search_range: int
    rag: bool
    verbose: bool
    source: BenchmarkSource
    benchmark_limit: int
    benchmark_offset: int
    benchmark_fail_fast: bool
    benchmark_show_output: bool


def launch_terminal_ui(
    default_model: str = "gemma2",
    default_base: str = "http://localhost:11434",
    default_loop: int = 10,
    default_range: int = 0,
    default_file: str = "",
    default_benchmark_source: BenchmarkSource = "all-gists",
    default_tool: str = "pllm",
) -> int:
    available_tools = list_tools()
    selected_tool = default_tool if default_tool in available_tools else (available_tools[0] if available_tools else "pllm")
    options = UIOptions(
        tool=selected_tool,
        model=default_model,
        base=default_base,
        temp=0.7,
        loop=max(1, default_loop),
        search_range=max(0, default_range),
        rag=True,
        verbose=False,
        source=_normalize_source(default_benchmark_source),
        benchmark_limit=30,
        benchmark_offset=0,
        benchmark_fail_fast=False,
        benchmark_show_output=False,
    )

    if not PROMPT_TOOLKIT_AVAILABLE:
        return _launch_plain_text_ui(default_file=default_file, options=options)

    while True:
        action = button_dialog(
            title="PLLM Command Center",
            text=_menu_dialog_text(options),
            buttons=[
                ("Run", "run"),
                ("Report", "report"),
                ("Config", "config"),
                ("Loadout", "loadout"),
                ("Doctor", "doctor"),
                ("Help", "help"),
                ("Quit", "quit"),
            ],
            style=UI_STYLE,
        ).run()

        if action in (None, "quit"):
            return 0
        if action == "doctor":
            _show_doctor_dialog(options.base)
            continue
        if action == "report":
            _report_menu(options)
            continue
        if action == "config":
            _configure_menu(options)
            continue
        if action == "loadout":
            _loadout_menu(options)
            continue
        if action == "run":
            _run_menu(options=options, default_file=default_file)
            continue

        if action == "help":
            message_dialog(
                title="PLLM UI Help",
                text=(
                    "Run snippet: launches tools/pllm/test_executor.py with a live dashboard.\n\n"
                    "Run benchmark case: pick source + case id from copied gistable assets.\n\n"
                    "Run benchmark: execute many cases, including competition-run filtered ids.\n\n"
                    "Config and loadouts: change defaults and save/load named profiles.\n\n"
                    "Reports: source breakdown and competition filter status.\n\n"
                    "Doctor checks: verifies Docker, Ollama API, and local executor files.\n\n"
                    "CLI equivalents:\n"
                    "  python3 cli/pllm_cli.py run --file /abs/path/snippet.py\n"
                    "  python3 cli/pllm_cli.py run --case-id <id> --benchmark-source all-gists\n"
                    "  python3 cli/pllm_cli.py benchmark run --source competition-run --limit 100\n"
                    "  python3 cli/pllm_cli.py benchmark breakdown\n"
                    "  python3 cli/pllm_cli.py doctor\n"
                ),
                style=UI_STYLE,
            ).run()
            continue


def run_config_with_dashboard(config: RunConfig) -> tuple[int, RunStats]:
    dashboard = RunDashboard()
    dashboard.start(config)

    stop_event = threading.Event()

    def on_line(line: str, stats: RunStats) -> None:
        dashboard.on_line(line, stats)
        if dashboard.stop_requested():
            stop_event.set()

    return_code, stats = stream_executor(config, line_callback=on_line, stop_event=stop_event)
    dashboard.finish(return_code, stats)
    return return_code, stats


@dataclass
class RunDashboard:
    refresh_interval: float = 0.2
    config: RunConfig | None = None
    lines: int = 0
    build_successes: int = 0
    build_failures: int = 0
    completed_processes: int = 0
    last_line: str = ""
    recent_lines: list[str] = field(default_factory=list)
    return_code: int | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    _cancel_requested: bool = False

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._app: Application[None] | None = None
        self._app_thread: threading.Thread | None = None
        self._isatty = _is_tty()
        self._expected_attempts = 1

    def start(self, config: RunConfig) -> None:
        with self._lock:
            self.config = config
            self.started_at = time.monotonic()
            self._expected_attempts = max(1, config.loop * ((config.search_range * 2) + 1))

        if not self._isatty:
            print("PLLM run started")
            return

        if PROMPT_TOOLKIT_AVAILABLE:
            self._start_app()

            def refresh_loop() -> None:
                while not self._stop_event.wait(self.refresh_interval):
                    self._refresh()

            self._thread = threading.Thread(target=refresh_loop, name="pllm-dashboard-refresh", daemon=True)
            self._thread.start()

    def on_line(self, line: str, stats: RunStats) -> None:
        with self._lock:
            self.lines = stats.lines
            self.build_successes = stats.build_successes
            self.build_failures = stats.build_failures
            self.completed_processes = stats.completed_processes
            self.last_line = line
            if line:
                self.recent_lines.append(line[:160])
                self.recent_lines = self.recent_lines[-8:]

        if not self._isatty:
            print(line)
            return

        self._refresh()

    def finish(self, return_code: int, stats: RunStats) -> None:
        with self._lock:
            self.return_code = return_code
            self.finished_at = time.monotonic()
            self.lines = stats.lines
            self.build_successes = stats.build_successes
            self.build_failures = stats.build_failures
            self.completed_processes = stats.completed_processes
            self.last_line = stats.last_line

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.refresh_interval + 0.1)

        self._refresh()
        if self._app is not None:
            self._app.exit(result=None)
        if self._app_thread is not None:
            self._app_thread.join(timeout=1)

        if not self._isatty:
            elapsed = stats.elapsed_seconds
            print(f"PLLM run finished with exit code {return_code} in {elapsed:.1f}s")

    def request_stop(self) -> None:
        with self._lock:
            self._cancel_requested = True
        self._refresh()

    def stop_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def _start_app(self) -> None:
        assert PROMPT_TOOLKIT_AVAILABLE
        bindings = KeyBindings()

        @bindings.add("c-c")
        @bindings.add("q")
        def request_stop(event) -> None:  # type: ignore[no-untyped-def]
            self.request_stop()
            event.app.invalidate()

        control = FormattedTextControl(self._formatted_text, focusable=False)
        body = Box(
            body=Frame(
                body=Window(content=control, always_hide_cursor=True, wrap_lines=False),
                title="PLLM Dashboard",
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

        def runner() -> None:
            assert self._app is not None
            self._app.run(set_exception_handler=False)

        self._app_thread = threading.Thread(target=runner, name="pllm-dashboard-app", daemon=True)
        self._app_thread.start()

    def _refresh(self) -> None:
        if self._app is not None:
            self._app.invalidate()

    def _formatted_text(self) -> AnyFormattedText:
        with self._lock:
            config = self.config
            lines = self.lines
            success = self.build_successes
            failure = self.build_failures
            completed = self.completed_processes
            last_line = self.last_line
            recent_lines = list(self.recent_lines)
            return_code = self.return_code
            cancel = self._cancel_requested

        elapsed = _format_elapsed((self.finished_at or time.monotonic()) - self.started_at)
        attempts = success + failure
        ratio = min(1.0, attempts / max(1, self._expected_attempts))
        bar = _progress_bar(ratio, width=38)

        fragments: list[tuple[str, str]] = [
            ("class:headline", "PLLM run in progress\n"),
            ("class:muted", "Press q or Ctrl+C to stop the current run.\n\n"),
            ("class:label", "File          "),
            ("class:value", f"{(config.file if config else '')}\n"),
            ("class:label", "Model         "),
            ("class:value", f"{(config.model if config else '')}\n"),
            ("class:label", "Ollama base   "),
            ("class:value", f"{(config.base if config else '')}\n"),
            ("class:label", "Loop / range  "),
            ("class:value", f"{(config.loop if config else 0)} / {(config.search_range if config else 0)}\n"),
            ("class:label", "RAG / verbose "),
            ("class:value", f"{(config.rag if config else False)} / {(config.verbose if config else False)}\n\n"),
            ("class:label", "Build attempts "),
            ("class:accent", f"{attempts}/{self._expected_attempts}\n"),
            ("class:bar.complete", bar[0]),
            ("class:bar.remaining", bar[1]),
            ("", "\n\n"),
            ("class:good", f"Build successes: {success}"),
            ("", "    "),
            ("class:bad", f"Build failures: {failure}"),
            ("", "    "),
            ("class:accent", f"Completed processes: {completed}"),
            ("", "    "),
            ("class:accent", f"Lines: {lines}"),
            ("", "    "),
            ("class:accent", f"Elapsed: {elapsed}\n"),
        ]

        if cancel:
            fragments.extend(
                [
                    ("class:bad", "\nStop requested.\n"),
                    ("class:muted", "The executor is being terminated.\n"),
                ]
            )

        if last_line:
            fragments.extend(
                [
                    ("class:label", "\nLast line\n"),
                    ("class:value", f"{last_line[:160]}\n"),
                ]
            )

        if recent_lines:
            fragments.append(("class:label", "\nRecent output\n"))
            for line in recent_lines[-6:]:
                fragments.append(("class:value", f"  {line}\n"))

        if return_code is not None:
            fragments.extend(
                [
                    ("class:label", "\nExit code\n"),
                    ("class:value", f"{return_code}\n"),
                ]
            )
        return fragments


@dataclass
class TerminalBenchmarkDashboard:
    refresh_interval: float = 0.2

    def __post_init__(self) -> None:
        self.run_id = ""
        self.total = 0
        self.completed = 0
        self.successes = 0
        self.failures = 0
        self.source: BenchmarkSource = "all-gists"
        self.model = ""
        self.loop = 0
        self.search_range = 0
        self.rag = True
        self.verbose = False
        self.current_cases: list[str] = []
        self.last_case_id = ""
        self.last_status = ""
        self.started_at = time.monotonic()
        self.summary: BenchmarkSummary | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._cancel_requested = False
        self._thread: threading.Thread | None = None
        self._app_thread: threading.Thread | None = None
        self._app: Application[None] | None = None
        self._isatty = _is_tty()

    def start(
        self,
        *,
        run_id: str,
        total: int,
        source: BenchmarkSource,
        model: str,
        loop: int,
        search_range: int,
        rag: bool,
        verbose: bool,
        artifacts_dir: Path,
    ) -> None:
        self.run_id = run_id
        self.total = total
        self.source = source
        self.model = model
        self.loop = loop
        self.search_range = search_range
        self.rag = rag
        self.verbose = verbose
        self.started_at = time.monotonic()
        if self._isatty:
            self._start_prompt_toolkit_app()
        else:
            self._render_text()

        def refresh_loop() -> None:
            while not self._stop_event.wait(self.refresh_interval):
                self._refresh()

        self._thread = threading.Thread(target=refresh_loop, name="pllm-bench-refresh", daemon=True)
        self._thread.start()

    def case_started(self, case_id: str) -> None:
        with self._lock:
            if case_id not in self.current_cases:
                self.current_cases.append(case_id)
        self._refresh()

    def advance(self, result: dict[str, object]) -> None:
        case_id = str(result.get("case_id", ""))
        success = bool(result.get("success", False))
        status = str(result.get("status", "unknown"))
        with self._lock:
            self.completed = min(self.total, self.completed + 1)
            self.successes += int(success)
            self.failures += int(not success and status != "skipped")
            self.current_cases = [item for item in self.current_cases if item != case_id]
            self.last_case_id = case_id
            self.last_status = status
        self._refresh()

    def finish(self, *, summary: BenchmarkSummary, status: str = "completed") -> None:
        self.summary = summary
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
        @bindings.add("q")
        def request_stop(event) -> None:  # type: ignore[no-untyped-def]
            self.request_stop()
            event.app.invalidate()

        control = FormattedTextControl(self._formatted_text, focusable=False)
        body = Box(
            body=Frame(
                body=Window(content=control, always_hide_cursor=True, wrap_lines=False),
                title="PLLM Benchmark Dashboard",
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

        def runner() -> None:
            assert self._app is not None
            self._app.run(set_exception_handler=False)

        self._app_thread = threading.Thread(target=runner, name="pllm-bench-app", daemon=True)
        self._app_thread.start()

    def _formatted_text(self) -> AnyFormattedText:
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        elapsed_seconds = time.monotonic() - self.started_at
        success_rate = _format_success_rate(self.successes, self.completed)
        seconds_per_case = _format_seconds_per_case(self._seconds_per_completed_case(elapsed_seconds))
        eta = _format_eta(self._eta_seconds(elapsed_seconds))
        bar = _format_progress_bar(self.completed, self.total)
        fragments: list[tuple[str, str]] = [
            ("class:headline", "PLLM benchmark in progress\n"),
            ("class:muted", "Press q or Ctrl+C to stop scheduling new cases.\n\n"),
            ("class:label", "Run ID       "), ("class:value", f"{self.run_id}\n"),
            ("class:label", "Source       "), ("class:value", f"{self.source}\n"),
            ("class:label", "Model        "), ("class:value", f"{self.model}\n"),
            ("class:label", "Loop / range "), ("class:value", f"{self.loop} / {self.search_range}\n"),
            ("class:label", "RAG / verbose"), ("class:value", f"{self.rag} / {self.verbose}\n\n"),
            ("class:label", "Progress     "), ("class:accent", f"{self.completed}/{self.total} ({percent:5.1f}%)\n"),
            ("class:bar.complete", bar[: int((percent / 100.0) * len(bar))]),
            ("class:bar.remaining", bar[int((percent / 100.0) * len(bar)):]),
            ("", "\n\n"),
            ("class:good", f"Successes: {self.successes}"),
            ("", "    "),
            ("class:bad", f"Failures: {self.failures}"),
            ("", "    "),
            ("class:accent", f"Elapsed: {_format_elapsed(elapsed_seconds)}"),
            ("", "    "),
            ("class:accent", f"Success rate: {success_rate}"),
            ("", "    "),
            ("class:accent", f"Speed: {seconds_per_case}"),
            ("", "    "),
            ("class:accent", f"ETA: {eta}\n"),
        ]
        if self._cancel_requested:
            fragments.extend(
                [
                    ("class:bad", "\nStop requested\n"),
                    ("class:muted", "Benchmark will stop after active case(s) complete.\n"),
                ]
            )
        if self.current_cases:
            fragments.append(("class:label", "\nActive cases\n"))
            for case_id in self.current_cases[: min(6, len(self.current_cases))]:
                fragments.append(("class:value", f"  - {case_id}\n"))
        elif self.last_case_id:
            fragments.extend(
                [
                    ("class:label", "\nLast completed\n"),
                    ("class:value", f"  - {self.last_case_id} ({self.last_status})\n"),
                ]
            )
        if self.summary is not None:
            fragments.extend(
                [
                    ("class:label", "\nSummary\n"),
                    ("class:value", f"  - selected={self.summary.total_selected}\n"),
                    ("class:value", f"  - attempted={self.summary.attempted}\n"),
                    ("class:value", f"  - succeeded={self.summary.succeeded}\n"),
                    ("class:value", f"  - failed={self.summary.failed}\n"),
                    ("class:value", f"  - skipped={self.summary.skipped}\n"),
                ]
            )
        width = max(72, min(shutil.get_terminal_size((100, 24)).columns - 4, 116))
        trailing = max(0, width - 1)
        fragments.append(("", " " * trailing))
        return fragments

    def _render_text(self, final: bool = False) -> None:
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        elapsed_seconds = time.monotonic() - self.started_at
        success_rate = _format_success_rate(self.successes, self.completed)
        seconds_per_case = _format_seconds_per_case(self._seconds_per_completed_case(elapsed_seconds))
        eta = _format_eta(self._eta_seconds(elapsed_seconds))
        lines = [
            "=" * 80,
            "PLLM Benchmark Dashboard",
            "=" * 80,
            f"Run ID: {self.run_id}",
            f"Source: {self.source}",
            f"Model: {self.model}",
            f"Loop/range: {self.loop}/{self.search_range}",
            f"Progress: {_format_progress_bar(self.completed, self.total)} {self.completed}/{self.total} ({percent:5.1f}%)",
            (
                f"Successes: {self.successes}    Failures: {self.failures}    "
                f"Success rate: {success_rate}    Elapsed: {_format_elapsed(elapsed_seconds)}"
            ),
            f"Speed: {seconds_per_case}    ETA: {eta}",
        ]
        if self._cancel_requested:
            lines.append("Stop requested: benchmark will stop after active case(s) finish.")
        if self.current_cases:
            lines.append("Active cases:")
            lines.extend(f"  - {case_id}" for case_id in self.current_cases[: min(6, len(self.current_cases))])
        elif self.last_case_id:
            lines.append(f"Last completed: {self.last_case_id} ({self.last_status})")
        if final and self.summary is not None:
            lines.extend(
                [
                    "",
                    f"Summary selected={self.summary.total_selected}",
                    f"attempted={self.summary.attempted} succeeded={self.summary.succeeded}",
                    f"failed={self.summary.failed} skipped={self.summary.skipped}",
                ]
            )
        print("\n".join(lines), file=sys.stdout, flush=True)

    def _seconds_per_completed_case(self, elapsed_seconds: float | None = None) -> float | None:
        if self.completed <= 0:
            return None
        current_elapsed = elapsed_seconds if elapsed_seconds is not None else (time.monotonic() - self.started_at)
        return current_elapsed / self.completed

    def _eta_seconds(self, elapsed_seconds: float | None = None) -> float | None:
        if self.total <= 0 or self.completed <= 0 or self.completed >= self.total:
            return 0.0 if self.total > 0 and self.completed >= self.total else None
        current_elapsed = elapsed_seconds if elapsed_seconds is not None else (time.monotonic() - self.started_at)
        seconds_per_case = self._seconds_per_completed_case(current_elapsed)
        if seconds_per_case is None:
            return None
        return max(0.0, seconds_per_case * (self.total - self.completed))


def _menu_dialog_text(options: UIOptions) -> AnyFormattedText:
    model_count = len(fetch_ollama_models(options.base))
    loadout_count = len(_list_loadout_names())
    return HTML(
        "<b><ansibrightyellow>PLLM Command Center</ansibrightyellow></b>\n"
        "<style fg='#98c1d9'>Run, report, and configure without memorizing commands.</style>\n\n"
        f"<b>Tool:</b> {options.tool}\n"
        f"<b>Benchmark source:</b> {options.source}\n"
        f"<b>Model:</b> {options.model}\n"
        f"<b>Ollama models:</b> {model_count}\n"
        f"<b>Runtime:</b> loop {options.loop} | range {options.search_range} | "
        f"RAG {'on' if options.rag else 'off'} | verbose {'on' if options.verbose else 'off'}\n"
        f"<b>Benchmark defaults:</b> limit {options.benchmark_limit} | offset {options.benchmark_offset} | "
        f"fail-fast {'on' if options.benchmark_fail_fast else 'off'}\n"
        f"<b>Loadouts:</b> {loadout_count}\n"
        f"<b>Ollama base:</b> {options.base}"
    )


def _run_menu(*, options: UIOptions, default_file: str) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    if not _ensure_tool_supported(options):
        return
    run_action = button_dialog(
        title="Run",
        text="Choose run type",
        buttons=[
            ("Snippet", "snippet"),
            ("Case", "case"),
            ("Benchmark", "benchmark"),
            ("Back", "back"),
        ],
        style=UI_STYLE,
    ).run()
    if run_action in {None, "back"}:
        return
    if run_action == "benchmark":
        _run_benchmark_dialog(options=options)
        return
    if run_action == "case":
        config = _collect_case_run_config(options=options)
    else:
        config = _collect_run_config(options=options, default_file=default_file)
    if config is None:
        return

    return_code, stats = run_config_with_dashboard(config)
    message_dialog(
        title="Run finished",
        text=(
            f"Exit code: {return_code}\n"
            f"Elapsed: {stats.elapsed_seconds:.1f}s\n"
            f"Lines: {stats.lines}\n"
            f"Build successes: {stats.build_successes}\n"
            f"Build failures: {stats.build_failures}\n"
        ),
        style=UI_STYLE,
    ).run()


def _report_menu(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    choice = button_dialog(
        title="Reports",
        text="View benchmark source and filter reports.",
        buttons=[
            ("Breakdown", "breakdown"),
            ("Filter IDs", "filter"),
            ("Back", "back"),
        ],
        style=UI_STYLE,
    ).run()
    if choice in {None, "back"}:
        return
    if choice == "breakdown":
        message_dialog(
            title="Benchmark breakdowns",
            text=breakdown_summary(options.source),
            style=UI_STYLE,
        ).run()
    elif choice == "filter":
        path, matched, csv_total = rebuild_competition_filter()
        message_dialog(
            title="Competition filter",
            text=f"{path}\nCSV ids: {csv_total}\nMatched ids: {matched}",
            style=UI_STYLE,
        ).run()


def _configure_menu(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    while True:
        choice = button_dialog(
            title="Configure",
            text="Adjust source and runtime defaults.",
            buttons=[
                ("Tool", "tool"),
                ("Model", "model"),
                ("Source", "source"),
                ("Runtime", "runtime"),
                ("Benchmark", "benchmark"),
                ("Back", "back"),
            ],
            style=UI_STYLE,
        ).run()
        if choice in {None, "back"}:
            return
        if choice == "tool":
            _choose_tool_dialog(options)
            continue
        if choice == "model":
            _choose_model_dialog(options)
            continue
        if choice == "source":
            selected = _choose_benchmark_source_dialog(options.source)
            if selected is not None:
                options.source = selected
                message_dialog(title="Source", text=f"Benchmark source set to {selected}.", style=UI_STYLE).run()
            continue
        if choice == "runtime":
            _configure_runtime_dialog(options)
            continue
        if choice == "benchmark":
            _configure_benchmark_defaults_dialog(options)


def _choose_tool_dialog(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    tools = list_tools()
    if not tools:
        message_dialog(title="Tool", text="No tool folders found under tools/.", style=UI_STYLE).run()
        return
    selected = radiolist_dialog(
        title="Select tool",
        text="Choose tool from tools/ directory",
        values=[(tool, tool) for tool in tools],
        default=options.tool if options.tool in tools else tools[0],
        style=UI_STYLE,
    ).run()
    if selected is None:
        return
    options.tool = selected
    message_dialog(title="Tool", text=f"Active tool set to {selected}.", style=UI_STYLE).run()


def _choose_model_dialog(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    models = fetch_ollama_models(options.base)
    if models:
        values: list[tuple[str, str]] = [(model, model) for model in models]
        values.append(("__manual__", "Manual entry"))
        selected = radiolist_dialog(
            title="Select model",
            text=f"Available Ollama models from {options.base}",
            values=values,
            default=options.model if options.model in models else models[0],
            style=UI_STYLE,
        ).run()
        if selected is None:
            return
        if selected == "__manual__":
            manual = input_dialog(
                title="Manual model",
                text="Model name",
                default=options.model,
                style=UI_STYLE,
            ).run()
            if manual:
                options.model = manual.strip()
        else:
            options.model = selected
    else:
        manual = input_dialog(
            title="Select model",
            text="No models discovered automatically. Enter model name.",
            default=options.model,
            style=UI_STYLE,
        ).run()
        if manual:
            options.model = manual.strip()

    message_dialog(title="Model", text=f"Active model set to {options.model}.", style=UI_STYLE).run()


def _configure_runtime_dialog(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    model = input_dialog(title="Model", text="Default model", default=options.model, style=UI_STYLE).run()
    if model is None:
        return
    base = input_dialog(title="Ollama base", text="Default base URL", default=options.base, style=UI_STYLE).run()
    if base is None:
        return
    temp_raw = input_dialog(title="Temperature", text="Default temp", default=str(options.temp), style=UI_STYLE).run()
    if temp_raw is None:
        return
    loop_raw = input_dialog(title="Loop", text="Default loop", default=str(options.loop), style=UI_STYLE).run()
    if loop_raw is None:
        return
    range_raw = input_dialog(
        title="Python range",
        text="Default search range",
        default=str(options.search_range),
        style=UI_STYLE,
    ).run()
    if range_raw is None:
        return
    flags = checkboxlist_dialog(
        title="Runtime options",
        text="Default runtime flags",
        values=[("rag", "Enable RAG"), ("verbose", "Verbose output")],
        default_values=[
            *([] if not options.rag else ["rag"]),
            *([] if not options.verbose else ["verbose"]),
        ],
        style=UI_STYLE,
    ).run()
    if flags is None:
        return
    options.model = model.strip() or options.model
    options.base = base.strip() or options.base
    options.temp = _parse_float(temp_raw, options.temp)
    options.loop = max(1, _parse_int(loop_raw, options.loop))
    options.search_range = max(0, _parse_int(range_raw, options.search_range))
    options.rag = "rag" in flags
    options.verbose = "verbose" in flags
    message_dialog(title="Runtime", text="Runtime defaults updated.", style=UI_STYLE).run()


def _configure_benchmark_defaults_dialog(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    limit_raw = input_dialog(
        title="Benchmark limit",
        text="Default case limit (0=all)",
        default=str(options.benchmark_limit),
        style=UI_STYLE,
    ).run()
    if limit_raw is None:
        return
    offset_raw = input_dialog(
        title="Benchmark offset",
        text="Default offset",
        default=str(options.benchmark_offset),
        style=UI_STYLE,
    ).run()
    if offset_raw is None:
        return
    flags = checkboxlist_dialog(
        title="Benchmark flags",
        text="Default benchmark behavior",
        values=[
            ("fail_fast", "Fail fast"),
            ("show_output", "Show per-case output"),
        ],
        default_values=[
            *([] if not options.benchmark_fail_fast else ["fail_fast"]),
            *([] if not options.benchmark_show_output else ["show_output"]),
        ],
        style=UI_STYLE,
    ).run()
    if flags is None:
        return
    options.benchmark_limit = max(0, _parse_int(limit_raw, options.benchmark_limit))
    options.benchmark_offset = max(0, _parse_int(offset_raw, options.benchmark_offset))
    options.benchmark_fail_fast = "fail_fast" in flags
    options.benchmark_show_output = "show_output" in flags
    message_dialog(title="Benchmark defaults", text="Benchmark defaults updated.", style=UI_STYLE).run()


def _loadout_menu(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    while True:
        choice = button_dialog(
            title="Loadouts",
            text="Save, load, or delete named UI profiles.",
            buttons=[
                ("Save", "save"),
                ("Load", "load"),
                ("Delete", "delete"),
                ("Back", "back"),
            ],
            style=UI_STYLE,
        ).run()
        if choice in {None, "back"}:
            return
        if choice == "save":
            _save_loadout_dialog(options)
            continue
        if choice == "load":
            _load_loadout_dialog(options)
            continue
        if choice == "delete":
            _delete_loadout_dialog()


def _ensure_tool_supported(options: UIOptions) -> bool:
    if options.tool == "pllm":
        return True
    if PROMPT_TOOLKIT_AVAILABLE:
        message_dialog(
            title="Tool not wired",
            text=(
                f"Tool '{options.tool}' is selectable but not wired to this runner yet.\n"
                "Switch tool to 'pllm' in Config -> Tool to run now."
            ),
            style=UI_STYLE,
        ).run()
    else:
        print(f"Tool '{options.tool}' is not wired yet. Select tool 'pllm' to run.")
    return False


def _save_loadout_dialog(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    name = input_dialog(
        title="Save loadout",
        text="Loadout name",
        default="default",
        style=UI_STYLE,
    ).run()
    if not name:
        return
    path = _loadouts_dir() / f"{name.strip()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_options_to_dict(options), indent=2), encoding="utf-8")
    message_dialog(title="Loadout saved", text=str(path), style=UI_STYLE).run()


def _load_loadout_dialog(options: UIOptions) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    names = _list_loadout_names()
    if not names:
        message_dialog(title="Load loadout", text="No loadouts saved yet.", style=UI_STYLE).run()
        return
    selected = radiolist_dialog(
        title="Load loadout",
        text="Choose a saved loadout",
        values=[(name, name) for name in names],
        default=names[0],
        style=UI_STYLE,
    ).run()
    if selected is None:
        return
    path = _loadouts_dir() / f"{selected}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message_dialog(title="Load loadout", text=f"Failed to load: {exc}", style=UI_STYLE).run()
        return
    _apply_options_dict(options, payload if isinstance(payload, dict) else {})
    message_dialog(title="Loadout loaded", text=f"Applied {selected}.", style=UI_STYLE).run()


def _delete_loadout_dialog() -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    names = _list_loadout_names()
    if not names:
        message_dialog(title="Delete loadout", text="No loadouts to delete.", style=UI_STYLE).run()
        return
    selected = radiolist_dialog(
        title="Delete loadout",
        text="Choose a loadout to delete",
        values=[(name, name) for name in names],
        default=names[0],
        style=UI_STYLE,
    ).run()
    if selected is None:
        return
    path = _loadouts_dir() / f"{selected}.json"
    if path.exists():
        path.unlink()
    message_dialog(title="Loadout deleted", text=selected, style=UI_STYLE).run()


def _loadouts_dir() -> Path:
    return ROOT_DIR / "data" / "loadouts" / "pllm"


def _list_loadout_names() -> list[str]:
    directory = _loadouts_dir()
    if not directory.exists():
        return []
    names = [path.stem for path in directory.glob("*.json") if path.is_file()]
    names.sort()
    return names


def _options_to_dict(options: UIOptions) -> dict[str, object]:
    return {
        "tool": options.tool,
        "model": options.model,
        "base": options.base,
        "temp": options.temp,
        "loop": options.loop,
        "search_range": options.search_range,
        "rag": options.rag,
        "verbose": options.verbose,
        "source": options.source,
        "benchmark_limit": options.benchmark_limit,
        "benchmark_offset": options.benchmark_offset,
        "benchmark_fail_fast": options.benchmark_fail_fast,
        "benchmark_show_output": options.benchmark_show_output,
    }


def _apply_options_dict(options: UIOptions, payload: dict[str, object]) -> None:
    available_tools = list_tools()
    loaded_tool = str(payload.get("tool", options.tool) or options.tool)
    if loaded_tool in available_tools:
        options.tool = loaded_tool
    options.model = str(payload.get("model", options.model) or options.model)
    options.base = str(payload.get("base", options.base) or options.base)
    options.temp = _parse_float(str(payload.get("temp", options.temp)), options.temp)
    options.loop = max(1, _parse_int(str(payload.get("loop", options.loop)), options.loop))
    options.search_range = max(
        0,
        _parse_int(str(payload.get("search_range", options.search_range)), options.search_range),
    )
    options.rag = bool(payload.get("rag", options.rag))
    options.verbose = bool(payload.get("verbose", options.verbose))
    options.source = _normalize_source(str(payload.get("source", options.source)))
    options.benchmark_limit = max(
        0,
        _parse_int(str(payload.get("benchmark_limit", options.benchmark_limit)), options.benchmark_limit),
    )
    options.benchmark_offset = max(
        0,
        _parse_int(str(payload.get("benchmark_offset", options.benchmark_offset)), options.benchmark_offset),
    )
    options.benchmark_fail_fast = bool(payload.get("benchmark_fail_fast", options.benchmark_fail_fast))
    options.benchmark_show_output = bool(payload.get("benchmark_show_output", options.benchmark_show_output))


def _run_benchmark_dialog(
    *,
    options: UIOptions,
) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    selected = _choose_benchmark_source_dialog(options.source)
    if selected is None:
        return

    limit_raw = input_dialog(
        title="Benchmark limit",
        text="How many cases to run (0 = all in source)",
        default=str(options.benchmark_limit),
        style=UI_STYLE,
    ).run()
    if limit_raw is None:
        return
    offset_raw = input_dialog(
        title="Benchmark offset",
        text="Start offset in sorted case ids",
        default=str(options.benchmark_offset),
        style=UI_STYLE,
    ).run()
    if offset_raw is None:
        return

    flags = checkboxlist_dialog(
        title="Benchmark options",
        text="Toggle benchmark options",
        values=[
            ("fail_fast", "Stop on first failure"),
            ("show_output", "Show full per-case output"),
            ("rag", "Enable RAG"),
            ("verbose", "Verbose test_executor mode"),
        ],
        default_values=[
            *([] if not options.benchmark_fail_fast else ["fail_fast"]),
            *([] if not options.benchmark_show_output else ["show_output"]),
            *([] if not options.rag else ["rag"]),
            *([] if not options.verbose else ["verbose"]),
        ],
        style=UI_STYLE,
    ).run()
    if flags is None:
        return

    limit = max(0, _parse_int(limit_raw, 0))
    offset = max(0, _parse_int(offset_raw, 0))
    fail_fast = "fail_fast" in flags
    show_output = "show_output" in flags
    rag = "rag" in flags
    verbose = "verbose" in flags
    options.source = selected
    options.benchmark_limit = limit
    options.benchmark_offset = offset
    options.benchmark_fail_fast = fail_fast
    options.benchmark_show_output = show_output
    options.rag = rag
    options.verbose = verbose

    if show_output:
        def handler(line: str) -> None:
            print(line)
    else:
        def handler(_line: str) -> None:
            return

    observer = None
    if not show_output:
        observer = TerminalBenchmarkDashboard()

    return_code, summary = run_benchmark(
        source=selected,
        model=options.model,
        base=options.base,
        temp=options.temp,
        loop=max(1, options.loop),
        search_range=max(0, options.search_range),
        rag=rag,
        verbose=verbose,
        limit=limit,
        offset=offset,
        fail_fast=fail_fast,
        show_case_output=show_output,
        line_handler=handler,
        observer=observer,
    )

    message_dialog(
        title="Benchmark finished",
        text=(
            f"Source: {summary.source}\n"
            f"Selected: {summary.total_selected}\n"
            f"Attempted: {summary.attempted}\n"
            f"Succeeded: {summary.succeeded}\n"
            f"Failed: {summary.failed}\n"
            f"Skipped: {summary.skipped}\n"
            f"Elapsed: {summary.elapsed_seconds:.1f}s\n"
            f"Exit code: {return_code}"
        ),
        style=UI_STYLE,
    ).run()


def _choose_benchmark_source_dialog(current: BenchmarkSource) -> BenchmarkSource | None:
    assert PROMPT_TOOLKIT_AVAILABLE
    return radiolist_dialog(
        title="Benchmark source",
        text="Choose source for case-based runs.",
        values=[
            ("all-gists", "all-gists"),
            ("dockerized-gists", "dockerized-gists"),
            ("competition-run", "competition-run (filtered by competition ids)"),
        ],
        default=current,
        style=UI_STYLE,
    ).run()


def _collect_case_run_config(
    *,
    options: UIOptions,
) -> RunConfig | None:
    assert PROMPT_TOOLKIT_AVAILABLE
    case_id = input_dialog(
        title="Run benchmark case",
        text=f"Case ID from source '{options.source}'",
        default="",
        style=UI_STYLE,
    ).run()
    if not case_id:
        return None
    snippet_path = resolve_snippet_path(case_id.strip(), options.source)
    if snippet_path is None:
        available = list_case_ids(options.source)
        sample = ", ".join(available[:8]) if available else "none found"
        message_dialog(
            title="Case not found",
            text=(
                f"Case '{case_id}' was not found in source '{options.source}'.\n"
                f"Available sample: {sample}"
            ),
            style=UI_STYLE,
        ).run()
        return None
    return _collect_run_config(
        options=options,
        default_file=str(snippet_path),
    )


def _collect_run_config(
    options: UIOptions,
    default_file: str,
) -> RunConfig | None:
    assert PROMPT_TOOLKIT_AVAILABLE

    file_path = input_dialog(
        title="Run snippet",
        text="Snippet path to run",
        default=default_file or "/abs/path/to/snippet.py",
        style=UI_STYLE,
    ).run()
    if not file_path:
        return None

    model = input_dialog(
        title="Model",
        text="Ollama model name",
        default=options.model,
        style=UI_STYLE,
    ).run()
    if model is None:
        return None

    base = input_dialog(
        title="Ollama base URL",
        text="Base URL for Ollama",
        default=options.base,
        style=UI_STYLE,
    ).run()
    if base is None:
        return None

    temp_str = input_dialog(
        title="Temperature",
        text="Model temperature",
        default=str(options.temp),
        style=UI_STYLE,
    ).run()
    if temp_str is None:
        return None

    loop_str = input_dialog(
        title="Loop count",
        text="How many PLLM loops",
        default=str(options.loop),
        style=UI_STYLE,
    ).run()
    if loop_str is None:
        return None

    range_str = input_dialog(
        title="Python range",
        text="Search range around inferred Python version",
        default=str(options.search_range),
        style=UI_STYLE,
    ).run()
    if range_str is None:
        return None

    selected_flags = checkboxlist_dialog(
        title="Options",
        text="Toggle runtime options",
        values=[
            ("rag", "Enable RAG"),
            ("verbose", "Verbose output"),
        ],
        default_values=[
            *([] if not options.rag else ["rag"]),
            *([] if not options.verbose else ["verbose"]),
        ],
        style=UI_STYLE,
    ).run()
    if selected_flags is None:
        return None

    parsed_temp = _parse_float(temp_str, options.temp)
    parsed_loop = max(1, _parse_int(loop_str, options.loop))
    parsed_range = max(0, _parse_int(range_str, options.search_range))
    parsed_rag = "rag" in selected_flags
    parsed_verbose = "verbose" in selected_flags

    options.model = model.strip() or options.model
    options.base = base.strip() or options.base
    options.temp = parsed_temp
    options.loop = parsed_loop
    options.search_range = parsed_range
    options.rag = parsed_rag
    options.verbose = parsed_verbose

    return RunConfig(
        file=file_path.strip(),
        model=options.model,
        base=options.base,
        temp=options.temp,
        loop=options.loop,
        search_range=options.search_range,
        rag=options.rag,
        verbose=options.verbose,
    )


def _show_doctor_dialog(base_url: str) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    checks = run_doctor(base_url)
    message_dialog(title="Doctor checks", text=format_doctor_report(checks), style=UI_STYLE).run()


def _launch_plain_text_ui(*, default_file: str, options: UIOptions) -> int:
    print("prompt_toolkit is not installed. Falling back to plain text mode.")
    active_source: BenchmarkSource = options.source
    while True:
        model_count = len(fetch_ollama_models(options.base))
        print("\nPLLM Command Center")
        print(
            f"Tool={options.tool} source={active_source} model={options.model} "
            f"models={model_count} loop={options.loop} range={options.search_range}"
        )
        print("1) Run snippet")
        print("2) Run benchmark case")
        print("3) Run benchmark")
        print("4) Reports")
        print("5) Configure")
        print("6) Loadouts")
        print("7) Doctor checks")
        print("8) Quit")
        choice = input("Select option: ").strip()
        if choice == "8":
            return 0
        if choice == "7":
            print(format_doctor_report(run_doctor(options.base)))
            continue
        if choice == "6":
            print("Loadouts")
            print("1) Save 2) Load 3) Delete 4) Back")
            sub = input("Select option: ").strip()
            if sub == "1":
                name = input("Loadout name [default]: ").strip() or "default"
                path = _loadouts_dir() / f"{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(_options_to_dict(options), indent=2), encoding="utf-8")
                print(f"Saved {path}")
            elif sub == "2":
                names = _list_loadout_names()
                print("Available:", ", ".join(names) if names else "(none)")
                name = input("Loadout to load: ").strip()
                if name:
                    path = _loadouts_dir() / f"{name}.json"
                    if path.exists():
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        _apply_options_dict(options, payload if isinstance(payload, dict) else {})
                        active_source = options.source
                        print(f"Loaded {name}")
            elif sub == "3":
                names = _list_loadout_names()
                print("Available:", ", ".join(names) if names else "(none)")
                name = input("Loadout to delete: ").strip()
                if name:
                    path = _loadouts_dir() / f"{name}.json"
                    if path.exists():
                        path.unlink()
                        print(f"Deleted {name}")
            continue
        if choice == "5":
            print("Configure")
            print("1) Tool 2) Model 3) Source 4) Runtime 5) Benchmark defaults 6) Back")
            sub = input("Select option: ").strip()
            if sub == "1":
                tools = list_tools()
                if not tools:
                    print("No tools found under tools/.")
                    continue
                print("Available tools:")
                for index, tool in enumerate(tools, start=1):
                    marker = "*" if tool == options.tool else " "
                    print(f"  {index}){marker} {tool}")
                selected = input(f"Tool [{options.tool}]: ").strip()
                if selected:
                    if selected.isdigit() and 1 <= int(selected) <= len(tools):
                        options.tool = tools[int(selected) - 1]
                    elif selected in tools:
                        options.tool = selected
                    else:
                        print("Invalid tool selection.")
                        continue
                print(f"Tool set to {options.tool}")
            elif sub == "2":
                models = fetch_ollama_models(options.base)
                if models:
                    print("Discovered Ollama models:")
                    for index, model_name in enumerate(models, start=1):
                        marker = "*" if model_name == options.model else " "
                        print(f"  {index}){marker} {model_name}")
                    selected = input(f"Model [{options.model}] (index/name): ").strip()
                    if selected:
                        if selected.isdigit() and 1 <= int(selected) <= len(models):
                            options.model = models[int(selected) - 1]
                        elif selected in models:
                            options.model = selected
                        else:
                            options.model = selected
                    print(f"Model set to {options.model}")
                else:
                    options.model = input(f"Model [{options.model}]: ").strip() or options.model
                    print(f"Model set to {options.model}")
            elif sub == "3":
                print("1) all-gists 2) dockerized-gists 3) competition-run")
                sel = input("Source: ").strip()
                if sel == "1":
                    active_source = "all-gists"
                elif sel == "2":
                    active_source = "dockerized-gists"
                elif sel == "3":
                    active_source = "competition-run"
                options.source = active_source
            elif sub == "4":
                options.model = input(f"Model [{options.model}]: ").strip() or options.model
                options.base = input(f"Ollama base [{options.base}]: ").strip() or options.base
                options.temp = _parse_float(input(f"Temp [{options.temp}]: ").strip() or str(options.temp), options.temp)
                options.loop = max(1, _parse_int(input(f"Loop [{options.loop}]: ").strip() or str(options.loop), options.loop))
                options.search_range = max(
                    0,
                    _parse_int(input(f"Range [{options.search_range}]: ").strip() or str(options.search_range), options.search_range),
                )
                options.rag = (input(f"RAG [{options.rag}] y/n: ").strip().lower() or ("y" if options.rag else "n")) in {"y", "yes"}
                options.verbose = (input(f"Verbose [{options.verbose}] y/n: ").strip().lower() or ("y" if options.verbose else "n")) in {"y", "yes"}
            elif sub == "5":
                options.benchmark_limit = max(
                    0,
                    _parse_int(
                        input(f"Benchmark limit [{options.benchmark_limit}]: ").strip() or str(options.benchmark_limit),
                        options.benchmark_limit,
                    ),
                )
                options.benchmark_offset = max(
                    0,
                    _parse_int(
                        input(f"Benchmark offset [{options.benchmark_offset}]: ").strip() or str(options.benchmark_offset),
                        options.benchmark_offset,
                    ),
                )
                options.benchmark_fail_fast = (
                    input(f"Fail fast [{options.benchmark_fail_fast}] y/n: ").strip().lower()
                    or ("y" if options.benchmark_fail_fast else "n")
                ) in {"y", "yes"}
                options.benchmark_show_output = (
                    input(f"Show output [{options.benchmark_show_output}] y/n: ").strip().lower()
                    or ("y" if options.benchmark_show_output else "n")
                ) in {"y", "yes"}
            continue
        if choice in {"1", "2", "3"} and not _ensure_tool_supported(options):
            continue
        if choice == "4":
            print(breakdown_summary(active_source))
            path, matched, csv_total = rebuild_competition_filter()
            print(f"Filter file: {path}")
            print(f"CSV ids parsed: {csv_total}")
            print(f"Matched all-gists ids: {matched}")
            continue
        if choice == "3":
            limit = max(
                0,
                _parse_int(
                    input(f"Limit (0=all) [{options.benchmark_limit}]: ").strip() or str(options.benchmark_limit),
                    options.benchmark_limit,
                ),
            )
            offset = max(
                0,
                _parse_int(
                    input(f"Offset [{options.benchmark_offset}]: ").strip() or str(options.benchmark_offset),
                    options.benchmark_offset,
                ),
            )
            fail_fast = (
                input(f"Fail fast [{options.benchmark_fail_fast}] y/n: ").strip().lower()
                or ("y" if options.benchmark_fail_fast else "n")
            ) in {"y", "yes"}
            show_output = (
                input(f"Show output [{options.benchmark_show_output}] y/n: ").strip().lower()
                or ("y" if options.benchmark_show_output else "n")
            ) in {"y", "yes"}
            observer = None if show_output else TerminalBenchmarkDashboard()
            return_code, summary = run_benchmark(
                source=active_source,
                model=options.model,
                base=options.base,
                temp=options.temp,
                loop=max(1, options.loop),
                search_range=max(0, options.search_range),
                rag=options.rag,
                verbose=options.verbose,
                limit=limit,
                offset=offset,
                fail_fast=fail_fast,
                show_case_output=show_output,
                observer=observer,
            )
            print(
                f"Benchmark source={summary.source} selected={summary.total_selected} "
                f"attempted={summary.attempted} succeeded={summary.succeeded} "
                f"failed={summary.failed} skipped={summary.skipped} "
                f"elapsed={summary.elapsed_seconds:.1f}s rc={return_code}"
            )
            continue
        if choice == "2":
            case_id = input(f"Case id from {active_source}: ").strip()
            if not case_id:
                continue
            snippet_path = resolve_snippet_path(case_id, active_source)
            if snippet_path is None:
                sample = ", ".join(list_case_ids(active_source)[:8]) or "none found"
                print(f"Case not found. Sample ids: {sample}")
                continue
            config = RunConfig(
                file=str(snippet_path),
                model=options.model,
                base=options.base,
                temp=options.temp,
                loop=options.loop,
                search_range=options.search_range,
                rag=options.rag,
                verbose=options.verbose,
            )
            return_code, stats = run_config_with_dashboard(config)
            print(
                f"Exit code {return_code}. Elapsed {stats.elapsed_seconds:.1f}s. "
                f"Build ok/fail: {stats.build_successes}/{stats.build_failures}"
            )
            continue
        if choice != "1":
            continue

        prompt = f"Snippet path [{default_file}]: " if default_file else "Snippet path: "
        file_path = input(prompt).strip() or default_file
        if not file_path:
            continue
        options.model = input(f"Model [{options.model}]: ").strip() or options.model
        options.base = input(f"Ollama base [{options.base}]: ").strip() or options.base
        options.temp = _parse_float(input(f"Temp [{options.temp}]: ").strip() or str(options.temp), options.temp)
        options.loop = max(1, _parse_int(input(f"Loop [{options.loop}]: ").strip() or str(options.loop), options.loop))
        options.search_range = max(
            0,
            _parse_int(input(f"Range [{options.search_range}]: ").strip() or str(options.search_range), options.search_range),
        )
        options.rag = (input(f"RAG [{options.rag}] y/n: ").strip().lower() or ("y" if options.rag else "n")) in {"y", "yes"}
        options.verbose = (input(f"Verbose [{options.verbose}] y/n: ").strip().lower() or ("y" if options.verbose else "n")) in {"y", "yes"}

        config = RunConfig(
            file=file_path,
            model=options.model,
            base=options.base,
            temp=options.temp,
            loop=options.loop,
            search_range=options.search_range,
            rag=options.rag,
            verbose=options.verbose,
        )
        return_code, stats = run_config_with_dashboard(config)
        print(
            f"Exit code {return_code}. "
            f"Elapsed {stats.elapsed_seconds:.1f}s. "
            f"Build ok/fail: {stats.build_successes}/{stats.build_failures}"
        )


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _format_progress_bar(completed: int, total: int, width: int = 36) -> str:
    if total <= 0:
        return "#" * width
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return ("#" * filled) + ("-" * (width - filled))


def _format_seconds_per_case(seconds_per_case: float | None) -> str:
    if seconds_per_case is None:
        return "n/a"
    return f"{seconds_per_case:.1f}s/case"


def _format_success_rate(successes: int, completed: int) -> str:
    if completed <= 0:
        return "n/a"
    return f"{(successes / completed) * 100.0:.1f}%"


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return _format_elapsed(seconds)


def _progress_bar(ratio: float, width: int = 30) -> tuple[str, str]:
    clamped = max(0.0, min(1.0, ratio))
    filled = int(width * clamped)
    return ("#" * filled, "-" * (width - filled))


def _parse_int(raw: str, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _is_tty() -> bool:
    import sys

    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _normalize_source(value: str) -> BenchmarkSource:
    if value in {"all-gists", "dockerized-gists", "competition-run"}:
        return value
    return "all-gists"
