from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from cli.pllm.benchmark_data import (
    BenchmarkSource,
    breakdown_summary,
    list_case_ids,
    rebuild_competition_filter,
    resolve_snippet_path,
)
from cli.pllm.benchmark_runner import run_benchmark
from cli.pllm.core import RunConfig, RunStats, format_doctor_report, run_doctor, stream_executor

PROMPT_TOOLKIT_AVAILABLE = True
try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import AnyFormattedText
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


def launch_terminal_ui(
    default_model: str = "gemma2",
    default_base: str = "http://localhost:11434",
    default_loop: int = 10,
    default_range: int = 0,
    default_file: str = "",
    default_benchmark_source: BenchmarkSource = "all-gists",
) -> int:
    if not PROMPT_TOOLKIT_AVAILABLE:
        return _launch_plain_text_ui(
            default_model,
            default_base,
            default_loop,
            default_range,
            default_file,
            default_benchmark_source,
        )

    active_source: BenchmarkSource = _normalize_source(default_benchmark_source)

    while True:
        action = button_dialog(
            title="PLLM Control Center",
            text="Select an action",
            buttons=[
                ("Run", "run"),
                ("Bench", "bench"),
                ("Doctor", "doctor"),
                ("Help", "help"),
                ("Quit", "quit"),
            ],
            style=UI_STYLE,
        ).run()

        if action in (None, "quit"):
            return 0
        if action == "doctor":
            _show_doctor_dialog(default_base)
            continue
        if action == "bench":
            active_source = _benchmark_menu(active_source)
            continue
        if action == "run":
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
                continue
            if run_action == "benchmark":
                _run_benchmark_dialog(
                    active_source=active_source,
                    default_model=default_model,
                    default_base=default_base,
                    default_loop=default_loop,
                    default_range=default_range,
                )
                continue
            if run_action == "case":
                config = _collect_case_run_config(
                    default_model=default_model,
                    default_base=default_base,
                    default_loop=default_loop,
                    default_range=default_range,
                    active_source=active_source,
                )
            else:
                config = _collect_run_config(
                    default_model,
                    default_base,
                    default_loop,
                    default_range,
                    default_file,
                )
            if config is None:
                continue

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
            continue

        if action == "help":
            message_dialog(
                title="PLLM UI Help",
                text=(
                    "Run snippet: launches tools/pllm/test_executor.py with a live dashboard.\n\n"
                    "Run benchmark case: pick source + case id from copied gistable assets.\n\n"
                    "Run benchmark: execute many cases, including competition-run filtered ids.\n\n"
                    "Benchmark setup: source selection, competition filter rebuild, and breakdown views.\n\n"
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


def _benchmark_menu(active_source: BenchmarkSource) -> BenchmarkSource:
    assert PROMPT_TOOLKIT_AVAILABLE
    current_source = active_source
    while True:
        choice = button_dialog(
            title="Benchmark setup",
            text="Manage source selection, filter, and breakdowns.",
            buttons=[
                ("Select source", "source"),
                ("Show breakdowns", "breakdown"),
                ("Rebuild competition filter", "filter"),
                ("Back", "back"),
            ],
            style=UI_STYLE,
        ).run()
        if choice in {None, "back"}:
            return current_source
        if choice == "source":
            selected = _choose_benchmark_source_dialog(current_source)
            if selected is not None:
                current_source = selected
                message_dialog(
                    title="Benchmark source",
                    text=f"Active source set to {current_source}.",
                    style=UI_STYLE,
                ).run()
            continue
        if choice == "breakdown":
            message_dialog(
                title="Benchmark breakdowns",
                text=breakdown_summary(current_source),
                style=UI_STYLE,
            ).run()
            continue
        if choice == "filter":
            path, matched, csv_total = rebuild_competition_filter()
            message_dialog(
                title="Competition filter rebuilt",
                text=(
                    f"Filter file: {path}\n"
                    f"CSV ids parsed: {csv_total}\n"
                    f"Matched all-gists ids: {matched}"
                ),
                style=UI_STYLE,
            ).run()


def _run_benchmark_dialog(
    *,
    active_source: BenchmarkSource,
    default_model: str,
    default_base: str,
    default_loop: int,
    default_range: int,
) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    selected = _choose_benchmark_source_dialog(active_source)
    if selected is None:
        return

    limit_raw = input_dialog(
        title="Benchmark limit",
        text="How many cases to run (0 = all in source)",
        default="30" if selected == "competition-run" else "10",
        style=UI_STYLE,
    ).run()
    if limit_raw is None:
        return
    offset_raw = input_dialog(
        title="Benchmark offset",
        text="Start offset in sorted case ids",
        default="0",
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
        default_values=["rag"],
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

    if show_output:
        def handler(line: str) -> None:
            print(line)
    else:
        def handler(_line: str) -> None:
            return

    return_code, summary = run_benchmark(
        source=selected,
        model=default_model,
        base=default_base,
        temp=0.7,
        loop=max(1, default_loop),
        search_range=max(0, default_range),
        rag=rag,
        verbose=verbose,
        limit=limit,
        offset=offset,
        fail_fast=fail_fast,
        show_case_output=show_output,
        line_handler=handler,
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
    default_model: str,
    default_base: str,
    default_loop: int,
    default_range: int,
    active_source: BenchmarkSource,
) -> RunConfig | None:
    assert PROMPT_TOOLKIT_AVAILABLE
    case_id = input_dialog(
        title="Run benchmark case",
        text=f"Case ID from source '{active_source}'",
        default="",
        style=UI_STYLE,
    ).run()
    if not case_id:
        return None
    snippet_path = resolve_snippet_path(case_id.strip(), active_source)
    if snippet_path is None:
        available = list_case_ids(active_source)
        sample = ", ".join(available[:8]) if available else "none found"
        message_dialog(
            title="Case not found",
            text=(
                f"Case '{case_id}' was not found in source '{active_source}'.\n"
                f"Available sample: {sample}"
            ),
            style=UI_STYLE,
        ).run()
        return None
    return _collect_run_config(
        default_model=default_model,
        default_base=default_base,
        default_loop=default_loop,
        default_range=default_range,
        default_file=str(snippet_path),
    )


def _collect_run_config(
    default_model: str,
    default_base: str,
    default_loop: int,
    default_range: int,
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
        default=default_model,
        style=UI_STYLE,
    ).run()
    if model is None:
        return None

    base = input_dialog(
        title="Ollama base URL",
        text="Base URL for Ollama",
        default=default_base,
        style=UI_STYLE,
    ).run()
    if base is None:
        return None

    temp_str = input_dialog(
        title="Temperature",
        text="Model temperature",
        default="0.7",
        style=UI_STYLE,
    ).run()
    if temp_str is None:
        return None

    loop_str = input_dialog(
        title="Loop count",
        text="How many PLLM loops",
        default=str(default_loop),
        style=UI_STYLE,
    ).run()
    if loop_str is None:
        return None

    range_str = input_dialog(
        title="Python range",
        text="Search range around inferred Python version",
        default=str(default_range),
        style=UI_STYLE,
    ).run()
    if range_str is None:
        return None

    options = checkboxlist_dialog(
        title="Options",
        text="Toggle runtime options",
        values=[
            ("rag", "Enable RAG"),
            ("verbose", "Verbose output"),
        ],
        default_values=["rag"],
        style=UI_STYLE,
    ).run()
    if options is None:
        return None

    return RunConfig(
        file=file_path.strip(),
        model=model.strip() or default_model,
        base=base.strip() or default_base,
        temp=_parse_float(temp_str, 0.7),
        loop=max(1, _parse_int(loop_str, default_loop)),
        search_range=max(0, _parse_int(range_str, default_range)),
        rag="rag" in options,
        verbose="verbose" in options,
    )


def _show_doctor_dialog(base_url: str) -> None:
    assert PROMPT_TOOLKIT_AVAILABLE
    checks = run_doctor(base_url)
    message_dialog(title="Doctor checks", text=format_doctor_report(checks), style=UI_STYLE).run()


def _launch_plain_text_ui(
    default_model: str,
    default_base: str,
    default_loop: int,
    default_range: int,
    default_file: str,
    default_benchmark_source: BenchmarkSource,
) -> int:
    print("prompt_toolkit is not installed. Falling back to plain text mode.")
    active_source: BenchmarkSource = _normalize_source(default_benchmark_source)
    while True:
        print("\nPLLM Control Center")
        print("1) Run snippet")
        print("2) Run benchmark case")
        print("3) Run benchmark")
        print("4) Benchmark setup")
        print("5) Doctor checks")
        print("6) Quit")
        choice = input("Select option: ").strip()
        if choice == "6":
            return 0
        if choice == "5":
            print(format_doctor_report(run_doctor(default_base)))
            continue
        if choice == "4":
            print("\nBenchmark setup")
            print(f"Current source: {active_source}")
            print("1) Change source")
            print("2) Show breakdowns")
            print("3) Rebuild competition filter")
            print("4) Back")
            sub = input("Select option: ").strip()
            if sub == "1":
                print("1) all-gists")
                print("2) dockerized-gists")
                print("3) competition-run")
                sel = input("Select source: ").strip()
                if sel == "1":
                    active_source = "all-gists"
                elif sel == "2":
                    active_source = "dockerized-gists"
                elif sel == "3":
                    active_source = "competition-run"
            elif sub == "2":
                print(breakdown_summary(active_source))
            elif sub == "3":
                path, matched, csv_total = rebuild_competition_filter()
                print(f"Filter file: {path}")
                print(f"CSV ids parsed: {csv_total}")
                print(f"Matched all-gists ids: {matched}")
            continue
        if choice == "3":
            limit = max(0, _parse_int(input("Limit (0=all) [30]: ").strip() or "30", 30))
            offset = max(0, _parse_int(input("Offset [0]: ").strip() or "0", 0))
            fail_fast = (input("Fail fast? [y/N]: ").strip().lower() or "n") in {"y", "yes"}
            show_output = (input("Show full case output? [y/N]: ").strip().lower() or "n") in {"y", "yes"}
            rag = (input("Enable RAG? [Y/n]: ").strip().lower() or "y") in {"y", "yes"}
            verbose = (input("Verbose executor mode? [y/N]: ").strip().lower() or "n") in {"y", "yes"}
            return_code, summary = run_benchmark(
                source=active_source,
                model=default_model,
                base=default_base,
                temp=0.7,
                loop=max(1, default_loop),
                search_range=max(0, default_range),
                rag=rag,
                verbose=verbose,
                limit=limit,
                offset=offset,
                fail_fast=fail_fast,
                show_case_output=show_output,
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
            default_file = str(snippet_path)
        if choice != "1":
            continue

        prompt = f"Snippet path [{default_file}]: " if default_file else "Snippet path: "
        file_path = input(prompt).strip() or default_file
        if not file_path:
            continue
        model = input(f"Model [{default_model}]: ").strip() or default_model
        base = input(f"Ollama base [{default_base}]: ").strip() or default_base
        temp = _parse_float(input("Temperature [0.7]: ").strip() or "0.7", 0.7)
        loop = max(1, _parse_int(input(f"Loop [{default_loop}]: ").strip() or str(default_loop), default_loop))
        search_range = max(
            0,
            _parse_int(input(f"Range [{default_range}]: ").strip() or str(default_range), default_range),
        )
        rag = (input("Enable RAG? [Y/n]: ").strip().lower() or "y") in {"y", "yes"}
        verbose = (input("Verbose logs? [y/N]: ").strip().lower() or "n") in {"y", "yes"}

        config = RunConfig(
            file=file_path,
            model=model,
            base=base,
            temp=temp,
            loop=loop,
            search_range=search_range,
            rag=rag,
            verbose=verbose,
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
