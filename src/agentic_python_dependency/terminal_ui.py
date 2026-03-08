from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML, AnyFormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.shortcuts import button_dialog, checkboxlist_dialog, input_dialog, message_dialog, radiolist_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Box, Frame

from agentic_python_dependency.config import MODEL_PROFILE_DEFAULTS, Settings
from agentic_python_dependency.presets import RESEARCH_BUNDLE_DEFAULTS, RESEARCH_FEATURES, PRESET_CONFIGS
from agentic_python_dependency.router import OllamaStatsSnapshot, OllamaStatsTracker
from agentic_python_dependency.tools.official_baselines import validate_pyego_runtime


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


def _resolve_apdr_version() -> str:
    try:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        version = str(payload.get("project", {}).get("version", "") or "").strip()
        if version:
            return version
    except Exception:
        pass
    try:
        return package_version("agentic-python-dependency")
    except PackageNotFoundError:
        pass
    except Exception:
        pass
    return "unknown"


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


def _format_tokens_per_second(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f} tok/s"


def _fit_table_cell(value: str, width: int) -> str:
    if width <= 0:
        return ""
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= width:
        return normalized.ljust(width)
    if width <= 3:
        return normalized[:width]
    return f"{normalized[: width - 3]}..."


@dataclass
class TerminalBenchmarkDashboard:
    refresh_interval: float = 0.2

    def __post_init__(self) -> None:
        self.app_version = _resolve_apdr_version()
        self.run_id = ""
        self.total = 0
        self.completed = 0
        self.successes = 0
        self.failures = 0
        self.resolver = "apdr"
        self.preset = "optimized"
        self.prompt_profile = "optimized"
        self.research_bundle = "baseline"
        self.research_features: tuple[str, ...] = ()
        self.benchmark_source = "all-gists"
        self.model_summary = "gemma-moe: gemma3:4b / gemma3:12b"
        self.runtime_config: dict[str, object] = {}
        self.jobs = 1
        self.target = "benchmark"
        self.artifacts_dir = Path(".")
        self.current_cases: list[str] = []
        self.current_case_activity: dict[str, dict[str, object]] = {}
        self.recent_case_activity: list[dict[str, object]] = []
        self.recent_case_results: list[dict[str, object]] = []
        self.last_case_id = ""
        self.last_status = ""
        self.started_at = time.monotonic()
        self.summary_path: Path | None = None
        self.warnings_path: Path | None = None
        self.ollama_stats: OllamaStatsTracker | None = None
        self.docker_build_seconds_total = 0.0
        self.docker_run_seconds_total = 0.0
        self.llm_wall_clock_seconds_total = 0.0
        self.image_cache_hits = 0
        self.build_skips = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._cancel_requested = False
        self._hard_cancel_requested = False
        self._hard_exit_callback: Callable[[], None] | None = None
        self._thread: threading.Thread | None = None
        self._app_thread: threading.Thread | None = None
        self._app: Application[None] | None = None
        self._results_window: Window | None = None
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
        research_bundle: str = "baseline",
        research_features: tuple[str, ...] = (),
        benchmark_source: str = "all-gists",
        jobs: int,
        target: str,
        artifacts_dir: Path,
        elapsed_seconds: float = 0.0,
        ollama_stats: OllamaStatsTracker | None = None,
        runtime_config: dict[str, object] | None = None,
        completed_results: list[dict[str, object]] | None = None,
    ) -> None:
        self.run_id = run_id
        self.total = total
        self.completed = completed
        self.successes = successes
        self.failures = failures
        self.resolver = resolver
        self.preset = preset
        self.prompt_profile = prompt_profile
        self.research_bundle = research_bundle
        self.research_features = tuple(research_features)
        self.benchmark_source = benchmark_source
        self.model_summary = model_summary
        self.runtime_config = dict(runtime_config or {})
        self.jobs = jobs
        self.target = target
        self.artifacts_dir = artifacts_dir
        self.ollama_stats = ollama_stats
        self.started_at = time.monotonic() - elapsed_seconds
        self.recent_case_results = [self._case_result_row(result) for result in (completed_results or [])[:200]]
        self.docker_build_seconds_total = sum(
            float(result.get("docker_build_seconds_total", 0.0) or 0.0)
            for result in (completed_results or [])
        )
        self.docker_run_seconds_total = sum(
            float(result.get("docker_run_seconds_total", 0.0) or 0.0)
            for result in (completed_results or [])
        )
        self.llm_wall_clock_seconds_total = sum(
            float(result.get("llm_wall_clock_seconds", 0.0) or 0.0)
            for result in (completed_results or [])
        )
        self.image_cache_hits = sum(int(result.get("image_cache_hits", 0) or 0) for result in (completed_results or []))
        self.build_skips = sum(int(result.get("build_skips", 0) or 0) for result in (completed_results or []))
        if self._isatty:
            self._start_prompt_toolkit_app()
        else:
            self._render_text()

        def _refresh_loop() -> None:
            while not self._stop_event.wait(self.refresh_interval):
                self._refresh()

        self._thread = threading.Thread(target=_refresh_loop, name="apdr-ui-benchmark-refresh", daemon=True)
        self._thread.start()

    def case_started(self, case_id: str) -> None:
        with self._lock:
            if case_id not in self.current_cases:
                self.current_cases.append(case_id)
        self._refresh()

    def case_event(self, case_id: str, *, attempt: int = 0, kind: str, detail: str) -> None:
        with self._lock:
            self.current_case_activity[case_id] = {
                "case_id": case_id,
                "attempt": attempt,
                "kind": kind,
                "detail": detail,
            }
            self.recent_case_activity.insert(
                0,
                {
                    "case_id": case_id,
                    "attempt": attempt,
                    "kind": kind,
                    "detail": detail,
                },
            )
            del self.recent_case_activity[20:]
        self._refresh()

    def advance(self, result: dict[str, object]) -> None:
        case_id = str(result.get("case_id", ""))
        success = bool(result.get("success", False))
        row = self._case_result_row(result)
        with self._lock:
            self.completed = min(self.total, self.completed + 1)
            self.successes += int(success)
            self.failures += int(not success)
            self.last_case_id = case_id
            self.last_status = "success" if success else str(result.get("final_error_category", "failure"))
            self.current_cases = [item for item in self.current_cases if item != case_id]
            self.current_case_activity.pop(case_id, None)
            self.recent_case_results = [item for item in self.recent_case_results if item.get("case_id") != case_id]
            self.recent_case_results.insert(0, row)
            del self.recent_case_results[200:]
            self.docker_build_seconds_total += float(result.get("docker_build_seconds_total", 0.0) or 0.0)
            self.docker_run_seconds_total += float(result.get("docker_run_seconds_total", 0.0) or 0.0)
            self.llm_wall_clock_seconds_total += float(result.get("llm_wall_clock_seconds", 0.0) or 0.0)
            self.image_cache_hits += int(result.get("image_cache_hits", 0) or 0)
            self.build_skips += int(result.get("build_skips", 0) or 0)
        self._refresh()

    @staticmethod
    def _case_result_row(result: dict[str, object]) -> dict[str, object]:
        success = bool(result.get("success", False))
        attempts = int(result.get("attempts", 0) or 0)
        target_python = str(result.get("target_python", "") or "")
        wall_clock_seconds = float(result.get("wall_clock_seconds", 0.0) or 0.0)
        error = "Success" if success else str(result.get("final_error_category", "failure") or "failure")
        result_matches_csv = str(result.get("result_matches_csv", "") or "").strip().upper()
        dependencies = result.get("dependencies", [])
        if isinstance(dependencies, list):
            dependency_preview = ", ".join(str(item) for item in dependencies[:3] if item)
            if len(dependencies) > 3:
                dependency_preview = f"{dependency_preview}, +{len(dependencies) - 3} more"
        else:
            dependency_preview = str(dependencies or "")
        return {
            "case_id": str(result.get("case_id", "") or ""),
            "success": success,
            "error": error,
            "attempts": attempts,
            "target_python": target_python,
            "seconds": wall_clock_seconds,
            "pllm_match": "MATCH" if result_matches_csv == "PASS" else ("MISS" if result_matches_csv == "FAIL" else "-"),
            "dependencies": dependency_preview or "-",
        }

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

    def request_hard_stop(self, *, invoke_callback: bool = True) -> None:
        callback: Callable[[], None] | None
        with self._lock:
            self._cancel_requested = True
            self._hard_cancel_requested = True
            callback = self._hard_exit_callback
        self._refresh()
        if invoke_callback and callback is not None:
            callback()

    def stop_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def hard_stop_requested(self) -> bool:
        with self._lock:
            return self._hard_cancel_requested

    def set_hard_exit_callback(self, callback: Callable[[], None] | None) -> None:
        with self._lock:
            self._hard_exit_callback = callback

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
            if self.stop_requested():
                self.request_hard_stop()
            else:
                self.request_stop()
            event.app.invalidate()

        @bindings.add("up")
        def _scroll_up(event) -> None:
            if self._results_window is None:
                return
            self._results_window.vertical_scroll = max(0, self._results_window.vertical_scroll - 1)
            event.app.invalidate()

        @bindings.add("down")
        def _scroll_down(event) -> None:
            if self._results_window is None:
                return
            self._results_window.vertical_scroll += 1
            event.app.invalidate()

        @bindings.add("pageup")
        def _page_up(event) -> None:
            if self._results_window is None:
                return
            self._results_window.vertical_scroll = max(0, self._results_window.vertical_scroll - 10)
            event.app.invalidate()

        @bindings.add("pagedown")
        def _page_down(event) -> None:
            if self._results_window is None:
                return
            self._results_window.vertical_scroll += 10
            event.app.invalidate()

        @bindings.add("home")
        def _go_home(event) -> None:
            if self._results_window is None:
                return
            self._results_window.vertical_scroll = 0
            event.app.invalidate()

        @bindings.add("end")
        def _go_end(event) -> None:
            if self._results_window is None:
                return
            self._results_window.vertical_scroll = max(0, len(self.recent_case_results) + 4)
            event.app.invalidate()

        control = FormattedTextControl(self._formatted_text, focusable=False)
        self._results_window = Window(
            content=FormattedTextControl(self._results_table_formatted_text, focusable=True),
            always_hide_cursor=True,
            wrap_lines=False,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )
        body = Box(
            body=Frame(
                body=HSplit(
                    [
                        Window(
                            content=control,
                            always_hide_cursor=True,
                            wrap_lines=False,
                            dont_extend_height=True,
                        ),
                        Frame(
                            body=self._results_window,
                            title="Completed Cases (newest first; arrows/PageUp/PageDown/Home/End scroll)",
                            style="class:frame",
                        ),
                    ]
                ),
                title="APDR Benchmark Dashboard",
                style="class:frame",
            ),
            padding=1,
        )
        self._app = Application(
            layout=Layout(HSplit([body]), focused_element=self._results_window),
            full_screen=True,
            mouse_support=False,
            style=DASHBOARD_STYLE,
            key_bindings=bindings,
        )

        def _runner() -> None:
            assert self._app is not None
            self._app.run(set_exception_handler=False)

        self._app_thread = threading.Thread(target=_runner, name="apdr-ui-benchmark-app", daemon=True)
        self._app_thread.start()

    def _formatted_text(self) -> AnyFormattedText:
        width = max(72, min(shutil.get_terminal_size((100, 24)).columns - 4, 116))
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        elapsed_seconds = time.monotonic() - self.started_at
        success_rate = _format_success_rate(self.successes, self.completed)
        seconds_per_case = _format_seconds_per_case(self._seconds_per_completed_case(elapsed_seconds))
        eta = _format_eta(self._eta_seconds(elapsed_seconds))
        bar = _format_progress_bar(self.completed, self.total)
        ollama_snapshot = self.ollama_stats.snapshot() if self.ollama_stats is not None else OllamaStatsSnapshot()
        fragments: list[tuple[str, str]] = [
            ("class:headline", "APDR benchmark in progress\n"),
            ("class:muted", "Use Ctrl+C only if you intend to stop the benchmark process itself.\n\n"),
            ("class:label", "Run ID       "), ("class:value", f"{self.run_id}\n"),
            ("class:label", "Version      "), ("class:value", f"{self.app_version}\n"),
            ("class:label", "Target       "), ("class:value", f"{self.target}\n"),
            ("class:label", "Resolver     "), ("class:value", f"{self.resolver}\n"),
            ("class:label", "Preset       "), ("class:value", f"{self.preset}\n"),
            ("class:label", "Research     "), ("class:value", f"{self.research_bundle}\n"),
            ("class:label", "Prompt       "), ("class:value", f"{self.prompt_profile}\n"),
            ("class:label", "Source       "), ("class:value", f"{self.benchmark_source}\n"),
            ("class:label", "Models       "), ("class:value", f"{getattr(self, 'model_summary', 'default')}\n"),
            ("class:label", "Effective    "), ("class:value", f"{self._runtime_summary()}\n"),
            ("class:label", "Ollama       "), ("class:value", f"{self._ollama_summary(ollama_snapshot)}\n"),
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
            ("class:accent", f"Elapsed: {_format_elapsed(elapsed_seconds)}"),
            ("", "    "),
            ("class:accent", f"Success rate: {success_rate}"),
            ("", "    "),
            ("class:accent", f"Speed: {seconds_per_case}"),
            ("", "    "),
            ("class:accent", f"ETA: {eta}\n"),
            ("class:label", "Perf         "),
            (
                "class:value",
                (
                    f"build {self.docker_build_seconds_total:.1f}s, "
                    f"run {self.docker_run_seconds_total:.1f}s, "
                    f"llm {self.llm_wall_clock_seconds_total:.1f}s, "
                    f"cache hits {self.image_cache_hits}, "
                    f"build skips {self.build_skips}\n"
                ),
            ),
        ]
        if self.research_features:
            fragments.extend(
                [
                    ("class:label", "Research feats "),
                    ("class:value", f"{', '.join(self.research_features)}\n"),
                ]
            )
        last_ollama = self._last_ollama_summary(ollama_snapshot)
        if last_ollama is not None:
            fragments.extend(
                [
                    ("class:label", "Last LLM     "),
                    ("class:value", f"{last_ollama}\n"),
                ]
            )
        if self._cancel_requested:
            fragments.extend(
                [
                    ("class:bad", "\nStop requested\n"),
                    (
                        "class:muted",
                        "APDR will stop scheduling new cases and exit after the active work finishes. "
                        "Press Ctrl+C again to hard quit immediately.\n",
                    ),
                ]
            )
        if self.current_cases:
            fragments.append(("class:label", "\nActive cases\n"))
            for case_id in self.current_cases[: min(6, len(self.current_cases))]:
                activity = self.current_case_activity.get(case_id)
                if activity:
                    detail = str(activity.get("detail", "") or "").replace("\n", " ").strip()
                    kind = str(activity.get("kind", "") or "activity")
                    attempt = int(activity.get("attempt", 0) or 0)
                    if len(detail) > 92:
                        detail = f"{detail[:89]}..."
                    fragments.append(("class:value", f"  • {case_id} [a{attempt} {kind}] {detail}\n"))
                else:
                    fragments.append(("class:value", f"  • {case_id}\n"))
        elif self.last_case_id:
            fragments.extend(
                [
                    ("class:label", "\nLast completed\n"),
                    ("class:value", f"  • {self.last_case_id} ({self.last_status})\n"),
                ]
            )
        if self.recent_case_activity:
            fragments.append(("class:label", "\nRecent activity\n"))
            for item in self.recent_case_activity[: min(5, len(self.recent_case_activity))]:
                case_id = str(item.get("case_id", "") or "")
                kind = str(item.get("kind", "") or "activity")
                detail = str(item.get("detail", "") or "").replace("\n", " ").strip()
                if len(detail) > 92:
                    detail = f"{detail[:89]}..."
                fragments.append(("class:value", f"  • {case_id} [{kind}] {detail}\n"))
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

    def _results_table_formatted_text(self) -> AnyFormattedText:
        width = max(72, min(shutil.get_terminal_size((100, 24)).columns - 8, 156))
        status_width = 4
        case_width = 18
        py_width = 6
        attempts_width = 4
        seconds_width = 7
        match_width = 5
        error_width = 18
        fixed_width = status_width + case_width + py_width + attempts_width + seconds_width + match_width + error_width + 14
        deps_width = max(20, width - fixed_width)
        header = (
            f"{_fit_table_cell('STAT', status_width)}  "
            f"{_fit_table_cell('CASE ID', case_width)}  "
            f"{_fit_table_cell('PY', py_width)}  "
            f"{_fit_table_cell('TRY', attempts_width)}  "
            f"{_fit_table_cell('SEC', seconds_width)}  "
            f"{_fit_table_cell('PLLM', match_width)}  "
            f"{_fit_table_cell('RESULT', error_width)}  "
            f"{_fit_table_cell('DEPENDENCIES', deps_width)}"
        )
        divider = "-" * len(header)
        fragments: list[tuple[str, str]] = [
            ("class:label", f"{header}\n"),
            ("class:muted", f"{divider}\n"),
        ]
        if not self.recent_case_results:
            fragments.append(
                (
                    "class:muted",
                    "No completed benchmark cases yet. Passed and failed cases will appear here as the run advances.\n",
                )
            )
            return fragments
        for row in self.recent_case_results:
            status_text = "PASS" if bool(row.get("success", False)) else "FAIL"
            status_style = "class:good" if status_text == "PASS" else "class:bad"
            seconds = float(row.get("seconds", 0.0) or 0.0)
            match_text = str(row.get("pllm_match", "-") or "-")
            match_style = "class:accent" if match_text == "MATCH" else ("class:bad" if match_text == "MISS" else "class:muted")
            fragments.extend(
                [
                    (status_style, _fit_table_cell(status_text, status_width)),
                    ("", "  "),
                    ("class:value", _fit_table_cell(str(row.get("case_id", "")), case_width)),
                    ("", "  "),
                    ("class:value", _fit_table_cell(str(row.get("target_python", "") or "-"), py_width)),
                    ("", "  "),
                    ("class:value", _fit_table_cell(str(row.get("attempts", "") or "-"), attempts_width)),
                    ("", "  "),
                    ("class:value", _fit_table_cell(f"{seconds:.1f}", seconds_width)),
                    ("", "  "),
                    (match_style, _fit_table_cell(match_text, match_width)),
                    ("", "  "),
                    ("class:value", _fit_table_cell(str(row.get("error", "")), error_width)),
                    ("", "  "),
                    ("class:value", _fit_table_cell(str(row.get("dependencies", "")), deps_width)),
                    ("", "\n"),
                ]
            )
        return fragments

    def _render_text(self, final: bool = False) -> None:
        percent = (self.completed / self.total * 100.0) if self.total else 100.0
        elapsed_seconds = time.monotonic() - self.started_at
        success_rate = _format_success_rate(self.successes, self.completed)
        seconds_per_case = _format_seconds_per_case(self._seconds_per_completed_case(elapsed_seconds))
        eta = _format_eta(self._eta_seconds(elapsed_seconds))
        ollama_snapshot = self.ollama_stats.snapshot() if self.ollama_stats is not None else OllamaStatsSnapshot()
        lines = [
            "=" * 80,
            "APDR Benchmark Dashboard",
            "=" * 80,
            f"Run ID: {self.run_id}",
            f"Version: {self.app_version}",
            f"Target: {self.target}",
            f"Resolver: {self.resolver}",
            f"Preset: {self.preset}",
            f"Research bundle: {self.research_bundle}",
            f"Prompt profile: {self.prompt_profile}",
            f"Benchmark source: {self.benchmark_source}",
            f"Models: {getattr(self, 'model_summary', 'default')}",
            f"Effective runtime: {self._runtime_summary()}",
            f"Ollama: {self._ollama_summary(ollama_snapshot)}",
            f"Jobs: {self.jobs}",
            f"Artifacts: {self.artifacts_dir}",
            "",
            f"Progress: {_format_progress_bar(self.completed, self.total)} {self.completed}/{self.total} ({percent:5.1f}%)",
            (
                f"Successes: {self.successes}    Failures: {self.failures}    "
                f"Success rate: {success_rate}    Elapsed: {_format_elapsed(elapsed_seconds)}"
            ),
            f"Speed: {seconds_per_case}    ETA: {eta}",
            (
                f"Docker build: {self.docker_build_seconds_total:.1f}s    "
                f"Docker run: {self.docker_run_seconds_total:.1f}s    "
                f"LLM: {self.llm_wall_clock_seconds_total:.1f}s    "
                f"Cache hits: {self.image_cache_hits}    Build skips: {self.build_skips}"
            ),
        ]
        last_ollama = self._last_ollama_summary(ollama_snapshot)
        if last_ollama is not None:
            lines.append(f"Last LLM: {last_ollama}")
        if self.research_features:
            lines.append(f"Research features: {', '.join(self.research_features)}")
        if self._cancel_requested:
            lines.append(
                "Stop requested: APDR will stop after the current active cases finish. "
                "Press Ctrl+C again to hard quit immediately."
            )
        if self.current_cases:
            lines.append("Active cases:")
            for case_id in self.current_cases[: min(6, len(self.current_cases))]:
                activity = self.current_case_activity.get(case_id)
                if activity:
                    lines.append(
                        "  - "
                        f"{case_id} "
                        f"[a{int(activity.get('attempt', 0) or 0)} {activity.get('kind', 'activity')}] "
                        f"{activity.get('detail', '')}"
                    )
                else:
                    lines.append(f"  - {case_id}")
        elif self.last_case_id:
            lines.append(f"Last completed: {self.last_case_id} ({self.last_status})")
        if self.recent_case_activity:
            lines.append("Recent activity:")
            for item in self.recent_case_activity[: min(5, len(self.recent_case_activity))]:
                lines.append(
                    "  - "
                    f"{item.get('case_id', '')} "
                    f"[{item.get('kind', 'activity')}] "
                    f"{item.get('detail', '')}"
                )
        if self.recent_case_results:
            lines.append("")
            lines.append("Recent completed cases:")
            for row in self.recent_case_results[: min(8, len(self.recent_case_results))]:
                status = "PASS" if bool(row.get("success", False)) else "FAIL"
                lines.append(
                    "  "
                    f"{status} {row.get('case_id', '')} "
                    f"py={row.get('target_python', '') or '-'} "
                    f"try={row.get('attempts', '') or '-'} "
                    f"pllm={row.get('pllm_match', '-') or '-'} "
                    f"result={row.get('error', '')}"
                )
        if final:
            lines.extend(
                [
                    "",
                    f"Summary: {self.summary_path}",
                    f"Warnings: {self.warnings_path or 'none recorded'}",
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

    @staticmethod
    def _ollama_summary(snapshot: OllamaStatsSnapshot) -> str:
        if snapshot.calls <= 0:
            return "waiting for first response"
        parts = [f"{snapshot.calls} calls"]
        if snapshot.eval_tokens > 0 or snapshot.eval_duration_ns > 0:
            parts.append(f"out {snapshot.eval_tokens} tok @ {_format_tokens_per_second(snapshot.eval_tokens_per_second)}")
        if snapshot.prompt_tokens > 0 or snapshot.prompt_duration_ns > 0:
            parts.append(f"prompt {snapshot.prompt_tokens} tok @ {_format_tokens_per_second(snapshot.prompt_tokens_per_second)}")
        return ", ".join(parts)

    @staticmethod
    def _last_ollama_summary(snapshot: OllamaStatsSnapshot) -> str | None:
        if snapshot.calls <= 0:
            return None
        model = snapshot.last_model or "unknown"
        stage = snapshot.last_stage or "unknown"
        return f"{stage} / {model} @ {_format_tokens_per_second(snapshot.last_eval_tokens_per_second)}"

    def _runtime_summary(self) -> str:
        model_profile = str(
            self.runtime_config.get("effective_model_profile", self.runtime_config.get("model_profile", "default"))
        )
        rag_mode = str(self.runtime_config.get("effective_rag_mode", self.runtime_config.get("rag_mode", "pypi")))
        structured = bool(
            self.runtime_config.get(
                "effective_structured_prompting",
                self.runtime_config.get("structured_prompting", False),
            )
        )
        repair_limit = int(
            self.runtime_config.get(
                "effective_repair_cycle_limit",
                self.runtime_config.get("repair_cycle_limit", 0),
            )
            or 0
        )
        fallback = bool(
            self.runtime_config.get(
                "effective_candidate_fallback_before_repair",
                self.runtime_config.get("allow_candidate_fallback_before_repair", False),
            )
        )
        return (
            f"profile={model_profile} rag={rag_mode} "
            f"structured={'on' if structured else 'off'} "
            f"repair={repair_limit} fallback={'on' if fallback else 'off'}"
        )


@dataclass
class TerminalUI:
    settings: Settings
    doctor_command: ActionCallback
    run_benchmark: ActionCallback
    run_failed_cases: ActionCallback
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
        self._app_version = _resolve_apdr_version()

    def run(self) -> int:
        if self._use_prompt_toolkit:
            return self._run_prompt_toolkit()
        return self._run_basic()

    def _run_prompt_toolkit(self) -> int:
        while True:
            choice = button_dialog(
                title="APDR Command Center",
                text=self._menu_dialog_text(),
                buttons=[
                    ("Run", "run"),
                    ("Reports", "report"),
                    ("Configure", "config"),
                    ("Loadouts", "loadouts"),
                    ("Doctor", "doctor"),
                    ("Quit", "quit"),
                ],
                style=UI_STYLE,
            ).run()
            if choice in {None, "quit"}:
                message_dialog(title="APDR", text="Exiting APDR UI.", style=UI_STYLE).run()
                return 0
            exit_code = self._dispatch_choice(choice)
            if choice in {"run", "report"} and exit_code >= 0:
                self._show_status_dialog(f"Command finished with exit code {exit_code}.")

    def _run_basic(self) -> int:
        while True:
            self._clear()
            self._print_header()
            self._print_menu()
            choice = self.input_fn("\nSelect an option: ").strip().lower()
            if choice in {"q", "quit", "8"}:
                self.output("\nExiting APDR UI.")
                return 0
            exit_code = self._dispatch_choice(choice)
            if choice in {"1", "2", "3", "4", "5"} and exit_code >= 0:
                self._pause_after(exit_code)

    def _dispatch_choice(self, choice: str | None) -> int:
        if choice in {"doctor", "1"}:
            return self._run_captured(self.doctor_command, self.settings, None)
        if choice in {"run", "2"}:
            return self._run_menu()
        if choice in {"report", "3"}:
            return self._report_menu()
        if choice in {"config", "4"}:
            return self._configure_menu()
        if choice in {"loadouts", "5"}:
            return self._loadouts_menu()
        self._show_status_dialog("Invalid choice.")
        return 0

    def _loadouts_menu(self) -> int:
        if self._use_prompt_toolkit:
            return self._loadouts_menu_prompt_toolkit()
        return self._loadouts_menu_basic()

    def _loadouts_menu_prompt_toolkit(self) -> int:
        while True:
            choice = button_dialog(
                title="Loadouts",
                text="Save, load, or delete reusable APDR settings profiles.",
                buttons=[
                    ("Save current", "save"),
                    ("Load", "load"),
                    ("Delete", "delete"),
                    ("Back", "back"),
                ],
                style=UI_STYLE,
            ).run()
            if choice in {None, "back"}:
                return 0
            if choice == "save":
                self._save_current_loadout()
            elif choice == "load":
                self._load_saved_loadout()
            elif choice == "delete":
                self._delete_saved_loadout()

    def _loadouts_menu_basic(self) -> int:
        self.output("\nLoadouts")
        self.output("  1. Save current settings")
        self.output("  2. Load saved settings")
        self.output("  3. Delete saved settings")
        self.output("  4. Back")
        choice = self.input_fn("Select loadout option: ").strip().lower()
        if choice == "1":
            self._save_current_loadout()
        elif choice == "2":
            self._load_saved_loadout()
        elif choice == "3":
            self._delete_saved_loadout()
        return 0

    def _apply_preset_config(self, preset: str, *, prompt_profile_override: str | None = None) -> None:
        preset_config = PRESET_CONFIGS[preset]
        self.settings.preset = preset
        self.settings.prompt_profile = prompt_profile_override or preset_config.prompt_profile
        self.settings.max_attempts = preset_config.max_attempts
        self.settings.default_module_grouping = preset_config.reporting_grouping
        self.settings.rag_mode = preset_config.rag_mode
        self.settings.structured_prompting = preset_config.structured_prompting
        self.settings.candidate_plan_count = preset_config.candidate_plan_count
        self.settings.allow_candidate_fallback_before_repair = preset_config.allow_candidate_fallback_before_repair
        self.settings.repair_cycle_limit = preset_config.repair_cycle_limit
        self.settings.repo_evidence_enabled = preset_config.repo_evidence_enabled

    def _run_menu(self) -> int:
        if self._use_prompt_toolkit:
            return self._run_menu_prompt_toolkit()
        return self._run_menu_basic()

    def _run_menu_prompt_toolkit(self) -> int:
        while True:
            choice = button_dialog(
                title="Run Workflows",
                text="Choose a run command.",
                buttons=[
                    ("Smoke benchmark", "2"),
                    ("Full benchmark", "3"),
                    ("Resume benchmark", "u"),
                    ("Retry failed cases", "9"),
                    ("Solve local project", "4"),
                    ("Back", "back"),
                ],
                style=UI_STYLE,
            ).run()
            if choice in {None, "back"}:
                return -1
            if choice == "2":
                return self._run_smoke()
            if choice == "3":
                return self._run_full()
            if choice == "u":
                return self._run_resume_benchmark()
            if choice == "9":
                return self._run_failed_cases_from_run()
            if choice == "4":
                return self._run_project_solve()

    def _run_menu_basic(self) -> int:
        self.output("\nRun workflows")
        self.output("  1. Smoke benchmark")
        self.output("  2. Full benchmark")
        self.output("  3. Resume benchmark")
        self.output("  4. Retry failed cases")
        self.output("  5. Solve local project")
        self.output("  6. Back")
        choice = self.input_fn("Select run option: ").strip().lower()
        if choice == "1":
            return self._run_smoke()
        if choice == "2":
            return self._run_full()
        if choice == "3":
            return self._run_resume_benchmark()
        if choice == "4":
            return self._run_failed_cases_from_run()
        if choice == "5":
            return self._run_project_solve()
        return -1

    def _report_menu(self) -> int:
        if self._use_prompt_toolkit:
            return self._report_menu_prompt_toolkit()
        return self._report_menu_basic()

    def _report_menu_prompt_toolkit(self) -> int:
        while True:
            choice = button_dialog(
                title="Reports",
                text="Choose a report command.",
                buttons=[
                    ("Summarize run", "5"),
                    ("Failure report", "6"),
                    ("Module report", "7"),
                    ("Timeline view", "l"),
                    ("Back", "back"),
                ],
                style=UI_STYLE,
            ).run()
            if choice in {None, "back"}:
                return -1
            if choice == "5":
                return self._run_summary()
            if choice == "6":
                return self._run_failures()
            if choice == "7":
                return self._run_modules()
            if choice == "l":
                return self._run_timeline()

    def _report_menu_basic(self) -> int:
        self.output("\nReports")
        self.output("  1. Summarize run")
        self.output("  2. Failure report")
        self.output("  3. Module report")
        self.output("  4. Timeline view")
        self.output("  5. Back")
        choice = self.input_fn("Select report option: ").strip().lower()
        if choice == "1":
            return self._run_summary()
        if choice == "2":
            return self._run_failures()
        if choice == "3":
            return self._run_modules()
        if choice == "4":
            return self._run_timeline()
        return -1

    def _configure_menu(self) -> int:
        if self._use_prompt_toolkit:
            return self._configure_menu_prompt_toolkit()
        return self._configure_menu_basic()

    def _configure_menu_prompt_toolkit(self) -> int:
        while True:
            choice = button_dialog(
                title="Configure",
                text="Adjust resolver, preset, models, and runtime options.",
                buttons=[
                    ("Resolver", "v"),
                    ("Preset", "p"),
                    ("Models", "m"),
                    ("Research", "x"),
                    ("Runtime", "r"),
                    ("Official setup", "o"),
                    ("Benchmark source", "s"),
                    ("Fresh run", "f"),
                    ("Trace LLM", "t"),
                    ("Back", "back"),
                ],
                style=UI_STYLE,
            ).run()
            if choice in {None, "back"}:
                return 0
            if choice == "v":
                self._choose_resolver()
            elif choice == "p":
                self._choose_preset()
            elif choice == "m":
                self._choose_model_profile()
            elif choice == "x":
                self._configure_research()
            elif choice == "r":
                self._configure_runtime()
            elif choice == "o":
                self._configure_official_setup()
            elif choice == "s":
                self._choose_benchmark_source()
            elif choice == "f":
                self._fresh_run = not self._fresh_run
                self._show_status_dialog(f"Fresh run is now {'on' if self._fresh_run else 'off'}.")
            elif choice == "t":
                self.settings.trace_llm = not self.settings.trace_llm
                self._show_status_dialog(f"LLM tracing is now {'on' if self.settings.trace_llm else 'off'}.")

    def _configure_menu_basic(self) -> int:
        self.output("\nConfigure")
        self.output("  1. Change resolver")
        self.output("  2. Change preset")
        self.output("  3. Change model bundle")
        self.output("  4. Configure research bundle/features")
        self.output("  5. Runtime controls")
        self.output("  6. Official baseline setup")
        self.output("  7. Change benchmark source")
        self.output("  8. Toggle fresh run / no LLM cache")
        self.output("  9. Toggle LLM tracing")
        self.output("  10. Back")
        choice = self.input_fn("Select configuration option: ").strip().lower()
        if choice == "1":
            self._choose_resolver()
        elif choice == "2":
            self._choose_preset()
        elif choice == "3":
            self._choose_model_profile()
        elif choice == "4":
            self._configure_research()
        elif choice == "5":
            self._configure_runtime()
        elif choice == "6":
            self._configure_official_setup()
        elif choice == "7":
            self._choose_benchmark_source()
        elif choice == "8":
            self._fresh_run = not self._fresh_run
            self._show_status_dialog(f"Fresh run is now {'on' if self._fresh_run else 'off'}.")
        elif choice == "9":
            self.settings.trace_llm = not self.settings.trace_llm
            self._show_status_dialog(f"LLM tracing is now {'on' if self.settings.trace_llm else 'off'}.")
        return 0

    def _configure_official_setup(self) -> None:
        if self._use_prompt_toolkit:
            choice = button_dialog(
                title="Official setup",
                text="Set up official baseline dependencies and services.",
                buttons=[
                    ("Setup local PyEGo Neo4j (recommended)", "pyego-neo4j"),
                    ("Back", "back"),
                ],
                style=UI_STYLE,
            ).run()
            if choice == "pyego-neo4j":
                self._setup_local_pyego_neo4j()
            return

        self.output("\nOfficial baseline setup")
        self.output("  1. Setup local PyEGo Neo4j (recommended)")
        self.output("  2. Back")
        choice = self.input_fn("Select setup option: ").strip().lower()
        if choice == "1":
            self._setup_local_pyego_neo4j()

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
            saved_prompt_profile = str(run_entry.get("prompt_profile", "") or "").strip() or None
            self._apply_preset_config(saved_preset, prompt_profile_override=saved_prompt_profile)
        saved_bundle = str(run_entry.get("research_bundle", "") or "")
        if saved_bundle in RESEARCH_BUNDLE_DEFAULTS:
            self.settings.research_bundle = saved_bundle
        saved_features = run_entry.get("research_features", [])
        if isinstance(saved_features, list):
            self.settings.research_features = tuple(
                feature for feature in saved_features if feature in RESEARCH_FEATURES
            )
        saved_benchmark_source = str(run_entry.get("benchmark_source", "") or "")
        if saved_benchmark_source in {"all-gists", "dockerized-gists", "competition-run"}:
            self.settings.benchmark_case_source = saved_benchmark_source
        self.settings.apply_runtime_config(run_entry)
        if self.settings.research_bundle not in RESEARCH_BUNDLE_DEFAULTS:
            self.settings.research_bundle = "baseline"
        self.settings.research_features = tuple(
            feature for feature in self.settings.research_features if feature in RESEARCH_FEATURES
        )
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

    def _run_failed_cases_from_run(self) -> int:
        if not self._validate_runtime_selection():
            return 1
        source_run = self._prompt_failed_case_run()
        if source_run is None:
            return 1
        jobs = self._prompt_int("Jobs", 1)
        run_id = self._prompt_optional("Run ID", "")
        dashboard = TerminalBenchmarkDashboard()
        return self.run_failed_cases(
            self.settings,
            str(source_run["run_id"]),
            None,
            run_id or None,
            jobs=jobs,
            observer=dashboard,
            notify_paths=False,
            fresh_run=self._fresh_run,
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
            resolver = str(state.get("resolver", "apdr") or "apdr")
            preset = str(state.get("preset", "optimized") or "optimized")
            research_bundle = str(state.get("research_bundle", "baseline") or "baseline")
            benchmark_source = str(state.get("benchmark_source", "all-gists") or "all-gists")
            label = (
                f"{run_id} [{status}] {completed}/{total} target={target} jobs={jobs} "
                f"resolver={resolver} preset={preset}/{research_bundle} source={benchmark_source}"
            )
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
                    "research_bundle": research_bundle,
                    "benchmark_source": benchmark_source,
                    "research_features": state.get("research_features", []),
                }
            )
        return entries

    def _available_failed_case_runs(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for entry in self._available_run_entries():
            run_id = str(entry["run_id"])
            failed_count = 0
            for result_path in (self.settings.artifacts_dir / run_id).glob("*/result.json"):
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if isinstance(payload, dict) and not bool(payload.get("success", False)):
                    failed_count += 1
            if failed_count <= 0:
                continue
            updated = dict(entry)
            updated["failed_count"] = failed_count
            updated["label"] = f"{entry['label']} failed={failed_count}"
            entries.append(updated)
        return entries

    def _loadouts_dir(self) -> Path:
        return self.settings.data_dir / "loadouts"

    def _sanitize_loadout_name(self, raw_name: str) -> str:
        normalized = raw_name.strip()
        normalized = re.sub(r"\s+", "-", normalized)
        normalized = re.sub(r"[^A-Za-z0-9._-]", "", normalized)
        return normalized.strip(".-_")

    def _loadout_path(self, loadout_name: str) -> Path:
        return self._loadouts_dir() / f"{loadout_name}.json"

    def _list_loadout_names(self) -> list[str]:
        loadouts_dir = self._loadouts_dir()
        if not loadouts_dir.exists():
            return []
        names = [path.stem for path in loadouts_dir.glob("*.json") if path.is_file()]
        return sorted(set(name for name in names if name))

    def _serialize_current_loadout(self) -> dict[str, object]:
        payload = {
            "resolver": self.settings.resolver,
            "preset": self.settings.preset,
            "prompt_profile": self.settings.prompt_profile,
            "benchmark_case_source": self.settings.benchmark_case_source,
            "trace_llm": self.settings.trace_llm,
            "pyego_python": self.settings.pyego_python,
            "fresh_run": self._fresh_run,
        }
        payload.update(self.settings.effective_runtime_config())
        return payload

    def _apply_loadout(self, payload: dict[str, object]) -> None:
        resolver = str(payload.get("resolver", self.settings.resolver) or self.settings.resolver)
        if resolver in {"apdr", "pyego", "readpye"}:
            self.settings.resolver = resolver

        preset = str(payload.get("preset", self.settings.preset) or self.settings.preset)
        if preset in PRESET_CONFIGS:
            prompt_profile = str(payload.get("prompt_profile", "") or "").strip() or None
            self._apply_preset_config(preset, prompt_profile_override=prompt_profile)

        benchmark_source = str(
            payload.get("benchmark_case_source", self.settings.benchmark_case_source)
            or self.settings.benchmark_case_source
        )
        if benchmark_source in {"all-gists", "dockerized-gists", "competition-run"}:
            self.settings.benchmark_case_source = benchmark_source
        self.settings.apply_runtime_config(payload)
        effective_model_profile = str(
            payload.get("effective_model_profile", payload.get("model_profile", self.settings.model_profile))
            or self.settings.model_profile
        )
        if effective_model_profile in MODEL_PROFILE_DEFAULTS:
            self.settings.model_profile = effective_model_profile

        for key, attr in (("trace_llm", "trace_llm"), ("fresh_run", "_fresh_run")):
            value = payload.get(key, None)
            if isinstance(value, bool):
                if attr == "_fresh_run":
                    self._fresh_run = value
                else:
                    setattr(self.settings, attr, value)

        pyego_python = str(payload.get("pyego_python", "")).strip()
        if pyego_python:
            self.settings.pyego_python = pyego_python

        if self.settings.research_bundle not in RESEARCH_BUNDLE_DEFAULTS:
            self.settings.research_bundle = "baseline"
        self.settings.research_features = tuple(
            feature for feature in self.settings.research_features if feature in RESEARCH_FEATURES
        )

        if self.settings.preset in {"research", "experimental"} and self.settings.resolver != "apdr":
            self.settings.resolver = "apdr"
        if self.settings.preset != "research":
            self.settings.research_bundle = "baseline"
            self.settings.research_features = ()

    def _prompt_loadout_name(self, label: str) -> str | None:
        candidate = self._prompt_optional(label, "").strip()
        if not candidate:
            self._show_status_dialog("Loadout name is required.")
            return None
        normalized = self._sanitize_loadout_name(candidate)
        if not normalized:
            self._show_status_dialog("Loadout name contains only unsupported characters.")
            return None
        return normalized

    def _prompt_existing_loadout(self, title: str, prompt: str) -> str | None:
        names = self._list_loadout_names()
        if not names:
            self._show_status_dialog(f"No saved loadouts in {self._loadouts_dir()}.")
            return None
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title=title,
                text=prompt,
                values=[(name, name) for name in names],
                default=names[0],
                style=UI_STYLE,
            ).run()
            return selected
        self.output("\nSaved loadouts:")
        for index, name in enumerate(names, start=1):
            self.output(f"  {index}. {name}")
        choice = self.input_fn("Choose loadout number: ").strip()
        if not choice.isdigit():
            self._show_status_dialog("Invalid loadout selection.")
            return None
        index = int(choice) - 1
        if index < 0 or index >= len(names):
            self._show_status_dialog("Invalid loadout selection.")
            return None
        return names[index]

    def _save_current_loadout(self) -> None:
        loadout_name = self._prompt_loadout_name("Loadout name")
        if not loadout_name:
            return
        loadouts_dir = self._loadouts_dir()
        loadouts_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": loadout_name,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "settings": self._serialize_current_loadout(),
        }
        loadout_path = self._loadout_path(loadout_name)
        loadout_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._show_status_dialog(f"Saved loadout '{loadout_name}' to {loadout_path}.")

    def _load_saved_loadout(self) -> None:
        loadout_name = self._prompt_existing_loadout("Load loadout", "Choose a saved loadout to apply.")
        if not loadout_name:
            return
        loadout_path = self._loadout_path(loadout_name)
        try:
            payload = json.loads(loadout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._show_status_dialog(f"Failed to read loadout '{loadout_name}'.")
            return
        if not isinstance(payload, dict):
            self._show_status_dialog(f"Loadout '{loadout_name}' is invalid.")
            return
        settings_payload = payload.get("settings", payload)
        if not isinstance(settings_payload, dict):
            self._show_status_dialog(f"Loadout '{loadout_name}' is invalid.")
            return
        self._apply_loadout(settings_payload)
        self._show_status_dialog(f"Loaded loadout '{loadout_name}'.")

    def _delete_saved_loadout(self) -> None:
        loadout_name = self._prompt_existing_loadout("Delete loadout", "Choose a saved loadout to delete.")
        if not loadout_name:
            return
        loadout_path = self._loadout_path(loadout_name)
        try:
            loadout_path.unlink()
        except OSError as exc:
            self._show_status_dialog(f"Failed to delete loadout '{loadout_name}': {exc}")
            return
        self._show_status_dialog(f"Deleted loadout '{loadout_name}'.")

    def _bootstrap_dir(self) -> Path:
        return self.settings.data_dir / "runtime_bootstrap"

    def _pyego_neo4j_container_name(self) -> str:
        return "apdr-pyego-neo4j"

    def _pyego_neo4j_volume_name(self) -> str:
        return "apdr-pyego-neo4j-data"

    def _pyego_neo4j_loaded_marker(self) -> Path:
        return self._bootstrap_dir() / "pyego-neo4j-loaded.json"

    def _docker_available(self) -> bool:
        return shutil.which("docker") is not None

    def _run_subprocess_checked(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd is not None else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            detail = (stderr or stdout).strip()
            timeout_value = f"{timeout:.0f}s" if isinstance(timeout, (int, float)) else "configured limit"
            if detail:
                raise RuntimeError(
                    f"{' '.join(command)}\nCommand timed out after {timeout_value}.\n{detail}"
                ) from exc
            raise RuntimeError(f"{' '.join(command)}\nCommand timed out after {timeout_value}.") from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or "command failed"
            raise RuntimeError(f"{' '.join(command)}\n{detail}")
        return completed

    def _ensure_pykg_dump_file(self) -> Path:
        pykg_dir = self.settings.pyego_root / "PyKG"
        dump_path = pykg_dir / "PyKG.dump"
        part_paths = sorted(pykg_dir.glob("PyKG.dump.a*"))
        if dump_path.exists():
            if part_paths:
                newest_part = max(part.stat().st_mtime for part in part_paths)
                if dump_path.stat().st_mtime >= newest_part:
                    return dump_path
            else:
                return dump_path
        if not part_paths:
            raise RuntimeError(f"PyKG dump parts not found under {pykg_dir}.")
        pykg_dir.mkdir(parents=True, exist_ok=True)
        with dump_path.open("wb") as destination:
            for part_path in part_paths:
                destination.write(part_path.read_bytes())
        return dump_path

    def _wait_for_local_tcp(self, host: str, port: int, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except OSError:
                time.sleep(1.0)
        return False

    def _is_docker_platform_manifest_error(self, detail: str) -> bool:
        lowered = detail.lower()
        return (
            "no matching manifest for linux/arm64" in lowered
            or "no match for platform in manifest" in lowered
        )

    def _rewrite_pyego_local_neo4j_config(self) -> Path:
        config_path = self.settings.pyego_root / "config.py"
        if not config_path.exists():
            raise RuntimeError(f"PyEGo config not found at {config_path}.")
        content = config_path.read_text(encoding="utf-8")
        replacements = {
            "NEO4J_URI": 'NEO4J_URI = "bolt://localhost:7687"',
            "NEO4J_PWD": "NEO4J_PWD = None",
            "NEO4J_USERNAME": "NEO4J_USERNAME = None",
            "NEO4J_DATABASE": "NEO4J_DATABASE = None",
        }
        updated = content
        for key, replacement in replacements.items():
            pattern = rf"^{key}\s*=.*$"
            if re.search(pattern, updated, flags=re.MULTILINE):
                updated = re.sub(pattern, replacement, updated, flags=re.MULTILINE)
            else:
                if not updated.endswith("\n"):
                    updated += "\n"
                updated += replacement + "\n"
        config_path.write_text(updated, encoding="utf-8")
        return config_path

    def _setup_local_pyego_neo4j(self) -> None:
        if not self._docker_available():
            self._show_status_dialog("Docker CLI not found on PATH. Install Docker Desktop first.")
            return
        try:
            dump_path = self._ensure_pykg_dump_file()
        except Exception as exc:
            self._show_status_dialog(f"Failed preparing PyKG dump: {exc}")
            return

        volume_name = self._pyego_neo4j_volume_name()
        container_name = self._pyego_neo4j_container_name()
        marker_path = self._pyego_neo4j_loaded_marker()
        marker_path.parent.mkdir(parents=True, exist_ok=True)

        reset_choice = self._prompt_choice(
            "Reset and reload local PyEGo Neo4j data volume?",
            ["no", "yes"],
            "no",
        )
        force_reload = reset_choice == "yes"
        messages: list[str] = [f"Using dump: {dump_path}"]
        neo4j_image = "neo4j:3.5.26"

        try:
            if force_reload:
                subprocess.run(
                    ["docker", "volume", "rm", "-f", volume_name],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )

            should_load = force_reload or not marker_path.exists()
            selected_platform = "native"
            attempted_amd64 = False
            platform_flags: list[str] = []
            volume_status = "Existing local Neo4j volume already initialized with PyKG."
            while True:
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )

                    self._run_subprocess_checked(["docker", "volume", "create", volume_name], timeout=30)
                    self._run_subprocess_checked(
                        ["docker", "pull", *platform_flags, neo4j_image],
                        timeout=1800,
                    )

                    if should_load:
                        load_preamble = (
                            "set -e; "
                            "mkdir -p /data/databases /data/databases/graph.db /data/dbms; "
                            "cp /import/PyKG.dump /data/databases/PyKG.dump; "
                            "if command -v neo4j-admin >/dev/null 2>&1; then "
                            "NEO4J_ADMIN=$(command -v neo4j-admin); "
                            "elif [ -x /var/lib/neo4j/bin/neo4j-admin ]; then "
                            "NEO4J_ADMIN=/var/lib/neo4j/bin/neo4j-admin; "
                            "elif [ -x /neo4j/bin/neo4j-admin ]; then "
                            "NEO4J_ADMIN=/neo4j/bin/neo4j-admin; "
                            "else echo 'neo4j-admin command not found'; exit 127; fi; "
                        )
                        primary_load_script = (
                            load_preamble
                            + "$NEO4J_ADMIN load --from=/data/databases/PyKG.dump --database=graph.db --force"
                        )
                        try:
                            self._run_subprocess_checked(
                                [
                                    "docker",
                                    "run",
                                    "--rm",
                                    *platform_flags,
                                    "-v",
                                    f"{volume_name}:/data",
                                    "-v",
                                    f"{dump_path.parent.resolve()}:/import",
                                    neo4j_image,
                                    "bash",
                                    "-lc",
                                    primary_load_script,
                                ],
                                timeout=3600,
                            )
                        except Exception as load_exc:
                            if "NoSuchFileException: /data/databases/graph.db" not in str(load_exc):
                                raise
                            messages.append(
                                "Primary graph.db load mode failed; retrying Neo4j 3.5 compatibility load mode."
                            )
                            compatibility_load_script = (
                                load_preamble + "$NEO4J_ADMIN load --from=/data/databases/PyKG.dump --force"
                            )
                            self._run_subprocess_checked(
                                [
                                    "docker",
                                    "run",
                                    "--rm",
                                    *platform_flags,
                                    "-v",
                                    f"{volume_name}:/data",
                                    "-v",
                                    f"{dump_path.parent.resolve()}:/import",
                                    neo4j_image,
                                    "bash",
                                    "-lc",
                                    compatibility_load_script,
                                ],
                                timeout=3600,
                            )
                        marker_path.write_text(
                            json.dumps(
                                {
                                    "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "volume": volume_name,
                                    "dump_path": str(dump_path),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        volume_status = "Loaded PyKG into local Neo4j volume."

                    self._run_subprocess_checked(
                        [
                            "docker",
                            "run",
                            "-d",
                            *platform_flags,
                            "--name",
                            container_name,
                            "-p",
                            "7474:7474",
                            "-p",
                            "7687:7687",
                            "-e",
                            "NEO4J_AUTH=none",
                            "-v",
                            f"{volume_name}:/data",
                            neo4j_image,
                        ],
                        timeout=300,
                    )
                    break
                except Exception as exc:
                    if attempted_amd64 or not self._is_docker_platform_manifest_error(str(exc)):
                        raise
                    attempted_amd64 = True
                    platform_flags = ["--platform", "linux/amd64"]
                    selected_platform = "linux/amd64 emulation"
                    messages.append(
                        "Neo4j image has no native arm64 manifest; retrying with linux/amd64 emulation."
                    )

            messages.append(volume_status)
            messages.append(f"Docker platform mode: {selected_platform}")
            if not self._wait_for_local_tcp("127.0.0.1", 7687, timeout_seconds=60):
                raise RuntimeError("Neo4j container started, but Bolt port 7687 did not become ready in time.")
            config_path = self._rewrite_pyego_local_neo4j_config()
            messages.append(f"Updated PyEGo config: {config_path}")

            self.settings.resolver = "pyego"
            version_text = self._ensure_ui_pyego_python311()
            if version_text:
                messages.append(f"Using PyEGo Python: {self.settings.pyego_python} ({version_text})")
            messages.append("Local PyEGo Neo4j setup complete.")
            self._show_status_dialog("\n".join(messages))
        except Exception as exc:
            self._show_status_dialog(
                "Failed to set up local PyEGo Neo4j automatically.\n\n"
                f"{exc}\n\n"
                "Tip: inspect docker logs with:\n"
                f"docker logs {container_name}"
            )

    def _official_runtime_paths(self, resolver: str) -> tuple[str, Path] | None:
        if resolver == "pyego":
            return self.settings.pyego_python, self.settings.pyego_root / "requirements.txt"
        if resolver == "readpye":
            return self.settings.readpye_python, self.settings.readpye_root / "requirements.txt"
        return None

    def _ensure_python311_for_resolver(self, resolver: str) -> tuple[str, tuple[int, int, int] | None]:
        if resolver == "pyego":
            self._ensure_ui_pyego_python311()
            python_exec = self.settings.pyego_python
        elif resolver == "readpye":
            version = self._probe_python_version(self.settings.readpye_python)
            if version is None or version[:2] != (3, 11):
                selected, _ = self._discover_python311_for_pyego()
                if selected:
                    self.settings.readpye_python = selected
            python_exec = self.settings.readpye_python
        else:
            return "", None
        return python_exec, self._probe_python_version(python_exec)

    def _bootstrap_marker_path(self, resolver: str, python_exec: str, requirements_path: Path) -> Path:
        digest = hashlib.sha1()
        digest.update(resolver.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(Path(python_exec)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(requirements_path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(requirements_path.read_bytes())
        return self._bootstrap_dir() / f"{resolver}-{digest.hexdigest()[:16]}.json"

    def _maybe_auto_install_official_requirements(self, resolver: str) -> tuple[bool, str]:
        runtime = self._official_runtime_paths(resolver)
        if runtime is None:
            return True, f"{resolver} has no official requirements bootstrap."
        _, requirements_path = runtime
        if not requirements_path.exists():
            return True, f"Skipped auto-install: requirements file not found at {requirements_path}."

        python_exec, version = self._ensure_python311_for_resolver(resolver)
        if version is None:
            return True, f"Skipped auto-install: unable to probe interpreter '{python_exec}'."
        if version[:2] != (3, 11):
            return (
                True,
                f"Skipped auto-install: {python_exec} is Python {version[0]}.{version[1]}.{version[2]} (need 3.11).",
            )

        marker_path = self._bootstrap_marker_path(resolver, python_exec, requirements_path)
        if marker_path.exists():
            return True, f"{resolver} requirements already bootstrapped for {python_exec}."

        install = subprocess.run(
            [python_exec, "-m", "pip", "install", "-r", str(requirements_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if install.returncode != 0:
            stderr = (install.stderr or "").strip()
            stdout = (install.stdout or "").strip()
            details = stderr or stdout or "pip install failed"
            return False, f"Auto-install failed for {resolver}: {details}"

        bootstrap_dir = self._bootstrap_dir()
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps(
                {
                    "resolver": resolver,
                    "python": python_exec,
                    "requirements": str(requirements_path),
                    "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return True, f"Auto-installed {resolver} requirements with {python_exec}."

    def _default_pyego_venv_python(self) -> Path:
        if sys.platform.startswith("win"):
            return self.settings.project_root / ".venv-pyego" / "Scripts" / "python.exe"
        return self.settings.project_root / ".venv-pyego" / "bin" / "python"

    def _probe_python_version(self, executable: str) -> tuple[int, int, int] | None:
        try:
            completed = subprocess.run(
                [
                    executable,
                    "-c",
                    "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=5,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        version_text = (completed.stdout or "").strip()
        parts = version_text.split(".")
        if len(parts) < 2:
            return None
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            return None
        return major, minor, patch

    def _discover_python311_for_pyego(self) -> tuple[str | None, tuple[int, int, int] | None]:
        candidates: list[str] = []
        default_venv = self._default_pyego_venv_python()
        if default_venv.exists():
            candidates.append(str(default_venv))
        if self.settings.pyego_python:
            candidates.append(self.settings.pyego_python)
        for command in ("python3.11", "python311", "python3", "python"):
            resolved = shutil.which(command)
            if resolved:
                candidates.append(resolved)
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        for candidate in deduped:
            version = self._probe_python_version(candidate)
            if version and version[:2] == (3, 11):
                return candidate, version
        return None, None

    def _ensure_ui_pyego_python311(self) -> str | None:
        selected, version = self._discover_python311_for_pyego()
        if selected is None:
            return None
        self.settings.pyego_python = selected
        version_text = f"{version[0]}.{version[1]}.{version[2]}" if version else "3.11"
        return version_text

    def _pyego_setup_instructions(self) -> str:
        if sys.platform.startswith("win"):
            return (
                "py -3.11 -m venv .venv-pyego\n"
                ".venv-pyego\\Scripts\\python -m pip install --upgrade pip\n"
                ".venv-pyego\\Scripts\\python -m pip install -r external\\PyEGo\\requirements.txt"
            )
        return (
            "python3.11 -m venv .venv-pyego\n"
            "source .venv-pyego/bin/activate\n"
            "python -m pip install --upgrade pip\n"
            "python -m pip install -r external/PyEGo/requirements.txt"
        )

    def _is_pyego_neo4j_connectivity_failure(self, detail: str) -> bool:
        lowered = detail.lower()
        markers = (
            "cannot reach neo4j",
            "cannot open connection to",
            "connection refused",
            "configured uri:",
            "py2neo",
        )
        return any(marker in lowered for marker in markers)

    def _validate_runtime_selection(self) -> bool:
        if self.settings.preset in {"research", "experimental"} and self.settings.resolver != "apdr":
            self._show_status_dialog("Research and experimental presets are only supported with the apdr resolver.")
            return False
        if self.settings.resolver == "pyego":
            install_ok, install_detail = self._maybe_auto_install_official_requirements("pyego")
            if not install_ok:
                self._show_status_dialog(install_detail)
                return False
            self._ensure_ui_pyego_python311()
            pyego_ok, pyego_detail = validate_pyego_runtime(self.settings)
            if not pyego_ok and self._is_pyego_neo4j_connectivity_failure(pyego_detail):
                auto_fix = self._prompt_choice(
                    "PyEGo cannot reach Neo4j. Auto-setup local Neo4j now?",
                    ["yes", "no"],
                    "yes",
                )
                if auto_fix == "yes":
                    self._setup_local_pyego_neo4j()
                    pyego_ok, pyego_detail = validate_pyego_runtime(self.settings)
            if not pyego_ok:
                default_venv = self._default_pyego_venv_python()
                self._show_status_dialog(
                    "PyEGo runtime check failed.\n\n"
                    f"{install_detail}\n\n"
                    f"{pyego_detail}\n\n"
                    "Set up a dedicated Python 3.11 env, then retry from the UI:\n"
                    f"{self._pyego_setup_instructions()}\n\n"
                    f"APDR auto-detects: {default_venv}\n\n"
                    "For automatic local Neo4j setup: Configure -> Official setup -> Setup local PyEGo Neo4j."
                )
                return False
        if self.settings.resolver == "readpye":
            install_ok, install_detail = self._maybe_auto_install_official_requirements("readpye")
            if not install_ok:
                self._show_status_dialog(install_detail)
                return False
            if not (self.settings.readpye_root / "run.py").exists():
                self._show_status_dialog(f"ReadPyE entrypoint missing at {self.settings.readpye_root / 'run.py'}.")
                return False
            if not self.settings.readpye_language_dir:
                self._show_status_dialog("ReadPyE requires APDR_READPYE_LANGDIR to be configured.")
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

    def _prompt_failed_case_run(self) -> dict[str, object] | None:
        run_entries = self._available_failed_case_runs()
        if not run_entries:
            self._show_status_dialog("No prior runs with failed cases were found.")
            return None
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title="Retry failed cases",
                text="Choose a prior run and APDR will rerun only its failed benchmark cases.",
                values=[(str(entry["run_id"]), str(entry["label"])) for entry in run_entries],
                default=str(run_entries[0]["run_id"]),
                style=UI_STYLE,
            ).run()
            if selected is None:
                return None
            return next((entry for entry in run_entries if entry["run_id"] == selected), None)

        self.output("\nRuns with failed cases:")
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
        self._apply_preset_config(selected)
        if selected in {"research", "experimental"} and self.settings.resolver != "apdr":
            self.settings.resolver = "apdr"
        if selected != "research":
            self.settings.research_bundle = "baseline"
            self.settings.research_features = ()
        self._show_status_dialog(f"Preset switched to {selected}.")

    def _choose_resolver(self) -> None:
        options = [
            ("apdr", "apdr"),
            ("pyego", "pyego (official only)"),
            ("readpye", "readpye (official repo if available)"),
        ]
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
        if selected != "apdr" and self.settings.preset in {"research", "experimental"}:
            self._apply_preset_config("accuracy")
            self.settings.research_bundle = "baseline"
            self.settings.research_features = ()
        message = f"Resolver switched to {selected}."
        if selected == "pyego":
            version_text = self._ensure_ui_pyego_python311()
            if version_text:
                message += (
                    "\n"
                    f"PyEGo interpreter set to {self.settings.pyego_python} "
                    f"(Python {version_text})."
                )
            else:
                message += (
                    "\nNo Python 3.11 interpreter was auto-detected yet. "
                    "Runs will be blocked until PyEGo runtime preflight passes."
                )
        self._show_status_dialog(message)

    def _choose_benchmark_source(self) -> None:
        options = [
            ("all-gists", "all-gists (default)"),
            ("dockerized-gists", "dockerized-gists"),
            ("competition-run", "competition-run (all-gists filtered by official CSV gist ids)"),
        ]
        if self._use_prompt_toolkit:
            selected = radiolist_dialog(
                title="Benchmark source",
                text="Choose which Gistable case collection APDR runs.",
                values=options,
                default=self.settings.benchmark_case_source,
                style=UI_STYLE,
            ).run()
            if selected is None:
                return
        else:
            self.output("\nBenchmark case sources:")
            visible_sources = [source for source, _ in options]
            for index, source in enumerate(visible_sources, start=1):
                marker = "*" if source == self.settings.benchmark_case_source else " "
                self.output(f"  {index}. [{marker}] {source}")
            choice = self.input_fn("Choose source number: ").strip()
            if not choice.isdigit():
                return
            index = int(choice) - 1
            if index < 0 or index >= len(visible_sources):
                return
            selected = visible_sources[index]
        self.settings.benchmark_case_source = selected
        self._show_status_dialog(f"Benchmark source switched to {selected}.")

    def _configure_research(self) -> None:
        if self.settings.preset != "research":
            self._show_status_dialog("Research controls are only available when the preset is research.")
            return
        if self._use_prompt_toolkit:
            bundle = radiolist_dialog(
                title="Research bundle",
                text="Choose the research bundle for the next runs.",
                values=[(bundle_name, bundle_name) for bundle_name in RESEARCH_BUNDLE_DEFAULTS],
                default=self.settings.research_bundle,
                style=UI_STYLE,
            ).run()
            if bundle is None:
                return
            selected_features = checkboxlist_dialog(
                title="Research features",
                text="Enable or disable research feature flags.",
                values=[(feature, feature) for feature in RESEARCH_FEATURES],
                default_values=list(self.settings.research_features or RESEARCH_BUNDLE_DEFAULTS[bundle]),
                style=UI_STYLE,
            ).run()
            if selected_features is None:
                return
        else:
            bundle = self._prompt_choice("Research bundle", list(RESEARCH_BUNDLE_DEFAULTS), self.settings.research_bundle)
            default_value = ",".join(self.settings.research_features or RESEARCH_BUNDLE_DEFAULTS[bundle])
            raw_features = self._prompt_optional("Research features (comma-separated)", default_value)
            selected_features = tuple(
                feature.strip() for feature in raw_features.split(",") if feature.strip() in RESEARCH_FEATURES
            )
        self.settings.research_bundle = bundle
        self.settings.research_features = tuple(
            feature for feature in selected_features if feature in RESEARCH_FEATURES
        )
        self._show_status_dialog(
            f"Research bundle set to {bundle} with {len(self.settings.research_features)} feature(s)."
        )

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
            "<b><ansibrightyellow>APDR Command Center</ansibrightyellow></b>\n"
            "<style fg='#98c1d9'>Run, report, and configure without memorizing commands.</style>\n\n"
            f"<b>Version:</b> {self._app_version}\n"
            f"<b>Preset/Resolver:</b> {self.settings.preset} / {self.settings.resolver}\n"
            f"<b>Benchmark source:</b> {self.settings.benchmark_case_source}\n"
            f"<b>Model bundle:</b> {self.settings.model_profile}\n"
            f"<b>Models:</b> {self.settings.extraction_model} / {self.settings.reasoning_model}\n"
            f"<b>Runtime:</b> MoE {'on' if self.settings.use_moe else 'off'} | "
            f"RAG {'on' if self.settings.use_rag else 'off'} | "
            f"LangChain {'on' if self.settings.use_langchain else 'off'}\n"
            f"<b>Research:</b> {self.settings.research_bundle} "
            f"({len(self.settings.research_features)} feature(s))\n"
            f"<b>PyEGo Python:</b> {self.settings.pyego_python}\n"
            f"<b>Loadouts:</b> {len(self._list_loadout_names())}\n"
            f"<b>Fresh run:</b> {'on' if self._fresh_run else 'off'} | "
            f"<b>Trace LLM:</b> {'on' if self.settings.trace_llm else 'off'}\n"
            f"<b>Artifacts:</b> {self.settings.artifacts_dir}"
        )

    def _show_status_dialog(self, text: str) -> None:
        if self._use_prompt_toolkit:
            message_dialog(title="APDR", text=text, style=UI_STYLE).run()
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
            value = input_dialog(title="APDR", text=f"{label} [{default}]", style=UI_STYLE).run()
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
                title="APDR",
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
        title = "APDR Command Center"
        self.output("=" * width)
        self.output(title.center(width))
        self.output("=" * width)
        self.output(f"Version: {self._app_version}")
        self.output(f"Preset: {self.settings.preset}")
        self.output(f"Resolver: {self.settings.resolver}")
        self.output(f"Benchmark source: {self.settings.benchmark_case_source}")
        self.output(f"Model bundle: {self.settings.model_profile}")
        self.output(f"Research bundle: {self.settings.research_bundle}")
        self.output(f"Research features: {', '.join(self.settings.research_features) or 'none'}")
        self.output(f"PyEGo Python: {self.settings.pyego_python}")
        self.output(f"Loadouts: {len(self._list_loadout_names())}")
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
        self.output("  2. Run workflows")
        self.output("  3. Reports")
        self.output("  4. Configure")
        self.output("  5. Loadouts")
        self.output("  8. Quit")


def launch_terminal_ui(
    settings: Settings,
    doctor_command: ActionCallback,
    run_benchmark: ActionCallback,
    run_failed_cases: ActionCallback,
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
        run_failed_cases=run_failed_cases,
        run_project=run_project,
        summarize_command=summarize_command,
        failures_command=failures_command,
        modules_command=modules_command,
        timeline_command=timeline_command,
        ensure_smoke_subset=ensure_smoke_subset,
    )
    return ui.run()
