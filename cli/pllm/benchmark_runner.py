from __future__ import annotations

import ast
import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Protocol

from cli.pllm.benchmark_data import BenchmarkSource, list_case_ids, resolve_snippet_path
from cli.pllm.core import ROOT_DIR, RunConfig, stream_executor

LineHandler = Callable[[str], None]
_OFFICIAL_PLLM_RESULTS_CSV = ROOT_DIR / "data" / "benchmarks" / "gistable" / "competition" / "summary-all-runs.csv"
_OFFICIAL_CASE_ID_COLUMNS = ("name", "gist_id", "gistid", "case_id", "id")
_OFFICIAL_CASE_TOKEN_PATTERN = re.compile(r"^[0-9A-Za-z]{6,40}$")


class BenchmarkObserver(Protocol):
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
    ) -> None: ...

    def case_started(self, case_id: str) -> None: ...

    def advance(self, result: dict[str, object]) -> None: ...

    def stop_requested(self) -> bool: ...

    def finish(self, *, summary: "BenchmarkSummary", status: str = "completed") -> None: ...


@dataclass
class BenchmarkSummary:
    source: BenchmarkSource
    total_selected: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    elapsed_seconds: float
    run_id: str = ""
    report_dir: str = ""
    summary_path: str = ""
    summary_csv_path: str = ""
    results_csv_path: str = ""
    results_json_path: str = ""
    results_markdown_path: str = ""
    leaderboard_markdown_path: str = ""
    timeline_json_path: str = ""
    timeline_csv_path: str = ""
    timeline_markdown_path: str = ""
    llm_trace_path: str = ""
    warnings_path: str = ""
    run_vs_official_csv_path: str = ""
    report_markdown_path: str = ""


def run_benchmark(
    *,
    source: BenchmarkSource,
    model: str,
    base: str,
    temp: float,
    loop: int,
    search_range: int,
    rag: bool,
    verbose: bool,
    limit: int = 0,
    offset: int = 0,
    fail_fast: bool = False,
    show_case_output: bool = False,
    line_handler: LineHandler | None = None,
    observer: BenchmarkObserver | None = None,
) -> tuple[int, BenchmarkSummary]:
    run_id = f"pllm-bench-{int(time.time())}"
    run_dir = _prepare_run_dir(run_id)
    case_ids = list_case_ids(source)
    if offset > 0:
        case_ids = case_ids[offset:]
    if limit > 0:
        case_ids = case_ids[:limit]
    rows: list[dict[str, object]] = []
    run_started_at = _utc_now_iso()
    run_trace_path = run_dir / "llm-trace.log"
    warnings_path = run_dir / "warnings.log"
    warnings_written = False
    _initialize_trace_log(
        trace_path=run_trace_path,
        run_id=run_id,
        source=source,
        model=model,
        base=base,
        loop=loop,
        search_range=search_range,
        rag=rag,
        verbose=verbose,
    )
    current_cases: list[str] = []
    last_case_id = ""
    last_status = "starting"
    stop_requested = False

    total_selected = len(case_ids)

    _refresh_runtime_artifacts(
        run_dir=run_dir,
        run_id=run_id,
        source=source,
        model=model,
        base=base,
        temp=temp,
        loop=loop,
        search_range=search_range,
        rag=rag,
        verbose=verbose,
        total_selected=total_selected,
        attempted=0,
        succeeded=0,
        failed=0,
        skipped=0,
        elapsed_seconds=0.0,
        rows=rows,
        started_at=run_started_at,
        current_cases=current_cases,
        last_case_id=last_case_id,
        last_status=last_status,
        status="running" if total_selected > 0 else "empty",
        stop_requested=False,
        llm_trace_path=str(run_trace_path),
        warnings_path=None,
    )

    if observer is not None:
        observer.start(
            run_id=run_id,
            total=total_selected,
            source=source,
            model=model,
            loop=loop,
            search_range=search_range,
            rag=rag,
            verbose=verbose,
            artifacts_dir=run_dir,
        )
    if total_selected == 0:
        summary = BenchmarkSummary(
            source=source,
            total_selected=0,
            attempted=0,
            succeeded=0,
            failed=0,
            skipped=0,
            elapsed_seconds=0.0,
            run_id=run_id,
        )
        _refresh_runtime_artifacts(
            run_dir=run_dir,
            run_id=run_id,
            source=source,
            model=model,
            base=base,
            temp=temp,
            loop=loop,
            search_range=search_range,
            rag=rag,
            verbose=verbose,
            total_selected=0,
            attempted=0,
            succeeded=0,
            failed=0,
            skipped=0,
            elapsed_seconds=0.0,
            rows=rows,
            started_at=run_started_at,
            current_cases=[],
            last_case_id="",
            last_status="no_cases_selected",
            status="empty",
            stop_requested=False,
            llm_trace_path=str(run_trace_path),
            warnings_path=None,
        )
        summary.report_dir = str(run_dir)
        summary.summary_path = str(run_dir / "summary.json")
        summary.summary_csv_path = str(run_dir / "summary.csv")
        summary.results_csv_path = str(run_dir / "results.csv")
        summary.results_json_path = str(run_dir / "results.json")
        summary.results_markdown_path = str(run_dir / "results.md")
        summary.leaderboard_markdown_path = str(run_dir / "leaderboard.md")
        summary.timeline_json_path = str(run_dir / "timeline.json")
        summary.timeline_csv_path = str(run_dir / "timeline.csv")
        summary.timeline_markdown_path = str(run_dir / "timeline.md")
        summary.llm_trace_path = str(run_dir / "llm-trace.log")
        summary.warnings_path = ""
        summary.run_vs_official_csv_path = str(run_dir / "run-vs-official.csv")
        summary.report_markdown_path = str(run_dir / "report.md")
        if observer is not None:
            observer.finish(summary=summary, status="empty")
        return 1, summary

    attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0
    elapsed_seconds = 0.0

    for index, case_id in enumerate(case_ids, start=1):
        if observer is not None and observer.stop_requested():
            _emit(line_handler, "Stop requested, ending benchmark loop.")
            stop_requested = True
            break
        current_cases = [case_id]
        _refresh_runtime_artifacts(
            run_dir=run_dir,
            run_id=run_id,
            source=source,
            model=model,
            base=base,
            temp=temp,
            loop=loop,
            search_range=search_range,
            rag=rag,
            verbose=verbose,
            total_selected=total_selected,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            elapsed_seconds=elapsed_seconds,
            rows=rows,
            started_at=run_started_at,
            current_cases=current_cases,
            last_case_id=last_case_id,
            last_status=last_status,
            status="running",
            stop_requested=stop_requested,
            llm_trace_path=str(run_trace_path),
            warnings_path=str(warnings_path) if warnings_written else None,
        )

        snippet_path = resolve_snippet_path(case_id, source=source)
        if snippet_path is None:
            skipped += 1
            now = _utc_now_iso()
            row = {
                "index": index,
                "case_id": case_id,
                "status": "skipped",
                "success": False,
                "return_code": 2,
                "elapsed_seconds": 0.0,
                "started_at": now,
                "finished_at": now,
                "source_path": "",
                "trace_path": str(run_dir / case_id / "llm-trace.log"),
            }
            _append_log_line(run_trace_path, f"[{_utc_now_iso()}] CASE SKIPPED {case_id} (missing snippet)")
            _append_log_line(run_dir / case_id / "llm-trace.log", f"[{_utc_now_iso()}] skipped: missing snippet")
            _append_log_line(warnings_path, f"{now} {case_id} skipped: missing snippet")
            warnings_written = True
            rows.append(row)
            _write_case_result(run_dir=run_dir, run_id=run_id, source=source, row=row)
            last_case_id = case_id
            last_status = "skipped"
            current_cases = []
            _refresh_runtime_artifacts(
                run_dir=run_dir,
                run_id=run_id,
                source=source,
                model=model,
                base=base,
                temp=temp,
                loop=loop,
                search_range=search_range,
                rag=rag,
                verbose=verbose,
                total_selected=total_selected,
                attempted=attempted,
                succeeded=succeeded,
                failed=failed,
                skipped=skipped,
                elapsed_seconds=elapsed_seconds,
                rows=rows,
                started_at=run_started_at,
                current_cases=current_cases,
                last_case_id=last_case_id,
                last_status=last_status,
                status="running",
                stop_requested=stop_requested,
                llm_trace_path=str(run_trace_path),
                warnings_path=str(warnings_path) if warnings_written else None,
            )
            _emit(line_handler, f"[{index}/{total_selected}] {case_id} skipped (missing snippet)")
            if observer is not None:
                observer.advance(
                    {
                        "case_id": case_id,
                        "success": False,
                        "status": "skipped",
                        "return_code": 2,
                        "elapsed_seconds": 0.0,
                    }
                )
            continue

        if observer is not None:
            observer.case_started(case_id)
        _emit(line_handler, f"[{index}/{total_selected}] running {case_id}")
        source_copy_path = _write_case_source(run_dir=run_dir, case_id=case_id, snippet_path=snippet_path)
        case_trace_path = run_dir / case_id / "llm-trace.log"
        _append_log_line(run_trace_path, f"[{_utc_now_iso()}] CASE START {case_id}")
        _append_log_line(case_trace_path, f"[{_utc_now_iso()}] CASE START {case_id}")
        config = RunConfig(
            file=str(snippet_path),
            model=model,
            base=base,
            temp=temp,
            loop=loop,
            search_range=search_range,
            rag=rag,
            verbose=verbose,
        )

        case_output_lines: list[str] = []
        case_env = {
            "PLLM_CASE_ARTIFACT_DIR": str(run_dir / case_id),
            "PLLM_RUN_ID": run_id,
            "PLLM_CASE_ID": case_id,
        }

        def callback(line: str, _stats: object) -> None:
            case_output_lines.append(line)
            _append_log_line(run_trace_path, line)
            _append_log_line(case_trace_path, line)
            if show_case_output:
                _emit(line_handler, line)

        case_started_at = _utc_now_iso()
        return_code, stats = stream_executor(config, line_callback=callback, env=case_env)
        finished_at = _utc_now_iso()
        attempt_meta = _materialize_case_attempt_artifacts(
            run_dir=run_dir,
            case_id=case_id,
            snippet_path=snippet_path,
            case_output_lines=case_output_lines,
        )
        attempted += 1
        elapsed_seconds += stats.elapsed_seconds
        success = return_code == 0
        status = "success" if success else "failed"
        row = {
            "index": index,
            "case_id": case_id,
            "status": status,
            "success": success,
            "return_code": return_code,
            "elapsed_seconds": round(stats.elapsed_seconds, 6),
            "started_at": case_started_at,
            "finished_at": finished_at,
            "source_path": str(source_copy_path) if source_copy_path is not None else "",
            "trace_path": str(case_trace_path),
            "run_log_path": attempt_meta.get("run_log_path", ""),
            "build_log_path": attempt_meta.get("build_log_path", ""),
            "dockerfile_path": attempt_meta.get("dockerfile_path", ""),
            "model_outputs_path": attempt_meta.get("model_outputs_path", ""),
            "prompt_a_path": attempt_meta.get("prompt_a_path", ""),
            "prompt_b_path": attempt_meta.get("prompt_b_path", ""),
            "attempt_count": attempt_meta.get("attempt_count", 1),
        }
        _append_log_line(run_trace_path, f"[{_utc_now_iso()}] CASE END {case_id} rc={return_code}")
        _append_log_line(case_trace_path, f"[{_utc_now_iso()}] case end rc={return_code}")
        rows.append(row)
        _write_case_result(run_dir=run_dir, run_id=run_id, source=source, row=row)
        last_case_id = case_id
        last_status = status
        current_cases = []

        if observer is not None:
            observer.advance(
                {
                    "case_id": case_id,
                    "success": success,
                    "status": status,
                    "return_code": return_code,
                    "elapsed_seconds": stats.elapsed_seconds,
                }
            )

        if success:
            succeeded += 1
            _emit(line_handler, f"[{index}/{total_selected}] {case_id} ok ({stats.elapsed_seconds:.1f}s)")
        else:
            failed += 1
            _append_log_line(
                warnings_path,
                f"{finished_at} {case_id} failed: return_code={return_code} elapsed={stats.elapsed_seconds:.6f}",
            )
            warnings_written = True
            _emit(
                line_handler,
                f"[{index}/{total_selected}] {case_id} failed rc={return_code} ({stats.elapsed_seconds:.1f}s)",
            )
            if fail_fast:
                break
        _refresh_runtime_artifacts(
            run_dir=run_dir,
            run_id=run_id,
            source=source,
            model=model,
            base=base,
            temp=temp,
            loop=loop,
            search_range=search_range,
            rag=rag,
            verbose=verbose,
            total_selected=total_selected,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            elapsed_seconds=elapsed_seconds,
            rows=rows,
            started_at=run_started_at,
            current_cases=current_cases,
            last_case_id=last_case_id,
            last_status=last_status,
            status="running",
            stop_requested=stop_requested,
            llm_trace_path=str(run_trace_path),
            warnings_path=str(warnings_path) if warnings_written else None,
        )

    summary = BenchmarkSummary(
        source=source,
        total_selected=total_selected,
        attempted=attempted,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        elapsed_seconds=elapsed_seconds,
        run_id=run_id,
    )
    final_status = "stopped" if stop_requested else "completed"
    warnings_path_value = str(warnings_path) if warnings_written else None
    _append_log_line(
        run_trace_path,
        f"[{_utc_now_iso()}] RUN END status={final_status} attempted={attempted} succeeded={succeeded} failed={failed} skipped={skipped}",
    )
    _write_run_reports(
        run_dir=run_dir,
        run_id=run_id,
        source=source,
        model=model,
        base=base,
        temp=temp,
        loop=loop,
        search_range=search_range,
        rag=rag,
        verbose=verbose,
        summary=summary,
        rows=rows,
        status=final_status,
        started_at=run_started_at,
        llm_trace_path=str(run_trace_path),
        warnings_path=warnings_path_value,
    )
    _write_run_state(
        run_dir=run_dir,
        payload=_build_run_state_payload(
            run_id=run_id,
            status=final_status,
            source=source,
            model=model,
            base=base,
            temp=temp,
            loop=loop,
            search_range=search_range,
            rag=rag,
            verbose=verbose,
            total_selected=total_selected,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            elapsed_seconds=elapsed_seconds,
            started_at=run_started_at,
            current_cases=[],
            last_case_id=last_case_id,
            last_status=last_status,
            summary=summary,
            completed=len(rows),
            stop_requested=stop_requested,
            llm_trace_path=str(run_trace_path),
            warnings_path=warnings_path_value,
        ),
    )
    if observer is not None:
        status = "stopped" if observer.stop_requested() or stop_requested else "completed"
        observer.finish(summary=summary, status=status)
    return (0 if failed == 0 else 1), summary


def _emit(handler: LineHandler | None, line: str) -> None:
    if handler is None:
        print(line)
    else:
        handler(line)


def _prepare_run_dir(run_id: str) -> Path:
    run_dir = ROOT_DIR / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_run_reports(
    *,
    run_dir: Path,
    run_id: str,
    source: BenchmarkSource,
    model: str,
    base: str,
    temp: float,
    loop: int,
    search_range: int,
    rag: bool,
    verbose: bool,
    summary: BenchmarkSummary,
    rows: list[dict[str, object]],
    status: str = "completed",
    started_at: str | None = None,
    llm_trace_path: str = "",
    warnings_path: str | None = None,
) -> None:
    summary_path = run_dir / "summary.json"
    summary_csv_path = run_dir / "summary.csv"
    results_json_path = run_dir / "results.json"
    results_csv_path = run_dir / "results.csv"
    run_vs_official_csv_path = run_dir / "run-vs-official.csv"
    results_markdown_path = run_dir / "results.md"
    leaderboard_markdown_path = run_dir / "leaderboard.md"
    timeline_json_path = run_dir / "timeline.json"
    timeline_csv_path = run_dir / "timeline.csv"
    timeline_markdown_path = run_dir / "timeline.md"
    report_markdown_path = run_dir / "report.md"
    now = _utc_now_iso()
    normalized_rows = [_normalize_result_row(row) for row in rows]
    normalized_rows.sort(key=lambda row: (row["index"], row["case_id"]))
    results_rows = [_to_results_row(row, case_number=index) for index, row in enumerate(normalized_rows, start=1)]

    summary_payload = {
        "run_id": run_id,
        "status": status,
        "source": source,
        "benchmark_source": source,
        "total_selected": summary.total_selected,
        "total": summary.total_selected,
        "completed": len(rows),
        "attempted": summary.attempted,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "successes": summary.succeeded,
        "failures": summary.failed,
        "skipped": summary.skipped,
        "success_rate": (summary.succeeded / summary.attempted) if summary.attempted else 0.0,
        "elapsed_seconds": round(summary.elapsed_seconds, 6),
        "started_at": started_at or now,
        "last_updated_at": now,
        "model": model,
        "base": base,
        "temp": temp,
        "loop": loop,
        "search_range": search_range,
        "rag": rag,
        "verbose": verbose,
        "model_summary": model,
        "artifacts_dir": str(run_dir),
        "llm_trace_path": llm_trace_path,
        "warnings_path": warnings_path or "",
        "run_vs_official_csv_path": str(run_vs_official_csv_path),
        "generated_at": now,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "success",
                "attempts",
                "initial_eval",
                "final_error_category",
                "wall_clock_seconds",
                "started_at",
                "finished_at",
            ],
        )
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "success": row["success"],
                    "attempts": row["attempts"],
                    "initial_eval": row["initial_eval"],
                    "final_error_category": row["final_error_category"],
                    "wall_clock_seconds": row["wall_clock_seconds"],
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                }
            )

    results_json_path.write_text(json.dumps({"run_id": run_id, "rows": results_rows}, indent=2), encoding="utf-8")

    with results_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_number",
                "case_id",
                "modules",
                "result",
                "attempts",
                "initial_eval",
                "final_error_category",
                "dependencies",
                "dependency_reason",
                "research_bundle",
                "research_features",
                "selected_candidate_rank",
                "strategy_type",
                "retry_severity",
                "conflict_precheck_failed",
                "status",
                "return_code",
                "wall_clock_seconds",
                "started_at",
                "finished_at",
                "source_path",
                "trace_path",
                "run_log_path",
                "build_log_path",
                "dockerfile_path",
                "model_outputs_path",
                "prompt_a_path",
                "prompt_b_path",
                "attempt_count",
            ],
        )
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    _write_run_vs_official_csv(path=run_vs_official_csv_path, run_rows=normalized_rows)

    result_markdown_lines = [
        "# Case Results",
        "",
        f"Run ID: `{run_id}`",
        "",
        "| # | Case ID | Modules | Result | Attempts | Final Status | Dependencies | Time (s) |",
        "| ---: | --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in results_rows:
        result_markdown_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_number"]),
                    str(row["case_id"]),
                    str(row["modules"] or "-"),
                    str(row["result"]),
                    str(row["attempts"]),
                    str(row["final_error_category"] or "-"),
                    str(row["dependencies"] or "-"),
                    f"{float(row['wall_clock_seconds']):.2f}",
                ]
            )
            + " |"
        )
    results_markdown_path.write_text("\n".join(result_markdown_lines) + "\n", encoding="utf-8")

    leaderboard_lines = [
        "# Benchmark Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Source: `{source}`",
        f"- Model: `{model}`",
        f"- Runtime: `loop={loop}` `range={search_range}` `rag={rag}` `verbose={verbose}`",
        f"- Total cases: `{summary.total_selected}`",
        f"- Attempted: `{summary.attempted}`",
        f"- Successes: `{summary.succeeded}`",
        f"- Failures: `{summary.failed}`",
        f"- Skipped: `{summary.skipped}`",
        f"- Success rate: `{((summary.succeeded / summary.attempted) if summary.attempted else 0.0):.2%}`",
        f"- Time to finish: `{_format_elapsed(summary.elapsed_seconds)}`",
    ]
    leaderboard_markdown_path.write_text("\n".join(leaderboard_lines) + "\n", encoding="utf-8")

    timeline_report = _build_timeline_report(run_id=run_id, rows=normalized_rows, run_started_at=started_at or now)
    timeline_json_path.write_text(json.dumps(timeline_report, indent=2), encoding="utf-8")
    with timeline_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "success",
                "final_error_category",
                "attempts",
                "started_at",
                "finished_at",
                "wall_clock_seconds",
                "duration_human",
                "relative_start_seconds",
                "relative_end_seconds",
            ],
        )
        writer.writeheader()
        for row in timeline_report["rows"]:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "success": row["success"],
                    "final_error_category": row["final_error_category"],
                    "attempts": row["attempts"],
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                    "wall_clock_seconds": row["wall_clock_seconds"],
                    "duration_human": row["duration_human"],
                    "relative_start_seconds": row["relative_start_seconds"],
                    "relative_end_seconds": row["relative_end_seconds"],
                }
            )

    timeline_lines = [
        "# Case Timeline",
        "",
        f"Run ID: `{run_id}`",
        f"Run started: `{timeline_report['run_started_at'] or 'unknown'}`",
        f"Run finished: `{timeline_report['run_finished_at'] or 'unknown'}`",
        f"Timeline span: `{timeline_report['total_span_human']}`",
        "",
        "| Case ID | Start | End | Duration | Attempts | Status | Timeline |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in timeline_report["rows"]:
        row_status = "Success" if row["success"] else (row["final_error_category"] or "failed")
        timeline_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_id"]),
                    str(row["started_at"] or "unknown"),
                    str(row["finished_at"] or "unknown"),
                    str(row["duration_human"]),
                    str(row["attempts"]),
                    row_status,
                    f"`{row['timeline_bar']}`",
                ]
            )
            + " |"
        )
    timeline_markdown_path.write_text("\n".join(timeline_lines) + "\n", encoding="utf-8")

    lines = [
        "# PLLM Benchmark Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Status: `{status}`",
        f"- Source: `{source}`",
        f"- Model: `{model}`",
        f"- Base: `{base}`",
        f"- Runtime: `loop={loop}` `range={search_range}` `rag={rag}` `verbose={verbose}`",
        f"- Selected: `{summary.total_selected}`",
        f"- Completed: `{len(rows)}`",
        f"- Attempted: `{summary.attempted}`",
        f"- Succeeded: `{summary.succeeded}`",
        f"- Failed: `{summary.failed}`",
        f"- Skipped: `{summary.skipped}`",
        f"- Elapsed: `{_format_elapsed(summary.elapsed_seconds)}`",
        "",
        f"- Summary JSON: `{summary_path}`",
        f"- Summary CSV: `{summary_csv_path}`",
        f"- Results CSV: `{results_csv_path}`",
        f"- Run vs Official CSV: `{run_vs_official_csv_path}`",
        f"- Results JSON: `{results_json_path}`",
        f"- Results Markdown: `{results_markdown_path}`",
        f"- Leaderboard: `{leaderboard_markdown_path}`",
        f"- Timeline JSON: `{timeline_json_path}`",
        f"- Timeline CSV: `{timeline_csv_path}`",
        f"- Timeline Markdown: `{timeline_markdown_path}`",
        f"- LLM Trace: `{llm_trace_path}`",
        f"- Warnings: `{warnings_path or '(none)'}`",
    ]
    report_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary.report_dir = str(run_dir)
    summary.summary_path = str(summary_path)
    summary.summary_csv_path = str(summary_csv_path)
    summary.results_csv_path = str(results_csv_path)
    summary.run_vs_official_csv_path = str(run_vs_official_csv_path)
    summary.results_json_path = str(results_json_path)
    summary.results_markdown_path = str(results_markdown_path)
    summary.leaderboard_markdown_path = str(leaderboard_markdown_path)
    summary.timeline_json_path = str(timeline_json_path)
    summary.timeline_csv_path = str(timeline_csv_path)
    summary.timeline_markdown_path = str(timeline_markdown_path)
    summary.llm_trace_path = llm_trace_path
    summary.warnings_path = warnings_path or ""
    summary.report_markdown_path = str(report_markdown_path)


def _write_case_result(*, run_dir: Path, run_id: str, source: BenchmarkSource, row: dict[str, object]) -> None:
    case_id = str(row.get("case_id", "")).strip()
    if not case_id:
        return
    case_dir = run_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_result_row(row)
    payload: dict[str, Any] = dict(normalized)
    payload["run_id"] = run_id
    payload["source"] = source
    payload["case_source"] = source
    payload["dependencies"] = []
    payload["attempt_records"] = []
    payload["candidate_provenance"] = {}
    (case_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_state(*, run_dir: Path, payload: dict[str, object]) -> None:
    state_path = run_dir / "run-state.json"
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    active_cases = payload.get("current_cases", [])
    if not isinstance(active_cases, list):
        active_cases = []
    total = _as_int(payload.get("total"), default=_as_int(payload.get("total_selected"), default=0))
    completed = _as_int(payload.get("completed"), default=0)
    elapsed = _format_elapsed(_as_float(payload.get("elapsed_seconds"), default=0.0))
    lines = [
        "# Run Status",
        "",
        f"- Run ID: `{payload.get('run_id', run_dir.name)}`",
        f"- Status: `{payload.get('status', 'unknown')}`",
        f"- Resolver: `{payload.get('resolver', 'pllm')}`",
        f"- Preset: `{payload.get('preset', 'default')}`",
        f"- Benchmark source: `{payload.get('benchmark_source', payload.get('source', 'all-gists'))}`",
        f"- Prompt profile: `{payload.get('prompt_profile', 'default')}`",
        f"- Research bundle: `{payload.get('research_bundle', 'baseline')}`",
        f"- Research features: `{', '.join(payload.get('research_features', [])) if isinstance(payload.get('research_features', []), list) and payload.get('research_features', []) else 'none'}`",
        f"- Jobs: `{payload.get('jobs', 1)}`",
        f"- Target: `{payload.get('target', 'benchmark')}`",
        f"- Progress: `{completed}/{total}`",
        f"- Successes: `{payload.get('successes', payload.get('succeeded', 0))}`",
        f"- Failures: `{payload.get('failures', payload.get('failed', 0))}`",
        f"- Skipped: `{payload.get('skipped', 0)}`",
        f"- Elapsed: `{elapsed}`",
        f"- Started at: `{payload.get('started_at', '') or 'unknown'}`",
        f"- Last updated: `{payload.get('last_updated_at', '') or 'unknown'}`",
        f"- Last case: `{payload.get('last_case_id', '') or 'none'}`",
        f"- Last status: `{payload.get('last_status', '') or 'none'}`",
        f"- Stop requested: `{payload.get('stop_requested', False)}`",
        "",
    ]
    if active_cases:
        lines.append("## Active Cases")
        lines.append("")
        for case_id in active_cases:
            lines.append(f"- `{case_id}`")
        lines.append("")
    artifacts = [
        ("Summary JSON", payload.get("summary_path")),
        ("Summary CSV", payload.get("summary_csv_path")),
        ("Results CSV", payload.get("results_csv_path")),
        ("Run vs Official CSV", payload.get("run_vs_official_csv_path")),
        ("Results JSON", payload.get("results_json_path")),
        ("Results Markdown", payload.get("results_markdown_path")),
        ("Leaderboard", payload.get("leaderboard_markdown_path")),
        ("Timeline JSON", payload.get("timeline_json_path")),
        ("Timeline CSV", payload.get("timeline_csv_path")),
        ("Timeline Markdown", payload.get("timeline_markdown_path")),
        ("LLM Trace", payload.get("llm_trace_path")),
        ("Report", payload.get("report_markdown_path")),
        ("Warnings", payload.get("warnings_path")),
    ]
    if any(path for _, path in artifacts):
        lines.append("## Artifacts")
        lines.append("")
        for label, path in artifacts:
            if path:
                lines.append(f"- {label}: `{path}`")
        lines.append("")
    error_text = str(payload.get("last_error", "") or "").strip()
    if error_text:
        lines.extend(["## Last Error", "", error_text, ""])
    (run_dir / "run-state.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_runtime_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    source: BenchmarkSource,
    model: str,
    base: str,
    temp: float,
    loop: int,
    search_range: int,
    rag: bool,
    verbose: bool,
    total_selected: int,
    attempted: int,
    succeeded: int,
    failed: int,
    skipped: int,
    elapsed_seconds: float,
    rows: list[dict[str, object]],
    started_at: str,
    current_cases: list[str],
    last_case_id: str,
    last_status: str,
    status: str,
    stop_requested: bool = False,
    llm_trace_path: str = "",
    warnings_path: str | None = None,
) -> None:
    summary = BenchmarkSummary(
        source=source,
        total_selected=total_selected,
        attempted=attempted,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        elapsed_seconds=elapsed_seconds,
        run_id=run_id,
    )
    _write_run_reports(
        run_dir=run_dir,
        run_id=run_id,
        source=source,
        model=model,
        base=base,
        temp=temp,
        loop=loop,
        search_range=search_range,
        rag=rag,
        verbose=verbose,
        summary=summary,
        rows=rows,
        status=status,
        started_at=started_at,
        llm_trace_path=llm_trace_path,
        warnings_path=warnings_path,
    )
    _write_run_state(
        run_dir=run_dir,
        payload={
            **_build_run_state_payload(
                run_id=run_id,
                status=status,
                source=source,
                model=model,
                base=base,
                temp=temp,
                loop=loop,
                search_range=search_range,
                rag=rag,
                verbose=verbose,
                total_selected=total_selected,
                attempted=attempted,
                succeeded=succeeded,
                failed=failed,
                skipped=skipped,
                elapsed_seconds=elapsed_seconds,
                started_at=started_at,
                current_cases=current_cases,
                last_case_id=last_case_id,
                last_status=last_status,
                summary=summary,
                completed=len(rows),
                stop_requested=stop_requested,
                llm_trace_path=llm_trace_path,
                warnings_path=warnings_path,
            )
        },
    )


def _build_run_state_payload(
    *,
    run_id: str,
    status: str,
    source: BenchmarkSource,
    model: str,
    base: str,
    temp: float,
    loop: int,
    search_range: int,
    rag: bool,
    verbose: bool,
    total_selected: int,
    attempted: int,
    succeeded: int,
    failed: int,
    skipped: int,
    elapsed_seconds: float,
    started_at: str,
    current_cases: list[str],
    last_case_id: str,
    last_status: str,
    summary: BenchmarkSummary,
    completed: int,
    stop_requested: bool,
    llm_trace_path: str,
    warnings_path: str | None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": status,
        "resolver": "pllm",
        "preset": "default",
        "benchmark_source": source,
        "prompt_profile": "default",
        "research_bundle": "baseline",
        "research_features": [],
        "jobs": 1,
        "target": "benchmark",
        "source": source,
        "model": model,
        "model_summary": model,
        "base": base,
        "temp": temp,
        "loop": loop,
        "search_range": search_range,
        "rag": rag,
        "verbose": verbose,
        "total": total_selected,
        "total_selected": total_selected,
        "completed": completed,
        "attempted": attempted,
        "successes": succeeded,
        "failures": failed,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "success_rate": (succeeded / attempted) if attempted else 0.0,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "started_at": started_at,
        "last_updated_at": _utc_now_iso(),
        "current_cases": current_cases,
        "last_case_id": last_case_id,
        "last_status": last_status,
        "summary_path": summary.summary_path,
        "summary_csv_path": summary.summary_csv_path,
        "results_csv_path": summary.results_csv_path,
        "run_vs_official_csv_path": summary.run_vs_official_csv_path,
        "results_json_path": summary.results_json_path,
        "results_markdown_path": summary.results_markdown_path,
        "leaderboard_markdown_path": summary.leaderboard_markdown_path,
        "timeline_json_path": summary.timeline_json_path,
        "timeline_csv_path": summary.timeline_csv_path,
        "timeline_markdown_path": summary.timeline_markdown_path,
        "llm_trace_path": llm_trace_path,
        "report_markdown_path": summary.report_markdown_path,
        "report_dir": summary.report_dir,
        "artifacts_dir": summary.report_dir,
        "warnings_path": warnings_path or "",
        "stop_requested": stop_requested,
    }


def _initialize_trace_log(
    *,
    trace_path: Path,
    run_id: str,
    source: BenchmarkSource,
    model: str,
    base: str,
    loop: int,
    search_range: int,
    rag: bool,
    verbose: bool,
) -> None:
    lines = [
        f"[{_utc_now_iso()}] RUN START {run_id}",
        f"source={source}",
        f"model={model}",
        f"base={base}",
        f"loop={loop} range={search_range} rag={rag} verbose={verbose}",
        "",
    ]
    trace_path.write_text("\n".join(lines), encoding="utf-8")


def _append_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


def _write_case_source(*, run_dir: Path, case_id: str, snippet_path: Path) -> Path | None:
    case_dir = run_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_path = case_dir / "source.py"
    try:
        source_text = snippet_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    source_path.write_text(source_text, encoding="utf-8")
    return source_path


def _materialize_case_attempt_artifacts(
    *,
    run_dir: Path,
    case_id: str,
    snippet_path: Path,
    case_output_lines: list[str],
) -> dict[str, object]:
    case_dir = run_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    snippet_dir = snippet_path.parent

    run_log_path = case_dir / "run.log"
    run_log_payload = "\n".join(case_output_lines).strip()
    if run_log_payload:
        run_log_path.write_text(run_log_payload + "\n", encoding="utf-8")
    elif (snippet_dir / "run.log").exists():
        shutil.copy2(snippet_dir / "run.log", run_log_path)

    build_log_path = case_dir / "build.log"
    build_lines = [line for line in case_output_lines if _looks_like_build_log_line(line)]
    if build_lines:
        build_log_path.write_text("\n".join(build_lines) + "\n", encoding="utf-8")
    elif (snippet_dir / "build.log").exists():
        shutil.copy2(snippet_dir / "build.log", build_log_path)

    dockerfiles = sorted(snippet_dir.glob("Dockerfile-llm-*"))
    yaml_outputs = sorted(snippet_dir.glob("output_data_*.yml"))
    existing_attempt_dirs = sorted(path for path in case_dir.glob("attempt_*") if path.is_dir())
    attempt_count = max(1, len(existing_attempt_dirs), len(dockerfiles), len(yaml_outputs))
    attempt_dirs = [case_dir / f"attempt_{index:02d}" for index in range(1, attempt_count + 1)]
    for attempt_dir in attempt_dirs:
        attempt_dir.mkdir(parents=True, exist_ok=True)

    for attempt_dir in attempt_dirs:
        if run_log_path.exists():
            shutil.copy2(run_log_path, attempt_dir / "run.log")
        if build_log_path.exists():
            shutil.copy2(build_log_path, attempt_dir / "build.log")

    dockerfile_path: Path | None = None
    if dockerfiles:
        for index, dockerfile in enumerate(dockerfiles):
            target_dir = attempt_dirs[min(index, len(attempt_dirs) - 1)]
            shutil.copy2(dockerfile, target_dir / "Dockerfile.generated")
        dockerfile_path = case_dir / "Dockerfile.generated"
        shutil.copy2(dockerfiles[0], dockerfile_path)

    for index, yaml_file in enumerate(yaml_outputs):
        target_dir = attempt_dirs[min(index, len(attempt_dirs) - 1)]
        shutil.copy2(yaml_file, target_dir / yaml_file.name)

    prompt_a_path: Path | None = None
    prompt_b_path: Path | None = None
    for attempt_dir in attempt_dirs:
        prompt_a = attempt_dir / "prompt_a.txt"
        prompt_b = attempt_dir / "prompt_b.txt"
        if prompt_a.exists() and prompt_a_path is None:
            prompt_a_path = case_dir / "prompt_a.txt"
            shutil.copy2(prompt_a, prompt_a_path)
        if prompt_b.exists() and prompt_b_path is None:
            prompt_b_path = case_dir / "prompt_b.txt"
            shutil.copy2(prompt_b, prompt_b_path)
    if prompt_a_path is None:
        prompt_a_path = case_dir / "prompt_a.txt"
        prompt_a_path.write_text(_build_extract_prompt(snippet_path), encoding="utf-8")
    if prompt_b_path is None:
        prompt_b_path = case_dir / "prompt_b.txt"
        prompt_b_path.write_text(_build_version_prompt(snippet_dir), encoding="utf-8")
    first_attempt_dir = attempt_dirs[0]
    if not (first_attempt_dir / "prompt_a.txt").exists():
        shutil.copy2(prompt_a_path, first_attempt_dir / "prompt_a.txt")
    if not (first_attempt_dir / "prompt_b.txt").exists():
        shutil.copy2(prompt_b_path, first_attempt_dir / "prompt_b.txt")
    first_attempt_prompts_dir = first_attempt_dir / "prompts"
    first_attempt_prompts_dir.mkdir(parents=True, exist_ok=True)
    if not any(first_attempt_prompts_dir.glob("prompt_*.txt")):
        shutil.copy2(prompt_a_path, first_attempt_prompts_dir / "prompt_001_extract.txt")
        shutil.copy2(prompt_b_path, first_attempt_prompts_dir / "prompt_002_version.txt")

    combined_model_outputs = _combine_attempt_model_outputs(attempt_dirs)
    if not any(combined_model_outputs[stage] for stage in ("extract", "version", "repair", "adjudicate")):
        inferred_outputs = _infer_model_outputs_from_logs(case_output_lines)
        for stage in combined_model_outputs:
            combined_model_outputs[stage].extend(inferred_outputs[stage])
    model_outputs_path: Path | None = case_dir / "model_outputs.json"
    assert model_outputs_path is not None
    model_outputs_path.write_text(json.dumps(combined_model_outputs, indent=2), encoding="utf-8")
    if not (first_attempt_dir / "model_outputs.json").exists():
        shutil.copy2(model_outputs_path, first_attempt_dir / "model_outputs.json")

    return {
        "run_log_path": str(run_log_path) if run_log_path.exists() else "",
        "build_log_path": str(build_log_path) if build_log_path.exists() else "",
        "dockerfile_path": str(dockerfile_path) if dockerfile_path is not None else "",
        "model_outputs_path": str(model_outputs_path) if model_outputs_path is not None else "",
        "prompt_a_path": str(prompt_a_path) if prompt_a_path is not None else "",
        "prompt_b_path": str(prompt_b_path) if prompt_b_path is not None else "",
        "attempt_count": attempt_count,
    }


def _combine_attempt_model_outputs(attempt_dirs: list[Path]) -> dict[str, list[dict[str, object]]]:
    combined: dict[str, list[dict[str, object]]] = {
        "extract": [],
        "version": [],
        "repair": [],
        "adjudicate": [],
    }
    for attempt_dir in attempt_dirs:
        payload_path = attempt_dir / "model_outputs.json"
        if not payload_path.exists():
            continue
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for stage in combined:
            entries = payload.get(stage, [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                merged = dict(entry)
                merged.setdefault("attempt_folder", attempt_dir.name)
                combined[stage].append(merged)
    return combined


def _write_run_vs_official_csv(*, path: Path, run_rows: list[dict[str, object]]) -> None:
    official_lookup = _load_official_pllm_lookup()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gistid", "result_a", "result_b", "both_matched"])
        writer.writeheader()
        for row in sorted(run_rows, key=lambda item: str(item.get("case_id", ""))):
            case_id = str(row.get("case_id", "") or "")
            result_a = _run_result_label_for_official(row)
            result_b = official_lookup.get(case_id, "")
            both_matched: str | bool = ""
            if result_b:
                both_matched = _normalize_result_label(result_a) == _normalize_result_label(result_b)
            writer.writerow(
                {
                    "gistid": case_id,
                    "result_a": result_a,
                    "result_b": result_b,
                    "both_matched": both_matched,
                }
            )


def _load_official_pllm_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    path = _OFFICIAL_PLLM_RESULTS_CSV
    if not path.exists():
        return lookup
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_ids = _extract_case_ids_from_official_row(row)
                if not case_ids:
                    continue
                official_result = _official_lookup_value(row, "result")
                if not official_result:
                    official_passed = _official_lookup_value(row, "passed")
                    if _parse_official_passed_flag(official_passed) is True:
                        official_result = "OtherPass"
                    elif _parse_official_passed_flag(official_passed) is False:
                        official_result = "OtherFailure"
                if not official_result:
                    continue
                for case_id in case_ids:
                    lookup.setdefault(case_id, official_result)
    except OSError:
        return {}
    return lookup


def _extract_case_ids_from_official_row(row: dict[str, str]) -> set[str]:
    normalized = {str(key).strip().lower(): str(value or "") for key, value in row.items() if key}
    values = [normalized[column] for column in _OFFICIAL_CASE_ID_COLUMNS if column in normalized]
    case_ids: set[str] = set()
    for value in values:
        for token in re.split(r"[^0-9A-Za-z]+", value):
            stripped = token.strip()
            if stripped and _OFFICIAL_CASE_TOKEN_PATTERN.fullmatch(stripped):
                case_ids.add(stripped)
    return case_ids


def _official_lookup_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _normalize_result_label(value: object) -> str:
    return str(value or "").strip().lower()


def _parse_official_passed_flag(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"true", "t", "yes", "y", "pass", "passed"}:
        return True
    if text in {"false", "f", "no", "n", "fail", "failed"}:
        return False
    try:
        return float(text) > 0.0
    except ValueError:
        return None


def _run_result_label_for_official(row: dict[str, object]) -> str:
    if bool(row.get("success", False)):
        return "OtherPass"
    category = _normalize_result_label(row.get("final_error_category", ""))
    status = _normalize_result_label(row.get("status", ""))
    if status == "skipped" or category == "skipped":
        return "Skipped"
    if category in {"modulenotfounderror", "modulenotfound"}:
        return "ModuleNotFound"
    if category in {"importerror", "syntaxerror", "nameerror", "typeerror", "attributeerror"}:
        return {
            "importerror": "ImportError",
            "syntaxerror": "SyntaxError",
            "nameerror": "NameError",
            "typeerror": "TypeError",
            "attributeerror": "AttributeError",
        }[category]
    if category in {"versionnotfound", "dependencyconflict", "nonzerocode", "invalidversion"}:
        return {
            "versionnotfound": "VersionNotFound",
            "dependencyconflict": "DependencyConflict",
            "nonzerocode": "NonZeroCode",
            "invalidversion": "InvalidVersion",
        }[category]
    if category:
        return category
    return "OtherFailure"


def _build_extract_prompt(snippet_path: Path) -> str:
    snippet_text = ""
    try:
        snippet_text = snippet_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        snippet_text = ""
    return (
        "List the required third-party libraries that need to be installed for the following Python code to execute "
        "successfully. Provide just the library names without any additional explanation.\n\n"
        f"Python code:\n{snippet_text}"
    )


def _build_version_prompt(snippet_dir: Path) -> str:
    modules_dir = snippet_dir / "modules"
    if not modules_dir.exists():
        return "# skipped: no compatible PyPI version options were inferred\n"
    package_lines: list[str] = []
    for path in sorted(modules_dir.glob("*_*.txt")):
        stem = path.stem
        if "_" not in stem:
            continue
        module_name = stem.rsplit("_", 1)[0]
        try:
            raw_versions = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        versions = [token.strip() for token in raw_versions.replace("\n", ",").split(",") if token.strip()]
        if not versions:
            continue
        package_lines.append(f"{module_name}: {', '.join(versions[:25])}")
    if not package_lines:
        return "# skipped: no compatible PyPI version options were inferred\n"
    return (
        "You are given a list of Python packages and their available versions from PyPI. Determine the most "
        "appropriate version for each package to ensure compatibility and successful execution. Only consider the "
        "versions explicitly listed.\n\n"
        "Return the result as a plain list using the format package_name==version, with no additional explanation or "
        "formatting.\n\n"
        "Packages and versions:\n"
        + "\n".join(package_lines)
    )


def _infer_model_outputs_from_logs(case_output_lines: list[str]) -> dict[str, list[dict[str, object]]]:
    payload: dict[str, list[dict[str, object]]] = {
        "extract": [],
        "version": [],
        "repair": [],
        "adjudicate": [],
    }
    for line in case_output_lines:
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        output_text = json.dumps(parsed, indent=2)
        if "python_version" in parsed and "python_modules" in parsed:
            payload["extract"].append({"attempt": len(payload["extract"]) + 1, "output": output_text, "source": "stdout"})
            continue
        if "module" in parsed and "version" in parsed:
            payload["version"].append({"attempt": len(payload["version"]) + 1, "output": output_text, "source": "stdout"})
            continue
    return payload


def _looks_like_build_log_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return stripped.startswith("#") or "docker build" in stripped.lower() or "build details:" in stripped.lower()


def _normalize_result_row(row: dict[str, object]) -> dict[str, object]:
    status = str(row.get("status", "unknown") or "unknown")
    success = bool(row.get("success", False))
    attempts = 0 if status == "skipped" else 1
    final_error_category = ""
    if not success:
        final_error_category = "Skipped" if status == "skipped" else "ExecutionFailed"
    elapsed = _as_float(row.get("elapsed_seconds"), default=0.0)
    return {
        "index": _as_int(row.get("index"), default=0),
        "case_id": str(row.get("case_id", "") or ""),
        "status": status,
        "success": success,
        "return_code": _as_int(row.get("return_code"), default=0),
        "elapsed_seconds": round(elapsed, 6),
        "wall_clock_seconds": round(elapsed, 6),
        "started_at": str(row.get("started_at", "") or ""),
        "finished_at": str(row.get("finished_at", "") or ""),
        "attempts": attempts,
        "initial_eval": "",
        "final_error_category": final_error_category,
        "source_path": str(row.get("source_path", "") or ""),
        "trace_path": str(row.get("trace_path", "") or ""),
        "run_log_path": str(row.get("run_log_path", "") or ""),
        "build_log_path": str(row.get("build_log_path", "") or ""),
        "dockerfile_path": str(row.get("dockerfile_path", "") or ""),
        "model_outputs_path": str(row.get("model_outputs_path", "") or ""),
        "prompt_a_path": str(row.get("prompt_a_path", "") or ""),
        "prompt_b_path": str(row.get("prompt_b_path", "") or ""),
        "attempt_count": _as_int(row.get("attempt_count"), default=1),
    }


def _to_results_row(row: dict[str, object], *, case_number: int) -> dict[str, object]:
    return {
        "case_number": case_number,
        "case_id": row["case_id"],
        "modules": "",
        "result": "success" if row["success"] else "failure",
        "attempts": row["attempts"],
        "initial_eval": row["initial_eval"],
        "final_error_category": row["final_error_category"],
        "dependencies": "",
        "dependency_reason": "",
        "research_bundle": "baseline",
        "research_features": "",
        "selected_candidate_rank": "",
        "strategy_type": "",
        "retry_severity": "",
        "conflict_precheck_failed": False,
        "status": row["status"],
        "return_code": row["return_code"],
        "wall_clock_seconds": row["wall_clock_seconds"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "source_path": row.get("source_path", ""),
        "trace_path": row.get("trace_path", ""),
        "run_log_path": row.get("run_log_path", ""),
        "build_log_path": row.get("build_log_path", ""),
        "dockerfile_path": row.get("dockerfile_path", ""),
        "model_outputs_path": row.get("model_outputs_path", ""),
        "prompt_a_path": row.get("prompt_a_path", ""),
        "prompt_b_path": row.get("prompt_b_path", ""),
        "attempt_count": row.get("attempt_count", 1),
    }


def _build_timeline_report(*, run_id: str, rows: list[dict[str, object]], run_started_at: str) -> dict[str, object]:
    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        started_dt = _parse_iso8601(row.get("started_at"))
        finished_dt = _parse_iso8601(row.get("finished_at"))
        parsed_rows.append(
            {
                **row,
                "_started_dt": started_dt,
                "_finished_dt": finished_dt,
            }
        )
    parsed_rows.sort(key=lambda row: (row.get("index", 0), str(row.get("case_id", ""))))

    parsed_starts = [row["_started_dt"] for row in parsed_rows if row["_started_dt"] is not None]
    parsed_finishes = [row["_finished_dt"] for row in parsed_rows if row["_finished_dt"] is not None]
    run_start_dt = _parse_iso8601(run_started_at) or (parsed_starts[0] if parsed_starts else None)
    run_finish_dt = parsed_finishes[-1] if parsed_finishes else None

    base_dt = run_start_dt or (parsed_starts[0] if parsed_starts else None)
    if base_dt is None:
        total_span_seconds = 0.0
    else:
        latest_dt = run_finish_dt or base_dt
        total_span_seconds = max(0.0, (latest_dt - base_dt).total_seconds())

    timeline_rows: list[dict[str, object]] = []
    for row in parsed_rows:
        started_dt = row["_started_dt"]
        finished_dt = row["_finished_dt"]
        duration = _as_float(row.get("wall_clock_seconds"), default=0.0)
        if base_dt is None or started_dt is None:
            relative_start = 0.0
        else:
            relative_start = max(0.0, (started_dt - base_dt).total_seconds())
        if base_dt is None or finished_dt is None:
            relative_end = relative_start + duration
        else:
            relative_end = max(relative_start, (finished_dt - base_dt).total_seconds())
        timeline_rows.append(
            {
                "case_id": row.get("case_id", ""),
                "success": row.get("success", False),
                "final_error_category": row.get("final_error_category", ""),
                "attempts": row.get("attempts", 0),
                "started_at": row.get("started_at", ""),
                "finished_at": row.get("finished_at", ""),
                "wall_clock_seconds": row.get("wall_clock_seconds", 0.0),
                "duration_human": _format_elapsed(duration),
                "relative_start_seconds": round(relative_start, 6),
                "relative_end_seconds": round(relative_end, 6),
                "timeline_bar": _render_timeline_bar(
                    relative_start_seconds=relative_start,
                    duration_seconds=max(duration, relative_end - relative_start),
                    total_span_seconds=total_span_seconds,
                ),
            }
        )

    return {
        "run_id": run_id,
        "run_started_at": run_started_at,
        "run_finished_at": run_finish_dt.isoformat() if run_finish_dt is not None else "",
        "total_span_seconds": round(total_span_seconds, 6),
        "total_span_human": _format_elapsed(total_span_seconds),
        "rows": timeline_rows,
    }


def _render_timeline_bar(*, relative_start_seconds: float, duration_seconds: float, total_span_seconds: float) -> str:
    width = 36
    if total_span_seconds <= 0.0:
        return "#" + "." * (width - 1)
    start_index = min(width - 1, max(0, int((relative_start_seconds / total_span_seconds) * (width - 1))))
    segment = max(1, int((duration_seconds / total_span_seconds) * width))
    end_index = min(width, start_index + segment)
    bar = ["." for _ in range(width)]
    for index in range(start_index, end_index):
        bar[index] = "#"
    return "".join(bar)


def _parse_iso8601(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _as_int(value: object, *, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: object, *, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _format_elapsed(seconds: float) -> str:
    remaining = max(0, int(round(seconds)))
    hours, remaining = divmod(remaining, 3600)
    minutes, secs = divmod(remaining, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"
