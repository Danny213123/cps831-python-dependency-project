from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Callable, Protocol

from cli.pllm.benchmark_data import BenchmarkSource, list_case_ids, resolve_snippet_path
from cli.pllm.core import ROOT_DIR, RunConfig, stream_executor

LineHandler = Callable[[str], None]


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
    results_csv_path: str = ""
    results_json_path: str = ""
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

    total_selected = len(case_ids)
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
        )
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
            break

        snippet_path = resolve_snippet_path(case_id, source=source)
        if snippet_path is None:
            skipped += 1
            now = _utc_now_iso()
            rows.append(
                {
                    "index": index,
                    "case_id": case_id,
                    "status": "skipped",
                    "success": False,
                    "return_code": 2,
                    "elapsed_seconds": 0.0,
                    "started_at": now,
                    "finished_at": now,
                }
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

        callback = (lambda _line, _stats: None)
        if show_case_output:
            callback = lambda line, _stats: _emit(line_handler, line)

        started_at = _utc_now_iso()
        return_code, stats = stream_executor(config, line_callback=callback)
        finished_at = _utc_now_iso()
        attempted += 1
        elapsed_seconds += stats.elapsed_seconds
        success = return_code == 0
        status = "success" if success else "failed"
        rows.append(
            {
                "index": index,
                "case_id": case_id,
                "status": status,
                "success": success,
                "return_code": return_code,
                "elapsed_seconds": round(stats.elapsed_seconds, 6),
                "started_at": started_at,
                "finished_at": finished_at,
            }
        )

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
            _emit(
                line_handler,
                f"[{index}/{total_selected}] {case_id} failed rc={return_code} ({stats.elapsed_seconds:.1f}s)",
            )
            if fail_fast:
                break

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
    )
    if observer is not None:
        status = "stopped" if observer.stop_requested() else "completed"
        observer.finish(summary=summary, status=status)
    return (0 if failed == 0 else 1), summary


def _emit(handler: LineHandler | None, line: str) -> None:
    if handler is None:
        print(line)
    else:
        handler(line)


def _prepare_run_dir(run_id: str) -> Path:
    run_dir = ROOT_DIR / "artifacts" / "pllm-benchmark" / run_id
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
) -> None:
    summary_path = run_dir / "summary.json"
    results_json_path = run_dir / "results.json"
    results_csv_path = run_dir / "results.csv"
    report_markdown_path = run_dir / "report.md"

    summary_payload = {
        "run_id": run_id,
        "source": source,
        "total_selected": summary.total_selected,
        "attempted": summary.attempted,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "skipped": summary.skipped,
        "success_rate": (summary.succeeded / summary.attempted) if summary.attempted else 0.0,
        "elapsed_seconds": round(summary.elapsed_seconds, 6),
        "model": model,
        "base": base,
        "temp": temp,
        "loop": loop,
        "search_range": search_range,
        "rag": rag,
        "verbose": verbose,
        "generated_at": _utc_now_iso(),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    results_json_path.write_text(json.dumps({"run_id": run_id, "rows": rows}, indent=2), encoding="utf-8")

    with results_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "case_id",
                "status",
                "success",
                "return_code",
                "elapsed_seconds",
                "started_at",
                "finished_at",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# PLLM Benchmark Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Source: `{source}`",
        f"- Model: `{model}`",
        f"- Base: `{base}`",
        f"- Runtime: `loop={loop}` `range={search_range}` `rag={rag}` `verbose={verbose}`",
        f"- Selected: `{summary.total_selected}`",
        f"- Attempted: `{summary.attempted}`",
        f"- Succeeded: `{summary.succeeded}`",
        f"- Failed: `{summary.failed}`",
        f"- Skipped: `{summary.skipped}`",
        f"- Elapsed: `{summary.elapsed_seconds:.1f}s`",
        "",
        f"- Summary JSON: `{summary_path}`",
        f"- Results CSV: `{results_csv_path}`",
        f"- Results JSON: `{results_json_path}`",
    ]
    report_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary.report_dir = str(run_dir)
    summary.summary_path = str(summary_path)
    summary.results_csv_path = str(results_csv_path)
    summary.results_json_path = str(results_json_path)
    summary.report_markdown_path = str(report_markdown_path)
