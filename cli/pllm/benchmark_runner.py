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
    started_at = _utc_now_iso()
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
        started_at=started_at,
        current_cases=current_cases,
        last_case_id=last_case_id,
        last_status=last_status,
        status="running" if total_selected > 0 else "empty",
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
            started_at=started_at,
            current_cases=[],
            last_case_id="",
            last_status="no_cases_selected",
            status="empty",
        )
        summary.report_dir = str(run_dir)
        summary.summary_path = str(run_dir / "summary.json")
        summary.results_csv_path = str(run_dir / "results.csv")
        summary.results_json_path = str(run_dir / "results.json")
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
            started_at=started_at,
            current_cases=current_cases,
            last_case_id=last_case_id,
            last_status=last_status,
            status="running",
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
            }
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
                started_at=started_at,
                current_cases=current_cases,
                last_case_id=last_case_id,
                last_status=last_status,
                status="running",
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
        row = {
            "index": index,
            "case_id": case_id,
            "status": status,
            "success": success,
            "return_code": return_code,
            "elapsed_seconds": round(stats.elapsed_seconds, 6),
            "started_at": started_at,
            "finished_at": finished_at,
        }
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
            started_at=started_at,
            current_cases=current_cases,
            last_case_id=last_case_id,
            last_status=last_status,
            status="running",
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
        started_at=started_at,
    )
    _write_run_state(
        run_dir=run_dir,
        payload={
            "run_id": run_id,
            "status": final_status,
            "source": source,
            "model": model,
            "base": base,
            "temp": temp,
            "loop": loop,
            "search_range": search_range,
            "rag": rag,
            "verbose": verbose,
            "total_selected": total_selected,
            "completed": len(rows),
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (succeeded / attempted) if attempted else 0.0,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "started_at": started_at,
            "last_updated_at": _utc_now_iso(),
            "current_cases": [],
            "last_case_id": last_case_id,
            "last_status": last_status,
            "summary_path": summary.summary_path,
            "results_csv_path": summary.results_csv_path,
            "results_json_path": summary.results_json_path,
            "report_markdown_path": summary.report_markdown_path,
            "report_dir": summary.report_dir,
        },
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
) -> None:
    summary_path = run_dir / "summary.json"
    results_json_path = run_dir / "results.json"
    results_csv_path = run_dir / "results.csv"
    report_markdown_path = run_dir / "report.md"
    now = _utc_now_iso()

    summary_payload = {
        "run_id": run_id,
        "status": status,
        "source": source,
        "total_selected": summary.total_selected,
        "completed": len(rows),
        "attempted": summary.attempted,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
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
        "generated_at": now,
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


def _write_case_result(*, run_dir: Path, run_id: str, source: BenchmarkSource, row: dict[str, object]) -> None:
    case_id = str(row.get("case_id", "")).strip()
    if not case_id:
        return
    case_dir = run_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload["run_id"] = run_id
    payload["source"] = source
    (case_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_state(*, run_dir: Path, payload: dict[str, object]) -> None:
    state_path = run_dir / "run-state.json"
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# PLLM Run State",
        "",
        f"- Run ID: `{payload.get('run_id', '')}`",
        f"- Status: `{payload.get('status', '')}`",
        f"- Source: `{payload.get('source', '')}`",
        f"- Model: `{payload.get('model', '')}`",
        f"- Progress: `{payload.get('completed', 0)}/{payload.get('total_selected', 0)}`",
        f"- Attempted/Succeeded/Failed/Skipped: "
        f"`{payload.get('attempted', 0)}/{payload.get('succeeded', 0)}/{payload.get('failed', 0)}/{payload.get('skipped', 0)}`",
        f"- Last case: `{payload.get('last_case_id', '')}` (`{payload.get('last_status', '')}`)",
        f"- Updated: `{payload.get('last_updated_at', '')}`",
    ]
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
    )
    _write_run_state(
        run_dir=run_dir,
        payload={
            "run_id": run_id,
            "status": status,
            "source": source,
            "model": model,
            "base": base,
            "temp": temp,
            "loop": loop,
            "search_range": search_range,
            "rag": rag,
            "verbose": verbose,
            "total_selected": total_selected,
            "completed": len(rows),
            "attempted": attempted,
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
            "results_csv_path": summary.results_csv_path,
            "results_json_path": summary.results_json_path,
            "report_markdown_path": summary.report_markdown_path,
            "report_dir": summary.report_dir,
        },
    )
