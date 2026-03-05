from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cli.pllm.benchmark_data import BenchmarkSource, list_case_ids, resolve_snippet_path
from cli.pllm.core import RunConfig, stream_executor

LineHandler = Callable[[str], None]


@dataclass
class BenchmarkSummary:
    source: BenchmarkSource
    total_selected: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    elapsed_seconds: float


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
) -> tuple[int, BenchmarkSummary]:
    case_ids = list_case_ids(source)
    if offset > 0:
        case_ids = case_ids[offset:]
    if limit > 0:
        case_ids = case_ids[:limit]

    total_selected = len(case_ids)
    if total_selected == 0:
        summary = BenchmarkSummary(
            source=source,
            total_selected=0,
            attempted=0,
            succeeded=0,
            failed=0,
            skipped=0,
            elapsed_seconds=0.0,
        )
        return 1, summary

    attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0
    elapsed_seconds = 0.0

    for index, case_id in enumerate(case_ids, start=1):
        snippet_path = resolve_snippet_path(case_id, source=source)
        if snippet_path is None:
            skipped += 1
            _emit(line_handler, f"[{index}/{total_selected}] {case_id} skipped (missing snippet)")
            continue

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

        return_code, stats = stream_executor(config, line_callback=callback)
        attempted += 1
        elapsed_seconds += stats.elapsed_seconds

        if return_code == 0:
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
    )
    return (0 if failed == 0 else 1), summary


def _emit(handler: LineHandler | None, line: str) -> None:
    if handler is None:
        print(line)
    else:
        handler(line)

