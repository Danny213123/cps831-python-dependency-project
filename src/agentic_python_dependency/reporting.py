from __future__ import annotations

import csv
import json
import math
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_python_dependency.benchmark.gistable import GistableDataset
from agentic_python_dependency.presets import GroupingMode
from agentic_python_dependency.state import BenchmarkSummary
from agentic_python_dependency.tools.import_extractor import (
    extract_import_roots_from_code,
    filter_third_party_imports,
    normalize_candidate_packages,
)


CANONICAL_MODULE_FAMILIES = {
    "pil": "pillow",
    "pillow": "pillow",
    "bs4": "beautifulsoup4",
    "beautifulsoup": "beautifulsoup4",
    "beautifulsoup4": "beautifulsoup4",
    "tensorflow": "tensorflow",
    "tensorflow-gpu": "tensorflow",
    "tensorflow-cpu": "tensorflow",
    "cv2": "opencv-python",
    "opencv-python": "opencv-python",
    "yaml": "pyyaml",
    "pyyaml": "pyyaml",
    "sklearn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "dateutil": "python-dateutil",
    "python-dateutil": "python-dateutil",
    "openssl": "pyopenssl",
    "pyopenssl": "pyopenssl",
    "serial": "pyserial",
    "pyserial": "pyserial",
    "git": "gitpython",
    "gitpython": "gitpython",
}


def canonical_module_name(module_name: str, grouping: GroupingMode) -> str:
    normalized = module_name.lower()
    if grouping == "raw":
        return normalized
    return CANONICAL_MODULE_FAMILIES.get(normalized, normalized)


def paper_hard_subset_case_ids(dataset: GistableDataset, ref: str | None = None) -> list[str]:
    hard_ids = {
        row["id"]
        for row in dataset.load_results_rows(ref)
        if row.get("initial-eval") == "ImportError"
        and (dataset.dataset_root(ref) / "all-gists" / row["id"] / "snippet.py").exists()
    }
    return sorted(hard_ids)


def safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def load_run_results(run_dir: Path) -> list[dict[str, Any]]:
    result_paths = sorted(run_dir.glob("*/result.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(math.floor(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _render_timeline_bar(
    start_seconds: float | None,
    duration_seconds: float,
    total_span_seconds: float,
    width: int = 32,
) -> str:
    if total_span_seconds <= 0 or start_seconds is None:
        return "█" * min(width, max(1, int(round(duration_seconds)))) if duration_seconds > 0 else "·"
    offset = int((start_seconds / total_span_seconds) * max(0, width - 1))
    span = max(1, int((duration_seconds / total_span_seconds) * width))
    filled = min(width - offset, span)
    return ("·" * offset) + ("█" * filled) + ("·" * max(0, width - offset - filled))


def build_timeline_view(run_dir: Path) -> dict[str, Any]:
    results = load_run_results(run_dir)
    timed_rows: list[dict[str, Any]] = []
    all_started = [parse_timestamp(item.get("started_at")) for item in results]
    all_finished = [parse_timestamp(item.get("finished_at")) for item in results]
    valid_started = [value for value in all_started if value is not None]
    valid_finished = [value for value in all_finished if value is not None]
    run_started_at = min(valid_started) if valid_started else None
    run_finished_at = max(valid_finished) if valid_finished else None
    total_span_seconds = (
        max(0.0, (run_finished_at - run_started_at).total_seconds())
        if run_started_at is not None and run_finished_at is not None
        else 0.0
    )

    for item in results:
        started_at = parse_timestamp(item.get("started_at"))
        finished_at = parse_timestamp(item.get("finished_at"))
        relative_start = (
            max(0.0, (started_at - run_started_at).total_seconds())
            if started_at is not None and run_started_at is not None
            else None
        )
        relative_end = (
            max(0.0, (finished_at - run_started_at).total_seconds())
            if finished_at is not None and run_started_at is not None
            else None
        )
        duration_seconds = float(item.get("wall_clock_seconds", 0.0))
        timed_rows.append(
            {
                "case_id": item.get("case_id", ""),
                "success": bool(item.get("success", False)),
                "final_error_category": item.get("final_error_category", ""),
                "attempts": int(item.get("attempts", 0)),
                "started_at": item.get("started_at", ""),
                "finished_at": item.get("finished_at", ""),
                "wall_clock_seconds": duration_seconds,
                "duration_human": format_duration(duration_seconds),
                "relative_start_seconds": relative_start,
                "relative_end_seconds": relative_end,
                "timeline_bar": _render_timeline_bar(relative_start, duration_seconds, total_span_seconds),
            }
        )

    timed_rows.sort(
        key=lambda item: (
            item["relative_start_seconds"] is None,
            float(item["relative_start_seconds"] or 0.0),
            item["case_id"],
        )
    )

    report = {
        "run_id": run_dir.name,
        "run_started_at": run_started_at.isoformat() if run_started_at is not None else "",
        "run_finished_at": run_finished_at.isoformat() if run_finished_at is not None else "",
        "total_span_seconds": total_span_seconds,
        "total_span_human": format_duration(total_span_seconds),
        "rows": timed_rows,
    }
    write_timeline_artifacts(run_dir, report)
    return report


def summarize_run(run_dir: Path, total_elapsed_seconds: float | None = None) -> BenchmarkSummary:
    results = load_run_results(run_dir)
    total_cases = len(results)
    successes = sum(1 for item in results if item["success"])
    failures = total_cases - successes
    initial_import_errors = sum(1 for item in results if item.get("initial_eval") == "ImportError")
    final_import_errors = sum(1 for item in results if item.get("final_error_category") == "ImportError")
    successful_attempts = [item["attempts"] for item in results if item["success"]]
    mean_attempts = sum(successful_attempts) / len(successful_attempts) if successful_attempts else 0.0
    summed_wall_clock = sum(item.get("wall_clock_seconds", 0.0) for item in results)
    mean_wall_clock = summed_wall_clock / total_cases if results else 0.0
    total_wall_clock = total_elapsed_seconds if total_elapsed_seconds is not None else summed_wall_clock
    transitions: dict[str, int] = {}
    dependency_reason_counts: dict[str, int] = {}
    resolver = results[0].get("resolver", "apd") if results else "apd"
    preset = results[0].get("preset", "optimized") if results else "optimized"
    prompt_profile = results[0].get("prompt_profile", "optimized") if results else "optimized"
    model_profile = results[0].get("model_profile", "gemma-moe") if results else "gemma-moe"
    use_moe = bool(results[0].get("use_moe", True)) if results else True
    use_rag = bool(results[0].get("use_rag", True)) if results else True
    use_langchain = bool(results[0].get("use_langchain", True)) if results else True
    rag_mode = str(results[0].get("rag_mode", "pypi")) if results else "pypi"
    structured_prompting = bool(results[0].get("structured_prompting", False)) if results else False
    extraction_model = results[0].get("extraction_model", "gemma3:4b") if results else "gemma3:4b"
    runner_model = results[0].get("runner_model", "gemma3:12b") if results else "gemma3:12b"
    version_model = results[0].get("version_model", runner_model) if results else "gemma3:12b"
    repair_model = results[0].get("repair_model", runner_model) if results else "gemma3:12b"
    adjudication_model = results[0].get("adjudication_model", runner_model) if results else "gemma3:12b"
    experimental_case_count = sum(1 for item in results if bool(item.get("experimental_path", False)))
    candidate_plan_attempts = sum(int(item.get("candidate_plan_count", 0) or 0) for item in results)
    selected_candidate_ranks = [float(item.get("selected_candidate_rank", 0) or 0) for item in results if item.get("selected_candidate_rank")]
    repair_cycle_count = sum(int(item.get("repair_cycle_count", 0) or 0) for item in results)
    structured_prompt_failures = sum(int(item.get("structured_prompt_failures", 0) or 0) for item in results)
    for item in results:
        key = f'{item.get("initial_eval", "")}->{item.get("final_error_category", "Success" if item["success"] else "UnknownError")}'
        transitions[key] = transitions.get(key, 0) + 1
        reason = item.get("dependency_reason")
        if reason:
            dependency_reason_counts[reason] = dependency_reason_counts.get(reason, 0) + 1

    summary = BenchmarkSummary(
        run_id=run_dir.name,
        total_cases=total_cases,
        successes=successes,
        failures=failures,
        success_rate=(successes / total_cases) if total_cases else 0.0,
        initial_import_errors=initial_import_errors,
        final_import_errors=final_import_errors,
        mean_attempts_to_success=mean_attempts,
        mean_wall_clock_time=mean_wall_clock,
        resolver=resolver,
        preset=preset,
        prompt_profile=prompt_profile,
        model_profile=model_profile,
        use_moe=use_moe,
        use_rag=use_rag,
        use_langchain=use_langchain,
        rag_mode=rag_mode,
        structured_prompting=structured_prompting,
        extraction_model=extraction_model,
        runner_model=runner_model,
        version_model=version_model,
        repair_model=repair_model,
        adjudication_model=adjudication_model,
        total_wall_clock_time=total_wall_clock,
        total_wall_clock_human=format_duration(total_wall_clock),
        transitions=transitions,
        dependency_reason_counts=dependency_reason_counts,
        experimental_case_count=experimental_case_count,
        candidate_plan_attempts=candidate_plan_attempts,
        average_candidate_rank_selected=(
            sum(selected_candidate_ranks) / len(selected_candidate_ranks) if selected_candidate_ranks else 0.0
        ),
        repair_cycle_count=repair_cycle_count,
        structured_prompt_failures=structured_prompt_failures,
    )
    write_summary_artifacts(run_dir, results, summary)
    build_timeline_view(run_dir)
    return summary


def write_summary_artifacts(run_dir: Path, results: list[dict], summary: BenchmarkSummary) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
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
        for row in results:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "success": row["success"],
                    "attempts": row["attempts"],
                    "initial_eval": row.get("initial_eval", ""),
                    "final_error_category": row.get("final_error_category", ""),
                    "wall_clock_seconds": row.get("wall_clock_seconds", 0.0),
                    "started_at": row.get("started_at", ""),
                    "finished_at": row.get("finished_at", ""),
                }
            )

    lines = [
        "# Benchmark Summary",
        "",
        f"- Run ID: `{summary.run_id}`",
        f"- Resolver: `{summary.resolver}`",
        f"- Preset: `{summary.preset}`",
        f"- Prompt profile: `{summary.prompt_profile}`",
        f"- Model profile: `{summary.model_profile}`",
        f"- Runtime: `{'moe' if summary.use_moe else 'single'}` / `{'rag' if summary.use_rag else 'no-rag'}` / `{'langchain' if summary.use_langchain else 'direct'}`",
        f"- RAG mode: `{summary.rag_mode}`",
        f"- Structured prompting: `{'enabled' if summary.structured_prompting else 'disabled'}`",
        f"- Models: `{summary.extraction_model}` / `{summary.runner_model}` / `{summary.version_model}` / `{summary.repair_model}` / `{summary.adjudication_model}`",
        f"- Total cases: `{summary.total_cases}`",
        f"- Success rate: `{summary.success_rate:.2%}`",
        f"- Initial ImportErrors: `{summary.initial_import_errors}`",
        f"- Final ImportErrors: `{summary.final_import_errors}`",
        f"- Experimental cases: `{summary.experimental_case_count}`",
        f"- Candidate plan attempts: `{summary.candidate_plan_attempts}`",
        f"- Average selected candidate rank: `{summary.average_candidate_rank_selected:.2f}`",
        f"- Repair cycles: `{summary.repair_cycle_count}`",
        f"- Structured prompt failures: `{summary.structured_prompt_failures}`",
        f"- Time to finish: `{summary.total_wall_clock_human}`",
    ]
    (run_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_results_artifacts(run_dir, results)


def _modules_from_result(row: dict[str, Any]) -> list[str]:
    candidate_provenance = row.get("candidate_provenance", {})
    if isinstance(candidate_provenance, dict) and candidate_provenance:
        return sorted(str(module) for module in candidate_provenance.keys())

    dependencies = row.get("dependencies", [])
    modules: list[str] = []
    if isinstance(dependencies, list):
        for dependency in dependencies:
            value = str(dependency).strip()
            if not value:
                continue
            modules.append(value.split("==", 1)[0])
    return sorted(set(modules))


def write_results_artifacts(run_dir: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(results, key=lambda item: str(item.get("case_id", ""))), start=1):
        modules = _modules_from_result(row)
        rows.append(
            {
                "case_number": index,
                "case_id": row.get("case_id", ""),
                "modules": ", ".join(modules),
                "result": "success" if row.get("success", False) else "failure",
                "attempts": row.get("attempts", 0),
                "initial_eval": row.get("initial_eval", ""),
                "final_error_category": row.get("final_error_category", ""),
                "dependencies": ", ".join(str(item) for item in row.get("dependencies", [])),
                "dependency_reason": row.get("dependency_reason", ""),
                "wall_clock_seconds": row.get("wall_clock_seconds", 0.0),
                "started_at": row.get("started_at", ""),
                "finished_at": row.get("finished_at", ""),
            }
        )

    (run_dir / "results.json").write_text(json.dumps({"run_id": run_dir.name, "rows": rows}, indent=2), encoding="utf-8")

    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
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
                "wall_clock_seconds",
                "started_at",
                "finished_at",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Case Results",
        "",
        f"Run ID: `{run_dir.name}`",
        "",
        "| # | Case ID | Modules | Result | Attempts | Final Status | Dependencies | Time (s) |",
        "| ---: | --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
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
    (run_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_failures(run_dir: Path, limit: int = 10, category: str | None = None) -> dict[str, Any]:
    results = load_run_results(run_dir)
    failures = [item for item in results if not item.get("success", False)]
    counts: dict[str, int] = {}
    for item in failures:
        failure_category = item.get("final_error_category", "UnknownError")
        counts[failure_category] = counts.get(failure_category, 0) + 1

    if category:
        failures = [item for item in failures if item.get("final_error_category") == category]

    failures.sort(
        key=lambda item: (
            -int(item.get("attempts", 0)),
            -float(item.get("wall_clock_seconds", 0.0)),
            item.get("case_id", ""),
        )
    )

    selected_cases = []
    for item in failures[:limit]:
        attempt_records = item.get("attempt_records", [])
        last_attempt = attempt_records[-1] if attempt_records else {}
        error_details = str(last_attempt.get("error_details", "")).strip()
        excerpt = error_details[:800]
        if len(error_details) > 800:
            excerpt += "..."
        selected_cases.append(
            {
                "case_id": item.get("case_id", ""),
                "initial_eval": item.get("initial_eval", ""),
                "final_error_category": item.get("final_error_category", "UnknownError"),
                "attempts": item.get("attempts", 0),
                "wall_clock_seconds": item.get("wall_clock_seconds", 0.0),
                "dependencies": item.get("dependencies", []),
                "last_attempt": {
                    "attempt_number": last_attempt.get("attempt_number"),
                    "exit_code": last_attempt.get("exit_code"),
                    "error_category": last_attempt.get("error_category", item.get("final_error_category", "")),
                    "error_excerpt": excerpt,
                },
            }
        )

    analysis = {
        "run_id": run_dir.name,
        "total_failures": len([item for item in results if not item.get("success", False)]),
        "selected_failures": len(failures),
        "selected_category": category or "ALL",
        "categories": dict(sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))),
        "cases": selected_cases,
    }
    output_name = "failure-analysis.json" if not category else f"failure-analysis-{category}.json"
    (run_dir / output_name).write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis


def write_timeline_artifacts(run_dir: Path, report: dict[str, Any]) -> None:
    rows = report["rows"]
    (run_dir / "timeline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (run_dir / "timeline.csv").open("w", newline="", encoding="utf-8") as handle:
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
        for row in rows:
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

    lines = [
        "# Case Timeline",
        "",
        f"Run ID: `{report['run_id']}`",
        f"Run started: `{report['run_started_at'] or 'unknown'}`",
        f"Run finished: `{report['run_finished_at'] or 'unknown'}`",
        f"Timeline span: `{report['total_span_human']}`",
        "",
        "| Case ID | Start | End | Duration | Attempts | Status | Timeline |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        status = "Success" if row["success"] else row["final_error_category"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["started_at"] or "unknown",
                    row["finished_at"] or "unknown",
                    row["duration_human"],
                    str(row["attempts"]),
                    status,
                    f"`{row['timeline_bar']}`",
                ]
            )
            + " |"
        )
    (run_dir / "timeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_module_success_table(
    run_dir: Path,
    dataset: GistableDataset,
    ref: str | None = None,
    top_n: int = 15,
    grouping: GroupingMode = "canonical",
    paper_compatible: bool = False,
) -> dict[str, Any]:
    results = load_run_results(run_dir)
    result_map = {item.get("case_id", ""): item for item in results}
    counts: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)
    covered: dict[str, int] = defaultdict(int)
    skipped_case_ids: list[str] = []
    all_rows: list[dict[str, Any]] = []

    if paper_compatible:
        cohort = "paper-compatible"
        case_ids = paper_hard_subset_case_ids(dataset, ref)
    else:
        cohort = "run"
        case_ids = [item["case_id"] for item in results if item.get("case_id")]

    def process_case(case_id: str) -> tuple[str, list[str] | None]:
        if paper_compatible:
            snippet_path = dataset.dataset_root(ref) / "all-gists" / case_id / "snippet.py"
        else:
            snippet_path = dataset.load_case(case_id, ref).snippet_path
        source_code = safe_read_text(snippet_path)
        if source_code is None:
            return case_id, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            import_roots = filter_third_party_imports(extract_import_roots_from_code(source_code))
            modules = normalize_candidate_packages(import_roots, import_roots)
        return case_id, sorted(set(modules))

    worker_count = min(32, max(1, (os.cpu_count() or 1) * 2))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for case_id, modules in executor.map(process_case, case_ids):
            if modules is None:
                skipped_case_ids.append(case_id)
                continue
            item = result_map.get(case_id)
            for module in modules:
                module_name = canonical_module_name(module, grouping)
                counts[module_name] += 1
                if item is not None:
                    covered[module_name] += 1
                if item is not None and item.get("success", False):
                    successes[module_name] += 1

    for module_name, project_count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0])):
        success_count = successes.get(module_name, 0)
        covered_count = covered.get(module_name, 0)
        denominator = covered_count if paper_compatible and covered_count else project_count
        success_rate = (success_count / denominator * 100.0) if denominator else 0.0
        all_rows.append(
            {
                "module_name": module_name,
                "projects": project_count,
                "covered_projects": covered_count,
                "successes": success_count,
                "coverage_rate": round((covered_count / project_count * 100.0), 2) if project_count else 0.0,
                "apd_success_rate": round(success_rate, 2),
                "apd_rate_denominator": denominator,
            }
        )
    if paper_compatible and any(row["covered_projects"] > 0 for row in all_rows):
        covered_rows = [row for row in all_rows if row["covered_projects"] > 0]
        uncovered_rows = [row for row in all_rows if row["covered_projects"] == 0]
        top_rows = covered_rows[:top_n]
        if len(top_rows) < top_n:
            top_rows.extend(uncovered_rows[: top_n - len(top_rows)])
        display_strategy = "covered-first"
    else:
        top_rows = all_rows[:top_n]
        display_strategy = "global-frequency"

    report = {
        "run_id": run_dir.name,
        "cohort": cohort,
        "grouping": grouping,
        "top_n": top_n,
        "paper_compatible": paper_compatible,
        "total_cohort_cases": len(case_ids),
        "covered_case_count": sum(1 for case_id in case_ids if case_id in result_map),
        "skipped_case_count": len(skipped_case_ids),
        "skipped_case_ids": skipped_case_ids,
        "display_strategy": display_strategy,
        "rows": top_rows,
        "top_rows": top_rows,
        "all_rows": all_rows,
    }
    write_module_success_artifacts(run_dir, report)
    return report


def write_module_success_artifacts(run_dir: Path, report: dict[str, Any]) -> None:
    rows = report["top_rows"]
    all_rows = report["all_rows"]
    cohort = report.get("cohort", "run")
    covered_case_count = report.get("covered_case_count", 0)
    total_cohort_cases = report.get("total_cohort_cases", 0)
    skipped_case_count = report.get("skipped_case_count", 0)
    display_strategy = report.get("display_strategy", "global-frequency")
    suffix_parts: list[str] = []
    if report.get("paper_compatible"):
        suffix_parts.append("paper")
    if report["grouping"] == "raw":
        suffix_parts.append("raw")
    suffix = "" if not suffix_parts else "-" + "-".join(suffix_parts)
    (run_dir / f"module-success{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (run_dir / f"module-success{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "module_name",
                "projects",
                "covered_projects",
                "successes",
                "coverage_rate",
                "apd_rate_denominator",
                "apd_success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

        lines = [
            "# Module Success Table",
            "",
            f"Run ID: `{report['run_id']}`",
            f"Cohort: `{cohort}`",
            f"Covered cases: `{covered_case_count}/{total_cohort_cases}`",
            f"Skipped unreadable cases: `{skipped_case_count}`",
            f"Display strategy: `{display_strategy}`",
            f"APD rate denominator: `covered projects`{' within the paper cohort' if report.get('paper_compatible') else ''}",
            "",
            "| Module Name | # Projects | APD |",
            "| --- | ---: | ---: |",
        ]
    for row in rows:
        lines.append(
            f"| {row['module_name']} | {row['projects']} | {row['apd_success_rate']:.2f} |"
        )
    (run_dir / f"module-success{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
