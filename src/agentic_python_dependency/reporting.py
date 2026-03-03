from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
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
        if row.get("final-eval") == "ImportError"
        and (dataset.dataset_root(ref) / "all-gists" / row["id"] / "snippet.py").exists()
    }
    return sorted(hard_ids)


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
    preset = results[0].get("preset", "optimized") if results else "optimized"
    prompt_profile = results[0].get("prompt_profile", "optimized") if results else "optimized"
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
        preset=preset,
        prompt_profile=prompt_profile,
        total_wall_clock_time=total_wall_clock,
        total_wall_clock_human=format_duration(total_wall_clock),
        transitions=transitions,
        dependency_reason_counts=dependency_reason_counts,
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
        f"- Preset: `{summary.preset}`",
        f"- Prompt profile: `{summary.prompt_profile}`",
        f"- Total cases: `{summary.total_cases}`",
        f"- Success rate: `{summary.success_rate:.2%}`",
        f"- Initial ImportErrors: `{summary.initial_import_errors}`",
        f"- Final ImportErrors: `{summary.final_import_errors}`",
        f"- Time to finish: `{summary.total_wall_clock_human}`",
    ]
    (run_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    if paper_compatible:
        cohort = "paper-compatible"
        case_ids = paper_hard_subset_case_ids(dataset, ref)
    else:
        cohort = "run"
        case_ids = [item["case_id"] for item in results if item.get("case_id")]

    for case_id in case_ids:
        if paper_compatible:
            snippet_path = dataset.dataset_root(ref) / "all-gists" / case_id / "snippet.py"
        else:
            snippet_path = dataset.load_case(case_id, ref).snippet_path
        source_code = snippet_path.read_text(encoding="utf-8", errors="replace")
        import_roots = filter_third_party_imports(extract_import_roots_from_code(source_code))
        modules = normalize_candidate_packages(import_roots, import_roots)
        item = result_map.get(case_id)
        for module in set(modules):
            module_name = canonical_module_name(module, grouping)
            counts[module_name] += 1
            if item is not None:
                covered[module_name] += 1
            if item is not None and item.get("success", False):
                successes[module_name] += 1

    all_rows = []
    for module_name, project_count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0])):
        success_count = successes.get(module_name, 0)
        covered_count = covered.get(module_name, 0)
        success_rate = (success_count / project_count * 100.0) if project_count else 0.0
        all_rows.append(
            {
                "module_name": module_name,
                "projects": project_count,
                "covered_projects": covered_count,
                "successes": success_count,
                "coverage_rate": round((covered_count / project_count * 100.0), 2) if project_count else 0.0,
                "apd_success_rate": round(success_rate, 2),
            }
        )
    top_rows = all_rows[:top_n]

    report = {
        "run_id": run_dir.name,
        "cohort": cohort,
        "grouping": grouping,
        "top_n": top_n,
        "paper_compatible": paper_compatible,
        "total_cohort_cases": len(case_ids),
        "covered_case_count": sum(1 for case_id in case_ids if case_id in result_map),
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
            "",
            "| Module Name | # Projects | APD |",
            "| --- | ---: | ---: |",
        ]
    for row in rows:
        lines.append(
            f"| {row['module_name']} | {row['projects']} | {row['apd_success_rate']:.2f} |"
        )
    (run_dir / f"module-success{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
