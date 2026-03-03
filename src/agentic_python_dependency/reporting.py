from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentic_python_dependency.benchmark.gistable import GistableDataset
from agentic_python_dependency.state import BenchmarkSummary
from agentic_python_dependency.tools.import_extractor import (
    extract_import_roots_from_code,
    filter_third_party_imports,
    normalize_candidate_packages,
)


def load_run_results(run_dir: Path) -> list[dict[str, Any]]:
    result_paths = sorted(run_dir.glob("*/result.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(math.floor(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
    for item in results:
        key = f'{item.get("initial_eval", "")}->{item.get("final_error_category", "Success" if item["success"] else "UnknownError")}'
        transitions[key] = transitions.get(key, 0) + 1

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
        total_wall_clock_time=total_wall_clock,
        total_wall_clock_human=format_duration(total_wall_clock),
        transitions=transitions,
    )
    write_summary_artifacts(run_dir, results, summary)
    return summary


def write_summary_artifacts(run_dir: Path, results: list[dict], summary: BenchmarkSummary) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "success", "attempts", "initial_eval", "final_error_category", "wall_clock_seconds"],
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
                }
            )

    lines = [
        "# Benchmark Summary",
        "",
        f"- Run ID: `{summary.run_id}`",
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


def build_module_success_table(
    run_dir: Path,
    dataset: GistableDataset,
    ref: str | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    results = load_run_results(run_dir)
    counts: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)

    for item in results:
        case = dataset.load_case(item["case_id"], ref)
        source_code = case.snippet_path.read_text(encoding="utf-8")
        import_roots = filter_third_party_imports(extract_import_roots_from_code(source_code))
        modules = normalize_candidate_packages(import_roots, import_roots)
        for module in set(modules):
            module_name = module.lower()
            counts[module_name] += 1
            if item.get("success", False):
                successes[module_name] += 1

    rows = []
    for module_name, project_count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))[:top_n]:
        success_count = successes.get(module_name, 0)
        success_rate = (success_count / project_count * 100.0) if project_count else 0.0
        rows.append(
            {
                "module_name": module_name,
                "projects": project_count,
                "successes": success_count,
                "apd_success_rate": round(success_rate, 2),
            }
        )

    report = {
        "run_id": run_dir.name,
        "top_n": top_n,
        "rows": rows,
    }
    write_module_success_artifacts(run_dir, report)
    return report


def write_module_success_artifacts(run_dir: Path, report: dict[str, Any]) -> None:
    rows = report["rows"]
    (run_dir / "module-success.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (run_dir / "module-success.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["module_name", "projects", "successes", "apd_success_rate"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Module Success Table",
        "",
        f"Run ID: `{report['run_id']}`",
        "",
        "| Module Name | # Projects | APD |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['module_name']} | {row['projects']} | {row['apd_success_rate']:.2f} |"
        )
    (run_dir / "module-success.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
