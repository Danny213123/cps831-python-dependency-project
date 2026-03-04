import json
from pathlib import Path

from agentic_python_dependency.reporting import (
    build_timeline_view,
    canonical_module_name,
    summarize_run,
    write_module_success_artifacts,
)


def test_canonical_module_name_merges_known_families() -> None:
    assert canonical_module_name("PIL", "canonical") == "pillow"
    assert canonical_module_name("beautifulsoup", "canonical") == "beautifulsoup4"
    assert canonical_module_name("tensorflow-gpu", "canonical") == "tensorflow"
    assert canonical_module_name("PIL", "raw") == "pil"


def test_summarize_run_collects_preset_and_dependency_reasons(tmp_path: Path) -> None:
    run_dir = tmp_path / "run123"
    case_dir = run_dir / "case1"
    case_dir.mkdir(parents=True)
    (case_dir / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case1",
                "success": True,
                "attempts": 1,
                "initial_eval": "ImportError",
                "final_error_category": "Success",
                "wall_clock_seconds": 1.5,
                "started_at": "2026-03-03T10:00:00+00:00",
                "finished_at": "2026-03-03T10:00:01.500000+00:00",
                "preset": "balanced",
                "prompt_profile": "optimized-strict",
                "dependency_reason": "deterministic_version_selector",
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_run(run_dir)

    assert summary.preset == "balanced"
    assert summary.prompt_profile == "optimized-strict"
    assert summary.dependency_reason_counts == {"deterministic_version_selector": 1}
    assert (run_dir / "timeline.json").exists()
    assert (run_dir / "timeline.csv").exists()
    assert (run_dir / "timeline.md").exists()


def test_write_module_success_artifacts_uses_raw_suffix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run123"
    run_dir.mkdir(parents=True)
    report = {
        "run_id": "run123",
        "grouping": "raw",
        "top_n": 1,
        "rows": [{"module_name": "requests", "projects": 1, "successes": 1, "apd_rate_denominator": 1, "apd_success_rate": 100.0}],
        "top_rows": [{"module_name": "requests", "projects": 1, "successes": 1, "apd_rate_denominator": 1, "apd_success_rate": 100.0}],
        "all_rows": [{"module_name": "requests", "projects": 1, "successes": 1, "apd_rate_denominator": 1, "apd_success_rate": 100.0}],
    }

    write_module_success_artifacts(run_dir, report)

    assert (run_dir / "module-success-raw.json").exists()
    assert (run_dir / "module-success-raw.csv").exists()
    assert (run_dir / "module-success-raw.md").exists()


def test_build_timeline_view_orders_cases_and_emits_relative_offsets(tmp_path: Path) -> None:
    run_dir = tmp_path / "run123"
    case_one = run_dir / "case1"
    case_two = run_dir / "case2"
    case_one.mkdir(parents=True)
    case_two.mkdir(parents=True)
    (case_one / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case1",
                "success": True,
                "attempts": 1,
                "final_error_category": "Success",
                "wall_clock_seconds": 3.0,
                "started_at": "2026-03-03T10:00:00+00:00",
                "finished_at": "2026-03-03T10:00:03+00:00",
            }
        ),
        encoding="utf-8",
    )
    (case_two / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case2",
                "success": False,
                "attempts": 2,
                "final_error_category": "TimeoutError",
                "wall_clock_seconds": 5.0,
                "started_at": "2026-03-03T10:00:04+00:00",
                "finished_at": "2026-03-03T10:00:09+00:00",
            }
        ),
        encoding="utf-8",
    )

    report = build_timeline_view(run_dir)

    assert report["run_started_at"] == "2026-03-03T10:00:00+00:00"
    assert report["run_finished_at"] == "2026-03-03T10:00:09+00:00"
    assert report["rows"][0]["case_id"] == "case1"
    assert report["rows"][0]["relative_start_seconds"] == 0.0
    assert report["rows"][1]["case_id"] == "case2"
    assert report["rows"][1]["relative_start_seconds"] == 4.0
    assert (run_dir / "timeline.md").read_text(encoding="utf-8").startswith("# Case Timeline")
