import json
from pathlib import Path

from agentic_python_dependency.reporting import canonical_module_name, summarize_run, write_module_success_artifacts


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


def test_write_module_success_artifacts_uses_raw_suffix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run123"
    run_dir.mkdir(parents=True)
    report = {
        "run_id": "run123",
        "grouping": "raw",
        "top_n": 1,
        "rows": [{"module_name": "requests", "projects": 1, "successes": 1, "apd_success_rate": 100.0}],
        "top_rows": [{"module_name": "requests", "projects": 1, "successes": 1, "apd_success_rate": 100.0}],
        "all_rows": [{"module_name": "requests", "projects": 1, "successes": 1, "apd_success_rate": 100.0}],
    }

    write_module_success_artifacts(run_dir, report)

    assert (run_dir / "module-success-raw.json").exists()
    assert (run_dir / "module-success-raw.csv").exists()
    assert (run_dir / "module-success-raw.md").exists()
