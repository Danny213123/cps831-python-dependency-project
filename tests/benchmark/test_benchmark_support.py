import csv
import json
from pathlib import Path

from agentic_python_dependency.benchmark.gistable import GistableDataset
from agentic_python_dependency.benchmark.subsets import build_smoke30
from agentic_python_dependency.config import Settings
from agentic_python_dependency.reporting import (
    analyze_failures,
    build_module_success_table,
    format_duration,
    summarize_run,
)


def make_dataset(settings: Settings) -> Path:
    root = settings.benchmark_dir / "gistable" / settings.benchmark_ref
    (root / "dockerized-gists").mkdir(parents=True, exist_ok=True)
    (root / "all-gists").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def write_case(root: Path, case_id: str) -> None:
    case_root = root / "dockerized-gists" / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "snippet.py").write_text("print('ok')\n", encoding="utf-8")
    (case_root / "Dockerfile").write_text("FROM python:3.12-slim\nCMD [\"python\", \"snippet.py\"]\n", encoding="utf-8")
    all_case_root = root / "all-gists" / case_id
    all_case_root.mkdir(parents=True, exist_ok=True)
    (all_case_root / "snippet.py").write_text("print('ok')\n", encoding="utf-8")


def test_build_smoke30_is_deterministic(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    root = make_dataset(settings)
    rows = []
    for index in range(20):
        case_id = f"imp{index:02d}"
        write_case(root, case_id)
        rows.append({"id": case_id, "url": "", "initial-eval": "ImportError", "final-eval": "", "error": ""})
    for index in range(5):
        case_id = f"succ{index:02d}"
        write_case(root, case_id)
        rows.append({"id": case_id, "url": "", "initial-eval": "Success", "final-eval": "", "error": ""})
    for index in range(3):
        case_id = f"syn{index:02d}"
        write_case(root, case_id)
        rows.append({"id": case_id, "url": "", "initial-eval": "SyntaxError", "final-eval": "", "error": ""})
    for index in range(2):
        case_id = f"other{index:02d}"
        write_case(root, case_id)
        rows.append({"id": case_id, "url": "", "initial-eval": "NameError", "final-eval": "", "error": ""})

    with (root / "results" / "naive-inference-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "url", "initial-eval", "final-eval", "error"])
        writer.writeheader()
        writer.writerows(rows)

    dataset = GistableDataset(settings)
    smoke = build_smoke30(dataset)

    assert len(smoke) == 30
    assert smoke[:2] == ["imp00", "imp01"]


def test_build_smoke30_backfills_when_buckets_are_short(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    root = make_dataset(settings)
    rows = []
    for index in range(20):
        case_id = f"imp{index:02d}"
        write_case(root, case_id)
        rows.append({"id": case_id, "url": "", "initial-eval": "ImportError", "final-eval": "", "error": ""})
    for index in range(5):
        case_id = f"succ{index:02d}"
        write_case(root, case_id)
        rows.append({"id": case_id, "url": "", "initial-eval": "Success", "final-eval": "", "error": ""})
    for index in range(8):
        case_id = f"extra{index:02d}"
        write_case(root, case_id)
        rows.append({"id": case_id, "url": "", "initial-eval": "RuntimeError", "final-eval": "", "error": ""})

    with (root / "results" / "naive-inference-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "url", "initial-eval", "final-eval", "error"])
        writer.writeheader()
        writer.writerows(rows)

    dataset = GistableDataset(settings)
    smoke = build_smoke30(dataset)

    assert len(smoke) == 30
    assert smoke[-5:] == ["extra00", "extra01", "extra02", "extra03", "extra04"]


def test_summarize_run_writes_summary_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "runs" / "run123"
    case_one = run_dir / "case1"
    case_two = run_dir / "case2"
    case_one.mkdir(parents=True, exist_ok=True)
    case_two.mkdir(parents=True, exist_ok=True)
    (case_one / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case1",
                "success": True,
                "attempts": 1,
                "initial_eval": "ImportError",
                "final_error_category": "Success",
                "wall_clock_seconds": 1.0,
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
                "initial_eval": "SyntaxError",
                "final_error_category": "SyntaxError",
                "wall_clock_seconds": 2.0,
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_run(run_dir, total_elapsed_seconds=5.0)

    assert summary.total_cases == 2
    assert summary.total_wall_clock_time == 5.0
    assert summary.total_wall_clock_human == "00:00:05"
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "leaderboard.md").exists()


def test_format_duration_renders_hms() -> None:
    assert format_duration(0.0) == "00:00:00"
    assert format_duration(65.2) == "00:01:05"
    assert format_duration(3661.9) == "01:01:01"


def test_analyze_failures_writes_failure_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "runs" / "run123"
    case_one = run_dir / "case1"
    case_two = run_dir / "case2"
    case_one.mkdir(parents=True, exist_ok=True)
    case_two.mkdir(parents=True, exist_ok=True)
    (case_one / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case1",
                "success": False,
                "attempts": 2,
                "initial_eval": "ImportError",
                "final_error_category": "UnknownError",
                "dependencies": ["requests==2.32.3"],
                "wall_clock_seconds": 3.0,
                "attempt_records": [
                    {
                        "attempt_number": 2,
                        "exit_code": 1,
                        "error_category": "UnknownError",
                        "error_details": "some long failure details",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (case_two / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case2",
                "success": False,
                "attempts": 1,
                "initial_eval": "SyntaxError",
                "final_error_category": "SyntaxError",
                "dependencies": [],
                "wall_clock_seconds": 1.0,
                "attempt_records": [
                    {
                        "attempt_number": 1,
                        "exit_code": 1,
                        "error_category": "SyntaxError",
                        "error_details": "SyntaxError: invalid syntax",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_failures(run_dir, limit=1, category="UnknownError")

    assert analysis["total_failures"] == 2
    assert analysis["selected_failures"] == 1
    assert analysis["categories"] == {"SyntaxError": 1, "UnknownError": 1}
    assert analysis["cases"][0]["case_id"] == "case1"
    assert (run_dir / "failure-analysis-UnknownError.json").exists()


def test_build_module_success_table_writes_artifacts(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    root = make_dataset(settings)
    write_case(root, "case1")
    write_case(root, "case2")
    (root / "dockerized-gists" / "case1" / "snippet.py").write_text("import requests\nimport yaml\n", encoding="utf-8")
    (root / "dockerized-gists" / "case2" / "snippet.py").write_text("import requests\n", encoding="utf-8")
    with (root / "results" / "naive-inference-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "url", "initial-eval", "final-eval", "error"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "case1", "url": "", "initial-eval": "ImportError", "final-eval": "", "error": ""},
                {"id": "case2", "url": "", "initial-eval": "ImportError", "final-eval": "", "error": ""},
            ]
        )

    run_dir = tmp_path / "artifacts" / "runs" / "run123"
    (run_dir / "case1").mkdir(parents=True, exist_ok=True)
    (run_dir / "case2").mkdir(parents=True, exist_ok=True)
    (run_dir / "case1" / "result.json").write_text(
        json.dumps({"case_id": "case1", "success": True, "attempts": 1, "initial_eval": "ImportError"}),
        encoding="utf-8",
    )
    (run_dir / "case2" / "result.json").write_text(
        json.dumps({"case_id": "case2", "success": False, "attempts": 1, "initial_eval": "ImportError"}),
        encoding="utf-8",
    )

    dataset = GistableDataset(settings)
    report = build_module_success_table(run_dir, dataset, top_n=5)

    assert report["rows"][0]["module_name"] == "requests"
    assert report["rows"][0]["projects"] == 2
    assert report["rows"][0]["apd_success_rate"] == 50.0
    assert (run_dir / "module-success.json").exists()
    assert (run_dir / "module-success.csv").exists()
    assert (run_dir / "module-success.md").exists()


def test_build_module_success_table_supports_paper_compatible_cohort(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    root = make_dataset(settings)
    write_case(root, "case1")
    write_case(root, "case2")
    write_case(root, "case3")
    (root / "all-gists" / "case1" / "snippet.py").write_text("import django\n", encoding="utf-8")
    (root / "all-gists" / "case2" / "snippet.py").write_text("import requests\n", encoding="utf-8")
    (root / "all-gists" / "case3" / "snippet.py").write_text("import requests\n", encoding="utf-8")
    with (root / "results" / "naive-inference-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "url", "initial-eval", "final-eval", "error"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "case1", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
                {"id": "case2", "url": "", "initial-eval": "ImportError", "final-eval": "", "error": ""},
                {"id": "case3", "url": "", "initial-eval": "Success", "final-eval": "ImportError", "error": ""},
            ]
        )

    run_dir = tmp_path / "artifacts" / "runs" / "run123"
    (run_dir / "case1").mkdir(parents=True, exist_ok=True)
    (run_dir / "case2").mkdir(parents=True, exist_ok=True)
    (run_dir / "case1" / "result.json").write_text(
        json.dumps({"case_id": "case1", "success": True, "attempts": 1, "initial_eval": "ImportError"}),
        encoding="utf-8",
    )
    (run_dir / "case2" / "result.json").write_text(
        json.dumps({"case_id": "case2", "success": False, "attempts": 1, "initial_eval": "ImportError"}),
        encoding="utf-8",
    )

    dataset = GistableDataset(settings)
    report = build_module_success_table(run_dir, dataset, top_n=5, paper_compatible=True)

    assert report["cohort"] == "paper-compatible"
    assert report["total_cohort_cases"] == 2
    assert report["covered_case_count"] == 2
    assert report["rows"][0]["module_name"] == "django"
    assert report["rows"][0]["apd_success_rate"] == 100.0
    assert report["rows"][1]["module_name"] == "requests"
    assert report["rows"][1]["apd_success_rate"] == 0.0
    assert (run_dir / "module-success-paper.json").exists()
    assert (run_dir / "module-success-paper.csv").exists()
    assert (run_dir / "module-success-paper.md").exists()


def test_build_module_success_table_uses_covered_projects_for_paper_preview_rates(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    root = make_dataset(settings)
    write_case(root, "case1")
    write_case(root, "case2")
    write_case(root, "case3")
    (root / "all-gists" / "case1" / "snippet.py").write_text("import numpy\n", encoding="utf-8")
    (root / "all-gists" / "case2" / "snippet.py").write_text("import numpy\n", encoding="utf-8")
    (root / "all-gists" / "case3" / "snippet.py").write_text("import numpy\n", encoding="utf-8")
    with (root / "results" / "naive-inference-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "url", "initial-eval", "final-eval", "error"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "case1", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
                {"id": "case2", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
                {"id": "case3", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
            ]
        )

    run_dir = tmp_path / "artifacts" / "runs" / "run123"
    (run_dir / "case1").mkdir(parents=True, exist_ok=True)
    (run_dir / "case1" / "result.json").write_text(
        json.dumps({"case_id": "case1", "success": True, "attempts": 1, "initial_eval": "ImportError"}),
        encoding="utf-8",
    )

    dataset = GistableDataset(settings)
    report = build_module_success_table(run_dir, dataset, top_n=5, paper_compatible=True)

    assert report["rows"][0]["projects"] == 3
    assert report["rows"][0]["covered_projects"] == 1
    assert report["rows"][0]["apd_rate_denominator"] == 1
    assert report["rows"][0]["apd_success_rate"] == 100.0


def test_build_module_success_table_prefers_covered_rows_for_partial_paper_preview(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    root = make_dataset(settings)
    write_case(root, "case1")
    write_case(root, "case2")
    write_case(root, "case3")
    write_case(root, "case4")
    (root / "all-gists" / "case1" / "snippet.py").write_text("import requests\n", encoding="utf-8")
    (root / "all-gists" / "case2" / "snippet.py").write_text("import requests\n", encoding="utf-8")
    (root / "all-gists" / "case3" / "snippet.py").write_text("import django\n", encoding="utf-8")
    (root / "all-gists" / "case4" / "snippet.py").write_text("import django\n", encoding="utf-8")
    with (root / "results" / "naive-inference-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "url", "initial-eval", "final-eval", "error"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "case1", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
                {"id": "case2", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
                {"id": "case3", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
                {"id": "case4", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
            ]
        )

    run_dir = tmp_path / "artifacts" / "runs" / "run123"
    (run_dir / "case1").mkdir(parents=True, exist_ok=True)
    (run_dir / "case1" / "result.json").write_text(
        json.dumps({"case_id": "case1", "success": True, "attempts": 1, "initial_eval": "ImportError"}),
        encoding="utf-8",
    )

    dataset = GistableDataset(settings)
    report = build_module_success_table(run_dir, dataset, top_n=2, paper_compatible=True)

    assert report["display_strategy"] == "covered-first"
    assert report["rows"][0]["module_name"] == "requests"
    assert report["rows"][0]["covered_projects"] == 1


def test_build_module_success_table_skips_unreadable_paper_case(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    root = make_dataset(settings)
    write_case(root, "case1")
    write_case(root, "case2")
    (root / "all-gists" / "case1" / "snippet.py").write_text("import django\n", encoding="utf-8")
    (root / "all-gists" / "case2" / "snippet.py").write_text("import requests\n", encoding="utf-8")
    with (root / "results" / "naive-inference-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "url", "initial-eval", "final-eval", "error"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "case1", "url": "", "initial-eval": "ImportError", "final-eval": "ImportError", "error": ""},
                {"id": "case2", "url": "", "initial-eval": "ImportError", "final-eval": "", "error": ""},
            ]
        )

    run_dir = tmp_path / "artifacts" / "runs" / "run123"
    (run_dir / "case1").mkdir(parents=True, exist_ok=True)
    (run_dir / "case1" / "result.json").write_text(
        json.dumps({"case_id": "case1", "success": True, "attempts": 1, "initial_eval": "ImportError"}),
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def flaky_read_text(self: Path, *args, **kwargs):
        if self.name == "snippet.py" and self.parent.name == "case2":
            raise OSError(22, "Invalid argument")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    dataset = GistableDataset(settings)
    report = build_module_success_table(run_dir, dataset, top_n=5, paper_compatible=True)

    assert report["skipped_case_count"] == 1
    assert report["skipped_case_ids"] == ["case2"]
    assert report["rows"][0]["module_name"] == "django"
