import json
from pathlib import Path

from agentic_python_dependency.config import Settings
from agentic_python_dependency.web_dashboard import get_case_detail, get_home_payload, get_run_detail, list_runs


def make_settings(tmp_path: Path) -> Settings:
    return Settings.from_env(project_root=tmp_path)


def test_list_runs_prefers_run_state_and_summary(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    run_dir = settings.artifacts_dir / "run123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "resolver": "apdr",
                "preset": "research",
                "prompt_profile": "research-rag",
                "research_bundle": "full",
                "research_features": ["repair_memory"],
                "benchmark_source": "competition-run",
                "target": "full",
                "total": 20,
                "completed": 5,
                "successes": 3,
                "failures": 2,
                "elapsed_seconds": 42.0,
                "docker_build_seconds_total": 30.0,
                "docker_run_seconds_total": 5.0,
                "llm_wall_clock_seconds_total": 7.0,
                "image_cache_hits": 2,
                "build_skips": 1,
                "last_updated_at": "2026-03-08T17:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"successes": 4, "failures": 1}), encoding="utf-8")

    runs = list_runs(settings)

    assert len(runs) == 1
    assert runs[0]["runId"] == "run123"
    assert runs[0]["status"] == "running"
    assert runs[0]["completed"] == 5
    assert runs[0]["dockerBuildSeconds"] == 30.0
    assert runs[0]["benchmarkSource"] == "competition-run"


def test_get_home_payload_exposes_command_center_fields(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    loadouts_dir = settings.data_dir / "loadouts"
    loadouts_dir.mkdir(parents=True, exist_ok=True)
    (loadouts_dir / "research.json").write_text("{}", encoding="utf-8")

    payload = get_home_payload(settings, host="0.0.0.0", port=8765)

    assert payload["home"]["title"] == "APDR Command Center"
    assert payload["home"]["server"]["localUrl"] == "http://127.0.0.1:8765"
    assert payload["home"]["loadouts"] == 1
    field_labels = [field["label"] for field in payload["home"]["fields"]]
    assert "Preset/Resolver" in field_labels
    assert "Artifacts" in field_labels
    action_labels = [action["label"] for action in payload["home"]["actions"]]
    assert action_labels == ["Run", "Reports", "Configure", "Loadouts", "Doctor", "Web"]


def test_get_run_detail_merges_case_runtime_rows(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    run_dir = settings.artifacts_dir / "run123"
    case_dir = run_dir / "case-1"
    case_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "resolver": "apdr",
                "preset": "research",
                "prompt_profile": "research-rag",
                "research_bundle": "baseline",
                "research_features": [],
                "benchmark_source": "all-gists",
                "target": "smoke30",
                "total": 10,
                "completed": 1,
                "successes": 1,
                "failures": 0,
                "elapsed_seconds": 12.0,
                "current_cases": ["case-2"],
                "recent_case_activity": [{"case_id": "case-2", "kind": "docker_build_start", "detail": "Building"}],
                "current_case_activity": [{"case_id": "case-2", "attempt": 1, "kind": "docker_build_start", "detail": "Building"}],
                "model_summary": "mistral-nemo-12b",
                "effective_model_profile": "mistral-nemo-12b",
                "effective_rag_mode": "hybrid",
                "effective_structured_prompting": True,
                "effective_repair_cycle_limit": 2,
                "effective_candidate_fallback_before_repair": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run-vs-csv.csv").write_text(
        "case_id,result_matches_csv,official_result,official_passed,official_python_modules\n"
        "case-1,PASS,OtherPass,True,requests\n",
        encoding="utf-8",
    )
    (case_dir / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "success": True,
                "attempts": 2,
                "target_python": "3.12",
                "runtime_profile": "import_smoke",
                "wall_clock_seconds": 3.5,
                "dependencies": ["requests==2.32.3", "urllib3==2.2.2"],
                "docker_build_seconds_total": 2.5,
                "docker_run_seconds_total": 0.5,
                "llm_wall_clock_seconds": 0.2,
                "candidate_plan_strategy": "llm-selected",
                "classifier_origin": "run",
                "root_cause_bucket": "success",
                "started_at": "2026-03-08T16:00:00+00:00",
                "finished_at": "2026-03-08T16:00:03+00:00",
                "attempt_records": [],
            }
        ),
        encoding="utf-8",
    )

    payload = get_run_detail(settings, "run123")

    assert payload is not None
    assert payload["run"]["runId"] == "run123"
    assert payload["activeCases"] == ["case-2"]
    assert payload["cases"][0]["pllmMatch"] == "MATCH"
    assert payload["cases"][0]["dependencyPreview"] == "requests==2.32.3, urllib3==2.2.2"


def test_get_case_detail_groups_attempt_activity_and_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    run_dir = settings.artifacts_dir / "run123"
    case_dir = run_dir / "case-1"
    attempt_dir = case_dir / "attempt_01"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-vs-csv.csv").write_text(
        "case_id,result_matches_csv,official_result,official_passed\ncase-1,FAIL,ImportError,False\n",
        encoding="utf-8",
    )
    (case_dir / "activity.log").write_text(
        "[2026-03-08T17:00:00+00:00] case=case-1 attempt=1 kind=docker_build_start Starting docker build.\n"
        "[2026-03-08T17:00:02+00:00] case=case-1 attempt=1 kind=docker_run_finish Container run failed.\n",
        encoding="utf-8",
    )
    (attempt_dir / "build.log").write_text("build output", encoding="utf-8")
    (attempt_dir / "run.log").write_text("run output", encoding="utf-8")
    (case_dir / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "success": False,
                "attempts": 1,
                "target_python": "2.7.18",
                "wall_clock_seconds": 4.2,
                "final_error_category": "ImportError",
                "dependencies": ["rx==1.5.8", "twisted==19.2.0"],
                "attempt_records": [
                    {
                        "attempt_number": 1,
                        "dependencies": ["rx==1.5.8", "twisted==19.2.0"],
                        "build_succeeded": True,
                        "run_succeeded": False,
                        "error_category": "ImportError",
                        "error_details": "cannot import name Disposable",
                        "llm_failure_analysis": "Runtime failure. The selected RxPY line is missing Disposable in rx.disposables.",
                        "llm_failure_analysis_model": "mistral-nemo:12b",
                        "validation_command": "python snippet.py",
                        "wall_clock_seconds": 4.2,
                        "build_wall_clock_seconds": 2.7,
                        "run_wall_clock_seconds": 0.8,
                        "build_skipped": False,
                        "image_cache_hit": False,
                        "artifact_dir": str(attempt_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = get_case_detail(settings, "run123", "case-1")

    assert payload is not None
    assert payload["case"]["pllmMatch"] == "MISS"
    assert payload["attempts"][0]["llmFailureAnalysis"].startswith("Runtime failure.")
    assert payload["attempts"][0]["llmFailureAnalysisModel"] == "mistral-nemo:12b"
    assert payload["attempts"][0]["activity"][0]["kind"] == "docker_build_start"
    assert payload["attempts"][0]["files"][0]["path"] == "attempt_01/build.log"
