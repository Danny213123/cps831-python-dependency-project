import json
import time
import warnings
from pathlib import Path
import pytest

from agentic_python_dependency.cli import (
    BenchmarkProgress,
    PersistentBenchmarkObserver,
    build_runtime_comparison_row,
    build_parser,
    collect_doctor_report,
    format_elapsed,
    format_progress_bar,
    gist_match_detailed_row_from_runtime_row,
    gist_match_row_from_runtime_row,
    load_official_csv_lookup,
    load_run_state,
    probe_docker_daemon,
    redirect_runtime_warnings,
    resolve_trace_path,
    main,
    run_benchmark,
    run_case_batch,
)
from agentic_python_dependency.config import Settings
from agentic_python_dependency.router import OllamaInvocationStats
from agentic_python_dependency.state import BenchmarkCase


def test_benchmark_run_parser_accepts_jobs_flag() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--preset",
            "thorough",
            "--resolver",
            "readpye",
            "--model-profile",
            "gpt-oss-20b",
            "--fresh-run",
            "--benchmark-source",
            "competition-run",
            "--competition-csv",
            "/tmp/official-results.csv",
            "--competition-filter-file",
            "/tmp/competition-case-ids.txt",
            "benchmark",
            "run",
            "--subset",
            "smoke30",
            "--jobs",
            "3",
        ]
    )

    assert args.command == "benchmark"
    assert args.benchmark_command == "run"
    assert args.jobs == 3
    assert args.preset == "thorough"
    assert args.resolver == "readpye"
    assert args.model_profile == "gpt-oss-20b"
    assert args.fresh_run is True
    assert args.benchmark_source == "competition-run"
    assert args.competition_csv == ["/tmp/official-results.csv"]
    assert args.competition_filter_file == "/tmp/competition-case-ids.txt"


def test_web_parser_accepts_host_and_port() -> None:
    parser = build_parser()

    args = parser.parse_args(["web", "--host", "0.0.0.0", "--port", "9000"])

    assert args.command == "web"
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_benchmark_run_parser_accepts_new_moe_model_profiles() -> None:
    parser = build_parser()

    gemma_args = parser.parse_args(["--model-profile", "gemma-moe-lite", "smoke"])
    qwen_args = parser.parse_args(["--model-profile", "qwen35-moe-lite", "smoke"])
    mistral_args = parser.parse_args(["--model-profile", "mistral-nemo-12b", "smoke"])

    assert gemma_args.model_profile == "gemma-moe-lite"
    assert qwen_args.model_profile == "qwen35-moe-lite"
    assert mistral_args.model_profile == "mistral-nemo-12b"


def test_benchmark_run_parser_accepts_runtime_controls() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--no-moe",
            "--no-rag",
            "--no-langchain",
            "--extraction-model",
            "extract:model",
            "--runner-model",
            "runner:model",
            "--version-model",
            "version:model",
            "--repair-model",
            "repair:model",
            "--adjudication-model",
            "adj:model",
            "smoke",
        ]
    )

    assert args.moe is False
    assert args.rag is False
    assert args.langchain is False
    assert args.extraction_model == "extract:model"
    assert args.runner_model == "runner:model"
    assert args.version_model == "version:model"
    assert args.repair_model == "repair:model"
    assert args.adjudication_model == "adj:model"


def test_benchmark_segment_parser_accepts_subset_and_jobs() -> None:
    parser = build_parser()

    args = parser.parse_args(["benchmark", "segment", "--subset", "smoke30", "--jobs", "2"])

    assert args.command == "benchmark"
    assert args.benchmark_command == "segment"
    assert args.subset == "smoke30"
    assert args.jobs == 2


def test_benchmark_save_competition_filter_parser_accepts_ref() -> None:
    parser = build_parser()

    args = parser.parse_args(["benchmark", "save-competition-filter", "--ref", "abc123"])

    assert args.command == "benchmark"
    assert args.benchmark_command == "save-competition-filter"
    assert args.ref == "abc123"


def test_benchmark_full_parser_accepts_jobs() -> None:
    parser = build_parser()

    args = parser.parse_args(["benchmark", "full", "--jobs", "4"])

    assert args.command == "benchmark"
    assert args.benchmark_command == "full"
    assert args.jobs == 4


def test_top_level_smoke_parser_accepts_jobs() -> None:
    parser = build_parser()

    args = parser.parse_args(["smoke", "--jobs", "2"])

    assert args.command == "smoke"
    assert args.jobs == 2


def test_top_level_full_parser_accepts_jobs() -> None:
    parser = build_parser()

    args = parser.parse_args(["full", "--jobs", "3"])

    assert args.command == "full"
    assert args.jobs == 3


def test_top_level_solve_parser_accepts_path() -> None:
    parser = build_parser()

    args = parser.parse_args(["solve", "--path", "/tmp/example"])

    assert args.command == "solve"
    assert args.path == "/tmp/example"


def test_top_level_doctor_parser_accepts_ref() -> None:
    parser = build_parser()

    args = parser.parse_args(["doctor", "--ref", "abc123"])

    assert args.command == "doctor"
    assert args.ref == "abc123"


def test_top_level_ui_parser_is_available() -> None:
    parser = build_parser()

    args = parser.parse_args(["ui"])

    assert args.command == "ui"


def test_parser_program_name_is_apdr() -> None:
    parser = build_parser()

    assert parser.prog == "apdr"


def test_load_official_csv_lookup_reads_case_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "official.csv"
    csv_path.write_text(
        "name,result,duration,passed,python_modules,file\n"
        "00a4835bf36513ca58a3,ImportError,12.3,0,c4d,output_data_2.7.yml\n",
        encoding="utf-8",
    )
    settings = Settings.from_env(project_root=tmp_path, competition_result_csvs_override=[str(csv_path)])

    lookup = load_official_csv_lookup(settings)

    assert "00a4835bf36513ca58a3" in lookup
    row = lookup["00a4835bf36513ca58a3"]
    assert row["official_result"] == "ImportError"
    assert row["official_duration"] == "12.3"
    assert row["official_passed"] == "0"
    assert row["official_python_modules"] == "c4d"
    assert row["official_file"] == "output_data_2.7.yml"
    assert "official.csv" in row["official_csv_sources"]


def test_load_official_csv_lookup_reads_pllm_style_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "hard-gists-l10-r1-10-final.csv"
    csv_path.write_text(
        "id,name,file,result,python_modules,duration,passed\n"
        "10,00a4835bf36513ca58a3,output_data_2.7.yml,ImportError,c4d,12.3,False\n",
        encoding="utf-8",
    )
    settings = Settings.from_env(project_root=tmp_path, competition_result_csvs_override=[str(csv_path)])

    lookup = load_official_csv_lookup(settings)

    assert "00a4835bf36513ca58a3" in lookup
    row = lookup["00a4835bf36513ca58a3"]
    assert row["official_result"] == "ImportError"
    assert row["official_duration"] == "12.3"
    assert row["official_passed"] == "False"
    assert row["official_python_modules"] == "c4d"
    assert row["official_file"] == "output_data_2.7.yml"
    assert "hard-gists-l10-r1-10-final.csv" in row["official_csv_sources"]


def test_build_runtime_comparison_row_includes_official_values() -> None:
    official_lookup = {
        "abc123": {
            "official_result": "ModuleNotFound",
            "official_passed": "False",
            "official_duration": "14.48",
            "official_python_modules": "requests",
            "official_file": "output.yml",
            "official_csv_sources": "pyego_results.csv",
        }
    }
    result = {
        "case_id": "abc123",
        "success": False,
        "final_error_category": "ModuleNotFoundError",
        "attempts": 1,
        "wall_clock_seconds": 7.5,
        "started_at": "2026-03-04T00:00:00+00:00",
        "finished_at": "2026-03-04T00:00:08+00:00",
    }

    row = build_runtime_comparison_row(result, official_lookup, case_number=3)

    assert row["case_number"] == 3
    assert row["case_id"] == "abc123"
    assert row["run_result"] == "failure"
    assert row["run_final_error_category"] == "ModuleNotFoundError"
    assert row["run_attempts"] == 1
    assert row["run_wall_clock_seconds"] == 7.5
    assert row["official_in_csv"] is True
    assert row["official_result"] == "ModuleNotFound"
    assert row["result_matches_csv"] == "PASS"
    assert row["official_passed"] == "False"
    assert row["official_duration"] == "14.48"
    assert row["official_csv_sources"] == "pyego_results.csv"


def test_build_runtime_comparison_row_marks_result_mismatches_as_fail() -> None:
    official_lookup = {
        "abc123": {
            "official_result": "OtherPass",
            "official_passed": "True",
            "official_duration": "14.48",
            "official_python_modules": "requests",
            "official_file": "output.yml",
            "official_csv_sources": "hard-gists-l10-r1-10-final.csv",
        }
    }
    result = {
        "case_id": "abc123",
        "success": False,
        "final_error_category": "UnknownError",
        "attempts": 1,
        "wall_clock_seconds": 7.5,
    }

    row = build_runtime_comparison_row(result, official_lookup, case_number=3)

    assert row["result_matches_csv"] == "FAIL"


def test_build_runtime_comparison_row_treats_any_failure_as_match_when_pllm_failed() -> None:
    official_lookup = {
        "abc123": {
            "official_result": "ImportError",
            "official_passed": "False",
            "official_duration": "14.48",
            "official_python_modules": "requests",
            "official_file": "output.yml",
            "official_csv_sources": "hard-gists-l10-r1-10-final.csv",
        }
    }
    result = {
        "case_id": "abc123",
        "success": False,
        "final_error_category": "ModuleNotFoundError",
        "attempts": 1,
        "wall_clock_seconds": 7.5,
    }

    row = build_runtime_comparison_row(result, official_lookup, case_number=3)

    assert row["result_matches_csv"] == "PASS"


def test_gist_match_row_from_runtime_row_compares_success_and_official_flag() -> None:
    assert gist_match_row_from_runtime_row(
        {
            "case_id": "case-success",
            "run_success": True,
            "official_passed": "10",
        }
    ) == {"gistid": "case-success", "matches": True}

    assert gist_match_row_from_runtime_row(
        {
            "case_id": "case-failure",
            "run_success": False,
            "official_passed": "False",
        }
    ) == {"gistid": "case-failure", "matches": True}

    assert gist_match_row_from_runtime_row(
        {
            "case_id": "case-mismatch",
            "run_success": False,
            "official_passed": "True",
        }
    ) == {"gistid": "case-mismatch", "matches": False}

    assert gist_match_row_from_runtime_row(
        {
            "case_id": "case-unknown",
            "run_success": True,
            "official_passed": "",
        }
    ) == {"gistid": "case-unknown", "matches": ""}


def test_gist_match_detailed_row_from_runtime_row_includes_both_match_modes() -> None:
    row = gist_match_detailed_row_from_runtime_row(
        {
            "case_id": "case-x",
            "run_success": False,
            "run_final_error_category": "ModuleNotFoundError",
            "official_passed": "False",
            "official_result": "ModuleNotFound",
        }
    )
    assert row["gistid"] == "case-x"
    assert row["matches_passed"] is True
    assert row["run_official_result"] == "modulenotfound"
    assert row["matches_official_result"] is True

    row_mismatch = gist_match_detailed_row_from_runtime_row(
        {
            "case_id": "case-y",
            "run_success": False,
            "run_final_error_category": "UnknownError",
            "official_passed": "True",
            "official_result": "OtherPass",
        }
    )
    assert row_mismatch["matches_passed"] is False
    assert row_mismatch["run_official_result"] == "otherfailure"
    assert row_mismatch["matches_official_result"] is False


def test_report_modules_parser_accepts_grouping() -> None:
    parser = build_parser()

    args = parser.parse_args(["report", "modules", "--run-id", "run123", "--grouping", "raw"])

    assert args.command == "report"
    assert args.report_command == "modules"
    assert args.grouping == "raw"


def test_report_modules_parser_accepts_paper_compatible() -> None:
    parser = build_parser()

    args = parser.parse_args(["report", "modules", "--run-id", "run123", "--paper-compatible"])

    assert args.command == "report"
    assert args.report_command == "modules"
    assert args.paper_compatible is True


def test_report_trace_parser_accepts_case_and_tail() -> None:
    parser = build_parser()

    args = parser.parse_args(["report", "trace", "--run-id", "run123", "--case-id", "case1", "--tail", "20"])

    assert args.command == "report"
    assert args.report_command == "trace"
    assert args.case_id == "case1"
    assert args.tail == 20


def test_report_timeline_parser_accepts_run_id() -> None:
    parser = build_parser()

    args = parser.parse_args(["report", "timeline", "--run-id", "run123"])

    assert args.command == "report"
    assert args.report_command == "timeline"
    assert args.run_id == "run123"


def test_parser_accepts_experimental_preset_and_prompt_profile() -> None:
    parser = build_parser()

    args = parser.parse_args(["--preset", "experimental", "--prompt-profile", "research-rag", "smoke"])

    assert args.preset == "experimental"
    assert args.prompt_profile == "research-rag"


def test_parser_accepts_research_bundle_and_feature_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--preset",
            "research",
            "--research-bundle",
            "enhanced",
            "--research-feature",
            "dynamic_imports",
            "--no-research-feature",
            "repair_memory",
            "smoke",
        ]
    )

    assert args.preset == "research"
    assert args.research_bundle == "enhanced"
    assert args.research_feature == ["dynamic_imports"]
    assert args.no_research_feature == ["repair_memory"]


def test_main_rejects_experimental_with_non_apdr_resolver(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["--preset", "experimental", "--resolver", "pyego", "doctor"])

    assert excinfo.value.code == 2


def test_main_rejects_research_with_non_apdr_resolver(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["--preset", "research", "--resolver", "readpye", "doctor"])

    assert excinfo.value.code == 2


def test_main_rejects_research_feature_flags_without_research_preset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["--preset", "optimized", "--research-bundle", "enhanced", "doctor"])

    assert excinfo.value.code == 2


def test_collect_doctor_report_marks_missing_tools_and_dataset(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path)

    monkeypatch.setattr("agentic_python_dependency.cli.shutil.which", lambda _: None)
    monkeypatch.setattr("agentic_python_dependency.cli.probe_docker_daemon", lambda _settings: (False, "docker not found on PATH"))
    monkeypatch.setattr("agentic_python_dependency.cli.urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))

    report = collect_doctor_report(settings)

    assert report["overall_status"] == "warning"
    assert report["resolver"] == "apdr"
    names = {check["name"]: check for check in report["checks"]}
    assert names["docker_cli"]["status"] == "missing"
    assert names["docker_daemon"]["status"] == "missing"
    assert names["ollama_server"]["status"] == "warning"
    assert names["gistable_dataset"]["status"] == "warning"


def test_collect_doctor_report_marks_unhealthy_docker_daemon(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path)

    monkeypatch.setattr("agentic_python_dependency.cli.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        "agentic_python_dependency.cli.probe_docker_daemon",
        lambda _settings: (False, "Error response from daemon: Docker Desktop is unable to start"),
    )
    monkeypatch.setattr("agentic_python_dependency.cli.urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))

    report = collect_doctor_report(settings)

    names = {check["name"]: check for check in report["checks"]}
    assert names["docker_cli"]["status"] == "ok"
    assert names["docker_daemon"]["status"] == "warning"
    assert "unable to start" in names["docker_daemon"]["detail"]


def test_run_case_batch_fails_when_no_cases_are_selected(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    run_id = "empty-run"

    class DummyDataset:
        def __init__(self, _settings: Settings):
            return None

        def fetch(self, _ref: str | None) -> None:
            return None

    monkeypatch.setattr("agentic_python_dependency.cli.GistableDataset", DummyDataset)
    monkeypatch.setattr("agentic_python_dependency.cli.load_official_csv_lookup", lambda _settings: {})
    monkeypatch.setattr("agentic_python_dependency.cli.probe_docker_daemon", lambda _settings: (True, "25.0.0"))

    exit_code = run_case_batch(
        settings,
        ref=None,
        case_ids=[],
        run_id=run_id,
        jobs=1,
        notify_paths=False,
    )

    assert exit_code == 2
    run_dir = settings.artifacts_dir / run_id
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["total_cases"] == 0
    assert summary_payload["preset"] == "research"
    state_payload = load_run_state(run_dir)
    assert state_payload["status"] == "empty"


def test_run_case_batch_fails_fast_on_invalid_research_full_runtime(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    settings.research_bundle = "full"
    settings.structured_prompting = False

    class DummyDataset:
        def __init__(self, _settings: Settings):
            return None

        def fetch(self, _ref: str | None) -> None:
            return None

    monkeypatch.setattr("agentic_python_dependency.cli.GistableDataset", DummyDataset)

    exit_code = run_case_batch(
        settings,
        ref=None,
        case_ids=["case-a"],
        run_id="invalid-runtime",
        jobs=1,
        notify_paths=False,
    )

    assert exit_code == 2
    warnings_text = (settings.artifacts_dir / "invalid-runtime" / "warnings.log").read_text(encoding="utf-8")
    assert "structured_prompting=true" in warnings_text


def test_run_case_batch_fails_fast_when_docker_preflight_fails(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")

    class DummyDataset:
        def __init__(self, _settings: Settings):
            return None

        def fetch(self, _ref: str | None) -> None:
            return None

    monkeypatch.setattr("agentic_python_dependency.cli.GistableDataset", DummyDataset)
    monkeypatch.setattr(
        "agentic_python_dependency.cli.probe_docker_daemon",
        lambda _settings: (False, "Error response from daemon: Docker Desktop is unable to start"),
    )

    exit_code = run_case_batch(
        settings,
        ref=None,
        case_ids=["case-a"],
        run_id="docker-preflight-failed",
        jobs=1,
        notify_paths=False,
    )

    assert exit_code == 2
    warnings_text = (settings.artifacts_dir / "docker-preflight-failed" / "warnings.log").read_text(encoding="utf-8")
    assert "Docker preflight failed" in warnings_text
    assert "unable to start" in warnings_text


def test_run_case_batch_passes_pllm_match_to_observer(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    case_root = tmp_path / "case-1"
    case_root.mkdir()
    snippet = case_root / "snippet.py"
    snippet.write_text("import rx\n", encoding="utf-8")
    case = BenchmarkCase(case_id="case-1", root_dir=case_root, snippet_path=snippet)

    class DummyDataset:
        def load_case(self, case_id: str, ref: str | None, case_source: str = "all-gists"):
            assert case_id == "case-1"
            assert ref is None
            assert case_source
            return case

    class DummyPromptRunner:
        def __init__(self, *args, **kwargs) -> None:
            return None

    class FakeWorkflow:
        def __init__(self, workflow_settings: Settings, prompt_runner=None, activity_callback=None) -> None:
            self.workflow_settings = workflow_settings
            self.activity_callback = activity_callback

        def initial_state_for_case(self, benchmark_case: BenchmarkCase, run_id: str | None = None) -> dict[str, object]:
            return {"case": benchmark_case, "run_id": run_id}

        def run(self, state: dict[str, object]) -> dict[str, object]:
            benchmark_case = state["case"]
            assert isinstance(benchmark_case, BenchmarkCase)
            artifact_dir = self.workflow_settings.artifacts_dir / str(state["run_id"]) / benchmark_case.case_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "case_id": benchmark_case.case_id,
                "success": False,
                "final_error_category": "ImportError",
                "attempts": 1,
                "wall_clock_seconds": 1.2,
                "target_python": "2.7.18",
                "dependencies": ["rx==1.2.4", "twisted==19.10.0"],
            }
            (artifact_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            return {"artifact_dir": str(artifact_dir), "final_result": result}

    class RecordingObserver:
        def __init__(self) -> None:
            self.results: list[dict[str, object]] = []

        def start(self, **kwargs) -> None:
            return None

        def case_started(self, case_id: str) -> None:
            return None

        def case_event(self, case_id: str, *, attempt: int = 0, kind: str, detail: str) -> None:
            return None

        def advance(self, result: dict[str, object]) -> None:
            self.results.append(dict(result))

        def stop_requested(self) -> bool:
            return False

        def finish(self, *, summary_path: Path, warnings_path: Path | None, status: str = "completed") -> None:
            return None

    def fake_summarize_run(run_dir: Path, **kwargs) -> dict[str, object]:
        payload = {"total_cases": 1, "successes": 0, "failures": 1}
        (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    observer = RecordingObserver()
    monkeypatch.setattr("agentic_python_dependency.cli.probe_docker_daemon", lambda _settings: (True, "25.0.0"))
    monkeypatch.setattr(
        "agentic_python_dependency.cli.load_official_csv_lookup",
        lambda _settings: {
            "case-1": {
                "official_result": "ImportError",
                "official_passed": "False",
                "official_duration": "6.63",
                "official_python_modules": "rx;twisted",
                "official_file": "output_data_2.7.yml",
                "official_csv_sources": "hard-gists-l10-r1-10-final.csv",
            }
        },
    )
    monkeypatch.setattr("agentic_python_dependency.cli.OllamaPromptRunner", DummyPromptRunner)
    monkeypatch.setattr("agentic_python_dependency.cli.ResolutionWorkflow", FakeWorkflow)
    monkeypatch.setattr("agentic_python_dependency.cli.summarize_run", fake_summarize_run)

    exit_code = run_case_batch(
        settings,
        ref=None,
        case_ids=["case-1"],
        run_id="match-run",
        dataset=DummyDataset(),
        jobs=1,
        observer=observer,
        notify_paths=False,
    )

    assert exit_code == 0
    assert observer.results
    assert observer.results[0]["case_id"] == "case-1"
    assert observer.results[0]["result_matches_csv"] == "PASS"


def test_run_benchmark_restores_saved_settings_before_case_selection(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    run_id = "resume-run"
    run_dir = settings.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "interrupted",
                "resolver": "apdr",
                "preset": "research",
                "prompt_profile": "research-rag",
                "benchmark_source": "competition-run",
                "research_bundle": "full",
                "research_features": ["dynamic_aliases", "repair_memory"],
                "effective_model_profile": "mistral-nemo-12b",
                "model_profile": "mistral-nemo-12b",
                "use_moe": True,
                "use_rag": True,
                "use_langchain": True,
                "extraction_model": "mistral-nemo:12b",
                "runner_model": "mistral-nemo:12b",
                "version_model": "mistral-nemo:12b",
                "repair_model": "mistral-nemo:12b",
                "adjudication_model": "mistral-nemo:12b",
                "rag_mode": "hybrid",
                "effective_rag_mode": "hybrid",
                "structured_prompting": True,
                "effective_structured_prompting": True,
                "candidate_plan_count": 3,
                "allow_candidate_fallback_before_repair": True,
                "effective_candidate_fallback_before_repair": True,
                "repair_cycle_limit": 2,
                "effective_repair_cycle_limit": 2,
                "repo_evidence_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class DummyDataset:
        def __init__(self, dataset_settings: Settings):
            captured["init_model_profile"] = dataset_settings.model_profile
            captured["init_preset"] = dataset_settings.preset
            captured["init_benchmark_source"] = dataset_settings.benchmark_case_source

        def fetch(self, _ref: str | None) -> None:
            return None

        def valid_case_ids(self, ref: str | None, case_source: str = "all-gists") -> list[str]:
            captured["valid_case_ids_ref"] = ref
            captured["valid_case_ids_source"] = case_source
            return ["case-a"]

    def fake_run_case_batch(
        batch_settings: Settings,
        ref: str | None,
        case_ids: list[str],
        batch_run_id: str | None,
        **kwargs,
    ) -> int:
        captured["batch_ref"] = ref
        captured["batch_case_ids"] = list(case_ids)
        captured["batch_run_id"] = batch_run_id
        captured["batch_model_profile"] = batch_settings.model_profile
        captured["batch_preset"] = batch_settings.preset
        captured["batch_benchmark_source"] = batch_settings.benchmark_case_source
        return 0

    monkeypatch.setattr("agentic_python_dependency.cli.GistableDataset", DummyDataset)
    monkeypatch.setattr("agentic_python_dependency.cli.run_case_batch", fake_run_case_batch)

    exit_code = run_benchmark(
        settings,
        ref=None,
        subset=None,
        full=False,
        run_id=run_id,
        jobs=1,
        fresh_run=False,
    )

    assert exit_code == 0
    assert captured["init_model_profile"] == "mistral-nemo-12b"
    assert captured["init_preset"] == "research"
    assert captured["init_benchmark_source"] == "competition-run"
    assert captured["valid_case_ids_source"] == "competition-run"
    assert captured["batch_case_ids"] == ["case-a"]
    assert captured["batch_model_profile"] == "mistral-nemo-12b"
    assert captured["batch_preset"] == "research"
    assert captured["batch_benchmark_source"] == "competition-run"


def test_run_case_batch_restores_saved_runtime_config_before_resume_validation(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    run_id = "resume-run"
    run_dir = settings.artifacts_dir / run_id
    case_dir = run_dir / "case-a"
    case_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run-state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "interrupted",
                "resolver": "apdr",
                "preset": "research",
                "prompt_profile": "research-rag",
                "benchmark_source": "competition-run",
                "research_bundle": "full",
                "research_features": ["dynamic_aliases", "repair_memory"],
                "elapsed_seconds": 42.0,
                "effective_model_profile": "mistral-nemo-12b",
                "model_profile": "mistral-nemo-12b",
                "use_moe": True,
                "use_rag": True,
                "use_langchain": True,
                "extraction_model": "mistral-nemo:12b",
                "runner_model": "mistral-nemo:12b",
                "version_model": "mistral-nemo:12b",
                "repair_model": "mistral-nemo:12b",
                "adjudication_model": "mistral-nemo:12b",
                "rag_mode": "hybrid",
                "effective_rag_mode": "hybrid",
                "structured_prompting": True,
                "effective_structured_prompting": True,
                "candidate_plan_count": 3,
                "allow_candidate_fallback_before_repair": True,
                "effective_candidate_fallback_before_repair": True,
                "repair_cycle_limit": 2,
                "effective_repair_cycle_limit": 2,
                "repo_evidence_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "result.json").write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "success": True,
                "final_error_category": "Success",
                "attempts": 1,
                "wall_clock_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    class DummyDataset:
        def fetch(self, _ref: str | None) -> None:
            return None

        def load_case(self, case_id: str, ref: str | None, case_source: str = "all-gists"):
            raise AssertionError(f"load_case should not run for completed case {case_id}")

    def fake_summarize_run(run_dir: Path, **kwargs) -> dict[str, object]:
        payload = {"total_cases": 1, "successes": 1, "failures": 0}
        (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    class RecordingObserver:
        def __init__(self) -> None:
            self.start_kwargs: dict[str, object] = {}

        def start(self, **kwargs) -> None:
            self.start_kwargs = dict(kwargs)

        def case_started(self, case_id: str) -> None:
            return None

        def case_event(self, case_id: str, *, attempt: int = 0, kind: str, detail: str) -> None:
            return None

        def advance(self, result: dict[str, object]) -> None:
            return None

        def stop_requested(self) -> bool:
            return False

        def finish(self, *, summary_path: Path, warnings_path: Path | None, status: str = "completed") -> None:
            return None

    observer = RecordingObserver()

    monkeypatch.setattr(
        "agentic_python_dependency.cli.load_official_csv_lookup",
        lambda _settings: {
            "case-a": {
                "official_result": "OtherPass",
                "official_passed": "True",
                "official_duration": "1.0",
                "official_python_modules": "requests",
                "official_file": "output_data_3.12.yml",
                "official_csv_sources": "hard-gists-l10-r1-10-final.csv",
            }
        },
    )
    monkeypatch.setattr("agentic_python_dependency.cli.probe_docker_daemon", lambda _settings: (True, "25.0.0"))
    monkeypatch.setattr("agentic_python_dependency.cli.summarize_run", fake_summarize_run)

    exit_code = run_case_batch(
        settings,
        ref=None,
        case_ids=["case-a"],
        run_id=run_id,
        dataset=DummyDataset(),
        jobs=1,
        observer=observer,
        notify_paths=False,
    )

    assert exit_code == 0
    assert settings.preset == "research"
    assert settings.prompt_profile == "research-rag"
    assert settings.model_profile == "mistral-nemo-12b"
    assert settings.benchmark_case_source == "competition-run"
    assert observer.start_kwargs["completed_results"] == [
        {
            "case_id": "case-a",
            "success": True,
            "final_error_category": "Success",
            "attempts": 1,
            "wall_clock_seconds": 1.0,
            "result_matches_csv": "PASS",
        }
    ]
    warnings_path = run_dir / "warnings.log"
    if warnings_path.exists():
        assert "Refusing to resume run with mismatched runtime config" not in warnings_path.read_text(encoding="utf-8")


def test_format_progress_bar_renders_partial_progress() -> None:
    assert format_progress_bar(3, 4, width=8) == "[######--]"


def test_format_elapsed_formats_hms() -> None:
    assert format_elapsed(3661.9) == "01:01:01"


def test_benchmark_progress_line_contains_run_id_and_counts(monkeypatch) -> None:
    progress = BenchmarkProgress("run123", total=10, completed=4)
    monkeypatch.setattr(progress, "started_at", progress.started_at - 65)

    line = progress._line()

    assert "Benchmark run123" in line
    assert "4/10" in line
    assert "40.0%" in line
    assert "ok 0" in line
    assert "fail 0" in line
    assert "elapsed 00:01:05" in line


def test_benchmark_progress_line_includes_ollama_tokens_per_second() -> None:
    progress = BenchmarkProgress("run123", total=10, completed=4)
    progress.ollama_stats = PersistentBenchmarkObserver(progress, Path("/tmp/run123")).ollama_stats
    progress.ollama_stats.record(
        OllamaInvocationStats(
            stage="repair",
            model="gemma3:12b",
            eval_count=48,
            eval_duration_ns=1_000_000_000,
        )
    )

    line = progress._line()

    assert "llm 48.0 tok/s" in line


def test_benchmark_progress_refresh_thread_starts_and_stops(monkeypatch) -> None:
    progress = BenchmarkProgress("run123", total=10, refresh_interval=0.01)
    monkeypatch.setattr(progress, "_isatty", True)

    progress.start(
        run_id="run123",
        total=10,
        completed=0,
        successes=0,
        failures=0,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe: gemma3:4b / gemma3:12b",
        jobs=1,
        target="smoke30",
        artifacts_dir=Path("/tmp/run123"),
    )
    time.sleep(0.03)
    progress.finish(summary_path=Path("/tmp/run123/summary.json"), warnings_path=None)

    assert progress._thread is not None
    assert not progress._thread.is_alive()


def test_benchmark_progress_tracks_case_results() -> None:
    progress = BenchmarkProgress("run123", total=3, completed=1)
    progress.start(
        run_id="run123",
        total=3,
        completed=1,
        successes=1,
        failures=0,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe: gemma3:4b / gemma3:12b",
        jobs=1,
        target="smoke30",
        artifacts_dir=Path("/tmp/run123"),
    )

    progress.case_started("case-2")
    progress.advance({"case_id": "case-2", "success": False, "final_error_category": "TimeoutError"})

    assert progress.completed == 2
    assert progress.successes == 1
    assert progress.failures == 1
    assert progress.last_case_id == "case-2"
    assert progress.last_status == "TimeoutError"


def test_benchmark_progress_can_request_stop() -> None:
    progress = BenchmarkProgress("run123", total=3)

    assert progress.stop_requested() is False

    progress.request_stop()

    assert progress.stop_requested() is True


def test_benchmark_progress_can_request_hard_stop() -> None:
    progress = BenchmarkProgress("run123", total=3)

    assert progress.hard_stop_requested() is False

    progress.request_hard_stop()

    assert progress.stop_requested() is True
    assert progress.hard_stop_requested() is True


def test_persistent_benchmark_observer_writes_run_state_files(tmp_path: Path) -> None:
    inner = BenchmarkProgress("run123", total=5)
    observer = PersistentBenchmarkObserver(inner, tmp_path / "run123")

    observer.start(
        run_id="run123",
        total=5,
        completed=1,
        successes=1,
        failures=0,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe",
        jobs=1,
        target="smoke30",
        artifacts_dir=tmp_path / "run123",
        runtime_config={
            "model_profile": "mistral-nemo-12b",
            "effective_model_profile": "mistral-nemo-12b",
            "rag_mode": "hybrid",
            "effective_rag_mode": "hybrid",
            "structured_prompting": True,
            "effective_structured_prompting": True,
            "repair_cycle_limit": 2,
            "effective_repair_cycle_limit": 2,
            "allow_candidate_fallback_before_repair": True,
            "effective_candidate_fallback_before_repair": True,
        },
    )
    observer.case_started("case-2")
    observer.advance({"case_id": "case-2", "success": True, "final_error_category": "Success"})
    observer.finish(summary_path=tmp_path / "run123" / "summary.json", warnings_path=None, status="paused")

    payload = load_run_state(tmp_path / "run123")
    assert payload["status"] == "paused"
    assert payload["completed"] == 2
    assert payload["successes"] == 2
    assert payload["effective_model_profile"] == "mistral-nemo-12b"
    assert payload["effective_structured_prompting"] is True
    assert (tmp_path / "run123" / "run-state.md").exists()


def test_persistent_benchmark_observer_persists_ollama_stats(tmp_path: Path) -> None:
    inner = BenchmarkProgress("run123", total=5)
    observer = PersistentBenchmarkObserver(inner, tmp_path / "run123")

    observer.start(
        run_id="run123",
        total=5,
        completed=0,
        successes=0,
        failures=0,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe",
        jobs=1,
        target="smoke30",
        artifacts_dir=tmp_path / "run123",
    )
    observer.ollama_stats.record(
        OllamaInvocationStats(
            stage="version",
            model="gemma3:12b",
            prompt_eval_count=12,
            prompt_eval_duration_ns=60_000_000,
            eval_count=30,
            eval_duration_ns=500_000_000,
        )
    )
    observer.case_started("case-1")

    payload = load_run_state(tmp_path / "run123")
    markdown = (tmp_path / "run123" / "run-state.md").read_text(encoding="utf-8")

    assert payload["ollama_stats"]["eval_tokens"] == 30
    assert payload["ollama_stats"]["last_model"] == "gemma3:12b"
    assert "## Ollama" in markdown
    assert "60.0 tok/s" in markdown


def test_persistent_benchmark_observer_persists_hard_stop_state(tmp_path: Path) -> None:
    inner = BenchmarkProgress("run123", total=5)
    observer = PersistentBenchmarkObserver(inner, tmp_path / "run123")

    observer.start(
        run_id="run123",
        total=5,
        completed=0,
        successes=0,
        failures=0,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe",
        jobs=1,
        target="smoke30",
        artifacts_dir=tmp_path / "run123",
    )
    observer.request_hard_stop(reason="Hard quit requested.")

    payload = load_run_state(tmp_path / "run123")

    assert payload["status"] == "interrupted"
    assert payload["hard_stop_requested"] is True
    assert payload["last_error"] == "Hard quit requested."


def test_persistent_benchmark_observer_persists_case_activity(tmp_path: Path) -> None:
    inner = BenchmarkProgress("run123", total=5)
    observer = PersistentBenchmarkObserver(inner, tmp_path / "run123")

    observer.start(
        run_id="run123",
        total=5,
        completed=0,
        successes=0,
        failures=0,
        resolver="apdr",
        preset="research",
        prompt_profile="research-rag",
        model_summary="mistral-nemo-12b",
        jobs=1,
        target="full",
        artifacts_dir=tmp_path / "run123",
    )
    observer.case_started("case-1")
    observer.case_event("case-1", attempt=1, kind="docker_build_start", detail="Starting docker build.")

    payload = load_run_state(tmp_path / "run123")
    markdown = (tmp_path / "run123" / "run-state.md").read_text(encoding="utf-8")
    activity_log = (tmp_path / "run123" / "activity.log").read_text(encoding="utf-8")

    assert payload["current_case_activity"][0]["case_id"] == "case-1"
    assert payload["current_case_activity"][0]["kind"] == "docker_build_start"
    assert payload["recent_case_activity"][0]["detail"] == "Starting docker build."
    assert "## Current Activity" in markdown
    assert "## Recent Activity" in markdown
    assert "docker_build_start" in activity_log


def test_persistent_benchmark_observer_restores_elapsed_seconds(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class RecordingObserver(BenchmarkProgress):
        def start(self, **kwargs):  # type: ignore[override]
            captured.update(kwargs)

        def case_started(self, case_id: str) -> None:
            return None

        def advance(self, result: dict[str, object]) -> None:
            return None

        def finish(self, *, summary_path: Path, warnings_path: Path | None, status: str = "completed") -> None:
            return None

        def stop_requested(self) -> bool:
            return False

    observer = PersistentBenchmarkObserver(
        RecordingObserver("run123", total=5),
        tmp_path / "run123",
        {"elapsed_seconds": 42.0, "started_at": "2026-03-03T00:00:00+00:00"},
    )

    observer.start(
        run_id="run123",
        total=5,
        completed=2,
        successes=1,
        failures=1,
        resolver="apdr",
        preset="optimized",
        prompt_profile="optimized",
        model_summary="gemma-moe",
        jobs=1,
        target="smoke30",
        artifacts_dir=tmp_path / "run123",
    )

    assert captured["elapsed_seconds"] == 42.0


def test_redirect_runtime_warnings_writes_warning_to_file(tmp_path: Path) -> None:
    warning_path = tmp_path / "warnings.log"

    with redirect_runtime_warnings(warning_path):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(r"invalid escape sequence '\s'", SyntaxWarning)

    contents = warning_path.read_text(encoding="utf-8")
    assert "SyntaxWarning" in contents
    assert "invalid escape sequence" in contents


def test_resolve_trace_path_supports_run_and_case_scope(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)

    assert resolve_trace_path(settings, "run1") == settings.artifacts_dir / "run1" / "llm-trace.log"
    assert resolve_trace_path(settings, "run1", "case1") == settings.artifacts_dir / "run1" / "case1" / "llm-trace.log"


def test_run_case_batch_hard_interrupt_cancels_executor_and_exits(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(project_root=tmp_path, preset_override="research")
    shutdown_calls: list[tuple[bool, bool]] = []

    class DummyDataset:
        def fetch(self, _ref: str | None) -> None:
            return None

        def load_case(self, case_id: str, ref: str | None, case_source: str = "all-gists"):
            raise AssertionError(f"load_case should not run for {case_id}")

    class FakeExecutor:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def submit(self, fn, case_id: str) -> object:
            return object()

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr("agentic_python_dependency.cli.load_official_csv_lookup", lambda _settings: {})
    monkeypatch.setattr("agentic_python_dependency.cli.probe_docker_daemon", lambda _settings: (True, "25.0.0"))
    monkeypatch.setattr("agentic_python_dependency.cli.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        "agentic_python_dependency.cli.as_completed",
        lambda futures: (_ for _ in ()).throw(KeyboardInterrupt("hard quit")),
    )
    monkeypatch.setattr(
        "agentic_python_dependency.cli._force_exit_now",
        lambda code=130: (_ for _ in ()).throw(SystemExit(code)),
    )

    with pytest.raises(SystemExit) as excinfo:
        run_case_batch(
            settings,
            ref=None,
            case_ids=["case-1"],
            run_id="hard-exit-run",
            dataset=DummyDataset(),
            jobs=2,
            notify_paths=False,
        )

    assert excinfo.value.code == 130
    assert shutdown_calls == [(False, True)]
    payload = load_run_state(settings.artifacts_dir / "hard-exit-run")
    assert payload["status"] == "interrupted"
    assert payload["hard_stop_requested"] is True
