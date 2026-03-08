from __future__ import annotations

import csv
import json
import mimetypes
import re
import signal
import socket
import tomllib
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from agentic_python_dependency.config import Settings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _web_root() -> Path:
    return _repo_root() / "web"


def _network_runs_root(settings: Settings) -> Path:
    return settings.artifacts_dir / "_network"


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned.strip("._") or "unknown"


def _local_source_label() -> str:
    return socket.gethostname() or "local"


def _make_run_key(source_type: str, run_id: str, source_id: str = "") -> str:
    if source_type == "remote":
        return f"remote:{source_id}:{run_id}"
    return f"local:{run_id}"


def _parse_run_key(run_key: str) -> tuple[str, str, str]:
    raw = str(run_key or "").strip()
    if raw.startswith("remote:"):
        _, remainder = raw.split("remote:", 1)
        source_id, _, run_id = remainder.partition(":")
        return "remote", source_id, run_id
    if raw.startswith("local:"):
        return "local", "", raw.split("local:", 1)[1]
    return "local", "", raw


def _resolve_run_location(settings: Settings, run_key: str) -> dict[str, Any] | None:
    source_type, source_id, run_id = _parse_run_key(run_key)
    if source_type == "remote":
        run_dir = _network_runs_root(settings) / _safe_component(source_id) / _safe_component(run_id)
        if not run_dir.exists():
            return None
        state = _read_json(run_dir / "run-state.json")
        return {
            "runKey": _make_run_key("remote", run_id, source_id),
            "runId": run_id,
            "runDir": run_dir,
            "sourceType": "remote",
            "sourceId": source_id,
            "sourceLabel": str(state.get("source_label", "") or source_id or "remote"),
        }
    local_dir = settings.artifacts_dir / run_id
    if not local_dir.exists():
        return None
    return {
        "runKey": _make_run_key("local", run_id),
        "runId": run_id,
        "runDir": local_dir,
        "sourceType": "local",
        "sourceId": "local",
        "sourceLabel": _local_source_label(),
    }


def _resolve_apdr_version() -> str:
    try:
        payload = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return "unknown"
    return str(payload.get("project", {}).get("version", "") or "unknown")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_runtime_row_lookup(run_dir: Path) -> dict[str, dict[str, str]]:
    path = run_dir / "run-vs-csv.csv"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return {}
    return {
        str(row.get("case_id", "") or ""): {str(key): str(value or "") for key, value in row.items()}
        for row in rows
        if str(row.get("case_id", "") or "")
    }


def _format_dependency_preview(dependencies: list[str]) -> str:
    preview = ", ".join(item for item in dependencies[:3] if item)
    if len(dependencies) > 3:
        preview = f"{preview}, +{len(dependencies) - 3} more"
    return preview or "-"


def _pllm_match(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized == "PASS":
        return "MATCH"
    if normalized == "FAIL":
        return "MISS"
    return "-"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _latest_timestamp(*values: object) -> str:
    timestamps = [str(value or "").strip() for value in values if str(value or "").strip()]
    return max(timestamps) if timestamps else ""


def _hardware_payload(state: dict[str, Any]) -> dict[str, Any]:
    hardware = state.get("hardware_info", {})
    if not isinstance(hardware, dict):
        return {}
    return {
        "host": str(hardware.get("host", "") or ""),
        "os": str(hardware.get("os", "") or ""),
        "platform": str(hardware.get("platform", "") or ""),
        "machine": str(hardware.get("machine", "") or ""),
        "cpu": str(hardware.get("cpu", "") or ""),
        "gpu": str(hardware.get("gpu", "") or ""),
        "memory": str(hardware.get("memory", "") or ""),
        "memoryBytes": _safe_int(hardware.get("memory_bytes", 0)),
        "logicalCores": _safe_int(hardware.get("logical_cores", 0)),
    }


def _case_summary_payload(result: dict[str, Any], runtime_row: dict[str, str] | None = None) -> dict[str, Any]:
    runtime_row = runtime_row or {}
    dependencies = result.get("dependencies", [])
    deps = [str(item) for item in dependencies] if isinstance(dependencies, list) else []
    success = bool(result.get("success", False))
    return {
        "caseId": str(result.get("case_id", "") or ""),
        "success": success,
        "status": "PASS" if success else "FAIL",
        "result": "Success" if success else str(result.get("final_error_category", "failure") or "failure"),
        "attempts": _safe_int(result.get("attempts", 0)),
        "targetPython": str(result.get("target_python", "") or ""),
        "runtimeProfile": str(result.get("runtime_profile", "") or ""),
        "seconds": _safe_float(result.get("wall_clock_seconds", 0.0)),
        "pllmMatch": _pllm_match(runtime_row.get("result_matches_csv", "")),
        "dependencies": deps,
        "dependencyPreview": _format_dependency_preview(deps),
        "dockerBuildSeconds": _safe_float(result.get("docker_build_seconds_total", 0.0)),
        "dockerRunSeconds": _safe_float(result.get("docker_run_seconds_total", 0.0)),
        "llmSeconds": _safe_float(result.get("llm_wall_clock_seconds", 0.0)),
        "imageCacheHits": _safe_int(result.get("image_cache_hits", 0)),
        "buildSkips": _safe_int(result.get("build_skips", 0)),
        "classifierOrigin": str(result.get("classifier_origin", "") or ""),
        "rootCauseBucket": str(result.get("root_cause_bucket", "") or ""),
        "candidatePlanStrategy": str(result.get("candidate_plan_strategy", "") or ""),
        "pythonFallbackUsed": bool(result.get("python_fallback_used", False)),
        "startedAt": str(result.get("started_at", "") or ""),
        "finishedAt": str(result.get("finished_at", "") or ""),
        "sortTimestamp": _latest_timestamp(result.get("finished_at"), result.get("started_at")),
    }


def _parse_activity_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[") or "]" not in line:
            continue
        try:
            timestamp, remainder = line.split("] ", 1)
        except ValueError:
            continue
        pieces = remainder.split(" ", 3)
        if len(pieces) < 3:
            continue
        case_id = pieces[0].split("=", 1)[-1]
        attempt = _safe_int(pieces[1].split("=", 1)[-1], 0)
        kind = pieces[2].split("=", 1)[-1]
        detail = pieces[3] if len(pieces) > 3 else ""
        events.append(
            {
                "timestamp": timestamp.lstrip("["),
                "caseId": case_id,
                "attempt": attempt,
                "kind": kind,
                "detail": detail,
            }
        )
    return events


def _file_entry(run_id: str, case_id: str, relative_path: str, *, label: str | None = None) -> dict[str, str]:
    return {
        "label": label or relative_path,
        "path": relative_path,
        "url": (
            f"/api/runs/{quote(run_id, safe='')}/cases/{quote(case_id, safe='')}/"
            f"artifacts/{quote(relative_path, safe='/')}"
        ),
    }


def ingest_network_run_state(settings: Settings, source_id: str, run_id: str, payload: dict[str, Any]) -> Path:
    run_dir = _network_runs_root(settings) / _safe_component(source_id) / _safe_component(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    stored = dict(payload)
    stored["source_type"] = "remote"
    stored["source_id"] = source_id
    stored["source_label"] = str(payload.get("source_label", "") or source_id)
    if not stored.get("run_id"):
        stored["run_id"] = run_id
    (run_dir / "run-state.json").write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return run_dir


def ingest_network_case_bundle(
    settings: Settings,
    source_id: str,
    run_id: str,
    case_id: str,
    payload: dict[str, Any],
) -> Path:
    run_dir = _network_runs_root(settings) / _safe_component(source_id) / _safe_component(run_id)
    case_dir = run_dir / _safe_component(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    result = payload.get("result", {})
    if isinstance(result, dict) and result:
        (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for item in payload.get("files", []):
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get("path", "") or "").strip()
        if not relative_path:
            continue
        target = (case_dir / relative_path).resolve()
        try:
            target.relative_to(case_dir.resolve())
        except ValueError:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(item.get("content", "") or ""), encoding="utf-8")
    return case_dir


def _iter_run_locations(settings: Settings) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for run_dir in sorted(settings.artifacts_dir.glob("*")):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        locations.append(
            {
                "runKey": _make_run_key("local", run_dir.name),
                "runId": run_dir.name,
                "runDir": run_dir,
                "sourceType": "local",
                "sourceId": "local",
                "sourceLabel": _local_source_label(),
            }
        )
    network_root = _network_runs_root(settings)
    if network_root.exists():
        for source_dir in sorted(network_root.glob("*")):
            if not source_dir.is_dir():
                continue
            for run_dir in sorted(source_dir.glob("*")):
                if not run_dir.is_dir():
                    continue
                state = _read_json(run_dir / "run-state.json")
                source_id = source_dir.name
                locations.append(
                    {
                        "runKey": _make_run_key("remote", run_dir.name, source_id),
                        "runId": run_dir.name,
                        "runDir": run_dir,
                        "sourceType": "remote",
                        "sourceId": source_id,
                        "sourceLabel": str(state.get("source_label", "") or source_id),
                    }
                )
    return locations


def list_runs(settings: Settings) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    app_version = _resolve_apdr_version()
    for location in _iter_run_locations(settings):
        run_dir = Path(location["runDir"])
        state = _read_json(run_dir / "run-state.json")
        summary = _read_json(run_dir / "summary.json")
        if not state and not summary:
            continue
        total = _safe_int(state.get("total", summary.get("total_cases", 0)))
        completed = (
            _safe_int(state.get("completed", 0))
            if "completed" in state
            else _safe_int(summary.get("successes", 0)) + _safe_int(summary.get("failures", 0))
        )
        successes = _safe_int(state.get("successes", summary.get("successes", 0)))
        failures = _safe_int(state.get("failures", summary.get("failures", 0)))
        runs.append(
            {
                "runKey": str(location["runKey"]),
                "runId": str(location["runId"]),
                "status": str(state.get("status", "completed" if summary else "unknown") or "unknown"),
                "resolver": str(state.get("resolver", summary.get("resolver", "apdr")) or "apdr"),
                "preset": str(state.get("preset", summary.get("preset", "optimized")) or "optimized"),
                "promptProfile": str(state.get("prompt_profile", summary.get("prompt_profile", "optimized")) or "optimized"),
                "researchBundle": str(state.get("research_bundle", summary.get("research_bundle", "baseline")) or "baseline"),
                "researchFeatures": state.get("research_features", summary.get("research_features", [])) or [],
                "benchmarkSource": str(state.get("benchmark_source", summary.get("benchmark_source", "all-gists")) or "all-gists"),
                "target": str(state.get("target", "benchmark") or "benchmark"),
                "total": total,
                "completed": completed,
                "successes": successes,
                "failures": failures,
                "successRate": ((successes / completed) * 100.0) if completed else 0.0,
                "elapsedSeconds": _safe_float(state.get("elapsed_seconds", summary.get("total_wall_clock_time", 0.0))),
                "dockerBuildSeconds": _safe_float(
                    state.get("docker_build_seconds_total", summary.get("total_docker_build_time", 0.0))
                ),
                "dockerRunSeconds": _safe_float(
                    state.get("docker_run_seconds_total", summary.get("total_docker_run_time", 0.0))
                ),
                "llmSeconds": _safe_float(state.get("llm_wall_clock_seconds_total", summary.get("total_llm_time", 0.0))),
                "imageCacheHits": _safe_int(state.get("image_cache_hits", summary.get("image_cache_hits", 0))),
                "buildSkips": _safe_int(state.get("build_skips", summary.get("build_skips", 0))),
                "modelSummary": str(state.get("model_summary", "") or ""),
                "appVersion": app_version,
                "sourceType": str(location["sourceType"]),
                "sourceId": str(location["sourceId"]),
                "sourceLabel": str(location["sourceLabel"]),
                "hardware": _hardware_payload(state),
                "lastCaseId": str(state.get("last_case_id", "") or ""),
                "lastStatus": str(state.get("last_status", "") or ""),
                "lastUpdatedAt": _latest_timestamp(state.get("last_updated_at"), state.get("started_at")),
                "artifactsDir": str(run_dir),
            }
        )
    runs.sort(key=lambda item: item.get("lastUpdatedAt", ""), reverse=True)
    return runs


def get_run_detail(settings: Settings, run_id: str) -> dict[str, Any] | None:
    location = _resolve_run_location(settings, run_id)
    if location is None:
        return None
    run_dir = Path(location["runDir"])
    state = _read_json(run_dir / "run-state.json")
    summary = _read_json(run_dir / "summary.json")
    runtime_rows = _read_runtime_row_lookup(run_dir)
    case_summaries: list[dict[str, Any]] = []
    for result_path in sorted(run_dir.glob("*/result.json")):
        result = _read_json(result_path)
        if not result:
            continue
        case_id = str(result.get("case_id", "") or result_path.parent.name)
        case_summaries.append(_case_summary_payload(result, runtime_rows.get(case_id, {})))
    case_summaries.sort(key=lambda item: (item.get("sortTimestamp", ""), item.get("caseId", "")), reverse=True)
    completed = _safe_int(state.get("completed", len(case_summaries)))
    successes = _safe_int(state.get("successes", summary.get("successes", 0)))
    failures = _safe_int(state.get("failures", summary.get("failures", 0)))
    return {
        "run": {
            "runKey": str(location["runKey"]),
            "runId": str(location["runId"]),
            "status": str(state.get("status", "completed" if summary else "unknown") or "unknown"),
            "resolver": str(state.get("resolver", summary.get("resolver", "apdr")) or "apdr"),
            "preset": str(state.get("preset", summary.get("preset", "optimized")) or "optimized"),
            "promptProfile": str(state.get("prompt_profile", summary.get("prompt_profile", "optimized")) or "optimized"),
            "researchBundle": str(state.get("research_bundle", summary.get("research_bundle", "baseline")) or "baseline"),
            "researchFeatures": state.get("research_features", summary.get("research_features", [])) or [],
            "benchmarkSource": str(state.get("benchmark_source", summary.get("benchmark_source", "all-gists")) or "all-gists"),
            "target": str(state.get("target", "benchmark") or "benchmark"),
            "total": _safe_int(state.get("total", summary.get("total_cases", len(case_summaries)))),
            "completed": completed,
            "successes": successes,
            "failures": failures,
            "successRate": ((successes / completed) * 100.0) if completed else 0.0,
            "elapsedSeconds": _safe_float(state.get("elapsed_seconds", summary.get("total_wall_clock_time", 0.0))),
            "dockerBuildSeconds": _safe_float(state.get("docker_build_seconds_total", summary.get("total_docker_build_time", 0.0))),
            "dockerRunSeconds": _safe_float(state.get("docker_run_seconds_total", summary.get("total_docker_run_time", 0.0))),
            "llmSeconds": _safe_float(state.get("llm_wall_clock_seconds_total", summary.get("total_llm_time", 0.0))),
            "imageCacheHits": _safe_int(state.get("image_cache_hits", summary.get("image_cache_hits", 0))),
            "buildSkips": _safe_int(state.get("build_skips", summary.get("build_skips", 0))),
            "modelSummary": str(state.get("model_summary", "") or ""),
            "appVersion": _resolve_apdr_version(),
            "sourceType": str(location["sourceType"]),
            "sourceId": str(location["sourceId"]),
            "sourceLabel": str(location["sourceLabel"]),
            "hardware": _hardware_payload(state),
            "runtimeConfig": {
                "effectiveModelProfile": state.get("effective_model_profile", state.get("model_profile", "")),
                "effectiveRagMode": state.get("effective_rag_mode", state.get("rag_mode", "")),
                "effectiveStructuredPrompting": bool(
                    state.get("effective_structured_prompting", state.get("structured_prompting", False))
                ),
                "effectiveRepairCycleLimit": _safe_int(
                    state.get("effective_repair_cycle_limit", state.get("repair_cycle_limit", 0))
                ),
                "effectiveCandidateFallbackBeforeRepair": bool(
                    state.get(
                        "effective_candidate_fallback_before_repair",
                        state.get("allow_candidate_fallback_before_repair", False),
                    )
                ),
            },
            "jobs": _safe_int(state.get("jobs", 1)),
            "startedAt": str(state.get("started_at", "") or ""),
            "lastUpdatedAt": str(state.get("last_updated_at", "") or ""),
            "lastCaseId": str(state.get("last_case_id", "") or ""),
            "lastStatus": str(state.get("last_status", "") or ""),
            "artifactsDir": str(run_dir),
            "summaryPath": str(state.get("summary_path", run_dir / "summary.json") or (run_dir / "summary.json")),
            "warningsPath": str(state.get("warnings_path", "") or ""),
            "ollamaStats": state.get("ollama_stats", {}),
        },
        "activeCases": list(state.get("current_cases", [])) if isinstance(state.get("current_cases"), list) else [],
        "currentCaseActivity": list(state.get("current_case_activity", []))
        if isinstance(state.get("current_case_activity"), list)
        else [],
        "recentCaseActivity": list(state.get("recent_case_activity", []))
        if isinstance(state.get("recent_case_activity"), list)
        else [],
        "cases": case_summaries,
    }


def get_case_detail(settings: Settings, run_id: str, case_id: str) -> dict[str, Any] | None:
    location = _resolve_run_location(settings, run_id)
    if location is None:
        return None
    run_dir = Path(location["runDir"])
    case_dir = run_dir / (_safe_component(case_id) if location["sourceType"] == "remote" else case_id)
    if not case_dir.exists():
        return None
    result = _read_json(case_dir / "result.json")
    if not result:
        return None
    runtime_row = _read_runtime_row_lookup(run_dir).get(case_id, {})
    activity_events = _parse_activity_log(case_dir / "activity.log")
    attempts: list[dict[str, Any]] = []
    for attempt in result.get("attempt_records", []):
        if not isinstance(attempt, dict):
            continue
        attempt_no = _safe_int(attempt.get("attempt_number", 0), 0)
        attempt_dir = Path(str(attempt.get("artifact_dir", "") or ""))
        if not attempt_dir.is_absolute():
            attempt_dir = case_dir / f"attempt_{attempt_no:02d}"
        files: list[dict[str, str]] = []
        for name in ("build.log", "run.log"):
            candidate = attempt_dir / name
            if candidate.exists():
                relative = candidate.relative_to(case_dir).as_posix()
                files.append(_file_entry(run_id, case_id, relative, label=name))
        attempts.append(
            {
                "attemptNumber": attempt_no,
                "dependencies": list(attempt.get("dependencies", [])) if isinstance(attempt.get("dependencies"), list) else [],
                "buildSucceeded": bool(attempt.get("build_succeeded", False)),
                "runSucceeded": bool(attempt.get("run_succeeded", False)),
                "exitCode": attempt.get("exit_code"),
                "errorCategory": str(attempt.get("error_category", "") or ""),
                "errorDetails": str(attempt.get("error_details", "") or ""),
                "llmFailureAnalysis": str(attempt.get("llm_failure_analysis", "") or ""),
                "llmFailureAnalysisModel": str(attempt.get("llm_failure_analysis_model", "") or ""),
                "validationCommand": str(attempt.get("validation_command", "") or ""),
                "wallClockSeconds": _safe_float(attempt.get("wall_clock_seconds", 0.0)),
                "buildWallClockSeconds": _safe_float(attempt.get("build_wall_clock_seconds", 0.0)),
                "runWallClockSeconds": _safe_float(attempt.get("run_wall_clock_seconds", 0.0)),
                "buildSkipped": bool(attempt.get("build_skipped", False)),
                "imageCacheHit": bool(attempt.get("image_cache_hit", False)),
                "environmentCacheKey": str(attempt.get("environment_cache_key", "") or ""),
                "startedAt": str(attempt.get("started_at", "") or ""),
                "finishedAt": str(attempt.get("finished_at", "") or ""),
                "activity": [event for event in activity_events if _safe_int(event.get("attempt", 0), 0) == attempt_no],
                "files": files,
            }
        )
    top_files: list[dict[str, str]] = []
    for name in (
        "build.log",
        "run.log",
        "activity.log",
        "model_outputs.json",
        "candidate-plans.json",
        "strategy-history.json",
        "structured-outputs.json",
        "repair-memory-summary.json",
        "constraint-pack.json",
        "result.json",
    ):
        candidate = case_dir / name
        if candidate.exists():
            top_files.append(_file_entry(run_id, case_id, name))
    payload = {
        "case": _case_summary_payload(result, runtime_row),
        "source": {
            "runKey": str(location["runKey"]),
            "runId": str(location["runId"]),
            "sourceType": str(location["sourceType"]),
            "sourceId": str(location["sourceId"]),
            "sourceLabel": str(location["sourceLabel"]),
        },
        "result": result,
        "attempts": attempts,
        "activity": activity_events,
        "files": top_files,
        "official": runtime_row,
    }
    return payload


def read_case_artifact(settings: Settings, run_id: str, case_id: str, relative_path: str) -> tuple[bytes, str] | None:
    location = _resolve_run_location(settings, run_id)
    if location is None:
        return None
    case_component = _safe_component(case_id) if location["sourceType"] == "remote" else case_id
    case_dir = (Path(location["runDir"]) / case_component).resolve()
    target = (case_dir / relative_path).resolve()
    try:
        target.relative_to(case_dir)
    except ValueError:
        return None
    if not target.exists() or not target.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(target))
    if target.suffix == ".json":
        mime = "application/json; charset=utf-8"
    elif not mime:
        mime = "text/plain; charset=utf-8"
    data = target.read_bytes()
    return data, mime


def _candidate_lan_ip() -> str | None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
        except OSError:
            return None
    return host if host and not host.startswith("127.") else None


def _count_loadouts(settings: Settings) -> int:
    loadouts_dir = settings.data_dir / "loadouts"
    if not loadouts_dir.exists():
        return 0
    return sum(1 for path in loadouts_dir.glob("*.json") if path.is_file())


def _dashboard_urls(host: str, port: int) -> tuple[str, str]:
    local_host = "127.0.0.1" if host == "0.0.0.0" else host
    local_url = f"http://{local_host}:{port}"
    network_url = ""
    if host == "0.0.0.0":
        lan_ip = _candidate_lan_ip()
        if lan_ip:
            network_url = f"http://{lan_ip}:{port}"
    elif host not in {"127.0.0.1", "localhost"}:
        network_url = f"http://{host}:{port}"
    return local_url, network_url


def get_home_payload(settings: Settings, *, host: str = "0.0.0.0", port: int = 8765) -> dict[str, Any]:
    app_version = _resolve_apdr_version()
    local_url, network_url = _dashboard_urls(host, port)
    runtime_summary = (
        f"MoE {'on' if settings.use_moe else 'off'} | "
        f"RAG {'on' if settings.use_rag else 'off'} | "
        f"LangChain {'on' if settings.use_langchain else 'off'}"
    )
    research_summary = f"{settings.research_bundle} ({len(settings.research_features)} feature(s))"
    fields = [
        {"label": "Version", "value": app_version},
        {"label": "Preset/Resolver", "value": f"{settings.preset} / {settings.resolver}"},
        {"label": "Benchmark source", "value": settings.benchmark_case_source},
        {"label": "Model bundle", "value": settings.model_profile},
        {"label": "Prompt profile", "value": settings.prompt_profile},
        {"label": "Models", "value": f"{settings.extraction_model} / {settings.reasoning_model}"},
        {"label": "Runtime", "value": runtime_summary},
        {"label": "Research", "value": research_summary},
        {"label": "PyEGo Python", "value": settings.pyego_python},
        {"label": "Loadouts", "value": str(_count_loadouts(settings))},
        {"label": "Fresh run / Trace LLM", "value": f"n/a / {'on' if settings.trace_llm else 'off'}"},
        {"label": "Artifacts", "value": str(settings.artifacts_dir)},
    ]
    actions = [
        {
            "label": "Run",
            "title": "Run workflows",
            "description": "Monitor smoke, full, resume, and retry flows from the run dashboard below.",
            "href": "#run-dashboard",
        },
        {
            "label": "Reports",
            "title": "Inspect outcomes",
            "description": "Use the case table and per-attempt dropdowns as the web equivalent of CLI reports.",
            "href": "#cases-panel",
        },
        {
            "label": "Configure",
            "title": "Review active settings",
            "description": "This page mirrors the current CLI command-center settings used on this host.",
            "href": "#command-center",
        },
        {
            "label": "Loadouts",
            "title": "Saved profiles",
            "description": f"{_count_loadouts(settings)} saved loadout(s) currently live under the data directory.",
            "href": "#command-center",
        },
        {
            "label": "Doctor",
            "title": "Environment checks",
            "description": "Run doctor from the CLI when you need repair guidance for Docker, Ollama, or external tools.",
            "href": "#command-center",
        },
        {
            "label": "Web",
            "title": "Network-ready viewer",
            "description": "Hosted on this device and reachable across the network when the server is bound to 0.0.0.0.",
            "href": "#run-dashboard",
        },
    ]
    return {
        "home": {
            "title": "APDR Command Center",
            "subtitle": "Run, report, and configure without memorizing commands.",
            "description": "The web dashboard mirrors the CLI command-center layout and live benchmark monitor.",
            "version": app_version,
            "preset": settings.preset,
            "resolver": settings.resolver,
            "benchmarkSource": settings.benchmark_case_source,
            "modelProfile": settings.model_profile,
            "promptProfile": settings.prompt_profile,
            "modelSummary": f"{settings.extraction_model} / {settings.reasoning_model}",
            "runtimeSummary": runtime_summary,
            "researchSummary": research_summary,
            "pyegoPython": settings.pyego_python,
            "loadouts": _count_loadouts(settings),
            "traceLlm": settings.trace_llm,
            "artifactsDir": str(settings.artifacts_dir),
            "fields": fields,
            "actions": actions,
            "server": {
                "host": host,
                "port": port,
                "localUrl": local_url,
                "networkUrl": network_url,
                "scope": "network" if host == "0.0.0.0" else "local",
            },
        }
    }


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "APDRDashboard/1.0"

    @property
    def settings(self) -> Settings:
        return self.server.settings  # type: ignore[attr-defined]

    @property
    def web_root(self) -> Path:
        return self.server.web_root  # type: ignore[attr-defined]

    @property
    def dashboard_host(self) -> str:
        return self.server.dashboard_host  # type: ignore[attr-defined]

    @property
    def dashboard_port(self) -> int:
        return self.server.dashboard_port  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, *, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        mime = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed.path)
            return
        self._handle_frontend(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._handle_api_post(parsed.path)

    def _handle_api(self, path: str) -> None:
        if path == "/api/health":
            self._send_json({"ok": True, "timestamp": datetime.utcnow().isoformat() + "Z"})
            return
        if path == "/api/home":
            self._send_json(get_home_payload(self.settings, host=self.dashboard_host, port=self.dashboard_port))
            return
        if path == "/api/runs":
            self._send_json({"runs": list_runs(self.settings)})
            return
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 3 and segments[1] == "runs":
            run_id = unquote(segments[2])
            if len(segments) == 3:
                payload = get_run_detail(self.settings, run_id)
                if payload is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_json(payload)
                return
            if len(segments) >= 5 and segments[3] == "cases":
                case_id = unquote(segments[4])
                if len(segments) == 5:
                    payload = get_case_detail(self.settings, run_id, case_id)
                    if payload is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    self._send_json(payload)
                    return
                if len(segments) >= 7 and segments[5] == "artifacts":
                    relative_path = "/".join(unquote(segment) for segment in segments[6:])
                    artifact = read_case_artifact(self.settings, run_id, case_id, relative_path)
                    if artifact is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    data, mime = artifact
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mime)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _handle_api_post(self, path: str) -> None:
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 6 and segments[1] == "ingest" and segments[2] == "runs":
            source_id = _safe_component(unquote(segments[3]))
            run_id = _safe_component(unquote(segments[4]))
            body = self._read_json_body()
            if body is None:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if len(segments) == 6 and segments[5] == "state":
                ingest_network_run_state(self.settings, source_id, run_id, body)
                self._send_json({"ok": True})
                return
            if len(segments) == 7 and segments[5] == "cases":
                case_id = unquote(segments[6])
                ingest_network_case_bundle(self.settings, source_id, run_id, case_id, body)
                self._send_json({"ok": True})
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_frontend(self, path: str) -> None:
        request_path = path or "/"
        if request_path == "/":
            request_path = "/index.html"
        target = (self.web_root / request_path.lstrip("/")).resolve()
        try:
            target.relative_to(self.web_root)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.exists() or target.is_dir():
            target = self.web_root / "index.html"
        self._send_file(target)


def serve_web_dashboard(settings: Settings, host: str = "0.0.0.0", port: int = 8765) -> int:
    web_root = _web_root()
    if not web_root.exists():
        print(f"[ERROR] web root not found: {web_root}")
        return 2

    class DashboardServer(ThreadingHTTPServer):
        def __init__(self, server_address: tuple[str, int]):
            super().__init__(server_address, DashboardRequestHandler)
            self.settings = settings
            self.web_root = web_root
            self.dashboard_host = host
            self.dashboard_port = port

    server = DashboardServer((host, port))
    server.daemon_threads = True
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"[INFO] APDR web dashboard serving {web_root}")
    print(f"[INFO] local: http://{bind_host}:{port}")
    lan_ip = _candidate_lan_ip()
    if host == "0.0.0.0" and lan_ip:
        print(f"[INFO] network: http://{lan_ip}:{port}")
    print(f"[INFO] artifacts: {settings.artifacts_dir}")
    print("[INFO] Press Ctrl+C to stop.")
    shutdown_requested = threading.Event()
    previous_handlers: dict[int, object] = {}

    def _request_shutdown(signum: int, _frame: object) -> None:
        if shutdown_requested.is_set():
            raise KeyboardInterrupt
        shutdown_requested.set()
        threading.Thread(target=server.shutdown, name="apdr-web-shutdown", daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _request_shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown_requested.set()
    finally:
        if threading.current_thread() is threading.main_thread():
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
        if shutdown_requested.is_set():
            print("\n[INFO] stopping APDR web dashboard")
        server.server_close()
    return 0
