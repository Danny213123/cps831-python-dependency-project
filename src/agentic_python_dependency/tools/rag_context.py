from __future__ import annotations

import json
from typing import Any

from agentic_python_dependency.state import ResolutionState


def _json_compact(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _json_minified(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _compact_strings(values: Any, *, max_items: int) -> list[str]:
    if not isinstance(values, list):
        return []
    compact: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            compact.append(text)
        if len(compact) >= max_items:
            break
    return compact


def _compact_repo_evidence(repo_evidence: Any, *, file_limit: int, hint_limit: int) -> dict[str, Any]:
    if not isinstance(repo_evidence, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("mode", "case_id", "initial_eval", "validation_command", "source_url"):
        value = repo_evidence.get(key)
        if value:
            summary[key] = str(value)
    declared_packages = _compact_strings(repo_evidence.get("declared_packages", []), max_items=hint_limit)
    if declared_packages:
        summary["declared_packages"] = declared_packages
    hint_lines = _compact_strings(repo_evidence.get("hint_lines", []), max_items=hint_limit)
    if hint_lines:
        summary["hint_lines"] = hint_lines
    if hint_limit > 4:
        for key in ("dockerfile_summary", "source_summary"):
            value = str(repo_evidence.get(key, "")).strip()
            if value:
                summary[key] = value[:220]
    files = repo_evidence.get("files", [])
    if isinstance(files, list):
        compact_files: list[dict[str, str]] = []
        for file_payload in files[:file_limit]:
            if not isinstance(file_payload, dict):
                continue
            path = str(file_payload.get("path", "")).strip()
            file_summary = str(file_payload.get("summary", "")).strip()
            if path or file_summary:
                compact_files.append({"path": path, "summary": file_summary[:220]})
        if compact_files:
            summary["files"] = compact_files
    return summary


def _compact_alias_resolution(alias_resolution: Any, *, max_items: int) -> dict[str, Any]:
    if not isinstance(alias_resolution, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("resolved_aliases", "rejected_aliases"):
        values = alias_resolution.get(key, [])
        if not isinstance(values, list):
            continue
        compact_values: list[dict[str, str]] = []
        for item in values[:max_items]:
            if not isinstance(item, dict):
                continue
            compact_item = {
                subkey: str(item.get(subkey, "")).strip()
                for subkey in ("import_name", "pypi_package", "reason")
                if str(item.get(subkey, "")).strip()
            }
            if compact_item:
                compact_values.append(compact_item)
        if compact_values:
            summary[key] = compact_values
    return summary


def _compact_version_summaries(
    summaries: Any,
    *,
    package_limit: int,
    version_limit: int,
    requires_dist_version_limit: int,
    requires_dist_entry_limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(summaries, list):
        return []
    compact_summaries: list[dict[str, Any]] = []
    for item in summaries[:package_limit]:
        if not isinstance(item, dict):
            continue
        versions = _compact_strings(item.get("versions", []), max_items=version_limit)
        summary: dict[str, Any] = {
            "package": str(item.get("package", "")).strip(),
            "versions": versions,
        }
        policy_notes = _compact_strings(item.get("policy_notes", []), max_items=4)
        if policy_notes:
            summary["policy_notes"] = policy_notes
        requires_python_payload = item.get("requires_python", {})
        if isinstance(requires_python_payload, dict):
            requires_python = {
                version: str(requires_python_payload.get(version, "")).strip()
                for version in versions[:requires_dist_version_limit]
                if str(requires_python_payload.get(version, "")).strip()
            }
            if requires_python:
                summary["requires_python"] = requires_python
        requires_dist_payload = item.get("requires_dist", {})
        if isinstance(requires_dist_payload, dict):
            requires_dist = {}
            for version in versions[:requires_dist_version_limit]:
                entries = _compact_strings(
                    requires_dist_payload.get(version, []),
                    max_items=requires_dist_entry_limit,
                )
                if entries:
                    requires_dist[version] = entries
            if requires_dist:
                summary["requires_dist"] = requires_dist
        platform_notes_payload = item.get("platform_notes", {})
        if isinstance(platform_notes_payload, dict):
            platform_notes = {}
            for version in versions[:requires_dist_version_limit]:
                entries = _compact_strings(
                    platform_notes_payload.get(version, []),
                    max_items=requires_dist_entry_limit,
                )
                if entries:
                    platform_notes[version] = entries
            if platform_notes:
                summary["platform_notes"] = platform_notes
        if summary["package"]:
            compact_summaries.append(summary)
    return compact_summaries


def _compact_conflict_notes(notes: Any, *, max_items: int) -> list[dict[str, str]]:
    if not isinstance(notes, list):
        return []
    compact_notes: list[dict[str, str]] = []
    for item in notes[:max_items]:
        if not isinstance(item, dict):
            continue
        compact_item = {
            key: str(item.get(key, "")).strip()
            for key in ("package", "related_package", "kind", "reason", "severity")
            if str(item.get(key, "")).strip()
        }
        if compact_item:
            compact_notes.append(compact_item)
    return compact_notes


def _build_summary_payload(
    context: dict[str, Any],
    *,
    package_limit: int,
    version_limit: int,
    requires_dist_version_limit: int,
    requires_dist_entry_limit: int,
    conflict_limit: int,
    hint_limit: int,
) -> dict[str, Any]:
    pypi_evidence = context.get("pypi_evidence", {})
    alias_resolution = {}
    if isinstance(pypi_evidence, dict):
        alias_resolution = pypi_evidence.get("alias_resolution", {})
    payload: dict[str, Any] = {
        "target_python": str(context.get("target_python", "")).strip(),
        "research_bundle": str(context.get("research_bundle", "")).strip(),
        "research_features": _compact_strings(context.get("research_features", []), max_items=hint_limit),
        "imports": _compact_strings(context.get("imports", []), max_items=package_limit * 2),
        "dynamic_imports": _compact_strings(context.get("dynamic_imports", []), max_items=hint_limit),
        "inferred_packages": _compact_strings(context.get("inferred_packages", []), max_items=package_limit),
        "unresolved_packages": _compact_strings(context.get("unresolved_packages", []), max_items=package_limit),
        "unsupported_imports": _compact_strings(context.get("unsupported_imports", []), max_items=hint_limit),
        "ambiguous_imports": _compact_strings(context.get("ambiguous_imports", []), max_items=hint_limit),
        "alias_resolution": _compact_alias_resolution(alias_resolution, max_items=hint_limit),
        "package_versions": _compact_version_summaries(
            context.get("version_summaries", []),
            package_limit=package_limit,
            version_limit=version_limit,
            requires_dist_version_limit=requires_dist_version_limit,
            requires_dist_entry_limit=requires_dist_entry_limit,
        ),
        "version_conflict_notes": _compact_conflict_notes(
            context.get("version_conflict_notes", []),
            max_items=conflict_limit,
        ),
        "python_constraint_intersection": _compact_strings(
            context.get("python_constraint_intersection", []),
            max_items=hint_limit,
        ),
        "platform_compatibility_notes": _compact_strings(
            context.get("platform_compatibility_notes", []),
            max_items=hint_limit * 2,
        ),
        "repo_evidence_summary": _compact_repo_evidence(
            context.get("repo_evidence", {}),
            file_limit=2,
            hint_limit=hint_limit,
        ),
    }
    repair_memory = context.get("repair_memory_summary", {})
    if isinstance(repair_memory, dict):
        compact_memory = {
            key: _compact_strings(repair_memory.get(key, []), max_items=hint_limit)
            for key in ("recent_strategies", "blocked_strategy_families", "orthogonal_hints")
        }
        compact_memory = {key: value for key, value in compact_memory.items() if value}
        if compact_memory:
            payload["repair_memory_summary"] = compact_memory
    return {key: value for key, value in payload.items() if value not in ("", [], {}, None)}


def build_research_rag_context(
    state: ResolutionState,
    *,
    repo_evidence: dict[str, Any],
    pypi_evidence: dict[str, Any],
) -> dict[str, Any]:
    version_summaries = []
    for option in state.get("version_options", []):
        version_summaries.append(
            {
                "package": option.package,
                "versions": option.versions[:10],
                "policy_notes": option.policy_notes,
                "requires_python": {
                    version: option.requires_python.get(version, "")
                    for version in option.versions[:8]
                    if option.requires_python.get(version, "")
                },
                "platform_notes": {
                    version: option.platform_notes.get(version, [])
                    for version in option.versions[:5]
                    if option.platform_notes.get(version, [])
                },
                "requires_dist": {version: option.requires_dist.get(version, [])[:10] for version in option.versions[:5]},
            }
        )
    return {
        "target_python": state.get("target_python", ""),
        "research_bundle": state.get("research_bundle", "baseline"),
        "research_features": list(state.get("research_features", ())),
        "imports": state.get("extracted_imports", []),
        "dynamic_imports": state.get("dynamic_import_candidates", []),
        "inferred_packages": state.get("inferred_packages", []),
        "unresolved_packages": state.get("unresolved_packages", []),
        "unsupported_imports": state.get("unsupported_imports", []),
        "ambiguous_imports": state.get("ambiguous_imports", []),
        "repo_alias_candidates": state.get("repo_alias_candidates", {}),
        "platform_compatibility_notes": state.get("platform_compatibility_notes", []),
        "python_constraint_intersection": state.get("python_constraint_intersection", []),
        "version_conflict_notes": [
            {
                "package": note.package,
                "related_package": note.related_package,
                "kind": note.kind,
                "reason": note.reason,
                "severity": note.severity,
            }
            for note in state.get("version_conflict_notes", [])
        ],
        "repair_memory_summary": (
            {
                "recent_strategies": state["repair_memory_summary"].recent_strategies,
                "blocked_strategy_families": state["repair_memory_summary"].blocked_strategy_families,
                "orthogonal_hints": state["repair_memory_summary"].orthogonal_hints,
            }
            if state.get("repair_memory_summary") is not None
            else {}
        ),
        "repo_evidence": repo_evidence,
        "pypi_evidence": pypi_evidence,
        "version_summaries": version_summaries,
    }


def summarize_rag_context(context: dict[str, Any], *, limit: int = 6000) -> str:
    budgets = (
        (8, 8, 4, 4, 10, 8),
        (6, 6, 3, 3, 8, 6),
        (5, 4, 2, 2, 6, 5),
        (4, 3, 1, 2, 4, 4),
        (3, 2, 1, 1, 3, 2),
        (2, 1, 1, 1, 2, 1),
    )
    for package_limit, version_limit, requires_dist_version_limit, requires_dist_entry_limit, conflict_limit, hint_limit in budgets:
        payload = _build_summary_payload(
            context,
            package_limit=package_limit,
            version_limit=version_limit,
            requires_dist_version_limit=requires_dist_version_limit,
            requires_dist_entry_limit=requires_dist_entry_limit,
            conflict_limit=conflict_limit,
            hint_limit=hint_limit,
        )
        rendered = _json_compact(payload)
        if len(rendered) <= limit:
            return rendered
        minified = _json_minified(payload)
        if len(minified) <= limit:
            return minified
    minimal = _json_compact(
        _build_summary_payload(
            context,
            package_limit=2,
            version_limit=1,
            requires_dist_version_limit=1,
            requires_dist_entry_limit=1,
            conflict_limit=1,
            hint_limit=1,
        )
    )
    if len(minimal) <= limit:
        return minimal
    minimal_payload = _build_summary_payload(
        context,
        package_limit=2,
        version_limit=1,
        requires_dist_version_limit=1,
        requires_dist_entry_limit=1,
        conflict_limit=1,
        hint_limit=1,
    )
    minified = _json_minified(minimal_payload)
    return minified if len(minified) <= limit else minified[: limit - 3] + "..."
