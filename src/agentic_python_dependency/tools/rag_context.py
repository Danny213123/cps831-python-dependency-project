from __future__ import annotations

import json
from typing import Any

from agentic_python_dependency.state import PackageVersionOptions, ResolutionState


def _json_compact(payload: dict[str, Any], *, limit: int) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)[:limit]


def build_research_rag_context(
    state: ResolutionState,
    *,
    repo_evidence: dict[str, Any],
    pypi_evidence: dict[str, Any],
) -> dict[str, Any]:
    version_summaries = []
    for option in state.get("version_options", []):
        option = option
        version_summaries.append(
            {
                "package": option.package,
                "versions": option.versions[:10],
                "policy_notes": option.policy_notes,
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
        "repo_alias_candidates": state.get("repo_alias_candidates", {}),
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
    return _json_compact(context, limit=limit)
