from __future__ import annotations

import json
from typing import Any

from agentic_python_dependency.state import CandidateDependency, CandidatePlan


class StructuredOutputError(ValueError):
    pass


def _load_json(raw_output: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise StructuredOutputError("Structured output must be a JSON object")
    return payload


def parse_experimental_package_payload(raw_output: str) -> list[dict[str, Any]]:
    payload = _load_json(raw_output)
    packages = payload.get("packages", [])
    if not isinstance(packages, list):
        raise StructuredOutputError("'packages' must be a list")
    parsed: list[dict[str, Any]] = []
    for item in packages:
        if not isinstance(item, dict):
            raise StructuredOutputError("Package entries must be objects")
        package = str(item.get("package", "")).strip()
        if not package:
            raise StructuredOutputError("Package entry missing 'package'")
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            raise StructuredOutputError("'evidence' must be a list")
        parsed.append(
            {
                "package": package,
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "source": str(item.get("source", "llm") or "llm"),
                "evidence": [str(entry) for entry in evidence if str(entry).strip()],
            }
        )
    return parsed


def parse_cross_validation_payload(raw_output: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = _load_json(raw_output)
    accepted = payload.get("accepted_packages", [])
    rejected = payload.get("rejected_packages", [])
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise StructuredOutputError("'accepted_packages' and 'rejected_packages' must be lists")
    parsed_accepted = []
    for item in accepted:
        if not isinstance(item, dict):
            raise StructuredOutputError("Accepted package entries must be objects")
        package = str(item.get("package", "")).strip()
        if not package:
            raise StructuredOutputError("Accepted package missing 'package'")
        sources = item.get("sources", [])
        if not isinstance(sources, list):
            raise StructuredOutputError("'sources' must be a list")
        parsed_accepted.append(
            {
                "package": package,
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "sources": [str(source) for source in sources if str(source).strip()],
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    parsed_rejected = []
    for item in rejected:
        if not isinstance(item, dict):
            raise StructuredOutputError("Rejected package entries must be objects")
        package = str(item.get("package", "")).strip()
        if not package:
            raise StructuredOutputError("Rejected package missing 'package'")
        parsed_rejected.append({"package": package, "reason": str(item.get("reason", "")).strip()})
    return parsed_accepted, parsed_rejected


def parse_candidate_plan_payload(
    raw_output: str,
    *,
    allowed_packages: set[str],
    allowed_versions: dict[str, set[str]],
) -> list[CandidatePlan]:
    payload = _load_json(raw_output)
    plans = payload.get("plans", [])
    if not isinstance(plans, list):
        raise StructuredOutputError("'plans' must be a list")
    parsed: list[CandidatePlan] = []
    seen_ranks: set[int] = set()
    for item in plans:
        if not isinstance(item, dict):
            raise StructuredOutputError("Plan entries must be objects")
        rank = item.get("rank")
        if not isinstance(rank, int) or rank < 1:
            raise StructuredOutputError("Plan rank must be a positive integer")
        if rank in seen_ranks:
            raise StructuredOutputError("Plan ranks must be unique")
        seen_ranks.add(rank)
        reason = str(item.get("reason", "")).strip()
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise StructuredOutputError("'dependencies' must be a list")
        parsed_dependencies: list[CandidateDependency] = []
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                raise StructuredOutputError("Dependency entries must be objects")
            name = str(dependency.get("name", "")).strip()
            version = str(dependency.get("version", "")).strip()
            if not name or not version:
                raise StructuredOutputError("Dependency entries require name and version")
            normalized_name = name.replace("-", "_").lower()
            if normalized_name not in allowed_packages:
                raise StructuredOutputError(f"Dependency '{name}' is not in the allowed package set")
            if version not in allowed_versions.get(normalized_name, set()):
                raise StructuredOutputError(f"Version '{version}' is not allowed for package '{name}'")
            parsed_dependencies.append(CandidateDependency(name=name, version=version))
        parsed.append(CandidatePlan(rank=rank, reason=reason, dependencies=parsed_dependencies))
    return sorted(parsed, key=lambda item: item.rank)


def parse_repair_plan_payload(
    raw_output: str,
    *,
    allowed_packages: set[str],
    allowed_versions: dict[str, set[str]],
) -> tuple[bool, list[CandidatePlan]]:
    payload = _load_json(raw_output)
    repair_applicable = bool(payload.get("repair_applicable", False))
    plans = parse_candidate_plan_payload(
        json.dumps({"plans": payload.get("plans", [])}),
        allowed_packages=allowed_packages,
        allowed_versions=allowed_versions,
    )
    return repair_applicable, plans


def parse_version_negotiation_payload(
    raw_output: str,
    *,
    allowed_packages: set[str],
    allowed_versions: dict[str, set[str]],
) -> list[CandidatePlan]:
    payload = _load_json(raw_output)
    bundles = payload.get("selected_bundles", [])
    return parse_candidate_plan_payload(
        json.dumps({"plans": bundles}),
        allowed_packages=allowed_packages,
        allowed_versions=allowed_versions,
    )
