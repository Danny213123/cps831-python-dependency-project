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


def parse_alias_resolution_payload(raw_output: str) -> list[dict[str, str]]:
    payload = _load_json(raw_output)
    aliases = payload.get("aliases", [])
    if not isinstance(aliases, list):
        raise StructuredOutputError("'aliases' must be a list")
    parsed: list[dict[str, str]] = []
    for item in aliases:
        if not isinstance(item, dict):
            raise StructuredOutputError("Alias entries must be objects")
        import_name = str(item.get("import_name", "")).strip()
        pypi_package = str(item.get("pypi_package", "")).strip()
        if not import_name or not pypi_package:
            raise StructuredOutputError("Alias entries require import_name and pypi_package")
        parsed.append({"import_name": import_name, "pypi_package": pypi_package})
    return parsed


def parse_candidate_plan_payload(
    raw_output: str,
    *,
    allowed_packages: set[str],
    allowed_versions: dict[str, set[str]],
    required_packages: set[str] | None = None,
    allowed_runtime_profiles: set[str] | None = None,
) -> list[CandidatePlan]:
    payload = _load_json(raw_output)
    plans = payload.get("plans", [])
    if not isinstance(plans, list):
        raise StructuredOutputError("'plans' must be a list")
    parsed: list[CandidatePlan] = []
    seen_ranks: set[int] = set()
    normalized_required = {package.replace("-", "_").lower() for package in (required_packages or set())}
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
        runtime_profile = str(item.get("runtime_profile", "") or "").strip()
        if runtime_profile and allowed_runtime_profiles is not None and runtime_profile not in allowed_runtime_profiles:
            raise StructuredOutputError(f"Runtime profile '{runtime_profile}' is not in the allowed runtime profile set")
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise StructuredOutputError("'dependencies' must be a list")
        parsed_dependencies: list[CandidateDependency] = []
        dependency_names: set[str] = set()
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
            if normalized_name in dependency_names:
                raise StructuredOutputError(
                    f"Dependency '{name}' duplicates another package after normalization"
                )
            parsed_dependencies.append(CandidateDependency(name=name, version=version))
            dependency_names.add(normalized_name)
        if normalized_required and not dependency_names:
            raise StructuredOutputError("Plan dependencies may not be empty when packages are required")
        missing_packages = sorted(normalized_required - dependency_names)
        if missing_packages:
            raise StructuredOutputError(
                "Plan is missing required packages: " + ", ".join(missing_packages)
            )
        parsed.append(
            CandidatePlan(
                rank=rank,
                reason=reason,
                dependencies=parsed_dependencies,
                runtime_profile=runtime_profile,
            )
        )
    return sorted(parsed, key=lambda item: item.rank)


def parse_repair_plan_payload(
    raw_output: str,
    *,
    allowed_packages: set[str],
    allowed_versions: dict[str, set[str]],
    required_packages: set[str] | None = None,
    allowed_runtime_profiles: set[str] | None = None,
    previous_plan: list[CandidateDependency] | None = None,
) -> tuple[bool, list[CandidatePlan]]:
    payload = _load_json(raw_output)
    repair_applicable = bool(payload.get("repair_applicable", False))
    plans = payload.get("plans", [])
    if not isinstance(plans, list):
        raise StructuredOutputError("'plans' must be a list")

    previous_dependencies = {
        dependency.name.replace("-", "_").lower(): CandidateDependency(name=dependency.name, version=dependency.version)
        for dependency in (previous_plan or [])
        if dependency.name and dependency.version
    }
    normalized_required = {package.replace("-", "_").lower() for package in (required_packages or set())}
    parsed: list[CandidatePlan] = []
    seen_ranks: set[int] = set()

    for item in plans:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank")
        if not isinstance(rank, int) or rank < 1 or rank in seen_ranks:
            continue
        seen_ranks.add(rank)
        reason = str(item.get("reason", "")).strip()
        runtime_profile = str(item.get("runtime_profile", "") or "").strip()
        if runtime_profile and allowed_runtime_profiles is not None and runtime_profile not in allowed_runtime_profiles:
            continue
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list):
            continue

        merged_dependencies: dict[str, CandidateDependency] = dict(previous_dependencies)
        valid_item = True
        explicit_dependency_count = 0
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                valid_item = False
                break
            name = str(dependency.get("name", "")).strip()
            version = str(dependency.get("version", "")).strip()
            if not name or not version:
                valid_item = False
                break
            normalized_name = name.replace("-", "_").lower()
            if normalized_name not in allowed_packages:
                valid_item = False
                break
            if version not in allowed_versions.get(normalized_name, set()):
                valid_item = False
                break
            merged_dependencies[normalized_name] = CandidateDependency(name=name, version=version)
            explicit_dependency_count += 1
        if not valid_item:
            continue
        if explicit_dependency_count == 0 and not runtime_profile:
            continue
        if normalized_required and any(package not in merged_dependencies for package in normalized_required):
            continue
        parsed.append(
            CandidatePlan(
                rank=rank,
                reason=reason,
                dependencies=sorted(merged_dependencies.values(), key=lambda dependency: dependency.name.lower()),
                runtime_profile=runtime_profile,
            )
        )
    return repair_applicable, sorted(parsed, key=lambda item: item.rank)


def parse_version_negotiation_payload(
    raw_output: str,
    *,
    allowed_packages: set[str],
    allowed_versions: dict[str, set[str]],
    required_packages: set[str] | None = None,
) -> list[CandidatePlan]:
    payload = _load_json(raw_output)
    bundles = payload.get("selected_bundles", [])
    return parse_candidate_plan_payload(
        json.dumps({"plans": bundles}),
        allowed_packages=allowed_packages,
        allowed_versions=allowed_versions,
        required_packages=required_packages,
    )
