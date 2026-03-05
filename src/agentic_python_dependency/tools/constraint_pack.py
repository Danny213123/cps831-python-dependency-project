from __future__ import annotations

from collections import defaultdict
from itertools import islice
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from agentic_python_dependency.state import CandidateDependency, ConflictNote, ConstraintPack, PackageVersionOptions


def _normalize_package(value: str) -> str:
    return value.strip().replace("-", "_").lower()


def _parse_requires_dist(entries: list[str]) -> list[Requirement]:
    parsed: list[Requirement] = []
    for entry in entries:
        try:
            parsed.append(Requirement(entry))
        except InvalidRequirement:
            continue
    return parsed


def _candidate_versions(option: PackageVersionOptions, top_k: int = 5) -> list[str]:
    return list(islice(option.versions, top_k))


def build_constraint_pack(
    options: list[PackageVersionOptions],
    *,
    target_python: str,
    top_k: int = 5,
) -> ConstraintPack:
    target_version = Version(target_python)
    direct_packages = {_normalize_package(option.package): option for option in options}
    candidate_versions = {option.package: _candidate_versions(option, top_k=top_k) for option in options}
    requires_python = {
        option.package: {version: option.requires_python.get(version, "") for version in candidate_versions[option.package]}
        for option in options
    }
    requires_dist = {
        option.package: {version: option.requires_dist.get(version, []) for version in candidate_versions[option.package]}
        for option in options
    }
    conflict_notes: list[ConflictNote] = []
    matrix: dict[str, list[str]] = defaultdict(list)
    python_intersection: list[str] = []
    python_intersection_valid = True

    for option in options:
        package_versions = candidate_versions[option.package]
        allowed_for_target = [
            version
            for version in package_versions
            if not option.requires_python.get(version)
            or target_version in SpecifierSet(option.requires_python.get(version, ""))
        ]
        python_intersection.append(f"{option.package}:{','.join(allowed_for_target)}")
        if not allowed_for_target:
            python_intersection_valid = False
            conflict_notes.append(
                ConflictNote(
                    package=option.package,
                    related_package="python",
                    kind="requires_python",
                    reason=f"No candidate versions for {option.package} support Python {target_python}",
                    severity="error",
                )
            )
            matrix[option.package].append("python")

    for option in options:
        normalized_package = _normalize_package(option.package)
        for version in candidate_versions[option.package]:
            for requirement in _parse_requires_dist(option.requires_dist.get(version, [])):
                requirement_name = _normalize_package(requirement.name)
                if requirement_name not in direct_packages:
                    continue
                candidate_target = direct_packages[requirement_name]
                compatible_target_versions = [
                    candidate_version
                    for candidate_version in candidate_versions[candidate_target.package]
                    if not requirement.specifier or Version(candidate_version) in requirement.specifier
                ]
                if compatible_target_versions:
                    continue
                note = ConflictNote(
                    package=option.package,
                    related_package=candidate_target.package,
                    kind="requires_dist",
                    reason=(
                        f"{option.package} {version} requires {requirement.name}{requirement.specifier}, "
                        f"but {candidate_target.package} candidates are incompatible"
                    ),
                    severity="error",
                )
                conflict_notes.append(note)
                if candidate_target.package not in matrix[option.package]:
                    matrix[option.package].append(candidate_target.package)
                if option.package not in matrix[candidate_target.package]:
                    matrix[candidate_target.package].append(option.package)

    return ConstraintPack(
        target_python=target_python,
        candidate_versions=candidate_versions,
        requires_python=requires_python,
        requires_dist=requires_dist,
        conflict_notes=conflict_notes,
        package_conflict_matrix=dict(matrix),
        python_intersection=python_intersection,
        python_intersection_valid=python_intersection_valid,
        conflict_precheck_failed=not python_intersection_valid,
    )


def generate_candidate_bundles(
    constraint_pack: ConstraintPack,
    *,
    max_bundles: int = 50,
    beam_width: int = 12,
) -> list[list[CandidateDependency]]:
    packages = list(constraint_pack.candidate_versions)
    beams: list[list[CandidateDependency]] = [[]]
    for package in packages:
        next_beams: list[list[CandidateDependency]] = []
        versions = constraint_pack.candidate_versions.get(package, [])
        for beam in beams:
            for version in versions:
                next_beams.append([*beam, CandidateDependency(name=package, version=version)])
        next_beams.sort(key=lambda candidate: (len(candidate), tuple(dep.version for dep in candidate)), reverse=True)
        beams = next_beams[:beam_width]
        if not beams:
            break
    return beams[:max_bundles]


def constraint_pack_to_dict(pack: ConstraintPack) -> dict[str, Any]:
    return {
        "target_python": pack.target_python,
        "candidate_versions": pack.candidate_versions,
        "requires_python": pack.requires_python,
        "requires_dist": pack.requires_dist,
        "conflict_notes": [
            {
                "package": note.package,
                "related_package": note.related_package,
                "kind": note.kind,
                "reason": note.reason,
                "severity": note.severity,
            }
            for note in pack.conflict_notes
        ],
        "package_conflict_matrix": pack.package_conflict_matrix,
        "python_intersection": pack.python_intersection,
        "python_intersection_valid": pack.python_intersection_valid,
        "conflict_precheck_failed": pack.conflict_precheck_failed,
    }
