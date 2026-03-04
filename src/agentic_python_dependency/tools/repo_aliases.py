from __future__ import annotations

from collections import defaultdict
from typing import Any

from agentic_python_dependency.tools.package_metadata import PackageMetadataStore
from agentic_python_dependency.tools.pypi_store import PyPIMetadataStore


def build_repo_alias_candidates(
    repo_evidence: dict[str, Any],
    *,
    target_python: str,
    pypi_store: PyPIMetadataStore,
    package_metadata_store: PackageMetadataStore,
    preset: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    declared_packages = [str(item).strip() for item in repo_evidence.get("declared_packages", []) if str(item).strip()]
    alias_map: dict[str, list[str]] = defaultdict(list)
    top_level_module_map: dict[str, list[str]] = {}
    for package in declared_packages:
        try:
            options = pypi_store.get_version_options(package, target_python, preset=preset)
        except FileNotFoundError:
            continue
        if not options.versions:
            continue
        version = options.versions[0]
        release_files = pypi_store.release_files(package, version)
        metadata = package_metadata_store.parse_release_metadata(package, version, release_files=release_files)
        modules = [str(item).strip() for item in metadata.get("top_level_modules", []) if str(item).strip()]
        if not modules:
            continue
        top_level_module_map[package] = modules
        for module in modules:
            normalized = module.replace("-", "_").lower()
            if package not in alias_map[normalized]:
                alias_map[normalized].append(package)
    return dict(alias_map), top_level_module_map
