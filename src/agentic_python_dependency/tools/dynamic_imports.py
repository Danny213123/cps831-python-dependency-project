from __future__ import annotations

import ast
import configparser
import tomllib
from pathlib import Path
from typing import Any


def _string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def extract_dynamic_imports_from_code(source: str) -> dict[str, list[str]]:
    resolved: set[str] = set()
    ambiguous: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"resolved": [], "ambiguous": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module" and node.args:
                value = _string_value(node.args[0])
                (resolved if value else ambiguous).add((value or "importlib.import_module").split(".", 1)[0])
            elif isinstance(node.func, ast.Name) and node.func.id == "__import__" and node.args:
                value = _string_value(node.args[0])
                (resolved if value else ambiguous).add((value or "__import__").split(".", 1)[0])
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "iter_entry_points":
                ambiguous.add("pkg_resources_entry_points")
    return {"resolved": sorted(resolved), "ambiguous": sorted(ambiguous)}


def extract_entry_point_modules(project_root: Path) -> list[str]:
    discovered: set[str] = set()
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        try:
            payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            scripts = payload.get("project", {}).get("scripts", {})
            for value in scripts.values():
                module = str(value).split(":", 1)[0].strip()
                if module:
                    discovered.add(module.split(".", 1)[0])
        except (OSError, tomllib.TOMLDecodeError):
            pass
    setup_cfg = project_root / "setup.cfg"
    if setup_cfg.exists():
        parser = configparser.ConfigParser()
        try:
            parser.read(setup_cfg, encoding="utf-8")
            if parser.has_section("options.entry_points"):
                for _, value in parser.items("options.entry_points"):
                    for line in value.splitlines():
                        module = line.split("=", 1)[-1].strip().split(":", 1)[0]
                        if module:
                            discovered.add(module.split(".", 1)[0])
        except (OSError, configparser.Error):
            pass
    return sorted(discovered)


def collect_dynamic_import_candidates(
    source_files: dict[str, str],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    resolved: set[str] = set()
    ambiguous: set[str] = set()
    per_file: dict[str, dict[str, list[str]]] = {}
    for name, source in source_files.items():
        result = extract_dynamic_imports_from_code(source)
        per_file[name] = result
        resolved.update(result["resolved"])
        ambiguous.update(result["ambiguous"])
    if project_root is not None:
        resolved.update(extract_entry_point_modules(project_root))
    return {
        "resolved": sorted(resolved),
        "ambiguous": sorted(ambiguous),
        "per_file": per_file,
    }
