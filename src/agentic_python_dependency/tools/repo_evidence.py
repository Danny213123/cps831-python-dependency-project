from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentic_python_dependency.state import ResolutionState


EVIDENCE_FILE_CANDIDATES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "setup.cfg",
    "Dockerfile",
    "README.md",
    "README.rst",
    "pytest.ini",
    "tox.ini",
)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _trim_text(text: str, *, limit: int = 1200) -> str:
    normalized = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    return normalized[:limit].strip()


def _extract_hint_lines(text: str) -> list[str]:
    hints: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if any(token in lowered for token in ("pip install", "poetry add", "requirements", "pytest", "python", "from python:")):
            hints.append(stripped)
        elif re.search(r"[A-Za-z0-9_.-]+(==|>=|<=|~=)[A-Za-z0-9_.-]+", stripped):
            hints.append(stripped)
    return hints[:20]


def build_repo_evidence(state: ResolutionState) -> dict[str, Any]:
    if state["mode"] == "gistable":
        case = state["benchmark_case"]
        dockerfile_text = _read_text(case.dockerfile_path) or ""
        source_text = _read_text(case.snippet_path) or ""
        return {
            "mode": "gistable",
            "case_id": case.case_id,
            "source_url": case.source_url,
            "initial_eval": case.initial_eval,
            "dockerfile_summary": _trim_text(dockerfile_text, limit=900),
            "source_summary": _trim_text(source_text, limit=1200),
            "hint_lines": _extract_hint_lines(dockerfile_text),
        }

    target = state["project_target"]
    files: list[dict[str, str]] = []
    hint_lines: list[str] = []
    for name in EVIDENCE_FILE_CANDIDATES:
        candidate = target.root_dir / name
        if not candidate.exists():
            continue
        text = _read_text(candidate)
        if text is None:
            continue
        files.append({"path": name, "summary": _trim_text(text)})
        hint_lines.extend(_extract_hint_lines(text))

    return {
        "mode": "project",
        "root_dir": str(target.root_dir),
        "validation_command": target.validation_command,
        "files": files,
        "hint_lines": hint_lines[:30],
    }
