from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Literal

from cli.pllm.core import ROOT_DIR

BenchmarkSource = Literal["all-gists", "dockerized-gists", "competition-run"]

BENCHMARK_REF_DEFAULT = "665d39a2bd82543d5196555f0801ef8fd4a3ee48"
BENCHMARK_ROOT = ROOT_DIR / "data" / "benchmarks" / "gistable" / BENCHMARK_REF_DEFAULT
COMPETITION_FILTER_FILE = ROOT_DIR / "competition" / "competition-case-ids.txt"
DEFAULT_COMPETITION_CSVS = (
    ROOT_DIR / "data" / "benchmarks" / "gistable" / "competition" / "summary-all-runs.csv",
    ROOT_DIR / "data" / "benchmarks" / "gistable" / "competition" / "pyego_results.csv",
    ROOT_DIR / "data" / "benchmarks" / "gistable" / "competition" / "readpy_results_total.csv",
)

_TOKEN_PATTERN = re.compile(r"^[0-9A-Za-z]{6,40}$")
_CSV_ID_COLUMNS = ("name", "gist_id", "gistid", "case_id", "id")


def resolve_snippet_path(case_id: str, source: BenchmarkSource = "all-gists") -> Path | None:
    case_id = case_id.strip()
    root = _source_path(source)
    primary = root / case_id / "snippet.py"
    if primary.exists():
        return primary

    if source == "competition-run":
        return None

    fallback_source: BenchmarkSource = "dockerized-gists" if source == "all-gists" else "all-gists"
    fallback = _source_path(fallback_source) / case_id / "snippet.py"
    if fallback.exists():
        return fallback
    return None


def list_case_ids(source: BenchmarkSource = "all-gists") -> list[str]:
    source_root = _source_path(source)
    if not source_root.exists():
        return []

    allowed = None
    if source == "competition-run":
        allowed = load_competition_filter_ids()

    case_ids: list[str] = []
    for case_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if allowed is not None and case_dir.name not in allowed:
            continue
        if not (case_dir / "snippet.py").exists():
            continue
        if source == "dockerized-gists" and not (case_dir / "Dockerfile").exists():
            continue
        case_ids.append(case_dir.name)
    return case_ids


def load_competition_filter_ids(path: Path = COMPETITION_FILTER_FILE) -> set[str]:
    if not path.exists():
        return set()
    selected: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in re.split(r"[^0-9A-Za-z]+", stripped):
            if _is_case_token(token):
                selected.add(token)
    return selected


def rebuild_competition_filter(
    csv_paths: Iterable[Path] = DEFAULT_COMPETITION_CSVS,
    *,
    filter_path: Path = COMPETITION_FILTER_FILE,
) -> tuple[Path, int, int]:
    known_case_ids = set(list_case_ids("all-gists"))
    csv_selected = _load_case_ids_from_csvs(csv_paths)
    effective = csv_selected & known_case_ids

    filter_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(sorted(effective))
    if payload:
        payload += "\n"
    filter_path.write_text(payload, encoding="utf-8")
    return filter_path, len(effective), len(csv_selected)


def breakdown_summary(active_source: BenchmarkSource = "all-gists") -> str:
    all_total, all_snippet, all_docker = _source_counts(_source_path("all-gists"))
    docker_total, docker_snippet, docker_docker = _source_counts(_source_path("dockerized-gists"))

    csv_selected = _load_case_ids_from_csvs(DEFAULT_COMPETITION_CSVS)
    filter_ids = load_competition_filter_ids()
    known_all = set(list_case_ids("all-gists"))

    lines = [
        f"Benchmark root: {BENCHMARK_ROOT}",
        f"Active source: {active_source}",
        "",
        "Source breakdowns",
        f"- all-gists dirs: {all_total}",
        f"- all-gists with snippet.py: {all_snippet}",
        f"- all-gists with Dockerfile: {all_docker}",
        f"- dockerized-gists dirs: {docker_total}",
        f"- dockerized-gists with snippet.py: {docker_snippet}",
        f"- dockerized-gists with Dockerfile: {docker_docker}",
        "",
        "Competition filter",
        f"- ids from configured CSVs: {len(csv_selected)}",
        f"- ids in filter file: {len(filter_ids)}",
        f"- ids valid in all-gists: {len(filter_ids & known_all)}",
        f"- runnable competition cases: {len(list_case_ids('competition-run'))}",
    ]
    return "\n".join(lines)


def _source_path(source: BenchmarkSource) -> Path:
    if source == "competition-run":
        return BENCHMARK_ROOT / "all-gists"
    return BENCHMARK_ROOT / source


def _source_counts(source_root: Path) -> tuple[int, int, int]:
    if not source_root.exists():
        return 0, 0, 0
    total = 0
    snippet = 0
    docker = 0
    for case_dir in source_root.iterdir():
        if not case_dir.is_dir():
            continue
        total += 1
        if (case_dir / "snippet.py").exists():
            snippet += 1
        if (case_dir / "Dockerfile").exists():
            docker += 1
    return total, snippet, docker


def _load_case_ids_from_csvs(csv_paths: Iterable[Path]) -> set[str]:
    selected: set[str] = set()
    for path in csv_paths:
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    selected.update(_extract_case_ids_from_row(row))
        except OSError:
            continue
    return selected


def _extract_case_ids_from_row(row: dict[str, str]) -> set[str]:
    normalized = {key.strip().lower(): str(value) for key, value in row.items() if key}
    values = [normalized[column] for column in _CSV_ID_COLUMNS if column in normalized]
    ids: set[str] = set()
    for value in values:
        for token in re.split(r"[^0-9A-Za-z]+", value):
            if _is_case_token(token):
                ids.add(token)
    return ids


def _is_case_token(token: str) -> bool:
    stripped = token.strip()
    return bool(stripped and _TOKEN_PATTERN.fullmatch(stripped))

