from __future__ import annotations

import hashlib
import json
import ssl
import threading
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from agentic_python_dependency.presets import get_preset_config
from agentic_python_dependency.state import PackageVersionOptions


SCHEMA_VERSION = 1
PYTHON2_UPLOAD_CUTOFF = "2020-01-01T00:00:00"
PYTHON2_VERSION_CEILINGS = {
    "django": Version("2.0"),
    "sqlalchemy": Version("1.4"),
    "fabric": Version("2.0"),
    "boto3": Version("1.18"),
    "emoji": Version("1.0"),
    "pygame": Version("2.0"),
    "luigi": Version("3.0"),
}


@dataclass(slots=True)
class PyPIReleaseRecord:
    version: str
    requires_python: str
    yanked: bool
    upload_time: str


class PyPIMetadataStore:
    _global_lock = threading.RLock()
    _MAX_CACHE_STEM = 80

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.raw_dir = cache_dir / "raw"
        self.db_path = cache_dir / "index.duckdb"
        self._db_enabled = True
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def _safe_cache_name(package: str) -> str:
        normalized = []
        for character in package:
            if character.isalnum() or character in {"-", "_", "."}:
                normalized.append(character)
            else:
                normalized.append("_")
        candidate = "".join(normalized).strip(" .")
        if not candidate:
            candidate = "package"
        reserved = {
            "con",
            "prn",
            "aux",
            "nul",
            *(f"com{index}" for index in range(1, 10)),
            *(f"lpt{index}" for index in range(1, 10)),
        }
        if candidate.lower() in reserved:
            candidate = f"pkg_{candidate}"
        if len(candidate) > PyPIMetadataStore._MAX_CACHE_STEM:
            digest = hashlib.sha256(package.encode("utf-8")).hexdigest()[:16]
            candidate = f"{candidate[:PyPIMetadataStore._MAX_CACHE_STEM - 17]}-{digest}"
        return candidate

    def _connect(self):
        try:
            import duckdb
            return duckdb.connect(str(self.db_path))
        except ImportError:  # pragma: no cover - lightweight test fallback
            import sqlite3

            return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._global_lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS metadata_schema (
                            schema_version INTEGER
                        )
                        """
                    )
                    if conn.execute("SELECT COUNT(*) FROM metadata_schema").fetchone()[0] == 0:
                        conn.execute("INSERT INTO metadata_schema VALUES (?)", [SCHEMA_VERSION])
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS package_releases (
                            package TEXT,
                            version TEXT,
                            requires_python TEXT,
                            yanked BOOLEAN,
                            upload_time TEXT
                        )
                        """
                    )
            except Exception:
                self._db_enabled = False

    def _download_json(self, package: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(package)
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()

        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{encoded}/json",
            timeout=30,
            context=context,
        ) as response:
            return json.load(response)

    def fetch_package_json(self, package: str) -> dict[str, Any]:
        raw_path = self.raw_dir / f"{self._safe_cache_name(package)}.json"
        if raw_path.exists():
            return json.loads(raw_path.read_text(encoding="utf-8"))

        try:
            payload = self._download_json(package)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(f"Package not found on PyPI: {package}") from exc
            raise
        with self._global_lock:
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._index_payload(package, payload)
        return payload

    def _index_payload(self, package: str, payload: dict[str, Any]) -> None:
        if not self._db_enabled:
            return
        rows: list[tuple[str, str, str, bool, str]] = []
        for version, files in payload.get("releases", {}).items():
            for file_meta in files:
                rows.append(
                    (
                        package,
                        version,
                        file_meta.get("requires_python") or "",
                        bool(file_meta.get("yanked", False)),
                        file_meta.get("upload_time_iso_8601") or file_meta.get("upload_time") or "",
                    )
                )
        with self._global_lock:
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM package_releases WHERE package = ?", [package])
                    if rows:
                        conn.executemany("INSERT INTO package_releases VALUES (?, ?, ?, ?, ?)", rows)
            except Exception:
                self._db_enabled = False

    @staticmethod
    def compatible_release_records(
        payload: dict[str, Any], target_python: str, limit: int = 20
    ) -> list[PyPIReleaseRecord]:
        compatible: list[PyPIReleaseRecord] = []
        target_version = Version(target_python)
        target_is_python2 = target_version < Version("3")
        for version, files in payload.get("releases", {}).items():
            if not files:
                continue
            accepted_files = []
            for file_meta in files:
                if file_meta.get("yanked", False):
                    continue
                requires_python = file_meta.get("requires_python") or ""
                upload_time = file_meta.get("upload_time_iso_8601") or file_meta.get("upload_time") or ""
                if target_is_python2 and upload_time and upload_time >= PYTHON2_UPLOAD_CUTOFF:
                    continue
                if requires_python:
                    try:
                        if target_version not in SpecifierSet(requires_python):
                            continue
                    except InvalidSpecifier:
                        continue
                accepted_files.append(file_meta)

            if not accepted_files:
                continue

            try:
                parsed_version = Version(version)
            except InvalidVersion:
                continue

            newest_file = max(
                accepted_files,
                key=lambda entry: entry.get("upload_time_iso_8601") or entry.get("upload_time") or "",
            )
            compatible.append(
                PyPIReleaseRecord(
                    version=str(parsed_version),
                    requires_python=newest_file.get("requires_python") or "",
                    yanked=False,
                    upload_time=newest_file.get("upload_time_iso_8601")
                    or newest_file.get("upload_time")
                    or "",
                )
            )

        stable_releases = [record for record in compatible if not Version(record.version).is_prerelease]
        if stable_releases:
            compatible = stable_releases

        compatible.sort(key=lambda item: Version(item.version), reverse=True)
        return compatible[:limit]

    @staticmethod
    def _apply_policy(
        package: str,
        records: list[PyPIReleaseRecord],
        target_python: str,
        preset: str,
    ) -> tuple[list[PyPIReleaseRecord], list[str]]:
        notes: list[str] = []
        if not records:
            return records, notes

        normalized_package = package.strip().replace("-", "_").lower()
        target_version = Version(target_python)
        target_is_python2 = target_version < Version("3")
        preset_config = get_preset_config(preset)

        if target_is_python2 and preset_config.compatibility_policy != "essential":
            ceiling = PYTHON2_VERSION_CEILINGS.get(normalized_package)
            if ceiling is not None:
                filtered = [record for record in records if Version(record.version) < ceiling]
                if filtered:
                    records = filtered
                    notes.append(f"python2_ceiling<{ceiling}")
        return records, notes

    def get_version_options(
        self,
        package: str,
        target_python: str,
        limit: int = 20,
        *,
        preset: str = "optimized",
    ) -> PackageVersionOptions:
        payload = self.fetch_package_json(package)
        compatible = self.compatible_release_records(payload, target_python=target_python, limit=limit)
        compatible, policy_notes = self._apply_policy(package, compatible, target_python, preset)
        package_requires_dist = payload.get("info", {}).get("requires_dist") or []
        return PackageVersionOptions(
            package=package,
            versions=[record.version for record in compatible],
            requires_python={record.version: record.requires_python for record in compatible},
            upload_time={record.version: record.upload_time for record in compatible},
            policy_notes=policy_notes,
            requires_dist={record.version: list(package_requires_dist) for record in compatible},
        )

    @staticmethod
    def format_prompt_block(options: list[PackageVersionOptions]) -> str:
        blocks = []
        for option in options:
            versions = ", ".join(option.versions)
            blocks.append(f"{option.package}: {versions}")
        return "\n".join(blocks)
