from __future__ import annotations

import hashlib
import json
import os
import ssl
import tarfile
import urllib.parse
import urllib.request
import zipfile
from email.parser import Parser
from io import BytesIO
from pathlib import Path
from typing import Any


PARSED_SCHEMA_VERSION = 2


def _safe_name(package: str, version: str) -> str:
    digest = hashlib.sha256(f"{package}=={version}".encode("utf-8")).hexdigest()[:16]
    stem = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in package)
    return f"{stem[:48]}-{version[:24]}-{digest}"


def _default_top_level_candidates(package: str) -> list[str]:
    normalized = package.strip().lower().replace("-", "_").replace(".", "_")
    parts = [part for part in normalized.split("_") if part]
    candidates = {normalized}
    if parts:
        candidates.add(parts[0])
    return sorted(candidates)


def _parse_top_level_text(text: str) -> list[str]:
    return sorted({line.strip() for line in text.splitlines() if line.strip()})


def _parse_metadata_names(text: str) -> list[str]:
    parsed = Parser().parsestr(text)
    names = set()
    for key in ("Name",):
        value = (parsed.get(key) or "").strip()
        if not value:
            continue
        names.update(_default_top_level_candidates(value))
    return sorted(names)


def _parse_requires_python(text: str) -> str:
    parsed = Parser().parsestr(text)
    return str(parsed.get("Requires-Python") or "").strip()


def _file_size_bytes(file_meta: dict[str, Any]) -> int | None:
    raw_size = file_meta.get("size")
    if isinstance(raw_size, int):
        return raw_size
    if isinstance(raw_size, float):
        return int(raw_size)
    if isinstance(raw_size, str):
        try:
            return int(raw_size.strip())
        except ValueError:
            return None
    return None


class PackageMetadataStore:
    def __init__(self, cache_dir: Path, *, keep_raw: bool | None = None):
        self.cache_dir = cache_dir
        self.raw_dir = cache_dir / "raw"
        self.parsed_dir = cache_dir / "parsed"
        self.keep_raw = (
            keep_raw
            if keep_raw is not None
            else os.environ.get("APDR_PACKAGE_METADATA_KEEP_RAW", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        raw_limit = os.environ.get("APDR_PACKAGE_METADATA_MAX_DOWNLOAD_MB", "").strip()
        try:
            max_download_mb = int(raw_limit) if raw_limit else 25
        except ValueError:
            max_download_mb = 25
        self.max_download_bytes = max(0, max_download_mb) * 1024 * 1024
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    def _raw_path(self, package: str, version: str) -> Path:
        return self.raw_dir / f"{_safe_name(package, version)}.bin"

    def _parsed_path(self, package: str, version: str) -> Path:
        return self.parsed_dir / f"{_safe_name(package, version)}.json"

    def _download_bytes(self, url: str) -> bytes:
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=30, context=context) as response:
            return response.read()

    def _cleanup_raw_cache(self, raw_path: Path) -> None:
        if self.keep_raw or not raw_path.exists():
            return
        try:
            raw_path.unlink()
        except OSError:
            return

    def _fetch_release_bytes(self, release_files: list[dict[str, Any]], raw_path: Path) -> bytes | None:
        preferred = sorted(
            release_files,
            key=lambda item: (
                item.get("packagetype") != "bdist_wheel",
                "py3" not in str(item.get("filename", "")).lower(),
                _file_size_bytes(item) or 0,
            ),
        )
        for file_meta in preferred:
            url = str(file_meta.get("url", "")).strip()
            if not url:
                continue
            size = _file_size_bytes(file_meta)
            if self.max_download_bytes and size is not None and size > self.max_download_bytes:
                continue
            try:
                payload = self._download_bytes(url)
            except OSError:
                continue
            if self.keep_raw:
                raw_path.write_bytes(payload)
            return payload
        return None

    @staticmethod
    def _extract_from_wheel(payload: bytes) -> tuple[list[str], list[str], str]:
        top_level: set[str] = set()
        requires_dist: list[str] = []
        requires_python = ""
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in archive.namelist():
                lowered = name.lower()
                if lowered.endswith("top_level.txt"):
                    top_level.update(_parse_top_level_text(archive.read(name).decode("utf-8", errors="replace")))
                elif lowered.endswith("metadata"):
                    text = archive.read(name).decode("utf-8", errors="replace")
                    top_level.update(_parse_metadata_names(text))
                    parsed = Parser().parsestr(text)
                    requires_dist.extend(parsed.get_all("Requires-Dist") or [])
                    if not requires_python:
                        requires_python = _parse_requires_python(text)
        return (
            sorted(top_level),
            sorted({entry.strip() for entry in requires_dist if entry.strip()}),
            requires_python,
        )

    @staticmethod
    def _extract_from_sdist(payload: bytes) -> tuple[list[str], list[str], str]:
        top_level: set[str] = set()
        requires_dist: list[str] = []
        requires_python = ""
        with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
            for member in archive.getmembers():
                lowered = member.name.lower()
                if not member.isfile():
                    continue
                if lowered.endswith("top_level.txt") or lowered.endswith("metadata") or lowered.endswith("pkg-info"):
                    file_obj = archive.extractfile(member)
                    if file_obj is None:
                        continue
                    text = file_obj.read().decode("utf-8", errors="replace")
                    if lowered.endswith("top_level.txt"):
                        top_level.update(_parse_top_level_text(text))
                    else:
                        top_level.update(_parse_metadata_names(text))
                        parsed = Parser().parsestr(text)
                        requires_dist.extend(parsed.get_all("Requires-Dist") or [])
                        if not requires_python:
                            requires_python = _parse_requires_python(text)
        return (
            sorted(top_level),
            sorted({entry.strip() for entry in requires_dist if entry.strip()}),
            requires_python,
        )

    @staticmethod
    def _parsed_cache_is_fresh(payload: dict[str, Any]) -> bool:
        schema_version = payload.get("schema_version")
        if schema_version != PARSED_SCHEMA_VERSION:
            return False
        return "requires_python" in payload

    def parse_release_metadata(
        self,
        package: str,
        version: str,
        *,
        release_files: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        parsed_path = self._parsed_path(package, version)
        raw_path = self._raw_path(package, version)
        if parsed_path.exists():
            cached = json.loads(parsed_path.read_text(encoding="utf-8"))
            if self._parsed_cache_is_fresh(cached):
                self._cleanup_raw_cache(raw_path)
                return cached

        payload = raw_path.read_bytes() if raw_path.exists() else None
        if payload is None and release_files:
            payload = self._fetch_release_bytes(release_files, raw_path)

        top_level: list[str] = []
        requires_dist: list[str] = []
        requires_python = ""
        source = "derived"
        if payload:
            try:
                if zipfile.is_zipfile(BytesIO(payload)):
                    top_level, requires_dist, requires_python = self._extract_from_wheel(payload)
                    source = "wheel"
                else:
                    top_level, requires_dist, requires_python = self._extract_from_sdist(payload)
                    source = "sdist"
            except (zipfile.BadZipFile, tarfile.TarError, OSError):
                top_level = []
                requires_dist = []
                requires_python = ""

        if not top_level:
            top_level = _default_top_level_candidates(package)
        parsed = {
            "schema_version": PARSED_SCHEMA_VERSION,
            "package": package,
            "version": version,
            "top_level_modules": sorted({item for item in top_level if item}),
            "requires_dist": requires_dist,
            "requires_python": requires_python,
            "source": source,
        }
        parsed_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        self._cleanup_raw_cache(raw_path)
        return parsed
