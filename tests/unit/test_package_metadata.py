from __future__ import annotations

import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

from agentic_python_dependency.tools.package_metadata import PARSED_SCHEMA_VERSION, PackageMetadataStore


def _write_sdist(path: Path, metadata_text: str) -> None:
    payload = BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        metadata_bytes = metadata_text.encode("utf-8")
        metadata_info = tarfile.TarInfo("demo-1.0.0/PKG-INFO")
        metadata_info.size = len(metadata_bytes)
        archive.addfile(metadata_info, BytesIO(metadata_bytes))
    path.write_bytes(payload.getvalue())


def _write_wheel(path: Path, metadata_text: str, top_level_text: str = "demo\n") -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, mode="w") as archive:
        archive.writestr("demo-1.0.0.dist-info/METADATA", metadata_text)
        archive.writestr("demo-1.0.0.dist-info/top_level.txt", top_level_text)
    path.write_bytes(payload.getvalue())


def test_parse_release_metadata_extracts_requires_python_from_wheel(tmp_path: Path) -> None:
    store = PackageMetadataStore(tmp_path)
    raw_path = store._raw_path("demo", "1.0.0")
    _write_wheel(
        raw_path,
        (
            "Metadata-Version: 2.1\n"
            "Name: demo\n"
            "Version: 1.0.0\n"
            "Requires-Python: >=3.8\n"
            "Requires-Dist: requests>=2\n"
        ),
    )

    metadata = store.parse_release_metadata("demo", "1.0.0")

    assert metadata["requires_python"] == ">=3.8"
    assert metadata["requires_dist"] == ["requests>=2"]
    assert metadata["schema_version"] == PARSED_SCHEMA_VERSION
    assert metadata["top_level_modules"] == ["demo"]


def test_parse_release_metadata_extracts_requires_python_from_sdist(tmp_path: Path) -> None:
    store = PackageMetadataStore(tmp_path)
    raw_path = store._raw_path("demo", "1.0.0")
    _write_sdist(
        raw_path,
        (
            "Metadata-Version: 1.2\n"
            "Name: demo\n"
            "Version: 1.0.0\n"
            "Requires-Python: >=2.7,<3\n"
            "Requires-Dist: numpy>=1.16\n"
        ),
    )

    metadata = store.parse_release_metadata("demo", "1.0.0")

    assert metadata["requires_python"] == ">=2.7,<3"
    assert metadata["requires_dist"] == ["numpy>=1.16"]


def test_parse_release_metadata_rebuilds_stale_cached_payload_without_requires_python(tmp_path: Path) -> None:
    store = PackageMetadataStore(tmp_path)
    raw_path = store._raw_path("demo", "1.0.0")
    parsed_path = store._parsed_path("demo", "1.0.0")
    _write_wheel(
        raw_path,
        (
            "Metadata-Version: 2.1\n"
            "Name: demo\n"
            "Version: 1.0.0\n"
            "Requires-Python: >=3.9\n"
        ),
    )
    parsed_path.write_text(
        json.dumps(
            {
                "package": "demo",
                "version": "1.0.0",
                "top_level_modules": ["demo"],
                "requires_dist": [],
                "source": "wheel",
            }
        ),
        encoding="utf-8",
    )

    metadata = store.parse_release_metadata("demo", "1.0.0")
    persisted = json.loads(parsed_path.read_text(encoding="utf-8"))

    assert metadata["requires_python"] == ">=3.9"
    assert persisted["requires_python"] == ">=3.9"
    assert persisted["schema_version"] == PARSED_SCHEMA_VERSION


def test_parse_release_metadata_does_not_persist_downloaded_raw_payload_by_default(tmp_path: Path) -> None:
    store = PackageMetadataStore(tmp_path)
    wheel_path = tmp_path / "demo-1.0.0.whl"
    _write_wheel(
        wheel_path,
        (
            "Metadata-Version: 2.1\n"
            "Name: demo\n"
            "Version: 1.0.0\n"
            "Requires-Python: >=3.8\n"
        ),
    )
    release_files = [{"url": "https://example.invalid/demo-1.0.0.whl", "packagetype": "bdist_wheel", "filename": "demo-1.0.0.whl"}]
    store._download_bytes = lambda url: wheel_path.read_bytes()  # type: ignore[method-assign]

    metadata = store.parse_release_metadata("demo", "1.0.0", release_files=release_files)

    assert metadata["requires_python"] == ">=3.8"
    assert not store._raw_path("demo", "1.0.0").exists()


def test_parse_release_metadata_deletes_stale_raw_payload_when_parsed_cache_is_fresh(tmp_path: Path) -> None:
    store = PackageMetadataStore(tmp_path)
    raw_path = store._raw_path("demo", "1.0.0")
    parsed_path = store._parsed_path("demo", "1.0.0")
    _write_wheel(
        raw_path,
        (
            "Metadata-Version: 2.1\n"
            "Name: demo\n"
            "Version: 1.0.0\n"
            "Requires-Python: >=3.10\n"
        ),
    )
    parsed_path.write_text(
        json.dumps(
            {
                "schema_version": PARSED_SCHEMA_VERSION,
                "package": "demo",
                "version": "1.0.0",
                "top_level_modules": ["demo"],
                "requires_dist": [],
                "requires_python": ">=3.10",
                "source": "wheel",
            }
        ),
        encoding="utf-8",
    )

    metadata = store.parse_release_metadata("demo", "1.0.0")

    assert metadata["requires_python"] == ">=3.10"
    assert not raw_path.exists()


def test_parse_release_metadata_skips_oversized_release_artifacts(tmp_path: Path) -> None:
    store = PackageMetadataStore(tmp_path)
    release_files = [
        {
            "url": "https://example.invalid/huge.whl",
            "packagetype": "bdist_wheel",
            "filename": "huge.whl",
            "size": 500 * 1024 * 1024,
        }
    ]

    calls: list[str] = []

    def _download(url: str) -> bytes:
        calls.append(url)
        raise AssertionError("oversized artifacts should not be downloaded")

    store._download_bytes = _download  # type: ignore[method-assign]

    metadata = store.parse_release_metadata("torch", "2.4.1", release_files=release_files)

    assert calls == []
    assert metadata["top_level_modules"] == ["torch"]
    assert metadata["requires_dist"] == []
