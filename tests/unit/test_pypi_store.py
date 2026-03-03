from pathlib import Path

from agentic_python_dependency.tools.pypi_store import PyPIMetadataStore


def test_compatible_release_records_filters_yanked_and_python_incompatible_versions() -> None:
    payload = {
        "releases": {
            "1.0.0": [
                {"yanked": False, "requires_python": ">=3.8", "upload_time_iso_8601": "2024-01-01T00:00:00"}
            ],
            "1.1.0": [
                {"yanked": True, "requires_python": ">=3.8", "upload_time_iso_8601": "2024-02-01T00:00:00"}
            ],
            "2.0.0": [
                {"yanked": False, "requires_python": ">=3.13", "upload_time_iso_8601": "2024-03-01T00:00:00"}
            ],
            "1.0.1": [
                {"yanked": False, "requires_python": ">=3.10", "upload_time_iso_8601": "2024-01-15T00:00:00"}
            ],
        }
    }

    records = PyPIMetadataStore.compatible_release_records(payload, target_python="3.12")

    assert [record.version for record in records] == ["1.0.1", "1.0.0"]


def test_compatible_release_records_avoids_post_eol_python2_releases_without_metadata() -> None:
    payload = {
        "releases": {
            "1.0.0": [
                {"yanked": False, "requires_python": "", "upload_time_iso_8601": "2019-06-01T00:00:00"}
            ],
            "2.0.0": [
                {"yanked": False, "requires_python": "", "upload_time_iso_8601": "2024-06-01T00:00:00"}
            ],
            "2.1.0": [
                {"yanked": False, "requires_python": ">=2.7,<3", "upload_time_iso_8601": "2024-07-01T00:00:00"}
            ],
        }
    }

    records = PyPIMetadataStore.compatible_release_records(payload, target_python="2.7")

    assert [record.version for record in records] == ["1.0.0"]


def test_safe_cache_name_handles_windows_reserved_names() -> None:
    assert PyPIMetadataStore._safe_cache_name("con") == "pkg_con"
    assert PyPIMetadataStore._safe_cache_name("aux") == "pkg_aux"
    assert PyPIMetadataStore._safe_cache_name("Flask-SQLAlchemy") == "Flask-SQLAlchemy"


def test_safe_cache_name_truncates_long_names_with_hash_suffix() -> None:
    long_name = "a" * 200

    safe_name = PyPIMetadataStore._safe_cache_name(long_name)

    assert len(safe_name) <= PyPIMetadataStore._MAX_CACHE_STEM
    assert safe_name.startswith("a")
    assert "-" in safe_name


def test_fetch_package_json_uses_sanitized_cache_path(tmp_path: Path, monkeypatch) -> None:
    store = PyPIMetadataStore(tmp_path)

    monkeypatch.setattr(
        store,
        "_download_json",
        lambda package: {"info": {"name": package}, "releases": {}},
    )

    payload = store.fetch_package_json("con")

    assert payload["info"]["name"] == "con"
    assert (tmp_path / "raw" / "pkg_con.json").exists()


def test_index_payload_disables_db_on_connect_failure(tmp_path: Path, monkeypatch) -> None:
    store = PyPIMetadataStore(tmp_path)

    def boom():
        raise OSError("locked")

    monkeypatch.setattr(store, "_connect", boom)
    store._db_enabled = True

    store._index_payload("requests", {"releases": {}})

    assert store._db_enabled is False
