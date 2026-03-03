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
