from pathlib import Path

from agentic_python_dependency.benchmark.gistable import GistableDataset
from agentic_python_dependency.config import Settings


def _create_case(dataset_root: Path, case_id: str) -> None:
    case_dir = dataset_root / "all-gists" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "snippet.py").write_text("import requests\n", encoding="utf-8")


def test_competition_case_ids_falls_back_to_filter_file_when_csvs_missing(tmp_path: Path) -> None:
    settings = Settings.from_env(
        project_root=tmp_path,
        competition_result_csvs_override=[str(tmp_path / "missing.csv")],
        competition_case_ids_file_override=str(tmp_path / "competition" / "competition-case-ids.txt"),
    )
    dataset = GistableDataset(settings)
    root = dataset.dataset_root(None)
    _create_case(root, "abc123")
    _create_case(root, "def456")
    settings.competition_case_ids_file.parent.mkdir(parents=True, exist_ok=True)
    settings.competition_case_ids_file.write_text("abc123\nzzzzzz\n", encoding="utf-8")

    assert dataset.competition_case_ids() == {"abc123"}
    assert dataset.valid_case_ids(case_source="competition-run") == ["abc123"]


def test_competition_case_ids_syncs_filter_file_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "official.csv"
    csv_path.write_text("name,result\nabc123,OtherPass\ndef456,ImportError\n", encoding="utf-8")
    settings = Settings.from_env(
        project_root=tmp_path,
        competition_result_csvs_override=[str(csv_path)],
        competition_case_ids_file_override=str(tmp_path / "competition" / "competition-case-ids.txt"),
    )
    dataset = GistableDataset(settings)
    root = dataset.dataset_root(None)
    _create_case(root, "abc123")
    _create_case(root, "def456")

    ids = dataset.competition_case_ids()

    assert ids == {"abc123", "def456"}
    assert settings.competition_case_ids_file.exists()
    assert settings.competition_case_ids_file.read_text(encoding="utf-8") == "abc123\ndef456\n"
