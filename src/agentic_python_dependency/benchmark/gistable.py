from __future__ import annotations

import csv
import json
import shutil
import ssl
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from agentic_python_dependency.config import BenchmarkCaseSource, Settings
from agentic_python_dependency.state import BenchmarkCase


class GistableDataset:
    def __init__(self, settings: Settings):
        self.settings = settings

    def dataset_root(self, ref: str | None = None) -> Path:
        return self.settings.benchmark_dir / "gistable" / (ref or self.settings.benchmark_ref)

    def _copy_tree_contents(self, source_root: Path, destination_root: Path) -> None:
        for child in source_root.iterdir():
            if child.name == ".git":
                continue
            target = destination_root / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

    def _download_tarball_via_python(self, ref: str, tarball: Path) -> None:
        url = f"https://github.com/gistable/gistable/archive/{ref}.tar.gz"
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()

        with urllib.request.urlopen(url, context=context) as response:
            tarball.write_bytes(response.read())

    def _populate_via_git(self, ref: str, root: Path) -> None:
        checkout_root = root / "_git_checkout"
        if checkout_root.exists():
            shutil.rmtree(checkout_root)
        checkout_root.mkdir(parents=True, exist_ok=True)

        subprocess.run(["git", "init"], cwd=checkout_root, capture_output=True, text=True, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/gistable/gistable.git"],
            cwd=checkout_root,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", ref],
            cwd=checkout_root,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "FETCH_HEAD"],
            cwd=checkout_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self._copy_tree_contents(checkout_root, root)
        shutil.rmtree(checkout_root)

    def fetch(self, ref: str | None = None) -> Path:
        benchmark_ref = ref or self.settings.benchmark_ref
        root = self.dataset_root(benchmark_ref)
        marker = root / ".fetch-complete"
        if marker.exists():
            return root

        root.mkdir(parents=True, exist_ok=True)
        tarball = root / f"{benchmark_ref}.tar.gz"
        try:
            self._download_tarball_via_python(benchmark_ref, tarball)

            extract_root = root / "_extract"
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tarball, "r:gz") as archive:
                archive.extractall(extract_root)

            extracted_root = next(extract_root.iterdir())
            self._copy_tree_contents(extracted_root, root)
            shutil.rmtree(extract_root)
            tarball.unlink(missing_ok=True)
        except (urllib.error.URLError, ssl.SSLError, tarfile.TarError):
            tarball.unlink(missing_ok=True)
            self._populate_via_git(benchmark_ref, root)

        marker.write_text(benchmark_ref, encoding="utf-8")
        return root

    def load_results_rows(self, ref: str | None = None) -> list[dict[str, str]]:
        root = self.dataset_root(ref)
        results_path = root / "results" / "naive-inference-results.csv"
        with results_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def results_by_id(self, ref: str | None = None) -> dict[str, dict[str, str]]:
        return {row["id"]: row for row in self.load_results_rows(ref)}

    def _resolve_case_source(self, case_source: BenchmarkCaseSource | None) -> BenchmarkCaseSource:
        return case_source or self.settings.benchmark_case_source

    def snippet_path_for_case(
        self,
        case_id: str,
        ref: str | None = None,
        *,
        case_source: BenchmarkCaseSource | None = None,
    ) -> Path | None:
        root = self.dataset_root(ref)
        resolved_source = self._resolve_case_source(case_source)
        primary = root / resolved_source / case_id / "snippet.py"
        if primary.exists():
            return primary
        fallback_source: BenchmarkCaseSource = "dockerized-gists" if resolved_source == "all-gists" else "all-gists"
        fallback = root / fallback_source / case_id / "snippet.py"
        if fallback.exists():
            return fallback
        return None

    def dockerfile_path_for_case(
        self,
        case_id: str,
        ref: str | None = None,
        *,
        case_source: BenchmarkCaseSource | None = None,
    ) -> Path | None:
        root = self.dataset_root(ref)
        resolved_source = self._resolve_case_source(case_source)
        if resolved_source == "all-gists":
            candidates = [root / "all-gists" / case_id / "Dockerfile"]
        else:
            candidates = [
                root / "dockerized-gists" / case_id / "Dockerfile",
                root / "all-gists" / case_id / "Dockerfile",
            ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def valid_case_ids(self, ref: str | None = None, case_source: BenchmarkCaseSource | None = None) -> list[str]:
        root = self.dataset_root(ref)
        resolved_source = self._resolve_case_source(case_source)
        ids = []
        case_root = root / resolved_source
        if not case_root.exists():
            return ids
        for case_dir in sorted(path for path in case_root.iterdir() if path.is_dir()):
            has_snippet = (case_dir / "snippet.py").exists()
            if not has_snippet:
                continue
            if resolved_source == "dockerized-gists" and not (case_dir / "Dockerfile").exists():
                continue
            ids.append(case_dir.name)
        return ids

    def load_case(
        self,
        case_id: str,
        ref: str | None = None,
        case_source: BenchmarkCaseSource | None = None,
    ) -> BenchmarkCase:
        root = self.dataset_root(ref)
        resolved_source = self._resolve_case_source(case_source)
        case_root = root / resolved_source / case_id
        snippet_path = self.snippet_path_for_case(case_id, ref, case_source=resolved_source)
        if snippet_path is None:
            raise FileNotFoundError(f"Snippet not found for case {case_id} in {resolved_source} or fallback source")
        dockerfile_path = self.dockerfile_path_for_case(case_id, ref, case_source=resolved_source)
        if not case_root.exists():
            case_root = snippet_path.parent
        row = self.results_by_id(ref).get(case_id, {})
        return BenchmarkCase(
            case_id=case_id,
            root_dir=case_root,
            snippet_path=snippet_path,
            dockerfile_path=dockerfile_path,
            case_source=resolved_source,
            initial_eval=row.get("initial-eval", ""),
            final_eval=row.get("final-eval", ""),
            source_url=row.get("url", ""),
        )

    def save_subset(self, name: str, case_ids: list[str], ref: str | None = None) -> Path:
        root = self.dataset_root(ref)
        subset_dir = root / "subsets"
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_path = subset_dir / f"{name}.json"
        subset_path.write_text(json.dumps(case_ids, indent=2), encoding="utf-8")
        return subset_path

    def load_subset(self, name: str, ref: str | None = None) -> list[str]:
        root = self.dataset_root(ref)
        subset_path = root / "subsets" / f"{name}.json"
        return json.loads(subset_path.read_text(encoding="utf-8"))
