from __future__ import annotations

from collections import defaultdict

from agentic_python_dependency.benchmark.gistable import GistableDataset


def build_smoke30(dataset: GistableDataset, ref: str | None = None) -> list[str]:
    rows = sorted(dataset.load_results_rows(ref), key=lambda row: row["id"])
    valid_ids = set(dataset.valid_case_ids(ref))

    buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        case_id = row["id"]
        if case_id not in valid_ids:
            continue
        initial = row["initial-eval"]
        if initial == "ImportError":
            buckets["ImportError"].append(case_id)
        elif initial == "Success":
            buckets["Success"].append(case_id)
        elif initial == "SyntaxError":
            buckets["SyntaxError"].append(case_id)
        elif initial in {"NameError", "IOError", "AttributeError"}:
            buckets["Other"].append(case_id)

    selected = (
        buckets["ImportError"][:20]
        + buckets["Success"][:5]
        + buckets["SyntaxError"][:3]
        + buckets["Other"][:2]
    )
    if len(selected) < 30:
        seen = set(selected)
        for row in rows:
            case_id = row["id"]
            if case_id not in valid_ids or case_id in seen:
                continue
            selected.append(case_id)
            seen.add(case_id)
            if len(selected) == 30:
                break
    return selected
