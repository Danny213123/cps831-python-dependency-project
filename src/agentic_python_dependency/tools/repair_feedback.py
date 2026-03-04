from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_feedback_event(memory_dir: Path, event: dict[str, Any]) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with (memory_dir / "repair_feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize_feedback_memory(memory_dir: Path, *, limit: int = 5) -> dict[str, Any]:
    jsonl_path = memory_dir / "repair_feedback.jsonl"
    summary_path = memory_dir / "repair_feedback_summary.json"
    if not jsonl_path.exists():
        summary = {"entries": [], "count": 0}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"successes": 0, "failures": 0})
    count = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += 1
        key = (
            str(payload.get("error_category", "")),
            str(payload.get("target_python", "")),
            str(payload.get("strategy_type", "")),
        )
        bucket = grouped[key]
        if payload.get("success"):
            bucket["successes"] += 1
        else:
            bucket["failures"] += 1
    entries = []
    for (error_category, target_python, strategy_type), bucket in grouped.items():
        total = bucket["successes"] + bucket["failures"]
        entries.append(
            {
                "error_category": error_category,
                "target_python": target_python,
                "strategy_type": strategy_type,
                "successes": bucket["successes"],
                "failures": bucket["failures"],
                "success_rate": (bucket["successes"] / total) if total else 0.0,
            }
        )
    entries.sort(key=lambda item: (item["success_rate"], item["successes"]), reverse=True)
    summary = {"entries": entries[:limit], "count": count}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
