from __future__ import annotations

import hashlib
import json
from threading import RLock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import urllib.error
import urllib.request

from agentic_python_dependency.config import Settings


@dataclass(slots=True)
class OllamaInvocationStats:
    stage: str
    model: str
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    backend: str = "direct"
    done_reason: str = ""


@dataclass(slots=True)
class OllamaStatsSnapshot:
    calls: int = 0
    prompt_tokens: int = 0
    prompt_duration_ns: int = 0
    eval_tokens: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    last_model: str = ""
    last_stage: str = ""
    last_backend: str = ""
    last_done_reason: str = ""
    last_prompt_tokens: int = 0
    last_prompt_duration_ns: int = 0
    last_eval_tokens: int = 0
    last_eval_duration_ns: int = 0

    @staticmethod
    def _tokens_per_second(tokens: int, duration_ns: int) -> float | None:
        if tokens <= 0 or duration_ns <= 0:
            return None
        return tokens / (duration_ns / 1_000_000_000)

    @property
    def eval_tokens_per_second(self) -> float | None:
        return self._tokens_per_second(self.eval_tokens, self.eval_duration_ns)

    @property
    def prompt_tokens_per_second(self) -> float | None:
        return self._tokens_per_second(self.prompt_tokens, self.prompt_duration_ns)

    @property
    def last_eval_tokens_per_second(self) -> float | None:
        return self._tokens_per_second(self.last_eval_tokens, self.last_eval_duration_ns)

    @classmethod
    def from_payload(cls, payload: object) -> "OllamaStatsSnapshot":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            calls=_coerce_int(payload.get("calls")),
            prompt_tokens=_coerce_int(payload.get("prompt_tokens")),
            prompt_duration_ns=_coerce_int(payload.get("prompt_duration_ns")),
            eval_tokens=_coerce_int(payload.get("eval_tokens")),
            eval_duration_ns=_coerce_int(payload.get("eval_duration_ns")),
            total_duration_ns=_coerce_int(payload.get("total_duration_ns")),
            load_duration_ns=_coerce_int(payload.get("load_duration_ns")),
            last_model=str(payload.get("last_model", "") or ""),
            last_stage=str(payload.get("last_stage", "") or ""),
            last_backend=str(payload.get("last_backend", "") or ""),
            last_done_reason=str(payload.get("last_done_reason", "") or ""),
            last_prompt_tokens=_coerce_int(payload.get("last_prompt_tokens")),
            last_prompt_duration_ns=_coerce_int(payload.get("last_prompt_duration_ns")),
            last_eval_tokens=_coerce_int(payload.get("last_eval_tokens")),
            last_eval_duration_ns=_coerce_int(payload.get("last_eval_duration_ns")),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "prompt_duration_ns": self.prompt_duration_ns,
            "eval_tokens": self.eval_tokens,
            "eval_duration_ns": self.eval_duration_ns,
            "total_duration_ns": self.total_duration_ns,
            "load_duration_ns": self.load_duration_ns,
            "last_model": self.last_model,
            "last_stage": self.last_stage,
            "last_backend": self.last_backend,
            "last_done_reason": self.last_done_reason,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_prompt_duration_ns": self.last_prompt_duration_ns,
            "last_eval_tokens": self.last_eval_tokens,
            "last_eval_duration_ns": self.last_eval_duration_ns,
        }


class OllamaStatsTracker:
    def __init__(self, initial: object | None = None):
        self._lock = RLock()
        self._snapshot = OllamaStatsSnapshot.from_payload(initial)

    def record(self, stats: OllamaInvocationStats) -> None:
        with self._lock:
            self._snapshot.calls += 1
            self._snapshot.prompt_tokens += max(0, stats.prompt_eval_count)
            self._snapshot.prompt_duration_ns += max(0, stats.prompt_eval_duration_ns)
            self._snapshot.eval_tokens += max(0, stats.eval_count)
            self._snapshot.eval_duration_ns += max(0, stats.eval_duration_ns)
            self._snapshot.total_duration_ns += max(0, stats.total_duration_ns)
            self._snapshot.load_duration_ns += max(0, stats.load_duration_ns)
            self._snapshot.last_model = stats.model
            self._snapshot.last_stage = stats.stage
            self._snapshot.last_backend = stats.backend
            self._snapshot.last_done_reason = stats.done_reason
            self._snapshot.last_prompt_tokens = max(0, stats.prompt_eval_count)
            self._snapshot.last_prompt_duration_ns = max(0, stats.prompt_eval_duration_ns)
            self._snapshot.last_eval_tokens = max(0, stats.eval_count)
            self._snapshot.last_eval_duration_ns = max(0, stats.eval_duration_ns)

    def snapshot(self) -> OllamaStatsSnapshot:
        with self._lock:
            return OllamaStatsSnapshot.from_payload(self._snapshot.to_payload())

    def to_payload(self) -> dict[str, object]:
        return self.snapshot().to_payload()


def _coerce_int(value: object) -> int:
    try:
        if value is None or value is False:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass
class OllamaPromptRunner:
    settings: Settings
    prompt_dir: Path
    scripted_responses: dict[str, list[str]] = field(default_factory=dict)
    stats_callback: Callable[[OllamaInvocationStats], None] | None = None
    _chains: dict[tuple[str, str], Any] = field(default_factory=dict, init=False)

    def stage_model(self, stage: str) -> str:
        return self.settings.stage_model(stage)

    @staticmethod
    def _stringify_response(response: Any) -> str:
        return getattr(response, "content", str(response)).strip()

    def _cache_path(self, stage: str, prompt_text: str) -> Path:
        payload = json.dumps(
            {
                "stage": stage,
                "model": self.stage_model(stage),
                "temperature": self.settings.temperature,
                "num_ctx": self.settings.num_ctx,
                "prompt": prompt_text,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.settings.llm_cache_dir / f"{digest}.txt"

    def _read_cache(self, stage: str, prompt_text: str) -> str | None:
        if self.settings.disable_llm_cache:
            return None
        cache_path = self._cache_path(stage, prompt_text)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, stage: str, prompt_text: str, response_text: str) -> None:
        if self.settings.disable_llm_cache:
            return
        cache_path = self._cache_path(stage, prompt_text)
        cache_path.write_text(response_text, encoding="utf-8")

    def _record_stats(self, stats: OllamaInvocationStats | None) -> None:
        if stats is not None and self.stats_callback is not None:
            self.stats_callback(stats)

    def _stats_from_payload(self, stage: str, model: str, payload: object, *, backend: str) -> OllamaInvocationStats:
        metadata = payload if isinstance(payload, dict) else {}
        usage = metadata.get("usage_metadata", {}) if isinstance(metadata, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        prompt_eval_count = _coerce_int(metadata.get("prompt_eval_count")) or _coerce_int(usage.get("input_tokens"))
        eval_count = _coerce_int(metadata.get("eval_count")) or _coerce_int(usage.get("output_tokens"))
        return OllamaInvocationStats(
            stage=stage,
            model=str(metadata.get("model", model) or model),
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration_ns=_coerce_int(metadata.get("prompt_eval_duration")),
            eval_count=eval_count,
            eval_duration_ns=_coerce_int(metadata.get("eval_duration")),
            total_duration_ns=_coerce_int(metadata.get("total_duration")),
            load_duration_ns=_coerce_int(metadata.get("load_duration")),
            backend=backend,
            done_reason=str(metadata.get("done_reason", "") or ""),
        )

    def _invoke_via_ollama_http(self, stage: str, prompt_text: str) -> tuple[str, OllamaInvocationStats]:
        model = self.stage_model(stage)
        request = urllib.request.Request(
            f"{self.settings.ollama_base_url}/api/generate",
            data=json.dumps(
                {
                    "model": model,
                    "prompt": prompt_text,
                    "stream": False,
                    "options": {
                        "temperature": self.settings.temperature,
                        "num_ctx": self.settings.num_ctx,
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed for stage '{stage}': {exc}") from exc
        return str(payload.get("response", "")).strip(), self._stats_from_payload(stage, model, payload, backend="direct")

    def invoke_template(self, stage: str, template_name: str, variables: dict[str, Any]) -> str:
        if self.scripted_responses.get(stage):
            return self.scripted_responses[stage].pop(0)

        prompt_text = (self.prompt_dir / template_name).read_text(encoding="utf-8")
        rendered_prompt = prompt_text.format(**variables)
        cached = self._read_cache(stage, rendered_prompt)
        if cached is not None:
            return cached
        if not self.settings.use_langchain:
            rendered_response, stats = self._invoke_via_ollama_http(stage, rendered_prompt)
            self._record_stats(stats)
            self._write_cache(stage, rendered_prompt, rendered_response)
            return rendered_response
        key = (stage, template_name)
        if key not in self._chains:
            try:
                from langchain_core.prompts import PromptTemplate
                from langchain_ollama import ChatOllama
            except ImportError as exc:  # pragma: no cover - dependency install gap
                raise RuntimeError(
                    "LangChain and langchain-ollama must be installed to invoke Gemma models."
                ) from exc

            llm = ChatOllama(
                model=self.stage_model(stage),
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.temperature,
                num_ctx=self.settings.num_ctx,
            )
            self._chains[key] = PromptTemplate.from_template(prompt_text) | llm

        response = self._chains[key].invoke(variables)
        rendered_response = self._stringify_response(response)
        self._record_stats(self._stats_from_payload(stage, self.stage_model(stage), getattr(response, "response_metadata", {}), backend="langchain"))
        self._write_cache(stage, rendered_prompt, rendered_response)
        return rendered_response

    def invoke_text(self, stage: str, prompt_text: str) -> str:
        if self.scripted_responses.get(stage):
            return self.scripted_responses[stage].pop(0)

        cached = self._read_cache(stage, prompt_text)
        if cached is not None:
            return cached
        if not self.settings.use_langchain:
            rendered_response, stats = self._invoke_via_ollama_http(stage, prompt_text)
            self._record_stats(stats)
            self._write_cache(stage, prompt_text, rendered_response)
            return rendered_response

        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - dependency install gap
            raise RuntimeError(
                "LangChain and langchain-ollama must be installed to invoke Gemma models."
            ) from exc

        llm = ChatOllama(
            model=self.stage_model(stage),
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.temperature,
            num_ctx=self.settings.num_ctx,
        )
        response = llm.invoke(prompt_text)
        rendered_response = self._stringify_response(response)
        self._record_stats(self._stats_from_payload(stage, self.stage_model(stage), getattr(response, "response_metadata", {}), backend="langchain"))
        self._write_cache(stage, prompt_text, rendered_response)
        return rendered_response
