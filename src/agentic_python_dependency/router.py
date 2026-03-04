from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from agentic_python_dependency.config import Settings


@dataclass
class OllamaPromptRunner:
    settings: Settings
    prompt_dir: Path
    scripted_responses: dict[str, list[str]] = field(default_factory=dict)
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

    def _invoke_via_ollama_http(self, stage: str, prompt_text: str) -> str:
        request = urllib.request.Request(
            f"{self.settings.ollama_base_url}/api/generate",
            data=json.dumps(
                {
                    "model": self.stage_model(stage),
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
        return str(payload.get("response", "")).strip()

    def invoke_template(self, stage: str, template_name: str, variables: dict[str, Any]) -> str:
        if self.scripted_responses.get(stage):
            return self.scripted_responses[stage].pop(0)

        prompt_text = (self.prompt_dir / template_name).read_text(encoding="utf-8")
        rendered_prompt = prompt_text.format(**variables)
        cached = self._read_cache(stage, rendered_prompt)
        if cached is not None:
            return cached
        if not self.settings.use_langchain:
            rendered_response = self._invoke_via_ollama_http(stage, rendered_prompt)
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
        self._write_cache(stage, rendered_prompt, rendered_response)
        return rendered_response

    def invoke_text(self, stage: str, prompt_text: str) -> str:
        if self.scripted_responses.get(stage):
            return self.scripted_responses[stage].pop(0)

        cached = self._read_cache(stage, prompt_text)
        if cached is not None:
            return cached
        if not self.settings.use_langchain:
            rendered_response = self._invoke_via_ollama_http(stage, prompt_text)
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
        self._write_cache(stage, prompt_text, rendered_response)
        return rendered_response
