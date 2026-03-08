import io
import json
from pathlib import Path
import sys
import types

from agentic_python_dependency.config import Settings
from agentic_python_dependency.router import OllamaPromptRunner


def make_settings(tmp_path: Path) -> Settings:
    return Settings.from_env(project_root=tmp_path)


def test_invoke_text_uses_disk_cache_without_model_imports(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    runner = OllamaPromptRunner(settings, settings.prompts_dir)
    prompt_text = "Return only package==version"
    runner._write_cache("version", prompt_text, "requests==2.32.3")

    response = runner.invoke_text("version", prompt_text)

    assert response == "requests==2.32.3"


def test_invoke_template_uses_disk_cache_without_model_imports(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "example.txt").write_text("Hello {name}", encoding="utf-8")
    runner = OllamaPromptRunner(settings, prompt_dir)
    runner._write_cache("extract", "Hello world", "cached-response")

    response = runner.invoke_template("extract", "example.txt", {"name": "world"})

    assert response == "cached-response"


def test_invoke_text_skips_disk_cache_when_disabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.disable_llm_cache = True
    runner = OllamaPromptRunner(settings, settings.prompts_dir, scripted_responses={"version": ["fresh-response"]})
    prompt_text = "Return only package==version"
    runner._write_cache("version", prompt_text, "cached-response")

    response = runner.invoke_text("version", prompt_text)

    assert response == "fresh-response"
    assert not any(settings.llm_cache_dir.iterdir())


def test_stage_model_uses_reasoning_model_when_moe_disabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.use_moe = False
    settings.extraction_model = "extract:model"
    settings.reasoning_model = "runner:model"
    settings.version_model = "version:model"

    runner = OllamaPromptRunner(settings, settings.prompts_dir)

    assert runner.stage_model("extract") == "runner:model"
    assert runner.stage_model("version") == "runner:model"


def test_invoke_text_can_use_direct_ollama_http_without_langchain(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.use_langchain = False
    settings.disable_llm_cache = True
    settings.version_model = "version:model"
    recorded = []
    runner = OllamaPromptRunner(settings, settings.prompts_dir, stats_callback=recorded.append)
    payload = io.BytesIO(
        json.dumps(
            {
                "response": "direct-response",
                "model": "version:model",
                "prompt_eval_count": 18,
                "prompt_eval_duration": 12_000_000,
                "eval_count": 36,
                "eval_duration": 600_000_000,
            }
        ).encode("utf-8")
    )
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return payload

    monkeypatch.setattr("agentic_python_dependency.router.urllib.request.urlopen", fake_urlopen)

    response = runner.invoke_text("version", "Return only package==version")

    assert response == "direct-response"
    assert captured["url"] == f"{settings.ollama_base_url}/api/generate"
    assert captured["body"]["model"] == "version:model"
    assert len(recorded) == 1
    assert recorded[0].stage == "version"
    assert recorded[0].model == "version:model"
    assert recorded[0].eval_count == 36
    assert recorded[0].eval_duration_ns == 600_000_000


def test_invoke_text_can_record_langchain_ollama_stats(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    settings.disable_llm_cache = True
    settings.version_model = "version:model"
    recorded = []

    class FakeResponse:
        content = "langchain-response"
        response_metadata = {
            "model": "version:model",
            "prompt_eval_count": 24,
            "prompt_eval_duration": 80_000_000,
            "eval_count": 40,
            "eval_duration": 500_000_000,
            "done_reason": "stop",
        }

    class FakeChatOllama:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def invoke(self, prompt_text: str) -> FakeResponse:
            assert prompt_text == "Return only package==version"
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "langchain_ollama", types.SimpleNamespace(ChatOllama=FakeChatOllama))

    runner = OllamaPromptRunner(settings, settings.prompts_dir, stats_callback=recorded.append)

    response = runner.invoke_text("version", "Return only package==version")

    assert response == "langchain-response"
    assert len(recorded) == 1
    assert recorded[0].backend == "langchain"
    assert recorded[0].done_reason == "stop"
    assert recorded[0].prompt_eval_count == 24
    assert recorded[0].eval_count == 40
