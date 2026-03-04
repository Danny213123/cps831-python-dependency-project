from pathlib import Path
import io
import json

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
    runner = OllamaPromptRunner(settings, settings.prompts_dir)
    payload = io.BytesIO(json.dumps({"response": "direct-response"}).encode("utf-8"))
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
