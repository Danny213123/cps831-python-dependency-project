from pathlib import Path

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
