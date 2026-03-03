"""PLLM-inspired Python dependency resolver."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("agentic-python-dependency")
except PackageNotFoundError:  # pragma: no cover - editable source tree fallback
    __version__ = "0.1.0"
