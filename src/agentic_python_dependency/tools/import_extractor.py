from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


RUNTIME_ALIASES = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyopenssl",
    "fitz": "pymupdf",
    "dotenv": "python-dotenv",
    "flask_sqlalchemy": "Flask-SQLAlchemy",
    "flask_migrate": "Flask-Migrate",
    "serial": "pyserial",
    "dns": "dnspython",
    "git": "GitPython",
    "OpenGL": "PyOpenGL",
}

IGNORED_DIRS = {".git", ".venv", "venv", "build", "dist", "site-packages", "__pycache__"}
PY2_STDLIB_EXTRAS = {
    "BaseHTTPServer",
    "CGIHTTPServer",
    "ConfigParser",
    "HTMLParser",
    "Queue",
    "SimpleHTTPServer",
    "SocketServer",
    "StringIO",
    "Tkconstants",
    "Tkinter",
    "UserDict",
    "UserList",
    "UserString",
    "__builtin__",
    "commands",
    "cookielib",
    "cPickle",
    "cStringIO",
    "copy_reg",
    "httplib",
    "markupbase",
    "repr",
    "test",
    "thread",
    "ttk",
    "urllib2",
    "urlparse",
    "xmlrpclib",
}
STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", set())) | PY2_STDLIB_EXTRAS
REJECTED_PACKAGES = {"pip", "setuptools", "wheel", "distribute"}
IMPORT_RE = re.compile(r"^\s*import\s+(.+)$")
FROM_RE = re.compile(r"^\s*from\s+([A-Za-z_][\w\.]*)\s+import\s+")
PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]{0,99})$")


def _normalize_name(value: str) -> str:
    return value.strip().replace("-", "_").lower()


ALIASES_BY_NORMALIZED = {_normalize_name(key): value for key, value in RUNTIME_ALIASES.items()}


def runtime_package_alias(value: str) -> str | None:
    return ALIASES_BY_NORMALIZED.get(_normalize_name(value))


def looks_like_package_name(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if not PACKAGE_NAME_RE.fullmatch(candidate):
        return False
    return any(character.isalpha() for character in candidate)


def _extract_import_roots_fallback(code: str) -> list[str]:
    roots: set[str] = set()
    for raw_line in code.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        from_match = FROM_RE.match(line)
        if from_match:
            roots.add(from_match.group(1).split(".", 1)[0])
            continue

        import_match = IMPORT_RE.match(line)
        if not import_match:
            continue

        for segment in import_match.group(1).split(","):
            candidate = segment.strip()
            if not candidate:
                continue
            module = candidate.split(" as ", 1)[0].strip()
            if module:
                roots.add(module.split(".", 1)[0])

    return sorted(roots)


def extract_import_roots_from_code(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _extract_import_roots_fallback(code)

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                roots.add(node.module.split(".", 1)[0])
    return sorted(roots)


def filter_third_party_imports(import_roots: list[str]) -> list[str]:
    return sorted({root for root in import_roots if root and root not in STDLIB_MODULES})


def discover_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def load_python_sources(root: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in discover_python_files(root):
        sources[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    return sources


def normalize_candidate_packages_with_sources(
    packages: list[str],
    extracted_imports: list[str] | None = None,
) -> dict[str, str]:
    allowed: set[str] | None = None
    allowed_sources: dict[str, str] = {}
    if extracted_imports:
        allowed = set()
        for root in extracted_imports:
            normalized_root = _normalize_name(root)
            allowed.add(normalized_root)
            allowed_sources[normalized_root] = "extracted"
            alias = ALIASES_BY_NORMALIZED.get(normalized_root)
            if alias:
                normalized_alias = _normalize_name(alias)
                allowed.add(normalized_alias)
                allowed_sources[normalized_alias] = "alias"

    normalized: dict[str, str] = {}
    sources: dict[str, str] = {}
    for package in packages:
        if not package:
            continue
        package = package.strip()
        if not package:
            continue
        if not looks_like_package_name(package):
            continue
        normalized_package = _normalize_name(package)
        alias = ALIASES_BY_NORMALIZED.get(normalized_package, package)
        if not looks_like_package_name(alias):
            continue
        normalized_alias = _normalize_name(alias)
        if package in STDLIB_MODULES or alias in STDLIB_MODULES:
            continue
        if normalized_package in REJECTED_PACKAGES or normalized_alias in REJECTED_PACKAGES:
            continue
        if allowed is not None and normalized_package not in allowed and normalized_alias not in allowed:
            continue
        normalized[normalized_alias] = alias
        if normalized_alias in allowed_sources:
            sources[normalized_alias] = allowed_sources[normalized_alias]
        elif normalized_package in allowed_sources:
            sources[normalized_alias] = allowed_sources[normalized_package]
        else:
            sources[normalized_alias] = "llm"
    return {normalized[key]: sources[key] for key in sorted(normalized, key=str.lower)}


def normalize_candidate_packages(packages: list[str], extracted_imports: list[str] | None = None) -> list[str]:
    return list(normalize_candidate_packages_with_sources(packages, extracted_imports).keys())
