from agentic_python_dependency.tools.import_extractor import (
    extract_import_roots_from_code,
    filter_third_party_imports,
    looks_like_package_name,
    normalize_candidate_packages,
    normalize_candidate_packages_with_sources,
)


def test_extract_import_roots_filters_relative_imports() -> None:
    code = """
import os
import requests as req
from yaml import safe_load
from .local_module import helper

def run():
    import bs4
"""
    roots = extract_import_roots_from_code(code)
    assert roots == ["bs4", "os", "requests", "yaml"]
    assert filter_third_party_imports(roots) == ["bs4", "requests", "yaml"]


def test_normalize_candidate_packages_applies_alias_map() -> None:
    normalized = normalize_candidate_packages(["yaml", "PIL", "requests"])
    assert normalized == ["Pillow", "PyYAML", "requests"]


def test_normalize_candidate_packages_filters_stdlib_and_unanchored_noise() -> None:
    normalized = normalize_candidate_packages(
        ["yaml", "logging", "pip", "totally-made-up", "Flask-SQLAlchemy"],
        extracted_imports=["yaml", "flask_sqlalchemy"],
    )

    assert normalized == ["Flask-SQLAlchemy", "PyYAML"]


def test_extract_import_roots_falls_back_for_python2_syntax() -> None:
    code = """
import boto
from dateutil.parser import parse

print 'No access key is available.'
"""
    roots = extract_import_roots_from_code(code)

    assert roots == ["boto", "dateutil"]


def test_filter_third_party_imports_excludes_python2_stdlib_extras() -> None:
    roots = ["BaseHTTPServer", "SimpleHTTPServer", "requests"]

    filtered = filter_third_party_imports(roots)

    assert filtered == ["requests"]


def test_normalize_candidate_packages_rejects_prose_and_invalid_names() -> None:
    normalized = normalize_candidate_packages(
        [
            "__main__ function implied because the code suggests a main routine",
            "requests",
            "this-package-name-is-valid",
            "contains/slash",
        ]
    )

    assert normalized == ["requests", "this-package-name-is-valid"]
    assert looks_like_package_name("requests") is True
    assert looks_like_package_name("contains/slash") is False


def test_normalize_candidate_packages_with_sources_tracks_aliases() -> None:
    normalized = normalize_candidate_packages_with_sources(["yaml", "requests"], extracted_imports=["yaml", "requests"])

    assert normalized == {"PyYAML": "alias", "requests": "extracted"}
