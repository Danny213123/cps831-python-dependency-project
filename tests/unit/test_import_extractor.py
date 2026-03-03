from agentic_python_dependency.tools.import_extractor import (
    extract_import_roots_from_code,
    filter_third_party_imports,
    normalize_candidate_packages,
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
