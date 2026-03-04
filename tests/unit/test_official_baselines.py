from __future__ import annotations

from agentic_python_dependency.tools.official_baselines import (
    parse_dockerfile_plan,
    parse_pyego_dependency_json,
)


def test_parse_pyego_dependency_json_extracts_python_packages_and_system_libs() -> None:
    plan = parse_pyego_dependency_json(
        {
            "python_version": "2.7",
            "system_lib": [
                {"name": "libjpeg-dev", "version": "latest", "install_method": "apt"},
                {"name": "brew-only", "version": "latest", "install_method": "brew"},
            ],
            "python_packages": [
                {"name": "PyYAML", "version": "6.0.2", "install_method": "pip"},
                {"name": "requests", "version": "latest", "install_method": "pip"},
            ],
            "message": "",
        }
    )

    assert plan.target_python == "2.7"
    assert [dependency.pin() for dependency in plan.dependencies] == ["PyYAML==6.0.2", "requests"]
    assert plan.system_packages == ["libjpeg-dev"]
    assert plan.implementation == "official"


def test_parse_dockerfile_plan_extracts_dependencies_and_system_packages() -> None:
    plan = parse_dockerfile_plan(
        "\n".join(
            [
                "FROM python:2.7.18",
                "RUN apt-get update && apt-get install -y --no-install-recommends libjpeg-dev pkg-config && rm -rf /var/lib/apt/lists/*",
                "RUN pip install twisted==20.3 cryptography==3.3.2",
                "",
            ]
        )
    )

    assert plan.target_python == "2.7.18"
    assert [dependency.pin() for dependency in plan.dependencies] == [
        "cryptography==3.3.2",
        "twisted==20.3",
    ]
    assert plan.system_packages == ["libjpeg-dev", "pkg-config"]
