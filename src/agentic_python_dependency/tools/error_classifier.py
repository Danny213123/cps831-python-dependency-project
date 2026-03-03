from __future__ import annotations

import re

from agentic_python_dependency.state import ExecutionOutcome


DEPENDENCY_PATTERNS = [
    (re.compile(r"site-packages/.+SyntaxError", re.IGNORECASE | re.DOTALL), "PackageCompatibilityError"),
    (
        re.compile(
            r"Failed building wheel|Could not build wheels|error: command ['\"](?:gcc|g\+\+)['\"] failed|"
            r"unable to execute ['\"](?:gcc|g\+\+)['\"]|Python\.h: No such file|ffi\.h: No such file|"
            r"fatal error: .*: No such file or directory",
            re.IGNORECASE,
        ),
        "NativeBuildError",
    ),
    (re.compile(r"ModuleNotFoundError"), "ModuleNotFoundError"),
    (re.compile(r"ImportError"), "ImportError"),
    (re.compile(r"No matching distribution found"), "ResolutionError"),
    (re.compile(r"Requires-Python"), "RequiresPythonError"),
]

TERMINAL_PATTERNS = [
    (
        re.compile(
            r"docker run timed out after \d+ seconds.*(Serving Flask app|Running on http://|Uvicorn running on|"
            r"development server|Listening on http)",
            re.IGNORECASE | re.DOTALL,
        ),
        "ServiceTimeoutError",
    ),
    (
        re.compile(
            r"usage:\s+.*(\n|.).*(too few arguments|the following arguments are required|error: too few arguments)",
            re.IGNORECASE,
        ),
        "ArgumentContractError",
    ),
    (
        re.compile(
            r"no display name and no \$DISPLAY|couldn't connect to display|cannot connect to display|_tkinter\.TclError",
            re.IGNORECASE,
        ),
        "DisplayRuntimeError",
    ),
    (re.compile(r"timed out after \d+ seconds", re.IGNORECASE), "TimeoutError"),
    (re.compile(r"SyntaxError"), "SyntaxError"),
    (re.compile(r"NameError"), "NameError"),
    (re.compile(r"AttributeError"), "AttributeError"),
    (re.compile(r"NoCredentialsError|APIError|ConnectionError"), "EnvironmentError"),
]


def classify_error(build_log: str, run_log: str, exit_code: int | None = None) -> ExecutionOutcome:
    combined = "\n".join(part for part in (build_log, run_log) if part)
    if exit_code == 0 and not build_log and not run_log:
        return ExecutionOutcome(
            success=True,
            category="Success",
            message="Execution succeeded.",
            build_succeeded=True,
            run_succeeded=True,
            exit_code=0,
        )

    for pattern, category in DEPENDENCY_PATTERNS:
        if pattern.search(combined):
            return ExecutionOutcome(
                success=False,
                category=category,
                message=combined.strip(),
                build_succeeded="error" not in build_log.lower(),
                run_succeeded=False,
                exit_code=exit_code,
                build_log=build_log,
                run_log=run_log,
                dependency_retryable=True,
            )

    for pattern, category in TERMINAL_PATTERNS:
        if pattern.search(combined):
            return ExecutionOutcome(
                success=False,
                category=category,
                message=combined.strip(),
                build_succeeded="error" not in build_log.lower(),
                run_succeeded=False,
                exit_code=exit_code,
                build_log=build_log,
                run_log=run_log,
                dependency_retryable=False,
            )

    return ExecutionOutcome(
        success=False,
        category="UnknownError",
        message=combined.strip(),
        build_succeeded="error" not in build_log.lower(),
        run_succeeded=False,
        exit_code=exit_code,
        build_log=build_log,
        run_log=run_log,
        dependency_retryable=False,
    )
