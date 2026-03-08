from __future__ import annotations

import re

from agentic_python_dependency.state import ExecutionOutcome


COMPATIBILITY_PATTERNS = [
    (
        re.compile(r"site-packages/.+SyntaxError", re.IGNORECASE | re.DOTALL),
        "PackageCompatibilityError",
        True,
    ),
    (
        re.compile(
            r"configparser(?:\.|['\"].*SafeConfigParser)|"
            r"inspect\.getargspec|"
            r"collections\.(?:MutableSet|MutableMapping|MutableSequence)\b",
            re.IGNORECASE,
        ),
        "PackageCompatibilityError",
        True,
    ),
    (
        re.compile(
            r"cannot import name ['\"]?(?:Set|Mapping|MutableMapping|Sequence|Iterable)['\"]? from ['\"]collections['\"]|"
            r"module ['\"]numpy['\"] has no attribute ['\"](?:float|int|bool|object)['\"]|"
            r"No module named ['\"]?(?:urllib2|cookielib|django\.test\.simple|sklearn\.cross_validation|typing)['\"]?|"
            r"Missing parentheses in call to ['\"]print['\"]|"
            r"build_py_2to3|"
            r"Package ['\"].+['\"] requires a different Python:",
            re.IGNORECASE,
        ),
        "PackageCompatibilityError",
        True,
    ),
]

TF_API_DRIFT_PATTERNS = [
    (
        re.compile(
            r"module ['\"]tensorflow['\"] has no attribute ['\"](?:placeholder|flags)['\"]|"
            r"No module named ['\"]tensorflow\.(?:examples|contrib)['\"]",
            re.IGNORECASE,
        ),
        "PackageCompatibilityError",
        True,
    ),
]

NATIVE_BUILD_PATTERNS = [
    (
        re.compile(
            r"Failed building wheel|"
            r"Could not build wheels|"
            r"error: command ['\"](?:gcc|g\+\+)['\"] failed|"
            r"unable to execute ['\"](?:gcc|g\+\+)['\"]|"
            r"Python\.h: No such file|"
            r"ffi\.h: No such file|"
            r"fatal error: .*: No such file or directory|"
            r"sdl-config: not found|"
            r"Unable to run \"sdl-config\"|"
            r"pkg-config freetype2\" failed|"
            r"Unknown compiler\(s\)|"
            r"BackendUnavailable|"
            r"setuptools\.build_meta|"
            r"metadata-generation-failed|"
            r"meson\.build:.*ERROR|"
            r"subprocess-exited-with-error|"
            r"pg_config executable not found|"
            r"The headers or library files could not be found for zlib|"
            r"Please make sure the libxml2 and libxslt development packages are installed|"
            r"No module named ['\"]?Cython\.Build['\"]?|"
            r"Cython\.Compiler\.Errors\.CompileError",
            re.IGNORECASE,
        ),
        "NativeBuildError",
        True,
    ),
]

SYSTEM_DEPENDENCY_PATTERNS = [
    (
        re.compile(
            r"libxcb\.so\.1|"
            r"Exempi library not found|"
            r"libexempi|"
            r"cannot open shared object file|"
            r"gobject-introspection.*not found|"
            r"does not have a Release file|"
            r"Unable to locate package|"
            r"\bapt-get update\b.*404\s+Not Found",
            re.IGNORECASE,
        ),
        "SystemDependencyError",
        True,
    ),
]

BUILD_PATTERNS = [
    (re.compile(r"docker build timed out after \d+ seconds", re.IGNORECASE), "BuildTimeoutError", True),
    *COMPATIBILITY_PATTERNS,
    *NATIVE_BUILD_PATTERNS,
    *SYSTEM_DEPENDENCY_PATTERNS,
    (
        re.compile(
            r"ResolutionImpossible|"
            r"package versions have conflicting dependencies|"
            r"The conflict is caused by:",
            re.IGNORECASE,
        ),
        "ResolutionError",
        True,
    ),
    (re.compile(r"No matching distribution found", re.IGNORECASE), "ResolutionError", True),
    (re.compile(r"Requires-Python", re.IGNORECASE), "RequiresPythonError", False),
    (re.compile(r"ModuleNotFoundError", re.IGNORECASE), "ModuleNotFoundError", True),
    (re.compile(r"ImportError", re.IGNORECASE), "ImportError", True),
]

RUN_RETRYABLE_PATTERNS = [
    *TF_API_DRIFT_PATTERNS,
    *COMPATIBILITY_PATTERNS,
    *SYSTEM_DEPENDENCY_PATTERNS,
    (re.compile(r"ModuleNotFoundError", re.IGNORECASE), "ModuleNotFoundError", True),
    (re.compile(r"ImportError", re.IGNORECASE), "ImportError", True),
]

RUN_TERMINAL_PATTERNS = [
    (
        re.compile(
            r"docker run timed out after \d+ seconds.*(Serving Flask app|Running on http://|Uvicorn running on|"
            r"development server|Listening on http)",
            re.IGNORECASE | re.DOTALL,
        ),
        "ServiceTimeoutError",
        False,
    ),
    (
        re.compile(
            r"usage:\s+.*(\n|.).*(too few arguments|the following arguments are required|error: too few arguments)",
            re.IGNORECASE,
        ),
        "ArgumentContractError",
        False,
    ),
    (
        re.compile(
            r"no display name and no \$DISPLAY|couldn't connect to display|cannot connect to display|_tkinter\.TclError",
            re.IGNORECASE,
        ),
        "DisplayRuntimeError",
        False,
    ),
    (
        re.compile(
            r"ServerSelectionTimeoutError|Connection refused|"
            r"ConnectionError|NoCredentialsError|APIError|urllib2?\.\w+Error|HTTPError|"
            r"DatabaseError|OperationalError",
            re.IGNORECASE,
        ),
        "EnvironmentError",
        False,
    ),
    (
        re.compile(r"AttributeError:\s*'module'\s*object has no attribute", re.IGNORECASE),
        "LocalModuleMismatch",
        False,
    ),
    (re.compile(r"timed out after \d+ seconds", re.IGNORECASE), "TimeoutError", False),
    (re.compile(r"SyntaxError", re.IGNORECASE), "SyntaxError", False),
    (re.compile(r"NameError", re.IGNORECASE), "NameError", False),
    (re.compile(r"AttributeError", re.IGNORECASE), "AttributeError", False),
    (re.compile(r"ValueError", re.IGNORECASE), "ValueError", False),
    (re.compile(r"IndexError", re.IGNORECASE), "IndexError", False),
    (re.compile(r"OSError|IOError|FileNotFoundError", re.IGNORECASE), "EnvironmentError", False),
    (re.compile(r"TypeError", re.IGNORECASE), "TypeError", False),
    (re.compile(r"struct\.error", re.IGNORECASE), "RuntimeError", False),
    (re.compile(r"KeyError", re.IGNORECASE), "KeyError", False),
    (re.compile(r"BadZipfile|BadZipFile", re.IGNORECASE), "RuntimeError", False),
]


def _default_build_success(build_log: str) -> bool:
    lowered = build_log.lower()
    return "error" not in lowered and "failed to solve" not in lowered


def _match_category(
    text: str,
    patterns: list[tuple[re.Pattern[str], str, bool]],
) -> tuple[str, bool] | None:
    for pattern, category, dependency_retryable in patterns:
        if pattern.search(text):
            return category, dependency_retryable
    return None


def _build_outcome(
    *,
    category: str,
    message: str,
    dependency_retryable: bool,
    build_succeeded: bool,
    run_succeeded: bool,
    exit_code: int | None,
    build_log: str,
    run_log: str,
    classifier_origin: str,
) -> ExecutionOutcome:
    return ExecutionOutcome(
        success=False,
        category=category,
        message=message.strip(),
        build_succeeded=build_succeeded,
        run_succeeded=run_succeeded,
        exit_code=exit_code,
        build_log=build_log,
        run_log=run_log,
        dependency_retryable=dependency_retryable,
        classifier_origin=classifier_origin,
    )


def classify_error(
    build_log: str,
    run_log: str,
    exit_code: int | None = None,
    *,
    build_succeeded: bool | None = None,
    run_succeeded: bool | None = None,
) -> ExecutionOutcome:
    if build_succeeded is not None:
        inferred_build_succeeded = build_succeeded
    elif build_log.strip() and (exit_code not in (None, 0)) and not run_log.strip():
        inferred_build_succeeded = False
    else:
        inferred_build_succeeded = _default_build_success(build_log)
    inferred_run_succeeded = run_succeeded if run_succeeded is not None else (inferred_build_succeeded and exit_code == 0)
    build_text = build_log.strip()
    run_text = run_log.strip()

    if inferred_build_succeeded and (inferred_run_succeeded or not run_text) and exit_code == 0:
        return ExecutionOutcome(
            success=True,
            category="Success",
            message="Execution succeeded.",
            build_succeeded=True,
            run_succeeded=True,
            exit_code=0,
            classifier_origin="none",
        )

    if not inferred_build_succeeded and build_text:
        matched = _match_category(build_text, BUILD_PATTERNS)
        if matched is not None:
            category, dependency_retryable = matched
            return _build_outcome(
                category=category,
                message=build_text,
                dependency_retryable=dependency_retryable,
                build_succeeded=False,
                run_succeeded=False,
                exit_code=exit_code,
                build_log=build_log,
                run_log=run_log,
                classifier_origin="build",
            )
        return _build_outcome(
            category="UnknownError",
            message=build_text,
            dependency_retryable=False,
            build_succeeded=False,
            run_succeeded=False,
            exit_code=exit_code,
            build_log=build_log,
            run_log=run_log,
            classifier_origin="build",
        )

    if inferred_build_succeeded and not inferred_run_succeeded:
        matched = _match_category(run_text, RUN_RETRYABLE_PATTERNS)
        if matched is not None:
            category, dependency_retryable = matched
            return _build_outcome(
                category=category,
                message=run_text,
                dependency_retryable=dependency_retryable,
                build_succeeded=True,
                run_succeeded=False,
                exit_code=exit_code,
                build_log=build_log,
                run_log=run_log,
                classifier_origin="run",
            )
        matched = _match_category(run_text, RUN_TERMINAL_PATTERNS)
        if matched is not None:
            category, dependency_retryable = matched
            return _build_outcome(
                category=category,
                message=run_text,
                dependency_retryable=dependency_retryable,
                build_succeeded=True,
                run_succeeded=False,
                exit_code=exit_code,
                build_log=build_log,
                run_log=run_log,
                classifier_origin="run",
            )
        return _build_outcome(
            category="UnknownError",
            message=run_text,
            dependency_retryable=False,
            build_succeeded=True,
            run_succeeded=False,
            exit_code=exit_code,
            build_log=build_log,
            run_log=run_log,
            classifier_origin="run",
        )

    combined = "\n".join(part for part in (build_text, run_text) if part)
    return _build_outcome(
        category="UnknownError",
        message=combined,
        dependency_retryable=False,
        build_succeeded=inferred_build_succeeded,
        run_succeeded=inferred_run_succeeded,
        exit_code=exit_code,
        build_log=build_log,
        run_log=run_log,
        classifier_origin="combined",
    )
