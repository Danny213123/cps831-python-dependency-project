from agentic_python_dependency.tools.error_classifier import classify_error


def test_classify_error_marks_dependency_failures_as_retryable() -> None:
    outcome = classify_error("ERROR: No matching distribution found for foo", "", 1)

    assert outcome.category == "ResolutionError"
    assert outcome.dependency_retryable is True


def test_classify_error_marks_syntax_error_as_terminal() -> None:
    outcome = classify_error("", "SyntaxError: invalid syntax", 1)

    assert outcome.category == "SyntaxError"
    assert outcome.dependency_retryable is False


def test_classify_error_marks_timeout_as_terminal() -> None:
    outcome = classify_error("", "docker run timed out after 60 seconds.", 124)

    assert outcome.category == "TimeoutError"
    assert outcome.dependency_retryable is False


def test_classify_error_detects_argument_contract_failure() -> None:
    outcome = classify_error("", "usage: snippet.py [-h]\nsnippet.py: error: too few arguments", 2)

    assert outcome.category == "ArgumentContractError"
    assert outcome.dependency_retryable is False


def test_classify_error_detects_display_runtime_failure() -> None:
    outcome = classify_error("", "_tkinter.TclError: no display name and no $DISPLAY environment variable", 1)

    assert outcome.category == "DisplayRuntimeError"
    assert outcome.dependency_retryable is False


def test_classify_error_detects_service_timeout_failure() -> None:
    outcome = classify_error(
        "",
        "docker run timed out after 60 seconds.\nServing Flask app \"snippet\"\nRunning on http://127.0.0.1:5000/",
        124,
    )

    assert outcome.category == "ServiceTimeoutError"
    assert outcome.dependency_retryable is False


def test_classify_error_detects_native_build_failure_as_retryable() -> None:
    outcome = classify_error("error: command 'gcc' failed with exit status 1", "", 1)

    assert outcome.category == "NativeBuildError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_sdl_native_build_failure_as_retryable() -> None:
    outcome = classify_error(
        "",
        'Unable to run "sdl-config". Please make sure a development version of SDL is installed.',
        1,
    )

    assert outcome.category == "NativeBuildError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_package_compatibility_failure_as_retryable() -> None:
    outcome = classify_error("", 'File "/usr/local/lib/python2.7/site-packages/emoji/core.py", line 25\nSyntaxError', 1)

    assert outcome.category == "PackageCompatibilityError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_local_module_mismatch_as_terminal() -> None:
    outcome = classify_error("", "AttributeError: 'module' object has no attribute 'compress'", 1)

    assert outcome.category == "LocalModuleMismatch"
    assert outcome.dependency_retryable is False


def test_classify_error_detects_value_error_as_terminal() -> None:
    outcome = classify_error("", "ValueError: invalid literal for int()", 1)

    assert outcome.category == "ValueError"
    assert outcome.dependency_retryable is False


def test_classify_error_detects_oserror_family_as_environment_error() -> None:
    outcome = classify_error("", "FileNotFoundError: [Errno 2] No such file or directory", 1)

    assert outcome.category == "EnvironmentError"
    assert outcome.dependency_retryable is False
