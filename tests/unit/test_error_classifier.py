from agentic_python_dependency.tools.error_classifier import classify_error


def test_classify_error_marks_dependency_failures_as_retryable() -> None:
    outcome = classify_error("ERROR: No matching distribution found for foo", "", 1)

    assert outcome.category == "ResolutionError"
    assert outcome.dependency_retryable is True
    assert outcome.classifier_origin == "build"


def test_classify_error_marks_syntax_error_as_terminal() -> None:
    outcome = classify_error("", "SyntaxError: invalid syntax", 1)

    assert outcome.category == "SyntaxError"
    assert outcome.dependency_retryable is False
    assert outcome.classifier_origin == "run"


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
        'Unable to run "sdl-config". Please make sure a development version of SDL is installed.',
        "",
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


def test_classify_error_detects_system_library_failure_as_retryable() -> None:
    outcome = classify_error("", "ImportError: libxcb.so.1: cannot open shared object file", 1)

    assert outcome.category == "SystemDependencyError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_legacy_runtime_breakage_as_package_compatibility() -> None:
    outcome = classify_error("", "ImportError: cannot import name 'Set' from 'collections'", 1)

    assert outcome.category == "PackageCompatibilityError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_safe_config_parser_build_break_as_package_compatibility() -> None:
    outcome = classify_error(
        "AttributeError: module 'configparser' has no attribute 'SafeConfigParser'",
        "",
        1,
    )

    assert outcome.category == "PackageCompatibilityError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_apt_repository_failure_as_system_dependency() -> None:
    outcome = classify_error(
        "E: The repository 'http://deb.debian.org/debian buster Release' does not have a Release file.",
        "",
        100,
    )

    assert outcome.category == "SystemDependencyError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_install_time_python_mismatch_as_package_compatibility() -> None:
    outcome = classify_error(
        "ERROR: Package 'pymc3' requires a different Python: 2.7.18 not in '>=3.5.4'",
        "",
        1,
    )

    assert outcome.category == "PackageCompatibilityError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_docker_build_timeout_as_retryable_build_timeout() -> None:
    outcome = classify_error(
        "docker build timed out after 300 seconds.\n\n#15 224.5   Building wheel for pandas (setup.py): still running...",
        "",
        124,
    )

    assert outcome.category == "BuildTimeoutError"
    assert outcome.dependency_retryable is True


def test_classify_error_prefers_run_log_for_tensorflow_api_drift_after_successful_build() -> None:
    outcome = classify_error(
        "ImportError: libxcb.so.1: cannot open shared object file",
        "ModuleNotFoundError: No module named 'tensorflow.contrib'",
        1,
        build_succeeded=True,
        run_succeeded=False,
    )

    assert outcome.category == "PackageCompatibilityError"
    assert outcome.dependency_retryable is True
    assert outcome.classifier_origin == "run"


def test_classify_error_detects_tensorflow_attribute_api_drift() -> None:
    outcome = classify_error(
        "",
        "AttributeError: module 'tensorflow' has no attribute 'placeholder'",
        1,
        build_succeeded=True,
        run_succeeded=False,
    )

    assert outcome.category == "PackageCompatibilityError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_pg_config_build_failure() -> None:
    outcome = classify_error("Error: pg_config executable not found.", "", 1)

    assert outcome.category == "NativeBuildError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_zlib_header_build_failure() -> None:
    outcome = classify_error("The headers or library files could not be found for zlib,", "", 1)

    assert outcome.category == "NativeBuildError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_lxml_header_build_failure() -> None:
    outcome = classify_error(
        "Error: Please make sure the libxml2 and libxslt development packages are installed.",
        "",
        1,
    )

    assert outcome.category == "NativeBuildError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_pip_resolution_impossible_as_resolution_error() -> None:
    outcome = classify_error(
        "ERROR: Cannot install foo and bar because these package versions have conflicting dependencies.\n"
        "ERROR: ResolutionImpossible",
        "",
        1,
    )

    assert outcome.category == "ResolutionError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_missing_typing_backport_as_compatibility_issue() -> None:
    outcome = classify_error("ImportError: No module named typing", "", 1)

    assert outcome.category == "PackageCompatibilityError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_legacy_cython_build_breakage() -> None:
    outcome = classify_error("ImportError: No module named Cython.Build", "", 1)

    assert outcome.category == "NativeBuildError"
    assert outcome.dependency_retryable is True


def test_classify_error_detects_service_connection_refusal_as_environment_error() -> None:
    outcome = classify_error(
        "",
        "pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused",
        1,
        build_succeeded=True,
        run_succeeded=False,
    )

    assert outcome.category == "EnvironmentError"
    assert outcome.dependency_retryable is False


def test_classify_error_detects_illegal_instruction_as_environment_error() -> None:
    outcome = classify_error(
        "",
        "Illegal instruction",
        132,
        build_succeeded=True,
        run_succeeded=False,
    )

    assert outcome.category == "EnvironmentError"
    assert outcome.dependency_retryable is False
