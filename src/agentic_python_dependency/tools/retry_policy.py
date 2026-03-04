from __future__ import annotations

from agentic_python_dependency.state import RetryDecision


def classify_retry_decision(
    category: str,
    *,
    system_packages_injected: bool = False,
    native_retry_used: int = 0,
) -> RetryDecision:
    terminal_categories = {
        "RequiresPythonError",
        "SyntaxError",
        "EnvironmentError",
        "LocalModuleMismatch",
        "NameError",
        "ArgumentContractError",
        "DisplayRuntimeError",
        "ServiceTimeoutError",
        "TimeoutError",
        "ConstraintConflictError",
    }
    repair_retryable = {"ModuleNotFoundError", "ImportError", "ResolutionError"}
    candidate_retryable = {"PackageCompatibilityError"}
    if category in terminal_categories:
        return RetryDecision(
            category=category,
            severity="terminal",
            repair_allowed=False,
            candidate_fallback_allowed=False,
            repair_retry_budget=0,
            native_retry_budget=0,
            reason="terminal-category",
        )
    if category in repair_retryable:
        return RetryDecision(
            category=category,
            severity="repair_retryable",
            repair_allowed=True,
            candidate_fallback_allowed=True,
            repair_retry_budget=2,
            native_retry_budget=0,
            reason="repair-may-help",
        )
    if category in candidate_retryable:
        return RetryDecision(
            category=category,
            severity="candidate_retryable",
            repair_allowed=True,
            candidate_fallback_allowed=True,
            repair_retry_budget=1,
            native_retry_budget=0,
            reason="candidate-fallback-first",
        )
    if category == "NativeBuildError":
        return RetryDecision(
            category=category,
            severity="limited_retryable",
            repair_allowed=not system_packages_injected,
            candidate_fallback_allowed=native_retry_used < 1,
            repair_retry_budget=0 if system_packages_injected else 1,
            native_retry_budget=max(0, 1 - native_retry_used),
            reason="native-build-limited-retry",
        )
    return RetryDecision(
        category=category,
        severity="terminal",
        repair_allowed=False,
        candidate_fallback_allowed=False,
        repair_retry_budget=0,
        native_retry_budget=0,
        reason="fallback-terminal",
    )
