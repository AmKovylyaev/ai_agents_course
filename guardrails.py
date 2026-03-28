"""Guardrail functions that validate pipeline state, outputs, and generated code."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import config as cfg

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class GuardrailResult:
    """Outcome of a single guardrail check."""

    __slots__ = ("passed", "message")

    def __init__(self, passed: bool, message: str = ""):
        self.passed = passed
        self.message = message

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"GuardrailResult({status}, {self.message!r})"


def _log(result: GuardrailResult, name: str) -> None:
    if not cfg.logger:
        return
    if result.passed:
        cfg.logger.info("[Guardrail %s] PASS%s", name, f": {result.message}" if result.message else "")
    else:
        cfg.logger.warning("[Guardrail %s] FAIL: %s", name, result.message)


# ---------------------------------------------------------------------------
# 1. Required state fields
# ---------------------------------------------------------------------------

REQUIRED_STATE_KEYS: dict[str, list[str]] = {
    "step1_eda": [
        "target_column", "numeric_columns", "categorical_columns",
        "n_classes", "train_shape", "task_type",
    ],
    "step2_train": ["model_path"],
    "step3_eval": ["local_metrics"],
    "step4_submission": ["submission_path"],
    "step5_submit": ["submit_ok"],
    "step7_report": ["report_path"],
}


def check_required_state(step_name: str, state: dict[str, Any]) -> GuardrailResult:
    """Verify that all required state keys are populated after a step."""
    required = REQUIRED_STATE_KEYS.get(step_name, [])
    if not required:
        return GuardrailResult(True, f"No required keys defined for {step_name}")

    missing: list[str] = []
    empty: list[str] = []
    for key in required:
        if key not in state:
            missing.append(key)
        elif state[key] is None or state[key] == "" or state[key] == [] or state[key] == {}:
            empty.append(key)

    problems: list[str] = []
    if missing:
        problems.append(f"missing keys: {missing}")
    if empty:
        problems.append(f"empty keys: {empty}")

    if problems:
        result = GuardrailResult(False, f"After {step_name}: {'; '.join(problems)}")
    else:
        result = GuardrailResult(True, f"{len(required)} keys present")
    _log(result, "required_state")
    return result


# ---------------------------------------------------------------------------
# 2. Submission format
# ---------------------------------------------------------------------------

def check_submission_format(state: dict[str, Any]) -> GuardrailResult:
    """Validate that submission CSV matches the sample submission format."""
    try:
        import pandas as pd
    except ImportError:
        return GuardrailResult(False, "pandas not available")

    sub_path = state.get("submission_path", "")
    sample_path = state.get("sample_submission_path", "")

    if not sub_path or not Path(sub_path).exists():
        result = GuardrailResult(False, f"Submission file not found: {sub_path}")
        _log(result, "submission_format")
        return result

    if not sample_path or not Path(sample_path).exists():
        result = GuardrailResult(False, f"Sample submission not found: {sample_path}")
        _log(result, "submission_format")
        return result

    sub_df = pd.read_csv(sub_path)
    sample_df = pd.read_csv(sample_path)

    problems: list[str] = []

    if list(sub_df.columns) != list(sample_df.columns):
        problems.append(
            f"Column mismatch: got {list(sub_df.columns)}, expected {list(sample_df.columns)}"
        )

    if len(sub_df) != len(sample_df):
        problems.append(
            f"Row count mismatch: got {len(sub_df)}, expected {len(sample_df)}"
        )

    pred_col = sample_df.columns[-1]
    if pred_col in sub_df.columns and sub_df[pred_col].isna().any():
        nan_count = int(sub_df[pred_col].isna().sum())
        problems.append(f"Prediction column '{pred_col}' has {nan_count} NaN values")

    if problems:
        result = GuardrailResult(False, "; ".join(problems))
    else:
        result = GuardrailResult(
            True,
            f"{len(sub_df)} rows, {len(sub_df.columns)} cols — matches sample",
        )
    _log(result, "submission_format")
    return result


# ---------------------------------------------------------------------------
# 3. Model file exists and is loadable
# ---------------------------------------------------------------------------

def check_model_exists(state: dict[str, Any]) -> GuardrailResult:
    """Verify the saved pipeline file exists and can be loaded."""
    model_path = state.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        result = GuardrailResult(False, f"Model file not found: {model_path}")
        _log(result, "model_exists")
        return result

    try:
        import joblib
        pipeline = joblib.load(model_path)
        has_predict = hasattr(pipeline, "predict")
        if not has_predict:
            result = GuardrailResult(False, "Loaded object has no predict() method")
        else:
            result = GuardrailResult(True, f"Pipeline loaded ({type(pipeline).__name__})")
    except Exception as e:
        result = GuardrailResult(False, f"Failed to load model: {e}")

    _log(result, "model_exists")
    return result


# ---------------------------------------------------------------------------
# 4. Metrics sanity
# ---------------------------------------------------------------------------

CLASSIFICATION_METRIC_KEYS = {"accuracy", "precision", "recall", "f1"}
REGRESSION_METRIC_KEYS = {"rmse", "mae", "r2"}
RANDOM_BASELINE_THRESHOLD = 0.05


def check_metrics_sanity(state: dict[str, Any]) -> GuardrailResult:
    """Verify local metrics are present and valid (task-type aware)."""
    metrics = state.get("local_metrics", {})
    if not metrics:
        result = GuardrailResult(False, "local_metrics is empty or missing")
        _log(result, "metrics_sanity")
        return result

    task_type = state.get("task_type", "classification")
    is_regression = task_type == "regression"
    expected_keys = REGRESSION_METRIC_KEYS if is_regression else CLASSIFICATION_METRIC_KEYS

    problems: list[str] = []

    missing_keys = expected_keys - set(metrics.keys())
    if missing_keys:
        problems.append(f"Missing metric keys for {task_type}: {missing_keys}")

    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            problems.append(f"{key} is not numeric: {value!r}")
            continue

    if is_regression:
        rmse = metrics.get("rmse")
        if isinstance(rmse, (int, float)) and rmse < 0:
            problems.append(f"rmse={rmse:.4f} is negative (invalid)")
        mae = metrics.get("mae")
        if isinstance(mae, (int, float)) and mae < 0:
            problems.append(f"mae={mae:.4f} is negative (invalid)")
        r2 = metrics.get("r2")
        if isinstance(r2, (int, float)) and r2 < -1.0:
            problems.append(f"r2={r2:.4f} is suspiciously low")
    else:
        for key in ("accuracy", "precision", "recall", "f1"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not (0.0 <= value <= 1.0):
                problems.append(f"{key}={value:.4f} out of [0, 1] range")

        n_classes = state.get("n_classes")
        if n_classes and isinstance(n_classes, (int, float)) and n_classes > 0:
            random_acc = 1.0 / n_classes
            acc = metrics.get("accuracy")
            if isinstance(acc, (int, float)) and acc < random_acc - RANDOM_BASELINE_THRESHOLD:
                problems.append(
                    f"accuracy={acc:.4f} is below random baseline ({random_acc:.4f}) for {n_classes} classes"
                )

    if problems:
        result = GuardrailResult(False, "; ".join(problems))
    else:
        summary = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
        )
        result = GuardrailResult(True, f"[{task_type}] {summary}")
    _log(result, "metrics_sanity")
    return result


# ---------------------------------------------------------------------------
# 5. No data leakage (target not in feature columns)
# ---------------------------------------------------------------------------

def check_no_data_leakage(state: dict[str, Any]) -> GuardrailResult:
    """Verify target column is not listed among feature columns."""
    target = state.get("target_column", "")
    if not target:
        result = GuardrailResult(False, "target_column not set")
        _log(result, "no_data_leakage")
        return result

    problems: list[str] = []
    numeric = state.get("numeric_columns", [])
    categorical = state.get("categorical_columns", [])

    if target in numeric:
        problems.append(f"target '{target}' found in numeric_columns")
    if target in categorical:
        problems.append(f"target '{target}' found in categorical_columns")

    if problems:
        result = GuardrailResult(False, "; ".join(problems))
    else:
        result = GuardrailResult(
            True,
            f"target '{target}' absent from {len(numeric)} numeric + {len(categorical)} categorical cols",
        )
    _log(result, "no_data_leakage")
    return result


# ---------------------------------------------------------------------------
# 6. Code safety
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\bos\.system\s*\(", "os.system() call"),
    (r"\bsubprocess\.(run|call|Popen|check_output)\s*\(", "subprocess call"),
    (r"\bshutil\.rmtree\s*\(", "shutil.rmtree() call"),
    (r"\bos\.remove\s*\(", "os.remove() call"),
    (r"\bos\.unlink\s*\(", "os.unlink() call"),
    (r"\bos\.rmdir\s*\(", "os.rmdir() call"),
    (r"\b(eval|exec)\s*\(", "eval()/exec() call"),
    (r"\brequests\.(get|post|put|delete|patch)\s*\(", "HTTP request"),
    (r"\burllib\.request\.", "urllib network call"),
    (r"\bsocket\.", "raw socket usage"),
    (r"\b__import__\s*\(", "dynamic import via __import__()"),
]


def check_code_safety(code: str) -> GuardrailResult:
    """Reject generated code containing dangerous operations."""
    violations: list[str] = []
    for pattern, description in _DANGEROUS_PATTERNS:
        matches = re.findall(pattern, code)
        if matches:
            violations.append(description)

    if violations:
        result = GuardrailResult(False, f"Unsafe code detected: {', '.join(violations)}")
    else:
        result = GuardrailResult(True, "No dangerous patterns found")
    _log(result, "code_safety")
    return result


# ---------------------------------------------------------------------------
# Step-specific guardrail mapping
# ---------------------------------------------------------------------------

_STEP_GUARDRAILS: dict[str, list[str]] = {
    "step1_eda": ["required_state", "no_data_leakage"],
    "step2_train": ["required_state", "model_exists"],
    "step3_eval": ["required_state", "metrics_sanity"],
    "step4_submission": ["required_state", "submission_format"],
    "step5_submit": ["required_state"],
    "step7_report": ["required_state"],
}

_GUARDRAIL_FUNCS: dict[str, Any] = {
    "required_state": check_required_state,
    "no_data_leakage": check_no_data_leakage,
    "model_exists": check_model_exists,
    "metrics_sanity": check_metrics_sanity,
    "submission_format": check_submission_format,
}


def run_step_guardrails(step_name: str, state: dict[str, Any]) -> list[GuardrailResult]:
    """Run only the guardrails relevant to *step_name*. Returns list of results."""
    names = _STEP_GUARDRAILS.get(step_name, [])
    results: list[GuardrailResult] = []
    for name in names:
        fn = _GUARDRAIL_FUNCS[name]
        if name == "required_state":
            results.append(fn(step_name, state))
        else:
            results.append(fn(state))
    return results


def get_guardrail_names_for_step(step_name: str) -> list[str]:
    """Return the guardrail IDs relevant to a step (used to scope verifier tools)."""
    return list(_STEP_GUARDRAILS.get(step_name, []))