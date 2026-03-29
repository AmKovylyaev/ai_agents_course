"""LLM-driven agent steps (with automatic fallback on failure)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import config as cfg
from config import log as _log
from executor import create_step_chain, run_step_with_retry
from mini_feedback_loop import mini_feedback_loop
from prompts import (
    STEP1_PLANNER_PROMPT,
    STEP1_EDA_PROMPT,
    STEP1_VERIFIER_PROMPT,
    STEP2_PLANNER_PROMPT,
    STEP2_TRAIN_PROMPT,
    STEP2_VERIFIER_PROMPT,
    STEP3_PLANNER_PROMPT,
    STEP3_EVAL_PROMPT,
    STEP3_VERIFIER_PROMPT,
    STEP4_PLANNER_PROMPT,
    STEP4_SUBMISSION_PROMPT,
    STEP4_VERIFIER_PROMPT,
    STEP7_REPORT_PROMPT,
    STEP_JUDGE_PROMPT,
)
from rag.rag_tools import inject_rag_context_into_state
from steps_fallback import (
    step1_eda_fallback,
    step2_train_fallback,
    step3_local_eval_fallback,
    step4_submission_fallback,
    step7_report_fallback,
    step_judge_fallback,
)
from tools.web_context import inject_web_context_into_state


DEFAULT_RAG_QUERY_MODE = "hybrid_code"


def _run_feedback_step(
    step_name: str,
    planner_prompt: str,
    coder_prompt: str,
    verifier_prompt: str,
    fallback_fn,
    state: dict,
    max_attempts: int = 3,
    timeout_sec: int = 120,
) -> dict:
    """Generic wrapper: try LLM agents, fall back on failure."""
    state = dict(state)
    temp = state.get("temperature", 0.0)
    llm = cfg.get_llm(temperature=temp)

    if not llm:
        _log("%s — no LLM, using fallback", step_name, level="warning")
        return fallback_fn(state)

    state, success = mini_feedback_loop(
        step_name,
        planner_prompt,
        coder_prompt,
        verifier_prompt,
        state,
        llm,
        max_attempts=max_attempts,
        timeout_sec=timeout_sec,
    )

    if not success:
        _log("%s — agents failed, using fallback", step_name, level="warning")
        return fallback_fn(state)
    return state


def _get_rag_query_mode(state: dict) -> str:
    mode = str(state.get("rag_query_mode", DEFAULT_RAG_QUERY_MODE) or DEFAULT_RAG_QUERY_MODE).strip().lower()
    if mode not in {"text", "code", "hybrid_code"}:
        mode = DEFAULT_RAG_QUERY_MODE
    return mode


def _sanitize_model_type(model_type: str) -> str:
    model_type = (model_type or "catboost").strip()
    return model_type or "catboost"


def _build_eda_code_sketch(state: dict) -> str:
    return '''import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_column = train_df.columns[-1]
feature_df = train_df.drop(columns=[target_column])

numeric_columns = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_columns = [c for c in feature_df.columns if c not in numeric_columns]

missing_values = train_df.isna().sum().to_dict()
n_classes = train_df[target_column].nunique(dropna=False)
unique_ratio = float(n_classes) / max(len(train_df), 1)
if pd.api.types.is_numeric_dtype(train_df[target_column]) and (n_classes > 20 or unique_ratio > 0.05):
    task_type = "regression"
else:
    task_type = "classification"

# save eda summary and only the required figures
train_df.describe(include="all")
'''


def _build_train_code_sketch(state: dict) -> str:
    task_type = state.get("task_type", "classification") or "classification"
    model_type = _sanitize_model_type(str(state.get("model_type", "catboost"))).lower()
    if model_type == "xgboost":
        model_ctor = "XGBRegressor(objective='reg:squarederror')" if task_type == "regression" else "XGBClassifier(eval_metric='logloss')"
    elif model_type == "lightgbm":
        model_ctor = "LGBMRegressor(objective='mse')" if task_type == "regression" else "LGBMClassifier()"
    else:
        model_ctor = "CatBoostRegressor(verbose=False)" if task_type == "regression" else "CatBoostClassifier(verbose=False)"
    numeric_columns = repr(state.get("numeric_columns", []))
    categorical_columns = repr(state.get("categorical_columns", []))
    target_column = state.get("target_column", "target")
    test_size = 1.0 - float(state.get("train_sample_frac", 0.8))
    return f'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train_df = pd.read_csv(train_path)
target_col = {target_column!r}
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

numeric_features = [c for c in {numeric_columns} if c in X.columns]
categorical_features = [c for c in {categorical_columns} if c in X.columns]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_features),
    ("cat", cat_pipe, categorical_features),
], remainder="drop")
model = {model_ctor}
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size={test_size:.3f}, random_state=42
)
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_val)
'''


def _build_eval_code_sketch(state: dict) -> str:
    task_type = state.get("task_type", "classification") or "classification"
    target_column = state.get("target_column", "target")
    test_size = 1.0 - float(state.get("train_sample_frac", 0.8))
    metric_block = (
        'metrics = {"mse": mean_squared_error(y_val_out, preds_out), "mae": mean_absolute_error(y_val_out, preds_out), "r2": r2_score(y_val_out, preds_out)}'
        if task_type == "regression"
        else 'metrics = {"accuracy": accuracy_score(y_val, preds), "precision": precision_score(y_val, preds, average="macro", zero_division=0), "recall": recall_score(y_val, preds, average="macro", zero_division=0), "f1": f1_score(y_val, preds, average="macro", zero_division=0)}'
    )
    return f'''import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(train_path)
pipeline = joblib.load(model_path)
target_col = {target_column!r}
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

y_for_split = np.log1p(y) if target_transform == "log1p" else y
X_train, X_val, y_train, y_val = train_test_split(
    X, y_for_split, test_size={test_size:.3f}, random_state=42
)
train_preds = pipeline.predict(X_train)
preds = pipeline.predict(X_val)

# inverse-transform if needed and compute metrics on original scale
{metric_block}
with open(local_metrics_path, "w", encoding="utf-8") as f:
    json.dump({{"train": {{}}, "val": metrics}}, f, ensure_ascii=False, indent=2)
'''


def _build_submission_code_sketch(state: dict) -> str:
    target_column = state.get("target_column", "target")
    return f'''import joblib
import numpy as np
import pandas as pd

sample_submission = pd.read_csv(sample_submission_path)
test_df = pd.read_csv(test_path)
model = joblib.load(model_path)

target_col = {target_column!r}
if target_col in test_df.columns:
    test_df = test_df.drop(columns=[target_col])

preds = model.predict(test_df)
if target_transform == "log1p":
    preds = np.expm1(preds)

submission = sample_submission.copy()
submission.iloc[:, -1] = preds
submission.to_csv(submission_path, index=False)
'''


def _build_rag_query(intent_query: str, code_sketch: str, query_mode: str) -> str:
    if query_mode == "text":
        return intent_query
    if query_mode == "code":
        return code_sketch
    return f"Intent:\n{intent_query}\n\nCode sketch:\n{code_sketch}"


def _inject_rag_with_mode(state: dict, *, intent_query: str, code_sketch: str, log_label: str) -> dict:
    state = dict(state)
    query_mode = _get_rag_query_mode(state)
    rag_query = _build_rag_query(intent_query, code_sketch, query_mode)

    state["rag_query_mode"] = query_mode
    state["rag_code_sketch"] = code_sketch

    state = inject_rag_context_into_state(
        state=state,
        query=rag_query,
        top_k=state.get("rag_top_k", 5),
        search_type=state.get("rag_search_type", "hybrid"),
        section_filter=None,
    )

    return state


def _ensure_rag_defaults(state: dict) -> dict:
    state = dict(state)
    state.setdefault("rag_query_mode", DEFAULT_RAG_QUERY_MODE)
    state.setdefault("rag_code_sketch", "")
    state.setdefault("rag_query", "")
    state.setdefault("rag_context", "")
    state.setdefault("rag_search_type", state.get("rag_search_type", "hybrid"))
    return state


def _inject_eda_context(state: dict) -> dict:
    state = _ensure_rag_defaults(state)
    if state.get("rag_enabled", False):
        intent_query = (
            "tabular dataset exploratory data analysis notebook; "
            "dataset loading; column profiling; missing values; "
            "target detection; task type inference; numeric correlations; "
            "categorical summary"
        )
        state = _inject_rag_with_mode(
            state,
            intent_query=intent_query,
            code_sketch=_build_eda_code_sketch(state),
            log_label="EDA",
        )

    if state.get("web_search_enabled", False):
        web_query = (
            "best practices for tabular exploratory data analysis target detection "
            "missing values categorical features"
        )
        state = inject_web_context_into_state(
            state=state,
            query=web_query,
            max_results=state.get("web_search_max_results", 3),
        )
        _log("Web query for EDA step: %s", web_query)
    return state


def _inject_train_context(state: dict) -> dict:
    state = _ensure_rag_defaults(state)
    if state.get("rag_enabled", False):
        task_type = state.get("task_type", "")
        intent_query = (
            f"tabular machine learning notebook for {task_type}; "
            f"preprocessing categorical and numeric features; "
            f"columntransformer pipeline; training and evaluation; "
            f"submission pattern"
        )
        state = _inject_rag_with_mode(
            state,
            intent_query=intent_query,
            code_sketch=_build_train_code_sketch(state),
            log_label="training",
        )

    if state.get("web_search_enabled", False):
        web_query = (
            f"best practices for tabular {state.get('task_type', 'regression')} "
            f"with categorical features catboost cross validation"
        )
        state = inject_web_context_into_state(
            state=state,
            query=web_query,
            max_results=state.get("web_search_max_results", 3),
        )
        _log("Web query for training step: %s", web_query)
    return state


def _inject_eval_context(state: dict) -> dict:
    state = _ensure_rag_defaults(state)
    if state.get("rag_enabled", False):
        task_type = state.get("task_type", "")
        intent_query = (
            f"tabular machine learning notebook for {task_type} evaluation; "
            f"reproduce train validation split; metrics calculation; "
            f"pipeline prediction; local validation report"
        )
        state = _inject_rag_with_mode(
            state,
            intent_query=intent_query,
            code_sketch=_build_eval_code_sketch(state),
            log_label="eval",
        )

    if state.get("web_search_enabled", False):
        web_query = (
            f"how to evaluate tabular {state.get('task_type', 'regression')} "
            f"catboost cross validation metric selection"
        )
        state = inject_web_context_into_state(
            state=state,
            query=web_query,
            max_results=state.get("web_search_max_results", 3),
        )
        _log("Web query for eval step: %s", web_query)
    return state


def _inject_submission_context(state: dict) -> dict:
    state = _ensure_rag_defaults(state)
    if state.get("rag_enabled", False):
        task_type = state.get("task_type", "")
        intent_query = (
            f"tabular machine learning notebook for {task_type} submission; "
            f"load saved pipeline; predict on test dataframe; "
            f"match sample submission columns and row order; save submission csv"
        )
        state = _inject_rag_with_mode(
            state,
            intent_query=intent_query,
            code_sketch=_build_submission_code_sketch(state),
            log_label="submission",
        )

    if state.get("web_search_enabled", False):
        web_query = "kaggle submission csv format regression predict best practices"
        state = inject_web_context_into_state(
            state=state,
            query=web_query,
            max_results=state.get("web_search_max_results", 3),
        )
        _log("Web query for submission step: %s", web_query)
    return state


def step1_eda_agent(state: dict) -> dict:
    state = _inject_eda_context(state)
    return _run_feedback_step(
        "step1_eda",
        STEP1_PLANNER_PROMPT,
        STEP1_EDA_PROMPT,
        STEP1_VERIFIER_PROMPT,
        step1_eda_fallback,
        state,
        timeout_sec=120,
    )


def step2_train_agent(state: dict) -> dict:
    state = _inject_train_context(state)
    return _run_feedback_step(
        "step2_train",
        STEP2_PLANNER_PROMPT,
        STEP2_TRAIN_PROMPT,
        STEP2_VERIFIER_PROMPT,
        step2_train_fallback,
        state,
        timeout_sec=360,
    )


def step3_local_eval_agent(state: dict) -> dict:
    state = _inject_eval_context(state)
    return _run_feedback_step(
        "step3_eval",
        STEP3_PLANNER_PROMPT,
        STEP3_EVAL_PROMPT,
        STEP3_VERIFIER_PROMPT,
        step3_local_eval_fallback,
        state,
        timeout_sec=120,
    )


def step4_submission_agent(state: dict) -> dict:
    state = _inject_submission_context(state)
    return _run_feedback_step(
        "step4_submission",
        STEP4_PLANNER_PROMPT,
        STEP4_SUBMISSION_PROMPT,
        STEP4_VERIFIER_PROMPT,
        step4_submission_fallback,
        state,
        timeout_sec=120,
    )


# ---------------------------------------------------------------------------
# Step 7: report (single chain, no feedback loop)
# ---------------------------------------------------------------------------

def step7_report_agent(state: dict) -> dict:
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        _log("step7_report — no LLM, using fallback", level="warning")
        return step7_report_fallback(state)

    chain = create_step_chain(STEP7_REPORT_PROMPT, llm)
    state, success = run_step_with_retry(
        "step7_report", chain, state, max_attempts=3, timeout_sec=60,
    )

    if not success:
        _log("step7_report — agent failed, using fallback", level="warning")
        return step7_report_fallback(state)
    return state


# ---------------------------------------------------------------------------
# Judge (LLM-as-a-judge, outer refinement loop)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {
            "decision": "NEED_REFINEMENT",
            "reasoning": "Failed to parse JSON, using fallback",
            "eda_suggestions": "",
            "train_suggestions": "",
        }


def step_judge_result_agent(state: dict) -> dict:
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        _log("Judge — no LLM, using fallback", level="warning")
        return step_judge_fallback(state)

    model_info_parts = []
    model_type = state.get("model_type", "")
    if model_type:
        model_info_parts.append(f"Model: {model_type}")
    task_type = state.get("task_type", "")
    if task_type:
        model_info_parts.append(f"Task: {task_type}")
    target_transform = state.get("target_transform", "")
    if target_transform and target_transform != "none":
        model_info_parts.append(f"Target transform: {target_transform}")
    model_info = "; ".join(model_info_parts) if model_info_parts else "unknown"

    local_metrics = state.get("local_metrics", {})
    metrics_str = json.dumps(local_metrics, separators=(",", ":"))[:500] if local_metrics else "{}"

    prompt_data = {
        "local_metrics": metrics_str,
        "model_info": model_info,
    }

    chain = create_step_chain(STEP_JUDGE_PROMPT, llm)
    judge_prompt_filled = STEP_JUDGE_PROMPT.format(**prompt_data)

    iter_dir = Path(state["session_dir"]) / f"iter_{state.get('iteration', 1)}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "judge_prompt.txt").write_text(judge_prompt_filled, encoding="utf-8")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        _log("  Judge attempt %d/%d", attempt, max_retries)
        try:
            response = chain.invoke(prompt_data)
            if not response:
                continue

            result = _extract_json(response)
            state["verification_decision"] = result.get("decision", "NEED_REFINEMENT")
            state["verification_reasoning"] = result.get("reasoning", "")
            state["eda_improvement_hint"] = result.get("eda_suggestions", "")
            state["train_improvement_hint"] = result.get("train_suggestions", "")
            state["verification_raw"] = response

            (iter_dir / "judge_response.txt").write_text(response, encoding="utf-8")
            (iter_dir / "judge_decision.txt").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
            )

            _log("  Judge decision: %s", state["verification_decision"])
            if state["verification_decision"] == "NEED_REFINEMENT":
                _log("    EDA suggestions: %s", state["eda_improvement_hint"])
                _log("    Train suggestions: %s", state["train_improvement_hint"])
            return state

        except Exception as e:
            _log("  Judge attempt %d failed: %s", attempt, e, level="error")

    _log("Judge failed after %d attempts, using fallback", max_retries, level="error")
    return step_judge_fallback(state)
