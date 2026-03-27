"""Deterministic fallback functions that run without an LLM."""

from __future__ import annotations

import json
from pathlib import Path

import config as cfg


def step1_eda_fallback(state: dict) -> dict:
    """Fallback EDA: basic stats, no LLM."""
    state = dict(state)
    train_path = Path(state.get("train_path", ""))
    test_path = Path(state.get("test_path", ""))

    try:
        import pandas as pd
    except ImportError:
        if cfg.logger:
            cfg.logger.warning("pandas not installed; skipping EDA.")
        state["eda_report"] = ""
        return state

    report_parts: list[str] = []

    if train_path.exists():
        train_df = pd.read_csv(train_path)
        report_parts.append(f"Train: {len(train_df)} rows, {len(train_df.columns)} columns")
        report_parts.append(f"Columns: {list(train_df.columns)}")
        report_parts.append(str(train_df.describe()))
        report_parts.append(f"Missing: {train_df.isnull().sum().to_dict()}")
        state["train_df"] = train_df
        state["train_shape"] = train_df.shape
    else:
        if cfg.logger:
            cfg.logger.warning("Train file not found: %s", train_path)

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        report_parts.append(f"Test: {len(test_df)} rows, {len(test_df.columns)} columns")
        state["test_df"] = test_df
        state["test_shape"] = test_df.shape
    else:
        if cfg.logger:
            cfg.logger.warning("Test file not found: %s", test_path)

    eda_text = "\n\n".join(report_parts) if report_parts else "No data loaded."
    state["eda_report"] = eda_text

    eda_file = Path(state["session_dir"]) / "reports" / "eda_summary.txt"
    with open(eda_file, "w", encoding="utf-8") as f:
        f.write(eda_text)

    state["step1_eda_success"] = True
    if cfg.logger:
        cfg.logger.info("Step 1 EDA (fallback) done")
    return state


def step2_train_fallback(state: dict) -> dict:
    """Fallback training: RandomForestClassifier, no LLM."""
    state = dict(state)

    try:
        import pandas as pd
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        if cfg.logger:
            cfg.logger.warning("sklearn/joblib not available: %s", e)
        state["model_path"] = ""
        return state

    train_path = Path(state.get("train_path", ""))
    if not train_path.exists():
        if cfg.logger:
            cfg.logger.warning("No train data; skipping training.")
        state["model_path"] = ""
        return state

    train_df_full = pd.read_csv(train_path)
    train_df = train_df_full.sample(frac=cfg.TRAIN_SAMPLE_FRAC, random_state=42)

    target_candidates = ["target", "label", "y"]
    target_col = None
    for c in target_candidates:
        if c in train_df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = train_df.columns[-1]

    if cfg.logger:
        cfg.logger.info("Using target column: %s", target_col)

    X = train_df.drop(columns=[target_col], errors="ignore").select_dtypes(include=["number"])
    if X.empty:
        X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]
    state["target_column"] = target_col

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    model_path = Path(state["session_dir"]) / "models" / "model.joblib"
    joblib.dump(model, model_path)
    state["model_path"] = str(model_path)
    state["model"] = model

    state["step2_train_success"] = True
    if cfg.logger:
        cfg.logger.info("Step 2 train (fallback) done; model saved to %s", model_path)
    return state


def step3_local_eval_fallback(state: dict) -> dict:
    """Fallback evaluation: accuracy + F1-macro, no LLM."""
    state = dict(state)
    model_path_str = state.get("model_path", "")

    if not model_path_str or not Path(model_path_str).exists():
        state["local_metrics"] = {}
        if cfg.logger:
            cfg.logger.warning("No model found; skipping local eval.")
        return state

    model_path = Path(model_path_str)

    try:
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        state["local_metrics"] = {}
        return state

    model = joblib.load(model_path)

    train_path = Path(state.get("train_path", ""))
    train_df_full = pd.read_csv(train_path)
    train_df = train_df_full.sample(frac=cfg.TRAIN_SAMPLE_FRAC, random_state=42)

    target_col = state.get("target_column", train_df.columns[-1])
    X = train_df.drop(columns=[target_col], errors="ignore").select_dtypes(include=["number"])
    y = train_df[target_col]

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    try:
        f1 = f1_score(y_val, pred, average="macro")
    except Exception:
        f1 = 0.0

    state["local_metrics"] = {"accuracy": acc, "f1_macro": f1}

    metrics_file = Path(state["session_dir"]) / "reports" / "local_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(state["local_metrics"], f, indent=2)

    state["step3_eval_success"] = True
    if cfg.logger:
        cfg.logger.info("Step 3 local eval (fallback): accuracy=%.4f, f1_macro=%.4f", acc, f1)
    return state


def step4_submission_fallback(state: dict) -> dict:
    """Fallback submission builder, no LLM."""
    state = dict(state)
    model_path_str = state.get("model_path", "")

    if not model_path_str or not Path(model_path_str).exists():
        state["submission_path"] = ""
        if cfg.logger:
            cfg.logger.warning("No model; skipping submission build.")
        return state

    model_path = Path(model_path_str)

    try:
        import pandas as pd
        import joblib
    except ImportError:
        state["submission_path"] = ""
        return state

    model = joblib.load(model_path)
    test_path = Path(state.get("test_path", ""))
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    if test_df is None or test_df.empty:
        state["submission_path"] = ""
        return state

    if hasattr(model, "feature_names_in_"):
        feats = [c for c in model.feature_names_in_ if c in test_df.columns]
        X_test = test_df[feats] if feats else test_df.select_dtypes(include=["number"])
    else:
        X_test = test_df.select_dtypes(include=["number"])

    preds = model.predict(X_test)
    out_path = Path(state["session_dir"]) / "submission.csv"

    sample_path = Path(state.get("sample_submission_path", ""))
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        out_df = sample.copy()
        pred_col = sample.columns[1] if len(sample.columns) > 1 else "prediction"
        out_df[pred_col] = preds
        out_df = out_df[sample.columns]
    else:
        id_col = "id" if "id" in test_df.columns else test_df.columns[0]
        out_df = test_df[[id_col]].copy() if id_col in test_df.columns else pd.DataFrame({"id": range(len(preds))})
        out_df["prediction"] = preds

    out_df.to_csv(out_path, index=False)
    state["submission_path"] = str(out_path)

    state["step4_submission_success"] = True
    if cfg.logger:
        cfg.logger.info("Step 4 submission (fallback) saved to %s", out_path)
    return state


def step_verify_fallback(state: dict) -> dict:
    """Детерминированный фолбек для судьи: если LLM недоступна,
    проверяем просто факт наличия обученной модели и метрик."""
    state = dict(state)

    metrics_path = Path(state["session_dir"]) / "reports" / "local_metrics.json"
    model_path = Path(state["session_dir"]) / "models" / "model.joblib"

    if metrics_path.exists() and model_path.exists():
        state["verification_decision"] = "SUFFICIENT"
        state["verification_reasoning"] = "Fallback: Metrics and model files found."
    else:
        state["verification_decision"] = "NEED_REFINEMENT"
        state["verification_reasoning"] = "Fallback: Missing model or metrics files."

    if cfg.logger:
        cfg.logger.info("Step verify: Using fallback logic.")

    return state


def step7_report_fallback(state: dict) -> dict:
    """Fallback report: JSON + plain text summary, no LLM."""
    state = dict(state)

    report = {
        "eda_summary": state.get("eda_report", "")[:1000],
        "local_metrics": state.get("local_metrics", {}),
        "model_path": state.get("model_path", ""),
        "submission_path": state.get("submission_path", ""),
        "submit_ok": state.get("submit_ok"),
        "public_score": state.get("public_score"),
        "private_score": state.get("private_score"),
        "submission_status": state.get("submission_status", ""),
    }

    report_path_json = Path(state["session_dir"]) / "reports" / "final_report.json"
    report_path_txt = Path(state["session_dir"]) / "reports" / "final_report.txt"

    with open(report_path_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "Report — Agent Pipeline Run",
        "============================",
        "EDA summary: " + (report["eda_summary"] or "")[:500],
        "Local metrics: " + json.dumps(report["local_metrics"]),
        "Model: " + report["model_path"],
        "Submission: " + report["submission_path"],
        "Submitted: " + str(report["submit_ok"]),
        "Public score: " + str(report["public_score"]),
        "Private score: " + str(report["private_score"]),
        "Status: " + str(report["submission_status"]),
    ]
    text_report = "\n".join(lines)
    with open(report_path_txt, "w", encoding="utf-8") as f:
        f.write(text_report)

    state["report_path"] = str(report_path_txt)
    state["step7_report_success"] = True

    if cfg.logger:
        cfg.logger.info("Step 7 report (fallback) saved to %s", report_path_txt)
    return state
