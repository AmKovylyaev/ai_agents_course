"""Deterministic fallback functions that run without an LLM."""

from __future__ import annotations

import json
from pathlib import Path

import config as cfg


def step1_eda_fallback(state: dict) -> dict:
    """Fallback EDA: profile all columns, no LLM."""
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
        report_parts.append(f"Dtypes:\n{train_df.dtypes.to_string()}")
        report_parts.append(str(train_df.describe()))
        report_parts.append(f"Missing: {train_df.isnull().sum().to_dict()}")

        target_candidates = ["target", "label", "y"]
        target_col = None
        for c in target_candidates:
            if c in train_df.columns:
                target_col = c
                break
        if target_col is None:
            target_col = train_df.columns[-1]

        numeric_cols = list(
            train_df.drop(columns=[target_col], errors="ignore")
            .select_dtypes(include=["number"]).columns
        )
        categorical_cols = list(
            train_df.drop(columns=[target_col], errors="ignore")
            .select_dtypes(include=["object", "category"]).columns
        )
        n_classes = int(train_df[target_col].nunique())
        is_numeric_target = train_df[target_col].dtype.kind in ("i", "f")
        unique_ratio = n_classes / len(train_df) if len(train_df) > 0 else 0
        task_type = (
            "regression"
            if is_numeric_target and (n_classes > 20 or unique_ratio > 0.05)
            else "classification"
        )

        report_parts.append(f"Target: {target_col} ({n_classes} unique values)")
        report_parts.append(f"Task type: {task_type}")
        report_parts.append(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
        report_parts.append(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

        state["train_shape"] = list(train_df.shape)
        state["target_column"] = target_col
        state["numeric_columns"] = numeric_cols
        state["categorical_columns"] = categorical_cols
        state["n_classes"] = n_classes
        state["task_type"] = task_type
        state["columns"] = list(train_df.columns)
        state["missing_values"] = train_df.isnull().sum().to_dict()
    else:
        if cfg.logger:
            cfg.logger.warning("Train file not found: %s", train_path)

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        report_parts.append(f"Test: {len(test_df)} rows, {len(test_df.columns)} columns")
        state["test_shape"] = list(test_df.shape)
    else:
        if cfg.logger:
            cfg.logger.warning("Test file not found: %s", test_path)

    eda_text = "\n\n".join(report_parts) if report_parts else "No data loaded."
    state["eda_report"] = eda_text

    reports_dir = Path(state["session_dir"]) / "reports"
    eda_file = reports_dir / "eda_summary.txt"
    with open(eda_file, "w", encoding="utf-8") as f:
        f.write(eda_text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if train_path.exists():
            train_df = pd.read_csv(train_path)

            if target_col and target_col in train_df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                train_df[target_col].value_counts().plot.bar(ax=ax)
                ax.set_title("Target Distribution")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(reports_dir / "target_distribution.png", dpi=100)
                plt.close(fig)

            if numeric_cols:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = train_df[numeric_cols].corr()
                im = ax.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(numeric_cols)))
                ax.set_yticks(range(len(numeric_cols)))
                ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(numeric_cols, fontsize=7)
                ax.set_title("Feature Correlation Heatmap")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(reports_dir / "correlation_heatmap.png", dpi=100)
                plt.close(fig)

            missing = train_df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                missing.plot.bar(ax=ax)
                ax.set_title("Missing Values per Column")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(reports_dir / "missing_values.png", dpi=100)
                plt.close(fig)
    except ImportError:
        if cfg.logger:
            cfg.logger.warning("matplotlib not available; skipping EDA plots.")

    state["step1_eda_success"] = True
    state["step1_eda_code"] = "(fallback – no LLM-generated code)"
    if cfg.logger:
        cfg.logger.info("Step 1 EDA (fallback) done")
    return state


def step2_train_fallback(state: dict) -> dict:
    """Fallback training: CatBoost with native categorical support, no LLM."""
    state = dict(state)

    try:
        import numpy as np
        import pandas as pd
        import joblib
        from catboost import CatBoostClassifier, CatBoostRegressor
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        if cfg.logger:
            cfg.logger.warning("catboost/joblib not available: %s", e)
        state["model_path"] = ""
        return state

    train_path = Path(state.get("train_path", ""))
    if not train_path.exists():
        if cfg.logger:
            cfg.logger.warning("No train data; skipping training.")
        state["model_path"] = ""
        return state

    train_df = pd.read_csv(train_path)

    target_col = state.get("target_column")
    if not target_col:
        for c in ("target", "label", "y"):
            if c in train_df.columns:
                target_col = c
                break
        if not target_col:
            target_col = train_df.columns[-1]

    task_type = state.get("task_type", "classification")
    if cfg.logger:
        cfg.logger.info("Using target column: %s (task_type=%s)", target_col, task_type)

    X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]

    if task_type == "regression":
        y = np.log1p(y)
        state["target_transform"] = "log1p"
        if cfg.logger:
            cfg.logger.info("Applied log1p target transformation (regression)")
    else:
        state["target_transform"] = "none"

    cat_cols = [
        c for c in state.get(
            "categorical_columns",
            list(X.select_dtypes(include=["object", "category"]).columns),
        ) if c in X.columns
    ]
    for c in cat_cols:
        X[c] = X[c].fillna("__missing__").astype(str)

    cat_indices = [X.columns.get_loc(c) for c in cat_cols]

    if cfg.logger:
        cfg.logger.info(
            "Columns: %d total, %d categorical (native CatBoost)",
            len(X.columns), len(cat_cols),
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.TRAIN_SAMPLE_FRAC, random_state=42,
    )

    if task_type == "regression":
        model = CatBoostRegressor(
            iterations=500, learning_rate=0.1, depth=6,
            loss_function="RMSE",
            random_seed=42, verbose=0, cat_features=cat_indices,
        )
    else:
        model = CatBoostClassifier(
            iterations=500, learning_rate=0.1, depth=6,
            random_seed=42, verbose=0, cat_features=cat_indices,
        )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    state["target_column"] = target_col

    models_dir = Path(state["session_dir"]) / "models"
    model_path = models_dir / "pipeline.joblib"
    joblib.dump(model, model_path)
    state["model_path"] = str(model_path)
    state["model_type"] = type(model).__name__

    state["step2_train_success"] = True
    state["step2_train_code"] = "(fallback – no LLM-generated code)"
    if cfg.logger:
        y_pred = model.predict(X_val)
        if task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_val_orig = np.expm1(y_val)
            y_pred_orig = np.expm1(y_pred)
            mse = float(mean_squared_error(y_val_orig, y_pred_orig))
            mae = float(mean_absolute_error(y_val_orig, y_pred_orig))
            r2 = float(r2_score(y_val_orig, y_pred_orig))
            cfg.logger.info(
                "Step 2 train (fallback, original scale): mse=%.4f, mae=%.4f, r2=%.4f",
                mse, mae, r2,
            )
        else:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
            )
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            cfg.logger.info(
                "Step 2 train (fallback): accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f",
                acc, prec, rec, f1,
            )
        cfg.logger.info("Model saved to %s", model_path)
    return state


def step3_local_eval_fallback(state: dict) -> dict:
    """Fallback evaluation: load model/pipeline, predict, score. No LLM."""
    state = dict(state)
    model_path_str = state.get("model_path", "")

    if not model_path_str or not Path(model_path_str).exists():
        state["local_metrics"] = {}
        if cfg.logger:
            cfg.logger.warning("No model/pipeline found; skipping local eval.")
        return state

    try:
        import numpy as np
        import pandas as pd
        import joblib
        from sklearn.model_selection import train_test_split
    except ImportError:
        state["local_metrics"] = {}
        return state

    model = joblib.load(model_path_str)

    train_path = Path(state.get("train_path", ""))
    train_df = pd.read_csv(train_path)

    target_col = state.get("target_column", train_df.columns[-1])
    X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]

    target_transform = state.get("target_transform", "none")
    if target_transform == "log1p":
        y = np.log1p(y)

    cat_cols = [
        c for c in state.get("categorical_columns", []) if c in X.columns
    ]
    for c in cat_cols:
        X[c] = X[c].fillna("__missing__").astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.TRAIN_SAMPLE_FRAC, random_state=42,
    )

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    task_type = state.get("task_type", "classification")

    if task_type == "regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        if target_transform == "log1p":
            y_train, y_val = np.expm1(y_train), np.expm1(y_val)
            pred_train, pred_val = np.expm1(pred_train), np.expm1(pred_val)
        train_metrics = {
            "mse": float(mean_squared_error(y_train, pred_train)),
            "mae": float(mean_absolute_error(y_train, pred_train)),
            "r2": float(r2_score(y_train, pred_train)),
        }
        val_metrics = {
            "mse": float(mean_squared_error(y_val, pred_val)),
            "mae": float(mean_absolute_error(y_val, pred_val)),
            "r2": float(r2_score(y_val, pred_val)),
        }
    else:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
        )
        train_metrics = {
            "accuracy": accuracy_score(y_train, pred_train),
            "precision": precision_score(y_train, pred_train, average="macro", zero_division=0),
            "recall": recall_score(y_train, pred_train, average="macro", zero_division=0),
            "f1": f1_score(y_train, pred_train, average="macro", zero_division=0),
        }
        val_metrics = {
            "accuracy": accuracy_score(y_val, pred_val),
            "precision": precision_score(y_val, pred_val, average="macro", zero_division=0),
            "recall": recall_score(y_val, pred_val, average="macro", zero_division=0),
            "f1": f1_score(y_val, pred_val, average="macro", zero_division=0),
        }

    state["local_metrics"] = {"train": train_metrics, "val": val_metrics}

    metrics_file = Path(state["session_dir"]) / "reports" / "local_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(state["local_metrics"], f, indent=2)

    state["step3_eval_success"] = True
    state["step3_eval_code"] = "(fallback – no LLM-generated code)"
    if cfg.logger:
        cfg.logger.info("Step 3 local eval (fallback):")
        cfg.logger.info("  Train metrics: %s", train_metrics)
        cfg.logger.info("  Val metrics:   %s", val_metrics)
    return state


def step4_submission_fallback(state: dict) -> dict:
    """Fallback submission builder: load pipeline, predict on test. No LLM."""
    state = dict(state)
    model_path_str = state.get("model_path", "")

    if not model_path_str or not Path(model_path_str).exists():
        state["submission_path"] = ""
        if cfg.logger:
            cfg.logger.warning("No pipeline; skipping submission build.")
        return state

    try:
        import pandas as pd
        import joblib
    except ImportError:
        state["submission_path"] = ""
        return state

    try:
        model = joblib.load(model_path_str)
        if not hasattr(model, "predict"):
            raise ValueError(f"Loaded object ({type(model).__name__}) has no predict() method")
    except Exception as e:
        state["submission_path"] = ""
        if cfg.logger:
            cfg.logger.warning("Cannot load model for submission: %s", e)
        return state

    test_path = Path(state.get("test_path", ""))
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    if test_df is None or test_df.empty:
        state["submission_path"] = ""
        return state

    target_col = state.get("target_column", "")
    X_test = test_df.drop(columns=[target_col], errors="ignore")

    cat_cols = [
        c for c in state.get("categorical_columns", []) if c in X_test.columns
    ]
    for c in cat_cols:
        X_test[c] = X_test[c].fillna("__missing__").astype(str)

    preds = model.predict(X_test)

    target_transform = state.get("target_transform", "none")
    if target_transform == "log1p":
        import numpy as np
        preds = np.expm1(preds)

    out_path = Path(state["session_dir"]) / "submission.csv"

    sample_path = Path(state.get("sample_submission_path", ""))
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        out_df = sample[sample.columns].copy()
        pred_col = sample.columns[-1]
        out_df[pred_col] = preds
        assert list(out_df.columns) == list(sample.columns), (
            f"Column mismatch: {list(out_df.columns)} != {list(sample.columns)}"
        )
    else:
        id_col = "id" if "id" in test_df.columns else test_df.columns[0]
        out_df = test_df[[id_col]].copy() if id_col in test_df.columns else pd.DataFrame({"id": range(len(preds))})
        out_df["prediction"] = preds

    out_df.to_csv(out_path, index=False)
    state["submission_path"] = str(out_path)

    state["step4_submission_success"] = True
    if cfg.logger:
        cfg.logger.info(
            "Step 4 submission (fallback): %d rows, %d cols, saved to %s",
            len(out_df), len(out_df.columns), out_path,
        )
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


def step_judge_fallback(state: dict) -> dict:
    """Deterministic judge fallback: check if model and metrics exist."""
    state = dict(state)

    metrics_path = Path(state["session_dir"]) / "reports" / "local_metrics.json"
    model_path = Path(state.get("model_path", ""))

    if metrics_path.exists() and model_path.exists():
        state["verification_decision"] = "SUFFICIENT"
        state["verification_reasoning"] = "Fallback: Metrics and model files found."
    else:
        state["verification_decision"] = "NEED_REFINEMENT"
        state["verification_reasoning"] = "Fallback: Missing model or metrics files."

    if cfg.logger:
        cfg.logger.info("Step judge (fallback): %s", state["verification_decision"])

    return state
