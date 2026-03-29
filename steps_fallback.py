"""Deterministic fallback functions that run without an LLM."""

from __future__ import annotations

import json
from pathlib import Path

import config as cfg

def _prepare_catboost_categorical_columns(df, categorical_cols):
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("__missing__").astype(str)
    return df

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

        state["train_df"] = train_df
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
        state["test_df"] = test_df
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
    if cfg.logger:
        cfg.logger.info("Step 1 EDA (fallback) done")
    return state


def step2_train_fallback(state: dict) -> dict:
    """Fallback training: CatBoost for regression, no LLM."""
    state = dict(state)

    try:
        import pandas as pd
        import joblib
        import numpy as np
        from catboost import CatBoostRegressor, CatBoostClassifier
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        if cfg.logger:
            cfg.logger.warning("Required packages not available: %s", e)
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

    X = train_df.drop(columns=[target_col], errors="ignore").copy()
    y = train_df[target_col].copy()

    numeric_cols = [
        c for c in state.get(
            "numeric_columns",
            list(X.select_dtypes(include=["number"]).columns),
        ) if c in X.columns
    ]
    categorical_cols = [
        c for c in state.get(
            "categorical_columns",
            list(X.select_dtypes(include=["object", "category"]).columns),
        ) if c in X.columns
    ]

    # CatBoost prefers string/object categorical features explicitly marked
    X = _prepare_catboost_categorical_columns(X, categorical_cols)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.TRAIN_SAMPLE_FRAC,
        random_state=42,
    )

    X_train = _prepare_catboost_categorical_columns(X_train, categorical_cols)
    X_val = _prepare_catboost_categorical_columns(X_val, categorical_cols)

    if task_type == "regression":
        model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=100,
        )
    else:
        model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=100,
        )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols if categorical_cols else None,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    state["target_column"] = target_col

    models_dir = Path(state["session_dir"]) / "models"
    model_path = models_dir / "pipeline.joblib"
    joblib.dump(model, model_path)
    state["model_path"] = str(model_path)
    state["model_type"] = type(model).__name__
    state["categorical_columns"] = categorical_cols
    state["numeric_columns"] = numeric_cols

    state["step2_train_success"] = True

    if cfg.logger:
        y_pred = model.predict(X_val)

        if task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae = float(mean_absolute_error(y_val, y_pred))
            r2 = float(r2_score(y_val, y_pred))

            cfg.logger.info(
                "Step 2 train (fallback, CatBoost): rmse=%.4f, mae=%.4f, r2=%.4f",
                rmse, mae, r2,
            )
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            y_pred_cls = model.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred_cls))
            prec = float(precision_score(y_val, y_pred_cls, average="macro", zero_division=0))
            rec = float(recall_score(y_val, y_pred_cls, average="macro", zero_division=0))
            f1 = float(f1_score(y_val, y_pred_cls, average="macro", zero_division=0))

            cfg.logger.info(
                "Step 2 train (fallback, CatBoost): accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f",
                acc, prec, rec, f1,
            )

        cfg.logger.info("Model saved to %s", model_path)

    return state


def step3_local_eval_fallback(state: dict) -> dict:
    """Fallback evaluation: load pipeline, predict, score. No LLM."""
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

    pipeline = joblib.load(model_path_str)

    train_path = Path(state.get("train_path", ""))
    train_df = pd.read_csv(train_path)

    target_col = state.get("target_column", train_df.columns[-1])
    X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]

    _, X_val, _, y_val = train_test_split(X, y, test_size=cfg.TRAIN_SAMPLE_FRAC, random_state=42)

    categorical_cols = [c for c in state.get("categorical_columns", []) if c in X_val.columns]
    X_val = _prepare_catboost_categorical_columns(X_val, categorical_cols)

    pred = pipeline.predict(X_val)
    task_type = state.get("task_type", "classification")

    if task_type == "regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        mae = float(mean_absolute_error(y_val, pred))
        r2 = float(r2_score(y_val, pred))
        state["local_metrics"] = {"rmse": rmse, "mae": mae, "r2": r2}
        log_msg = f"Step 3 local eval (fallback): rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}"
    else:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
        )
        acc = accuracy_score(y_val, pred)
        prec = precision_score(y_val, pred, average="macro", zero_division=0)
        rec = recall_score(y_val, pred, average="macro", zero_division=0)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        state["local_metrics"] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        }
        log_msg = f"Step 3 local eval (fallback): accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}"

    metrics_file = Path(state["session_dir"]) / "reports" / "local_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(state["local_metrics"], f, indent=2)

    state["step3_eval_success"] = True
    if cfg.logger:
        cfg.logger.info(log_msg)
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

    pipeline = joblib.load(model_path_str)
    test_path = Path(state.get("test_path", ""))
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    if test_df is None or test_df.empty:
        state["submission_path"] = ""
        return state

    target_col = state.get("target_column", "")
    X_test = test_df.drop(columns=[target_col], errors="ignore")

    categorical_cols = [c for c in state.get("categorical_columns", []) if c in X_test.columns]
    X_test = _prepare_catboost_categorical_columns(X_test, categorical_cols)
    
    preds = pipeline.predict(X_test)
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
