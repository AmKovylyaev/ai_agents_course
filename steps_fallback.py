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

        report_parts.append(f"Target: {target_col} ({n_classes} classes)")
        report_parts.append(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
        report_parts.append(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

        state["train_df"] = train_df
        state["train_shape"] = list(train_df.shape)
        state["target_column"] = target_col
        state["numeric_columns"] = numeric_cols
        state["categorical_columns"] = categorical_cols
        state["n_classes"] = n_classes
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

    eda_file = Path(state["session_dir"]) / "reports" / "eda_summary.txt"
    with open(eda_file, "w", encoding="utf-8") as f:
        f.write(eda_text)

    state["step1_eda_success"] = True
    if cfg.logger:
        cfg.logger.info("Step 1 EDA (fallback) done")
    return state


def step2_train_fallback(state: dict) -> dict:
    """Fallback training: Pipeline with ColumnTransformer + RandomForest, no LLM."""
    state = dict(state)

    try:
        import pandas as pd
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
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

    target_col = state.get("target_column")
    if not target_col:
        for c in ("target", "label", "y"):
            if c in train_df.columns:
                target_col = c
                break
        if not target_col:
            target_col = train_df.columns[-1]

    if cfg.logger:
        cfg.logger.info("Using target column: %s", target_col)

    X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]

    numeric_cols = [
        c for c in state.get(
            "numeric_columns",
            list(X.select_dtypes(include=["number"]).columns),
        ) if c in X.columns
    ]
    all_cat_cols = [
        c for c in state.get(
            "categorical_columns",
            list(X.select_dtypes(include=["object", "category"]).columns),
        ) if c in X.columns
    ]
    low_card_cols = [c for c in all_cat_cols if X[c].nunique() <= 50]
    high_card_cols = [c for c in all_cat_cols if X[c].nunique() > 50]

    if cfg.logger:
        cfg.logger.info(
            "Columns: %d numeric, %d low-card cat, %d high-card cat (ordinal)",
            len(numeric_cols), len(low_card_cols), len(high_card_cols),
        )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    low_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = [
        ("num", num_pipe, numeric_cols),
        ("cat_low", low_cat_pipe, low_card_cols),
    ]
    if high_card_cols:
        transformers.append(("cat_high", high_cat_pipe, high_card_cols))

    preprocessor = ColumnTransformer(transformers)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    pipeline.fit(X_train, y_train)

    state["target_column"] = target_col

    models_dir = Path(state["session_dir"]) / "models"
    model_path = models_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    state["model_path"] = str(model_path)

    state["step2_train_success"] = True
    if cfg.logger:
        from sklearn.metrics import accuracy_score, f1_score

        y_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        cfg.logger.info(
            "Step 2 train (fallback): accuracy=%.4f, f1_macro=%.4f", acc, f1,
        )
        cfg.logger.info("Pipeline saved to %s", model_path)
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
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        state["local_metrics"] = {}
        return state

    pipeline = joblib.load(model_path_str)

    train_path = Path(state.get("train_path", ""))
    train_df_full = pd.read_csv(train_path)
    train_df = train_df_full.sample(frac=cfg.TRAIN_SAMPLE_FRAC, random_state=42)

    target_col = state.get("target_column", train_df.columns[-1])
    X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, pred)
    try:
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
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
