"""
Linear ML Pipeline for Kaggle Competitions (No Agent)

This module implements a simple linear pipeline that executes ML tasks
in sequence without using LLM agents. It demonstrates the traditional
approach to ML competitions.

Pipeline Steps:
    1. EDA (Exploratory Data Analysis) - Load and analyze train/test data
    2. Train - Train a RandomForest classifier on training data
    3. Local Eval - Evaluate model on validation split
    4. Submission - Generate submission CSV file
    5. Submit - Submit to Kaggle competition
    6. Wait Results - Poll for Kaggle leaderboard results
    7. Report - Generate final pipeline report

Usage:
    From project root:
        .venv/bin/python ai_agents_course/final_project/ai_agent_step_by_step/01_.py

    Or use the shell script:
        ./run_01.sh

Requirements:
    - pandas, scikit-learn, joblib for ML operations
    - kaggle package for Kaggle API access
    - langchain-huggingface for optional LLM summaries
    - .env file with API_KAGGLE_KEY (new token format: KGAT_xxx)

Environment Variables:
    - API_KAGGLE_KEY: Your Kaggle API token (new format: KGAT_xxx)
    - KAGGLE_COMPETITION: Competition name (default: mws-ai-agents-2026)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Конфиг и пути — только модели из config
try:
    from config import model_llm, model_embedding
except ImportError:
    model_llm = "z-ai/glm-4.7-flash"
    model_embedding = "google/gemini-embedding-001"

SCRIPT_DIR = Path(__file__).resolve().parent
# Данные соревнования: final_project/data (на уровень выше ai_agent_step_by_step)
DATA_DIR = SCRIPT_DIR.parent / "data"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

# Имена файлов данных (типичная структура Kaggle)
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submition.csv"  # Note: file has typo in name

# Kaggle competition configuration
# Can be overridden via environment variable KAGGLE_COMPETITION
COMPETITION = os.getenv("KAGGLE_COMPETITION", "mws-ai-agents-2026")

logger: logging.Logger | None = None


def _setup_logging() -> None:
    """Настройка логирования: консоль + файл artifacts/run_01.log."""
    global logger
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = ARTIFACTS_DIR / "run_01.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging to %s", log_file)


def _load_kaggle_env() -> None:
    """
    Load Kaggle credentials from project root .env file.

    The .env file should contain:
    - API_KAGGLE_KEY: Your Kaggle API token (new format: KGAT_xxx)

    For new KGAT_ tokens, we use KAGGLE_API_TOKEN (not KAGGLE_KEY).
    KAGGLE_USERNAME is not required with the new token format.
    """
    from dotenv import load_dotenv

    # Load from project root .env
    # Path: ai_agents_course/final_project/ai_agent_step_by_step -> parent.parent.parent = project root
    project_root = SCRIPT_DIR.parent.parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        if logger:
            logger.info("Loaded environment from %s", env_path)
    else:
        if logger:
            logger.warning("No .env file found at %s", env_path)

    # Map API_KAGGLE_KEY to KAGGLE_API_TOKEN (required for KGAT_ tokens)
    # The new token format KGAT_xxx is a single token that replaces username+key
    api_kaggle_key = os.getenv("API_KAGGLE_KEY")
    if api_kaggle_key:
        if api_kaggle_key.startswith("KGAT_"):
            # New token format: use KAGGLE_API_TOKEN
            os.environ["KAGGLE_API_TOKEN"] = api_kaggle_key
            if logger:
                logger.info("Kaggle API token configured (KGAT_ format)")
        else:
            # Legacy token format: use KAGGLE_KEY + KAGGLE_USERNAME
            os.environ["KAGGLE_KEY"] = api_kaggle_key
            if logger:
                logger.info("Kaggle API key configured (legacy format)")
    else:
        if logger:
            logger.warning("API_KAGGLE_KEY not found in environment")


def _get_llm():
    """
    Initialize LLM from config (model_llm).

    Uses HuggingFace provider for models like z-ai/glm-4.7-flash.

    Returns:
        ChatHuggingFace instance or None if unavailable
    """
    try:
        from langchain_huggingface import ChatHuggingFace
        return ChatHuggingFace(model_id=model_llm, temperature=0)
    except Exception as e:
        if logger:
            logger.warning("ChatHuggingFace unavailable (%s); LLM steps will use templates.", e)
        return None


def step1_eda(state: dict[str, Any]) -> dict[str, Any]:
    """
    Perform Exploratory Data Analysis (EDA) on competition data.

    This step loads the train and test CSV files, computes basic statistics,
    and optionally generates an LLM-based summary of the findings.

    Args:
        state: Pipeline state dictionary containing:
            - 'train_path': Path to training data CSV
            - 'test_path': Path to test data CSV
            - Other context variables

    Returns:
        Updated state dictionary with:
            - 'eda_report': Full EDA text report
            - 'eda_summary': Short summary (first 500 chars or LLM-generated)
            - 'train_df': Loaded training DataFrame (or None)
            - 'test_df': Loaded test DataFrame (or None)
            - 'train_path': Path string to train.csv
            - 'test_path': Path string to test.csv
            - 'sample_submission_path': Path string to sample submission

    Note:
        If pandas is not installed or data files don't exist, the function
        gracefully handles errors and returns empty/default values.
    """
    state = dict(state)
    train_path = DATA_DIR / TRAIN_FILE
    test_path = DATA_DIR / TEST_FILE
    sample_path = DATA_DIR / SAMPLE_SUBMISSION_FILE

    if not DATA_DIR.exists():
        if logger:
            logger.warning("DATA_DIR %s does not exist; skipping EDA.", DATA_DIR)
        state["eda_report"] = ""
        state["train_path"] = str(train_path)
        state["test_path"] = str(test_path)
        state["sample_submission_path"] = str(sample_path)
        state["train_df"] = None
        state["test_df"] = None
        return state

    try:
        import pandas as pd
    except ImportError:
        if logger:
            logger.warning("pandas not installed; skipping EDA.")
        state["eda_report"] = ""
        state["train_path"] = str(train_path)
        state["test_path"] = str(test_path)
        state["sample_submission_path"] = str(sample_path)
        state["train_df"] = None
        state["test_df"] = None
        return state

    report_parts = []
    train_df, test_df = None, None

    if train_path.exists():
        train_df = pd.read_csv(train_path)
        report_parts.append(f"Train: {len(train_df)} rows, {len(train_df.columns)} columns")
        report_parts.append(f"Columns: {list(train_df.columns)}")
        report_parts.append(str(train_df.describe()))
        report_parts.append(f"Missing: {train_df.isnull().sum().to_dict()}")
        state["train_df"] = train_df
    else:
        if logger:
            logger.warning("Train file not found: %s", train_path)
    state["train_path"] = str(train_path)

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        report_parts.append(f"Test: {len(test_df)} rows, {len(test_df.columns)} columns")
        state["test_df"] = test_df
    else:
        if logger:
            logger.warning("Test file not found: %s", test_path)
    state["test_path"] = str(test_path)
    state["sample_submission_path"] = str(sample_path)

    eda_text = "\n\n".join(report_parts) if report_parts else "No data loaded."
    state["eda_report"] = eda_text

    llm = _get_llm()
    if llm and eda_text and logger:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            prompt = ChatPromptTemplate.from_messages([
                ("human", "Кратко резюмируй исследовательский анализ данных (2–3 предложения):\n\n{eda}"),
            ])
            chain = prompt | llm | StrOutputParser()
            state["eda_summary"] = chain.invoke({"eda": eda_text[:4000]})
        except Exception as e:
            if logger:
                logger.warning("LLM EDA summary failed: %s", e)
            state["eda_summary"] = eda_text[:500]
    else:
        state["eda_summary"] = eda_text[:500] if eda_text else ""

    if logger:
        logger.info("Step 1 EDA done; train=%s, test=%s", train_df is not None, test_df is not None)
    return state


def step2_train(state: dict[str, Any]) -> dict[str, Any]:
    """
    Train a RandomForest classifier on the training data.

    This step trains a model using the loaded training data and saves
    the trained model to artifacts/model.joblib.

    Args:
        state: Pipeline state dictionary with 'train_df' from step1_eda

    Returns:
        Updated state dictionary with:
            - 'model_path': Path to saved model file
            - 'model': Trained model object
            - 'X_train', 'X_val': Training and validation features
            - 'y_train', 'y_val': Training and validation labels
            - 'target_column': Name of the target column used

    Note:
        Uses 80/20 train/validation split. Target column is auto-detected
        from common names (target, label, y) or defaults to last column.
    """
    state = dict(state)
    train_df = state.get("train_df")
    if train_df is None or train_df.empty:
        if logger:
            logger.warning("No train data; skipping training.")
        state["model_path"] = ""
        return state

    try:
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        if logger:
            logger.warning("sklearn/joblib not available: %s", e)
        state["model_path"] = ""
        return state

    # Целевая колонка: типичные имена
    target_candidates = ["target", "label", "y"]
    target_col = None
    for c in target_candidates:
        if c in train_df.columns:
            target_col = c
            break
    if target_col is None:
        # Последняя колонка как целевая
        target_col = train_df.columns[-1]
    if logger:
        logger.info("Using target column: %s", target_col)

    X = train_df.drop(columns=[target_col], errors="ignore").select_dtypes(include=["number"])
    if X.empty:
        X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]
    state["target_column"] = target_col

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    state["X_train"], state["X_val"] = X_train, X_val
    state["y_train"], state["y_val"] = y_train, y_val

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(model, model_path)
    state["model_path"] = str(model_path)
    state["model"] = model

    if logger:
        logger.info("Step 2 train done; model saved to %s", model_path)
    return state


def step3_local_eval(state: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate the trained model on the validation set.

    Computes accuracy and macro F1 score on the validation data.

    Args:
        state: Pipeline state dictionary with 'model', 'X_val', 'y_val'

    Returns:
        Updated state dictionary with:
            - 'local_metrics': Dict with 'accuracy' and 'f1_macro' scores
    """
    state = dict(state)
    model = state.get("model")
    X_val = state.get("X_val")
    y_val = state.get("y_val")

    if model is None or X_val is None or y_val is None:
        state["local_metrics"] = {}
        if logger:
            logger.warning("No model or val data; skipping local eval.")
        return state

    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        state["local_metrics"] = {}
        return state

    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    try:
        f1 = f1_score(y_val, pred, average="macro")
    except Exception:
        f1 = 0.0
    state["local_metrics"] = {"accuracy": acc, "f1_macro": f1}
    if logger:
        logger.info("Step 3 local eval: accuracy=%.4f, f1_macro=%.4f", acc, f1)
    return state


def step4_submission(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate submission CSV file from model predictions on test data.

    Creates a submission file matching the format of the sample submission,
    using the trained model to predict on test data.

    Args:
        state: Pipeline state dictionary with 'model', 'test_df', 'sample_submission_path'

    Returns:
        Updated state dictionary with:
            - 'submission_path': Path to generated submission.csv file

    Note:
        Uses the same features that were used during training. If sample
        submission exists, copies its structure for proper format.
    """
    state = dict(state)
    model = state.get("model")
    test_df = state.get("test_df")
    sample_path = Path(state.get("sample_submission_path", ""))

    if model is None or test_df is None or test_df.empty:
        state["submission_path"] = ""
        if logger:
            logger.warning("No model or test data; skipping submission build.")
        return state

    # Те же признаки, что при обучении (sklearn feature_names_in_ или по колонкам train)
    if hasattr(model, "feature_names_in_"):
        feats = [c for c in model.feature_names_in_ if c in test_df.columns]
        X_test = test_df[feats] if feats else test_df.select_dtypes(include=["number"])
    else:
        X_test = test_df.select_dtypes(include=["number"])
        if X_test.empty:
            X_test = test_df.copy()
        train_df = state.get("train_df")
        if train_df is not None and state.get("target_column"):
            train_cols = [c for c in train_df.columns if c != state["target_column"]]
            available = [c for c in train_cols if c in X_test.columns]
            if available:
                X_test = X_test[available]

    preds = model.predict(X_test)
    out_path = ARTIFACTS_DIR / "submission.csv"

    try:
        import pandas as pd
        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            # Copy sample structure entirely and replace predictions
            submission = sample.copy()
            pred_col = sample.columns[1] if len(sample.columns) > 1 else "prediction"
            submission[pred_col] = preds
            # Keep only sample columns in correct order
            submission = submission[sample.columns]
        else:
            # Fallback: create index,prediction format like sample_submition.csv
            submission = pd.DataFrame({
                "index": range(len(preds)),
                "prediction": preds
            })
        submission.to_csv(out_path, index=False)
        if logger:
            logger.info("Submission created: columns=%s, shape=%s", list(submission.columns), submission.shape)
    except Exception as e:
        if logger:
            logger.warning("Submission build failed: %s", e)
        import pandas as pd
        # Fallback with correct format
        pd.DataFrame({"index": range(len(preds)), "prediction": preds}).to_csv(out_path, index=False)

    state["submission_path"] = str(out_path)
    if logger:
        logger.info("Step 4 submission saved to %s", out_path)
    return state


def _get_submission_info(s) -> dict[str, Any]:
    """Как в run_submit_and_metric.py."""
    return {
        "fileName": getattr(s, "fileName", getattr(s, "file_name", "N/A")),
        "date": getattr(s, "date", "N/A"),
        "description": getattr(s, "description", "N/A"),
        "status": getattr(s, "status", "N/A"),
        "publicScore": getattr(s, "publicScore", getattr(s, "public_score", None)),
        "privateScore": getattr(s, "privateScore", getattr(s, "private_score", None)),
    }


def step5_submit(state: dict[str, Any]) -> dict[str, Any]:
    """
    Submit the generated submission file to Kaggle competition.

    Uses the Kaggle API to upload the submission file. Requires proper
    authentication via KAGGLE_USERNAME and API_KAGGLE_KEY environment
    variables.

    Args:
        state: Pipeline state dictionary with 'submission_path'

    Returns:
        Updated state dictionary with:
            - 'submit_ok': Boolean indicating submission success
            - 'submit_error': Error message if submission failed (or None)

    Note:
        Make sure to accept the competition rules on Kaggle website before
        submitting, otherwise you'll get a 403 error.
    """
    state = dict(state)
    sub_path = state.get("submission_path", "")
    if not sub_path or not Path(sub_path).exists():
        state["submit_ok"] = False
        state["submit_error"] = "No submission file"
        if logger:
            logger.warning("Step 5: no submission file to submit.")
        return state

    _load_kaggle_env()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        state["submit_ok"] = False
        state["submit_error"] = "kaggle package not installed (use project venv: .venv/bin/python -m pip install kaggle)"
        if logger:
            logger.warning("Step 5: kaggle not installed. Use venv: .venv/bin/python 01_.py")
        return state
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = f"Kaggle init/auth: {e}"
        if logger:
            logger.warning("Step 5: Kaggle error (check kaggle-mcp/.env or ~/.kaggle): %s", e)
        return state

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = str(e)
        if logger:
            logger.error("Kaggle authenticate failed: %s", e)
        return state

    try:
        api.competition_submit(
            sub_path,
            "Submission from 01_.py pipeline",
            COMPETITION,
            quiet=False,
        )
        state["submit_ok"] = True
        state["submit_error"] = None
        if logger:
            logger.info("Step 5: submission submitted to %s", COMPETITION)
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = str(e)
        if logger:
            logger.error("Step 5 submit failed: %s", e)
    return state


def step6_wait_results(state: dict[str, Any]) -> dict[str, Any]:
    """
    Confirm submission was sent successfully.

    Note: Kaggle API does not support retrieving individual submission scores
    with the new KGAT_ token format. Check Kaggle website for results.

    Args:
        state: Pipeline state dictionary with 'submit_ok'

    Returns:
        Updated state dictionary with:
            - 'public_score': None (check Kaggle website)
            - 'private_score': None
            - 'submission_status': 'submitted_check_website'
    """
    state = dict(state)

    if not state.get("submit_ok"):
        state["public_score"] = None
        state["private_score"] = None
        state["submission_status"] = "not_submitted"
        return state

    if logger:
        logger.info("Step 6: submission sent successfully!")
        logger.info("Check https://www.kaggle.com/competitions/%s for your score", COMPETITION)

    state["public_score"] = None
    state["private_score"] = None
    state["submission_status"] = "submitted_check_website"
    return state


def step7_report(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate final pipeline report.

    Creates both JSON and text reports summarizing the entire pipeline run,
    including EDA summary, local metrics, submission status, and scores.
    Optionally generates an LLM-based summary.

    Args:
        state: Pipeline state dictionary with all accumulated results

    Returns:
        Updated state dictionary with:
            - 'report_path': Path to the text report file
            - 'report_llm_text': LLM-generated report (if available)

    Note:
        Reports are saved to artifacts/report_01.json and artifacts/report_01.txt
    """
    state = dict(state)
    report = {
        "eda_summary": state.get("eda_summary", "")[:1000],
        "local_metrics": state.get("local_metrics", {}),
        "model_path": state.get("model_path", ""),
        "submission_path": state.get("submission_path", ""),
        "submit_ok": state.get("submit_ok"),
        "public_score": state.get("public_score"),
        "private_score": state.get("private_score"),
        "submission_status": state.get("submission_status", ""),
        "submission_error": state.get("submission_error", ""),
    }

    report_path_json = ARTIFACTS_DIR / "report_01.json"
    report_path_txt = ARTIFACTS_DIR / "report_01.txt"
    with open(report_path_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "Report 01 — pipeline run",
        "=========================",
        "EDA summary: " + (report["eda_summary"] or "")[:500],
        "Local metrics: " + json.dumps(report["local_metrics"]),
        "Model: " + report["model_path"],
        "Submission: " + report["submission_path"],
        "Submitted: " + str(report["submit_ok"]),
        "Public score: " + str(report["public_score"]),
        "Private score: " + str(report["private_score"]),
        "Status: " + str(report["submission_status"]),
    ]
    if report.get("submission_error"):
        lines.append("Submission error: " + str(report["submission_error"])[:300])
        if "403" in str(report["submission_error"]):
            lines.append("Hint: примите правила соревнования на https://www.kaggle.com/competitions/" + COMPETITION)
    text_report = "\n".join(lines)
    with open(report_path_txt, "w", encoding="utf-8") as f:
        f.write(text_report)

    llm = _get_llm()
    if llm and (report["eda_summary"] or report["local_metrics"]):
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            prompt = ChatPromptTemplate.from_messages([
                ("human", "Кратко составь итоговый отчёт по запуску пайплайна (2–4 предложения):\n\n{data}"),
            ])
            chain = prompt | llm | StrOutputParser()
            data_str = json.dumps(report, ensure_ascii=False)[:3000]
            state["report_llm_text"] = chain.invoke({"data": data_str})
            with open(ARTIFACTS_DIR / "report_01_llm.txt", "w", encoding="utf-8") as f:
                f.write(state["report_llm_text"])
        except Exception as e:
            if logger:
                logger.warning("LLM report failed: %s", e)
            state["report_llm_text"] = text_report
    else:
        state["report_llm_text"] = text_report

    state["report_path"] = str(report_path_txt)
    if logger:
        logger.info("Step 7 report saved to %s and %s", report_path_json, report_path_txt)
    return state


def run_pipeline() -> dict[str, Any]:
    """
    Execute the complete ML pipeline sequentially.

    Runs all 7 steps in order:
    1. EDA - Exploratory data analysis
    2. Train - Model training
    3. Local Eval - Validation metrics
    4. Submission - Generate submission file
    5. Submit - Upload to Kaggle
    6. Wait Results - Poll for scores
    7. Report - Generate final report

    Returns:
        Final state dictionary with all pipeline results

    Example:
        >>> state = run_pipeline()
        >>> print(state.get('public_score'))
    """
    _setup_logging()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "data_dir": str(DATA_DIR),
        "train_path": str(DATA_DIR / TRAIN_FILE),
        "test_path": str(DATA_DIR / TEST_FILE),
        "submission_sample_path": str(DATA_DIR / SAMPLE_SUBMISSION_FILE),
    }

    steps = [step1_eda, step2_train, step3_local_eval, step4_submission, step5_submit, step6_wait_results, step7_report]
    for i, step_fn in enumerate(steps, 1):
        if logger:
            logger.info("Running step %d/%d: %s", i, len(steps), step_fn.__name__)
        try:
            state = step_fn(state)
        except Exception as e:
            if logger:
                logger.exception("Step %d failed: %s", i, e)
            state["last_error"] = str(e)
            break

    if logger:
        logger.info("Pipeline finished. Report: %s", state.get("report_path", "N/A"))
    return state


if __name__ == "__main__":
    run_pipeline()
