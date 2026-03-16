"""
LLM Code Generation Agent for Kaggle Competitions

This module implements an agentic ML pipeline where an LLM generates
Python code for each pipeline step. The generated code is executed
in a subprocess with retry logic for error handling.

Key Features:
    - LLM-generated Python code for each step
    - Feedback loops for error correction (up to 3 attempts)
    - Timestamped session directories for artifacts
    - Uses only 20% of training data for speed
    - Fallback functions if LLM fails

Pipeline Steps:
    1. EDA - LLM generates code to analyze data
    2. Train - LLM generates model training code
    3. Eval - LLM generates evaluation code
    4. Submission - LLM generates submission creation code
    5. Submit - Direct Kaggle API submission (no LLM)
    6. Wait Results - Poll Kaggle leaderboard (no LLM)
    7. Report - LLM generates final report

Architecture:
    The agent uses OpenRouter API with ChatOpenAI for LLM access.
    Code is extracted from markdown code blocks, validated for syntax,
    and executed in subprocesses with state passed via JSON files.

Usage:
    .venv/bin/python ai_agents_course/final_project/ai_agent_step_by_step/02_.py

Requirements:
    - OPENROUTER_API_KEY in .env
    - API_KAGGLE_KEY in .env (new token format: KGAT_xxx)

Environment Variables:
    - OPENROUTER_API_KEY: API key for OpenRouter
    - API_KAGGLE_KEY: Your Kaggle API token (new format: KGAT_xxx)
    - KAGGLE_COMPETITION: Competition name (default: mws-ai-agents-2026)
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
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
SAMPLE_SUBMISSION_FILE = "sample_submition.csv"

# Kaggle competition configuration
# Can be overridden via environment variable KAGGLE_COMPETITION
COMPETITION = os.getenv("KAGGLE_COMPETITION", "mws-ai-agents-2026")

logger: logging.Logger | None = None


# ============================================================================
# ПРОМПТЫ ДЛЯ LLM
# ============================================================================

STEP1_EDA_PROMPT = """You are an ML engineer. Write Python code to perform Exploratory Data Analysis.

Context:
- Train data path: {train_path}
- Test data path: {test_path}
- Session directory: {session_dir}
- Use only 20% of training data for speed (sample with random_state=42)

Requirements:
1. Load train and test CSV files using pandas
2. Show basic statistics (shape, columns, dtypes, missing values)
3. Create visualizations if helpful (save to session_dir/reports/)
4. Save a summary report to session_dir/reports/eda_summary.txt
5. Save the loaded dataframes to JSON-serializable state dict with keys: train_shape, test_shape, columns, missing_values

Output ONLY executable Python code in a ```python code block.
Use pandas, numpy, and matplotlib/seaborn if needed.
The code should work standalone without user input.
At the end, update the 'state' dict with useful information.
"""

STEP2_TRAIN_PROMPT = """You are an ML engineer. Write Python code to train a classification model.

Context:
- Train data path: {train_path}
- Session directory: {session_dir}
- Previous error (if retry): {last_error}
- Previous code (if retry): {previous_code}

Requirements:
1. Load train data (20% subset with random_state=42)
2. Identify target column (usually 'target', 'label', or last column)
3. Prepare features (handle missing values, select numeric columns)
4. Split into train/validation (80/20, random_state=42)
5. Train a model (start simple: RandomForestClassifier with n_estimators=50, random_state=42)
6. Save model to session_dir/models/model.joblib
7. Print training metrics
8. Update state dict with: target_column, model_path, X_train_shape, X_val_shape

Output ONLY executable Python code in a ```python code block.
Use sklearn, joblib, pandas, numpy.
"""

STEP3_EVAL_PROMPT = """You are an ML engineer. Write Python code to evaluate the trained model locally.

Context:
- Model path: {model_path}
- Session directory: {session_dir}
- Previous error (if retry): {last_error}

Requirements:
1. Load the model from model_path using joblib
2. Load train data again (20% subset) and recreate the same train/val split (80/20, random_state=42)
3. Identify target column (same as training)
4. Prepare validation features (same columns as training)
5. Make predictions on validation set
6. Calculate metrics: accuracy, F1-macro, confusion matrix
7. Save metrics to session_dir/reports/local_metrics.json
8. Print evaluation results
9. Update state dict with: local_metrics (dict)

Output ONLY executable Python code in a ```python code block.
Use sklearn, joblib, pandas, numpy.
"""

STEP4_SUBMISSION_PROMPT = """You are an ML engineer. Write Python code to create a submission file.

Context:
- Model path: {model_path}
- Test data path: {test_path}
- Sample submission path: {sample_submission_path}
- Session directory: {session_dir}
- Target column used in training: {target_column}
- Previous error (if retry): {last_error}

Requirements:
1. Load the model using joblib
2. Load test data
3. Prepare features (same numeric columns as training)
4. Make predictions
5. Load sample_submission.csv to get correct format
6. Create submission CSV EXACTLY matching sample_submission.csv format:
   - Copy sample_submission to a new dataframe
   - REPLACE the prediction column values with your predictions (DO NOT add new columns!)
   - Keep ONLY the columns from sample_submission (usually id/index and prediction column)
   - The final submission must have THE SAME columns as sample_submission
7. Save to session_dir/submission.csv
8. Update state dict with: submission_path

CRITICAL: The submission file must have EXACTLY the same columns as sample_submission.csv!
If sample_submission has columns [index, prediction], your submission must have [index, prediction] only.

Output ONLY executable Python code in a ```python code block.
Use pandas, joblib, numpy.
"""

STEP7_REPORT_PROMPT = """You are an ML engineer. Write Python code to generate a final report.

Context:
- Session directory: {session_dir}
- EDA insights available in: session_dir/reports/eda_summary.txt
- Local metrics available in: session_dir/reports/local_metrics.json
- Submission status: {submit_ok}
- Kaggle scores: public={public_score}, private={private_score}

Requirements:
1. Load EDA summary from file if exists
2. Load local metrics from file if exists
3. Compile all results into a structured report
4. Save as JSON: session_dir/reports/final_report.json
5. Save as readable text: session_dir/reports/final_report.txt
6. Include in the report:
   - EDA insights
   - Model parameters
   - Local validation metrics
   - Kaggle submission results
   - Lessons learned and next steps

Output ONLY executable Python code in a ```python code block.
Use json, pathlib.
"""


# ============================================================================
# SETUP ФУНКЦИИ
# ============================================================================

def _setup_logging(session_dir: Path) -> None:
    """Настройка логирования: консоль + файл в сессии."""
    global logger
    log_file = session_dir / "run.log"
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


def _create_session_dir() -> Path:
    """Создает папку сессии с timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = ARTIFACTS_DIR / "sessions" / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    # Подпапки
    (session_dir / "code").mkdir(exist_ok=True)
    (session_dir / "models").mkdir(exist_ok=True)
    (session_dir / "reports").mkdir(exist_ok=True)

    return session_dir


def _get_llm():
    """LLM из config (model_llm). Провайдер — OpenRouter через ChatOpenAI."""
    try:
        import os
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        # Загружаем .env из корня проекта
        env_path = SCRIPT_DIR.parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")

        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_llm,
            temperature=0
        )
    except Exception as e:
        if logger:
            logger.warning("LLM unavailable (%s); using fallback functions.", e)
        return None


def load_data_subset(state: dict) -> dict:
    """Загружает данные, использует только 20% train для ускорения."""
    try:
        import pandas as pd
    except ImportError:
        if logger:
            logger.warning("pandas not installed; cannot load data.")
        return state

    state = dict(state)

    # Загружаем test полностью
    test_path = Path(state["test_path"])
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        state["test_df"] = test_df
        state["test_shape"] = test_df.shape
        if logger:
            logger.info("Loaded %d test samples", len(test_df))
    else:
        if logger:
            logger.warning("Test file not found: %s", test_path)

    # Загружаем только 20% train
    train_path = Path(state["train_path"])
    if train_path.exists():
        train_df_full = pd.read_csv(train_path)
        train_df = train_df_full.sample(frac=0.2, random_state=42)
        state["train_df"] = train_df
        state["train_df_full"] = train_df_full
        state["train_shape"] = train_df.shape
        if logger:
            logger.info("Loaded %d train samples (20%% of %d)", len(train_df), len(train_df_full))
    else:
        if logger:
            logger.warning("Train file not found: %s", train_path)

    return state


# ============================================================================
# CODE EXECUTION ENGINE
# ============================================================================

def extract_code_block(text: str) -> str | None:
    """Извлекает код из ```python ... ``` или ``` ... ```."""
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def validate_code(code: str) -> tuple[bool, str]:
    """Базовая валидация синтаксиса кода."""
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def execute_code(code: str, state: dict, timeout_sec: int = 60) -> tuple[bool, str, dict]:
    """
    Выполняет код в subprocess с доступом к state.
    Returns: (success, output, updated_state)
    """
    session_dir = state["session_dir"]

    # Подготовка: сохраняем state в JSON для передачи
    state_file = session_dir / "state_input.json"
    state_to_save = {}
    for k, v in state.items():
        # Пропускаем несериализуемые объекты
        if isinstance(v, Path):
            state_to_save[k] = str(v)
        elif hasattr(v, 'to_json'):  # pandas DataFrame
            continue
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            state_to_save[k] = v
        else:
            try:
                json.dumps(v)
                state_to_save[k] = v
            except:
                pass

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_to_save, f, ensure_ascii=False, indent=2, default=str)

    # Оборачиваем код для загрузки/сохранения state
    wrapped_code = f'''
import json
import sys
from pathlib import Path

# Загружаем state
with open("{state_file}", "r", encoding="utf-8") as f:
    state = json.load(f)

# Конвертируем строковые пути обратно в Path
for key in ["session_dir", "code_dir", "models_dir", "reports_dir", "data_dir",
            "train_path", "test_path", "sample_submission_path", "model_path", "submission_path"]:
    if key in state and isinstance(state[key], str):
        state[key] = Path(state[key])

# Пользовательский код
{code}

# Сохраняем обновленный state
state_output_file = Path("{session_dir}") / "state_output.json"
state_to_save = {{}}
for k, v in state.items():
    if isinstance(v, Path):
        state_to_save[k] = str(v)
    elif hasattr(v, 'to_json'):
        continue
    elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
        state_to_save[k] = v
    else:
        try:
            json.dumps(v)
            state_to_save[k] = v
        except:
            pass

with open(state_output_file, "w", encoding="utf-8") as f:
    json.dump(state_to_save, f, ensure_ascii=False, indent=2, default=str)

print("STATE_SAVED_SUCCESSFULLY")
'''

    # Выполняем в subprocess
    fd, path = tempfile.mkstemp(suffix=".py", prefix="agent_step_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(wrapped_code)

        result = subprocess.run(
            ["python3", path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(session_dir)
        )

        if result.returncode == 0:
            # Загружаем обновленный state
            output_file = session_dir / "state_output.json"
            if output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    updated_state = json.load(f)
                # Объединяем с оригинальным state, сохраняя несериализуемые объекты
                merged_state = dict(state)
                merged_state.update(updated_state)
                return True, result.stdout, merged_state
            return True, result.stdout, state
        else:
            error_msg = result.stderr or result.stdout or "Non-zero exit code"
            return False, error_msg, state

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Code execution exceeded time limit", state
    except Exception as e:
        return False, f"Execution error: {str(e)}", state
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ============================================================================
# LCEL CHAINS
# ============================================================================

def create_step_chain(prompt_template: str, llm):
    """Создает LCEL chain для шага."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])
    chain = prompt | llm | StrOutputParser()
    return chain


# ============================================================================
# FEEDBACK LOOP HANDLER
# ============================================================================

def run_step_with_retry(
    step_name: str,
    chain,
    state: dict,
    max_attempts: int = 3,
    timeout_sec: int = 120  # Increased from 60 to 120
) -> tuple[dict, bool]:
    """
    Универсальная функция для выполнения шага с retry.

    Returns: (updated_state, success)
    """
    attempts = 0
    errors = []

    while attempts < max_attempts:
        attempts += 1
        if logger:
            logger.info("%s: Attempt %d/%d", step_name, attempts, max_attempts)

        try:
            # 1. Генерируем код через LLM
            if logger:
                logger.info("%s: Generating code with LLM...", step_name)

            # Подготавливаем state для промпта (конвертируем Path в строки)
            prompt_state = {}
            for k, v in state.items():
                if isinstance(v, Path):
                    prompt_state[k] = str(v)
                elif isinstance(v, dict):
                    prompt_state[k] = str(v)
                else:
                    prompt_state[k] = v if v is not None else ""

            # Гарантируем наличие обязательных переменных для промптов
            prompt_state.setdefault("last_error", "")
            prompt_state.setdefault("previous_code", "")
            prompt_state.setdefault("model_path", state.get("model_path", ""))
            prompt_state.setdefault("target_column", state.get("target_column", ""))
            prompt_state.setdefault("submit_ok", state.get("submit_ok", False))
            prompt_state.setdefault("public_score", state.get("public_score", "N/A"))
            prompt_state.setdefault("private_score", state.get("private_score", "N/A"))
            prompt_state.setdefault("train_path", state.get("train_path", ""))
            prompt_state.setdefault("test_path", state.get("test_path", ""))
            prompt_state.setdefault("sample_submission_path", state.get("sample_submission_path", ""))
            prompt_state.setdefault("session_dir", str(state.get("session_dir", "")))

            code_response = chain.invoke(prompt_state)

            # 2. Извлекаем код из ответа
            code = extract_code_block(code_response)
            if not code:
                error_msg = "No code block found in LLM response"
                if logger:
                    logger.warning("%s: %s", step_name, error_msg)
                    logger.debug("LLM response: %s", code_response[:500])
                errors.append(error_msg)
                if attempts < max_attempts:
                    state["last_error"] = error_msg
                continue

            # 3. Валидируем синтаксис
            is_valid, validation_msg = validate_code(code)
            if not is_valid:
                if logger:
                    logger.warning("%s: Validation failed: %s", step_name, validation_msg)
                errors.append(validation_msg)
                if attempts < max_attempts:
                    state["last_error"] = validation_msg
                    state["previous_code"] = code
                continue

            # 4. Сохраняем сгенерированный код
            code_file = state["session_dir"] / "code" / f"{step_name}.py"
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            if logger:
                logger.info("%s: Code saved to %s", step_name, code_file)

            # 5. Выполняем код
            if logger:
                logger.info("%s: Executing code...", step_name)
            success, output, updated_state = execute_code(code, state, timeout_sec)

            if success:
                if logger:
                    logger.info("%s: Execution successful", step_name)
                    if output:
                        logger.debug("%s: Output: %s", step_name, output[:500])
                updated_state[f"{step_name}_attempts"] = attempts
                updated_state[f"{step_name}_success"] = True
                updated_state[f"{step_name}_code"] = code
                updated_state[f"{step_name}_output"] = output
                # Очищаем флаги ошибок
                updated_state.pop("last_error", None)
                updated_state.pop("previous_code", None)
                return updated_state, True
            else:
                error_msg = f"Execution failed: {output}"
                if logger:
                    logger.warning("%s: %s", step_name, error_msg)
                errors.append(error_msg)

                # Если это не последняя попытка, добавляем ошибку в state для retry
                if attempts < max_attempts:
                    state["last_error"] = error_msg
                    state["previous_code"] = code

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if logger:
                logger.error("%s: %s", step_name, error_msg)
            errors.append(error_msg)
            if attempts < max_attempts:
                state["last_error"] = error_msg

    # Все попытки исчерпаны
    if logger:
        logger.error("%s: All %d attempts failed", step_name, max_attempts)
    state[f"{step_name}_attempts"] = attempts
    state[f"{step_name}_success"] = False
    state[f"{step_name}_errors"] = errors
    return state, False


# ============================================================================
# AGENT STEPS (С LLM)
# ============================================================================

def step1_eda_agent(state: dict) -> dict:
    """Шаг 1: EDA сгенерированный LLM."""
    state = dict(state)
    llm = _get_llm()

    if not llm:
        if logger:
            logger.warning("No LLM available, using fallback EDA")
        return step1_eda_fallback(state)

    # Создаем chain
    chain = create_step_chain(STEP1_EDA_PROMPT, llm)

    # Выполняем с retry
    state, success = run_step_with_retry(
        "step1_eda",
        chain,
        state,
        max_attempts=3,
        timeout_sec=60
    )

    if not success:
        if logger:
            logger.warning("EDA agent failed, using fallback")
        return step1_eda_fallback(state)

    return state


def step2_train_agent(state: dict) -> dict:
    """Шаг 2: Обучение модели сгенерированное LLM."""
    state = dict(state)
    llm = _get_llm()

    if not llm:
        if logger:
            logger.warning("No LLM available, using fallback train")
        return step2_train_fallback(state)

    # Создаем chain
    chain = create_step_chain(STEP2_TRAIN_PROMPT, llm)

    # Выполняем с retry
    state, success = run_step_with_retry(
        "step2_train",
        chain,
        state,
        max_attempts=3,
        timeout_sec=120  # Обучение может занять больше времени
    )

    if not success:
        if logger:
            logger.warning("Train agent failed, using fallback")
        return step2_train_fallback(state)

    return state


def step3_local_eval_agent(state: dict) -> dict:
    """Шаг 3: Локальная оценка модели сгенерированная LLM."""
    state = dict(state)
    llm = _get_llm()

    if not llm:
        if logger:
            logger.warning("No LLM available, using fallback eval")
        return step3_local_eval_fallback(state)

    # Создаем chain
    chain = create_step_chain(STEP3_EVAL_PROMPT, llm)

    # Выполняем с retry
    state, success = run_step_with_retry(
        "step3_eval",
        chain,
        state,
        max_attempts=3,
        timeout_sec=60
    )

    if not success:
        if logger:
            logger.warning("Eval agent failed, using fallback")
        return step3_local_eval_fallback(state)

    return state


def step4_submission_agent(state: dict) -> dict:
    """Шаг 4: Создание submission сгенерированное LLM."""
    state = dict(state)
    llm = _get_llm()

    if not llm:
        if logger:
            logger.warning("No LLM available, using fallback submission")
        return step4_submission_fallback(state)

    # Создаем chain
    chain = create_step_chain(STEP4_SUBMISSION_PROMPT, llm)

    # Выполняем с retry
    state, success = run_step_with_retry(
        "step4_submission",
        chain,
        state,
        max_attempts=3,
        timeout_sec=60
    )

    if not success:
        if logger:
            logger.warning("Submission agent failed, using fallback")
        return step4_submission_fallback(state)

    return state


def step7_report_agent(state: dict) -> dict:
    """Шаг 7: Создание отчета сгенерированное LLM."""
    state = dict(state)
    llm = _get_llm()

    if not llm:
        if logger:
            logger.warning("No LLM available, using fallback report")
        return step7_report_fallback(state)

    # Создаем chain
    chain = create_step_chain(STEP7_REPORT_PROMPT, llm)

    # Выполняем с retry
    state, success = run_step_with_retry(
        "step7_report",
        chain,
        state,
        max_attempts=3,
        timeout_sec=60
    )

    if not success:
        if logger:
            logger.warning("Report agent failed, using fallback")
        return step7_report_fallback(state)

    return state


# ============================================================================
# FALLBACK ФУНКЦИИ (БЕЗ LLM, ИЗ 01_.py)
# ============================================================================

def step1_eda_fallback(state: dict) -> dict:
    """Fallback EDA без LLM (из 01_.py)."""
    state = dict(state)
    train_path = Path(state.get("train_path", ""))
    test_path = Path(state.get("test_path", ""))
    sample_path = Path(state.get("sample_submission_path", ""))

    try:
        import pandas as pd
    except ImportError:
        if logger:
            logger.warning("pandas not installed; skipping EDA.")
        state["eda_report"] = ""
        return state

    report_parts = []

    if train_path.exists():
        train_df = pd.read_csv(train_path)
        report_parts.append(f"Train: {len(train_df)} rows, {len(train_df.columns)} columns")
        report_parts.append(f"Columns: {list(train_df.columns)}")
        report_parts.append(str(train_df.describe()))
        report_parts.append(f"Missing: {train_df.isnull().sum().to_dict()}")
        state["train_df"] = train_df
        state["train_shape"] = train_df.shape
    else:
        if logger:
            logger.warning("Train file not found: %s", train_path)

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        report_parts.append(f"Test: {len(test_df)} rows, {len(test_df.columns)} columns")
        state["test_df"] = test_df
        state["test_shape"] = test_df.shape
    else:
        if logger:
            logger.warning("Test file not found: %s", test_path)

    eda_text = "\n\n".join(report_parts) if report_parts else "No data loaded."
    state["eda_report"] = eda_text

    # Сохраняем отчет
    eda_file = state["session_dir"] / "reports" / "eda_summary.txt"
    with open(eda_file, "w", encoding="utf-8") as f:
        f.write(eda_text)

    state["step1_eda_success"] = True
    if logger:
        logger.info("Step 1 EDA (fallback) done")
    return state


def step2_train_fallback(state: dict) -> dict:
    """Fallback train без LLM (из 01_.py)."""
    state = dict(state)

    try:
        import pandas as pd
        import joblib
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        if logger:
            logger.warning("sklearn/joblib not available: %s", e)
        state["model_path"] = ""
        return state

    # Загружаем данные (20% subset)
    train_path = Path(state.get("train_path", ""))
    if not train_path.exists():
        if logger:
            logger.warning("No train data; skipping training.")
        state["model_path"] = ""
        return state

    train_df_full = pd.read_csv(train_path)
    train_df = train_df_full.sample(frac=0.2, random_state=42)

    # Целевая колонка
    target_candidates = ["target", "label", "y"]
    target_col = None
    for c in target_candidates:
        if c in train_df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = train_df.columns[-1]

    if logger:
        logger.info("Using target column: %s", target_col)

    X = train_df.drop(columns=[target_col], errors="ignore").select_dtypes(include=["number"])
    if X.empty:
        X = train_df.drop(columns=[target_col], errors="ignore")
    y = train_df[target_col]
    state["target_column"] = target_col

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    model_path = state["session_dir"] / "models" / "model.joblib"
    joblib.dump(model, model_path)
    state["model_path"] = str(model_path)
    state["model"] = model

    state["step2_train_success"] = True
    if logger:
        logger.info("Step 2 train (fallback) done; model saved to %s", model_path)
    return state


def step3_local_eval_fallback(state: dict) -> dict:
    """Fallback eval без LLM (из 01_.py)."""
    state = dict(state)
    model_path = Path(state.get("model_path", ""))

    if not model_path.exists():
        state["local_metrics"] = {}
        if logger:
            logger.warning("No model found; skipping local eval.")
        return state

    try:
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        state["local_metrics"] = {}
        return state

    # Загружаем модель
    model = joblib.load(model_path)

    # Загружаем данные и валидируем
    train_path = Path(state.get("train_path", ""))
    train_df_full = pd.read_csv(train_path)
    train_df = train_df_full.sample(frac=0.2, random_state=42)

    target_col = state.get("target_column", train_df.columns[-1])
    X = train_df.drop(columns=[target_col], errors="ignore").select_dtypes(include=["number"])
    y = train_df[target_col]

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    try:
        f1 = f1_score(y_val, pred, average="macro")
    except Exception:
        f1 = 0.0

    state["local_metrics"] = {"accuracy": acc, "f1_macro": f1}

    # Сохраняем метрики
    metrics_file = state["session_dir"] / "reports" / "local_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(state["local_metrics"], f, indent=2)

    state["step3_eval_success"] = True
    if logger:
        logger.info("Step 3 local eval (fallback): accuracy=%.4f, f1_macro=%.4f", acc, f1)
    return state


def step4_submission_fallback(state: dict) -> dict:
    """Fallback submission без LLM (из 01_.py)."""
    state = dict(state)
    model_path = Path(state.get("model_path", ""))

    if not model_path.exists():
        state["submission_path"] = ""
        if logger:
            logger.warning("No model; skipping submission build.")
        return state

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

    # Признаки
    if hasattr(model, "feature_names_in_"):
        feats = [c for c in model.feature_names_in_ if c in test_df.columns]
        X_test = test_df[feats] if feats else test_df.select_dtypes(include=["number"])
    else:
        X_test = test_df.select_dtypes(include=["number"])

    preds = model.predict(X_test)
    out_path = state["session_dir"] / "submission.csv"

    sample_path = Path(state.get("sample_submission_path", ""))
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        # Копируем sample submission и заменяем только колонку предсказаний
        out_df = sample.copy()
        pred_col = sample.columns[1] if len(sample.columns) > 1 else "prediction"
        out_df[pred_col] = preds
        # Оставляем только те колонки, что в sample_submission
        out_df = out_df[sample.columns]
    else:
        id_col = "id" if "id" in test_df.columns else test_df.columns[0]
        out_df = test_df[[id_col]].copy() if id_col in test_df.columns else pd.DataFrame({"id": range(len(preds))})
        out_df["prediction"] = preds

    out_df.to_csv(out_path, index=False)
    state["submission_path"] = str(out_path)

    state["step4_submission_success"] = True
    if logger:
        logger.info("Step 4 submission (fallback) saved to %s", out_path)
    return state


def step7_report_fallback(state: dict) -> dict:
    """Fallback report без LLM (из 01_.py)."""
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

    report_path_json = state["session_dir"] / "reports" / "final_report.json"
    report_path_txt = state["session_dir"] / "reports" / "final_report.txt"

    with open(report_path_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "Report 02 — Agent Pipeline Run",
        "===============================",
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

    if logger:
        logger.info("Step 7 report (fallback) saved to %s", report_path_txt)
    return state


# ============================================================================
# KAGGLE STEPS (БЕЗ LLM)
# ============================================================================

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


def step5_submit(state: dict) -> dict:
    """Шаг 5: Отправка submission на Kaggle."""
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
        state["submit_error"] = "kaggle package not installed"
        if logger:
            logger.warning("Step 5: kaggle not installed: %s", e)
        return state
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = f"Kaggle init/auth: {e}"
        if logger:
            logger.warning("Step 5: Kaggle error: %s", e)
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
            "Submission from 02_.py agent pipeline",
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


def step6_wait_results(state: dict) -> dict:
    """Шаг 6: Подтверждение отправки submission.

    Примечание: Мы не получаем метрики с leaderboard - только подтверждаем,
    что файл был успешно отправлен в Kaggle.
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


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def run_pipeline() -> dict[str, Any]:
    """Запуск агентной цепочки."""
    # Создаем сессию
    session_dir = _create_session_dir()

    # Настройка логирования в сессию
    _setup_logging(session_dir)
    if logger:
        logger.info("Session directory: %s", session_dir)

    # Инициализируем state
    state = {
        "session_dir": session_dir,
        "code_dir": session_dir / "code",
        "models_dir": session_dir / "models",
        "reports_dir": session_dir / "reports",
        "data_dir": str(DATA_DIR),
        "train_path": str(DATA_DIR / TRAIN_FILE),
        "test_path": str(DATA_DIR / TEST_FILE),
        "sample_submission_path": str(DATA_DIR / SAMPLE_SUBMISSION_FILE),
    }

    # Загружаем данные (20% train)
    if logger:
        logger.info("Loading data subset (20%% of train)...")
    state = load_data_subset(state)

    # Выполняем агентные шаги (с LLM)
    agent_steps = [
        ("step1_eda", step1_eda_agent),
        ("step2_train", step2_train_agent),
        ("step3_eval", step3_local_eval_agent),
        ("step4_submission", step4_submission_agent),
    ]

    for step_name, step_fn in agent_steps:
        if logger:
            logger.info("=" * 60)
            logger.info("Running %s...", step_name)
        try:
            state = step_fn(state)
            if not state.get(f"{step_name}_success", True):
                if logger:
                    logger.warning("%s failed, continuing with fallback...", step_name)
        except Exception as e:
            if logger:
                logger.error("%s crashed: %s", step_name, e)
            state[f"{step_name}_error"] = str(e)

    # Шаги без LLM (Kaggle)
    if logger:
        logger.info("=" * 60)
        logger.info("Running step5_submit...")
    state = step5_submit(state)

    if logger:
        logger.info("=" * 60)
        logger.info("Running step6_wait_results...")
    state = step6_wait_results(state)

    # Финальный отчет
    if logger:
        logger.info("=" * 60)
        logger.info("Running step7_report...")
    state = step7_report_agent(state)

    if logger:
        logger.info("=" * 60)
        logger.info("Pipeline finished. Session: %s", session_dir)
        logger.info("Report: %s", state.get("report_path", "N/A"))

    return state


if __name__ == "__main__":
    run_pipeline()
