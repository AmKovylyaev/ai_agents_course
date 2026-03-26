"""Shared configuration: paths, model settings, logger, and setup helpers."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from _model_config import model_llm, model_embedding
except ImportError:
    model_llm = "z-ai/glm-4.7-flash"
    model_embedding = "google/gemini-embedding-001"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submition.csv"

COMPETITION = os.getenv("KAGGLE_COMPETITION", "mws-ai-agents-2026")

logger: logging.Logger | None = None


def setup_logging(session_dir: Path) -> None:
    """Configure console + file logging for a session."""
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


def create_session_dir() -> Path:
    """Create a timestamped session directory with sub-folders."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = ARTIFACTS_DIR / "sessions" / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    (session_dir / "code").mkdir(exist_ok=True)
    (session_dir / "models").mkdir(exist_ok=True)
    (session_dir / "reports").mkdir(exist_ok=True)

    return session_dir


def get_llm():
    """Return a ChatOpenAI instance via OpenRouter, or None on failure."""
    try:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        env_path = SCRIPT_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env")

        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_llm,
            temperature=0,
        )
    except Exception as e:
        if logger:
            logger.warning("LLM unavailable (%s); using fallback functions.", e)
        return None


def load_data_subset(state: dict) -> dict:
    """Load data files; use only 20 % of train for speed."""
    try:
        import pandas as pd
    except ImportError:
        if logger:
            logger.warning("pandas not installed; cannot load data.")
        return state

    state = dict(state)

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
