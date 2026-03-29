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
    model_llm = "openai/gpt-oss-120b"
    model_embedding = "google/gemini-embedding-001"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submition.csv"

COMPETITION = os.getenv("KAGGLE_COMPETITION", "mws-ai-agents-2026")

# TRAIN_SAMPLE_PCT: percentage of all training data to use (for lightweight runs).
#   100 = use all data, 20 = use 20% of data.
TRAIN_SAMPLE_PCT = 100

# TRAIN_SAMPLE_FRAC: fraction of the (subsetted) training data held out for validation.
#   0.2 = 80% train / 20% validation split.
TRAIN_SAMPLE_FRAC = 0.2

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
    for noisy in (
        "httpx", "httpcore", "openai",
        "langchain", "langchain_core", "langchain_openai",
        "langgraph", "langsmith",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from langchain_core.globals import set_verbose, set_debug
    set_verbose(False)
    set_debug(False)

    logger = logging.getLogger(__name__)
    logger.info("Logging to %s", log_file)


def create_session_dir() -> Path:
    """Create a timestamped session directory with sub-folders."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = ARTIFACTS_DIR / "sessions" / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("models", "reports"):
        (session_dir / sub).mkdir(exist_ok=True)

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
    """Load data, subset train to TRAIN_SAMPLE_PCT%, save subset to session dir.

    After this function, state["train_path"] points to the subsetted CSV
    so all downstream steps just load it directly — no sampling needed.
    """
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
        state["test_shape"] = list(test_df.shape)
        if logger:
            logger.info("Loaded %d test samples", len(test_df))
    else:
        if logger:
            logger.warning("Test file not found: %s", test_path)

    train_path_orig = Path(state["train_path"])
    if train_path_orig.exists():
        train_df_full = pd.read_csv(train_path_orig)
        train_df = train_df_full.sample(frac=TRAIN_SAMPLE_PCT / 100, random_state=42)

        subset_path = Path(state["session_dir"]) / "train_subset.csv"
        train_df.to_csv(subset_path, index=False)
        state["train_path"] = str(subset_path)
        state["train_path_original"] = str(train_path_orig)
        state["train_shape"] = list(train_df.shape)

        if logger:
            logger.info(
                "Subsetted %d train samples (%d%% of %d) → %s",
                len(train_df), TRAIN_SAMPLE_PCT, len(train_df_full), subset_path,
            )
    else:
        if logger:
            logger.warning("Train file not found: %s", train_path_orig)

    return state