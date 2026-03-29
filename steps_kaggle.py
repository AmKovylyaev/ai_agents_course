"""Kaggle-related pipeline steps (no LLM): submit and wait for results."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import config as cfg


def _load_kaggle_env() -> None:
    """Load Kaggle credentials from the project .env file."""
    from dotenv import load_dotenv

    env_path = cfg.SCRIPT_DIR / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        if cfg.logger:
            cfg.logger.info("Loaded environment from %s", env_path)
    else:
        if cfg.logger:
            cfg.logger.warning("No .env file found at %s", env_path)

    api_kaggle_key = os.getenv("API_KAGGLE_KEY")
    if api_kaggle_key:
        if api_kaggle_key.startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = api_kaggle_key
            if cfg.logger:
                cfg.logger.info("Kaggle API token configured (KGAT_ format)")
        else:
            os.environ["KAGGLE_KEY"] = api_kaggle_key
            if cfg.logger:
                cfg.logger.info("Kaggle API key configured (legacy format)")
    else:
        if cfg.logger:
            cfg.logger.warning("API_KAGGLE_KEY not found in environment")


def _get_submission_info(s) -> dict[str, Any]:
    """Extract useful fields from a Kaggle submission object."""
    return {
        "fileName": getattr(s, "fileName", getattr(s, "file_name", "N/A")),
        "date": getattr(s, "date", "N/A"),
        "description": getattr(s, "description", "N/A"),
        "status": getattr(s, "status", "N/A"),
        "publicScore": getattr(s, "publicScore", getattr(s, "public_score", None)),
        "privateScore": getattr(s, "privateScore", getattr(s, "private_score", None)),
    }


def step5_submit(state: dict) -> dict:
    """Step 5: Upload submission.csv to Kaggle."""
    state = dict(state)
    sub_path = state.get("submission_path", "")
    if not sub_path or not Path(sub_path).exists():
        state["submit_ok"] = False
        state["submit_error"] = "No submission file"
        if cfg.logger:
            cfg.logger.warning("Step 5: no submission file to submit.")
        return state

    _load_kaggle_env()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        state["submit_ok"] = False
        state["submit_error"] = "kaggle package not installed"
        if cfg.logger:
            cfg.logger.warning("Step 5: kaggle not installed: %s", e)
        return state
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = f"Kaggle init/auth: {e}"
        if cfg.logger:
            cfg.logger.warning("Step 5: Kaggle error: %s", e)
        return state

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = str(e)
        if cfg.logger:
            cfg.logger.error("Kaggle authenticate failed: %s", e)
        return state

    try:
        api.competition_submit(
            sub_path,
            "Submission from agent pipeline",
            cfg.COMPETITION,
            quiet=False,
        )
        state["submit_ok"] = True
        state["submit_error"] = None
        if cfg.logger:
            cfg.logger.info("Step 5: submission submitted to %s", cfg.COMPETITION)
    except Exception as e:
        state["submit_ok"] = False
        state["submit_error"] = str(e)
        if cfg.logger:
            cfg.logger.error("Step 5 submit failed: %s", e)
    return state


def step6_wait_results(state: dict) -> dict:
    """Step 6: Confirm submission status (scores checked on the website)."""
    state = dict(state)
    if not state.get("submit_ok"):
        state["public_score"] = None
        state["private_score"] = None
        state["submission_status"] = "not_submitted"
        return state

    if cfg.logger:
        cfg.logger.info("Step 6: submission sent successfully!")
        cfg.logger.info(
            "Check https://www.kaggle.com/competitions/%s for your score",
            cfg.COMPETITION,
        )

    state["public_score"] = None
    state["private_score"] = None
    state["submission_status"] = "submitted_check_website"
    return state
