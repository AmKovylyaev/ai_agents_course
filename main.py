"""
LLM Code Generation Agent for Kaggle Competitions

Runs a 7-step ML pipeline where an LLM generates Python code for each step.
Generated code is executed in a subprocess with retry logic (up to 3 attempts).
Deterministic fallbacks kick in when the LLM is unavailable or all retries fail.

Pipeline:
    1. EDA              (LLM)
    2. Train             (LLM)
    3. Eval              (LLM)
    4. Submission        (LLM)
    5. Submit to Kaggle  (API)
    6. Wait / confirm    (API)
    7. Report            (LLM)

Usage:
    venv/bin/python main.py

Requirements (in .env at project root):
    OPENROUTER_API_KEY   – for LLM access via OpenRouter
    API_KAGGLE_KEY       – Kaggle API token (KGAT_xxx or legacy)
"""

from __future__ import annotations

from typing import Any

import config as cfg
from steps_agent import (
    step1_eda_agent,
    step2_train_agent,
    step3_local_eval_agent,
    step4_submission_agent,
    step7_report_agent,
)
from steps_kaggle import step5_submit, step6_wait_results


def run_pipeline() -> dict[str, Any]:
    """Execute the full agent pipeline and return the final state dict."""
    session_dir = cfg.create_session_dir()

    cfg.setup_logging(session_dir)
    if cfg.logger:
        cfg.logger.info("Session directory: %s", session_dir)

    state: dict[str, Any] = {
        "session_dir": session_dir,
        "code_dir": session_dir / "code",
        "models_dir": session_dir / "models",
        "reports_dir": session_dir / "reports",
        "data_dir": str(cfg.DATA_DIR),
        "train_path": str(cfg.DATA_DIR / cfg.TRAIN_FILE),
        "test_path": str(cfg.DATA_DIR / cfg.TEST_FILE),
        "sample_submission_path": str(cfg.DATA_DIR / cfg.SAMPLE_SUBMISSION_FILE),
    }

    if cfg.logger:
        cfg.logger.info("Loading data subset (%d%% of train)...", cfg.TRAIN_SAMPLE_PCT)
    state = cfg.load_data_subset(state)

    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step1_eda...")
    state = step1_eda_agent(state)

    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step2_train...")
    state = step2_train_agent(state)

    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step3_eval...")
    state = step3_local_eval_agent(state)

    local_metrics = state.get("local_metrics", {})
    if cfg.logger and local_metrics:
        cfg.logger.info("Local evaluation metrics:")
        for metric_name, metric_value in local_metrics.items():
            if isinstance(metric_value, float):
                cfg.logger.info("  %s: %.4f", metric_name, metric_value)
            else:
                cfg.logger.info("  %s: %s", metric_name, metric_value)
    elif cfg.logger:
        cfg.logger.warning("No local metrics available after step3")

    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step4_submission...")
    state = step4_submission_agent(state)

    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step5_submit...")
    state = step5_submit(state)

    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step6_wait_results...")
    state = step6_wait_results(state)

    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step7_report...")
    state = step7_report_agent(state)

    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Pipeline finished. Session: %s", session_dir)
        cfg.logger.info("Report: %s", state.get("report_path", "N/A"))

    return state


if __name__ == "__main__":
    run_pipeline()
