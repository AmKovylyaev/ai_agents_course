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

from pathlib import Path
from rag.rag_tools import build_notebook_rag_index

import config as cfg
from steps_agent import (
    step1_eda_agent,
    step2_train_agent,
    step3_local_eval_agent,
    step4_submission_agent,
    step7_report_agent,
)
from steps_kaggle import step5_submit, step6_wait_results

from pathlib import Path
from rag.rag_tools import build_notebook_rag_index

def run_pipeline() -> dict[str, Any]:
    """Execute the full agent pipeline and return the final state dict."""
    session_dir = cfg.create_session_dir()

    cfg.setup_logging(session_dir)
    if cfg.logger:
        cfg.logger.info("Session directory: %s", session_dir)

    state: dict[str, Any] = {
        # directories
        "session_dir": str(session_dir),
        "code_dir": str(session_dir / "code"),
        "models_dir": str(session_dir / "models"),
        "reports_dir": str(session_dir / "reports"),
        "plans_dir": str(session_dir / "plans"),
        "feedback_dir": str(session_dir / "feedback"),
        "data_dir": str(cfg.DATA_DIR),
        # data paths
        "train_path": str(cfg.DATA_DIR / cfg.TRAIN_FILE),
        "test_path": str(cfg.DATA_DIR / cfg.TEST_FILE),
        "sample_submission_path": str(cfg.DATA_DIR / cfg.SAMPLE_SUBMISSION_FILE),
        # sampling config
        "train_sample_frac": cfg.TRAIN_SAMPLE_FRAC,
        "train_sample_pct": cfg.TRAIN_SAMPLE_PCT,
        # populated by later steps (defaults to conventional locations)
        "model_path": str(session_dir / "models" / "pipeline.joblib"),
        "submission_path": str(session_dir / "submission.csv"),
        "target_column": "",
        # rag
        "rag_enabled": cfg.RAG_ENABLED,
        "rag_top_k": cfg.RAG_TOP_K,
        "rag_search_type": cfg.RAG_SEARCH_TYPE,
        "rag_embedding_model": cfg.RAG_EMBEDDING_MODEL,
        "notebooks_kb_dir": str(cfg.NOTEBOOKS_KB_DIR),
        "rag_index_dir": str(cfg.RAG_INDEX_DIR),
        "rag_context": "",
        "rag_results": [],
        "rag_query": "",
        "web_search_enabled": cfg.WEB_SEARCH_ENABLED,
        "web_search_max_results": cfg.WEB_SEARCH_MAX_RESULTS,
        "web_query": "",
        "web_context": "",
        "web_results": [],
    }

    if cfg.logger:
        cfg.logger.info("Loading data subset (%d%% of train)...", cfg.TRAIN_SAMPLE_PCT)
    state = cfg.load_data_subset(state)

    if state.get("rag_enabled", False):
        kb_dir = Path(state["notebooks_kb_dir"])
        if kb_dir.exists():
            rag_stats = build_notebook_rag_index(
                notebooks_root=kb_dir,
                index_dir=state["rag_index_dir"],
                max_chars=1600,
                embedding_model=state["rag_embedding_model"],
            )
            state["rag_stats"] = rag_stats
            if cfg.logger:
                cfg.logger.info("RAG index built: %s", rag_stats)
        else:
            state["rag_stats"] = {
                "notebooks_root": str(kb_dir),
                "index_dir": state["rag_index_dir"],
                "n_chunks": 0,
            }
            if cfg.logger:
                cfg.logger.warning("Notebook KB directory does not exist: %s", kb_dir)

    # -- Step 1: EDA --------------------------------------------------------
    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step1_eda...")
    state = step1_eda_agent(state)

    # -- Step 2: Train ------------------------------------------------------
    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step2_train...")
    state = step2_train_agent(state)

    # -- Step 3: Eval -------------------------------------------------------
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

    # -- Step 4: Submission -------------------------------------------------
    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step4_submission...")
    state = step4_submission_agent(state)

    # -- Step 5: Submit to Kaggle -------------------------------------------
    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step5_submit...")
    state = step5_submit(state)

    # -- Step 6: Wait / confirm ---------------------------------------------
    if cfg.logger:
        cfg.logger.info("=" * 60)
        cfg.logger.info("Running step6_wait_results...")
    state = step6_wait_results(state)

    # -- Step 7: Report -----------------------------------------------------
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
