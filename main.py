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
    step_judge_result_agent,
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
    }

    if cfg.logger:
        cfg.logger.info("Loading data subset (%d%% of train)...", cfg.TRAIN_SAMPLE_PCT)
    state = cfg.load_data_subset(state)
    max_iterations = 3
    iteration = 0
    best_state = None
    best_metric = None

    while iteration < max_iterations:
        iteration += 1
        if cfg.logger:
            cfg.logger.info("=" * 60)
            cfg.logger.info("Refinement iteration %d/%d", iteration, max_iterations)

        # # Если есть предложения от судьи, передаём их в EDA/Train как подсказку
        # if state.get("verification_suggestions"):
        #     state["improvement_hint"] = state["verification_suggestions"]
        #     # if cfg.logger:
        #     #     cfg.logger.info("Using improvement hint: %s", state["improvement_hint"])
        # else:
        #     state["improvement_hint"] = ""

        state["improvement_hint"] = state.get("eda_improvement_hint", "")
        if cfg.logger and state["improvement_hint"]:
            cfg.logger.info(f"Using EDA hint: {state['improvement_hint']}")

        # -- Step 1: EDA --------------------------------------------------------
        if cfg.logger:
            cfg.logger.info("=" * 60)
            cfg.logger.info("Running step1_eda...")
        state = step1_eda_agent(state)

        state["improvement_hint"] = state.get("train_improvement_hint", "")
        if cfg.logger and state["improvement_hint"]:
            cfg.logger.info(f"Using Train hint: {state['improvement_hint']}")
        # -- Step 2: Train ------------------------------------------------------
        if cfg.logger:
            cfg.logger.info("=" * 60)
            cfg.logger.info("Running step2_train...")
        state = step2_train_agent(state)

        # state["improvement_hint"] = state.get("eval_improvement_hint", "")
        # if cfg.logger and state["improvement_hint"]:
        #     cfg.logger.info(f"Using Eval hint: {state['improvement_hint']}")
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
        if cfg.logger:
            cfg.logger.info("Running step_verify...")
        state = step_judge_result_agent(state)

        # Сохраняем лучшую модель по метрике (предполагаем, что в local_metrics есть "mse")
        current_metrics = state.get("local_metrics", {})
        current_mse = current_metrics.get("rmse")
        if current_mse is not None:
            if best_metric is None or current_mse < best_metric:
                best_metric = current_mse
                best_state = dict(state)  # сохраняем копию
                if cfg.logger:
                    cfg.logger.info("New best model found with MSE=%.4f", current_mse)

        # Проверяем решение судьи
        decision = state.get("verification_decision", "NEED_REFINEMENT")
        if decision == "SUFFICIENT":
            if cfg.logger:
                cfg.logger.info("Judge decision: SUFFICIENT – stopping refinement loop.")
            break
        else:
            if cfg.logger:
                cfg.logger.info("Judge decision: NEED_REFINEMENT – continuing loop.")

    if best_state is not None:
        if cfg.logger:
            cfg.logger.info("Using best model from iteration with MSE=%.4f", best_metric)
        state.update(best_state)  # заменить состояние на лучшее
    else:
        if cfg.logger:
            cfg.logger.warning("No valid iteration, using last state.")

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