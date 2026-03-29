"""
LLM Code Generation Agent for Kaggle Competitions

Runs a 7-step ML pipeline where an LLM generates Python code for each step.
Generated code is executed in a subprocess with retry logic (up to 3 attempts).
Deterministic fallbacks kick in when the LLM is unavailable or all retries fail.

Pipeline:
    1. EDA              (LLM)
    2. Train             (LLM)
    3. Eval              (LLM)
    -- Judge loop (repeat 1-3 if NEED_REFINEMENT) --
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

from pathlib import Path
from typing import Any

import config as cfg
from config import log as _log
from steps_agent import (
    step1_eda_agent,
    step2_train_agent,
    step3_local_eval_agent,
    step4_submission_agent,
    step_judge_result_agent,
    step7_report_agent,
)
from steps_kaggle import step5_submit, step6_wait_results


def _log_metrics(local_metrics: dict) -> None:
    if not cfg.logger or not local_metrics:
        return
    cfg.logger.info("Local evaluation metrics:")
    for split in ("train", "val"):
        split_m = local_metrics.get(split)
        if isinstance(split_m, dict):
            cfg.logger.info("  [%s]", split)
            for k, v in split_m.items():
                if isinstance(v, (int, float)):
                    cfg.logger.info("    %s: %.4f", k, v)
                else:
                    cfg.logger.info("    %s: %s", k, v)
    if "train" not in local_metrics and "val" not in local_metrics:
        for k, v in local_metrics.items():
            if isinstance(v, (int, float)):
                cfg.logger.info("  %s: %.4f", k, v)
            else:
                cfg.logger.info("  %s: %s", k, v)


def _setup_iteration_dirs(session_dir: Path, iteration: int) -> dict[str, str]:
    """Create per-iteration subdirectories and return path strings."""
    iter_dir = session_dir / f"iter_{iteration}"
    iter_dir.mkdir(exist_ok=True)
    return {"iter_dir": str(iter_dir)}


def run_pipeline(max_iterations: int = 3) -> dict[str, Any]:
    """Execute the full agent pipeline and return the final state dict."""
    session_dir = cfg.create_session_dir()
    cfg.setup_logging(session_dir)
    _log("Session directory: %s", session_dir)

    state: dict[str, Any] = {
        "session_dir": str(session_dir),
        "reports_dir": str(session_dir / "reports"),
        "data_dir": str(cfg.DATA_DIR),
        "train_path": str(cfg.DATA_DIR / cfg.TRAIN_FILE),
        "test_path": str(cfg.DATA_DIR / cfg.TEST_FILE),
        "sample_submission_path": str(cfg.DATA_DIR / cfg.SAMPLE_SUBMISSION_FILE),
        "train_sample_frac": cfg.TRAIN_SAMPLE_FRAC,
        "train_sample_pct": cfg.TRAIN_SAMPLE_PCT,
        "model_path": str(session_dir / "models" / "pipeline.joblib"),
        "submission_path": str(session_dir / "submission.csv"),
        "target_column": "",
    }

    _log("Loading data subset (%d%% of train)...", cfg.TRAIN_SAMPLE_PCT)
    state = cfg.load_data_subset(state)

    best_state = None
    best_metric = None

    for iteration in range(1, max_iterations + 1):
        _log("=" * 60)
        _log("Refinement iteration %d/%d", iteration, max_iterations)

        iter_paths = _setup_iteration_dirs(session_dir, iteration)
        state.update(iter_paths)
        state["iteration"] = iteration

        # -- Step 1: EDA --------------------------------------------------------
        _log("Step 1 — EDA")
        state["improvement_hint"] = state.get("eda_improvement_hint", "")
        state = step1_eda_agent(state)

        # -- Step 2: Train ------------------------------------------------------
        _log("Step 2 — Train")
        state["improvement_hint"] = state.get("train_improvement_hint", "")
        state = step2_train_agent(state)

        # -- Step 3: Eval -------------------------------------------------------
        _log("Step 3 — Eval")
        state = step3_local_eval_agent(state)

        _log_metrics(state.get("local_metrics", {}))

        # -- Judge --------------------------------------------------------------
        _log("Judge — evaluating results...")
        state = step_judge_result_agent(state)

        val_metrics = state.get("local_metrics", {}).get("val", state.get("local_metrics", {}))
        if isinstance(val_metrics, dict):
            task = state.get("task_type", "classification")
            if task == "regression":
                current = val_metrics.get("mse")
                is_better = current is not None and (best_metric is None or current < best_metric)
            else:
                current = val_metrics.get("f1") or val_metrics.get("accuracy")
                is_better = current is not None and (best_metric is None or current > best_metric)
            if is_better:
                best_metric = current
                best_state = dict(state)
                metric_name = "MSE" if task == "regression" else "F1/Acc"
                _log("New best model (%s=%.4f)", metric_name, current)

        decision = state.get("verification_decision", "NEED_REFINEMENT")
        if decision == "SUFFICIENT":
            _log("Judge: SUFFICIENT — stopping refinement loop.")
            break
        _log("Judge: NEED_REFINEMENT — continuing loop.")

    if best_state is not None:
        _log("Using best model from iterations (best=%.4f)", best_metric)
        state.update(best_state)
    else:
        _log("No valid metric tracked, using last state.", level="warning")

    # -- Step 4: Submission -------------------------------------------------
    _log("=" * 60)
    _log("Step 4 — Submission")
    state = step4_submission_agent(state)

    # -- Step 5: Submit to Kaggle -------------------------------------------
    _log("=" * 60)
    _log("Step 5 — Submit to Kaggle")
    state = step5_submit(state)

    # -- Step 6: Wait / confirm ---------------------------------------------
    _log("=" * 60)
    _log("Step 6 — Wait for results")
    state = step6_wait_results(state)

    # -- Step 7: Report -----------------------------------------------------
    _log("=" * 60)
    _log("Step 7 — Report")
    state = step7_report_agent(state)

    _log("=" * 60)
    _log("Pipeline finished. Session: %s", session_dir)
    _log("Report: %s", state.get("report_path", "N/A"))

    return state


if __name__ == "__main__":
    run_pipeline()
