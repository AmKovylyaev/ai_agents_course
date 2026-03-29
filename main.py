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
from rag.rag_tools import build_notebook_rag_index
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


def run_candidate_branch(branch_id: int, iteration: int, base_state: dict, session_dir: Path) -> dict[str, Any]:
    """
    Отдельная ветка исполнения: EDA -> Train -> Eval.
    Это позволяет запускать несколько гипотез параллельно.
    """
    _log(f"[Iter {iteration} | Branch {branch_id}] Starting research branch...")

    # Создаем изолированное состояние для ветки
    branch_state = dict(base_state)
    branch_temp = (branch_id - 1) * 0.4

    # Генерируем уникальный путь: session/iter_1/branch_1/
    branch_dir = session_dir / f"iter_{iteration}" / f"branch_{branch_id}"
    branch_dir.mkdir(parents=True, exist_ok=True)
    (branch_dir / "reports").mkdir(exist_ok=True)
    (branch_dir / "models").mkdir(exist_ok=True)

    branch_state.update({
        "temperature": branch_temp,
        "iter_dir": str(branch_dir),
        "reports_dir": str(branch_dir / "reports"),
        "model_path": str(branch_dir / "models" / "pipeline.joblib"),
        "iteration": iteration,
        "branch_id": branch_id
    })

    try:
        # 1. EDA
        branch_state["improvement_hint"] = branch_state.get("eda_improvement_hint", "")
        branch_state = step1_eda_agent(branch_state)

        # 2. Train
        branch_state["improvement_hint"] = branch_state.get("train_improvement_hint", "")
        branch_state = step2_train_agent(branch_state)

        # 3. Eval
        branch_state = step3_local_eval_agent(branch_state)

        return branch_state
    except Exception as e:
        _log(f"Branch {branch_id} failed with error: {e}", level="error")
        return branch_state


def _maybe_build_rag_index(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state)
    if not state.get("rag_enabled", False):
        return state

    kb_dir = Path(state["notebooks_kb_dir"])
    if not kb_dir.exists():
        state["rag_stats"] = {
            "notebooks_root": str(kb_dir),
            "index_dir": state["rag_index_dir"],
            "n_chunks": 0,
        }
        _log("Notebook KB directory does not exist: %s", kb_dir, level="warning")
        return state

    rag_stats = build_notebook_rag_index(
        notebooks_root=kb_dir,
        index_dir=state["rag_index_dir"],
        max_chars=1600,
        embedding_model=state["rag_embedding_model"],
    )
    state["rag_stats"] = rag_stats
    _log("RAG index built: %s", rag_stats)
    return state


def run_pipeline(max_iterations: int = 3, num_candidates: int = 3) -> dict[str, Any]:
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

    _log("Loading data subset (%d%% of train)...", cfg.TRAIN_SAMPLE_PCT)
    state = cfg.load_data_subset(state)

    state = _maybe_build_rag_index(state)

    best_overall_state = None
    best_overall_metric = None

    for iteration in range(1, max_iterations + 1):
        _log("=" * 60)
        _log(f"Refinement Iteration {iteration}/{max_iterations}")
        _log(f"Launching {num_candidates} parallel candidates...")

        iteration_results = []

        # --- ПАРАЛЛЕЛЬНЫЙ БЛОК ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_candidates) as executor:
            # Каждый кандидат начинает с текущего лучшего состояния (state)
            futures = [
                executor.submit(run_candidate_branch, b, iteration, state, session_dir)
                for b in range(1, num_candidates + 1)
            ]

            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                iteration_results.append(res)

        # --- JUDGE BLOCK ---
        _log(f"Iteration {iteration} complete. Filtering candidates by SUFFICIENT verdict...")

        # Сначала просим судью оценить каждого кандидата
        evaluated_results = []
        for branch_res in iteration_results:
            evaluated_results.append(step_judge_result_agent(branch_res))

        # # Фильтруем только те ветки, которые судья признал годными
        # sufficient_candidates = [
        #     res for res in evaluated_results
        #     if res.get("verification_decision") == "SUFFICIENT"
        # ]

        best_iter_state = None
        best_iter_metric = None

        # if not evaluated_results:
        #     _log(
        #         f"Iteration {iteration}: No candidates reached SUFFICIENT status. Refinement will continue using the best available attempt.",
        #         level="warning")
        #     # Для продолжения цикла рефинамента (передачи hints) возьмем лучшего из всех,
        #     # но НЕ будем считать его финальным победителем (best_overall_state)
        #     candidates_to_compare = evaluated_results
        # else:
        #     _log(f"Found {len(sufficient_candidates)} sufficient candidates. Selecting the best one.")
        #     candidates_to_compare = sufficient_candidates

        # Ищем лучшую метрику в выбранном пуле (либо среди SUFFICIENT, либо среди всех как fallback)
        for branch_res in evaluated_results:
            val_metrics = branch_res.get("local_metrics", {}).get("val", branch_res.get("local_metrics", {}))

            if isinstance(val_metrics, dict):
                task = branch_res.get("task_type", "regression")
                current = val_metrics.get("mse") if task == "regression" else (
                            val_metrics.get("f1") or val_metrics.get("accuracy"))

                if current is not None:
                    is_better = (best_iter_metric is None or
                                 (current < best_iter_metric if task == "regression" else current > best_iter_metric))
                    if is_better:
                        best_iter_metric = current
                        best_iter_state = branch_res

        # Обновляем глобальное состояние и решаем, выходить ли из цикла
        if best_iter_state:
            # Если этот кандидат был SUFFICIENT, проверяем, не лучше ли он наших прошлых рекордов
            # is_branch_sufficient = best_iter_state.get("verification_decision") == "SUFFICIENT"

            is_overall_better = (best_overall_metric is None or
                                 (best_iter_metric < best_overall_metric if task == "regression" else best_iter_metric > best_overall_metric))

            if is_overall_better:
                best_overall_metric = best_iter_metric
                best_overall_state = dict(best_iter_state)
                _log(f"Updated BEST OVERALL model: Metric = {best_overall_metric:.4f}")

            # Передаем знания на следующую итерацию
            state.update({
                "eda_improvement_hint": best_iter_state.get("eda_suggestions", ""),
                "train_improvement_hint": best_iter_state.get("train_suggestions", ""),
                "task_type": best_iter_state.get("task_type"),
                "target_column": best_iter_state.get("target_column"),
                "numeric_columns": best_iter_state.get("numeric_columns"),
                "categorical_columns": best_iter_state.get("categorical_columns"),
            })

            # # Если мы нашли хотя бы одного SUFFICIENT кандидата, можем завершить весь Pipeline
            # if is_branch_sufficient:
            #     _log("Stopping refinement: at least one branch is SUFFICIENT.")
            #     break

    # --- FINAL STEPS ---
    if best_overall_state:
        _log("Preparing final submission using best discovered state.")
        state = best_overall_state

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
