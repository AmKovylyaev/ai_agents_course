"""LLM-driven agent steps (with automatic fallback on failure)."""

from __future__ import annotations

import config as cfg
from rag.rag_tools import inject_rag_context_into_state
from executor import create_step_chain, run_step_with_retry
from rag.rag_tools import inject_rag_context_into_state
from mini_feedback_loop import mini_feedback_loop
from tools.web_context import inject_web_context_into_state
from prompts import (
    STEP1_PLANNER_PROMPT,
    STEP1_EDA_PROMPT,
    STEP1_VERIFIER_PROMPT,
    STEP2_PLANNER_PROMPT,
    STEP2_TRAIN_PROMPT,
    STEP2_VERIFIER_PROMPT,
    STEP3_PLANNER_PROMPT,
    STEP3_EVAL_PROMPT,
    STEP3_VERIFIER_PROMPT,
    STEP4_PLANNER_PROMPT,
    STEP4_SUBMISSION_PROMPT,
    STEP4_VERIFIER_PROMPT,
    STEP7_REPORT_PROMPT,
)
from steps_fallback import (
    step1_eda_fallback,
    step2_train_fallback,
    step3_local_eval_fallback,
    step4_submission_fallback,
    step7_report_fallback,
)


def step1_eda_agent(state: dict) -> dict:
    """Step 1: LLM-generated EDA."""
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        if cfg.logger:
            cfg.logger.warning("No LLM available, using fallback EDA")
        return step1_eda_fallback(state)

    state, success = mini_feedback_loop(
        "step1_eda",
        STEP1_PLANNER_PROMPT,
        STEP1_EDA_PROMPT,
        STEP1_VERIFIER_PROMPT,
        state,
        llm,
        max_attempts=3,
        timeout_sec=120,
    )

    if not success:
        if cfg.logger:
            cfg.logger.warning("EDA agent failed, using fallback")
        return step1_eda_fallback(state)
    return state


def step2_train_agent(state: dict) -> dict:
    """Step 2: LLM-generated model training."""
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        if cfg.logger:
            cfg.logger.warning("No LLM available, using fallback train")
        return step2_train_fallback(state)

    if state.get("rag_enabled", False):
        task_type = state.get("task_type", "")
        query = (
            f"tabular machine learning notebook for {task_type}; "
            f"preprocessing categorical and numeric features; "
            f"columntransformer pipeline; training and evaluation; "
            f"submission pattern"
        )
        state = inject_rag_context_into_state(
            state=state,
            query=query,
            top_k=state.get("rag_top_k", 5),
            search_type=state.get("rag_search_type", "hybrid"),
            section_filter=None,
        )
        if cfg.logger:
            cfg.logger.info("RAG query for training step: %s", query)

    state, success = mini_feedback_loop(
        "step2_train",
        STEP2_PLANNER_PROMPT,
        STEP2_TRAIN_PROMPT,
        STEP2_VERIFIER_PROMPT,
        state,
        llm,
        max_attempts=3,
        timeout_sec=360,
    )

    if state.get("web_search_enabled", False):
        web_query = (
            f"best practices for tabular {state.get('task_type', 'regression')} "
            f"with categorical features catboost cross validation"
        )
        state = inject_web_context_into_state(
            state=state,
            query=web_query,
            max_results=state.get("web_search_max_results", 3),
        )
        if cfg.logger:
            cfg.logger.info("Web query for training step: %s", web_query)

    if not success:
        if cfg.logger:
            cfg.logger.warning("Train agent failed, using fallback")
        return step2_train_fallback(state)
    return state


def step3_local_eval_agent(state: dict) -> dict:
    """Step 3: LLM-generated local model evaluation."""
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        if cfg.logger:
            cfg.logger.warning("No LLM available, using fallback eval")
        return step3_local_eval_fallback(state)

    if state.get("web_search_enabled", False):
        web_query = (
            f"how to evaluate tabular {state.get('task_type', 'regression')} "
            f"catboost cross validation metric selection"
        )
        state = inject_web_context_into_state(
            state=state,
            query=web_query,
            max_results=state.get("web_search_max_results", 3),
        )

    state, success = mini_feedback_loop(
        "step3_eval",
        STEP3_PLANNER_PROMPT,
        STEP3_EVAL_PROMPT,
        STEP3_VERIFIER_PROMPT,
        state,
        llm,
        max_attempts=3,
        timeout_sec=120,
    )

    if not success:
        if cfg.logger:
            cfg.logger.warning("Eval agent failed, using fallback")
        return step3_local_eval_fallback(state)
    return state


def step4_submission_agent(state: dict) -> dict:
    """Step 4: LLM-generated submission creation."""
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        if cfg.logger:
            cfg.logger.warning("No LLM available, using fallback submission")
        return step4_submission_fallback(state)

    if state.get("web_search_enabled", False):
        web_query = "kaggle submission csv format regression predict best practices"
        state = inject_web_context_into_state(
            state=state,
            query=web_query,
            max_results=state.get("web_search_max_results", 3),
        )

    state, success = mini_feedback_loop(
        "step4_submission",
        STEP4_PLANNER_PROMPT,
        STEP4_SUBMISSION_PROMPT,
        STEP4_VERIFIER_PROMPT,
        state,
        llm,
        max_attempts=3,
        timeout_sec=120,
    )

    if not success:
        if cfg.logger:
            cfg.logger.warning("Submission agent failed, using fallback")
        return step4_submission_fallback(state)
    return state


def step7_report_agent(state: dict) -> dict:
    """Step 7: LLM-generated final report."""
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        if cfg.logger:
            cfg.logger.warning("No LLM available, using fallback report")
        return step7_report_fallback(state)

    chain = create_step_chain(STEP7_REPORT_PROMPT, llm)
    state, success = run_step_with_retry("step7_report", chain, state, max_attempts=3, timeout_sec=60)

    if not success:
        if cfg.logger:
            cfg.logger.warning("Report agent failed, using fallback")
        return step7_report_fallback(state)
    return state
