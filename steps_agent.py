"""LLM-driven agent steps (with automatic fallback on failure)."""

from __future__ import annotations

import config as cfg
from executor import create_step_chain, run_step_with_retry
from mini_feedback_loop import mini_feedback_loop
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
