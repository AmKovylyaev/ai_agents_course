"""LLM-driven agent steps (with automatic fallback on failure)."""

from __future__ import annotations

import json
import re
from pathlib import Path

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
    STEP_JUDGE_PROMPT,
)
from steps_fallback import (
    step1_eda_fallback,
    step2_train_fallback,
    step3_local_eval_fallback,
    step4_submission_fallback,
    step7_report_fallback,
    step_judge_fallback,
)


def _log(msg: str, *args, level: str = "info") -> None:
    if cfg.logger:
        getattr(cfg.logger, level)(msg, *args)


# ---------------------------------------------------------------------------
# Steps 1–4: three-agent feedback loop
# ---------------------------------------------------------------------------

def _run_feedback_step(
    step_name: str,
    planner_prompt: str,
    coder_prompt: str,
    verifier_prompt: str,
    fallback_fn,
    state: dict,
    max_attempts: int = 3,
    timeout_sec: int = 120,
) -> dict:
    """Generic wrapper: try LLM agents, fall back on failure."""
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        _log("%s — no LLM, using fallback", step_name, level="warning")
        return fallback_fn(state)

    state, success = mini_feedback_loop(
        step_name,
        planner_prompt,
        coder_prompt,
        verifier_prompt,
        state,
        llm,
        max_attempts=max_attempts,
        timeout_sec=timeout_sec,
    )

    if not success:
        _log("%s — agents failed, using fallback", step_name, level="warning")
        return fallback_fn(state)
    return state


def step1_eda_agent(state: dict) -> dict:
    return _run_feedback_step(
        "step1_eda",
        STEP1_PLANNER_PROMPT, STEP1_EDA_PROMPT, STEP1_VERIFIER_PROMPT,
        step1_eda_fallback, state,
        timeout_sec=120,
    )


def step2_train_agent(state: dict) -> dict:
    return _run_feedback_step(
        "step2_train",
        STEP2_PLANNER_PROMPT, STEP2_TRAIN_PROMPT, STEP2_VERIFIER_PROMPT,
        step2_train_fallback, state,
        timeout_sec=360,
    )


def step3_local_eval_agent(state: dict) -> dict:
    return _run_feedback_step(
        "step3_eval",
        STEP3_PLANNER_PROMPT, STEP3_EVAL_PROMPT, STEP3_VERIFIER_PROMPT,
        step3_local_eval_fallback, state,
        timeout_sec=120,
    )


def step4_submission_agent(state: dict) -> dict:
    return _run_feedback_step(
        "step4_submission",
        STEP4_PLANNER_PROMPT, STEP4_SUBMISSION_PROMPT, STEP4_VERIFIER_PROMPT,
        step4_submission_fallback, state,
        timeout_sec=120,
    )


# ---------------------------------------------------------------------------
# Step 7: report (single chain, no feedback loop)
# ---------------------------------------------------------------------------

def step7_report_agent(state: dict) -> dict:
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        _log("step7_report — no LLM, using fallback", level="warning")
        return step7_report_fallback(state)

    chain = create_step_chain(STEP7_REPORT_PROMPT, llm)
    state, success = run_step_with_retry(
        "step7_report", chain, state, max_attempts=3, timeout_sec=60,
    )

    if not success:
        _log("step7_report — agent failed, using fallback", level="warning")
        return step7_report_fallback(state)
    return state


# ---------------------------------------------------------------------------
# Judge (LLM-as-a-judge, outer refinement loop)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM output."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {
            "decision": "NEED_REFINEMENT",
            "reasoning": "Failed to parse JSON, using fallback",
            "eda_suggestions": "",
            "train_suggestions": "",
        }


def step_judge_result_agent(state: dict) -> dict:
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        _log("Judge — no LLM, using fallback", level="warning")
        return step_judge_fallback(state)

    model_info_parts = []
    model_type = state.get("model_type", "")
    if model_type:
        model_info_parts.append(f"Model: {model_type}")
    task_type = state.get("task_type", "")
    if task_type:
        model_info_parts.append(f"Task: {task_type}")
    target_transform = state.get("target_transform", "")
    if target_transform and target_transform != "none":
        model_info_parts.append(f"Target transform: {target_transform}")
    model_info = "; ".join(model_info_parts) if model_info_parts else "unknown"

    local_metrics = state.get("local_metrics", {})
    metrics_str = json.dumps(local_metrics, separators=(",", ":"))[:500] if local_metrics else "{}"

    prompt_data = {
        "local_metrics": metrics_str,
        "previous_code": model_info,
    }

    chain = create_step_chain(STEP_JUDGE_PROMPT, llm)
    judge_prompt_filled = STEP_JUDGE_PROMPT.format(**prompt_data)

    iter_dir = Path(state["session_dir"]) / f"iter_{state.get('iteration', 1)}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "judge_prompt.txt").write_text(judge_prompt_filled, encoding="utf-8")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        _log("  Judge attempt %d/%d", attempt, max_retries)
        try:
            response = chain.invoke(prompt_data)
            if not response:
                continue

            result = _extract_json(response)
            state["verification_decision"] = result.get("decision", "NEED_REFINEMENT")
            state["verification_reasoning"] = result.get("reasoning", "")
            state["eda_improvement_hint"] = result.get("eda_suggestions", "")
            state["train_improvement_hint"] = result.get("train_suggestions", "")
            state["verification_raw"] = response

            (iter_dir / "judge_response.txt").write_text(response, encoding="utf-8")
            (iter_dir / "judge_decision.txt").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
            )

            _log("  Judge decision: %s", state["verification_decision"])
            if state["verification_decision"] == "NEED_REFINEMENT":
                _log("    EDA suggestions: %s", state["eda_improvement_hint"])
                _log("    Train suggestions: %s", state["train_improvement_hint"])
            return state

        except Exception as e:
            _log("  Judge attempt %d failed: %s", attempt, e, level="error")

    _log("Judge failed after %d attempts, using fallback", max_retries, level="error")
    return step_judge_fallback(state)
