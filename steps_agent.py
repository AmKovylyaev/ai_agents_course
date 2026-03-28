"""LLM-driven agent steps (with automatic fallback on failure)."""

from __future__ import annotations

import config as cfg
from executor import create_step_chain, run_step_with_retry
from prompts import (
    STEP1_EDA_PROMPT,
    STEP2_TRAIN_PROMPT,
    STEP3_EVAL_PROMPT,
    STEP4_SUBMISSION_PROMPT,
    STEP_VERIFY_PROMPT,
    STEP7_REPORT_PROMPT,
)
from steps_fallback import (
    step1_eda_fallback,
    step2_train_fallback,
    step3_local_eval_fallback,
    step4_submission_fallback,
    step_verify_fallback,
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

    chain = create_step_chain(STEP1_EDA_PROMPT, llm)
    state, success = run_step_with_retry("step1_eda", chain, state, max_attempts=3, timeout_sec=60)

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

    chain = create_step_chain(STEP2_TRAIN_PROMPT, llm)
    state, success = run_step_with_retry("step2_train", chain, state, max_attempts=3, timeout_sec=120)

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

    chain = create_step_chain(STEP3_EVAL_PROMPT, llm)
    state, success = run_step_with_retry("step3_eval", chain, state, max_attempts=3, timeout_sec=60)

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

    chain = create_step_chain(STEP4_SUBMISSION_PROMPT, llm)
    state, success = run_step_with_retry("step4_submission", chain, state, max_attempts=3, timeout_sec=60)

    if not success:
        if cfg.logger:
            cfg.logger.warning("Submission agent failed, using fallback")
        return step4_submission_fallback(state)
    return state


def step_verify_result_agent(state: dict) -> dict:
    import re
    import json
    state = dict(state)
    llm = cfg.get_llm()

    if not llm:
        if cfg.logger:
            cfg.logger.warning("No LLM available for Judge, using fallback")
        return step_verify_fallback(state)

    # codes = []
    # for step in ["step1_eda", "step2_train", "step3_eval"]:
    #     code_key = f"{step}_code"
    #     if code_key in state and state[code_key]:
    #         codes.append(state[code_key])
    # previous_code = "\n\n".join(codes) if codes else ""
    # if previous_code:
    #     previous_code = previous_code[:500] + "..." if len(previous_code) > 500 else previous_code

    eda_summary = state.get("eda_report", "No EDA summary available.")
    if isinstance(eda_summary, str):
        eda_summary = eda_summary[:1500]
    else:
        eda_summary = "No EDA summary."

    local_metrics = state.get("local_metrics", {})
    if local_metrics:
        metrics_str = json.dumps(local_metrics, separators=(',', ':'))[:500]
    else:
        metrics_str = "{}"

    if cfg.logger:
        cfg.logger.info(f"Judge input sizes: eda={len(eda_summary)}, metrics={len(metrics_str)}")

    prompt_data = {
        "eda_summary": eda_summary,
        "local_metrics": metrics_str,
        # "previous_code": previous_code,
    }
    if cfg.logger:
        cfg.logger.info(f"eda summary: {eda_summary}")
        cfg.logger.info(f"local metrics: {local_metrics}")
    chain = create_step_chain(STEP_VERIFY_PROMPT, llm)
    max_attempts: int = 3

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        if cfg.logger:
            cfg.logger.info(f"Judge attempt {attempt}/{max_attempts}")
        try:
            response = chain.invoke(prompt_data)
            if len(response) == 0:
                attempt -= 1
            if cfg.logger:
                cfg.logger.info(f"Judge raw response:\n{response}")

            # Пытаемся извлечь JSON
            def extract_json(text: str) -> dict:
                text = text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(0))
                        except:
                            pass
                    raise ValueError("No valid JSON found in LLM response")

            result = extract_json(response)
            state["verification_decision"] = result.get("decision", "NEED_REFINEMENT")
            state["verification_reasoning"] = result.get("reasoning", "")
            state["verification_suggestions"] = result.get("suggestions", "")
            state["verification_raw"] = response
            if cfg.logger:
                cfg.logger.info("Judge decision: %s", state["verification_decision"])
                if state["verification_decision"] == "NEED_REFINEMENT":
                    cfg.logger.info("Suggestions: %s", state["verification_suggestions"])
            return state

        except Exception as e:
            if cfg.logger:
                cfg.logger.error(f"Judge attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                continue

    # Если все попытки провалились
    if cfg.logger:
        cfg.logger.error(f"Judge failed after {max_attempts} attempts, using fallback")
    return step_verify_fallback(state)


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
