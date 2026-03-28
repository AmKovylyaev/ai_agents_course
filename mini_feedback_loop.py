"""Three-agent mini feedback loop: planner → coder → verifier."""

from __future__ import annotations

from pathlib import Path

import config as cfg
from executor import create_step_chain, extract_code_block, validate_code, execute_code


def _build_prompt_state(state: dict) -> dict:
    """Convert state to a dict of string values suitable for prompt formatting."""
    prompt_state: dict = {}
    for k, v in state.items():
        if isinstance(v, Path):
            prompt_state[k] = str(v)
        elif isinstance(v, dict):
            prompt_state[k] = str(v)
        else:
            prompt_state[k] = v if v is not None else ""

    prompt_state.setdefault("last_error", "")
    prompt_state.setdefault("previous_code", "")
    prompt_state.setdefault("plan", "")
    prompt_state.setdefault("verifier_feedback", "")
    prompt_state.setdefault("model_path", state.get("model_path", ""))
    prompt_state.setdefault("target_column", state.get("target_column", ""))
    prompt_state.setdefault("numeric_columns", state.get("numeric_columns", "[]"))
    prompt_state.setdefault("categorical_columns", state.get("categorical_columns", "[]"))
    prompt_state.setdefault("n_classes", state.get("n_classes", ""))
    prompt_state.setdefault("train_shape", state.get("train_shape", ""))
    prompt_state.setdefault("submit_ok", state.get("submit_ok", False))
    prompt_state.setdefault("public_score", state.get("public_score", "N/A"))
    prompt_state.setdefault("private_score", state.get("private_score", "N/A"))
    prompt_state.setdefault("train_path", state.get("train_path", ""))
    prompt_state.setdefault("test_path", state.get("test_path", ""))
    prompt_state.setdefault(
        "sample_submission_path", state.get("sample_submission_path", "")
    )
    prompt_state.setdefault("session_dir", str(state.get("session_dir", "")))
    prompt_state.setdefault("train_sample_frac", cfg.TRAIN_SAMPLE_FRAC)
    prompt_state.setdefault("train_sample_pct", cfg.TRAIN_SAMPLE_PCT)
    return prompt_state


def mini_feedback_loop(
    step_name: str,
    planner_prompt: str,
    coder_prompt: str,
    verifier_prompt: str,
    state: dict,
    llm,
    max_attempts: int = 3,
    timeout_sec: int = 120,
) -> tuple[dict, bool]:
    """
    Three-agent feedback loop: planner → coder → verifier.

    Each attempt:
      1. Planner produces a step-by-step plan (incorporating prior feedback)
      2. Coder generates executable Python code following the plan
      3. Code is extracted, validated, and executed
      4. On failure, verifier analyzes the error and provides actionable feedback
      5. Feedback carries into the next attempt

    Returns (updated_state, success).
    """
    logger = cfg.logger

    planner_chain = create_step_chain(planner_prompt, llm)
    coder_chain = create_step_chain(coder_prompt, llm)
    verifier_chain = create_step_chain(verifier_prompt, llm)

    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        if logger:
            logger.info("%s: Attempt %d/%d", step_name, attempt, max_attempts)

        try:
            session_dir = Path(state["session_dir"])
            plans_dir = session_dir / "plans"
            plans_dir.mkdir(exist_ok=True)
            code_dir = session_dir / "code"
            feedback_dir = session_dir / "feedback"
            feedback_dir.mkdir(exist_ok=True)
            prompt_state = _build_prompt_state(state)

            # --- PLANNER ---
            if logger:
                logger.info("%s: [Planner] Generating plan...", step_name)
            plan = planner_chain.invoke(prompt_state)
            if logger:
                logger.info(
                    "%s: [Planner] Plan ready (%d chars)", step_name, len(plan)
                )
            state["plan"] = plan
            prompt_state["plan"] = plan

            plan_file = plans_dir / f"{step_name}_attempt{attempt}.txt"
            with open(plan_file, "w", encoding="utf-8") as f:
                f.write(plan)
            if logger:
                logger.info("%s: Plan saved to %s", step_name, plan_file)

            # --- CODER ---
            if logger:
                logger.info("%s: [Coder] Generating code...", step_name)
            code_response = coder_chain.invoke(prompt_state)

            code = extract_code_block(code_response)
            if not code:
                error_msg = "No code block found in Coder response"
                if logger:
                    logger.warning("%s: %s", step_name, error_msg)
                errors.append(error_msg)
                state["last_error"] = error_msg
                continue

            is_valid, validation_msg = validate_code(code)
            if not is_valid:
                if logger:
                    logger.warning(
                        "%s: Validation failed: %s", step_name, validation_msg
                    )
                errors.append(validation_msg)
                state["last_error"] = validation_msg
                state["previous_code"] = code
                continue

            code_file = code_dir / f"{step_name}.py"
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            if logger:
                logger.info("%s: Code saved to %s", step_name, code_file)

            # --- EXECUTE ---
            if logger:
                logger.info("%s: Executing code...", step_name)
            success, output, updated_state = execute_code(code, state, timeout_sec)

            if success:
                if logger:
                    logger.info("%s: Execution successful", step_name)
                updated_state[f"{step_name}_attempts"] = attempt
                updated_state[f"{step_name}_success"] = True
                updated_state[f"{step_name}_code"] = code
                updated_state[f"{step_name}_output"] = output
                for key in ("last_error", "previous_code", "plan", "verifier_feedback"):
                    updated_state.pop(key, None)
                return updated_state, True

            # --- VERIFIER (on failure) ---
            error_msg = f"Execution failed: {output}"
            if logger:
                logger.warning("%s: %s", step_name, error_msg)
            errors.append(error_msg)
            state["last_error"] = error_msg
            state["previous_code"] = code

            if attempt < max_attempts:
                if logger:
                    logger.info("%s: [Verifier] Analyzing failure...", step_name)
                verify_state = _build_prompt_state(state)
                feedback = verifier_chain.invoke(verify_state)
                if logger:
                    logger.info(
                        "%s: [Verifier] Feedback ready (%d chars)",
                        step_name,
                        len(feedback),
                    )
                state["verifier_feedback"] = feedback

                feedback_file = feedback_dir / f"{step_name}_attempt{attempt}.txt"
                with open(feedback_file, "w", encoding="utf-8") as f:
                    f.write(feedback)
                if logger:
                    logger.info(
                        "%s: Verifier feedback saved to %s", step_name, feedback_file
                    )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if logger:
                logger.error("%s: %s", step_name, error_msg)
            errors.append(error_msg)
            state["last_error"] = error_msg

    if logger:
        logger.error("%s: All %d attempts failed", step_name, max_attempts)
    state[f"{step_name}_attempts"] = max_attempts
    state[f"{step_name}_success"] = False
    state[f"{step_name}_errors"] = errors
    return state, False
