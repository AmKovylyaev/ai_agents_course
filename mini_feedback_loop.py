"""Three-agent feedback loop using LangChain/LangGraph agents with tools.

Architecture:
  - Planner: simple LLM chain (no tools) — produces a step-by-step plan
  - Coder Agent: LangGraph ReAct agent with tools:
      * validate_syntax — check code for syntax errors
      * check_safety — reject dangerous operations
      * execute_code — run code in an isolated subprocess
    The agent can self-correct: validate → fix → execute → fix → re-execute
  - Verifier Agent: LangGraph ReAct agent with guardrail tools:
      * check_state_completeness — required state keys present
      * check_data_leakage — target not in feature columns
      * check_model_file — pipeline.joblib loadable
      * check_metrics — values sane
      * check_submission — CSV format matches sample
      * read_session_file — inspect generated artifacts
    The verifier diagnoses failures using concrete tool results
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import config as cfg
from config import log as _log, build_prompt_state as _build_prompt_state
from executor import (
    create_step_chain,
    validate_code,
    execute_code,
)


# ---------------------------------------------------------------------------
# Artifact directory helpers
# ---------------------------------------------------------------------------

def _attempt_dir(state: dict, step_name: str, attempt: int) -> Path:
    """Return the directory for a specific step/attempt within the current iteration.

    Layout: session_dir/iter_N/step_name/attempt_M/
    """
    session_dir = Path(state["session_dir"])
    iteration = state.get("iteration", 1)
    d = session_dir / f"iter_{iteration}" / step_name / f"attempt_{attempt}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_text(directory: Path, filename: str, text: str) -> Path:
    """Write text to a file and return the path."""
    p = directory / filename
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tool factories (closures capture mutable *state* and shared *exec_result*)
# ---------------------------------------------------------------------------

def _make_coder_tools(
    state: dict, step_name: str, attempt: int, timeout_sec: int,
) -> tuple[list, dict[str, Any]]:
    """Build tools for the coder agent. Returns (tools, exec_result_container)."""
    from langchain_core.tools import tool as lc_tool

    exec_result: dict[str, Any] = {
        "success": False, "output": "", "updated_state": None, "code": "",
    }
    run_counter: dict[str, int] = {"n": 0}

    @lc_tool
    def validate_syntax(code: str) -> str:
        """Check Python code for syntax errors.
        Returns 'VALID' or 'INVALID: <details>'."""
        ok, msg = validate_code(code)
        return "VALID" if ok else f"INVALID: {msg}"

    @lc_tool
    def check_safety(code: str) -> str:
        """Scan code for dangerous operations (os.system, subprocess, eval, exec,
        file deletion, network calls). Returns 'PASS' or 'FAIL: <details>'."""
        from guardrails import check_code_safety
        r = check_code_safety(code, quiet=True)
        return f"{'PASS' if r else 'FAIL'}: {r.message}"

    @lc_tool
    def run_code(code: str) -> str:
        """Validate, safety-check, then execute Python code in an isolated subprocess.
        The subprocess has a `state` dict with all pipeline paths.
        Returns 'SUCCESS: <stdout>' or 'ERROR: <details>'.
        If it fails, fix the code and call this tool again (up to 2 calls total)."""
        run_counter["n"] += 1
        n = run_counter["n"]

        ok, msg = validate_code(code)
        if not ok:
            return f"VALIDATION_ERROR: {msg}"

        from guardrails import check_code_safety
        safety = check_code_safety(code, quiet=True)
        if not safety:
            return f"SAFETY_ERROR: {safety.message}"

        att_dir = _attempt_dir(state, step_name, attempt)
        code_file = att_dir / f"code_v{n}.py"
        code_file.write_text(code, encoding="utf-8")

        success, output, updated = execute_code(code, state, timeout_sec)
        exec_result["code"] = code
        if success:
            exec_result.update(success=True, output=output, updated_state=updated)
            final_file = att_dir / "code.py"
            final_file.write_text(code, encoding="utf-8")
            _log("  run_code [%d]: SUCCESS", n)
            return f"SUCCESS:\n{output[:2000]}"
        exec_result["output"] = output
        short_err = output.strip().splitlines()[-1][:200] if output.strip() else "unknown error"
        _log("  run_code [%d]: FAILED — %s", n, short_err)
        return f"EXECUTION_ERROR:\n{output[:3000]}"

    return [validate_syntax, check_safety, run_code], exec_result


def _make_verifier_tools(state: dict, step_name: str) -> list:
    """Build guardrail tools for the verifier agent, scoped to the current step."""
    from langchain_core.tools import tool as lc_tool
    from guardrails import get_guardrail_names_for_step

    relevant = set(get_guardrail_names_for_step(step_name))
    tools: list = []

    if "required_state" in relevant:
        @lc_tool
        def check_state_completeness() -> str:
            """Check whether all required state keys are populated for the current step."""
            from guardrails import check_required_state
            r = check_required_state(step_name, state)
            return f"{'PASS' if r else 'FAIL'}: {r.message}"
        tools.append(check_state_completeness)

    if "no_data_leakage" in relevant:
        @lc_tool
        def check_data_leakage() -> str:
            """Check whether the target column leaked into feature column lists."""
            from guardrails import check_no_data_leakage
            r = check_no_data_leakage(state)
            return f"{'PASS' if r else 'FAIL'}: {r.message}"
        tools.append(check_data_leakage)

    if "model_exists" in relevant:
        @lc_tool
        def check_model_file() -> str:
            """Check whether the model pipeline file exists and is loadable."""
            from guardrails import check_model_exists
            r = check_model_exists(state)
            return f"{'PASS' if r else 'FAIL'}: {r.message}"
        tools.append(check_model_file)

    if "metrics_sanity" in relevant:
        @lc_tool
        def check_metrics() -> str:
            """Check whether evaluation metrics are valid (in [0,1], above random baseline)."""
            from guardrails import check_metrics_sanity
            r = check_metrics_sanity(state)
            return f"{'PASS' if r else 'FAIL'}: {r.message}"
        tools.append(check_metrics)

    if "submission_format" in relevant:
        @lc_tool
        def check_submission() -> str:
            """Check whether submission.csv matches the sample submission format."""
            from guardrails import check_submission_format
            r = check_submission_format(state)
            return f"{'PASS' if r else 'FAIL'}: {r.message}"
        tools.append(check_submission)

    @lc_tool
    def read_session_file(file_path: str) -> str:
        """Read a file from the session or data directory (for inspecting code, reports, etc.)."""
        try:
            p = Path(file_path)
            allowed_roots = [Path(state["session_dir"]), Path(state.get("data_dir", ""))]
            if not any(str(p).startswith(str(r)) for r in allowed_roots if str(r)):
                return "ACCESS_DENIED: path outside session/data directories"
            if not p.exists():
                return f"NOT_FOUND: {file_path}"
            text = p.read_text(encoding="utf-8")
            return text[:5000] + ("\n…(truncated)" if len(text) > 5000 else "")
        except Exception as e:
            return f"READ_ERROR: {e}"
    tools.append(read_session_file)

    return tools


# ---------------------------------------------------------------------------
# Agent runner (works with langgraph *or* falls back to simple chains)
# ---------------------------------------------------------------------------

_CODER_SYSTEM = (
    "You are an ML engineer with access to tools.\n"
    "Your ONLY job is to write Python code and execute it via the run_code tool.\n"
    "NEVER respond with just code in a text message — that does nothing.\n\n"
    "Mandatory workflow:\n"
    "1. Write the complete Python code as a single string\n"
    "2. Call the run_code tool with that full code string\n"
    "3. If run_code returns an error, fix the code and call run_code again\n"
    "4. You have at most 2 run_code calls. Make them count.\n\n"
    "CRITICAL RULES:\n"
    "- Follow the plan EXACTLY — use the model, encoder, and approach specified in the plan.\n"
    "- Do NOT switch to a different model or library than what the plan specifies.\n"
    "- If a library import fails, fix the import — do NOT replace the model/library.\n"
    "- You MUST call run_code at least once. If you finish without calling run_code, the attempt fails."
)

_VERIFIER_SYSTEM = (
    "You are a code verification expert with access to guardrail tools.\n"
    "You run after EVERY attempt — whether the code succeeded or failed.\n\n"
    "Mandatory workflow:\n"
    "1. Call ALL available guardrail tools to check requirements\n"
    "2. Analyze the results\n"
    "3. Provide your verdict:\n"
    '   - If ALL checks pass: state "APPROVED"\n'
    '   - If ANY check fails: list each failure with a specific fix, state "FAIL"\n\n'
    "You MUST call the guardrail tools — do not guess at results."
)


_MAX_AGENT_STEPS = 8


def _invoke_agent(llm, tools: list, system_prompt: str, user_message: str) -> str:
    """Run a LangGraph ReAct agent and return its final text response."""
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(llm, tools, prompt=system_prompt)
    try:
        result = agent.invoke(
            {"messages": [("user", user_message)]},
            config={"callbacks": [], "recursion_limit": _MAX_AGENT_STEPS},
        )
    except Exception as e:
        _log("  Agent hit recursion limit or error: %s", e, level="warning")
        return ""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        text = getattr(msg, "content", "")
        if text and not getattr(msg, "tool_calls", None):
            return text
    return ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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
    """Three-agent feedback loop: planner -> coder agent -> verifier agent.

    Artifacts for every attempt are saved under:
        session_dir/iter_N/<step_name>/attempt_M/
            planner_prompt.txt   — filled planner prompt
            plan.txt             — planner output
            coder_prompt.txt     — filled coder prompt
            code.py              — generated code
            verifier_prompt.txt  — filled verifier prompt
            feedback.txt         — verifier output

    Returns ``(updated_state, success)``.
    """
    planner_chain = create_step_chain(planner_prompt, llm)
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        _log("─" * 50)
        _log("%s  attempt %d/%d", step_name, attempt, max_attempts)

        try:
            prompt_state = _build_prompt_state(state)
            att_dir = _attempt_dir(state, step_name, attempt)

            # ── Planner (simple chain) ────────────────────────────────
            planner_input = planner_prompt.format(**prompt_state)
            _save_text(att_dir, "planner_prompt.txt", planner_input)

            plan = planner_chain.invoke(prompt_state)
            _save_text(att_dir, "plan.txt", plan)
            _log("  Planner → plan (%d chars)", len(plan))

            state["plan"] = plan
            prompt_state["plan"] = plan

            # ── Coder Agent ───────────────────────────────────────────
            coder_tools, exec_result = _make_coder_tools(
                state, step_name, attempt, timeout_sec,
            )
            coder_input = coder_prompt.format(**prompt_state)
            _save_text(att_dir, "coder_prompt.txt", coder_input)

            _invoke_agent(llm, coder_tools, _CODER_SYSTEM, coder_input)

            if not exec_result["success"]:
                error_output = exec_result.get("output") or "Coder agent did not execute code"
                error_msg = f"Execution failed: {error_output}"
                _log("  Coder FAILED: %s", error_msg[:300], level="warning")
                errors.append(error_msg)
                state["last_error"] = error_msg
                state["previous_code"] = exec_result.get("code", "")
            else:
                _log("  Coder OK")
                state.update(exec_result["updated_state"])
                state["previous_code"] = exec_result.get("code", "")
                state.pop("last_error", None)

                from guardrails import run_step_guardrails
                guardrail_results = run_step_guardrails(step_name, state)
                guardrail_failures = [r for r in guardrail_results if not r]
                if guardrail_failures:
                    guardrail_msg = "; ".join(f.message for f in guardrail_failures)
                    state["last_error"] = f"Guardrail check failed: {guardrail_msg}"
                    _log("  Guardrails FAILED: %s", guardrail_msg, level="warning")

            # ── Verifier Agent (always runs) ──────────────────────────
            verifier_tools = _make_verifier_tools(state, step_name)
            verifier_input = verifier_prompt.format(**_build_prompt_state(state))
            _save_text(att_dir, "verifier_prompt.txt", verifier_input)

            feedback = _invoke_agent(
                llm, verifier_tools, _VERIFIER_SYSTEM, verifier_input,
            )
            _save_text(att_dir, "feedback.txt", feedback)
            state["verifier_feedback"] = feedback

            # ── Accept or retry ───────────────────────────────────────
            verifier_approved = "FAIL" not in feedback.upper()
            is_approved = (
                exec_result["success"]
                and not state.get("last_error")
                and verifier_approved
            )
            verdict = "APPROVED" if is_approved else "REJECTED"
            _log(
                "  Verdict: %s  (exec=%s, guardrails=%s, verifier=%s)",
                verdict,
                "ok" if exec_result["success"] else "fail",
                "ok" if not state.get("last_error") else "fail",
                "ok" if verifier_approved else "fail",
            )

            if is_approved:
                state[f"{step_name}_attempts"] = attempt
                state[f"{step_name}_success"] = True
                state[f"{step_name}_code"] = exec_result["code"]
                state[f"{step_name}_output"] = exec_result["output"]
                for key in ("last_error", "previous_code", "plan", "verifier_feedback"):
                    state.pop(key, None)
                return state, True

            if not state.get("last_error"):
                state["last_error"] = f"Verifier rejected: {feedback[:500]}"
            errors.append(state["last_error"])

        except Exception as e:
            error_msg = f"Unexpected error: {e!s}"
            _log("  %s: %s", step_name, error_msg, level="error")
            errors.append(error_msg)
            state["last_error"] = error_msg

    _log("%s: all %d attempts failed", step_name, max_attempts, level="error")
    state[f"{step_name}_attempts"] = max_attempts
    state[f"{step_name}_success"] = False
    state[f"{step_name}_errors"] = errors
    return state, False
