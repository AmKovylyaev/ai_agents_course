"""Code execution engine: extraction, validation, subprocess runner, and retry loop."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import config as cfg


def extract_code_block(text: str) -> str | None:
    """Extract Python code from a markdown fenced code block."""
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def validate_code(code: str) -> tuple[bool, str]:
    """Check code for syntax errors via ast.parse."""
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def execute_code(code: str, state: dict, timeout_sec: int = 60) -> tuple[bool, str, dict]:
    """
    Run *code* in a subprocess, passing *state* via JSON files.

    Returns (success, output_or_error, updated_state).
    """
    session_dir = Path(state["session_dir"])

    state_file = session_dir / "state_input.json"
    state_to_save = {}
    for k, v in state.items():
        if isinstance(v, Path):
            state_to_save[k] = str(v)
        elif hasattr(v, "to_json"):
            continue
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            state_to_save[k] = v
        else:
            try:
                json.dumps(v)
                state_to_save[k] = v
            except Exception:
                pass

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_to_save, f, ensure_ascii=False, indent=2, default=str)

    wrapped_code = f'''
import json
import sys
from pathlib import Path

with open("{state_file}", "r", encoding="utf-8") as f:
    state = json.load(f)

for key in ["session_dir", "code_dir", "models_dir", "reports_dir", "data_dir",
            "train_path", "test_path", "sample_submission_path", "model_path", "submission_path"]:
    if key in state and isinstance(state[key], str):
        state[key] = Path(state[key])

{code}

state_output_file = Path("{session_dir}") / "state_output.json"
state_to_save = {{}}
for k, v in state.items():
    if isinstance(v, Path):
        state_to_save[k] = str(v)
    elif hasattr(v, 'to_json'):
        continue
    elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
        state_to_save[k] = v
    else:
        try:
            json.dumps(v)
            state_to_save[k] = v
        except:
            pass

with open(state_output_file, "w", encoding="utf-8") as f:
    json.dump(state_to_save, f, ensure_ascii=False, indent=2, default=str)

print("STATE_SAVED_SUCCESSFULLY")
'''

    fd, path = tempfile.mkstemp(suffix=".py", prefix="agent_step_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(wrapped_code)

        result = subprocess.run(
            ["python3", path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(session_dir),
        )

        if result.returncode == 0:
            output_file = session_dir / "state_output.json"
            if output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    updated_state = json.load(f)
                merged_state = dict(state)
                merged_state.update(updated_state)
                return True, result.stdout, merged_state
            return True, result.stdout, state
        else:
            error_msg = result.stderr or result.stdout or "Non-zero exit code"
            return False, error_msg, state

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Code execution exceeded time limit", state
    except Exception as e:
        return False, f"Execution error: {str(e)}", state
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def create_step_chain(prompt_template: str, llm):
    """Build an LCEL chain: prompt -> llm -> string output."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])
    return prompt | llm | StrOutputParser()


def run_step_with_retry(
    step_name: str,
    chain,
    state: dict,
    max_attempts: int = 3,
    timeout_sec: int = 120,
) -> tuple[dict, bool]:
    """
    Generate code via *chain*, execute it, and retry on failure.

    Returns (updated_state, success).
    """
    logger = cfg.logger
    attempts = 0
    errors: list[str] = []

    while attempts < max_attempts:
        attempts += 1
        if logger:
            logger.info("%s: Attempt %d/%d", step_name, attempts, max_attempts)

        try:
            if logger:
                logger.info("%s: Generating code with LLM...", step_name)

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
            prompt_state.setdefault("model_path", state.get("model_path", ""))
            prompt_state.setdefault("target_column", state.get("target_column", ""))
            prompt_state.setdefault("submit_ok", state.get("submit_ok", False))
            prompt_state.setdefault("public_score", state.get("public_score", "N/A"))
            prompt_state.setdefault("private_score", state.get("private_score", "N/A"))
            prompt_state.setdefault("train_path", state.get("train_path", ""))
            prompt_state.setdefault("test_path", state.get("test_path", ""))
            prompt_state.setdefault("sample_submission_path", state.get("sample_submission_path", ""))
            prompt_state.setdefault("session_dir", str(state.get("session_dir", "")))
            prompt_state.setdefault("train_sample_frac", cfg.TRAIN_SAMPLE_FRAC)
            prompt_state.setdefault("train_sample_pct", cfg.TRAIN_SAMPLE_PCT)

            code_response = chain.invoke(prompt_state)

            code = extract_code_block(code_response)
            if not code:
                error_msg = "No code block found in LLM response"
                if logger:
                    logger.warning("%s: %s", step_name, error_msg)
                    logger.debug("LLM response: %s", code_response[:500])
                errors.append(error_msg)
                if attempts < max_attempts:
                    state["last_error"] = error_msg
                continue

            is_valid, validation_msg = validate_code(code)
            if not is_valid:
                if logger:
                    logger.warning("%s: Validation failed: %s", step_name, validation_msg)
                errors.append(validation_msg)
                if attempts < max_attempts:
                    state["last_error"] = validation_msg
                    state["previous_code"] = code
                continue

            code_file = Path(state["session_dir"]) / "code" / f"{step_name}.py"
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            if logger:
                logger.info("%s: Code saved to %s", step_name, code_file)

            if logger:
                logger.info("%s: Executing code...", step_name)
            success, output, updated_state = execute_code(code, state, timeout_sec)

            if success:
                if logger:
                    logger.info("%s: Execution successful", step_name)
                    if output:
                        logger.debug("%s: Output: %s", step_name, output[:500])
                updated_state[f"{step_name}_attempts"] = attempts
                updated_state[f"{step_name}_success"] = True
                updated_state[f"{step_name}_code"] = code
                updated_state[f"{step_name}_output"] = output
                updated_state.pop("last_error", None)
                updated_state.pop("previous_code", None)
                return updated_state, True
            else:
                error_msg = f"Execution failed: {output}"
                if logger:
                    logger.warning("%s: %s", step_name, error_msg)
                errors.append(error_msg)
                if attempts < max_attempts:
                    state["last_error"] = error_msg
                    state["previous_code"] = code

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if logger:
                logger.error("%s: %s", step_name, error_msg)
            errors.append(error_msg)
            if attempts < max_attempts:
                state["last_error"] = error_msg

    if logger:
        logger.error("%s: All %d attempts failed", step_name, max_attempts)
    state[f"{step_name}_attempts"] = attempts
    state[f"{step_name}_success"] = False
    state[f"{step_name}_errors"] = errors
    return state, False
