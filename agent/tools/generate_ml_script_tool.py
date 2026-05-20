"""generate_ml_script tool — high-level ML script generation sub-agent.

The model calls this tool with structured parameters describing what it wants
to build. The tool:
  1. Loads the appropriate domain example script as a reference
  2. Makes a focused LLM call to adapt the example to the user's requirements
  3. Runs lint_python (ruff + mypy) and auto-fixes issues
  4. Runs code_review and auto-fixes CRITICAL/WARNING items
  5. Returns the finished, production-quality script

This is the primary way to generate ML code with local Ollama models, where
trying to write 200+ lines in a single text response is unreliable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from litellm import Message, acompletion

from agent.core import telemetry
from agent.core.llm_params import _resolve_llm_params
from agent.core.prompt_caching import with_prompt_caching
from agent.core.session import Event
from agent.tools.lint_tool import _lint_file
import tempfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain → example file mapping
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

_DOMAIN_EXAMPLES: dict[str, str] = {
    "cv": "cv_classification.py",
    "computer_vision": "cv_classification.py",
    "tabular": "tabular_classification.py",
    "nlp": "nlp_classification.py",
    "timeseries": "timeseries_forecasting.py",
    "time_series": "timeseries_forecasting.py",
    "rl": "rl_training.py",
    "reinforcement_learning": "rl_training.py",
}

# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------

GENERATE_ML_SCRIPT_TOOL_SPEC: dict[str, Any] = {
    "name": "generate_ml_script",
    "description": (
        "Generate a complete, production-quality ML Python script by adapting "
        "a domain reference example to the user's specific requirements.\n\n"
        "The tool handles all code quality steps internally: it adapts the "
        "example, runs ruff + mypy linting, runs code review, and auto-fixes "
        "issues before returning the finished script.\n\n"
        "Use this as the PRIMARY tool for any ML coding task:\n"
        "- Tabular classification/regression (LightGBM, XGBoost, sklearn)\n"
        "- Computer vision classification\n"
        "- NLP / text classification\n"
        "- Time-series forecasting\n"
        "- Reinforcement learning (Stable Baselines 3)\n\n"
        "Do NOT try to write ML scripts as plain text — call this tool instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": (
                    "ML domain. One of: tabular, cv, nlp, timeseries, rl. "
                    "Determines which reference example is used."
                ),
                "enum": ["tabular", "cv", "nlp", "timeseries", "rl"],
            },
            "task": {
                "type": "string",
                "description": (
                    "Clear description of what the script should do. Example: "
                    "'Binary classification on the Titanic dataset using LightGBM "
                    "with 5-fold CV and SHAP feature importance.' "
                    "Include dataset name, target variable, key requirements."
                ),
            },
            "framework": {
                "type": "string",
                "description": (
                    "Primary ML framework or library. Examples: lightgbm, xgboost, "
                    "sklearn, pytorch, transformers, stable-baselines3. "
                    "Leave empty to use the domain default."
                ),
            },
            "requirements": {
                "type": "string",
                "description": (
                    "Additional specific requirements, constraints, or context. "
                    "Examples: 'must run on CPU only', 'dataset columns: age, fare, "
                    "survived', 'output predictions to predictions.csv', "
                    "'use HuggingFace dataset: datasets/titanic'."
                ),
            },
        },
        "required": ["domain", "task"],
    },
}

# ---------------------------------------------------------------------------
# System prompt for the script-generation LLM call
# ---------------------------------------------------------------------------

_GENERATION_SYSTEM_PROMPT = """\
You are an ML engineering assistant that generates production-quality Python scripts.

You will be given:
1. A reference example script that demonstrates the expected code style and structure
2. A task description specifying what needs to be built

Your job: adapt the reference script to implement the requested task.

# Non-negotiable code quality requirements

Every script MUST have:
- TYPE HINTS: All function signatures fully annotated (args + return types)
- TYPED CONFIG: All hyperparameters in a dataclass or Pydantic BaseModel — no raw dicts
- CLI ENTRY POINT: argparse that populates the config. Runnable as `python script.py --arg val`
- LOGGING: Python logging module with module-level logger. Never use print() for status
- MODULAR STRUCTURE: Separate functions for data loading, model building, training, evaluation, export

# Output format

Return ONLY the Python script. No markdown fences, no explanation, no comments outside the code.
The output must be valid Python that can be saved directly to a .py file and run.
"""

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _load_example(domain: str) -> str | None:
    """Load the reference example script for the given domain."""
    filename = _DOMAIN_EXAMPLES.get(domain.lower())
    if not filename:
        return None
    path = _EXAMPLES_DIR / filename
    if not path.exists():
        logger.warning("Example file not found: %s", path)
        return None
    return path.read_text(encoding="utf-8")


def _get_fix_model(main_model: str) -> str:
    """Return a smaller/cheaper model for lint-fix and review passes.

    Generation needs the full model. Fixing lint errors and applying review
    feedback is a simpler task — a smaller model handles it fine and costs
    much less on the HF router free tier.
    """
    # Local Ollama — no cost, use same model
    if main_model.startswith("ollama/"):
        return main_model
    # Anthropic/OpenAI direct — use same model (billed separately, not HF credits)
    if main_model.startswith(("anthropic/", "openai/", "bedrock/")):
        return main_model
    # HF router — swap to Qwen2.5-7B for fix passes (14x cheaper than 72B)
    return "Qwen/Qwen2.5-7B-Instruct"


async def generate_ml_script_handler(
    arguments: dict[str, Any],
    session: Any = None,
    tool_call_id: str | None = None,
) -> tuple[str, bool]:
    """Generate, lint, and review an ML script, then return it."""
    domain = arguments.get("domain", "tabular").lower()
    task = arguments.get("task", "")
    framework = arguments.get("framework", "")
    requirements = arguments.get("requirements", "")

    if not task:
        return "No task description provided.", False
    if not session:
        return "No session available.", False

    # ── Unique agent id for UI status lines ─────────────────────────────
    if tool_call_id:
        _agent_id = tool_call_id
    else:
        import uuid
        _agent_id = uuid.uuid4().hex[:8]
    _label = f"generate_ml_script: {task[:60]}"

    async def _log(text: str) -> None:
        try:
            await session.send_event(
                Event(
                    event_type="tool_log",
                    data={"tool": "generate_ml_script", "log": text,
                          "agent_id": _agent_id, "label": _label},
                )
            )
        except Exception:
            pass

    # ── Load reference example ───────────────────────────────────────────
    # For local Ollama models, skip the full example — it doubles prompt size
    # and causes smaller models to spend all their time thinking about the example.
    _is_local = session.config.model_name.startswith("ollama/")
    if _is_local:
        await _log("Generating script (local model - no reference example)...")
        example_code = None
    else:
        await _log("Loading reference example...")
        example_code = _load_example(domain)
        if not example_code:
            await _log(f"No example found for domain '{domain}' — generating from scratch")

    if not example_code:
        example_section = ""
    else:
        example_section = (
            f"\n\n# REFERENCE EXAMPLE (match this style exactly)\n\n"
            f"```python\n{example_code}\n```"
        )

    # ── Build the generation prompt ──────────────────────────────────────
    user_prompt_parts = [f"Task: {task}"]
    if framework:
        user_prompt_parts.append(f"Framework: {framework}")
    if requirements:
        user_prompt_parts.append(f"Requirements:\n{requirements}")
    user_prompt_parts.append(
        "\nGenerate the complete Python script. "
        "Return ONLY the code — no markdown, no explanation."
    )
    user_prompt = "\n".join(user_prompt_parts) + example_section

    messages: list[Message] = [
        Message(role="system", content=_GENERATION_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]

    # ── Pick models — full model for generation, smaller for fix passes ──
    main_model = session.config.model_name
    gen_model = main_model
    fix_model = _get_fix_model(main_model)

    llm_params = _resolve_llm_params(
        gen_model,
        getattr(session, "hf_token", None),
    )
    fix_llm_params = _resolve_llm_params(
        fix_model,
        getattr(session, "hf_token", None),
    )

    # Local models: 240s is enough with think=False. Cloud models: 180s.
    _gen_timeout = 240 if _is_local else 180

    if not _is_local:
        await _log("Generating script...")

    # ── Generation call ──────────────────────────────────────────────────
    try:
        _msgs, _ = with_prompt_caching(messages, None, llm_params.get("model"))
        _t0 = time.monotonic()
        response = await acompletion(
            messages=_msgs,
            stream=False,
            timeout=_gen_timeout,
            **llm_params,
        )
        try:
            await telemetry.record_llm_call(
                session,
                model=gen_model,
                response=response,
                latency_ms=int((time.monotonic() - _t0) * 1000),
                finish_reason=(
                    response.choices[0].finish_reason if response.choices else None
                ),
                kind="generate_ml_script",
            )
        except Exception as _e:
            logger.debug("telemetry failed: %s", _e)

        script = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Script generation LLM call failed: %s", e)
        return f"Script generation failed: {e}", False

    # Strip markdown code fences if the model added them
    script = _strip_code_fences(script)

    if not script.strip():
        return "Script generation produced empty output.", False

    await _log("Running lint...")

    if _is_local:
        # Local models (Ollama): run lint as a quick local check only.
        # Skip LLM-based fix passes — each extra Ollama call adds 30-120s
        # of wait time for a 4B model. Return the script as-is; the user
        # can ask for fixes after reviewing the output.
        script, lint_report = await _lint_only(script, _log)
        review_report = "Code review: skipped for local model."
    else:
        # ── Lint + auto-fix loop (max 2 attempts) ───────────────────────
        script, lint_report = await _lint_and_fix(
            script, messages, fix_llm_params, session, fix_model, _log, max_attempts=2
        )
        await _log("Running code review...")
        # ── Code review + auto-fix (single pass) ────────────────────────
        script, review_report = await _review_and_fix(
            script, task, messages, fix_llm_params, session, fix_model, _log
        )

    await _log("Done.")

    # Suggest a filename from the task so the agent can write the file
    slug = task.lower().replace(" ", "_")[:40]
    suggested_filename = f"{slug}.py"

    # ── Return final script with a brief summary header ─────────────────
    if _is_local:
        quality_note = "Lint ran (no auto-fix on local model). Code review skipped to avoid extra wait time."
    else:
        quality_note = "Lint and code review ran internally; all issues were auto-fixed."
    output = (
        f"[generate_ml_script COMPLETE — SUCCESS]\n"
        f"{quality_note}\n"
        f"IMPORTANT: Do NOT call generate_ml_script again. The script is finished.\n"
        f"Next step: call the write tool to save the script to disk.\n"
        f"Suggested filename: {suggested_filename}\n\n"
        f"```python\n{script}\n```\n\n"
        f"---\n{lint_report}\n\n{review_report}"
    )
    return output, True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove markdown ```python ... ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```python or ```) and last (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].strip().startswith("```"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


async def _lint_only(script: str, log_fn: Any) -> tuple[str, str]:
    """Run lint once (no LLM fix). Returns (script, report)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="gml_lint_", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(script)
        tmp_path = Path(tmp.name)
    try:
        report = _lint_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    passed = "❌ FAILED" not in report
    if passed:
        return script, "Lint: ✅ PASSED"
    await log_fn("Lint issues found (auto-fix skipped for local model — review output).")
    return script, f"Lint: ⚠️ Issues found (not auto-fixed on local model)\n{report}"


async def _lint_and_fix(
    script: str,
    base_messages: list[Message],
    llm_params: dict[str, Any],
    session: Any,
    model: str,
    log_fn: Any,
    max_attempts: int = 2,
) -> tuple[str, str]:
    """Run lint; if it fails, ask LLM to fix and re-lint. Return (script, report)."""
    for attempt in range(max_attempts):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="gml_lint_", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(script)
            tmp_path = Path(tmp.name)

        try:
            report = _lint_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        passed = "❌ FAILED" not in report
        if passed:
            return script, "Lint: ✅ PASSED"

        if attempt < max_attempts - 1:
            await log_fn(f"Lint issues auto-correcting (attempt {attempt + 1})...")
            fix_messages = list(base_messages) + [
                Message(
                    role="user",
                    content=(
                        f"The script has lint errors. Fix ALL of them.\n\n"
                        f"Lint report:\n{report}\n\n"
                        f"Current script:\n```python\n{script}\n```\n\n"
                        f"Return ONLY the corrected Python script."
                    ),
                )
            ]
            try:
                _msgs, _ = with_prompt_caching(
                    fix_messages, None, llm_params.get("model")
                )
                _t0 = time.monotonic()
                response = await acompletion(
                    messages=_msgs, stream=False, timeout=180, **llm_params
                )
                try:
                    await telemetry.record_llm_call(
                        session,
                        model=model,
                        response=response,
                        latency_ms=int((time.monotonic() - _t0) * 1000),
                        finish_reason=(
                            response.choices[0].finish_reason
                            if response.choices
                            else None
                        ),
                        kind="generate_ml_script_lint_fix",
                    )
                except Exception:
                    pass
                fixed = response.choices[0].message.content or ""
                fixed = _strip_code_fences(fixed)
                if fixed.strip():
                    script = fixed
            except Exception as e:
                logger.warning("Lint fix LLM call failed: %s", e)

    return script, f"Lint: ❌ FAILED after {max_attempts} attempts\n{report}"


async def _review_and_fix(
    script: str,
    task: str,
    base_messages: list[Message],
    llm_params: dict[str, Any],
    session: Any,
    model: str,
    log_fn: Any,
) -> tuple[str, str]:
    """Run code review; if CRITICAL/WARNING found, do one fix pass."""
    from agent.tools.code_review_tool import CODE_REVIEW_SYSTEM_PROMPT

    user_content = f"Context: {task}\n\nReview this Python ML script:\n\n```python\n{script}\n```"
    review_messages: list[Message] = [
        Message(role="system", content=CODE_REVIEW_SYSTEM_PROMPT),
        Message(role="user", content=user_content),
    ]

    try:
        _msgs, _ = with_prompt_caching(
            review_messages, None, llm_params.get("model")
        )
        _t0 = time.monotonic()
        response = await acompletion(
            messages=_msgs, stream=False, timeout=120, **llm_params
        )
        try:
            await telemetry.record_llm_call(
                session,
                model=model,
                response=response,
                latency_ms=int((time.monotonic() - _t0) * 1000),
                finish_reason=(
                    response.choices[0].finish_reason if response.choices else None
                ),
                kind="generate_ml_script_review",
            )
        except Exception:
            pass
        review = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Code review call failed: %s", e)
        return script, f"Code review: skipped ({e})"

    has_issues = any(
        kw in review.upper() for kw in ("CRITICAL", "WARNING", "❌")
    )
    if not has_issues:
        return script, f"Code review: ✅ PASSED\n{review}"

    await log_fn("Code review issues auto-correcting...")
    fix_messages = list(base_messages) + [
        Message(
            role="user",
            content=(
                f"The script has code review issues. Fix ALL CRITICAL and WARNING items.\n\n"
                f"Review report:\n{review}\n\n"
                f"Current script:\n```python\n{script}\n```\n\n"
                f"Return ONLY the corrected Python script."
            ),
        )
    ]
    try:
        _msgs, _ = with_prompt_caching(fix_messages, None, llm_params.get("model"))
        _t0 = time.monotonic()
        response = await acompletion(
            messages=_msgs, stream=False, timeout=180, **llm_params
        )
        try:
            await telemetry.record_llm_call(
                session,
                model=model,
                response=response,
                latency_ms=int((time.monotonic() - _t0) * 1000),
                finish_reason=(
                    response.choices[0].finish_reason if response.choices else None
                ),
                kind="generate_ml_script_review_fix",
            )
        except Exception:
            pass
        fixed = response.choices[0].message.content or ""
        fixed = _strip_code_fences(fixed)
        if fixed.strip():
            script = fixed
    except Exception as e:
        logger.warning("Review fix LLM call failed: %s", e)

    return script, f"Code review: issues found and fixed.\n{review}"
