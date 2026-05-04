"""Code review sub-agent tool.

Runs a second LLM pass over generated Python code to catch ML bugs
that ruff and mypy cannot detect — data leakage, wrong metrics,
wrong loss functions, shape errors, missing eval/no_grad, GCP issues.

The review runs in an independent context (does not pollute the main
conversation) and returns a structured PASSED / FAILED report.

The main agent must fix all CRITICAL and WARNING items before returning
code to the user.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from litellm import Message, acompletion

from agent.core import telemetry
from agent.core.llm_params import _resolve_llm_params
from agent.core.prompt_caching import with_prompt_caching
from agent.core.session import Event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — ML bug checklist
# ---------------------------------------------------------------------------

CODE_REVIEW_SYSTEM_PROMPT = """\
You are a senior ML engineer doing a code review. Your job is to find bugs
that static analysis tools (ruff, mypy) cannot catch. Focus exclusively on
correctness issues — not style.

Review the code against each item in the checklist below. For each issue
found, report it with severity and a concrete fix.

# Checklist

## Data integrity
- [ ] DATA LEAKAGE: Is test/val data used during training (fitted scaler,
      computed statistics, label encoder trained on full dataset)?
- [ ] FUTURE LEAKAGE: In time-series, do features use future information
      (lag=0, rolling windows that include the current row)?
- [ ] TRAIN/VAL SPLIT: Is the split done BEFORE any feature engineering
      that uses global statistics? Is stratification used where needed?
- [ ] DATASET FORMAT: Are column names hard-coded without verification?
      Does the code assume dataset format without checking?
- [ ] SHUFFLE: Is training data shuffled? Is val/test data NOT shuffled?

## Model correctness
- [ ] LOSS FUNCTION: Is the loss function correct for the task?
      (CrossEntropy for multi-class, BCEWithLogits for binary/multi-label,
       MSE/MAE for regression, FocalLoss only when class imbalance warrants it)
- [ ] OUTPUT ACTIVATION: Is softmax/sigmoid applied at the right place?
      (Never inside the model when using CrossEntropyLoss/BCEWithLogits —
       those expect raw logits. Apply only at inference for probabilities.)
- [ ] METRIC CORRECTNESS: Is the right metric used for the task?
      (Accuracy inappropriate for imbalanced classes — use F1/AUC.
       RMSE hides outlier impact vs MAE. BLEU alone insufficient for generation.)
- [ ] CLASS IMBALANCE: If dataset is imbalanced, is it handled?
      (WeightedRandomSampler, class_weight, FocalLoss, or oversampling)
- [ ] TENSOR SHAPES: Are tensor shapes consistent through the forward pass?
      Flag any reshape/squeeze/unsqueeze that looks suspicious.
- [ ] DTYPE: Are inputs cast to the right dtype before model forward?
      (float32 for most models, long for class indices passed to loss)

## Training loop
- [ ] MODEL MODES: Is model.train() called at the start of each train epoch?
      Is model.eval() called before validation/inference?
- [ ] NO_GRAD: Is torch.no_grad() used during validation and inference?
- [ ] GRADIENT ACCUMULATION: If used, is optimizer.zero_grad() called at
      the right step (not every batch)?
- [ ] GRADIENT CLIPPING: If used, is it applied BEFORE optimizer.step()?
- [ ] LR SCHEDULER: Is scheduler.step() called at the right frequency
      (epoch vs batch)? Is it called AFTER optimizer.step()?
- [ ] EARLY STOPPING: Is the best model checkpoint saved and loaded at end?
      (Not just the last epoch's weights)
- [ ] SEED: Is a random seed set for reproducibility (torch, numpy, random)?

## Data loading
- [ ] DATALOADER WORKERS: Are num_workers > 0? Is pin_memory used with CUDA?
- [ ] AUGMENTATION LEAKAGE: Are augmentations applied to val/test data?
      (Only normalisation should apply to val/test, not random transforms)
- [ ] IMAGE NORMALISATION: Are the right mean/std used for the backbone?
      (ImageNet: [0.485,0.456,0.406] / [0.229,0.224,0.225])

## Persistence and export
- [ ] MODEL SAVED: Is the trained model explicitly saved before the script ends?
- [ ] GCS OUTPUT: If running on Vertex AI, is the model saved to the GCS
      output_dir (AIP_MODEL_DIR env var), not just local disk?
- [ ] CHECKPOINT COMPLETENESS: Does the checkpoint include everything needed
      for inference? (model weights, tokenizer, label mappings, scaler stats)

## GCP / Vertex AI specific
- [ ] TIMEOUT: Is the job timeout appropriate for the model size?
      (Never less than 2h for any real training run)
- [ ] EPHEMERAL STORAGE: Does the script assume local files will persist
      after the job ends? (They won't — save to GCS)
- [ ] RESOURCE UTILISATION: Is the batch size reasonable for the selected
      GPU? Is gradient checkpointing used for large models?

# Output format

Start with one of:
  ✅ PASSED — no issues found.
  ❌ FAILED — issues found (list below).

Then for each issue:
  [CRITICAL] <issue> — <why it's wrong> — Fix: <concrete fix>
  [WARNING]  <issue> — <why it's wrong> — Fix: <concrete fix>
  [INFO]     <minor observation, not blocking>

Severity guide:
  CRITICAL — will produce wrong results or crash silently
  WARNING  — will likely hurt model quality or cause subtle bugs
  INFO     — good practice, not strictly required

Be terse. One line per issue. No preamble. No praise.
If a check is not applicable (e.g. no time-series → skip FUTURE LEAKAGE),
skip it silently.
"""

# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------

CODE_REVIEW_TOOL_SPEC: dict[str, Any] = {
    "name": "code_review",
    "description": (
        "Run a second LLM pass over a generated Python ML script to catch bugs "
        "that ruff and mypy cannot detect.\n"
        "\n"
        "Checks for: data leakage, future leakage in time-series, wrong loss "
        "functions, wrong metrics for imbalanced data, missing model.eval() / "
        "torch.no_grad(), augmentation applied to val set, model not saved, "
        "GCS output missing for Vertex AI jobs, and more.\n"
        "\n"
        "Call this AFTER lint_python passes and BEFORE returning code to the user.\n"
        "\n"
        "Fix all CRITICAL and WARNING items, then re-run code_review until the "
        "result is ✅ PASSED.\n"
        "\n"
        "Provide either:\n"
        "  - `code`: the full script as a string\n"
        "  - `file_path`: path to an existing .py file\n"
        "\n"
        "Optionally provide `context` to describe the task (e.g. 'binary "
        "classification on imbalanced dataset, running on Vertex AI T4')."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "code": {
                "type": "string",
                "description": "Full Python script source to review.",
            },
            "file_path": {
                "type": "string",
                "description": "Absolute path to an existing .py file to review.",
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional task context: what the script does, dataset characteristics "
                    "(imbalanced? time-series?), target hardware (Vertex AI? local?)."
                ),
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _get_review_model(main_model: str) -> str:
    """Use same model for review — we're already local/free."""
    return main_model


async def code_review_handler(
    arguments: dict[str, Any],
    session=None,
    tool_call_id: str | None = None,
    **_kw,
) -> tuple[str, bool]:
    """Run ML code review as a sub-agent call."""
    code: str | None = arguments.get("code")
    file_path: str | None = arguments.get("file_path")
    context: str = arguments.get("context", "")

    if not code and not file_path:
        return "Provide either `code` or `file_path`.", False

    # Load code from file if needed
    if file_path and not code:
        try:
            from pathlib import Path
            code = Path(file_path).read_text(encoding="utf-8")
        except Exception as exc:
            return f"Could not read file {file_path}: {exc}", False

    if not session:
        # Offline fallback — no session, can't call LLM as sub-agent.
        return "No session available for code review sub-agent.", False

    # Progress event
    _agent_label = "code_review"
    if tool_call_id:
        _agent_id = tool_call_id
    else:
        import uuid
        _agent_id = uuid.uuid4().hex[:8]

    async def _log(text: str) -> None:
        try:
            await session.send_event(
                Event(event_type="tool_log", data={
                    "tool": "code_review",
                    "log": text,
                    "agent_id": _agent_id,
                    "label": _agent_label,
                })
            )
        except Exception:
            pass

    await _log("Running ML code review...")

    # Build messages for sub-agent
    user_content = "Review this Python ML script:\n\n```python\n" + code + "\n```"
    if context:
        user_content = f"Context: {context}\n\n{user_content}"

    messages: list[Message] = [
        Message(role="system", content=CODE_REVIEW_SYSTEM_PROMPT),
        Message(role="user", content=user_content),
    ]

    main_model = session.config.model_name
    review_model = _get_review_model(main_model)
    llm_params = _resolve_llm_params(
        review_model,
        getattr(session, "hf_token", None),
    )

    try:
        _msgs, _ = with_prompt_caching(messages, None, llm_params.get("model"))
        _t0 = time.monotonic()
        response = await acompletion(
            messages=_msgs,
            tools=None,   # single-shot review — no tool calls
            stream=False,
            timeout=180,
            **llm_params,
        )
        try:
            await telemetry.record_llm_call(
                session,
                model=review_model,
                response=response,
                latency_ms=int((time.monotonic() - _t0) * 1000),
                finish_reason=response.choices[0].finish_reason if response.choices else None,
                kind="code_review",
            )
        except Exception as _telem_err:
            logger.debug("code_review telemetry failed: %s", _telem_err)

        review = response.choices[0].message.content or ""
        passed = review.startswith("✅ PASSED")
        await _log("Review complete.")
        return review or "Code review returned no output.", passed

    except Exception as exc:
        logger.exception("code_review sub-agent error")
        return f"Code review error: {exc}", False
