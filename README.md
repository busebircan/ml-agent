# Buse's ML Agent

A personal ML engineering agent for building end-to-end ML projects across computer vision, tabular, time-series, NLP, and reinforcement learning — running on a Windows laptop, locally via Ollama or via the HF Router cloud.

Built on top of [huggingface/ml-intern](https://github.com/huggingface/ml-intern), heavily customised for local-first inference, production-quality code generation, and Windows workflows.

---

## What it does

- **Generates ML scripts** — types, lints, and reviews its own code before handing it over
- **Runs locally or in the cloud** — default is `Qwen/Qwen2.5-Coder-32B-Instruct` via HF Router; switch to a local Ollama model with `/model ollama/ml-agent`
- **Saves to your desktop** — tells the model your Windows username and paths so files land in the right place
- **Self-corrects** — cloud path: full ruff + mypy + code review + auto-fix loop; local path: lint-only (too slow otherwise)
- **Doom-loop proof** — repetition guard, duplicate-call guard, and timeout handling prevent infinite retry spirals
- **Switch models live** — `/model` while the agent is running, no restart needed

---

## Stack

| Layer | What | Notes |
|---|---|---|
| Default model | `Qwen/Qwen2.5-Coder-32B-Instruct` via HF Router | Best quality, cloud inference, free with HF token |
| Local model | `ollama/ml-agent` (qwen2.5-coder:7b-instruct) | Offline use, 4.4GB, 15–20 tok/s on hybrid GPU |
| Hardware | Windows laptop, 32GB RAM, NVIDIA T550 4GB VRAM | Local model uses partial GPU offload |
| Research | HF Papers, HF Docs, GitHub code search | Free |

---

## Domains

- **Tabular** — LightGBM/XGBoost pipelines, StratifiedKFold CV, SHAP feature importance
- **Computer vision** — classification with EfficientNet/ViT, ImageNet pretrained weights
- **NLP** — text classification with DistilBERT/RoBERTa/DeBERTa
- **Time-series** — forecasting with time-based train/val splits, lag features
- **Reinforcement learning** — PPO/SAC via Stable Baselines 3, vectorised envs, EvalCallback

---

## Setup

```bash
git clone https://github.com/busebircan/ml-agent.git
cd ml-agent
uv sync
uv tool install -e .
```

Create a `.env` or set environment variables:

```bash
HF_TOKEN=<your-hugging-face-token>     # required for HF Router cloud model + HF tools
GITHUB_TOKEN=<your-github-token>       # optional, for GitHub code search tool
```

Run:

```bash
ml-intern
```

The agent starts with `Qwen/Qwen2.5-Coder-32B-Instruct` via HF Router by default. No Ollama required for the default setup.

---

## Local model setup (offline / no cloud)

For fully offline use, set up the optimised local model:

```powershell
# Pull the model (~4.4GB)
ollama pull qwen2.5-coder:7b-instruct

# Create the optimised variant (temperature 0.1, hybrid GPU offload, 4096 ctx)
ollama create ml-agent -f Modelfile.qwen25coder

# Switch to it
ml-intern --model ollama/ml-agent
# or set "model_name": "ollama/ml-agent" in configs/cli_agent_config.json
```

The `Modelfile.qwen25coder` bakes in the settings from local LLM benchmarks:
- `temperature 0.1` — biggest fix for tool-call reliability (default 0.8 causes hallucinations)
- `num_gpu 18` — 18 layers on T550 VRAM, 14 on CPU; ~15–20 tok/s vs 8–12 CPU-only
- `num_ctx 4096` — half the KV cache vs 8192, keeps VRAM under 4GB
- `mmap 0` — model fully in RAM for faster steady-state generation

For pure CPU (no GPU), change `num_gpu 0` in the Modelfile. Adjust `num_thread` to your physical core count minus 2.

---

## Model options

Use `/model` while running, or change `model_name` in `configs/cli_agent_config.json`:

| Model | Where | Notes |
|---|---|---|
| `Qwen/Qwen2.5-Coder-32B-Instruct` | HF Router | **Default** — best quality, free with HF token |
| `meta-llama/Llama-3.3-70B-Instruct` | HF Router | Best tool calling on HF Router |
| `ollama/ml-agent` | Local | Offline, ~15–20 tok/s hybrid GPU |
| `ollama/qwen2.5-coder:7b-instruct` | Local | Same model, without custom Modelfile |
| `ollama/llama3.1:8b` | Local | Most validated Ollama tool calling |
| `anthropic/claude-opus-4-7` | Anthropic API | Best overall, requires `ANTHROPIC_API_KEY` |
| `openai/gpt-4o` | OpenAI API | Requires `OPENAI_API_KEY` |

> **Avoid `ollama/qwen3:4b`** — confirmed tool-name hallucination bug in Ollama (issue #11135). Use `qwen2.5-coder:7b-instruct` instead.

---

## Customisations over upstream ml-intern

### `generate_ml_script` tool
Primary tool for all ML coding tasks. Prevents the model from writing code as plain text:
- Generates the script via a focused sub-agent LLM call
- Runs `ruff` + `mypy` lint (always)
- Cloud models: code review pass + auto-fix via a cheaper 7B model
- Session-level duplicate-call guard — blocks the second call immediately so the model moves straight to `write`
- Explicit timeout error message: tells the model "DO NOT retry" on failure so it reports to the user instead of looping

### Ollama streaming + text tool call parsing
- Full streaming support for Ollama models
- `assistant_chunk` events suppressed for Ollama (chunks fired before knowing if content is a tool call or text — suppressing prevents raw JSON from leaking to the UI)
- `_parse_ollama_text_tool_calls` handles the `{ "name": ..., "arguments": ... }` plain-text format that Ollama models use as a fallback when native tool_calls aren't emitted
- `<think>...</think>` blocks from qwen3 filtered before reaching the UI

### Doom-loop and retry protection
- Repetition guard (doom loop detector) threshold: 2 identical consecutive calls (not 3)
- Guard message simplified — "do NOT call more tools, send a text message to the user" — vague instructions confused small models into calling `function_name` literally
- MCP tool calls wrapped in `asyncio.wait_for(15s)` — hallucinated tool names no longer hang the agent indefinitely
- Timeout error in `generate_ml_script` returns explicit "DO NOT retry" message

### Windows-specific fixes
- `_safe_char()` helper in `terminal_display.py` — falls back to ASCII for box-drawing characters (`▸`→`>`, `✓`→`+`) that Windows cp1252 terminals can't encode
- `curl whoami` at startup replaced with `huggingface_hub.HfApi().whoami()` — the curl subprocess was timing out (rc=28) while the Python library worked fine
- Ctrl+C quit window widened 1.0s → 3.0s — `add_signal_handler` unavailable on Windows so the double-press window is tighter than on Linux
- Linux shell commands (`mkdir -p`, etc.) banned from system prompt; `write` tool used for file creation instead
- `local_mode` derived from model name (`ollama/` prefix) rather than hardcoded `True` — cloud models now get all 30 tools and the full system prompt

### System prompt (`system_prompt_local.yaml`)
- Windows username and Desktop path explicitly stated so files land in the right place
- `generate_ml_script` = one call, then `write` — no second call under any circumstances
- Conversational guard: "just reply directly" for greetings/questions, no tools

### `_sanitize_messages`
Uses `getattr` per field instead of `vars()` on litellm Message objects — `vars()` silently drops Pydantic fields like `tool_calls` and `tool_call_id`, breaking multi-turn tool-call history.

---

## Roadmap

- [x] Phase 0 — Fork, setup, local Ollama configuration
- [x] Phase 1 — System prompt code quality contract
- [x] Phase 2 — Domain example scripts
- [x] Phase 3 — Ruff + mypy linting pass
- [x] Phase 4 — GCP Vertex AI tool
- [x] Phase 5 — Code review sub-agent
- [x] Phase 6 — Model evaluation harness
- [x] Phase 7 — Domain hardening (CV, tabular, time-series, RL, NLP)
- [x] Phase 8 — Local-first Ollama support (streaming, think=False, generate_ml_script tool)
- [x] Phase 9 — Windows hardening (Unicode, cp1252, Ctrl+C, whoami, Linux command guard)
- [x] Phase 10 — Reliability (doom loop guard, retry protection, session duplicate-call guard)
- [x] Phase 11 — CPU/GPU optimisation (qwen2.5-coder:7b, temperature 0.1, num_ctx 4096, hybrid offload)
- [ ] Phase 12 — Evaluation harness: benchmark generate_ml_script quality across models

---

## Credits

Built on [huggingface/ml-intern](https://github.com/huggingface/ml-intern) — an open-source ML engineering agent by the Hugging Face team.
