# Buse's ML Agent

A personal ML engineering agent for building end-to-end ML projects across computer vision, tabular, time-series, NLP, and reinforcement learning — running fully locally on a Windows laptop via Ollama.

Built on top of [huggingface/ml-intern](https://github.com/huggingface/ml-intern), customised for local-first inference, production-quality code generation, and Windows workflows.

---

## What it does

- **Runs locally** — default brain is `qwen3:4b` via Ollama (fits in 4GB VRAM), no cloud API required
- **Generates production-quality scripts** — typed configs, modular structure, argparse CLI, proper logging, ruff/mypy clean
- **Auto-pulls models** — fetches the configured Ollama model at startup if not already present
- **Self-corrects** — lints its own output before handing it over (cloud models also get a full code review pass)
- **Stays current** — swap the model brain live with `/model` while the agent is running

---

## Stack

| Layer | What | Cost |
|---|---|---|
| Agent brain | `ollama/qwen3:4b` — local, fits in 4GB VRAM | Free |
| Cloud fallback | HF Router (Qwen2.5-72B, Kimi-K2.6, DeepSeek V4 Pro…) | HF credits |
| Research | HF Papers, HF Docs, GitHub code search | Free |
| Agent hardware | Windows laptop, 32GB RAM + NVIDIA T550 4GB VRAM | — |

---

## Domains

- **Tabular** — LightGBM/XGBoost pipelines, StratifiedKFold CV, SHAP feature importance
- **Computer vision** — classification with EfficientNet/ViT, ImageNet pretrained weights
- **NLP** — text classification with DistilBERT/RoBERTa/DeBERTa
- **Time-series** — forecasting with time-based train/val splits, lag features
- **Reinforcement learning** — PPO/SAC via Stable Baselines 3, vectorised envs, EvalCallback

---

## Customisations over upstream ml-intern

### 1. `generate_ml_script` tool — NEW (`agent/tools/generate_ml_script_tool.py`)
Primary tool for all ML coding tasks. Instead of the model writing code as text, it calls this tool which:
- Adapts the matching domain reference example to the user's requirements
- Runs `ruff` + `mypy` lint locally (always)
- On cloud models: runs a code review pass + auto-fixes issues via a cheaper 7B fix model
- Returns the finished script ready to write to disk

### 2. Full Ollama streaming support (`agent/core/agent_loop.py`)
- Streaming enabled for Ollama — tokens appear as they're generated
- `<think>...</think>` blocks from qwen3 are filtered before reaching the UI
- `_sanitize_messages` uses `getattr` per field (not `vars()`) so `tool_calls`/`tool_call_id` are never dropped from multi-turn conversations

### 3. qwen3 think=False (`agent/core/llm_params.py`)
- `think=False` injected into every Ollama request for qwen3 models
- Eliminates silent 5–10 min extended-thinking pauses before each tool call
- `num_ctx: 16384` — fits in 4GB VRAM KV cache without overflow

### 4. Timestamp sanitization for all providers
- `_sanitize_messages` now strips non-standard fields for all providers except Anthropic/Bedrock
- Fixes 400 errors on Kimi-K2.6, Qwen3-32B, DeepSeek and any strict HF router provider

### 5. Auto-pull Ollama model at startup (`agent/main.py`)
- Checks `ollama list` on startup, pulls the configured model if missing
- Uses native terminal output for the pull progress bar

### 6. Custom banner
- "BUSE'S ML AGENT" in braille particle animation (apostrophe glyph added)
- "Built on Hugging Face's ML Intern" plain subtitle

### 7. Windows-first system prompt (`agent/prompts/system_prompt_local.yaml`)
- Instructs the model to use Windows paths everywhere
- Defines `generate_ml_script` as the primary coding tool with usage examples
- Includes domain knowledge: backbone selection, algorithm selection, common pitfalls

### 8. Reference examples (`examples/`)
- `tabular_classification.py`: updated to LightGBM 4.x callbacks API (`early_stopping()` + `log_evaluation()`), SHAP importance, `set_seeds()`

---

## Setup

```bash
git clone https://github.com/busebircan/ml-agent.git
cd ml-agent
uv sync
uv tool install -e .
```

Install [Ollama](https://ollama.com) — the agent will pull `qwen3:4b` (~2.5GB) automatically on first run.

Optionally create a `.env` file for cloud/research tools:

```bash
HF_TOKEN=<your-hugging-face-token>     # for HF Router fallback + HF tools
GITHUB_TOKEN=<your-github-token>       # for GitHub code search tool
```

Run:

```bash
ml-intern
```

---

## Switching the model brain

Use `/model` while the agent is running to switch live:

```
> /model ollama/qwen3:8b
> /model Qwen/Qwen2.5-72B-Instruct
> /model anthropic/claude-opus-4-6
```

Or change the default in `configs/cli_agent_config.json`:

```json
{ "model_name": "ollama/qwen3:4b" }
```

| Model | Where | Notes |
|---|---|---|
| `ollama/qwen3:4b` | Local | **Default** — fits in 4GB VRAM (~2.5GB), fast |
| `ollama/qwen3:8b` | Local | Better quality, CPU-only on T550 (too slow) |
| `Qwen/Qwen2.5-72B-Instruct` | HF Router | Cloud fallback, best HF router quality |
| `anthropic/claude-opus-4-6` | Anthropic | Best overall, requires `ANTHROPIC_API_KEY` |
| `deepseek-ai/DeepSeek-V4-Pro:deepinfra` | HF Router | Strong code model |

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
- [x] Phase 8 — Local-first Ollama support (qwen3:4b, streaming, think=False, generate_ml_script tool)

---

## Credits

Built on [huggingface/ml-intern](https://github.com/huggingface/ml-intern) — an open-source ML engineering agent by the Hugging Face team.
