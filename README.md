# ML Agent

A personal ML engineering agent for building end-to-end ML projects across computer vision, tabular, time-series, NLP, and reinforcement learning — with a research-first approach that stays current with the latest literature.

Built on top of [huggingface/ml-intern](https://github.com/huggingface/ml-intern), customised for production-quality code generation and GCP-based training workflows.

---

## What it does

- **Researches before it codes** — crawls papers, citation graphs, and live HF docs to find current best practices and APIs before writing a single line
- **Generates production-quality scripts** — typed configs, modular structure, argparse CLI, proper logging, ruff and mypy clean
- **Self-corrects** — lints and reviews its own output before handing it over
- **Runs training on GCP / Kaggle** — submits jobs to Vertex AI or generates Kaggle-compatible notebooks
- **Stays current** — swap the model brain with one config line as better models are released

---

## Stack

| Layer | What | Cost |
|---|---|---|
| Agent brain | Qwen2.5-Coder 32B via Ollama (local) | Free |
| Research | HF Papers, HF Docs, GitHub search | Free |
| Training compute | GCP Vertex AI / Kaggle | Pay per use / Free tier |
| Agent orchestration | Windows laptop, 32GB RAM + NVIDIA T550 | Free |

---

## Domains

**Primary**
- Computer vision — classification, object detection, image labelling
- Tabular — XGBoost/LightGBM pipelines, feature engineering
- Time-series — forecasting, optimisation

**Secondary**
- NLP — text classification, fine-tuning language models
- Reinforcement learning — Stable Baselines 3 / CleanRL workflows

**Project types**
- Time-series forecasting and optimisation
- Image labelling pipelines
- Recommendation engines
- End-to-end training → GCP Vertex AI → HF Hub

---

## Customisations over base ml-intern

### 1. Production code quality contract (`agent/prompts/system_prompt_v3.yaml`)
Every generated script must have:
- Type hints on all function signatures
- Typed config via `dataclass` or Pydantic — no raw dicts
- `argparse` or Hydra CLI entry point
- `logging` not `print`
- Modular structure: data loading, model, training, evaluation, export as separate functions

### 2. Domain example scripts (`examples/`)
Reference implementations for each domain that the agent matches in style and structure.

### 3. Linting pass (`agent/tools/lint_tool.py`)
Agent runs `ruff` and `mypy` on all generated code and self-corrects before returning output.

### 4. Code review sub-agent (`agent/tools/code_review_tool.py`)
A second LLM pass reviews generated code for ML bugs the linter cannot catch — data leakage, wrong metrics, wrong loss functions, GCP-specific issues.

### 5. GCP Vertex AI tool (`agent/tools/gcp_vertex_tool.py`)
Replaces HF job submission with native Vertex AI job submission — generates GCP-native training scripts from the start.

### 6. Model evaluation harness (`evals/agent_eval.py`)
Fixed prompt suite for benchmarking agent brain candidates. Swap models by changing one config line, compare JSON results.

---

## Setup

```bash
git clone https://github.com/busebircan/ml-agent.git
cd ml-agent
uv sync
uv tool install -e .
```

Create a `.env` file in the project root:

```bash
HF_TOKEN=<your-hugging-face-token>     # free at huggingface.co/settings/tokens
GITHUB_TOKEN=<your-github-token>       # for GitHub code search tool
```

Install [Ollama](https://ollama.com), then pull the model (~20GB):

```bash
ollama pull qwen2.5-coder:32b
```

Run:

```bash
ml-intern
```

> **Hardware note:** 32GB RAM is the minimum for Qwen2.5-Coder 32B (Q4, ~20GB). A discrete GPU with CUDA support (even 4GB VRAM) will offload layers and speed up inference. Mac Mini M4 Pro (24GB unified RAM + Neural Engine) is the upgrade path for faster local inference.

---

## Switching the agent brain

All LLM calls go through `litellm` — swap the model with one line in `configs/cli_agent_config.json`:

```json
{ "model_name": "ollama/qwen2.5-coder:32b" }
```

Models tracked:

| Model | Provider | Notes |
|---|---|---|
| `ollama/qwen2.5-coder:32b` | Local (Ollama) | Primary — best coding model at this size, fits 24GB |
| `ollama/llama3.3:70b` | Local (Ollama) | Larger reasoning, Q2 quant for 24GB |
| `deepseek-ai/DeepSeek-V4-Pro` | HF Router | Cloud fallback |

---

## Roadmap

- [x] Phase 0 — Fork, setup, local Ollama configuration
- [x] Phase 1 — System prompt code quality contract
- [x] Phase 2 — Domain example scripts
- [x] Phase 3 — Ruff + mypy linting pass
- [ ] Phase 4 — GCP Vertex AI tool
- [ ] Phase 5 — Code review sub-agent
- [ ] Phase 6 — Model evaluation harness
- [ ] Phase 7 — Domain hardening (CV, tabular, time-series, RL, NLP)

---

## Credits

Built on [huggingface/ml-intern](https://github.com/huggingface/ml-intern) — an open-source ML engineering agent by the Hugging Face team.
