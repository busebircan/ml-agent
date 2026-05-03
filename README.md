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
| Agent brain | Groq (Llama 3.3 70b / Qwen2.5 72b) | Free tier |
| Research | HF Papers, HF Docs, GitHub search | Free |
| Training compute | GCP Vertex AI / Kaggle | Pay per use / Free tier |
| Agent orchestration | Runs on laptop | Free |

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
GROQ_API_KEY=<your-groq-api-key>       # free at console.groq.com
HF_TOKEN=<your-hugging-face-token>     # free at huggingface.co/settings/tokens
GITHUB_TOKEN=<your-github-token>       # for GitHub code search tool
```

Run:

```bash
ml-intern
```

---

## Switching the agent brain

All LLM calls go through `litellm` — swap the model with one line in `configs/cli_agent_config.json`:

```json
{ "model_name": "groq/llama-3.3-70b-versatile" }
```

Models currently tracked:

| Model | Provider | Notes |
|---|---|---|
| `groq/llama-3.3-70b-versatile` | Groq free | Baseline, good reasoning |
| `groq/qwen-qwen2.5-72b-instruct` | Groq free | Strong on code and tool calling |
| `deepseek-ai/DeepSeek-V4-Pro` | HF Router | Strong alternative |

---

## Roadmap

- [x] Phase 0 — Fork, setup, Groq configuration
- [ ] Phase 1 — System prompt code quality contract
- [ ] Phase 2 — Domain example scripts
- [ ] Phase 3 — Ruff + mypy linting pass
- [ ] Phase 4 — GCP Vertex AI tool
- [ ] Phase 5 — Code review sub-agent
- [ ] Phase 6 — Model evaluation harness
- [ ] Phase 7 — Domain hardening (CV, tabular, time-series, RL, NLP)

---

## Credits

Built on [huggingface/ml-intern](https://github.com/huggingface/ml-intern) — an open-source ML engineering agent by the Hugging Face team.
