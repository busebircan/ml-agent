"""Agent brain evaluation harness.

Runs a fixed prompt suite against any LiteLLM-compatible model and scores
each output against the project code quality contract. Use this to compare
model candidates before committing to one.

Usage:
    python evals/agent_eval.py --model ollama/qwen2.5-coder:32b
    python evals/agent_eval.py --model gemini/gemini-2.5-flash --output results/gemini.json
    python evals/agent_eval.py --compare results/qwen.json results/gemini.json

Scoring (0-1 per criterion, averaged to a final score 0-100):
    - code_extracted    : did the model return a Python code block?
    - has_type_hints    : type annotations on function signatures
    - has_dataclass     : typed Config dataclass (no raw dicts)
    - has_argparse      : argparse CLI entry point
    - has_logging       : logging module (not print)
    - is_modular        : load_data / build_model / train / evaluate / export
    - passes_ruff       : ruff check passes
    - passes_mypy       : mypy passes (--ignore-missing-imports)
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from litellm import completion

# ---------------------------------------------------------------------------
# Eval prompts — one per domain, fixed across all runs
# ---------------------------------------------------------------------------

EVAL_PROMPTS: list[dict[str, str]] = [
    {
        "id": "cv_classification",
        "domain": "computer_vision",
        "prompt": (
            "Write a complete, production-quality Python script for image classification "
            "using a pretrained ResNet-50 on a custom dataset stored in data/images/ "
            "(ImageFolder format, 10 classes). The script must: train for 20 epochs with "
            "AdamW and cosine LR schedule, log train loss and val accuracy each epoch, "
            "save the best checkpoint, and export to TorchScript. "
            "Use argparse for all hyperparameters."
        ),
    },
    {
        "id": "tabular_classification",
        "domain": "tabular",
        "prompt": (
            "Write a complete, production-quality Python script for binary classification "
            "on a tabular CSV dataset (data/train.csv, data/test.csv) using LightGBM "
            "with 5-fold stratified cross-validation. Handle categorical columns with "
            "label encoding. Report OOF AUC and save a submission.csv. "
            "Use argparse for all hyperparameters."
        ),
    },
    {
        "id": "timeseries_forecasting",
        "domain": "timeseries",
        "prompt": (
            "Write a complete, production-quality Python script for multi-series time-series "
            "forecasting using LightGBM. The dataset is a CSV with columns: date, series_id, "
            "value. Create lag features (lags 1-28) and rolling mean/std features (windows "
            "7, 14, 28). Use a time-based train/val split. Report MAE and RMSE. "
            "Use argparse for all hyperparameters."
        ),
    },
    {
        "id": "nlp_classification",
        "domain": "nlp",
        "prompt": (
            "Write a complete, production-quality Python script to fine-tune "
            "distilbert-base-uncased for binary text classification on the imdb dataset "
            "from HuggingFace. Use the Trainer API. Report accuracy and macro-F1. "
            "Save the best model locally. Use argparse for all hyperparameters."
        ),
    },
    {
        "id": "rl_training",
        "domain": "reinforcement_learning",
        "prompt": (
            "Write a complete, production-quality Python script to train a PPO agent "
            "on LunarLander-v2 using Stable Baselines 3. Use 8 parallel environments, "
            "an EvalCallback that saves the best model, and stop early if mean reward "
            "exceeds 200. Report mean and std reward at the end. "
            "Use argparse for all hyperparameters."
        ),
    },
]

SYSTEM_PROMPT = """\
You are an expert ML engineer. When asked to write a Python script, you MUST follow these rules:

1. TYPE HINTS: All function signatures must have complete type annotations.
2. TYPED CONFIG: All hyperparameters must live in a typed dataclass called Config. No raw dicts.
3. CLI: Every script must have an argparse entry point. No hard-coded values in main().
4. LOGGING: Use Python's logging module. Never use print() for status output.
5. MODULAR: Break scripts into named functions — at minimum:
   load_data(), build_model(), train(), evaluate(), export() or domain equivalents.
   No 200-line main() functions.

Return ONLY the Python script inside a ```python ... ``` code block. No explanation.
"""

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _extract_code(response: str) -> str | None:
    """Extract first ```python ... ``` block from response."""
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: bare code block
    match = re.search(r"```\s*(import|from|def|class).*?```", response, re.DOTALL)
    if match:
        return match.group(0).replace("```", "").strip()
    return None


def _has_type_hints(code: str) -> bool:
    """Check if function signatures have type annotations (args + return)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return False
    annotated = sum(
        1 for f in funcs
        if f.returns is not None or any(a.annotation for a in f.args.args)
    )
    return annotated / len(funcs) >= 0.7  # 70%+ of functions annotated


def _has_dataclass(code: str) -> bool:
    """Check for @dataclass decorator."""
    return "@dataclass" in code and "class Config" in code


def _has_argparse(code: str) -> bool:
    """Check for argparse usage."""
    return "argparse" in code and "add_argument" in code


def _has_logging(code: str) -> bool:
    """Check for logging module (not just print)."""
    has_logging = "import logging" in code or "logging.getLogger" in code
    has_print = bool(re.search(r'\bprint\s*\(', code))
    return has_logging and not has_print


def _is_modular(code: str) -> bool:
    """Check for at least 3 of the required function names."""
    required = ["load_data", "build_model", "train", "evaluate", "export",
                "make_env", "make_data", "make_model", "make_loaders"]
    found = sum(1 for name in required if re.search(rf"\bdef {name}\b", code))
    return found >= 3


def _run_linter(cmd: list[str], code: str) -> bool:
    """Write code to temp file, run cmd, return True if exit 0."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="eval_", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            cmd + [tmp], capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        Path(tmp).unlink(missing_ok=True)


def _passes_ruff(code: str) -> bool:
    return _run_linter(
        [sys.executable, "-m", "ruff", "check", "--select", "E,F,W"],
        code,
    )


def _passes_mypy(code: str) -> bool:
    return _run_linter(
        [sys.executable, "-m", "mypy", "--ignore-missing-imports", "--no-error-summary"],
        code,
    )


@dataclass
class PromptResult:
    prompt_id: str
    domain: str
    model: str
    latency_s: float
    code_extracted: bool
    has_type_hints: bool
    has_dataclass: bool
    has_argparse: bool
    has_logging: bool
    is_modular: bool
    passes_ruff: bool
    passes_mypy: bool
    score: float = field(init=False)
    raw_response: str = ""
    code: str = ""

    def __post_init__(self) -> None:
        criteria = [
            self.code_extracted, self.has_type_hints, self.has_dataclass,
            self.has_argparse, self.has_logging, self.is_modular,
            self.passes_ruff, self.passes_mypy,
        ]
        self.score = round(sum(criteria) / len(criteria) * 100, 1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_prompt(prompt: dict[str, str], model: str, timeout: int = 120) -> PromptResult:
    """Call the model with one eval prompt and score the output."""
    print(f"  [{prompt['id']}] calling {model}...", end=" ", flush=True)
    t0 = time.monotonic()
    try:
        resp = completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt["prompt"]},
            ],
            timeout=timeout,
            # Point ollama models at local server
            **({"api_base": "http://localhost:11434"} if model.startswith("ollama/") else {}),
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        latency = time.monotonic() - t0
        print(f"ERROR ({exc})")
        return PromptResult(
            prompt_id=prompt["id"], domain=prompt["domain"], model=model,
            latency_s=round(latency, 1),
            code_extracted=False, has_type_hints=False, has_dataclass=False,
            has_argparse=False, has_logging=False, is_modular=False,
            passes_ruff=False, passes_mypy=False,
            raw_response=str(exc),
        )

    latency = time.monotonic() - t0
    code = _extract_code(raw) or ""
    extracted = bool(code)

    result = PromptResult(
        prompt_id=prompt["id"],
        domain=prompt["domain"],
        model=model,
        latency_s=round(latency, 1),
        code_extracted=extracted,
        has_type_hints=_has_type_hints(code) if extracted else False,
        has_dataclass=_has_dataclass(code) if extracted else False,
        has_argparse=_has_argparse(code) if extracted else False,
        has_logging=_has_logging(code) if extracted else False,
        is_modular=_is_modular(code) if extracted else False,
        passes_ruff=_passes_ruff(code) if extracted else False,
        passes_mypy=_passes_mypy(code) if extracted else False,
        raw_response=raw,
        code=code,
    )
    print(f"score={result.score:.0f}  latency={latency:.1f}s")
    return result


def run_eval(model: str, output: Path, timeout: int = 120) -> list[PromptResult]:
    """Run all eval prompts against model, save results to output."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model}")
    print(f"{'='*60}")

    results: list[PromptResult] = []
    for prompt in EVAL_PROMPTS:
        result = run_prompt(prompt, model, timeout=timeout)
        results.append(result)

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nOverall score: {avg_score:.1f}/100")

    # Save to JSON
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "avg_score": round(avg_score, 1),
        "n_prompts": len(results),
        "results": [asdict(r) for r in results],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved to {output}")
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(paths: list[Path]) -> None:
    """Print a side-by-side comparison table of multiple eval result files."""
    datasets: list[dict] = []
    for p in paths:
        datasets.append(json.loads(p.read_text(encoding="utf-8")))

    criteria = [
        "code_extracted", "has_type_hints", "has_dataclass",
        "has_argparse", "has_logging", "is_modular",
        "passes_ruff", "passes_mypy",
    ]

    models = [d["model"] for d in datasets]
    col_w = max(len(m) for m in models) + 2

    header = f"{'Criterion':<22}" + "".join(f"{m:<{col_w}}" for m in models)
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for criterion in criteria:
        row = f"{criterion:<22}"
        for ds in datasets:
            pct = sum(
                r[criterion] for r in ds["results"] if isinstance(r[criterion], bool)
            ) / len(ds["results"]) * 100
            row += f"{pct:>5.0f}%{'':<{col_w-7}}"
        print(row)

    print(f"{'-'*len(header)}")
    score_row = f"{'OVERALL SCORE':<22}"
    for ds in datasets:
        score_row += f"{ds['avg_score']:>5.1f} {'':<{col_w-7}}"
    print(score_row)

    latency_row = f"{'avg latency (s)':<22}"
    for ds in datasets:
        avg_lat = sum(r["latency_s"] for r in ds["results"]) / len(ds["results"])
        latency_row += f"{avg_lat:>5.1f}s{'':<{col_w-7}}"
    print(latency_row)
    print(f"{'='*len(header)}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an agent brain model against the fixed ML prompt suite."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run eval against a model")
    run_p.add_argument("--model", required=True,
                       help="LiteLLM model id, e.g. ollama/qwen2.5-coder:32b")
    run_p.add_argument("--output", type=Path, default=None,
                       help="Output JSON path (default: results/<model-slug>.json)")
    run_p.add_argument("--timeout", type=int, default=120,
                       help="Per-prompt timeout in seconds (default: 120)")
    run_p.add_argument("--prompts", nargs="+",
                       help="Subset of prompt IDs to run (default: all)")

    cmp_p = sub.add_parser("compare", help="Compare result JSON files")
    cmp_p.add_argument("files", nargs="+", type=Path,
                       help="Result JSON files to compare")

    # Shortcut: python agent_eval.py --model ... (legacy single-command form)
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=120, help=argparse.SUPPRESS)
    parser.add_argument("--compare", nargs="+", type=Path, dest="compare_files",
                        help="Compare result JSON files (shortcut)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Shortcut mode: --compare
    if hasattr(args, "compare_files") and args.compare_files:
        compare(args.compare_files)
        return

    if args.command == "compare":
        compare(args.files)
        return

    # run mode (either via subcommand or legacy --model flag)
    model = getattr(args, "model", None)
    if not model:
        print("Provide --model. Example: python evals/agent_eval.py run --model ollama/qwen2.5-coder:32b")
        sys.exit(1)

    slug = model.replace("/", "_").replace(":", "-").replace(".", "-")
    output = args.output or Path(f"evals/results/{slug}.json")
    timeout = args.timeout

    prompts = EVAL_PROMPTS
    if hasattr(args, "prompts") and args.prompts:
        prompts = [p for p in EVAL_PROMPTS if p["id"] in args.prompts]
        if not prompts:
            print(f"No prompts matched. Available: {[p['id'] for p in EVAL_PROMPTS]}")
            sys.exit(1)

    run_eval(model, output, timeout)


if __name__ == "__main__":
    main()
