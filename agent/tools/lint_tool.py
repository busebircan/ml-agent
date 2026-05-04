"""Ruff + mypy linting tool.

Runs ruff (style/correctness) and mypy (type checking) on generated Python
code. The agent calls this before returning any script and self-corrects
until it passes both checks.

Usage by the agent:
    lint_python({"code": "<full script as string>"})
    lint_python({"file_path": "/path/to/script.py"})

Returns a structured report: PASSED or FAILED with annotated output.
The agent must fix all issues and re-lint until the result is PASSED.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


LINT_TOOL_SPEC: dict[str, Any] = {
    "name": "lint_python",
    "description": (
        "Run ruff and mypy on a Python script and return the results.\n"
        "\n"
        "Call this tool on EVERY generated Python script before returning it to "
        "the user. Fix all reported issues and re-lint until the result is PASSED.\n"
        "\n"
        "Provide either:\n"
        "  - `code`: the full script as a string (written to a temp file)\n"
        "  - `file_path`: absolute path to an existing .py file\n"
        "\n"
        "Returns a PASSED / FAILED report with annotated ruff and mypy output.\n"
        "Common fixes:\n"
        "  - ruff E/W: style — fix imports, spacing, unused vars\n"
        "  - ruff ANN: missing type annotations — add them\n"
        "  - mypy errors: add/fix type hints, fix incompatible types\n"
        "  - mypy 'import could not be resolved': install the stub or add "
        "    `# type: ignore[import-untyped]`\n"
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "code": {
                "type": "string",
                "description": "Full Python script source to lint (written to a temp file).",
            },
            "file_path": {
                "type": "string",
                "description": "Absolute path to an existing .py file to lint.",
            },
        },
    },
}


def _run(cmd: list[str], cwd: str | None = None) -> tuple[str, int]:
    """Run a subprocess and return (stdout+stderr, returncode)."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return (result.stdout + result.stderr).strip(), result.returncode


def _lint_file(path: Path) -> str:
    """Run ruff and mypy on path, return formatted report."""
    python = sys.executable
    lines: list[str] = []

    # ── ruff check (lint) ────────────────────────────────────────────────
    ruff_out, ruff_code = _run([
        python, "-m", "ruff", "check",
        "--select", "E,W,F,ANN,N,B,I",
        "--ignore", "ANN101,ANN102,ANN401",
        str(path),
    ])
    if ruff_code == 0:
        lines.append("ruff check:   PASSED")
    else:
        lines.append("ruff check:   FAILED")
        lines.append(ruff_out)

    lines.append("")

    # ── ruff format (check only, no write) ──────────────────────────────
    fmt_out, fmt_code = _run([
        python, "-m", "ruff", "format", "--check", str(path),
    ])
    if fmt_code == 0:
        lines.append("ruff format:  PASSED")
    else:
        lines.append("ruff format:  FAILED (formatting issues)")
        lines.append(fmt_out)

    lines.append("")

    # ── mypy ─────────────────────────────────────────────────────────────
    mypy_out, mypy_code = _run([
        python, "-m", "mypy",
        "--ignore-missing-imports",
        "--no-error-summary",
        str(path),
    ])
    if mypy_code == 0:
        lines.append("mypy:         PASSED")
    else:
        lines.append("mypy:         FAILED")
        lines.append(mypy_out)

    # ── overall verdict ──────────────────────────────────────────────────
    passed = ruff_code == 0 and fmt_code == 0 and mypy_code == 0
    verdict = "✅ PASSED — all checks clean." if passed else (
        "❌ FAILED — fix the issues above and re-run lint_python."
    )
    lines.insert(0, f"{'='*60}")
    lines.insert(1, verdict)
    lines.insert(2, f"{'='*60}")
    lines.insert(3, "")

    return "\n".join(lines)


async def lint_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    """Tool handler: lint code string or file path."""
    code: str | None = args.get("code")
    file_path: str | None = args.get("file_path")

    if not code and not file_path:
        return "Provide either `code` or `file_path`.", False

    if file_path:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}", False
        report = _lint_file(path)
        passed = "❌ FAILED" not in report
        return report, passed

    # Write code to a temp file and lint it
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="lint_", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        report = _lint_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    passed = "❌ FAILED" not in report
    return report, passed
