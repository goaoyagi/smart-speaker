#!/usr/bin/env python3
"""
Guard test for the logging rule in AGENTS.md.

Unlike the other test modules this one does not mirror a single `src/`
module: it checks a repository-wide rule. `src/` must log through
`logging.getLogger(__name__)` so output is routed and filtered
consistently, while `scripts/` holds manually run tools that talk to the
operator directly and may print.
"""

import ast
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _print_calls(path):
    """Return the line numbers of every print() call in a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def test_src_modules_do_not_call_print():
    offenders = [
        f"{path.relative_to(SRC_DIR.parent)}:{lineno}"
        for path in sorted(SRC_DIR.rglob("*.py"))
        for lineno in _print_calls(path)
    ]

    assert not offenders, (
        "src/ must log instead of printing. Use logging.getLogger(__name__), "
        "or audio_utils.log_init / log_ready for startup messages. "
        "Offending calls: " + ", ".join(offenders)
    )
