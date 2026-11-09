"""Command-line entry point.

uv run python cli.py preprocess --case case-01 --log maven.log
"""

from __future__ import annotations

import argparse
import sys

from evals.loader import CASES_DIR, CaseError, load_agent_input
from guardrails.preprocess import preprocess


def _cmd_preprocess(args: argparse.Namespace) -> int:
    try:
        inp = load_agent_input(args.case, args.log)
    except CaseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    raw = (CASES_DIR / inp.case_id / inp.log).read_text(encoding="utf-8", errors="replace")
    excerpt = preprocess(raw)

    first = excerpt.first_error
    summary = (
        f"first fatal error: {first.kind} at line {first.line.number}"
        if first
        else "first fatal error: none found"
    )
    term = excerpt.termination
    if term:
        summary += f"; job {term.kind} at line {term.line.number}"
    print(
        f"# {inp.case_id}/{inp.log}: {excerpt.total_lines} raw lines, "
        f"{len(excerpt.lines)} kept; {summary}"
    )
    print(excerpt.render())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cli.py", description="CI failure analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="print the G8 excerpt of a case log")
    p.add_argument("--case", required=True, help="case id, e.g. case-01")
    p.add_argument("--log", required=True, help="log file name, e.g. maven.log")
    p.set_defaults(func=_cmd_preprocess)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
