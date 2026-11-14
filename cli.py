"""Command-line entry point.

uv run python cli.py preprocess --case case-01 --log maven.log
uv run python cli.py analyze --case case-01 [--log maven.log] --dry-run

``analyze`` prints the Diagnosis JSON on stdout; spans and the notification go to
stderr, so stdout can be piped to ``jq``.
"""

from __future__ import annotations

import argparse
import os
import sys

import telemetry
from adapters.base import AdapterError, NotFoundError, RepoReader
from adapters.logs.local import LocalLogSource
from adapters.notify.console import ConsoleNotifier
from adapters.repo.local import ENV_VAR, LocalRepoReader
from agent.stub import StubAgent
from evals.loader import CASES_DIR, CaseError, case_logs, load_agent_input
from guardrails.preprocess import preprocess
from pipeline import analyze


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


def _repo_reader() -> RepoReader | None:
    if not os.environ.get(ENV_VAR):
        print(f"note: {ENV_VAR} not set; file evidence cannot be verified", file=sys.stderr)
        return None
    try:
        return LocalRepoReader.from_env()
    except AdapterError as e:
        print(f"note: {e}; file evidence cannot be verified", file=sys.stderr)
        return None


def _cmd_analyze(args: argparse.Namespace) -> int:
    if not args.dry_run:
        print("error: the LLM agent is not built yet; use --dry-run", file=sys.stderr)
        return 2
    try:
        logs = case_logs(args.case)
        log = args.log or (logs[0] if logs else None)
        if log is None:
            raise CaseError(f"{args.case} has no failure logs")
        load_agent_input(args.case, log)  # validates case + log, agent-side view only
        source = LocalLogSource(CASES_DIR)
        ref = source.ref_for(args.case, log)
    except (CaseError, NotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    others = [name for name in logs if name != log]
    print(
        f"# {args.case}/{log} (dry run: stub agent, no model call)"
        + (f"; other logs: {', '.join(others)}" if others else ""),
        file=sys.stderr,
    )
    provider = telemetry.configure(telemetry.console_exporter(fmt=args.spans))
    try:
        result = analyze(
            ref,
            logs=source,
            repo=_repo_reader(),
            notifier=ConsoleNotifier(stream=sys.stderr),
            agent=StubAgent(),
            tracer=telemetry.get_tracer(provider),
        )
    finally:
        provider.shutdown()
    print(result.diagnosis.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cli.py", description="CI failure analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="print the preprocessed excerpt of a case log")
    p.add_argument("--case", required=True, help="case id, e.g. case-01")
    p.add_argument("--log", required=True, help="log file name, e.g. maven.log")
    p.set_defaults(func=_cmd_preprocess)

    a = sub.add_parser("analyze", help="run the full pipeline on a case log")
    a.add_argument("--case", required=True, help="case id, e.g. case-01")
    a.add_argument("--log", help="log file name (default: the case's first failure log)")
    a.add_argument("--dry-run", action="store_true", help="use the stub agent; no model is called")
    a.add_argument(
        "--spans", choices=["compact", "json"], default="compact", help="span output format"
    )
    a.set_defaults(func=_cmd_analyze)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
