"""Command-line entry point.

uv run python cli.py preprocess --case case-01 --log maven.log
uv run python cli.py analyze --case case-01 [--log maven.log] [--dry-run] [--debug]
uv run python cli.py ping-model [--role triage|specialist]

Settings come from the environment; a ``.env`` file next to this script is loaded
first (variables already set in the shell take precedence).

Results go to stdout as JSON (``analyze``: one Diagnosis per log, keyed by log
name); spans, notes, debug output and the notification go to stderr, so stdout can
be piped to ``jq``.

``analyze`` is fail-safe: if the model is down or times out, a budget runs out, or
analysis crashes, that log gets ``{"analysis_unavailable": "<reason>"}`` instead of
a diagnosis, nothing is posted for it, nothing is retried, and the exit code stays
0, so the analyzer never turns a CI run red.

Exit codes: 0 ok (including "analysis unavailable"), 1 misconfigured (e.g. no
model settings), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections.abc import MutableMapping
from pathlib import Path

import telemetry
from adapters.base import AdapterError, NotFoundError, RepoReader
from adapters.logs.local import LocalLogSource
from adapters.notify.console import ConsoleNotifier
from adapters.repo.local import ENV_VAR, LocalRepoReader
from agent.base import DiagnosisAgent
from agent.errors import ModelError, ModelUnavailableError
from agent.stub import StubAgent
from evals.loader import CASES_DIR, CaseError, load_agent_input, load_meta
from guardrails.budget import BudgetExceeded, BudgetLimits
from guardrails.preprocess import preprocess
from pipeline import PipelineResult, analyze


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


def _make_agent(args: argparse.Namespace, repo: RepoReader | None) -> DiagnosisAgent:
    if args.dry_run:
        return StubAgent()
    # Imported here so dry runs never load a model SDK.
    from agent.generalist import GeneralistAgent
    from agent.models import ModelSettings, build_model

    return GeneralistAgent(build_model(ModelSettings.from_env(), "specialist"), repo)


ANALYZE_CONCLUSIONS = frozenset({"failure", "cancelled"})


def _logs_to_analyze(case: str, log: str | None) -> list[str]:
    """``--log``, or every log of a run whose conclusion is failure/cancelled."""
    meta = load_meta(case)
    if log is not None:
        load_agent_input(case, log)  # validates the log name; agent-side view only
        return [log]
    logs = [r.log for r in meta.runs.values() if r.conclusion in ANALYZE_CONCLUSIONS and r.log]
    if not logs:
        raise CaseError(f"{case} has no failed or cancelled runs with a log")
    return logs


def _print_debug(log: str, result: PipelineResult) -> None:
    """Raw model output and every guardrail verdict, on stderr. Model text is shown
    unsanitized (JSON/repr-escaped, so no control characters reach the terminal)."""
    out = [f"--- debug: {log} ---", "raw model output (before guardrails):"]
    out.append(result.agent_output.raw_output or "(none)")
    if result.agent_output.structured_output_fallback:
        out.append("(the answer came from the constrained structured-output fallback)")
    calls = result.agent_output.tool_log
    out.append(f"tool calls ({len(calls)}):")
    for i, call in enumerate(calls, 1):
        args = json.dumps(dict(call.args), ensure_ascii=True)
        args = args if len(args) <= 200 else args[:200] + "…"
        out.append(f"  {i}. {call.name} {args} -> {call.result_chars} chars ({call.status})")
    if not calls:
        out.append("  (none)")
    out.append("evidence:")
    for c in result.evidence_checks:
        ev = c.evidence
        verdict = "accepted" if c.ok else "rejected"
        hint = f"line hint {ev.line}" if ev.line is not None else "no line hint"
        located = f", located_line {ev.located_line}" if ev.located_line is not None else ""
        offset = f", offset {c.offset:+d}" if c.offset else ""
        out.append(f"  [{verdict}] {ev.where()!r} ({hint}{located}{offset}): {c.reason}")
        out.append(f"      snippet: {ev.snippet!r}")
    if not result.evidence_checks:
        out.append("  (none cited)")
    fix = result.fix_check
    if fix is None:
        out.append("fix_diff: none proposed")
    else:
        out.append(f"fix_diff: {'accepted' if fix.ok else 'rejected (set to null)'}: {fix.reason}")
    req = result.requirements
    out.append(
        "category evidence requirements: "
        + ("met" if req.met else f"NOT met, missing {'; '.join(req.missing)}")
    )
    d = result.diagnosis
    out.append(
        f"after guardrails: {d.status.value} / {d.category.value} / confidence "
        f"{d.confidence:g}; alert: {result.decision.level.value}"
    )
    print("\n".join(out), file=sys.stderr)


def _cmd_analyze(args: argparse.Namespace) -> int:
    try:
        logs = _logs_to_analyze(args.case, args.log)
        source = LocalLogSource(CASES_DIR)
        refs = {log: source.ref_for(args.case, log) for log in logs}
    except (CaseError, NotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    repo = _repo_reader()
    try:
        agent = _make_agent(args, repo)
        limits = BudgetLimits.from_env()
    except ModelUnavailableError as e:  # model server down at startup: fail safe
        print(f"analysis unavailable: {e}", file=sys.stderr)
        print(json.dumps({log: {"analysis_unavailable": str(e)} for log in refs}, indent=2))
        return 0
    except (ModelError, ValueError) as e:  # misconfiguration: the operator must fix it
        print(f"error: {e}", file=sys.stderr)
        return 1
    mode = "dry run: stub agent, no model call" if args.dry_run else f"agent: {agent.name}"
    provider = telemetry.configure(telemetry.console_exporter(fmt=args.spans))
    tracer = telemetry.get_tracer(provider)
    diagnoses: dict[str, object] = {}
    try:
        for log, ref in refs.items():
            print(f"# {args.case}/{log} ({mode})", file=sys.stderr)
            try:
                result = analyze(
                    ref,
                    logs=source,
                    repo=repo,
                    notifier=ConsoleNotifier(stream=sys.stderr),
                    agent=agent,
                    tracer=tracer,
                    limits=limits,
                    log_name=log,
                )
            except (BudgetExceeded, ModelError) as e:
                reason = f"{type(e).__name__}: {e}"
                print(f"analysis unavailable for {log}: {reason}", file=sys.stderr)
                diagnoses[log] = {"analysis_unavailable": reason}
                continue
            except Exception as e:  # a crash must not fail CI either; report it loudly
                reason = f"internal error: {type(e).__name__}: {e}"
                print(f"analysis unavailable for {log}: {reason}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                diagnoses[log] = {"analysis_unavailable": reason}
                continue
            if args.debug:
                _print_debug(log, result)
            diagnoses[log] = result.diagnosis.model_dump(mode="json")
    finally:
        provider.shutdown()
    print(json.dumps(diagnoses, indent=2))
    return 0


def _cmd_ping_model(args: argparse.Namespace) -> int:
    # Imported here so commands that make no model call never load a model SDK.
    from agent.models import ModelError, ModelSettings, build_model, ping

    provider = telemetry.configure(telemetry.console_exporter(fmt=args.spans))
    tracer = telemetry.get_tracer(provider)
    try:
        with tracer.start_as_current_span("model.ping") as span:
            try:
                settings = ModelSettings.from_env()
                span.set_attributes({"provider": settings.provider, "role": args.role})
                handle = build_model(settings, args.role)
                span.set_attribute("model_id", handle.pinned_id)
                result = ping(handle)
            except ModelError as e:
                span.set_attribute("error", type(e).__name__)
                print(f"error: {e}", file=sys.stderr)
                return 1
            cost = "unknown" if result.cost_usd is None else result.cost_usd
            span.set_attributes(
                {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_s": round(result.latency_s, 3),
                    "cost_usd": cost,
                }
            )
    finally:
        provider.shutdown()

    print(
        json.dumps(
            {
                "provider": handle.provider,
                "role": handle.role,
                "model_id": handle.model_id,
                "pinned_id": handle.pinned_id,
                "reply": result.reply,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "latency_s": round(result.latency_s, 3),
            },
            indent=2,
        )
    )
    if result.cost_usd is None:
        print(
            "note: cost unknown; set MODEL_PRICE_INPUT_PER_MTOK and MODEL_PRICE_OUTPUT_PER_MTOK",
            file=sys.stderr,
        )
    return 0


ENV_FILE = Path(__file__).resolve().parent / ".env"
_ENV_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def load_env_file(path: Path, environ: MutableMapping[str, str]) -> list[str]:
    """Load ``KEY=VALUE`` lines into ``environ`` without overriding variables that are
    already set (the shell wins). Blank lines and ``#`` comments are ignored; values
    may be quoted. Nothing is executed. Returns the names that were set."""
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if key not in environ:
            environ[key] = value
            loaded.append(key)
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cli.py", description="CI failure analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="print the preprocessed excerpt of a case log")
    p.add_argument("--case", required=True, help="case id, e.g. case-01")
    p.add_argument("--log", required=True, help="log file name, e.g. maven.log")
    p.set_defaults(func=_cmd_preprocess)

    a = sub.add_parser("analyze", help="run the full pipeline on a case log")
    a.add_argument("--case", required=True, help="case id, e.g. case-01")
    a.add_argument(
        "--log", help="analyze only this log (default: every failed or cancelled run's log)"
    )
    a.add_argument("--dry-run", action="store_true", help="use the stub agent; no model is called")
    a.add_argument(
        "--debug",
        action="store_true",
        help="print raw model output, tool calls, and each evidence/fix/requirement verdict "
        "with its reason (stderr)",
    )
    a.add_argument(
        "--spans", choices=["compact", "json"], default="compact", help="span output format"
    )
    a.set_defaults(func=_cmd_analyze)

    m = sub.add_parser("ping-model", help="one minimal call to the configured model")
    m.add_argument(
        "--role", choices=["triage", "specialist"], default="triage", help="which model id"
    )
    m.add_argument(
        "--spans", choices=["compact", "json"], default="compact", help="span output format"
    )
    m.set_defaults(func=_cmd_ping_model)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    load_env_file(ENV_FILE, os.environ)  # script runs only: tests never read a local .env
    sys.exit(main())
