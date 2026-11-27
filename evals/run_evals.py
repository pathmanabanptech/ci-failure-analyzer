"""Eval runner: deterministic metrics over every case x log x N runs.

    uv run python -m evals.run_evals                 # generalist agent from the environment
    uv run python -m evals.run_evals --dry-run --n 1 # stub agent, no model call
    uv run python -m evals.run_evals --case case-01 --n 5

Each run goes through the real pipeline (preprocess, redact, agent, verify, gate,
sanitize; notifications are collected, not printed) and is scored against the
case's answer key:

* category and status (status after the guardrails, i.e. what would be posted)
* evidence: cited, verified, hallucinated (cited but failed verification), whether
  at least one item verified, and which expected evidence sources were covered
* confidence inside the answer key's ``expected_confidence`` range, when it has one
* whether a proposed fix applies to what the build compiled
* structured-output fallback, tool calls, model calls, tokens, cost, latency
* budget or model errors, recorded per run instead of stopping the eval
* guardrail cases: answer-key ``must_not_appear`` text (secrets, injected
  instructions or links) searched for in the posted diagnosis, and
  ``must_not_reach_model`` text (secrets) in everything the agent received
* with ``--judge``: an LLM judge's root cause (0-3) and local_vs_ci (0-2) scores
  against the answer key's key facts (see ``evals/judge``)

The table goes to stdout; the full results (every run, per-log aggregates, the
summary, and what produced them: git SHA, prompt hashes, model IDs, budget
limits, N) go to a JSON file. A real run writes ``evals/results/phase1.json``; a
dry run writes ``evals/results/dry-run.json`` so it never replaces a baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import telemetry
from adapters.base import AdapterError, RepoReader, Report
from adapters.logs.local import LocalLogSource
from adapters.repo.local import ENV_VAR, LocalRepoReader
from agent.base import AgentRequest, AgentResult, DiagnosisAgent, run_context
from agent.errors import ModelError
from agent.schema import Evidence
from agent.stub import StubAgent
from evals.judge.judge import Judge, judge_from_env, rubric_hash
from evals.loader import CASES_DIR, AnswerKey, case_logs, list_cases, load_answer_key
from guardrails.budget import Budget, BudgetExceeded, BudgetLimits
from pipeline import PipelineResult, analyze

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"
PROMPTS_DIR = REPO_ROOT / "prompts"
DEFAULT_N = 3
CONTEXT_SOURCES = frozenset({"log", "commit", "run_attempts"})


# ------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class RunScore:
    case: str
    log: str
    run: int  # 1-based run index for this case x log
    error: str | None = None  # budget or model failure: no diagnosis was produced
    category: str | None = None
    category_ok: bool | None = None
    status: str | None = None  # after the guardrails
    status_ok: bool | None = None
    agent_status: str | None = None  # as the agent returned it, before the guardrails
    confidence: float | None = None  # after the guardrails
    agent_confidence: float | None = None
    confidence_in_range: bool | None = None  # None: the answer key has no range
    evidence_cited: int = 0
    evidence_verified: int = 0
    sources_expected: tuple[str, ...] = ()
    sources_covered: tuple[str, ...] = ()
    fix_proposed: bool = False
    fix_applies: bool = False
    alert_level: str | None = None
    structured_output_fallback: bool = False
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None: price not configured
    latency_s: float = 0.0
    model_id: str | None = None
    # Final (sanitized) prose, kept so the judge can be re-checked against hand labels.
    root_cause: str | None = None
    local_vs_ci: str | None = None
    judge_root_cause: int | None = None  # 0-3
    judge_local_vs_ci: int | None = None  # 0-2
    judge_root_cause_reason: str | None = None
    judge_local_vs_ci_reason: str | None = None
    judge_error: str | None = None
    judge_cost_usd: float | None = None
    # Guardrail cases; None: the answer key has nothing to check on that side.
    leaked_to_model: tuple[str, ...] | None = None  # must_not_reach_model found in input
    leaked_to_output: tuple[str, ...] | None = None  # must_not_appear found in output

    @property
    def evidence_hallucinated(self) -> int:
        return self.evidence_cited - self.evidence_verified

    @property
    def has_verified_evidence(self) -> bool:
        return self.evidence_verified > 0

    @property
    def guardrail_ok(self) -> bool | None:
        checked = [x for x in (self.leaked_to_model, self.leaked_to_output) if x is not None]
        return None if not checked else not any(checked)


class InputRecorder:
    """Wraps the agent and keeps the last request, so the eval can check what the
    model could see. Everything an agent's tools return is drawn from the same
    redacted log lines and redacted file reads."""

    def __init__(self, agent: DiagnosisAgent) -> None:
        self.inner = agent
        self.name = agent.name
        self.last: AgentRequest | None = None

    def diagnose(self, request: AgentRequest, budget: Budget) -> AgentResult:
        self.last = request
        return self.inner.diagnose(request, budget)


def model_input_text(request: AgentRequest) -> str:
    parts = [
        request.excerpt.render(),
        *(line.text for line in request.log_lines),
        json.dumps(run_context(request.run)),
    ]
    return "\n".join(parts)


def leaks(needles: Sequence[str], text: str) -> tuple[str, ...]:
    low = text.lower()
    return tuple(n for n in needles if n.lower() in low)


def _source_label(ev: Evidence) -> str:
    return ev.path or "?" if ev.kind == "file" else ev.kind


def _covers(expected: str, labels: Iterable[str]) -> bool:
    """``log``/``commit``/``run_attempts`` match by kind; a file matches by full path,
    path suffix or file name (answer keys write ``pom.xml`` or the full path)."""
    for label in labels:
        if expected in CONTEXT_SOURCES or label in CONTEXT_SOURCES:
            if label == expected:
                return True
        elif label == expected or label.endswith("/" + expected) or expected.endswith("/" + label):
            return True
    return False


def score_run(
    key: AnswerKey,
    run: int,
    result: PipelineResult,
    latency_s: float,
    request: AgentRequest | None = None,
) -> RunScore:
    d = result.diagnosis
    agent = result.agent_output.diagnosis
    verified = [c.evidence for c in result.evidence_checks if c.ok]
    labels = [_source_label(ev) for ev in verified]
    in_range = None
    if key.expected_confidence is not None:
        lo, hi = key.expected_confidence
        in_range = lo <= d.confidence <= hi
    usage = result.usage
    leaked_model = leaked_output = None
    if key.must_not_reach_model:
        if request is None:
            raise ValueError("must_not_reach_model needs the agent's request")
        leaked_model = leaks(key.must_not_reach_model, model_input_text(request))
    if key.must_not_appear:
        leaked_output = leaks(key.must_not_appear, d.model_dump_json())
    return RunScore(
        case=key.case_id,
        log=key.log,
        run=run,
        category=d.category.value,
        category_ok=d.category is key.category,
        status=d.status.value,
        status_ok=d.status is key.status,
        agent_status=agent.status.value,
        confidence=d.confidence,
        agent_confidence=agent.confidence,
        confidence_in_range=in_range,
        evidence_cited=len(result.evidence_checks),
        evidence_verified=len(verified),
        sources_expected=tuple(key.evidence_sources),
        sources_covered=tuple(s for s in key.evidence_sources if _covers(s, labels)),
        fix_proposed=result.fix_check is not None,
        fix_applies=bool(result.fix_check and result.fix_check.ok),
        alert_level=result.decision.level.value,
        structured_output_fallback=result.agent_output.structured_output_fallback,
        tool_calls=result.agent_output.tool_calls,
        llm_calls=usage.llm_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        latency_s=round(latency_s, 3),
        model_id=usage.model_id,
        root_cause=d.root_cause,
        local_vs_ci=d.local_vs_ci,
        leaked_to_model=leaked_model,
        leaked_to_output=leaked_output,
    )


# --------------------------------------------------------------- aggregation


@dataclass(frozen=True)
class Stat:
    mean: float
    std: float  # population std over the runs (0 for one run)

    def fmt(self, digits: int = 2) -> str:
        if round(self.std, digits) == 0:
            return f"{self.mean:.{digits}f}"
        return f"{self.mean:.{digits}f}±{self.std:.{digits}f}"


def _stat(values: Sequence[float]) -> Stat | None:
    if not values:
        return None
    return Stat(mean=statistics.fmean(values), std=statistics.pstdev(values))


def _rate(flags: Iterable[bool | None]) -> float | None:
    known = [f for f in flags if f is not None]
    return sum(known) / len(known) if known else None


@dataclass(frozen=True)
class Aggregate:
    """Over a group of runs (one case x log, or everything)."""

    runs: int
    errors: int
    category_accuracy: float | None
    status_accuracy: float | None
    runs_with_verified_evidence: float | None
    evidence_cited: int
    evidence_verified: int
    verified_evidence_rate: float | None  # verified / cited
    hallucinated_evidence_rate: float | None  # (cited - verified) / cited
    sources_coverage: float | None  # expected sources covered / expected
    confidence_in_range: float | None  # None: no answer key in the group has a range
    confidence: Stat | None
    fix_applies_rate: float | None  # of runs that proposed a fix
    structured_output_fallback_rate: float | None
    tool_calls: Stat | None
    input_tokens: Stat | None
    output_tokens: Stat | None
    latency_s: Stat | None
    cost_usd_total: float | None  # None: some run had no known price
    judge_root_cause: Stat | None = None  # 0-3, over judged runs
    judge_local_vs_ci: Stat | None = None  # 0-2
    judge_errors: int = 0
    guardrail_pass: float | None = None  # None: no guardrail case in the group
    group: dict[str, str] = field(default_factory=dict)


def aggregate(scores: Sequence[RunScore], group: dict[str, str] | None = None) -> Aggregate:
    ok = [s for s in scores if s.error is None]
    cited = sum(s.evidence_cited for s in ok)
    verified = sum(s.evidence_verified for s in ok)
    expected = sum(len(s.sources_expected) for s in ok)
    covered = sum(len(s.sources_covered) for s in ok)
    costs = [s.cost_usd for s in ok]
    with_fix = [s for s in ok if s.fix_proposed]
    return Aggregate(
        runs=len(scores),
        errors=len(scores) - len(ok),
        category_accuracy=_rate(s.category_ok for s in ok),
        status_accuracy=_rate(s.status_ok for s in ok),
        runs_with_verified_evidence=_rate(s.has_verified_evidence for s in ok),
        evidence_cited=cited,
        evidence_verified=verified,
        verified_evidence_rate=verified / cited if cited else None,
        hallucinated_evidence_rate=(cited - verified) / cited if cited else None,
        sources_coverage=covered / expected if expected else None,
        confidence_in_range=_rate(s.confidence_in_range for s in ok),
        confidence=_stat([s.confidence for s in ok if s.confidence is not None]),
        fix_applies_rate=_rate(s.fix_applies for s in with_fix),
        structured_output_fallback_rate=_rate(s.structured_output_fallback for s in ok),
        tool_calls=_stat([s.tool_calls for s in ok]),
        input_tokens=_stat([s.input_tokens for s in ok]),
        output_tokens=_stat([s.output_tokens for s in ok]),
        latency_s=_stat([s.latency_s for s in ok]),
        cost_usd_total=None
        if any(c is None for c in costs)
        else round(sum(c or 0.0 for c in costs), 6),
        judge_root_cause=_stat([s.judge_root_cause for s in ok if s.judge_root_cause is not None]),
        judge_local_vs_ci=_stat(
            [s.judge_local_vs_ci for s in ok if s.judge_local_vs_ci is not None]
        ),
        judge_errors=sum(1 for s in ok if s.judge_error),
        guardrail_pass=_rate(s.guardrail_ok for s in ok),
        group=group or {},
    )


def per_log(scores: Sequence[RunScore]) -> list[Aggregate]:
    keys: dict[tuple[str, str], list[RunScore]] = {}
    for s in scores:
        keys.setdefault((s.case, s.log), []).append(s)
    return [aggregate(group, {"case": c, "log": log}) for (c, log), group in keys.items()]


# -------------------------------------------------------------------- table


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def _num(s: Stat | None, digits: int = 1) -> str:
    return "-" if s is None else s.fmt(digits)


def _cost(x: float | None) -> str:
    return "unknown" if x is None else f"${x:.4f}"


COLUMNS = (
    ("case/log", lambda a: f"{a.group.get('case', 'ALL')}/{a.group.get('log', '')}".rstrip("/")),
    ("runs", lambda a: f"{a.runs}" + (f" ({a.errors} err)" if a.errors else "")),
    ("category", lambda a: _pct(a.category_accuracy)),
    ("status", lambda a: _pct(a.status_accuracy)),
    ("≥1 verified", lambda a: _pct(a.runs_with_verified_evidence)),
    ("cited/verified", lambda a: f"{a.evidence_cited}/{a.evidence_verified}"),
    ("halluc.", lambda a: _pct(a.hallucinated_evidence_rate)),
    ("sources", lambda a: _pct(a.sources_coverage)),
    ("conf", lambda a: _num(a.confidence, 2)),
    ("conf range", lambda a: _pct(a.confidence_in_range)),
    ("guardrail", lambda a: _pct(a.guardrail_pass)),
    ("root cause /3", lambda a: _num(a.judge_root_cause, 1)),
    ("local vs CI /2", lambda a: _num(a.judge_local_vs_ci, 1)),
    ("fix applies", lambda a: _pct(a.fix_applies_rate)),
    ("fallback", lambda a: _pct(a.structured_output_fallback_rate)),
    ("in tok", lambda a: _num(a.input_tokens, 0)),
    ("latency s", lambda a: _num(a.latency_s, 1)),
    ("cost", lambda a: _cost(a.cost_usd_total)),
)


def render_table(rows: Sequence[Aggregate], total: Aggregate) -> str:
    cells = [[name for name, _ in COLUMNS]]
    cells += [[fn(a) for _, fn in COLUMNS] for a in rows]
    cells.append([fn(total) for _, fn in COLUMNS])
    widths = [max(len(r[i]) for r in cells) for i in range(len(COLUMNS))]

    def line(row: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(cells[0]), rule, *map(line, cells[1:-1]), rule, line(cells[-1])])


# --------------------------------------------------------------- provenance


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def prompt_hashes(prompts_dir: Path = PROMPTS_DIR) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(prompts_dir.glob("*.md"))
    }


def provenance(
    *,
    agent: str,
    n: int,
    limits: BudgetLimits,
    repo_reader: bool,
    scores: Sequence[RunScore],
    judge: str | None = None,
) -> dict[str, Any]:
    status = _git("status", "--porcelain", "--", ".", ":(exclude)evals/results")
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "agent": agent,
        "model_ids": sorted({s.model_id for s in scores if s.model_id}),
        "prompt_hashes": prompt_hashes(),
        "n": n,
        "budget_limits": asdict(limits),
        "repo_reader": repo_reader,
        "judge": judge,
        "judge_rubric_hash": rubric_hash() if judge else None,
        "holdout": "no holdout yet",
    }


# ------------------------------------------------------------------- runner


class CollectingNotifier:
    """Keeps reports instead of posting them (an eval must not alert anyone)."""

    def __init__(self) -> None:
        self.reports: list[Report] = []

    def notify(self, report: Report) -> None:
        self.reports.append(report)


def targets(cases: Sequence[str] | None, cases_dir: Path = CASES_DIR) -> list[tuple[str, str]]:
    chosen = list(cases) if cases else list_cases(cases_dir)
    return [(c, log) for c in chosen for log in case_logs(c, cases_dir)]


def run_eval(
    pairs: Sequence[tuple[str, str]],
    *,
    agent: DiagnosisAgent,
    repo: RepoReader | None,
    n: int,
    limits: BudgetLimits,
    cases_dir: Path = CASES_DIR,
    progress: Any = None,
    judge: Judge | None = None,
) -> list[RunScore]:
    source = LocalLogSource(cases_dir)
    tracer = telemetry.get_tracer(telemetry.make_provider(None))  # spans dropped
    recorder = InputRecorder(agent)
    scores: list[RunScore] = []
    for case, log in pairs:
        key = load_answer_key(case, log, cases_dir)
        ref = source.ref_for(case, log)
        for i in range(1, n + 1):
            start = time.perf_counter()
            try:
                result = analyze(
                    ref,
                    logs=source,
                    repo=repo,
                    notifier=CollectingNotifier(),
                    agent=recorder,
                    tracer=tracer,
                    limits=limits,
                    log_name=log,
                )
            except (BudgetExceeded, ModelError) as e:
                score = RunScore(
                    case=case,
                    log=log,
                    run=i,
                    error=f"{type(e).__name__}: {e}",
                    latency_s=round(time.perf_counter() - start, 3),
                )
            else:
                score = score_run(key, i, result, time.perf_counter() - start, recorder.last)
                if judge is not None:
                    score = _judged(score, judge, key, result)
            scores.append(score)
            if progress is not None:
                verdict = score.error or (
                    f"{score.category} {score.status} conf={score.confidence:g} "
                    f"verified={score.evidence_verified}/{score.evidence_cited}"
                )
                print(
                    f"{case}/{log} run {i}/{n}: {verdict} ({score.latency_s:.1f}s)",
                    file=progress,
                    flush=True,
                )
    return scores


def _judged(score: RunScore, judge: Judge, key: AnswerKey, result: PipelineResult) -> RunScore:
    try:
        j = judge.score(key, result.diagnosis)
    except ModelError as e:
        return replace(score, judge_error=f"{type(e).__name__}: {e}")
    return replace(
        score,
        judge_root_cause=j.root_cause,
        judge_local_vs_ci=j.local_vs_ci,
        judge_root_cause_reason=j.root_cause_reason,
        judge_local_vs_ci_reason=j.local_vs_ci_reason,
        judge_cost_usd=j.cost_usd,
    )


def results_json(
    meta: dict[str, Any], scores: Sequence[RunScore], rows: Sequence[Aggregate], total: Aggregate
) -> dict[str, Any]:
    def run_dict(s: RunScore) -> dict[str, Any]:
        return {
            **asdict(s),
            "evidence_hallucinated": s.evidence_hallucinated,
            "has_verified_evidence": s.has_verified_evidence,
        }

    return {
        "meta": meta,
        "summary": asdict(total),
        "per_log": [asdict(a) for a in rows],
        "runs": [run_dict(s) for s in scores],
    }


def _repo_reader() -> RepoReader | None:
    if not os.environ.get(ENV_VAR):
        return None
    try:
        return LocalRepoReader.from_env()
    except AdapterError:
        return None


def _make_agent(dry_run: bool, repo: RepoReader | None) -> DiagnosisAgent:
    if dry_run:
        return StubAgent()
    from agent.generalist import GeneralistAgent  # loads a model SDK: real runs only
    from agent.models import ModelSettings, build_model

    return GeneralistAgent(build_model(ModelSettings.from_env(), "specialist"), repo)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m evals.run_evals", description=__doc__.split("\n")[0]
    )
    p.add_argument("--case", action="append", help="limit to this case (repeatable)")
    p.add_argument(
        "--n", type=int, default=DEFAULT_N, help=f"runs per case x log (default {DEFAULT_N})"
    )
    p.add_argument("--dry-run", action="store_true", help="stub agent, no model call")
    p.add_argument("--out", type=Path, help="results JSON path")
    p.add_argument(
        "--judge",
        action="store_true",
        help="score root cause and local_vs_ci with the LLM judge (JUDGE_MODEL_* settings)",
    )
    args = p.parse_args(argv)
    if args.n < 1:
        p.error("--n must be at least 1")

    out = args.out or RESULTS_DIR / ("dry-run.json" if args.dry_run else "phase1.json")
    repo = _repo_reader()
    if repo is None:
        print(
            f"warning: {ENV_VAR} not set or unusable; file evidence cannot verify, "
            "so evidence metrics are pessimistic",
            file=sys.stderr,
        )
    try:
        pairs = targets(args.case)
        agent = _make_agent(args.dry_run, repo)
        limits = BudgetLimits.from_env()
        judge = judge_from_env(os.environ) if args.judge else None
    except (ModelError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    scores = run_eval(
        pairs, agent=agent, repo=repo, n=args.n, limits=limits, progress=sys.stderr, judge=judge
    )
    rows = per_log(scores)
    total = aggregate(scores)
    meta = provenance(
        agent=agent.name,
        n=args.n,
        limits=limits,
        repo_reader=repo is not None,
        scores=scores,
        judge=judge.name if judge else None,
    )
    if judge is not None and judge.name in meta["model_ids"]:
        print(
            f"warning: the judge ({judge.name}) is the model being evaluated; "
            "use a different JUDGE_MODEL_ID where possible",
            file=sys.stderr,
        )

    print(render_table(rows, total))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results_json(meta, scores, rows, total), indent=2) + "\n")
    print(
        f"\nwrote {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}"
        f" (agent {meta['agent']}, models {', '.join(meta['model_ids']) or 'none'}, "
        f"git {(meta['git_sha'] or '?')[:12]}{' dirty' if meta['git_dirty'] else ''})"
    )
    return 1 if total.errors else 0


if __name__ == "__main__":
    from cli import ENV_FILE, load_env_file

    load_env_file(ENV_FILE, os.environ)  # same settings as `cli.py analyze`
    sys.exit(main())
