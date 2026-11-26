"""Judge calibration: compare hand labels with the judge's scores.

Before trusting the judge, label a set of real diagnoses by hand and check that the
judge agrees with you on at least 80% of the scores.

1. Run the eval with the judge, so the results file holds diagnoses and judge scores:
       make eval ARGS="--judge"
2. Write a labels file to fill in. It shows each diagnosis and its key facts, but
   not the judge's scores, so your labels stay independent:
       uv run python -m evals.judge.agreement --results evals/results/phase1.json \\
           --template > evals/judge/labels.yaml
3. Replace every ``null`` with your score (root_cause 0-3, local_vs_ci 0-2, as in
   ``rubric.md``) and delete runs you don't want to label.
4. Measure agreement (exit 1 below the threshold):
       uv run python -m evals.judge.agreement --results evals/results/phase1.json \\
           --labels evals/judge/labels.yaml

Agreement is exact: a score agrees only if it equals your label. Within-one
agreement is reported too.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evals.loader import load_answer_key

THRESHOLD = 0.8
DIMENSIONS = ("root_cause", "local_vs_ci")
RunKey = tuple[str, str, int]


class AgreementError(ValueError):
    pass


@dataclass(frozen=True)
class Comparison:
    case: str
    log: str
    run: int
    dimension: str
    label: int
    judge: int

    @property
    def exact(self) -> bool:
        return self.label == self.judge

    @property
    def within_one(self) -> bool:
        return abs(self.label - self.judge) <= 1


@dataclass(frozen=True)
class AgreementReport:
    comparisons: tuple[Comparison, ...]
    unlabeled: int  # entries in the labels file still holding null

    def rate(self, dimension: str | None = None, *, within_one: bool = False) -> float | None:
        items = [c for c in self.comparisons if dimension in (None, c.dimension)]
        if not items:
            return None
        hits = sum(c.within_one if within_one else c.exact for c in items)
        return hits / len(items)

    @property
    def passed(self) -> bool:
        rate = self.rate()
        return rate is not None and rate >= THRESHOLD


def load_results(path: Path) -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise AgreementError(f"cannot read results {path}: {e}") from None
    if "runs" not in data:
        raise AgreementError(f"{path} is not an eval results file")
    return data


def _runs_by_key(results: Mapping[str, Any]) -> dict[RunKey, Mapping[str, Any]]:
    return {(r["case"], r["log"], int(r["run"])): r for r in results["runs"]}


def template(results: Mapping[str, Any], results_path: str) -> str:
    """A labels file to fill in by hand. Judge scores are deliberately left out."""
    meta = results.get("meta", {})
    out = [
        "# Hand labels for judge calibration. Score with evals/judge/rubric.md:",
        "#   root_cause 0-3, local_vs_ci 0-2. Replace every null; delete runs you skip.",
        "# The judge's scores are not shown here on purpose: label independently.",
        f"results: {results_path}",
        f"results_git_sha: {meta.get('git_sha')}",
        "labels:",
    ]
    for run in results["runs"]:
        if run.get("error") or run.get("root_cause") is None:
            continue
        key = load_answer_key(run["case"], run["log"])
        out.append(f"  # --- {run['case']}/{run['log']} run {run['run']} ---")
        out.append("  # key facts:")
        out += [f"  #   - {f}" for f in key.key_facts]
        for field in ("root_cause", "local_vs_ci"):
            text = " ".join(str(run[field]).split())
            out.append(f"  # {field}: {text}")
        out += [
            f"  - case: {run['case']}",
            f"    log: {run['log']}",
            f"    run: {run['run']}",
            "    root_cause: null",
            "    local_vs_ci: null",
            '    note: ""',
        ]
    return "\n".join(out) + "\n"


def _label_value(entry: Mapping[str, Any], dim: str, hi: int) -> int | None:
    value = entry.get(dim)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= hi:
        raise AgreementError(f"{dim} must be an integer 0-{hi}, got {value!r} in {dict(entry)}")
    return value


def compare(results: Mapping[str, Any], labels: Mapping[str, Any]) -> AgreementReport:
    runs = _runs_by_key(results)
    comparisons: list[Comparison] = []
    unlabeled = 0
    for entry in labels.get("labels") or []:
        key: RunKey = (entry["case"], entry["log"], int(entry["run"]))
        run = runs.get(key)
        if run is None:
            raise AgreementError(f"labelled run {key} is not in the results file")
        for dim, hi in (("root_cause", 3), ("local_vs_ci", 2)):
            label = _label_value(entry, dim, hi)
            if label is None:
                unlabeled += 1
                continue
            judge = run.get(f"judge_{dim}")
            if judge is None:
                raise AgreementError(
                    f"run {key} has no judge score; run the eval with --judge first"
                )
            comparisons.append(Comparison(*key, dimension=dim, label=label, judge=int(judge)))
    return AgreementReport(comparisons=tuple(comparisons), unlabeled=unlabeled)


def render(report: AgreementReport) -> str:
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{x * 100:.0f}%"

    lines = [f"{'dimension':<12}  {'items':>5}  {'exact':>6}  {'within 1':>8}"]
    for dim in (*DIMENSIONS, None):
        n = sum(1 for c in report.comparisons if dim in (None, c.dimension))
        name = dim or "ALL"
        lines.append(
            f"{name:<12}  {n:>5}  {pct(report.rate(dim)):>6}  "
            f"{pct(report.rate(dim, within_one=True)):>8}"
        )
    disagreements = [c for c in report.comparisons if not c.exact]
    if disagreements:
        lines.append("\ndisagreements (label vs judge):")
        lines += [
            f"  {c.case}/{c.log} run {c.run} {c.dimension}: {c.label} vs {c.judge}"
            for c in disagreements
        ]
    if report.unlabeled:
        lines.append(f"\n{report.unlabeled} score(s) still null in the labels file (ignored)")
    verdict = "PASS" if report.passed else "FAIL"
    lines.append(f"\n{verdict}: exact agreement {pct(report.rate())} (threshold {THRESHOLD:.0%})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m evals.judge.agreement", description="Compare hand labels with judge scores."
    )
    p.add_argument("--results", type=Path, required=True, help="eval results JSON")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--template", action="store_true", help="print a labels file to fill in")
    group.add_argument("--labels", type=Path, help="filled-in labels YAML")
    args = p.parse_args(argv)
    try:
        results = load_results(args.results)
        if args.template:
            print(template(results, str(args.results)), end="")
            return 0
        labels = yaml.safe_load(args.labels.read_text()) or {}
        report = compare(results, labels)
    except (AgreementError, OSError, yaml.YAMLError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not report.comparisons:
        print("error: no labelled scores to compare", file=sys.stderr)
        return 2
    print(render(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
