# ci-failure-analyzer

An AI agent that diagnoses failed CI builds. For each failure it explains the root
cause and why the build probably passed on the developer's machine, suggests a
minimal fix, and posts one deduplicated alert.

It is a team-level CI triage layer, not a replacement for per-run fix buttons such as
GitHub's "Fix with Copilot" or similar: it runs automatically on every failure, works
across CI systems, and hands fixable failures to a coding agent (Copilot, Claude,
Gemini, …).

## Status

Phase 1 (single agent, local) is in progress. The deterministic parts are built and
tested against real failure logs from
[`petclinic-ci-failures`](https://github.com/pathmanabanptech/petclinic-ci-failures):
log preprocessing, secret redaction, evidence verification, alert gating, budgets
and output sanitization. The LLM agent itself comes next.

Planned: multi-agent (Phase 2), live on GitHub Actions (Phase 3), deployed on AWS
Bedrock AgentCore (Phase 4).

## Guardrails

- The agent is read-only: it suggests diffs, never pushes code.
- Logs, files and commit messages are treated as untrusted input (prompt-injection defense).
- Secrets are redacted before anything reaches the model, memory or a notification.
- Every cited piece of evidence is checked by code; unverified evidence is dropped and confidence is capped.
- Alerts and the fix hand-off are gated by code, not by the model.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
make install lint test
uv run python cli.py preprocess --case case-01 --log maven.log   # excerpt around the first fatal error
```

Reading files from the test-bed repo needs a local clone with all branches fetched:

```bash
export PETCLINIC_REPO_PATH=~/path/to/petclinic-ci-failures
```

## Layout

| Path | Contents |
|---|---|
| `agent/` | Output schema (`Diagnosis`), later the agent itself |
| `guardrails/` | Preprocessing, redaction, evidence verifier, gating, budgets, sanitizer |
| `adapters/` | I/O interfaces (logs, repo, notifier, memory) and local implementations |
| `evals/` | Eval cases (raw logs + metadata + answer keys), loader, capture tool |
| `cli.py` | Command-line entry point |
