# Try it, one stage at a time

A failed CI log passes through these stages, in this order:

```
raw job log + meta.yaml
  → preprocess (first fatal error, original line numbers)
  → redact secrets
  → [agent: LLM diagnosis, not built yet]
  → verify evidence → gate (alert level, auto-fix) → sanitize → notify
```

Each section below runs one stage on the real eval cases, shows what you should
see, and suggests one change to try. Run everything from the repo root:

```bash
make install
export PETCLINIC_REPO_PATH=~/path/to/petclinic-ci-failures   # clone with `git fetch --all`
```

Each stage has a test file; `uv run pytest <file> -v` lists every behaviour it
checks by name.

The cases in `evals/cases/`:

| Case | Failure | Category |
|---|---|---|
| case-01 | A PR calls `getPet(String, boolean)`, which was renamed on the base branch; only the merge build fails | compile |
| case-02 | Guava added to `pom.xml` but not `build.gradle`; Maven passes, Gradle fails | dependency |
| case-03 | Test heap capped at 32m; Maven shows an OutOfMemoryError, Gradle shows the OOM and then hangs until cancelled | memory |

## 1. Case data: what the agent may see

```bash
uv run python - <<'EOF'
from evals.loader import load_agent_input, load_answer_key
inp = load_agent_input("case-01", "maven.log")
print("agent sees:", inp.meta.event, inp.meta.head_sha[:8], "runs:", list(inp.meta.runs))
load_answer_key("case-01", "maven.log")
EOF
```

You should see `agent sees: pull_request c9d5394f runs: ['maven', 'gradle']`,
followed by an `AnswerKeyAccessError`. That error is intended: the answer key
(`expected.yaml`) can only be read from `evals/` or `tests/`, never by agent code.

Check that every case's `meta.yaml` agrees with its own logs:

```bash
uv run python -c "from evals.consistency import check_all_cases; print(check_all_cases() or 'all cases consistent')"
```

Tests: `tests/test_loader.py`, `tests/test_consistency.py`

## 2. Preprocess: find the first real error

```bash
uv run python cli.py preprocess --case case-02 --log gradle.log | head -1
```

Expected: `298 raw lines, 100 kept; first fatal error: javac_error at line 239`

Try: drop `| head -1` to see the whole excerpt. `239>` marks the first error,
`... [N lines omitted] ...` marks gaps, and every number matches `grep -n` on the
raw file. Then try `--case case-03 --log gradle.log`: the OOM at line 543 is the
first error, and the cancellation at line 550 is reported separately.

Tests: `tests/test_preprocess.py`

## 3. Redact secrets

```bash
uv run python - <<'EOF'
from guardrails.redact import redact_with_report
r = redact_with_report("DB_PASSWORD=hunter2 url=postgres://app:s3cr3t@db/x token: ***")
print(r.text); print(r.counts)
EOF
```

Expected:

```
DB_PASSWORD=[REDACTED:password] url=postgres://[REDACTED:connection_string]@db/x token: ***
{'connection_string': 1, 'password': 1}
```

GitHub's own `***` masking is left alone. Try: run `redact` on its own output;
nothing changes the second time.

Tests: `tests/test_redact.py`

## 4. Read the run and the repo

```bash
uv run python - <<'EOF'
from adapters.logs.local import LocalLogSource
from adapters.repo.local import LocalRepoReader
from evals.loader import CASES_DIR
logs = LocalLogSource(CASES_DIR)
run = logs.get_run(logs.ref_for("case-02", "gradle.log"))
print(run.build_tool, run.conclusion, run.head_sha[:8])
repo = LocalRepoReader.from_env()
print("guava in build.gradle:", "guava" in repo.read_file(run.head_sha, "build.gradle").lower())
print("guava in pom.xml:     ", "guava" in repo.read_file(run.head_sha, "pom.xml").lower())
for c in repo.recent_commits(run.head_sha, limit=2): print(" ", c.sha[:8], c.subject)
try: repo.read_file(run.head_sha, ".env")
except Exception as e: print(type(e).__name__, "-", e)
EOF
```

Expected: `gradle failure f9cba0ce`, Guava present in `pom.xml` but not in
`build.gradle` (the case-02 bug itself), the commit `Simplify last-name handling
with Guava`, and `AccessDeniedError - path not in allowlist: '.env'`.

Try: `"../x"`, `"/etc/passwd"`, or `"HEAD"` as the SHA. All are refused: the repo
reader only reads allowlisted paths at hex commit SHAs.

Tests: `tests/test_adapters.py`

## 5. Verify evidence: real vs made up

```bash
uv run python - <<'EOF'
from agent.schema import Diagnosis
from evals.loader import CASES_DIR
from guardrails.verify_evidence import verify_evidence
log = (CASES_DIR / "case-02/gradle.log").read_text()
d = Diagnosis(status="diagnosed", category="dependency", specialist=None, root_cause="x",
  local_vs_ci="y", fix_diff=None, confidence=0.9, evidence=[
  {"source": "log", "line": 239, "snippet": "error: package com.google.common.base does not exist"},
  {"source": "log", "line": 239, "snippet": "error: cannot resolve guava 33.0"}])
r = verify_evidence(d, log=log, read_file=lambda p: None)
print("kept:", [e.snippet for e in r.diagnosis.evidence]); print("dropped:", [e.snippet for e in r.dropped])
print("status/conf:", r.diagnosis.status.value, r.diagnosis.confidence)
EOF
```

Expected: the real line is kept, the made-up one is dropped, and the diagnosis
stays `diagnosed 0.9`.

Try: delete the first evidence entry. With nothing verified, the result becomes
`possible_cause 0.3`. Changing `239` to `240` also makes the real line fail: log
evidence must cite the exact line.

Tests: `tests/test_guardrails.py -k "verif or evidence"`

## 6. Gate: who is alerted, and whether a coding agent may fix it

```bash
uv run python - <<'EOF'
from agent.schema import Diagnosis
from guardrails.gating import decide, apply_auto_fix_gating
for conf, status in [(0.9, "diagnosed"), (0.55, "diagnosed"), (0.3, "diagnosed"), (0.9, "undiagnosable")]:
    d = Diagnosis(status=status, category="compile", specialist=None, root_cause="x", local_vs_ci="y",
        fix_diff=None, confidence=conf, evidence=[{"source": "log", "line": 1, "snippet": "s", "verified": True}])
    g = decide(d)
    print(f"{conf:<4} {status:<14} -> {g.level.value:<15} {[c.value for c in g.channels]}  auto_fixable={apply_auto_fix_gating(d).auto_fixable}")
EOF
```

Expected:

```
0.9  diagnosed      -> alert           ['pr_comment', 'teams']  auto_fixable=True
0.55 diagnosed      -> possible_cause  ['pr_comment']  auto_fixable=False
0.3  diagnosed      -> none            []  auto_fixable=False
0.9  undiagnosable  -> none            []  auto_fixable=False
```

Try: change `category="compile"` to `"memory"`. `auto_fixable` turns False even at
0.9: only `compile`, `format` and `test_assertion` failures are handed to a coding
agent.

Tests: `tests/test_guardrails.py -k "alert_thresholds or auto_fix"`

## 7. Sanitize the output

```bash
uv run python -c "from guardrails.sanitize import sanitize_text as s; print(s('<b>Fix</b> @octocat see https://evil.example and List<String>', repo='pathmanabanptech/petclinic-ci-failures', limit=500))"
```

Expected: ``Fix `@octocat` see [link removed] and List<String>``. HTML is gone, the
mention pings nobody, the foreign link is removed, and Java generics survive.

Try: a link to `https://github.com/pathmanabanptech/petclinic-ci-failures/actions`
is kept, because it points into the repo itself.

Tests: `tests/test_guardrails.py -k "html or mention or url or cap"`

## 8. Budget

```bash
uv run python - <<'EOF'
from guardrails.budget import Budget, BudgetLimits, BudgetExceeded
b = Budget(BudgetLimits(max_tool_calls_per_agent=2))
b.charge_tool_call(); b.charge_tool_call()
try: b.charge_tool_call()
except BudgetExceeded as e: print("BudgetExceeded:", e)
EOF
```

Expected: `BudgetExceeded: tool_calls budget exceeded by agent 'generalist': 3 > 2`

Tests: `tests/test_guardrails.py -k budget`

## 9. All stages together

There is no single command for the whole flow yet; that comes with the pipeline
(`cli.py analyze --case case-01 --dry-run`). Until then, this chains the built
stages on case-01, with a hand-written diagnosis standing in for the LLM. It cites
two real pieces of evidence and one made-up one:

```bash
uv run python - <<'EOF'
from adapters.base import AdapterError, Report
from adapters.logs.local import LocalLogSource
from adapters.notify.console import ConsoleNotifier
from adapters.repo.local import LocalRepoReader
from agent.schema import Diagnosis
from evals.loader import CASES_DIR
from guardrails.gating import apply_auto_fix_gating, decide
from guardrails.sanitize import sanitize_diagnosis
from guardrails.verify_evidence import verify_evidence

logs = LocalLogSource(CASES_DIR)
ref = logs.ref_for("case-01", "maven.log")
run, raw = logs.get_run(ref), logs.get_log(ref)
repo = LocalRepoReader.from_env()

def read_file(path):
    try: return repo.read_file(run.head_sha, path)
    except AdapterError: return None

stand_in = Diagnosis(status="diagnosed", category="compile", specialist=None, confidence=0.9,
    root_cause="Owner.hasPet() calls getPet(name, true), renamed to findPetByName on the base. cc @octocat",
    local_vs_ci="The branch built green alone; only the PR merge build fails.",
    fix_diff="-\t\treturn getPet(name, true) != null;\n+\t\treturn findPetByName(name, true) != null;\n",
    evidence=[
      {"source": "log", "line": 279, "snippet": "Owner.java:[194,24] no suitable method found for getPet(java.lang.String,boolean)"},
      {"source": "src/main/java/org/springframework/samples/petclinic/owner/Owner.java", "line": 194,
       "snippet": "return getPet(name, true) != null;"},
      {"source": "log", "line": 279, "snippet": "cannot find symbol getPetByName"}])

result = verify_evidence(stand_in, log=raw, read_file=read_file)
print(f"evidence: {len(result.diagnosis.evidence)} verified, {len(result.dropped)} dropped")
gated = apply_auto_fix_gating(result.diagnosis)
print("gate:", decide(gated).level.value)
clean = sanitize_diagnosis(gated, repo=run.ref.repo)
ConsoleNotifier().notify(Report(run.ref.repo, run.head_sha, (run.build_tool,), clean))
EOF
```

Expected: `evidence: 2 verified, 1 dropped`, `gate: alert`, then the console alert
with `` `@octocat` ``, both verified evidence lines, the fix in a code block and
`Auto-fixable: yes`.

## Run everything

```bash
make lint test
```
