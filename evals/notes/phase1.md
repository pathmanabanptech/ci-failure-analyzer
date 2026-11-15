# Phase 1 notes: generalist agent, first hand review

Observations only. No prompt or agent tuning has been done from these yet.

## Run

| | |
|---|---|
| Date | 2026-11-14 |
| Model | Ollama `qwen2.5:7b-instruct@845dbda0ea48` (Q4_K_M, temperature 0, `num_ctx` 16384) |
| Prompt | `prompts/generalist.md`, sha256 `d5f925852dc2…` |
| Code | `dc6504b` + uncommitted 1.9 changes |
| Repo reader | local clone of `petclinic-ci-failures` (file evidence verifiable) |
| N | 1 run per log (no variance data yet) |

Command: `uv run python cli.py analyze --case <case> --log <log>`

## Results

| Log | Valid JSON | Agent said | After verify | Cited / verified | Tool calls | Model calls | Input tok | Time |
|---|---|---|---|---|---|---|---|---|
| case-01 maven | yes | diagnosed / compile / 1.0 | possible_cause / 0.3 | 1 / 0 | 2 | 3 | 12.4k | 27 s |
| case-01 gradle | yes | diagnosed / compile / 0.9 | diagnosed / 0.9 | 1 / 1 | 3 | 4 | 25.4k | 56 s |
| case-02 gradle | yes | diagnosed / dependency / 0.9 | diagnosed / 0.9 | 4 / 3 | 4 | 5 | 33.1k | 70 s |
| case-03 maven | yes¹ | diagnosed / memory / 1.0 | possible_cause / 0.3 | 1 / 0 | 3 | 4 | 48.9k² | 94 s |
| case-03 gradle | yes¹ | diagnosed / memory / 0.9 | possible_cause / 0.3 | 1 / 0 | 3 | 6 | 26.3k | 49 s |

¹ Through the structured-output fallback: the model ended with prose instead of calling
the structured-output tool; a second, schema-constrained call produced the JSON.
² Includes an estimate for the fallback call (the SDK does not report its usage).

**Gate status:** valid JSON 5/5. **≥ 1 verified evidence: 2/5 logs, so the 1.9 gate is not
met with this model.** Every dropped item was a real model mistake (wrong line, or no line).
None was a verifier false negative. Category is correct on 5/5. The hallucinated-evidence
rate is 4 of 8 cited items (50%).

## Mistakes, per log

### case-01 maven (compile, semantic merge)
- **Evidence:** cited the right text (`Owner.java:[194,24] no suitable method found for
  getPet(java.lang.String,boolean)`) but at line 277, the line marked `>`, which reads
  `[ERROR] COMPILATION ERROR :`. The text is at line 279. Dropped.
- **Root cause** says `getPet(String, boolean)` "does not exist". That is true but misses the
  rename (`findPetByName`) on the base side and the new caller `hasPet()` in the PR.
- **local_vs_ci** is vague ("might not have the exact merge result") but points the right way.
- **fix_diff** is invented: a fabricated `index` line, a made-up hunk
  (`pets.get(petName, false)`) that is not in the file. It would not apply.
- Never used `ref="base"` or `recent_commits`, so it never saw the rename.

### case-01 gradle (compile, semantic merge)
- **Evidence:** `Owner.java:194  return getPet(name, true) != null;` verified (head).
- **Root cause** is wrong: "likely due to a classpath issue".
- **local_vs_ci** is wrong (classpath/version of `Owner`); the merge was not mentioned.
- **fix_diff:** none.
- Same gap as maven: never checked the base commit.

### case-02 gradle (dependency, build drift)
- Best of the five. Verified: log 238 `> Task :compileJava FAILED`, `OwnerController.java:22`
  import, `OwnerController.java:103` usage.
- Dropped: `symbol:   variable Strings` cited at log 242; it is at line 245 (off by 3).
- **Root cause** says it is missing from `build.gradle` but never says it **is** in `pom.xml`,
  and never cites `pom.xml` (expected source). Did not use the passing Maven related run.
- **local_vs_ci** is backwards in substance ("local classpath has Guava"); the real reason
  is that the developer built with Maven.
- **fix_diff** has the right idea (add Guava to `build.gradle`), but the hunk context is
  invented (`spring-boot-starter-web` is not in this file; header line counts wrong).

### case-03 maven (memory, OOM)
- Needed the fallback.
- **Evidence:** `java.lang.OutOfMemoryError: Java heap space` with **no line number**.
  Log evidence requires a line, so it was dropped. The text exists (e.g. line 4078, as
  `Caused by: …`).
- **Root cause** blames "the JVM running the Maven build"; it is the Surefire fork heap
  (`-Xmx32m` argLine in `pom.xml`), which it never cited.
- **local_vs_ci** is generic but roughly right (larger heap locally).
- **fix_diff:** none.

### case-03 gradle (memory, OOM then hang until cancel)
- Needed the fallback.
- **Evidence:** `Caused by: java.lang.OutOfMemoryError at Arrays.java:3537`, no line and
  paraphrased. Dropped.
- **Root cause** found `32m` (good), but says "AOT compilation" (wrong; it is the test JVM).
- Did not mention the cancellation as a symptom (expected key fact).
- **fix_diff** is malformed: it keeps the old line, adds a second `maxHeapSize`, and
  deletes the closing brace. It uses `#` comments, which are invalid in Groovy.

## Cross-cutting observations (candidates for later increments, not changed yet)

1. **Line numbers are the main failure mode.** All 4 dropped items had the right (or nearly
   right) text with a wrong or missing line. One is caused by our own excerpt: the `>` marker sits on Maven's
   `[ERROR] COMPILATION ERROR :` header, and the model attached that line number to the
   real message two lines below. A preprocessing option is to mark the first *specific*
   error line.
2. **Base-side reasoning never happens.** On the semantic-merge case the model never read
   the PR base or commits, so it cannot reach the real cause. This is a capability/prompt
   question for evals, not something to patch now.
3. **Related runs are ignored.** The passing Maven run (case-02) and the passing branch
   builds (case-01) are the strongest `local_vs_ci` signal; the model did not use them.
4. **Every `fix_diff` was fabricated or malformed.** The diffs were not built from file content
   the model had read. Verifying or regenerating the fix is later work; for now a
   wrong fix is only ever shown with a downgraded status when its evidence fails.
5. **Confidence is overstated.** The model reported 0.9–1.0 on all five, including the
   three wrong root causes. The verifier's cap is the only thing keeping alerts quiet.
6. **Structured output with 7B Ollama is unreliable**: 2/5 runs needed the constrained
   fallback. Keep the fallback; record its rate in the eval runner.
7. **Token use is close to the budget**: up to ~49k of the 60k input budget, because every
   model call resends the excerpt plus tool results. A larger model with more tool calls
   will hit the budget.
8. A 7B model is enough to test the wiring but not to give a quality baseline. Compare
   with `qwen2.5-coder:32b` locally and with the target cloud model before tuning anything.
