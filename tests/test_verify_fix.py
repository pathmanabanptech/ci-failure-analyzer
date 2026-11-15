"""Fix verifier: fix_diff is kept only if it applies to what the build compiled.

The git side (temp index + ``git apply --cached --check`` against head or the merge)
is tested in tests/test_adapters.py with a temp repo; here: the verdict handling and
one end-to-end check against the real petclinic clone.
"""

from __future__ import annotations

import difflib
import os
from typing import Any

import pytest

from agent.schema import Diagnosis
from evals.loader import load_meta
from guardrails.verify_fix import FixCheck, verify_fix

OWNER = "src/main/java/org/springframework/samples/petclinic/owner/Owner.java"

# The fix_diff qwen2.5:7b proposed for case-01 maven: invented index line and hunk.
FABRICATED = """\
diff --git a/src/main/java/org/springframework/samples/petclinic/owner/Owner.java b/src/main/java/org/springframework/samples/petclinic/owner/Owner.java
index 1234567..89abcde 100644
--- a/src/main/java/org/springframework/samples/petclinic/owner/Owner.java
+++ b/src/main/java/org/springframework/samples/petclinic/owner/Owner.java
@@ -194,7 +194,7 @@ public class Owner {
     // Other method definitions
     public Pet getPet(String petName) {
         // Implementation
-        return pets.get(petName, false);
+        return pets.get(petName);
     }

     public Pet getPet(Integer petId) {
"""  # noqa: E501


def make_diff(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), f"a/{path}", f"b/{path}"
        )
    )


def diag(fix: str | None) -> Diagnosis:
    data: dict[str, Any] = {
        "status": "diagnosed",
        "category": "compile",
        "specialist": None,
        "root_cause": "r",
        "local_vs_ci": "l",
        "evidence": [],
        "fix_diff": fix,
        "confidence": 0.9,
    }
    return Diagnosis.model_validate(data)


def test_applying_fix_is_kept() -> None:
    seen: list[str] = []

    def check(patch: str) -> FixCheck:
        seen.append(patch)
        return FixCheck(True, "applies to merge")

    d, result = verify_fix(diag("--- a/x\n+++ b/x\n"), check)
    assert result == FixCheck(True, "applies to merge") and d.fix_diff == "--- a/x\n+++ b/x\n"
    assert seen == ["--- a/x\n+++ b/x\n"]


def test_non_applying_fix_is_dropped() -> None:
    d, result = verify_fix(diag(FABRICATED), lambda p: FixCheck(False, "does not apply to merge"))
    assert result is not None and not result.ok and d.fix_diff is None
    assert d.root_cause == "r" and d.confidence == 0.9  # nothing else changes


def test_without_a_repository_the_fix_is_dropped() -> None:
    d, result = verify_fix(diag(FABRICATED), None)
    assert d.fix_diff is None and result is not None and "not available" in result.reason


def test_no_fix_no_check() -> None:
    d, result = verify_fix(diag(None), lambda p: FixCheck(True, "never called"))
    assert result is None and d.fix_diff is None


# ------------------------------------------- against the real petclinic clone


@pytest.mark.skipif(not os.environ.get("PETCLINIC_REPO_PATH"), reason="PETCLINIC_REPO_PATH not set")
def test_real_case01_fix_head_vs_merge() -> None:
    from adapters.repo.local import LocalRepoReader

    repo = LocalRepoReader.from_env()
    meta = load_meta("case-01")
    assert meta.base_sha is not None
    merged = repo.read_file_at_merge(meta.base_sha, meta.head_sha, OWNER)
    assert "getPet(name, true)" in merged  # the PR's caller survives the merge ...
    fix = make_diff(
        OWNER, merged, merged.replace("getPet(name, true)", "findPetByName(name, true)")
    )

    assert repo.check_patch(fix, meta.head_sha, meta.base_sha).ok  # ... and this fixes it
    assert not repo.check_patch(FABRICATED, meta.head_sha, meta.base_sha).ok
    assert not repo.check_patch(FABRICATED, meta.head_sha).ok
