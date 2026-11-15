"""Fix verifier: a suggested ``fix_diff`` is kept only if it applies to the code the
build compiled: the merge of head into base for pull_request builds, head otherwise.

The check itself runs in the RepoReader (``check_patch``: a private temporary index
holding the head tree or the ``git merge-tree --write-tree`` result, then ``git apply
--cached --check``); nothing in the repository is modified and nothing from the diff
is executed. This module only decides what to do with the answer.

A diff that does not apply (invented context, wrong file, malformed hunk), names a
path outside the read allowlist, or has no file headers is dropped: ``fix_diff``
becomes ``None``. Without a repository the fix cannot be checked and is dropped too
(fail closed).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent.schema import Diagnosis


@dataclass(frozen=True)
class FixCheck:
    ok: bool
    reason: str


PatchChecker = Callable[[str], FixCheck]  # diff -> verdict against the compiled tree


def verify_fix(d: Diagnosis, check: PatchChecker | None) -> tuple[Diagnosis, FixCheck | None]:
    """Drop ``fix_diff`` unless it applies. Returns the check (None: no fix proposed)."""
    if not d.fix_diff:
        return d, None
    if check is None:
        result = FixCheck(False, "repository not available; the fix cannot be checked")
    else:
        result = check(d.fix_diff)
    return (d if result.ok else d.model_copy(update={"fix_diff": None})), result
