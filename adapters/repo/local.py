"""RepoReader over a local git clone (``PETCLINIC_REPO_PATH``).

Read-only (``git cat-file`` / ``git show`` / ``git log``); the agent never modifies the
repo. The merge a pull_request build compiles is computed with ``git merge-tree
--write-tree``; the tree objects it writes go to a private temporary object directory
(the clone's objects are only read, via ``GIT_ALTERNATE_OBJECT_DIRECTORIES``), so the
clone's ``.git`` is never written to. Requests are policed before git runs:

* ``sha`` must be 7-40 lowercase hex. Refs such as ``HEAD`` or anything starting
  with ``-`` (git option injection) are rejected.
* ``path`` must be relative POSIX, with no ``..``/``.`` segments, backslashes,
  colons or control characters, and must match the allowlist (build files, sources,
  workflows). ``.env``, keys and other files never become readable by accident.
* Binary and oversized files are refused.
* ``identifier`` (for ``symbol_history``) must be a plain identifier; it is escaped
  into a single ``-G<regex>`` argument.
* Patches are checked against a private temporary index (``git apply --cached
  --check``); every path a patch names must pass the read allowlist, so a patch
  cannot probe files the agent may not read.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from adapters.base import (
    AccessDeniedError,
    AdapterError,
    Commit,
    NotFoundError,
    PatchCheck,
    SymbolChange,
    SymbolLine,
)

ENV_VAR = "PETCLINIC_REPO_PATH"

DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = ("src/", ".github/workflows/", ".mvn/", "gradle/")
DEFAULT_ALLOWED_FILES: frozenset[str] = frozenset(
    {
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradle.properties",
    }
)

_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_BAD_PATH_CHARS = re.compile(r"[\\:\x00-\x1f\x7f]")
_MAX_COMMITS = 50
_MAX_SYMBOL = 200
_MAX_SYMBOL_LINES = 10  # per side per commit
_MAX_LINE_CHARS = 200
_FIELD_SEP = "\x1f"
_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]{0,199}$")
_PATCH_HEADER = re.compile(r"^(---|\+\+\+) (\S+)(?:\t.*)?$", re.M)


class LocalRepoReader:
    def __init__(
        self,
        repo_path: Path,
        *,
        allowed_prefixes: Iterable[str] = DEFAULT_ALLOWED_PREFIXES,
        allowed_files: Iterable[str] = DEFAULT_ALLOWED_FILES,
        max_bytes: int = 512_000,
        timeout_s: float = 15.0,
    ) -> None:
        if not (repo_path / ".git").exists():
            raise AdapterError(f"{repo_path} is not a git clone")
        self._repo = repo_path
        self._prefixes = tuple(allowed_prefixes)
        self._files = frozenset(allowed_files)
        self._max_bytes = max_bytes
        self._timeout = timeout_s
        self._merge_trees: dict[tuple[str, str], str] = {}
        self._objects: tempfile.TemporaryDirectory[str] | None = None  # merge-tree output

    @classmethod
    def from_env(cls) -> LocalRepoReader:
        value = os.environ.get(ENV_VAR)
        if not value:
            raise AdapterError(f"{ENV_VAR} is not set (path to the petclinic-ci-failures clone)")
        return cls(Path(value).expanduser())

    # ---------------------------------------------------------------- policy

    def _check_sha(self, sha: str) -> str:
        if not _SHA.match(sha):
            raise AccessDeniedError(f"not a commit SHA: {sha!r}")
        return sha

    def _check_path(self, path: str) -> str:
        if not path or path.startswith("/") or _BAD_PATH_CHARS.search(path):
            raise AccessDeniedError(f"invalid path: {path!r}")
        parts = path.split("/")
        if any(p in {"", ".", ".."} for p in parts) or parts[0].startswith(("-", "~")):
            raise AccessDeniedError(f"invalid path: {path!r}")
        if path not in self._files and not path.startswith(self._prefixes):
            raise AccessDeniedError(f"path not in allowlist: {path!r}")
        return path

    # ------------------------------------------------------------------- git

    def _object_env(self) -> dict[str, str]:
        """Write new objects to a private temp dir; read the clone's objects too."""
        if self._objects is None:
            self._objects = tempfile.TemporaryDirectory(prefix="cfa-objects-")
        repo_objects = self._git_path("objects")
        return {
            "GIT_OBJECT_DIRECTORY": self._objects.name,
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": repo_objects,
        }

    def _git_path(self, name: str) -> str:
        proc = self._git("rev-parse", "--path-format=absolute", "--git-path", name)
        if proc.returncode != 0:
            raise AdapterError(f"cannot locate {name} in {self._repo}")
        return proc.stdout.decode().strip()

    def _git(
        self, *args: str, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        env = {
            **os.environ,
            **(extra_env or {}),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "LC_ALL": "C",
        }
        try:
            return subprocess.run(
                ["git", "-C", str(self._repo), "-c", "core.quotepath=off", *args],
                capture_output=True,
                timeout=self._timeout,
                env=env,
                check=False,
            )
        except FileNotFoundError as e:
            raise AdapterError("git is not installed") from e
        except subprocess.TimeoutExpired as e:
            raise AdapterError(f"git {args[0]} timed out after {self._timeout}s") from e

    def _require_commit(self, sha: str) -> None:
        if self._git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            raise NotFoundError(f"commit {sha} not in {self._repo} (try `git fetch --all`)")

    # ------------------------------------------------------------------- API

    def read_file(self, sha: str, path: str) -> str:
        sha, path = self._check_sha(sha), self._check_path(path)
        self._require_commit(sha)
        return self._read_blob(sha, path, sha[:12])

    def read_file_at_merge(self, base_sha: str, head_sha: str, path: str) -> str:
        base_sha, head_sha = self._check_sha(base_sha), self._check_sha(head_sha)
        path = self._check_path(path)
        tree = self._merge_tree(base_sha, head_sha)
        return self._read_blob(
            tree,
            path,
            f"merge of {head_sha[:12]} into {base_sha[:12]}",
            extra_env=self._object_env(),
            show=True,
        )

    def _merge_tree(self, base_sha: str, head_sha: str) -> str:
        key = (base_sha, head_sha)
        if key not in self._merge_trees:
            self._require_commit(base_sha)
            self._require_commit(head_sha)
            proc = self._git(
                "merge-tree", "--write-tree", base_sha, head_sha, extra_env=self._object_env()
            )
            out = proc.stdout.decode(errors="replace").split("\n")
            if proc.returncode == 1:
                raise AdapterError(
                    f"{head_sha[:12]} does not merge cleanly into {base_sha[:12]} (conflicts)"
                )
            if proc.returncode != 0 or not _SHA.match(out[0].strip()):
                err = proc.stderr.decode(errors="replace").strip()
                raise AdapterError(f"git merge-tree failed (needs git >= 2.38): {err}")
            self._merge_trees[key] = out[0].strip()
        return self._merge_trees[key]

    def _read_blob(
        self,
        rev: str,
        path: str,
        label: str,
        *,
        extra_env: dict[str, str] | None = None,
        show: bool = False,
    ) -> str:
        kind = self._git("cat-file", "-t", f"{rev}:{path}", extra_env=extra_env)
        if kind.returncode != 0:
            raise NotFoundError(f"{path} does not exist at {label}")
        if kind.stdout.strip() != b"blob":  # e.g. a directory (tree), not a file
            raise NotFoundError(f"{path} is not a file at {label}")
        size = self._git("cat-file", "-s", f"{rev}:{path}", extra_env=extra_env)
        if size.returncode != 0 or int(size.stdout) > self._max_bytes:
            raise AccessDeniedError(f"{path} is larger than {self._max_bytes} bytes")
        cmd = ("show", "--no-textconv") if show else ("cat-file", "blob")
        proc = self._git(*cmd, f"{rev}:{path}", extra_env=extra_env)
        if proc.returncode != 0:
            raise NotFoundError(f"{path} could not be read at {label}")
        if b"\x00" in proc.stdout:
            raise AccessDeniedError(f"{path} is a binary file")
        return proc.stdout.decode("utf-8", errors="replace")

    def recent_commits(self, sha: str, limit: int = 10, path: str | None = None) -> list[Commit]:
        sha = self._check_sha(sha)
        limit = max(1, min(limit, _MAX_COMMITS))
        self._require_commit(sha)
        args = ["log", "--no-color", f"--format=%H{_FIELD_SEP}%aI{_FIELD_SEP}%s", f"-n{limit}", sha]
        if path is not None:
            args += ["--", self._check_path(path)]
        proc = self._git(*args)
        if proc.returncode != 0:
            raise AdapterError(f"git log failed: {proc.stderr.decode(errors='replace').strip()}")
        commits = []
        for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
            full_sha, authored, subject = line.split(_FIELD_SEP, 2)
            commits.append(
                Commit(sha=full_sha, authored_at=datetime.fromisoformat(authored), subject=subject)
            )
        return commits

    def _check_identifier(self, identifier: str) -> str:
        if not _IDENTIFIER.match(identifier):
            raise AccessDeniedError(
                f"not a plain identifier: {identifier!r} (letters, digits, _ or $; "
                f"max {_MAX_SYMBOL} chars), e.g. getPet"
            )
        return identifier

    def symbol_history(
        self, base_sha: str | None, head_sha: str, identifier: str, path: str | None = None
    ) -> list[SymbolChange]:
        head_sha = self._check_sha(head_sha)
        identifier = self._check_identifier(identifier)
        self._require_commit(head_sha)
        # POSIX ERE equivalent of \b<id>\s*\( : macOS git's regex engine has no \b / \s.
        pattern = f"(^|[^[:alnum:]_$]){re.escape(identifier)}[[:space:]]*\\("
        fmt = f"--format=%m{_FIELD_SEP}%H{_FIELD_SEP}%s"
        args = ["log", "--no-color", fmt, "-E", f"-G{pattern}", f"-n{_MAX_COMMITS}"]
        if base_sha is not None:
            base_sha = self._check_sha(base_sha)
            self._require_commit(base_sha)
            # base...head: reachable from head or from base, but not from their merge base
            args += ["--left-right", f"{base_sha}...{head_sha}"]
        else:
            args.append(head_sha)
        pathspec = [self._check_path(path)] if path else []
        proc = self._git(*args, "--", *pathspec)
        if proc.returncode != 0:
            raise AdapterError(f"git log -G failed: {proc.stderr.decode(errors='replace').strip()}")
        changes = []
        for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
            mark, sha, subject = line.split(_FIELD_SEP, 2)
            added, removed = self._identifier_lines(sha, identifier, pathspec)
            changes.append(
                SymbolChange(
                    sha=sha,
                    subject=subject,
                    side="base" if mark == "<" else "head",
                    added=added,
                    removed=removed,
                )
            )
        return changes

    def _identifier_lines(
        self, sha: str, identifier: str, pathspec: list[str]
    ) -> tuple[tuple[SymbolLine, ...], tuple[SymbolLine, ...]]:
        """The commit's +/- lines that contain ``identifier`` as a whole word."""
        word = re.compile(rf"(?<![\w$]){re.escape(identifier)}(?![\w$])")
        proc = self._git(
            "show", "--no-color", "--format=", "--unified=0", "--no-ext-diff", sha, "--", *pathspec
        )
        added: list[SymbolLine] = []
        removed: list[SymbolLine] = []
        current = ""
        for raw in proc.stdout.decode("utf-8", errors="replace").splitlines():
            if raw.startswith("+++ "):
                current = raw[6:] if raw.startswith("+++ b/") else raw[4:]
            elif raw.startswith("--- "):
                continue
            elif raw[:1] in "+-" and word.search(raw[1:]):
                target = added if raw[0] == "+" else removed
                if len(target) < _MAX_SYMBOL_LINES:
                    target.append(SymbolLine(path=current, text=raw[1:].strip()[:_MAX_LINE_CHARS]))
        return tuple(added), tuple(removed)

    # ------------------------------------------------------------- patches

    def check_patch(self, patch: str, head_sha: str, base_sha: str | None = None) -> PatchCheck:
        head_sha = self._check_sha(head_sha)
        self._require_commit(head_sha)
        try:
            strip = self._check_patch_paths(patch)
        except AccessDeniedError as e:
            return PatchCheck(False, str(e))
        if base_sha is not None:
            tree, label = self._merge_tree(self._check_sha(base_sha), head_sha), "merge"
        else:
            tree, label = f"{head_sha}^{{tree}}", "head"
        with tempfile.TemporaryDirectory(prefix="cfa-patch-") as tmp:
            # A private index holding the tree; `git apply --cached --check` reads it and
            # the objects, and writes nothing.
            env = {**self._object_env(), "GIT_INDEX_FILE": os.path.join(tmp, "index")}
            read = self._git("read-tree", tree, extra_env=env)
            if read.returncode != 0:
                raise AdapterError(f"git read-tree failed: {read.stderr.decode(errors='replace')}")
            patch_file = os.path.join(tmp, "fix.patch")
            with open(patch_file, "w", encoding="utf-8") as f:
                f.write(patch if patch.endswith("\n") else patch + "\n")
            proc = self._git(
                "apply", "--cached", "--check", f"-p{strip}", patch_file, extra_env=env
            )
        if proc.returncode == 0:
            return PatchCheck(True, f"applies to {label}")
        detail = " ".join(proc.stderr.decode(errors="replace").split())[:300]
        return PatchCheck(False, f"does not apply to {label}: {detail or 'git apply failed'}")

    def _check_patch_paths(self, patch: str) -> int:
        """Every path the patch names must pass the read allowlist; returns the -p level."""
        headers = _PATCH_HEADER.findall(patch)
        if not headers or {side for side, _ in headers} != {"---", "+++"}:
            raise AccessDeniedError("not a unified diff (no ---/+++ file headers)")
        named = [p for _, p in headers if p != "/dev/null"]
        strip = 1 if named and all(p.startswith(("a/", "b/")) for p in named) else 0
        for p in named:
            self._check_path(p[2:] if strip else p)
        return strip
