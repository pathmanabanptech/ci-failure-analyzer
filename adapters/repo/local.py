"""RepoReader over a local git clone (``PETCLINIC_REPO_PATH``).

Read-only (``git cat-file`` / ``git log``; design rule 6). Requests are policed
before git runs:

* ``sha`` must be 7-40 lowercase hex. Refs such as ``HEAD`` or anything starting
  with ``-`` (git option injection) are rejected.
* ``path`` must be relative POSIX, with no ``..``/``.`` segments, backslashes,
  colons or control characters, and must match the allowlist (build files, sources,
  workflows). ``.env``, keys and other files never become readable by accident.
* Binary and oversized files are refused.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from adapters.base import AccessDeniedError, AdapterError, Commit, NotFoundError

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
_FIELD_SEP = "\x1f"


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

    def _git(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        env = {
            **os.environ,
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
        size = self._git("cat-file", "-s", f"{sha}:{path}")
        if size.returncode != 0:
            raise NotFoundError(f"{path} does not exist at {sha[:12]}")
        if int(size.stdout) > self._max_bytes:
            raise AccessDeniedError(f"{path} is larger than {self._max_bytes} bytes")
        proc = self._git("cat-file", "blob", f"{sha}:{path}")
        if proc.returncode != 0:  # e.g. a directory (tree), not a file
            raise NotFoundError(f"{path} is not a file at {sha[:12]}")
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
