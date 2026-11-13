import io
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from adapters.base import (
    AccessDeniedError,
    AdapterError,
    LogSource,
    MemoryQuery,
    MemoryRecord,
    MemoryStore,
    NotFoundError,
    Notifier,
    RepoReader,
    Report,
    RunRef,
)
from adapters.logs.local import LocalLogSource
from adapters.memory.sqlite import SqliteMemoryStore
from adapters.notify.console import ConsoleNotifier
from adapters.repo.local import ENV_VAR, LocalRepoReader
from agent.schema import Category, Diagnosis
from evals.loader import CASES_DIR

REPO = "pathmanabanptech/petclinic-ci-failures"

# ===================================================================== LogSource


@pytest.fixture
def logs() -> LocalLogSource:
    return LocalLogSource(CASES_DIR)


def test_protocol_conformance(tmp_path: Path, git_repo: tuple[Path, list[str]]) -> None:
    assert isinstance(LocalLogSource(CASES_DIR), LogSource)
    assert isinstance(LocalRepoReader(git_repo[0]), RepoReader)
    assert isinstance(ConsoleNotifier(io.StringIO()), Notifier)
    assert isinstance(SqliteMemoryStore(), MemoryStore)


def test_ref_for_and_get_log(logs: LocalLogSource) -> None:
    ref = logs.ref_for("case-01", "maven.log")
    assert ref == RunRef(repo=REPO, run_id=42485422975, job_id=128746111502)
    assert logs.get_log(ref) == (CASES_DIR / "case-01/maven.log").read_text(encoding="utf-8")


def test_get_run_pr_case(logs: LocalLogSource) -> None:
    run = logs.get_run(RunRef(REPO, 42485422968))  # job_id optional
    assert run.build_tool == "gradle" and run.event == "pull_request"
    assert run.head_sha.startswith("c9d5394f") and run.base_sha is not None
    assert run.base_sha.startswith("afab46b7")
    assert [(a.attempt, a.conclusion) for a in run.attempts] == [(1, "failure")]
    passed_on_old_base = [r for r in run.related_runs if r.event == "pull_request"]
    assert {r.conclusion for r in passed_on_old_base} == {"success"}
    assert all(r.base_sha and r.base_sha.startswith("32cb980c") for r in passed_on_old_base)


def test_get_run_cancelled(logs: LocalLogSource) -> None:
    run = logs.get_run(logs.ref_for("case-03", "gradle.log"))
    assert run.conclusion == "cancelled" and run.base_sha is None


def test_passed_run_has_no_log(logs: LocalLogSource) -> None:
    ref = RunRef(REPO, 42479137740)  # case-02 maven: success
    assert logs.get_run(ref).conclusion == "success"
    with pytest.raises(NotFoundError, match="no failure log"):
        logs.get_log(ref)


@pytest.mark.parametrize(
    "ref",
    [RunRef(REPO, 1), RunRef("someone/else", 42485422975), RunRef(REPO, 42485422975, job_id=1)],
)
def test_unknown_run(logs: LocalLogSource, ref: RunRef) -> None:
    with pytest.raises(NotFoundError):
        logs.get_run(ref)


def test_ref_for_unknown_log(logs: LocalLogSource) -> None:
    with pytest.raises(NotFoundError):
        logs.ref_for("case-02", "maven.log")


# ==================================================================== RepoReader

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,  # ignore the user's config (signing, hooks, ...)
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(repo: Path, *args: str, date: str = "2026-11-01T10:00:00+00:00") -> str:
    env = {**os.environ, **GIT_ENV, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    out = subprocess.run(
        ["git", "-C", str(repo), *args], env=env, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _write(repo: Path, rel: str, content: str | bytes) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content)


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, list[str]]:
    """Repo with 3 commits; returns (path, [sha1, sha2, sha3])."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "pom.xml", "<project>v1</project>\n")
    _write(repo, "src/main/java/A.java", "class A { int getPet() { return 1; } }\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "Initial import", date="2026-11-01T10:00:00+00:00")
    _write(repo, "src/main/java/A.java", "class A { int findPetByName() { return 1; } }\n")
    _write(repo, ".env", "DB_PASSWORD=secret\n")
    _write(repo, "src/main/resources/logo.png", b"\x89PNG\r\n\x1a\n\x00\x00binary")
    _git(repo, "add", ".")
    _git(
        repo,
        "commit",
        "-q",
        "-m",
        "Rename getPet to findPetByName",
        date="2026-11-02T10:00:00+00:00",
    )
    _write(repo, "build.gradle", "plugins { id 'java' }\n" + "// pad\n" * 50)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "Add Gradle build", date="2026-11-03T10:00:00+00:00")
    shas = _git(repo, "log", "--reverse", "--format=%H").split("\n")
    return repo, shas


def test_read_file_at_each_sha(git_repo: tuple[Path, list[str]]) -> None:
    repo, (s1, s2, _) = git_repo
    reader = LocalRepoReader(repo)
    assert "getPet()" in reader.read_file(s1, "src/main/java/A.java")
    assert "findPetByName" in reader.read_file(s2, "src/main/java/A.java")
    assert reader.read_file(s1[:12], "pom.xml") == "<project>v1</project>\n"  # short SHA ok


def test_file_missing_at_sha(git_repo: tuple[Path, list[str]]) -> None:
    repo, (s1, _, s3) = git_repo
    reader = LocalRepoReader(repo)
    with pytest.raises(NotFoundError, match="does not exist"):
        reader.read_file(s1, "build.gradle")  # added in commit 3
    assert reader.read_file(s3, "build.gradle").startswith("plugins")


def test_unknown_commit(git_repo: tuple[Path, list[str]]) -> None:
    with pytest.raises(NotFoundError, match="git fetch --all"):
        LocalRepoReader(git_repo[0]).read_file("0" * 40, "pom.xml")


@pytest.mark.parametrize(
    "sha", ["HEAD", "main", "--output=/tmp/x", "abc", "ABCDEF1234", "", "a" * 41]
)
def test_rejects_non_sha_refs(git_repo: tuple[Path, list[str]], sha: str) -> None:
    with pytest.raises(AccessDeniedError, match="not a commit SHA"):
        LocalRepoReader(git_repo[0]).read_file(sha, "pom.xml")


@pytest.mark.parametrize(
    "path",
    [
        "../etc/passwd",
        "/etc/passwd",
        "src/../.env",
        "src/./main/java/A.java",
        "src//main/java/A.java",
        "src\\main\\java\\A.java",
        "src/main/java/A.java:extra",
        "",
        "~/.ssh/id_rsa",
        "src/main/java/A.java\n",
    ],
)
def test_rejects_traversal_and_malformed_paths(git_repo: tuple[Path, list[str]], path: str) -> None:
    repo, shas = git_repo
    with pytest.raises(AccessDeniedError, match="invalid path"):
        LocalRepoReader(repo).read_file(shas[-1], path)


@pytest.mark.parametrize("path", [".env", "README.md", "secrets/key.pem", "srcx/A.java"])
def test_rejects_paths_outside_allowlist(git_repo: tuple[Path, list[str]], path: str) -> None:
    repo, shas = git_repo
    with pytest.raises(AccessDeniedError, match="allowlist"):
        LocalRepoReader(repo).read_file(shas[-1], path)


def test_custom_allowlist(git_repo: tuple[Path, list[str]]) -> None:
    repo, shas = git_repo
    reader = LocalRepoReader(repo, allowed_prefixes=(), allowed_files=("pom.xml",))
    assert reader.read_file(shas[-1], "pom.xml")
    with pytest.raises(AccessDeniedError):
        reader.read_file(shas[-1], "src/main/java/A.java")


def test_rejects_binary_directory_and_oversized(git_repo: tuple[Path, list[str]]) -> None:
    repo, shas = git_repo
    reader = LocalRepoReader(repo, max_bytes=100)
    with pytest.raises(AccessDeniedError, match="binary"):
        reader.read_file(shas[-1], "src/main/resources/logo.png")
    with pytest.raises(NotFoundError, match="not a file"):
        reader.read_file(shas[-1], "src/main")
    with pytest.raises(AccessDeniedError, match="larger than 100"):
        reader.read_file(shas[-1], "build.gradle")


def test_recent_commits(git_repo: tuple[Path, list[str]]) -> None:
    repo, (s1, s2, s3) = git_repo
    reader = LocalRepoReader(repo)
    commits = reader.recent_commits(s3)
    assert [c.sha for c in commits] == [s3, s2, s1]
    assert commits[1].subject == "Rename getPet to findPetByName"
    assert commits[0].authored_at == datetime(2026, 11, 3, 10, tzinfo=UTC)
    assert [c.sha for c in reader.recent_commits(s3, limit=1)] == [s3]
    assert [c.sha for c in reader.recent_commits(s3, path="src/main/java/A.java")] == [s2, s1]
    assert [c.sha for c in reader.recent_commits(s2)] == [s2, s1]  # only reachable history
    with pytest.raises(AccessDeniedError):
        reader.recent_commits(s3, path=".env")


def test_not_a_clone(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="not a git clone"):
        LocalRepoReader(tmp_path)


def test_from_env(monkeypatch: pytest.MonkeyPatch, git_repo: tuple[Path, list[str]]) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(AdapterError, match=ENV_VAR):
        LocalRepoReader.from_env()
    monkeypatch.setenv(ENV_VAR, str(git_repo[0]))
    assert LocalRepoReader.from_env().read_file(git_repo[1][0], "pom.xml")


# Plan 1.5 manual check, automated when the real clone is configured.
@pytest.mark.skipif(not os.environ.get(ENV_VAR), reason=f"{ENV_VAR} not set")
def test_real_clone_reads_case_files() -> None:
    reader = LocalRepoReader.from_env()
    gradle = reader.read_file("f9cba0ceeb31fc1912f196ff77fa831edf805b54", "build.gradle")  # case-02
    assert "guava" not in gradle.lower()  # build drift: Guava only in pom.xml
    pom = reader.read_file("f9cba0ceeb31fc1912f196ff77fa831edf805b54", "pom.xml")
    assert "guava" in pom.lower()
    owner = reader.read_file(
        "afab46b773c6b540fad0e470685afff76f7363a3",  # case-01 base_sha
        "src/main/java/org/springframework/samples/petclinic/owner/Owner.java",
    )
    assert "findPetByName" in owner


# ============================================================ Notifier / Memory


def _diag(**kw: object) -> Diagnosis:
    data: dict[str, object] = {
        "status": "diagnosed",
        "category": "memory",
        "specialist": None,
        "root_cause": "Test heap capped at 32m.",
        "local_vs_ci": "IDE uses a larger default heap.",
        "evidence": [
            {"source": "log", "line": 543, "snippet": "OutOfMemoryError", "verified": True}
        ],
        "fix_diff": "-  maxHeapSize = '32m'\n+  maxHeapSize = '512m'\n",
        "confidence": 0.85,
    }
    data.update(kw)
    return Diagnosis.model_validate(data)


def test_console_notifier_prints_report_and_dedupes() -> None:
    out = io.StringIO()
    notifier = ConsoleNotifier(out)
    report = Report(
        repo=REPO,
        head_sha="259023cc" + "0" * 32,
        build_tools=("maven", "gradle"),
        diagnosis=_diag(),
    )
    assert report.dedupe_key == f"{'259023cc' + '0' * 32}:memory"
    notifier.notify(report)
    notifier.notify(report)
    first, second = out.getvalue().split("=== CI failure analysis")[1:]
    assert "diagnosed / memory (confidence 0.85)" in first and "(updated)" not in first
    assert "builds: maven, gradle" in first
    assert "log:543 (verified): OutOfMemoryError" in first
    assert "```diff\n-  maxHeapSize = '32m'" in first
    assert second.startswith(" (updated)")


def test_console_notifier_without_evidence_or_fix() -> None:
    out = io.StringIO()
    ConsoleNotifier(out).notify(
        Report(REPO, "a" * 40, (), _diag(evidence=[], fix_diff=None, status="undiagnosable"))
    )
    text = out.getvalue()
    assert "(none)" in text and "Suggested fix" not in text and "builds: -" in text


def test_memory_stub_never_stores() -> None:
    store = SqliteMemoryStore()
    assert store.recall(MemoryQuery(repo=REPO)) == []
    record = MemoryRecord(
        repo=REPO,
        category=Category.memory,
        signature="OutOfMemoryError",
        diagnosis=_diag(),
        confirmed_by="reaction",
        stored_at=datetime.now(UTC),
    )
    with pytest.raises(NotImplementedError, match="Phase 3"):
        store.store(record)
