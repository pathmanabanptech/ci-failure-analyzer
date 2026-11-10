"""G3 redaction tests.

Fake secrets are assembled at runtime so no secret-shaped literal sits in the source
(keeps GitHub secret scanning / push protection quiet).
"""

import base64
import json
from pathlib import Path

import pytest

from evals.loader import CASES_DIR
from guardrails.preprocess import preprocess
from guardrails.redact import redact, redact_with_report

ALNUM = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2W3x4Y5z6"


def _b64(obj: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"  # AWS documentation example key
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/" + "bPxRfiCYEXAMPLEKEY"  # AWS documentation example
GH_CLASSIC = "gh" + "p_" + ALNUM[:36]
GH_SERVER = "gh" + "s_" + ALNUM[2:38]
GH_FINE = "github" + "_pat_" + ALNUM[:22] + "_" + ALNUM[:40]
JWT = ".".join([_b64({"alg": "HS256", "typ": "JWT"}), _b64({"sub": "ci", "n": 1}), ALNUM[:43]])
KEY_BEGIN = "-----BEGIN " + "RSA PRIVATE KEY-----"
KEY_END = "-----END " + "RSA PRIVATE KEY-----"
KEY_BODY = "MIIEpAIBAAKCAQEA" + ALNUM[:40]


def ph(kind: str) -> str:
    return f"[REDACTED:{kind}]"


# ---------------------------------------- plan validation: fake secrets in a real log

# (raw line number, text appended to that line, expected appended text after redaction)
INLINE_SECRETS = [
    (150, f" AWS_ACCESS_KEY_ID={AWS_KEY}", f" AWS_ACCESS_KEY_ID={ph('aws_key')}"),
    (151, f" aws_secret_access_key = {AWS_SECRET}", f" aws_secret_access_key = {ph('aws_secret')}"),
    (160, f" Authorization: token {GH_CLASSIC}", f" Authorization: token {ph('github_token')}"),
    (161, f" GH_TOKEN={GH_FINE}", f" GH_TOKEN={ph('github_token')}"),
    (162, f" app token {GH_SERVER} used", f" app token {ph('github_token')} used"),
    (200, f" Bearer {JWT}", f" Bearer {ph('jwt')}"),
    (
        210,
        " url=postgresql://petclinic:hunter2@db:5432/petclinic",
        f" url=postgresql://{ph('connection_string')}@db:5432/petclinic",
    ),
    (
        211,
        " -Dspring.datasource.password=s3cr3t!",
        f" -Dspring.datasource.password={ph('password')}",
    ),
    (212, ' DB_PASSWORD="two words"', f" DB_PASSWORD={ph('password')}"),
]
KEY_LINES = (230, 231, 232)  # line contents replaced by a 3-line private key


def _dirty_and_expected(raw: str) -> tuple[str, str, set[int]]:
    lines = raw.split("\n")
    dirty, expected = list(lines), list(lines)
    for n, add, add_redacted in INLINE_SECRETS:
        dirty[n - 1] = lines[n - 1] + add
        expected[n - 1] = lines[n - 1] + add_redacted
    for n, content in zip(KEY_LINES, [KEY_BEGIN, KEY_BODY, KEY_END], strict=True):
        ts = lines[n - 1][:29]  # "2026-11-08T13:32:23.0000000Z "
        assert ts.endswith("Z ")
        dirty[n - 1] = ts + content
        expected[n - 1] = ts + ph("private_key")
    touched = {n for n, _, _ in INLINE_SECRETS} | set(KEY_LINES)
    return "\n".join(dirty), "\n".join(expected), touched


@pytest.fixture(scope="module")
def real_log() -> str:
    return (CASES_DIR / "case-01" / "maven.log").read_text(encoding="utf-8")


def test_fake_secrets_in_real_log_are_all_redacted(real_log: str) -> None:
    dirty, expected, _ = _dirty_and_expected(real_log)
    result = redact_with_report(dirty)
    assert result.text == expected
    for secret in (AWS_KEY, AWS_SECRET, GH_CLASSIC, GH_SERVER, GH_FINE, JWT, KEY_BODY, "hunter2"):
        assert secret not in result.text
    assert result.counts == {
        "aws_key": 1,
        "aws_secret": 1,
        "github_token": 3,
        "jwt": 1,
        "connection_string": 1,
        "password": 2,
        "private_key": 1,
    }


def test_nothing_else_changed(real_log: str) -> None:
    dirty, _, touched = _dirty_and_expected(real_log)
    before, after = dirty.split("\n"), redact(dirty).split("\n")
    assert len(before) == len(after)
    changed = {i for i, (a, b) in enumerate(zip(before, after, strict=True), 1) if a != b}
    assert changed == touched


def test_running_twice_gives_same_output(real_log: str) -> None:
    dirty, _, _ = _dirty_and_expected(real_log)
    once = redact(dirty)
    assert redact(once) == once
    assert redact_with_report(once).total == 0


@pytest.mark.parametrize(
    "log", sorted(CASES_DIR.glob("*/*.log")), ids=lambda p: p.parent.name + "/" + p.name
)
def test_real_logs_have_no_false_positives(log: Path) -> None:
    raw = log.read_text(encoding="utf-8")
    assert redact(raw) == raw


def test_works_on_preprocessed_excerpt(real_log: str) -> None:
    dirty, _, _ = _dirty_and_expected(real_log)
    rendered = preprocess(dirty).render()
    cleaned = redact(rendered)
    assert GH_CLASSIC not in cleaned and KEY_BODY not in cleaned
    assert len(cleaned.split("\n")) == len(rendered.split("\n"))


# ------------------------------------------------------------- unit: each pattern


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"key {AWS_KEY}.", f"key {ph('aws_key')}."),
        ("ASIA" + "Y34FZKBOKMUTVV7A", ph("aws_key")),
        (
            f'"aws_secret_access_key": "{AWS_SECRET}"',
            f'"aws_secret_access_key": "{ph("aws_secret")}"',
        ),
        (f"AWS_SECRET_ACCESS_KEY={AWS_SECRET}", f"AWS_SECRET_ACCESS_KEY={ph('aws_secret')}"),
        (f"x={GH_CLASSIC}", f"x={ph('github_token')}"),
        (f"{GH_FINE}", ph("github_token")),
        (JWT, ph("jwt")),
        (
            "mongodb+srv://u:p%40ss@cluster0.x.net/db",
            f"mongodb+srv://{ph('connection_string')}@cluster0.x.net/db",
        ),
        (
            "jdbc:postgresql://db:5432/app?user=app&password=s3cr3t&ssl=true",
            f"jdbc:postgresql://db:5432/app?user=app&password={ph('password')}&ssl=true",
        ),
        ("--password=abc", f"--password={ph('password')}"),
        ("PASSWORD = 'quoted value'", f"PASSWORD = {ph('password')}"),
        ("gpg.passphrase=xyz", f"gpg.passphrase={ph('password')}"),
    ],
)
def test_redacts(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_private_key_without_timestamps_and_types() -> None:
    for kind in ("RSA ", "EC ", "OPENSSH ", ""):
        block = f"-----BEGIN {kind}PRIVATE KEY-----\n{KEY_BODY}\n-----END {kind}PRIVATE KEY-----"
        assert redact(f"a\n{block}\nb") == "a\n" + "\n".join([ph("private_key")] * 3) + "\nb"


def test_truncated_private_key_redacts_begin_and_base64_lines() -> None:
    text = f"x\n{KEY_BEGIN}\n{KEY_BODY}\n{KEY_BODY}\n[INFO] next"
    assert redact(text) == "x\n" + "\n".join([ph("private_key")] * 3) + "\n[INFO] next"


@pytest.mark.parametrize(
    "text",
    [
        "  server-password: GITHUB_TOKEN",  # setup-java input: a variable *name*
        "  token: ***",  # already masked by GitHub
        "AUTHORIZATION: basic ***",
        "https://x-access-token:***@github.com/o/r",
        "password=${DB_PASSWORD}",
        "password=$DB_PASSWORD",
        "PWD=/home/runner/work/petclinic",  # env var, not a password
        "git@github.com:pathmanabanptech/petclinic-ci-failures.git",
        "https://repo.maven.apache.org/maven2/org/x/1.0/x-1.0.pom",
        "ghp_short",
        "AKIAIOSFODNN7EXAMPL",  # 15 chars after prefix: too short
        "eyJhbGciOi.notajwt",
        "Downloaded from central: https://repo.maven.apache.org/maven2/",
        "[REDACTED:password] password=[REDACTED:password]",
    ],
)
def test_leaves_non_secrets_alone(text: str) -> None:
    assert redact(text) == text


def test_preserves_line_count_everywhere() -> None:
    text = f"{KEY_BEGIN}\r\n{KEY_BODY}\r\n{KEY_END}\nnext"
    assert redact(text).count("\n") == text.count("\n")
