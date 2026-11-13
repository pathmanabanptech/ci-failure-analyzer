"""SqliteMemoryStore: Phase 1 stub.

Memory is implemented in Phase 3 together with the confirmation flow (G12: store a
diagnosis only after a 👍 reaction or a merged fix with a green build). Until then
nothing can be confirmed, so ``store`` refuses loudly and ``recall`` returns no
hints. That keeps the pipeline wired against the MemoryStore Protocol without ever
storing an unconfirmed diagnosis.
"""

from __future__ import annotations

from pathlib import Path

from adapters.base import MemoryQuery, MemoryRecord


class SqliteMemoryStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = path

    def store(self, record: MemoryRecord) -> None:
        raise NotImplementedError(
            "memory storage arrives in Phase 3 with the G12 confirmation flow"
        )

    def recall(self, query: MemoryQuery) -> list[MemoryRecord]:
        return []
