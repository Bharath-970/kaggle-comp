from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    score: float
    structural_hash: str
    functional_fingerprint: str
    metadata: dict[str, str] = field(default_factory=dict)


class DiversityBank:
    """Keeps strong candidates while preserving structural and behavioral diversity."""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._records: list[CandidateRecord] = []

    @property
    def records(self) -> tuple[CandidateRecord, ...]:
        return tuple(self._records)

    def add(self, record: CandidateRecord) -> bool:
        # Reject dominated duplicates by either structural or functional identity.
        for existing in self._records:
            if (
                existing.structural_hash == record.structural_hash
                or existing.functional_fingerprint == record.functional_fingerprint
            ):
                if record.score <= existing.score:
                    return False
                self._records.remove(existing)
                break

        self._records.append(record)
        self._records.sort(key=lambda r: r.score, reverse=True)
        if len(self._records) > self.max_size:
            self._records = self._records[: self.max_size]
        return True
