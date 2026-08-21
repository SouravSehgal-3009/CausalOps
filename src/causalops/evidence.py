"""Incident-scoped evidence: opaque IDs, content hashes, ordering, quotas, digests.

The bounds a tool result must respect live here too, beside the context quotas, so
that every backend shapes its result the same way.
"""

import json
import time
from collections.abc import Iterable
from datetime import datetime
from hashlib import sha256
from uuid import uuid4

from pydantic import JsonValue

from causalops.domain import (
    CheckOutcome,
    Evidence,
    EvidenceKind,
    ReasonCode,
    ToolOutcome,
)

MAX_RESULT_BYTES = 12 * 1024

# Fixed per-kind quotas keep the model context bounded and the same size for the
# same evidence, whatever order it arrived in.
CONTEXT_QUOTAS: dict[EvidenceKind, int] = {
    EvidenceKind.SYMPTOM: 2,
    EvidenceKind.TOPOLOGY: 2,
    EvidenceKind.METRIC: 3,
    EvidenceKind.LOG: 3,
    EvidenceKind.CHANGE: 3,
}

# `context_evidence` below does an unguarded `CONTEXT_QUOTAS[record.kind]`
# lookup -- the only unguarded per-kind lookup in `src/` (confirmed by
# review during Unit 3a). A new `EvidenceKind` member added without a
# matching entry here would not fail at the point it was added; it would
# crash the first investigation that ever produced evidence of that kind.
# Unit 3a removed the *motive* for that (no `EvidenceKind.RUNBOOK` --
# `RunbookCheckOutcome` has no `kind` field to populate one with) but not
# the *capability* -- nothing stops a future kind from being added here
# without its quota. This assertion is that guard, checked at import time
# rather than left to whichever investigation happens to hit the gap first.
assert set(CONTEXT_QUOTAS) == set(EvidenceKind), (
    "CONTEXT_QUOTAS is missing an entry for at least one EvidenceKind member "
    "-- context_evidence()'s per-kind lookup would crash on that kind's "
    "first evidence record"
)


def new_opaque_id() -> str:
    return uuid4().hex


def digest_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def content_hash(payload: dict[str, JsonValue]) -> str:
    return digest_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def build_evidence(
    incident_id: str,
    kind: EvidenceKind,
    source: str,
    observed_at: datetime,
    summary: str,
    payload: dict[str, JsonValue],
    receipt_id: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=new_opaque_id(),
        incident_id=incident_id,
        kind=kind,
        source=source,
        observed_at=observed_at,
        summary=summary,
        payload=payload,
        receipt_id=receipt_id,
        content_hash=content_hash(payload),
    )


def failed_check(
    kind: EvidenceKind, source: str, outcome: ToolOutcome, reason: ReasonCode, note: str
) -> CheckOutcome:
    return CheckOutcome(
        outcome=outcome, kind=kind, source=source, summary=note, reason_code=reason
    )


def executed_check(
    kind: EvidenceKind,
    source: str,
    summary: str,
    payload: dict[str, JsonValue],
    started: float,
) -> CheckOutcome:
    return CheckOutcome(
        outcome=ToolOutcome.EXECUTED,
        kind=kind,
        source=source,
        summary=summary,
        payload=payload,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def fits(payload: dict[str, JsonValue]) -> bool:
    return len(json.dumps(payload).encode("utf-8")) <= MAX_RESULT_BYTES


def trim_to_bytes(
    payload: dict[str, JsonValue], rows_key: str, rows: list[JsonValue]
) -> dict[str, JsonValue]:
    """Drop rows from the end until the whole result fits the byte bound."""
    kept = list(rows)
    payload[rows_key] = kept
    while kept and not fits(payload):
        kept.pop()
        payload[rows_key] = kept
        payload["truncated"] = True
    return payload


class EvidenceStore:
    """Holds one incident's evidence. Nothing from another incident enters or leaves."""

    def __init__(self, incident_id: str) -> None:
        self.incident_id = incident_id
        self.records: dict[str, Evidence] = {}

    def add(self, evidence: Evidence) -> None:
        if evidence.incident_id != self.incident_id:
            raise ValueError("evidence belongs to another incident")
        self.records[evidence.evidence_id] = evidence

    def unknown_ids(self, cited: Iterable[str]) -> tuple[str, ...]:
        """Cited IDs this incident cannot account for, which is a forged citation."""
        return tuple(cited_id for cited_id in cited if cited_id not in self.records)

    def ordered(self) -> tuple[Evidence, ...]:
        return tuple(
            sorted(
                self.records.values(),
                key=lambda record: (
                    record.observed_at,
                    record.kind,
                    record.evidence_id,
                ),
            )
        )

    def context_evidence(self) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
        """Evidence for the model context, plus a marker for anything left out."""
        kept: list[Evidence] = []
        used: dict[EvidenceKind, int] = {}
        dropped: dict[EvidenceKind, int] = {}
        for record in self.ordered():
            seen = used.get(record.kind, 0)
            if seen < CONTEXT_QUOTAS[record.kind]:
                kept.append(record)
                used[record.kind] = seen + 1
                continue
            dropped[record.kind] = dropped.get(record.kind, 0) + 1
        markers = tuple(
            f"[truncated: {count} more {kind.value} records omitted]"
            for kind, count in sorted(dropped.items())
        )
        return tuple(kept), markers
