import json
from datetime import timedelta

import pytest
from fake_incident import INCIDENT_ID, WINDOW_START, packet_evidence
from pydantic import JsonValue

from causalops.domain import Evidence, EvidenceKind
from causalops.evidence import (
    CONTEXT_QUOTAS,
    EvidenceStore,
    build_evidence,
    content_hash,
    digest_text,
    new_opaque_id,
)


# Kept for defense in depth alongside `evidence.py`'s own module-level
# `assert` -- that assertion runs once at import time and would be stripped
# under `python -O`; this test runs every time `pytest` does and cannot be
# silently skipped by an interpreter flag. Unit 3a made the gap this guards
# real: a `RunbookCheckOutcome` has no `kind` field (so nothing motivates a
# new `EvidenceKind.RUNBOOK` member any more), but nothing stops a future
# `EvidenceKind` addition from landing without a matching quota, and
# `context_evidence()`'s `CONTEXT_QUOTAS[record.kind]` lookup below is
# unguarded -- confirmed the only such lookup in `src/` during review.
def test_context_quotas_covers_every_evidence_kind() -> None:
    assert set(CONTEXT_QUOTAS) == set(EvidenceKind)


def log_evidence(minute: int, incident_id: str = INCIDENT_ID) -> Evidence:
    payload: dict[str, JsonValue] = {"rows": minute}
    return build_evidence(
        incident_id=incident_id,
        kind=EvidenceKind.LOG,
        source="query_logs",
        observed_at=WINDOW_START + timedelta(minutes=minute),
        summary=f"{minute} timeout rows",
        payload=payload,
    )


def filled_store() -> EvidenceStore:
    store = EvidenceStore(INCIDENT_ID)
    for record in packet_evidence():
        store.add(record)
    return store


def test_an_opaque_id_carries_no_meaning() -> None:
    first, second = new_opaque_id(), new_opaque_id()

    assert first != second
    assert len(first) == 32
    assert first.isalnum()


def test_the_content_hash_covers_the_payload_only() -> None:
    payload: dict[str, JsonValue] = {"b": 2, "a": 1}

    assert content_hash(payload) == digest_text(
        json.dumps({"a": 1, "b": 2}, separators=(",", ":"))
    )
    assert content_hash(payload) == content_hash({"a": 1, "b": 2})
    assert content_hash(payload) != content_hash({"a": 1, "b": 3})


def test_built_evidence_hashes_its_own_payload() -> None:
    record = log_evidence(1)

    assert record.content_hash == content_hash(record.payload)


def test_evidence_from_another_incident_cannot_enter_the_store() -> None:
    store = filled_store()

    with pytest.raises(ValueError, match="another incident"):
        store.add(log_evidence(1, incident_id="some-other-incident"))


def test_an_unknown_citation_is_reported() -> None:
    store = filled_store()
    known = store.ordered()[0].evidence_id

    assert store.unknown_ids([known]) == ()
    assert store.unknown_ids([known, "made-up"]) == ("made-up",)


def test_ordering_is_stable_whatever_order_evidence_arrived_in() -> None:
    forwards = filled_store()
    backwards = filled_store()
    records = [log_evidence(3), log_evidence(1), log_evidence(2)]
    for record in records:
        forwards.add(record)
    for record in reversed(records):
        backwards.add(record)

    assert [record.evidence_id for record in forwards.ordered()] == [
        record.evidence_id for record in backwards.ordered()
    ]


def test_context_evidence_applies_a_quota_and_says_what_it_left_out() -> None:
    store = filled_store()
    extra = CONTEXT_QUOTAS[EvidenceKind.LOG] + 2
    for minute in range(extra):
        store.add(log_evidence(minute))

    kept, markers = store.context_evidence()

    logs_kept = [record for record in kept if record.kind is EvidenceKind.LOG]
    assert len(logs_kept) == CONTEXT_QUOTAS[EvidenceKind.LOG]
    assert markers == ("[truncated: 2 more LOG records omitted]",)


def test_context_evidence_adds_no_marker_when_nothing_is_dropped() -> None:
    kept, markers = filled_store().context_evidence()

    assert len(kept) == 2
    assert markers == ()
