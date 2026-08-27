import json
from datetime import timedelta

import pytest
from fake_incident import INCIDENT_ID, WINDOW_START, packet_evidence
from pydantic import JsonValue

from causalops.domain import Evidence, EvidenceKind
from causalops.evidence import (
    CONTEXT_QUOTAS,
    MAX_RESULT_BYTES,
    EvidenceStore,
    build_evidence,
    content_hash,
    digest_text,
    fits,
    new_opaque_id,
    trim_to_bytes,
)


# Kept for defense in depth alongside `evidence.py`'s own module-level `assert`
# -- that assertion runs once at import time and would be stripped under
# `python -O`; this test runs every time `pytest` does and cannot be silently
# skipped by an interpreter flag. Runbook retrieval made the gap this guards
# real: a `RunbookCheckOutcome` has no `kind` field (so nothing motivates a new
# `EvidenceKind.RUNBOOK` member any more), but nothing stops a future
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


def test_trim_to_bytes_pops_rows_and_keeps_the_count_field_in_sync() -> None:
    rows: list[JsonValue] = [{"text": "x" * 2000} for _ in range(20)]
    payload: dict[str, JsonValue] = {"row_count": len(rows), "truncated": False}

    result = trim_to_bytes(payload, "rows", rows, "row_count")

    assert fits(result)
    assert result["truncated"] is True
    assert result["row_count"] == len(result["rows"])  # type: ignore[arg-type]
    assert result["row_count"] < 20


def test_trim_to_bytes_falls_back_to_shrinking_a_scalar_field() -> None:
    """Before this fix, once `rows` reached zero the
    trimming loop stopped even if `fits(payload)` was still `False` --
    reproducing `run_changes_check`'s bug directly against `trim_to_bytes`
    itself, with no `RunPaths`/telemetry scaffolding needed: a single
    oversized scalar field (`"summary"`, standing in for `run_changes_
    check`'s real `summaries`) is bigger than `MAX_RESULT_BYTES` all by
    itself, so no amount of row-popping can ever make `fits(payload)` true
    on its own -- rows are still popped first (unconditionally, the same
    as before this fix; see `trim_to_bytes`'s own docstring for why an
    earlier version of this fix that tried to skip that step broke on
    `run_changes_check`'s real shape), and the scalar fallback then
    finishes the job."""
    oversized_summary = "y" * (MAX_RESULT_BYTES + 2000)
    payload: dict[str, JsonValue] = {
        "row_count": 3,
        "summary": oversized_summary,
        "truncated": False,
    }
    rows: list[JsonValue] = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

    result = trim_to_bytes(payload, "rows", rows, "row_count")

    assert fits(result)
    assert result["truncated"] is True
    assert result["row_count"] == len(result["rows"])  # type: ignore[arg-type]
    # Shrunk, not silently dropped -- `"summary"` still exists and is a
    # (much smaller) prefix of the original, not a lie about what the
    # oversized field contained.
    assert isinstance(result["summary"], str)
    assert 0 < len(result["summary"]) < len(oversized_summary)
    assert oversized_summary.startswith(result["summary"])
