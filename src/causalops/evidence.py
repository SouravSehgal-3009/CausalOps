"""Incident-scoped evidence: opaque IDs, content hashes, ordering, quotas, digests.

The bounds a tool result must respect live here too, beside the context quotas, so
that every backend shapes its result the same way.
"""

import json
import time
from collections.abc import Iterable, Mapping
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

# SYMPTOM/TOPOLOGY never scale with the evidence budget:
# SYMPTOM is seeded once at incident creation, never tool-produced, so more
# executed-tool budget can never produce a second one. TOPOLOGY's single
# `get_topology` call takes no varying arguments, so a second call is always
# refused as `DUPLICATE_PROPOSAL` by the policy layer regardless of budget --
# that cap comes from policy, not from this quota, and this quota must not
# be read as the reason a second topology record never appears.
_FIXED_CONTEXT_QUOTAS: dict[EvidenceKind, int] = {
    EvidenceKind.SYMPTOM: 2,
    EvidenceKind.TOPOLOGY: 2,
}
# METRIC/LOG/CHANGE are the three kinds an executed check can actually keep
# producing more of as the evidence budget widens -- `derive_context_quotas`
# below scales each of these with `Budgets.executed_tools` rather than
# leaving them fixed, so a wider evidence budget is not silently capped by a
# context window still sized for the narrower default.
_TOOL_DERIVED_QUOTA_KINDS: tuple[EvidenceKind, ...] = (
    EvidenceKind.METRIC,
    EvidenceKind.LOG,
    EvidenceKind.CHANGE,
)

# `context_evidence` below does an unguarded `quotas[record.kind]` lookup --
# the only unguarded per-kind lookup in `src/`. A new `EvidenceKind` member
# added to neither the fixed set above nor the tool-derived set above would
# not fail at the point it was added; it would crash the first investigation
# that ever produced evidence of that kind. Runbook retrieval removed the
# *motive* for that (no `EvidenceKind.RUNBOOK` -- `RunbookCheckOutcome` has
# no `kind` field to populate one with) but not the *capability* -- nothing
# stops a future kind from being added to `EvidenceKind` without also being
# added here. This assertion is that guard, checked at import time rather
# than left to whichever investigation happens to hit the gap first; it
# covers `derive_context_quotas`'s output for every possible
# `executed_tools` value, not just the one default this file used to pin,
# since the two sets it checks are what the derivation is built from, not a
# concrete dict that could itself go stale.
assert set(_FIXED_CONTEXT_QUOTAS) | set(_TOOL_DERIVED_QUOTA_KINDS) == set(
    EvidenceKind
), (
    "a new EvidenceKind member is neither fixed nor tool-derived here -- "
    "context_evidence()'s per-kind lookup would crash on that kind's first "
    "evidence record"
)


def derive_context_quotas(executed_tools: int) -> dict[EvidenceKind, int]:
    """The per-kind evidence quota for a run whose `Budgets.executed_tools`
    is `executed_tools` -- keeps the model context bounded and the same size
    for the same evidence, whatever order it arrived in, while letting a
    wider evidence-gathering budget actually see more of what it gathered.
    METRIC/LOG/CHANGE scale as `executed_tools + 1` (room for one more
    record than the number of checks a run could ever execute, since a
    single check can produce more than one evidence record of the same kind
    over its history); SYMPTOM/TOPOLOGY stay fixed at 2 regardless -- see
    `_FIXED_CONTEXT_QUOTAS`'s own comment for why."""
    quotas = dict(_FIXED_CONTEXT_QUOTAS)
    for kind in _TOOL_DERIVED_QUOTA_KINDS:
        quotas[kind] = executed_tools + 1
    return quotas


# The quota an unbudgeted caller (a test, a direct `EvidenceStore.
# context_evidence()` call with no `Budgets` in scope) sees -- equal to
# `derive_context_quotas(2)`, `Budgets().executed_tools`'s own default, so
# behaviour for every such caller is unchanged from before quotas became
# budget-derived.
CONTEXT_QUOTAS: dict[EvidenceKind, int] = derive_context_quotas(2)


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
    payload: dict[str, JsonValue],
    rows_key: str,
    rows: list[JsonValue],
    count_key: str,
) -> dict[str, JsonValue]:
    """Drop rows from the front until the whole result fits the byte bound.
    `count_key` (`"row_count"`/`"change_count"`/`"edge_count"`/
    `"sample_count"`, one per caller) is kept equal to `len(kept)`
    throughout, not just set once from the pre-trim count and left stale --
    an owner reading the payload after trimming must see a count that
    actually matches what `payload[rows_key]` holds.

    Pops from the front (`kept.pop(0)`), not the
    end, so that whenever `rows` arrives in oldest-to-newest order --
    already true for `prometheus.py`'s samples and `telemetry.py`'s log
    rows, and true for `run_changes_check`'s changes once it sorts them
    before calling here -- the *newest* rows survive. Every real
    observation in this lab sits near the tail of the incident window
    (the fault only manifests close to `window_end`), so dropping from the
    end used to discard exactly the data-bearing region first. This is a
    property of the SHARED function, so `run_topology_check`'s two calls
    (`edges`, `services`) get it too -- harmless there, since a static
    topology snapshot has no chronological order for "newest" to mean
    anything about.

    Dropping rows alone cannot help when the
    payload's real weight is a SCALAR field assembled from those rows
    before this function ever sees them (`run_changes_check`'s
    `summaries`, joined from every matched change's own summary text,
    before `trim_to_bytes` is called at all). Once `rows` is fully
    emptied, the loop below has nothing left to drop -- previously
    the function returned there regardless of whether `fits(payload)` was
    actually `True`, silently shipping an over-budget result. This
    function's contract now is `fits(payload)` on return WHENEVER
    the overage is text this function can shrink -- once the row list is
    exhausted, it falls back to shrinking the largest remaining
    string-valued field (by half, repeatedly) instead of stopping. That
    guarantee is not unconditional: if every string field is already
    empty and the payload still does not fit, the loop below gives up
    rather than looping forever (see its own comment at the `break`) --
    the remaining weight is fixed JSON structure this function has no text
    left to shrink, a documented known limit, not a silent one.
    `"truncated"` is set on every path that changes what the payload
    actually holds, list-trimming or scalar-shrinking alike -- the signal
    an owner needs to know the result no longer represents everything that
    matched, never silently.

    Rows are ALWAYS popped first, unconditionally, even when a caller's
    oversized scalar (not the rows) turns out to be the real cause: an
    earlier version of this fix tried to skip popping when a cheap
    up-front check judged the rows "innocent," and broke on exactly
    `run_changes_check`'s real shape -- each kept row there is the raw
    entry dict, which still carries its OWN full-size `summary` text
    inside it, a second, undetected copy of the same oversized content the
    top-level `summaries` scalar holds. Skipping the row pop left that
    copy sitting inside `payload[rows_key]`, where the scalar-shrinking
    fallback below (which only inspects top-level string values) could
    never reach it. Popping rows first removes both copies in one step;
    the scalar fallback then only has to handle whatever is left over."""
    kept = list(rows)
    payload[rows_key] = kept
    payload[count_key] = len(kept)
    while kept and not fits(payload):
        kept.pop(0)
        payload[rows_key] = kept
        payload[count_key] = len(kept)
        payload["truncated"] = True
    while not fits(payload):
        widest_key = max(
            (key for key, value in payload.items() if isinstance(value, str) and value),
            key=lambda key: len(str(payload[key])),
            default=None,
        )
        if widest_key is None:
            # Every string field is already empty and the payload still
            # does not fit -- the remaining weight is fixed JSON structure
            # (keys, ints, bools), not text this function can shrink.
            # Documented as a known limit rather than looped forever.
            break
        text = str(payload[widest_key])
        payload[widest_key] = text[: len(text) // 2]
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

    def context_evidence(
        self, quotas: Mapping[EvidenceKind, int] = CONTEXT_QUOTAS
    ) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
        """Evidence for the model context, plus a marker for anything left
        out. `quotas` defaults to the unbudgeted `CONTEXT_QUOTAS` above;
        `graph.py`'s `_render_stage_request` -- the one production caller --
        always passes `derive_context_quotas(budgets.executed_tools)`
        instead, so a real run's quota tracks its own evidence budget."""
        kept: list[Evidence] = []
        used: dict[EvidenceKind, int] = {}
        dropped: dict[EvidenceKind, int] = {}
        for record in self.ordered():
            seen = used.get(record.kind, 0)
            if seen < quotas[record.kind]:
                kept.append(record)
                used[record.kind] = seen + 1
                continue
            dropped[record.kind] = dropped.get(record.kind, 0) + 1
        markers = tuple(
            f"[truncated: {count} more {kind.value} records omitted]"
            for kind, count in sorted(dropped.items())
        )
        return tuple(kept), markers
