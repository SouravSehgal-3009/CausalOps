"""The alert's own counts and the ambiguous/saturation log signatures, against
the real lab.

Marked `docker` for the same reason as `test_configuration_change.py`: these
assert against the real, live-scraped `logs/*.jsonl` a scenario run leaves
behind, not a fixture. `--seed evaluation` is used throughout because both
fixes checked here are about what the *evaluation* corpus looks like -- the
seed the paired evaluation actually runs under.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

from causalops.domain import StoredIncident
from causalops.scenario_control import reset_scenario, runs_root, start_scenario

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILIES = [
    "configuration_change",
    "downstream_timeout_retry_amplification",
    "resource_pool_saturation",
    "ambiguous_telemetry",
]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def stored_incident(root: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(root) / incident_id / "incident.json").read_text(encoding="utf-8")
    return StoredIncident.model_validate_json(text)


def gateway_events_in_window(
    root: Path, incident_id: str, start: datetime, end: datetime
) -> list[dict[str, object]]:
    """Every `logs/gateway.jsonl` row for this run whose `at` falls inside
    `[start, end]` -- the exact ground `build_incident`'s alert counts are
    generated from (`drive_traffic` drives the gateway, one HTTP round trip
    per row logged here), so this is the real-world check against the alert,
    not a re-derivation through the model-facing `query_logs` tool contract
    (which cannot see a successful request at all -- a deliberate design
    choice, not an oversight)."""
    rows = read_jsonl(runs_root(root) / incident_id / "logs" / "gateway.jsonl")
    in_window = []
    for row in rows:
        moment = row.get("at")
        assert isinstance(moment, str)
        at = datetime.fromisoformat(moment)
        if start <= at <= end:
            in_window.append(row)
    return in_window


@pytest.mark.parametrize("family", FAMILIES)
def test_alert_counts_match_the_real_gateway_log_over_the_whole_window(
    family: str,
) -> None:
    """`total_requests`/`failed_requests` must describe the whole
    recorded window (baseline + fault), not the fault phase alone -- proven
    against `logs/gateway.jsonl`, the same HTTP round trips `drive_traffic`
    counted to build the alert in the first place."""
    incident_id = start_scenario(REPOSITORY, family, "evaluation")

    try:
        incident = stored_incident(REPOSITORY, incident_id)
        alert_payload = dict(incident.evidence[0].payload)

        rows = gateway_events_in_window(
            REPOSITORY, incident_id, incident.scope.started_at, incident.scope.ended_at
        )
        total_requests = sum(
            1 for row in rows if row.get("event") == "request_received"
        )
        failed_requests = sum(
            1
            for row in rows
            if row.get("event") in ("upstream_error", "upstream_timeout")
        )
        served_requests = sum(1 for row in rows if row.get("event") == "request_served")

        # The self-check: if this fails, the counting rule above is wrong,
        # not the application code under test -- say so distinctly rather
        # than letting it read as the same failure as the assertions below.
        assert total_requests == failed_requests + served_requests, (
            "counting rule is broken: request_received rows should equal "
            "failed + served rows in logs/gateway.jsonl, independent of "
            "anything build_incident computed"
        )

        assert alert_payload["total_requests"] == total_requests, (
            "alert total_requests does not match the real gateway log over "
            "the whole incident window"
        )
        assert alert_payload["failed_requests"] == failed_requests
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()


def test_ambiguous_telemetry_shows_both_fault_signals_under_the_evaluation_seed() -> (
    None
):
    """Before this fix, the evaluation seed's `pool_capacity: 7`
    meant every fault request exhausted the pool before the retry loop that
    would ever hit `response_delay_seconds` even ran -- the family
    degenerated to a `resource_pool_saturation` clone with the opposite
    expected label. `pool_capacity: 13` splits the fault phase so both
    signals genuinely appear."""
    incident_id = start_scenario(REPOSITORY, "ambiguous_telemetry", "evaluation")

    try:
        rows = read_jsonl(runs_root(REPOSITORY) / incident_id / "logs" / "orders.jsonl")
        event_counts = Counter(row.get("event") for row in rows)
        # Exact counts, not just membership: this evaluation seed's fault
        # phase runs a fixed request count through single-threaded counter
        # arithmetic (never live timing), so the 5/5 split is deterministic,
        # not a race -- pinning it exactly proves both signals genuinely
        # appear at the split this fix intends, not merely that at least one
        # of each fired. A deliberate future `pool_capacity` retune for this
        # family should update these two numbers consciously, the same way
        # this project pins other deterministic literals elsewhere.
        assert event_counts["pool_exhausted"] == 5
        assert event_counts["upstream_timeout"] == 5
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()


def test_resource_pool_saturation_shows_only_pool_exhaustion() -> None:
    """The distinguishability control. `resource_pool_saturation`'s own
    evaluation-seed `pool_capacity` (7, unchanged by this fix) must keep
    producing pool exhaustion only -- proof that fixing `ambiguous_telemetry`
    did not blur the family it used to collapse into. `response_delay_seconds`
    is never configured for this family at all, so `upstream_timeout` cannot
    appear."""
    incident_id = start_scenario(REPOSITORY, "resource_pool_saturation", "evaluation")

    try:
        rows = read_jsonl(runs_root(REPOSITORY) / incident_id / "logs" / "orders.jsonl")
        events = {row.get("event") for row in rows}
        assert "pool_exhausted" in events
        assert "upstream_timeout" not in events
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()
