"""The two new evidence-budget-curve seeds' required-evidence predicates
and family distinguishability, against the real lab.

`lab/scenarios/*.json` gained `evaluation_b`/`evaluation_c` seed variants
(the 12-incident corpus, `causalops.evaluate_cli.EVALUATION_SEEDS`) beside
the pre-existing `evaluation` seed. Two claims must hold for both new seeds,
the same two `evaluation`'s own fixed corpus already proves elsewhere
(`test_ambiguous_telemetry_predicate_is_earnable.py`,
`test_incident_manifest_fidelity.py`):

1. Every family's own `expected.predicates` is still reachable through the
   real registered tool backend (`run_logs_check`/`run_changes_check`), not
   just present in a raw fixture file.
2. The families a log-event signature could confuse
   (`resource_pool_saturation` vs `ambiguous_telemetry`'s pool-exhaustion
   half, `downstream_timeout_retry_amplification` vs `ambiguous_telemetry`'s
   timeout half) stay observably distinguishable under each new seed's own
   fault magnitudes, not just under `evaluation`'s.
"""

from pathlib import Path

import pytest

from causalops.domain import StoredIncident, ToolOutcome
from causalops.scenario_control import reset_scenario, runs_root, start_scenario
from causalops.telemetry import RunPaths, run_changes_check, run_logs_check
from causalops.tools import ListRecentChangesArguments, LogFilter, QueryLogsArguments

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
NEW_SEEDS = ["evaluation_b", "evaluation_c"]


def stored_incident(root: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(root) / incident_id / "incident.json").read_text(encoding="utf-8")
    return StoredIncident.model_validate_json(text)


def event_codes_for(seed: str, family: str) -> str:
    """Runs `family` under `seed`, returns the `query_logs` `event_codes`
    string for its `orders` service over the incident's own scope window --
    the same real backend call `run_metric_check`'s sibling checks use,
    never a direct file read."""
    incident_id = start_scenario(REPOSITORY, family, seed)
    try:
        incident = stored_incident(REPOSITORY, incident_id)
        scope = incident.scope
        paths = RunPaths(root=runs_root(REPOSITORY) / incident_id)
        arguments = QueryLogsArguments(
            log_filter=LogFilter.ERRORS_ONLY,
            service="orders",
            window_start=scope.started_at,
            window_end=scope.ended_at,
            row_limit=40,
        )
        outcome = run_logs_check(arguments, paths)
        assert outcome.outcome is ToolOutcome.EXECUTED
        event_codes = outcome.payload["event_codes"]
        assert isinstance(event_codes, str)
        return event_codes
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()


@pytest.mark.parametrize("seed", NEW_SEEDS)
def test_ambiguous_telemetry_shows_both_fault_signals(seed: str) -> None:
    """Mirrors `test_ambiguous_telemetry_shows_both_fault_signals_under_the_
    evaluation_seed` in `test_incident_manifest_fidelity.py`, for each new
    seed's own `pool_capacity`/`response_delay_seconds` overrides -- both
    `pool_exhausted` and `upstream_timeout` must genuinely appear, not just
    one of the two, or the family collapses into a clone of one of its
    siblings under this seed."""
    event_codes = event_codes_for(seed, "ambiguous_telemetry")

    assert "pool_exhausted" in event_codes
    assert "upstream_timeout" in event_codes


@pytest.mark.parametrize("seed", NEW_SEEDS)
def test_resource_pool_saturation_shows_only_pool_exhaustion(seed: str) -> None:
    """The distinguishability control against `ambiguous_telemetry`'s own
    pool-exhaustion signal: `resource_pool_saturation` never configures
    `response_delay_seconds`, so `upstream_timeout` must never appear
    regardless of which seed's `pool_capacity` override is in effect."""
    event_codes = event_codes_for(seed, "resource_pool_saturation")

    assert "pool_exhausted" in event_codes
    assert "upstream_timeout" not in event_codes


@pytest.mark.parametrize("seed", NEW_SEEDS)
def test_downstream_timeout_retry_amplification_shows_only_timeouts(seed: str) -> None:
    """The distinguishability control against `ambiguous_telemetry`'s own
    timeout signal: this family's `pool_capacity` (50, never overridden by
    either new seed) stays far above any seed's cumulative traffic, so
    `pool_exhausted` must never appear."""
    event_codes = event_codes_for(seed, "downstream_timeout_retry_amplification")

    assert "upstream_timeout" in event_codes
    assert "pool_exhausted" not in event_codes


@pytest.mark.parametrize("seed", NEW_SEEDS)
def test_configuration_change_predicate_is_reachable_through_list_recent_changes(
    seed: str,
) -> None:
    """`configuration_change`'s own predicate is a `list_recent_changes`
    summary match, not a log-event signature -- proven through the real
    `run_changes_check` backend, the same way the log-based families above
    are proven through `run_logs_check`."""
    incident_id = start_scenario(REPOSITORY, "configuration_change", seed)
    try:
        incident = stored_incident(REPOSITORY, incident_id)
        scope = incident.scope
        paths = RunPaths(root=runs_root(REPOSITORY) / incident_id)
        arguments = ListRecentChangesArguments(
            service="orders",
            window_start=scope.started_at,
            window_end=scope.ended_at,
        )

        outcome = run_changes_check(arguments, paths)

        assert outcome.outcome is ToolOutcome.EXECUTED
        summaries = outcome.payload["summaries"]
        assert isinstance(summaries, str)
        assert "require_order_token" in summaries
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()
