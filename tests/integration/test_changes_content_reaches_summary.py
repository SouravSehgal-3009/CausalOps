"""Real change content in the summary the model reads,
against the real lab.

Marked `docker` for the same reason as `test_pool_metric_gauge_name.py`: every
non-docker test in this codebase writes its own `changes.json` fixture by
hand, so none of them can catch a mismatch between what the lab's own
scenario controller actually writes to that file and what `run_changes_check`
reads back from it. This test calls `run_changes_check` directly against a
real, freshly started scenario's real `changes.json`, not a hand-built
fixture.
"""

from pathlib import Path

import pytest

from causalops.domain import StoredIncident, ToolOutcome
from causalops.scenario_control import (
    reset_scenario,
    run_paths,
    runs_root,
    start_scenario,
)
from causalops.telemetry import run_changes_check
from causalops.tools import ListRecentChangesArguments

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILY = "configuration_change"


def stored_incident(root: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(root) / incident_id / "incident.json").read_text(encoding="utf-8")
    return StoredIncident.model_validate_json(text)


def test_the_real_change_content_reaches_the_model_visible_summary() -> None:
    """`configuration_change.json`'s own change entry names
    `require_order_token` in its `summary` field -- the exact string this
    family's ground-truth predicate requires (`expected.predicates[0].value`
    in the scenario JSON). Before this fix, `run_changes_check`'s returned
    `.summary` never carried this text at all, only the bare change count --
    `render_context` (`prompts.py`) puts only `.summary` in front of the
    model, so a model that correctly called `list_recent_changes` against a
    real running lab still could not see the content that answers the
    incident."""
    incident_id = start_scenario(REPOSITORY, FAMILY, "evaluation")

    try:
        incident = stored_incident(REPOSITORY, incident_id)
        scope = incident.scope
        paths = run_paths(REPOSITORY, incident_id)

        arguments = ListRecentChangesArguments(
            service="orders",
            window_start=scope.started_at,
            window_end=scope.ended_at,
        )
        outcome = run_changes_check(arguments, paths)

        assert outcome.outcome is ToolOutcome.EXECUTED
        assert "require_order_token" in outcome.summary
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()
