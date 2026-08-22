import json
from pathlib import Path

import pytest
from fake_incident import FIXTURE_DIR

from causalops.models import (
    ModelRequest,
    ReplayFixtureError,
    ReplayReasoningModel,
    Stage,
)

FIXTURES = (
    "valid_diagnosis.json",
    "correct_abstention.json",
    "repair_then_valid.json",
    "malformed_output.json",
    "lab_diagnosis.json",
)


def make_request(
    stage: Stage, system_text: str = "system", context_text: str = "context"
) -> ModelRequest:
    """`run_id`/`graph_phase`/`model_turn`/`context_digest` (Unit 3b-2) are
    inert to `ReplayReasoningModel.respond`, which matches fixtures on
    `request.stage` alone -- placeholder values below, not meaningful ones,
    present only because the fields are required."""
    return ModelRequest(
        stage=stage,
        system_text=system_text,
        context_text=context_text,
        run_id="run-1",
        graph_phase="INVESTIGATE",
        model_turn=0,
        context_digest="digest",
    )


def ask(model: ReplayReasoningModel, stage: Stage) -> dict[str, object]:
    return dict(model.respond(make_request(stage)).content)


def test_every_checked_in_fixture_loads() -> None:
    for name in FIXTURES:
        model = ReplayReasoningModel(FIXTURE_DIR / name)

        assert model.responses["initial_plan"]


def test_a_diagnosis_fixture_scripts_all_three_stages() -> None:
    model = ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")

    assert ask(model, Stage.INITIAL_PLAN)["proposal"]
    assert ask(model, Stage.HYPOTHESIS_UPDATE)["proposal"]
    assert ask(model, Stage.FINAL_ASSESSMENT)["disposition"] == "DIAGNOSED"


def test_an_abstention_fixture_stops_and_abstains() -> None:
    model = ReplayReasoningModel(FIXTURE_DIR / "correct_abstention.json")
    ask(model, Stage.INITIAL_PLAN)

    assert ask(model, Stage.HYPOTHESIS_UPDATE)["stop_reason"]
    assert ask(model, Stage.FINAL_ASSESSMENT)["root_cause"] == "UNDETERMINED"


def test_asking_the_same_stage_twice_returns_the_next_scripted_response() -> None:
    model = ReplayReasoningModel(FIXTURE_DIR / "repair_then_valid.json")

    assert ask(model, Stage.INITIAL_PLAN) == {"hypotheses": []}
    assert ask(model, Stage.INITIAL_PLAN)["proposal"]


def test_a_malformed_fixture_never_becomes_valid() -> None:
    model = ReplayReasoningModel(FIXTURE_DIR / "malformed_output.json")

    assert ask(model, Stage.INITIAL_PLAN) == {"hypotheses": []}
    assert ask(model, Stage.INITIAL_PLAN) == {"hypotheses": []}


def test_running_past_the_script_is_an_error_rather_than_a_guess() -> None:
    model = ReplayReasoningModel(FIXTURE_DIR / "malformed_output.json")
    ask(model, Stage.INITIAL_PLAN)
    ask(model, Stage.INITIAL_PLAN)

    with pytest.raises(ReplayFixtureError, match="no response 3"):
        ask(model, Stage.INITIAL_PLAN)


def test_the_caller_fills_in_what_a_static_fixture_cannot_know(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps({"responses": {"initial_plan": [{"id": "{{incident_id}}"}]}}),
        encoding="utf-8",
    )
    model = ReplayReasoningModel(fixture, substitutions={"incident_id": "abc123"})

    assert ask(model, Stage.INITIAL_PLAN) == {"id": "abc123"}


def test_a_placeholder_outside_the_closed_list_is_refused(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"responses": {}}), encoding="utf-8")

    with pytest.raises(ReplayFixtureError, match="unknown placeholders"):
        ReplayReasoningModel(fixture, substitutions={"anything_at_all": "x"})


def test_a_script_can_cite_evidence_it_only_saw_in_the_context(tmp_path: Path) -> None:
    """The one thing a static file cannot do for itself: name a runtime ID."""
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "responses": {
                    "final_assessment": [{"cite": "{{evidence_from_last_check}}"}]
                }
            }
        ),
        encoding="utf-8",
    )
    model = ReplayReasoningModel(
        fixture, substitutions={"symptom_evidence_id": "packet-1"}
    )
    context = "\n".join(
        [
            "<untrusted-telemetry>",
            "- packet-1 [SYMPTOM] 2026-08-16T10:00:00+00:00 from alert: errors",
            "- from-check-1 [LOG] 2026-08-16T10:05:00+00:00 from query_logs: rows",
            "</untrusted-telemetry>",
        ]
    )

    response = model.respond(make_request(Stage.FINAL_ASSESSMENT, context_text=context))

    assert response.content == {"cite": "from-check-1"}


def test_citing_a_check_that_never_ran_fails_loudly(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "responses": {
                    "final_assessment": [{"cite": "{{evidence_from_last_check}}"}]
                }
            }
        ),
        encoding="utf-8",
    )
    model = ReplayReasoningModel(
        fixture, substitutions={"symptom_evidence_id": "packet-1"}
    )
    context = "- packet-1 [SYMPTOM] 2026-08-16T10:00:00+00:00 from alert: errors"

    with pytest.raises(ReplayFixtureError, match="no check evidence"):
        model.respond(make_request(Stage.FINAL_ASSESSMENT, context_text=context))


def test_fixture_names_describe_the_script_not_the_cause() -> None:
    """Section 4 keeps fixture keys opaque about the incident cause."""
    causes = ("config", "timeout", "retry", "pool", "saturation")

    for name in FIXTURES:
        assert not any(cause in name for cause in causes)
