import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from causalops import scenario_control
from causalops.domain import RootCauseCode, StoredIncident
from causalops.scenario_control import (
    LabError,
    LabReasonCode,
    active_incident_file,
    compose,
    reset_scenario,
    runs_root,
    start_scenario,
)

FAMILY = "configuration_change"
REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway project root holding the real checked-in scenario definition."""
    scenarios = tmp_path / "lab" / "scenarios"
    scenarios.mkdir(parents=True)
    shutil.copy(
        REPOSITORY / "lab" / "scenarios" / f"{FAMILY}.json",
        scenarios / f"{FAMILY}.json",
    )
    monkeypatch.setattr(scenario_control, "REQUEST_PAUSE_SECONDS", 0.0)
    monkeypatch.setattr(
        scenario_control, "call_gateway", gateway_reading_config(tmp_path)
    )
    return tmp_path


def gateway_reading_config(root: Path) -> Callable[[], int]:
    """A stand-in gateway that fails exactly when the faulted setting is in place.

    It reads the same file the real orders service reads, so the test proves the
    controller flips the configuration before it drives the fault traffic.
    """

    def call() -> int:
        incident = active_incident_file(root).read_text(encoding="utf-8").strip()
        config = json.loads(
            (runs_root(root) / incident / "lab" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        return 500 if config.get("require_order_token") else 200

    return call


def stored_incident(project: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(project) / incident_id / "incident.json").read_text(
        encoding="utf-8"
    )
    return StoredIncident.model_validate_json(text)


def test_starting_a_scenario_leaves_a_complete_run_directory(project: Path) -> None:
    incident_id = start_scenario(project, FAMILY, "development")

    run = runs_root(project) / incident_id
    assert len(incident_id) == 32
    assert (run / "logs").is_dir()
    assert (run / "changes.json").is_file()
    assert (run / "topology.json").is_file()
    assert (run / "incident.json").is_file()
    assert (run / "evaluator" / "expected.json").is_file()
    assert active_incident_file(project).read_text(encoding="utf-8") == incident_id


def test_the_fault_is_applied_after_the_healthy_baseline(project: Path) -> None:
    incident_id = start_scenario(project, FAMILY, "development")

    config = json.loads(
        (runs_root(project) / incident_id / "lab" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    incident = stored_incident(project, incident_id)
    assert config["require_order_token"] is True
    assert incident.packet.symptom.value == "ELEVATED_ERRORS"
    payload = dict(incident.evidence[0].payload)
    assert payload["failed_requests"] == 10


def test_recorded_changes_fall_inside_the_recorded_window(project: Path) -> None:
    incident_id = start_scenario(project, FAMILY, "development")

    incident = stored_incident(project, incident_id)
    changes = json.loads(
        (runs_root(project) / incident_id / "changes.json").read_text(encoding="utf-8")
    )
    for change in changes:
        assert incident.scope.started_at.isoformat() <= change["at"]
        assert change["at"] <= incident.scope.ended_at.isoformat()


def test_nothing_the_investigator_can_read_names_the_family(project: Path) -> None:
    incident_id = start_scenario(project, FAMILY, "development")

    run = runs_root(project) / incident_id
    visible = [path for path in run.rglob("*") if "evaluator" not in path.parts]
    for path in visible:
        assert FAMILY not in path.name
        assert "config_change" not in path.name.lower()
    readable = "".join(
        path.read_text(encoding="utf-8") for path in visible if path.is_file()
    )
    assert FAMILY not in readable
    assert "development" not in readable
    for code in RootCauseCode:
        assert code.value not in readable


def test_the_expected_outcome_stays_on_the_evaluator_side(project: Path) -> None:
    incident_id = start_scenario(project, FAMILY, "development")

    expected = json.loads(
        (runs_root(project) / incident_id / "evaluator" / "expected.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected["root_cause"] == "CONFIG_CHANGE"
    assert expected["seed"] == "development"
    assert expected["predicates"]


def test_a_lab_that_never_worked_stops_before_faulting(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scenario_control, "call_gateway", lambda: 503)

    with pytest.raises(LabError) as refused:
        start_scenario(project, FAMILY, "development")

    assert refused.value.reason_code is LabReasonCode.BASELINE_NOT_HEALTHY


def test_a_fault_that_changes_nothing_is_reported(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scenario_control, "call_gateway", lambda: 200)

    with pytest.raises(LabError) as refused:
        start_scenario(project, FAMILY, "development")

    assert refused.value.reason_code is LabReasonCode.FAULT_NOT_OBSERVED


def test_only_one_scenario_may_be_active(project: Path) -> None:
    start_scenario(project, FAMILY, "development")

    with pytest.raises(LabError) as refused:
        start_scenario(project, FAMILY, "development")

    assert refused.value.reason_code is LabReasonCode.SCENARIO_ALREADY_ACTIVE


def test_an_unknown_family_is_refused(project: Path) -> None:
    with pytest.raises(LabError) as refused:
        start_scenario(project, "not_a_family", "development")

    assert refused.value.reason_code is LabReasonCode.UNKNOWN_FAMILY


def test_reset_deletes_the_run_and_never_the_results(project: Path) -> None:
    incident_id = start_scenario(project, FAMILY, "development")
    finalized = project / "results" / "investigations" / "inv-1"
    finalized.mkdir(parents=True)
    (finalized / "report.json").write_text("{}", encoding="utf-8")

    reset_scenario(project, incident_id)

    assert not (runs_root(project) / incident_id).exists()
    assert not active_incident_file(project).exists()
    assert (finalized / "report.json").read_text(encoding="utf-8") == "{}"


def test_reset_refuses_anything_that_is_not_an_incident(project: Path) -> None:
    for name in ("..", "../results", "unknown"):
        with pytest.raises(LabError) as refused:
            reset_scenario(project, name)
        assert refused.value.reason_code is LabReasonCode.INCIDENT_NOT_FOUND


def test_a_missing_docker_is_reported_rather_than_raised(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_docker(*arguments: object, **keywords: object) -> None:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(scenario_control.subprocess, "run", no_docker)

    with pytest.raises(LabError) as refused:
        compose(project, ["up", "-d"], 5)

    assert refused.value.reason_code is LabReasonCode.DOCKER_UNAVAILABLE
