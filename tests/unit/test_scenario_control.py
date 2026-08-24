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
    apply_seed_variant,
    compose,
    reset_scenario,
    runs_root,
    start_scenario,
    validated_run_paths,
)

FAMILY = "configuration_change"
FAMILIES = [
    "configuration_change",
    "downstream_timeout_retry_amplification",
    "resource_pool_saturation",
    "ambiguous_telemetry",
]
FAMILY_SYMPTOM = {
    "configuration_change": "ELEVATED_ERRORS",
    "downstream_timeout_retry_amplification": "ELEVATED_ERRORS_AND_LATENCY",
    "resource_pool_saturation": "ELEVATED_ERRORS_AND_LATENCY",
    "ambiguous_telemetry": "ELEVATED_ERRORS_AND_LATENCY",
}
FAMILY_ROOT_CAUSE = {
    "configuration_change": "CONFIG_CHANGE",
    "downstream_timeout_retry_amplification": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION",
    "resource_pool_saturation": "RESOURCE_POOL_SATURATION",
    "ambiguous_telemetry": "UNDETERMINED",
}
FAMILY_DISPOSITION = {
    "configuration_change": "DIAGNOSED",
    "downstream_timeout_retry_amplification": "DIAGNOSED",
    "resource_pool_saturation": "DIAGNOSED",
    "ambiguous_telemetry": "INSUFFICIENT_EVIDENCE",
}
REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway project root holding the real checked-in scenario definitions."""
    scenarios = tmp_path / "lab" / "scenarios"
    scenarios.mkdir(parents=True)
    for family in FAMILIES:
        shutil.copy(
            REPOSITORY / "lab" / "scenarios" / f"{family}.json",
            scenarios / f"{family}.json",
        )
    monkeypatch.setattr(scenario_control, "REQUEST_PAUSE_SECONDS", 0.0)
    monkeypatch.setattr(
        scenario_control, "call_gateway", gateway_reading_config(tmp_path)
    )
    return tmp_path


def use_family_gateway(
    monkeypatch: pytest.MonkeyPatch, root: Path, family: str
) -> None:
    monkeypatch.setattr(scenario_control, "call_gateway", GATEWAYS[family](root))


def read_active_config(root: Path) -> dict[str, object]:
    incident = active_incident_file(root).read_text(encoding="utf-8").strip()
    config: dict[str, object] = json.loads(
        (runs_root(root) / incident / "lab" / "config.json").read_text(encoding="utf-8")
    )
    return config


def test_validated_run_paths_refuses_a_symlink_loop(tmp_path: Path) -> None:
    incident_id = "loop"
    target = runs_root(tmp_path) / incident_id
    target.parent.mkdir(parents=True)
    target.symlink_to(target)

    with pytest.raises(LabError) as excinfo:
        validated_run_paths(tmp_path, incident_id)

    assert excinfo.value.reason_code is LabReasonCode.INCIDENT_NOT_FOUND


def gateway_reading_config(root: Path) -> Callable[[], int]:
    """A stand-in gateway that fails exactly when the faulted setting is in place.

    It reads the same file the real orders service reads, so the test proves the
    controller flips the configuration before it drives the fault traffic.
    """

    def call() -> int:
        config = read_active_config(root)
        return 500 if config.get("require_order_token") else 200

    return call


def gateway_reading_response_delay(root: Path) -> Callable[[], int]:
    """Mirrors orders: any injected inventory delay times the request out."""

    def call() -> int:
        config = read_active_config(root)
        return 504 if config.get("response_delay_seconds", 0) else 200

    return call


def gateway_reading_pool_capacity(root: Path) -> Callable[[], int]:
    """Mirrors LeakyPool: a per-incident count that never releases a slot."""
    counts: dict[str, int] = {}

    def call() -> int:
        incident = active_incident_file(root).read_text(encoding="utf-8").strip()
        config = read_active_config(root)
        counts[incident] = counts.get(incident, 0) + 1
        capacity = config.get("pool_capacity")
        if capacity is not None and counts[incident] > capacity:
            return 500
        return 200

    return call


def gateway_reading_pool_or_delay(root: Path) -> Callable[[], int]:
    """The ambiguous family: either active knob can make a request fail."""
    counts: dict[str, int] = {}

    def call() -> int:
        incident = active_incident_file(root).read_text(encoding="utf-8").strip()
        config = read_active_config(root)
        counts[incident] = counts.get(incident, 0) + 1
        capacity = config.get("pool_capacity")
        pool_exhausted = capacity is not None and counts[incident] > capacity
        delayed = bool(config.get("response_delay_seconds", 0))
        return 500 if pool_exhausted or delayed else 200

    return call


GATEWAYS: dict[str, Callable[[Path], Callable[[], int]]] = {
    "configuration_change": gateway_reading_config,
    "downstream_timeout_retry_amplification": gateway_reading_response_delay,
    "resource_pool_saturation": gateway_reading_pool_capacity,
    "ambiguous_telemetry": gateway_reading_pool_or_delay,
}


def stored_incident(project: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(project) / incident_id / "incident.json").read_text(
        encoding="utf-8"
    )
    return StoredIncident.model_validate_json(text)


@pytest.mark.parametrize("family", FAMILIES)
def test_starting_a_scenario_leaves_a_complete_run_directory(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    use_family_gateway(monkeypatch, project, family)

    incident_id = start_scenario(project, family, "development")

    run = runs_root(project) / incident_id
    assert len(incident_id) == 32
    assert (run / "logs").is_dir()
    assert (run / "changes.json").is_file()
    assert (run / "topology.json").is_file()
    assert (run / "incident.json").is_file()
    assert (run / "evaluator" / "expected.json").is_file()
    assert active_incident_file(project).read_text(encoding="utf-8") == incident_id


@pytest.mark.parametrize("family", FAMILIES)
def test_the_fault_is_applied_after_the_healthy_baseline(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    use_family_gateway(monkeypatch, project, family)

    incident_id = start_scenario(project, family, "development")

    incident = stored_incident(project, incident_id)
    assert incident.packet.symptom.value == FAMILY_SYMPTOM[family]
    payload = dict(incident.evidence[0].payload)
    assert payload["failed_requests"] > 0


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


@pytest.mark.parametrize("family", FAMILIES)
def test_the_expected_outcome_stays_on_the_evaluator_side(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    use_family_gateway(monkeypatch, project, family)

    incident_id = start_scenario(project, family, "development")

    expected = json.loads(
        (runs_root(project) / incident_id / "evaluator" / "expected.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected["root_cause"] == FAMILY_ROOT_CAUSE[family]
    assert expected["disposition"] == FAMILY_DISPOSITION[family]
    assert expected["seed"] == "development"


@pytest.mark.parametrize("family", FAMILIES)
def test_a_lab_that_never_worked_stops_before_faulting(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    monkeypatch.setattr(scenario_control, "call_gateway", lambda: 503)

    with pytest.raises(LabError) as refused:
        start_scenario(project, family, "development")

    assert refused.value.reason_code is LabReasonCode.BASELINE_NOT_HEALTHY


@pytest.mark.parametrize("family", FAMILIES)
def test_a_fault_that_changes_nothing_is_reported(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    monkeypatch.setattr(scenario_control, "call_gateway", lambda: 200)

    with pytest.raises(LabError) as refused:
        start_scenario(project, family, "development")

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


@pytest.mark.parametrize("family", FAMILIES)
def test_reset_deletes_the_run_and_never_the_results(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    use_family_gateway(monkeypatch, project, family)
    incident_id = start_scenario(project, family, "development")
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


def test_apply_seed_variant_is_identity_without_seed_variants() -> None:
    definition = {
        "changes": [{"service": "orders", "summary": "x", "offset_seconds": -10}]
    }

    result = apply_seed_variant(definition, "development")

    assert result is definition


def test_apply_seed_variant_is_identity_for_an_unlisted_seed() -> None:
    definition = {
        "changes": [{"service": "orders", "summary": "x", "offset_seconds": -10}],
        "seed_variants": {"development": {"change_offsets": [-20]}},
    }

    result = apply_seed_variant(definition, "evaluation")

    assert result is definition


def test_apply_seed_variant_replaces_offsets_and_merges_config_overrides() -> None:
    definition = {
        "changes": [
            {"service": "orders", "summary": "a", "offset_seconds": -10},
            {"service": "inventory", "summary": "b", "offset_seconds": -20},
        ],
        "faulted_config": {"pool_capacity": 9},
        "seed_variants": {
            "evaluation": {
                "change_offsets": [-30, -40],
                "faulted_config_overrides": {"pool_capacity": 7},
            }
        },
    }

    result = apply_seed_variant(definition, "evaluation")

    assert [entry["offset_seconds"] for entry in result["changes"]] == [-30, -40]
    assert result["changes"][0]["service"] == "orders"
    assert result["faulted_config"] == {"pool_capacity": 7}


def test_apply_seed_variant_requires_matching_offset_and_change_counts() -> None:
    definition = {
        "changes": [{"service": "orders", "summary": "a", "offset_seconds": -10}],
        "seed_variants": {"development": {"change_offsets": [-10, -20]}},
    }

    with pytest.raises(ValueError):
        apply_seed_variant(definition, "development")
