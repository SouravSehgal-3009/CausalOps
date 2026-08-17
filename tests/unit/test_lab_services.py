"""The lab services, exercised in process. Nothing here starts a container."""

import json
from pathlib import Path

import pytest
import service
from prometheus_client import generate_latest
from service import MAX_FIELD_LENGTH, MAX_LOG_FIELDS, LabService, bounded_fields

INCIDENT = "3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(service, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(
        service, "ACTIVE_INCIDENT_FILE", tmp_path / "active-incident.txt"
    )
    return tmp_path


def activate(runs: Path, incident: str = INCIDENT) -> None:
    (runs / "active-incident.txt").write_text(incident, encoding="utf-8")


def log_lines(runs: Path, name: str) -> list[dict[str, object]]:
    path = runs / INCIDENT / "logs" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_log_record_carries_what_section_three_asks_for(runs: Path) -> None:
    activate(runs)
    orders = LabService(name="orders", port=8081, path="/orders")

    orders.log("error", "config_rejected_request", "req-1", config_key="setting")

    record = log_lines(runs, "orders")[0]
    assert set(record) == {"at", "request_id", "service", "severity", "event", "fields"}
    assert record["service"] == "orders"
    assert record["fields"] == {"config_key": "setting"}


def test_nothing_is_logged_when_no_scenario_is_active(runs: Path) -> None:
    gateway = LabService(name="gateway", port=8080, path="/api/orders")

    gateway.log("info", "request_received", "req-1")

    assert not (runs / INCIDENT).exists()


def test_log_fields_are_bounded() -> None:
    crowded = {f"key{index}": "v" for index in range(MAX_LOG_FIELDS + 5)}
    crowded["long"] = "x" * (MAX_FIELD_LENGTH + 50)

    bounded = bounded_fields(crowded)

    assert len(bounded) <= MAX_LOG_FIELDS
    assert all(len(str(value)) <= MAX_FIELD_LENGTH for value in bounded.values())


def test_metrics_carry_the_active_incident_label(runs: Path) -> None:
    activate(runs)
    inventory = LabService(name="inventory", port=8082, path="/inventory")

    inventory.observe("success", 0.01)

    exposed = generate_latest(inventory.registry).decode("utf-8")
    assert f'incident="{INCIDENT}"' in exposed
    assert 'service="inventory"' in exposed
    assert "causalops_pool_in_use" in exposed


def test_configuration_is_read_from_the_active_run(runs: Path) -> None:
    activate(runs)
    config_file = runs / INCIDENT / "lab" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"require_order_token": True}), encoding="utf-8")

    assert service.read_lab_config(INCIDENT) == {"require_order_token": True}
    assert service.read_lab_config("no-such-incident") == {}


def test_orders_turns_requests_away_when_the_setting_is_on(runs: Path) -> None:
    import orders

    activate(runs)
    config_file = runs / INCIDENT / "lab" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"require_order_token": True}), encoding="utf-8")

    status, body, outcome = orders.handle("req-1")

    assert (status, outcome) == (500, "error")
    assert "token" in str(body)
    events = [record["event"] for record in log_lines(runs, "orders")]
    assert "config_rejected_request" in events


def test_the_gateway_reports_an_unreachable_upstream(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gateway

    activate(runs)
    monkeypatch.setattr(gateway, "ORDERS_URL", "http://127.0.0.1:1/orders")

    status, _, outcome = gateway.handle("req-1")

    assert (status, outcome) == (504, "timeout")
    events = [record["event"] for record in log_lines(runs, "gateway")]
    assert "upstream_timeout" in events
