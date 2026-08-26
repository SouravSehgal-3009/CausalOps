"""The lab services, exercised in process. Nothing here starts a container."""

import json
from pathlib import Path

import pytest
import service
from prometheus_client import generate_latest
from service import MAX_FIELD_LENGTH, MAX_LOG_FIELDS, LabService, bounded_fields

INCIDENT = "3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c"
# A distinct incident ID for pool tests, since LeakyPool's counters are keyed by
# incident ID and never released, so reusing INCIDENT would leak state between tests.
POOL_INCIDENT = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(service, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(
        service, "ACTIVE_INCIDENT_FILE", tmp_path / "active-incident.txt"
    )
    return tmp_path


def activate(runs: Path, incident: str = INCIDENT) -> None:
    (runs / "active-incident.txt").write_text(incident, encoding="utf-8")


def log_lines(
    runs: Path, name: str, incident: str = INCIDENT
) -> list[dict[str, object]]:
    path = runs / incident / "logs" / f"{name}.jsonl"
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


def test_the_latency_histogram_has_boundaries_near_the_lab_s_real_range(
    runs: Path,
) -> None:
    """Lab-defect-fix Unit 2, W6. Proves the ladder is actually wired into
    the exposed `/metrics` text, not just declared -- `histogram_quantile`
    reads bucket boundaries (`le=` labels) off exactly this output. Checks
    for boundaries near both landmarks this ladder was derived from
    (`service.py`'s own `LATENCY_BUCKETS_SECONDS` comment): the ~1.2s
    timeout path and the 1.5s/2.0s delayed-success path -- not the
    `prometheus_client` default ladder's `1.0`-to-`2.5` gap, which is what
    put this lab's entire dynamic range inside one bucket in the first
    place."""
    activate(runs)
    orders = LabService(name="orders", port=8081, path="/orders")

    orders.observe("success", 1.21)

    exposed = generate_latest(orders.registry).decode("utf-8")
    assert 'le="1.2"' in exposed
    assert 'le="1.5"' in exposed
    assert 'le="2.0"' in exposed
    # The default ladder's own coarse jump is gone: nothing between 1.0 and
    # 2.5 should be missing a real cut point.
    assert 'le="1.0"' in exposed
    assert 'le="2.5"' in exposed


def test_configuration_is_read_from_the_active_run(runs: Path) -> None:
    activate(runs)
    config_file = runs / INCIDENT / "lab" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"require_order_token": True}), encoding="utf-8")

    assert service.read_lab_config(INCIDENT) == {"require_order_token": True}
    assert service.read_lab_config("no-such-incident") == {}


def test_read_lab_config_returns_empty_on_malformed_json(runs: Path) -> None:
    """Lab-defect-fix Unit 5, W12. A reader hitting `config.json` mid-write
    (before `write_json` was made atomic, or from any other partial write)
    must fail safe -- an empty configuration -- not raise `JSONDecodeError`
    and crash the request the lab service is handling."""
    config_path = runs / INCIDENT / "lab" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{"require_order_token": tru', encoding="utf-8")

    assert service.read_lab_config(INCIDENT) == {}


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


def test_orders_exhausts_its_pool_past_capacity(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import orders

    activate(runs, POOL_INCIDENT)
    config_file = runs / POOL_INCIDENT / "lab" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"pool_capacity": 2}), encoding="utf-8")
    monkeypatch.setattr(
        orders, "call_inventory", lambda: {"sku": "widget-1", "available": 42}
    )

    orders.handle("req-1")
    orders.handle("req-2")
    status, body, outcome = orders.handle("req-3")

    exposed = generate_latest(orders.orders.registry).decode("utf-8")
    assert "causalops_pool_in_use" in exposed
    assert (status, outcome) == (500, "error")
    assert "pool" in str(body)
    events = [record["event"] for record in log_lines(runs, "orders", POOL_INCIDENT)]
    assert "pool_exhausted" in events


def test_orders_retries_the_upstream_before_giving_up(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import orders

    activate(runs)
    monkeypatch.setattr(orders, "INVENTORY_URL", "http://127.0.0.1:1/inventory")
    attempts = {"count": 0}
    real_call = orders.call_inventory

    def counting_call() -> dict[str, object]:
        attempts["count"] += 1
        return real_call()

    monkeypatch.setattr(orders, "call_inventory", counting_call)

    status, _, outcome = orders.handle("req-1")

    assert (status, outcome) == (504, "timeout")
    assert attempts["count"] == orders.INVENTORY_MAX_ATTEMPTS
    events = [record["event"] for record in log_lines(runs, "orders")]
    assert "upstream_timeout" in events


def test_inventory_sleeps_for_the_configured_delay(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inventory

    activate(runs)
    config_file = runs / INCIDENT / "lab" / "config.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        json.dumps({"response_delay_seconds": 1.5}), encoding="utf-8"
    )
    slept: list[float] = []
    monkeypatch.setattr(inventory.time, "sleep", slept.append)

    status, _, outcome = inventory.handle("req-1")

    assert (status, outcome) == (200, "success")
    assert slept == [1.5]


def test_inventory_does_not_sleep_when_no_delay_is_configured(
    runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inventory

    activate(runs)
    slept: list[float] = []
    monkeypatch.setattr(inventory.time, "sleep", slept.append)

    status, _, outcome = inventory.handle("req-1")

    assert (status, outcome) == (200, "success")
    assert slept == []
