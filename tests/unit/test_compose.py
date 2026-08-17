"""Section 12 asks for the container ceilings and the retention window to be tested.

Reading the compose file proves what the lab is declared to be without starting it.
"""

from pathlib import Path
from typing import Any

import yaml

LAB = Path(__file__).resolve().parents[2] / "lab"
CEILINGS = {
    "gateway": "128m",
    "orders": "192m",
    "inventory": "128m",
    "prometheus": "256m",
}


def compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        (LAB / "docker-compose.yml").read_text(encoding="utf-8")
    )
    return loaded


def test_every_container_declares_its_memory_ceiling() -> None:
    services = compose()["services"]

    assert set(services) == set(CEILINGS)
    for name, ceiling in CEILINGS.items():
        assert services[name]["mem_limit"] == ceiling


def test_prometheus_keeps_one_hour_of_history() -> None:
    command = compose()["services"]["prometheus"]["command"]

    assert "--storage.tsdb.retention.time=1h" in command


def test_prometheus_is_pinned_and_reads_the_project_scrape_config() -> None:
    prometheus = compose()["services"]["prometheus"]

    assert prometheus["image"].startswith("prom/prometheus:v")
    assert "./prometheus.yml:/etc/prometheus/prometheus.yml:ro" in prometheus["volumes"]


def test_the_services_share_the_run_directory() -> None:
    services = compose()["services"]

    for name in ("gateway", "orders", "inventory"):
        assert "../runs:/runs" in services[name]["volumes"]


def test_the_lab_listens_only_on_loopback() -> None:
    services = compose()["services"]

    published = [port for service in services.values() for port in service["ports"]]
    assert published
    for port in published:
        assert port.startswith("127.0.0.1:")


def test_prometheus_scrapes_all_three_services() -> None:
    scrape: dict[str, Any] = yaml.safe_load(
        (LAB / "prometheus.yml").read_text(encoding="utf-8")
    )

    targets = scrape["scrape_configs"][0]["static_configs"][0]["targets"]
    assert targets == ["gateway:8080", "orders:8081", "inventory:8082"]
