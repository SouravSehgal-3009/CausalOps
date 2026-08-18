"""The middle service: it applies its configuration, then calls inventory."""

import json
import os
import urllib.error
import urllib.request
from typing import Any

from service import LabService, RouteResult, read_lab_config

INVENTORY_URL = os.environ.get(
    "CAUSALOPS_INVENTORY_URL", "http://inventory:8082/inventory"
)
INVENTORY_MAX_ATTEMPTS = 3
INVENTORY_ATTEMPT_TIMEOUT_SECONDS = 0.4
TOKEN_SETTING = "require_order_token"
POOL_CAPACITY_SETTING = "pool_capacity"

orders = LabService(name="orders", port=8081, path="/orders")


class LeakyPool:
    """A per-incident slot counter that never releases what it hands out.

    This is not real concurrency tracking; it is just enough state to make a
    resource pool observably exhaust itself as fault traffic accumulates.
    It assumes sequential, single-threaded callers, matching how
    `drive_traffic` in `scenario_control.py` drives traffic today;
    concurrent use would need a lock around `acquire`.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def acquire(self, incident_id: str) -> int:
        self.counts[incident_id] = self.counts.get(incident_id, 0) + 1
        return self.counts[incident_id]


pool = LeakyPool()


def call_inventory() -> dict[str, Any]:
    with urllib.request.urlopen(
        INVENTORY_URL, timeout=INVENTORY_ATTEMPT_TIMEOUT_SECONDS
    ) as response:
        stock: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return stock


def handle(request_id: str) -> RouteResult:
    incident_id = orders.incident_id()
    configuration = read_lab_config(incident_id)
    orders.log(
        "info",
        "request_received",
        request_id,
        endpoint="/orders",
        settings_loaded=len(configuration),
    )
    in_use = pool.acquire(incident_id)
    orders.set_pool_in_use(in_use)
    capacity = configuration.get(POOL_CAPACITY_SETTING)
    if capacity is not None and in_use > int(capacity):
        orders.log(
            "error",
            "pool_exhausted",
            request_id,
            config_key=POOL_CAPACITY_SETTING,
            detail=f"{in_use} slots requested against a capacity of {capacity}",
        )
        return 500, {"error": "resource pool exhausted"}, "error"
    if configuration.get(TOKEN_SETTING, False):
        # The caller never sends a token, so this setting turns every request away.
        orders.log(
            "error",
            "config_rejected_request",
            request_id,
            config_key=TOKEN_SETTING,
            detail="request carried no order token",
        )
        return 500, {"error": "order token required"}, "error"
    for _ in range(INVENTORY_MAX_ATTEMPTS):
        try:
            stock = call_inventory()
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            continue
        orders.log("info", "request_served", request_id, upstream="inventory")
        return 200, {"order": request_id, "stock": stock}, "success"
    orders.log("error", "upstream_timeout", request_id, upstream="inventory")
    return 504, {"error": "inventory did not answer"}, "timeout"


if __name__ == "__main__":
    orders.serve(handle)
