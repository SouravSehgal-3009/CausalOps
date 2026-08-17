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
UPSTREAM_TIMEOUT_SECONDS = 2.0
TOKEN_SETTING = "require_order_token"

orders = LabService(name="orders", port=8081, path="/orders")


def handle(request_id: str) -> RouteResult:
    configuration = read_lab_config(orders.incident_id())
    orders.log(
        "info",
        "request_received",
        request_id,
        endpoint="/orders",
        settings_loaded=len(configuration),
    )
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
    try:
        with urllib.request.urlopen(
            INVENTORY_URL, timeout=UPSTREAM_TIMEOUT_SECONDS
        ) as response:
            stock: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
        orders.log("error", "upstream_timeout", request_id, upstream="inventory")
        return 504, {"error": "inventory did not answer"}, "timeout"
    orders.log("info", "request_served", request_id, upstream="inventory")
    return 200, {"order": request_id, "stock": stock}, "success"


if __name__ == "__main__":
    orders.serve(handle)
