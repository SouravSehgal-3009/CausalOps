"""The downstream service: it answers stock lookups."""

import time

from service import LabService, RouteResult, read_lab_config

inventory = LabService(name="inventory", port=8082, path="/inventory")

DELAY_SETTING = "response_delay_seconds"


def handle(request_id: str) -> RouteResult:
    inventory.log("info", "request_received", request_id, endpoint="/inventory")
    configuration = read_lab_config(inventory.incident_id())
    delay = float(configuration.get(DELAY_SETTING, 0))
    if delay > 0:
        time.sleep(delay)
    return 200, {"sku": "widget-1", "available": 42}, "success"


if __name__ == "__main__":
    inventory.serve(handle)
