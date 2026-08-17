"""The downstream service: it answers stock lookups."""

from service import LabService, RouteResult

inventory = LabService(name="inventory", port=8082, path="/inventory")


def handle(request_id: str) -> RouteResult:
    inventory.log("info", "request_received", request_id, endpoint="/inventory")
    return 200, {"sku": "widget-1", "available": 42}, "success"


if __name__ == "__main__":
    inventory.serve(handle)
