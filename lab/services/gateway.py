"""The public entry point of the lab: it calls orders and reports what happened."""

import json
import os
import urllib.error
import urllib.request
from typing import Any

from service import LabService, RouteResult

ORDERS_URL = os.environ.get("CAUSALOPS_ORDERS_URL", "http://orders:8081/orders")
UPSTREAM_TIMEOUT_SECONDS = 2.0

gateway = LabService(name="gateway", port=8080, path="/api/orders")


def handle(request_id: str) -> RouteResult:
    gateway.log("info", "request_received", request_id, endpoint="/api/orders")
    try:
        with urllib.request.urlopen(
            ORDERS_URL, timeout=UPSTREAM_TIMEOUT_SECONDS
        ) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        gateway.log(
            "error",
            "upstream_error",
            request_id,
            upstream="orders",
            status=error.code,
        )
        return 502, {"error": "orders rejected the request"}, "error"
    except (TimeoutError, urllib.error.URLError):
        gateway.log("error", "upstream_timeout", request_id, upstream="orders")
        return 504, {"error": "orders did not answer"}, "timeout"
    gateway.log("info", "request_served", request_id, upstream="orders")
    return 200, body, "success"


if __name__ == "__main__":
    gateway.serve(handle)
