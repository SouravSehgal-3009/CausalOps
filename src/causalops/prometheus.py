"""The metric backend: registered templates, one range query, and a checked answer.

Prometheus is the only tool backend that talks to the network, and its response is
untrusted, so it is validated into typed samples before anything else touches it.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

from pydantic import BaseModel, ConfigDict, JsonValue

from causalops.domain import (
    CheckOutcome,
    EvidenceKind,
    IncidentScope,
    ReasonCode,
    ToolOutcome,
)
from causalops.evidence import executed_check, failed_check, trim_to_bytes
from causalops.tools import MetricTemplate, QueryMetricArguments

MAX_METRIC_SAMPLES = 60
METRIC_STEP_SECONDS = 15
DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"

# A service name reaches PromQL as a label value. Policy already checks it against
# the incident allowlist; this is the second lock on the same door.
SAFE_SERVICE_NAME = re.compile(r"^[a-z][a-z0-9-]{0,30}$")

METRIC_QUERIES: dict[MetricTemplate, str] = {
    MetricTemplate.GATEWAY_ERROR_RATE: (
        'sum(rate(causalops_requests_total{{service="{service}",'
        'incident="{incident}",outcome="error"}}[1m]))'
    ),
    MetricTemplate.GATEWAY_LATENCY_P95: (
        "histogram_quantile(0.95, sum by (le) (rate("
        'causalops_request_latency_seconds_bucket{{service="{service}",'
        'incident="{incident}"}}[1m])))'
    ),
    MetricTemplate.DOWNSTREAM_TIMEOUT_RATE: (
        'sum(rate(causalops_requests_total{{service="{service}",'
        'incident="{incident}",outcome="timeout"}}[1m]))'
    ),
    MetricTemplate.RESOURCE_POOL_IN_USE: (
        'max(causalops_pool_in_use{{service="{service}",incident="{incident}"}})'
    ),
}


class MetricSample(BaseModel):
    """One point of a range query, after the untrusted response has been checked."""

    model_config = ConfigDict(frozen=True)

    at: float
    value: float


def read_sample(pair: JsonValue) -> MetricSample | None:
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    moment, reading = pair
    if not isinstance(moment, int | float) or not isinstance(
        reading, str | int | float
    ):
        return None
    try:
        return MetricSample(at=float(moment), value=float(reading))
    except ValueError:
        return None


def parse_samples(answered: JsonValue) -> list[MetricSample] | None:
    """Turn a Prometheus range response into typed samples, or None if unreadable.

    Section 7 treats telemetry as untrusted, so the response is checked into a shape
    this module understands rather than indexed as loose JSON.
    """
    if not isinstance(answered, dict) or answered.get("status") != "success":
        return None
    data = answered.get("data")
    if not isinstance(data, dict):
        return None
    series = data.get("result")
    if not isinstance(series, list):
        return None
    if not series:
        return []
    first = series[0]
    if not isinstance(first, dict):
        return None
    values = first.get("values")
    if not isinstance(values, list):
        return []
    samples = [read_sample(pair) for pair in values]
    return [sample for sample in samples if sample is not None]


def query_prometheus(
    base_url: str, promql: str, start: datetime, end: datetime, timeout: int
) -> list[MetricSample] | None:
    query = urllib.parse.urlencode(
        {
            "query": promql,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "step": f"{METRIC_STEP_SECONDS}s",
        }
    )
    url = f"{base_url.rstrip('/')}/api/v1/query_range?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        answered: JsonValue = json.loads(response.read().decode("utf-8"))
    return parse_samples(answered)


def fetch_metric_samples(
    base_url: str,
    promql: str,
    arguments: QueryMetricArguments,
    timeout: int,
    source: str,
) -> list[MetricSample] | CheckOutcome:
    """Run the range query, mapping each failure mode to its own reason code.

    Returns the samples on success, or the failed check outcome to return as-is.
    """
    try:
        samples = query_prometheus(
            base_url, promql, arguments.window_start, arguments.window_end, timeout
        )
    except TimeoutError:
        return failed_check(
            EvidenceKind.METRIC,
            source,
            ToolOutcome.TIMEOUT,
            ReasonCode.TOOL_TIMEOUT,
            "Prometheus did not answer in time",
        )
    except (urllib.error.URLError, OSError):
        return failed_check(
            EvidenceKind.METRIC,
            source,
            ToolOutcome.UNAVAILABLE,
            ReasonCode.TOOL_UNAVAILABLE,
            "Prometheus is not reachable",
        )
    except json.JSONDecodeError:
        samples = None
    if samples is None:
        return failed_check(
            EvidenceKind.METRIC,
            source,
            ToolOutcome.ERROR,
            ReasonCode.TOOL_ERROR,
            "Prometheus returned something this tool cannot read",
        )
    return samples


def run_metric_check(
    arguments: QueryMetricArguments, scope: IncidentScope, base_url: str, timeout: int
) -> CheckOutcome:
    source = "query_metric"
    if not SAFE_SERVICE_NAME.match(arguments.service):
        return failed_check(
            EvidenceKind.METRIC,
            source,
            ToolOutcome.ERROR,
            ReasonCode.TOOL_ERROR,
            "that service name cannot be used in a query",
        )
    promql = METRIC_QUERIES[arguments.template].format(
        service=arguments.service, incident=scope.incident_id
    )
    started = time.monotonic()
    fetched = fetch_metric_samples(base_url, promql, arguments, timeout, source)
    if isinstance(fetched, CheckOutcome):
        return fetched
    kept = fetched[:MAX_METRIC_SAMPLES]
    rows: list[JsonValue] = [[sample.at, sample.value] for sample in kept]
    payload: dict[str, JsonValue] = {
        "template": arguments.template.value,
        "service": arguments.service,
        "sample_count": len(kept),
        "max_value": max((sample.value for sample in kept), default=0.0),
        "truncated": len(fetched) > len(kept),
    }
    payload = trim_to_bytes(payload, "samples", rows, "sample_count")
    # Post-freeze review. `count_key="sample_count"` was added in this
    # same round to keep the PAYLOAD honest post-trim (Unit 3b-4 addendum,
    # C3's fix, applied here too) -- but this summary string kept reading
    # `len(kept)`, the PRE-trim count, four lines below where the payload
    # was already fixed. Reproduced live: payload said `sample_count: 371`
    # while this string still said "900 samples." Same bug, same fix
    # pattern already applied twice this round (`run_logs_check`,
    # `run_changes_check`).
    #
    # Round 4 review, F3. `max_value` (above) has the SAME pre-trim-
    # aggregate shape -- computed from `kept` before `trim_to_bytes` ever
    # runs, so a sample popped by byte trimming could still be reported as
    # the peak even though it no longer appears in `payload["samples"]`.
    # Not reachable with this suite's real data (`MetricSample.at`/
    # `.value` are both floats, always small -- byte trimming never fires
    # below the `MAX_METRIC_SAMPLES` cap already applied above), but fixed
    # anyway: the owner ruled this closes the CLASS, not just the
    # reachable instance. Rebuilt from `payload["samples"]`, the POST-trim
    # list.
    #
    # Round 6 review. The comment here used to also claim rebuilding
    # `max_value` "can only shrink or hold... never grow the payload back
    # over budget" -- true of the NUMERIC value (every kept sample was
    # already in the pre-trim set, so the max can only fall or stay the
    # same), but not a safe claim about JSON BYTE LENGTH: `900.0`
    # serializes to 5 bytes, `0.30000000000000004` to 19, so a smaller
    # float is not guaranteed to serialize to fewer bytes. The real safety
    # argument is unreachability, stated above, not this false general
    # property -- the same shape of overclaiming this round's own review
    # was about.
    kept_samples = payload["samples"]
    assert isinstance(kept_samples, list)
    kept_values: list[float] = []
    for kept_sample in kept_samples:
        if isinstance(kept_sample, list) and len(kept_sample) == 2:
            value = kept_sample[1]
            if isinstance(value, int | float):
                kept_values.append(float(value))
    payload["max_value"] = max(kept_values, default=0.0)
    # Round 6 review, the P1. `payload["truncated"]` already carries whether
    # this check's data was cut, but only `run_logs_check`/`run_changes_check`
    # rendered that into their summary string -- the only field of a
    # `CheckOutcome` `prompts.py`'s `render_context` puts in front of the
    # model. A truncated metric window read as complete with no signal the
    # true peak might lie outside what was kept.
    truncated_note = " (truncated)" if payload["truncated"] else ""
    return executed_check(
        EvidenceKind.METRIC,
        source,
        f"{arguments.template.value} for {arguments.service}: "
        f"{payload['sample_count']} samples, peak {payload['max_value']}"
        f"{truncated_note}",
        payload,
        started,
    )
