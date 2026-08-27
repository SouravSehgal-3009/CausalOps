"""The metric backend: registered templates, one range query, and a checked answer.

Prometheus is the only tool backend that talks to the network, and its response is
untrusted, so it is validated into typed samples before anything else touches it.
"""

import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, JsonValue

from causalops.domain import (
    CheckOutcome,
    EvidenceKind,
    IncidentScope,
    MetricSampleStatus,
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

# Lab-defect-fix Unit 2, W5. `GATEWAY_ERROR_RATE`/`DOWNSTREAM_TIMEOUT_RATE`
# narrow their `rate(...)` lookback from `[1m]` to `[30s]` -- confirmed
# against `TOOL_SELECTION_BIAS_FINDINGS.md` §8.5's own line citations
# (`prometheus.py:38-41`/`:47-50`, unchanged by this unit's earlier edits)
# that these two, and only these two, are the ones measured to understate
# a real fault's error rate ~6.5x: the fault band here is 4-15s, and a
# 1-minute rate window averages it against ~45s of nothing. `[30s]` at a
# 5s scrape still contains 6 samples (well above the 2-sample minimum
# `rate()` needs), cutting the dilution roughly in half, and W3's settle
# delay guarantees both samples the lookback needs are actually present.
# `GATEWAY_LATENCY_P95`'s own `rate(...[1m])` is deliberately UNCHANGED:
# it feeds `histogram_quantile`, a different computation (a quantile
# estimate, not a raw count), which W6's bucket-boundary fix addresses on
# its own terms -- narrowing its lookback too was never part of the §8.5
# finding this item fixes, and is not a decision this unit makes silently.
METRIC_QUERIES: dict[MetricTemplate, str] = {
    MetricTemplate.GATEWAY_ERROR_RATE: (
        'sum(rate(causalops_requests_total{{service="{service}",'
        'incident="{incident}",outcome="error"}}[30s]))'
    ),
    MetricTemplate.GATEWAY_LATENCY_P95: (
        "histogram_quantile(0.95, sum by (le) (rate("
        'causalops_request_latency_seconds_bucket{{service="{service}",'
        'incident="{incident}"}}[1m])))'
    ),
    MetricTemplate.DOWNSTREAM_TIMEOUT_RATE: (
        'sum(rate(causalops_requests_total{{service="{service}",'
        'incident="{incident}",outcome="timeout"}}[30s]))'
    ),
    # Fix F1 (revised). Queries `causalops_pool_attempts_per_capacity`: total
    # slot acquisition attempts for the incident divided by configured
    # capacity. This is honestly unbounded above -- it exceeds 1 once
    # attempts outstrip the pool, since a refused request is still a real
    # attempt -- so it is named for what it measures rather than implying a
    # bounded occupancy fraction the way "utilization" would. The old raw
    # `causalops_pool_in_use` counter is deliberately not kept as a second
    # registered template: that would preserve the exact trap this fix
    # exists to close (a model reading a monotonically growing counter and
    # mistaking it for pool occupancy).
    MetricTemplate.RESOURCE_POOL_ATTEMPTS_PER_CAPACITY: (
        "max(causalops_pool_attempts_per_capacity"
        '{{service="{service}",incident="{incident}"}})'
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
        at = float(moment)
        value = float(reading)
    except (ValueError, OverflowError):
        # Round 8 review, P2. `json.loads` produces a genuine Python `int`
        # for a large integer literal in a JSON response (no decimal point
        # or exponent, so `float()`'s string-parsing path -- which happily
        # rounds an oversized literal to inf rather than raising -- is
        # never involved). Converting a Python int that large to `float`
        # raises `OverflowError`, which `ValueError` alone does not catch --
        # so a single oversized integer sample used to escape this
        # function's own "unreadable field -> None, skip this row" contract
        # entirely, propagate through `graph.py`'s blanket exception
        # handler, and turn one bad sample into `FAILED_SAFE` for the whole
        # investigation.
        return None
    # `float()` parses "NaN", "Infinity", and an overflow literal like
    # "1e400" without raising -- and `histogram_quantile` (GATEWAY_LATENCY_P95)
    # is documented to return NaN over an all-zero-rate bucket, a realistic
    # quiet-minute case, not an exotic one. A NaN sample makes `max()` in
    # `run_metric_check` order-dependent (a genuine peak can be silently
    # replaced depending only on where the NaN sample sits in the fetched
    # list), and both non-finite kinds serialize outside the JSON spec
    # (Python's own encoder/decoder tolerate them; nothing else reading
    # evidence.jsonl is obliged to). Same "unreadable field -> None, skip
    # this row" contract this function already applies to unparsable rows.
    if not math.isfinite(at) or not math.isfinite(value):
        return None
    return MetricSample(at=at, value=value)


class ParsedSamples(NamedTuple):
    """The samples this module could read, how many raw rows Prometheus
    actually sent (`raw_count`), and how many series the response held
    (`series_count`).

    Round 8 review, P2. Before this type existed, `parse_samples` returned
    only the surviving samples -- the row-level distinction between "nothing
    was fetched" and "rows were fetched but every one of them was
    unreadable" (e.g. an all-NaN `histogram_quantile` window, a documented,
    realistic quiet-minute response, not an exotic one) was lost the moment
    `read_sample` dropped the bad rows. Both cases rendered identically:
    `sample_count: 0, max_value: 0.0` -- indistinguishable from a genuinely
    empty, valid response. `raw_count > len(samples)` is `run_metric_check`'s
    signal that rows existed and were discarded, not that nothing was there.

    Lab-defect-fix Unit 2, W14. `series_count` closes the same class of gap
    one level up: every registered template aggregates
    (`sum`/`max`/`histogram_quantile`), so a real Prometheus response is
    never meant to carry more than one series, but nothing enforced that --
    a multi-series response was silently narrowed to `series[0]`, with
    `series_count == 1` implied and never actually checked. `run_metric_
    check` surfaces this as its own `MULTIPLE_SERIES` status without
    withholding `series[0]`'s real samples -- see its own comment.
    """

    samples: list[MetricSample]
    raw_count: int
    series_count: int


def parse_samples(answered: JsonValue) -> ParsedSamples | None:
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
        return ParsedSamples([], 0, 0)
    first = series[0]
    if not isinstance(first, dict):
        return None
    values = first.get("values")
    if not isinstance(values, list):
        return ParsedSamples([], 0, len(series))
    samples = [read_sample(pair) for pair in values]
    kept = [sample for sample in samples if sample is not None]
    return ParsedSamples(kept, len(values), len(series))


def aligned_metric_window(
    window_start: datetime, window_end: datetime
) -> tuple[datetime, datetime, timedelta]:
    """Lab-defect-fix Unit 2, W2. Anchors the query grid on `window_end`
    instead of `window_start`, so `window_end` -- the instant closest to
    the fault, and the one most likely to hold data -- is itself an
    evaluated point, not a gap `query_range` silently steps past.

    All arithmetic is `timedelta`, never `float`/`total_seconds()`:
    `timedelta // timedelta` is exact integer division over the internal
    microsecond representation, so `aligned_start` lands exactly `points`
    whole steps before `window_end` with no rounding drift. Going through
    a float intermediate could leave `aligned_start` a microsecond off and
    silently un-align the very grid this fix exists to align.

    `aligned_start >= window_start` always holds by construction (the
    floor division can only lose up to `span mod step` off the head, never
    push `aligned_start` earlier than `window_start`) -- the query never
    reads outside the incident window policy already authorized. The lost
    head time falls inside `WINDOW_LEAD_IN`'s guaranteed-empty lead-in
    (`scenario_control.py`), the correct end to give up.

    Returns `(aligned_start, window_end, step)` -- `step` is returned
    alongside the endpoints, not left for a caller to re-read
    `METRIC_STEP_SECONDS` independently, so there is exactly one source of
    truth for the value that has to agree across the alignment arithmetic
    and the emitted request.
    """
    step = timedelta(seconds=METRIC_STEP_SECONDS)
    span = window_end - window_start
    points = span // step
    aligned_start = window_end - points * step
    return aligned_start, window_end, step


def query_prometheus(
    base_url: str,
    promql: str,
    start: datetime,
    end: datetime,
    step: timedelta,
    timeout: int,
) -> ParsedSamples | None:
    # Lab-defect-fix Unit 2, W2/#3. `step` is a parameter, not a second,
    # independent read of `METRIC_STEP_SECONDS` -- this function has
    # exactly one caller (`fetch_metric_samples`, confirmed by grep), which
    # already computed `step` once via `aligned_metric_window`. Two sites
    # each reading the same module constant is a real drift hazard,
    # especially with `METRIC_STEP_SECONDS` left open to future change
    # (owner decision Q12); passing it through keeps one source of truth.
    #
    # `int(step.total_seconds())`, not an f-string over the `timedelta`
    # directly: `f"{step}s"` on a `timedelta` renders its own `str()` form
    # (`"0:00:15s"` for 15 seconds), which Prometheus rejects with a 400 --
    # a failure this function's own `except (urllib.error.URLError,
    # OSError)` handler cannot distinguish from a genuinely unreachable
    # Prometheus, so it would misleadingly report `TOOL_UNAVAILABLE`
    # instead of the real formatting cause.
    query = urllib.parse.urlencode(
        {
            "query": promql,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "step": f"{int(step.total_seconds())}s",
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
) -> ParsedSamples | CheckOutcome:
    """Run the range query, mapping each failure mode to its own reason code.

    Returns the parsed samples on success, or the failed check outcome to
    return as-is.
    """
    # The wrapper always resolves a window before run_check is called
    # (tool_wrappers.resolve_effective_window) -- this is never None in
    # practice, but the argument model's own fields are typed Optional so
    # the model can omit them, so mypy needs this stated, not inferred.
    assert arguments.window_start is not None
    assert arguments.window_end is not None
    start, end, step = aligned_metric_window(
        arguments.window_start, arguments.window_end
    )
    try:
        parsed = query_prometheus(base_url, promql, start, end, step, timeout)
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
        parsed = None
    if parsed is None:
        return failed_check(
            EvidenceKind.METRIC,
            source,
            ToolOutcome.ERROR,
            ReasonCode.TOOL_ERROR,
            "Prometheus returned something this tool cannot read",
        )
    return parsed


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
    all_samples = fetched.samples
    # Lab-defect-fix Unit 2, W7 (+ W14's `MULTIPLE_SERIES`). Computed from
    # `fetched` before any trimming, so the status describes what
    # Prometheus actually returned, not what survived the cap/byte budget
    # below. `MULTIPLE_SERIES` is checked first and, deliberately, does NOT
    # short-circuit the rest of this function: `parse_samples` already
    # always reads `series[0]`'s real samples regardless of `series_count`
    # (see `MetricSampleStatus`'s own docstring), and this status must
    # never become a reason to withhold them -- a reader who cannot tell
    # `MULTIPLE_SERIES` apart from "no usable data" would be worse off than
    # before this status existed.
    #
    # Post-freeze review, P2 (three independent reviewers, same finding):
    # `MULTIPLE_SERIES` must be checked BEFORE `NO_USABLE_SAMPLES`/
    # `ALL_READINGS_DISCARDED`, not just for style -- `raw_count`/`all_
    # samples` are both computed from `series[0]` alone, so a response can
    # have `series_count > 1` AND an empty/unusable `series[0]` at the same
    # time. If the empty-series checks ran first, that exact response would
    # silently render as `NO_USABLE_SAMPLES` -- correctly describing
    # `series[0]`, but hiding the multi-series anomaly this status exists
    # to surface in the first place. Checking `series_count` first makes
    # that case unreachable: `MULTIPLE_SERIES` always wins the label,
    # whatever `series[0]` itself contains.
    if fetched.series_count > 1:
        status = MetricSampleStatus.MULTIPLE_SERIES
    elif fetched.series_count == 0:
        status = MetricSampleStatus.NO_RETURNED_SERIES
    elif fetched.raw_count == 0:
        status = MetricSampleStatus.NO_USABLE_SAMPLES
    elif not all_samples:
        status = MetricSampleStatus.ALL_READINGS_DISCARDED
    else:
        status = MetricSampleStatus.SAMPLED
    # Lab-defect-fix Unit 2, W4. Samples arrive already time-ordered
    # ascending from Prometheus's own `query_range` response (confirmed by
    # this module's own test fixtures, never independently sorted here) --
    # every real observation in this lab is at the tail of the window
    # (`SCRAPE_SETTLE_SECONDS`/W3 exists precisely because the fault signal
    # only appears near `window_end`), so keeping the *first* N samples, as
    # this line used to, silently discarded exactly the data-bearing region
    # once a query returned more than `MAX_METRIC_SAMPLES` rows. Keep the
    # newest N instead.
    kept = all_samples[-MAX_METRIC_SAMPLES:]
    rows: list[JsonValue] = [[sample.at, sample.value] for sample in kept]
    # Round 8 review, P2. `fetched.raw_count` is how many rows Prometheus
    # actually sent; `len(all_samples)` is how many of those `read_sample`
    # could parse into a finite `MetricSample`. The gap between them is
    # rows discarded as unreadable (malformed shape, or non-finite -- most
    # realistically an all-NaN `histogram_quantile` window over a quiet
    # minute) -- a DIFFERENT reduction than `truncated`, which only tracks
    # the `MAX_METRIC_SAMPLES` cap applied below. Without this, an all-NaN
    # window and a genuinely empty, valid response both rendered identically
    # (`sample_count: 0, max_value: 0.0`, `truncated: False`) -- the model
    # had no way to tell "confirmed zero" from "nothing measured."
    readings_discarded = fetched.raw_count - len(all_samples)
    payload: dict[str, JsonValue] = {
        "template": arguments.template.value,
        "service": arguments.service,
        "sample_count": len(kept),
        "max_value": max((sample.value for sample in kept), default=0.0),
        "truncated": len(all_samples) > len(kept),
        "readings_discarded": readings_discarded,
        "status": status.value,
    }
    # Q7 addendum. Receipt-internal audit data only: `render_context`
    # (`prompts.py`) puts only `Evidence.summary` in front of the model,
    # never `Evidence.payload`, so these five fields change nothing the
    # model sees -- they exist for an owner reading `evidence.jsonl` back,
    # recording exactly what was queried and against what resolved grid.
    # They are added before `trim_to_bytes` and so share the same byte
    # budget as everything else in the payload; at roughly 285 bytes
    # against `MAX_RESULT_BYTES`, they never approach it (a full
    # `MAX_METRIC_SAMPLES`-sample metric payload measures well under the
    # cap), and any trim that did reach them would set `truncated` like
    # every other path -- never silent.
    # `window_start`/`window_end` are asserted (not re-checked) because
    # `fetch_metric_samples` above already asserted the same and only
    # returned `ParsedSamples`, never `CheckOutcome`, once they held --
    # this is the same non-`None` guarantee restated for mypy at this
    # function's own scope, not a new runtime condition.
    assert arguments.window_start is not None
    assert arguments.window_end is not None
    start, end, step = aligned_metric_window(
        arguments.window_start, arguments.window_end
    )
    payload["promql"] = promql
    payload["query_window_start"] = start.isoformat()
    payload["query_window_end"] = end.isoformat()
    payload["query_step_seconds"] = int(step.total_seconds())
    payload["grid_points"] = int((end - start) // step) + 1
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
    # Lab-defect-fix Unit 2, W4. Names the *direction* kept, not just that
    # trimming happened -- a model silently reading a window trimmed at the
    # wrong end has no way to know from "(truncated)" alone which end it was.
    truncated_note = (
        f" (kept the newest {payload['sample_count']} of {len(all_samples)})"
        if payload["truncated"]
        else ""
    )
    # Round 8 review, P2. Deliberately a SEPARATE note from `(truncated)`
    # rather than folded into it: `truncated` means "there was more data
    # than fit the byte/count budget," `readings_discarded` means "some of
    # what Prometheus sent could not be read at all" (typically non-finite
    # readings -- see `read_sample`). Conflating the two would tell the
    # model the wrong story about WHY the count is smaller than it might
    # expect. This is also the only signal that distinguishes `sample_count:
    # 0, max_value: 0.0` meaning "confirmed zero, nothing to discard" from
    # "every reading this window returned was unreadable" -- before this,
    # both rendered identically.
    discarded = payload["readings_discarded"]
    assert isinstance(discarded, int)
    discarded_note = f" ({discarded} unreadable, discarded)" if discarded else ""
    # Lab-defect-fix Unit 2, W7/W14. `prompts.py`'s `render_context` puts
    # only `CheckOutcome.summary` in front of the model -- `payload["status"]`
    # existing is not enough on its own (§8.11's finding, restated by this
    # unit's plan: a fix that only touches the payload is invisible to the
    # thing it exists to fix). `MULTIPLE_SERIES` states both facts at once
    # -- the ambiguity AND that the sample figures above are real data from
    # one specific series, not withheld -- rather than only one.
    status_note = ""
    if status is MetricSampleStatus.MULTIPLE_SERIES:
        status_note = (
            f" (multiple series returned: showing series 1 of "
            f"{fetched.series_count}, others not shown)"
        )
    elif status is MetricSampleStatus.NO_RETURNED_SERIES:
        status_note = " (status: no series returned for this query)"
    elif status is MetricSampleStatus.NO_USABLE_SAMPLES:
        status_note = " (status: a series returned but had no usable samples)"
    elif status is MetricSampleStatus.ALL_READINGS_DISCARDED:
        status_note = " (status: every returned reading was unreadable)"
    return executed_check(
        EvidenceKind.METRIC,
        source,
        f"{arguments.template.value} for {arguments.service}: "
        f"{payload['sample_count']} samples, peak {payload['max_value']}"
        f"{truncated_note}{discarded_note}{status_note}",
        payload,
        started,
    )
