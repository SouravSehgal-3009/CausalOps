import json
import threading
import time
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fake_incident import (
    INCIDENT_ID,
    SERVICES,
    WINDOW_END,
    WINDOW_START,
    RecordingPrometheus,
    incident_scope,
    log_row,
    prometheus_body,
    write_log,
)
from pydantic import JsonValue

from causalops.domain import (
    EvidenceKind,
    IncidentScope,
    ReasonCode,
    ToolOutcome,
    ToolProposal,
)
from causalops.evidence import MAX_RESULT_BYTES, build_evidence
from causalops.prometheus import (
    MAX_METRIC_SAMPLES,
    METRIC_STEP_SECONDS,
    MetricSample,
    ParsedSamples,
    aligned_metric_window,
    fetch_metric_samples,
    parse_samples,
    read_sample,
    run_metric_check,
)
from causalops.telemetry import (
    MAX_LOG_ROWS,
    RunPaths,
    _registered_check_runner,
    read_json_file,
    read_json_line,
    run_changes_check,
    run_logs_check,
    run_topology_check,
    within_window,
)
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
    RunbookTopic,
    SearchRunbooksArguments,
)


@pytest.fixture
def stalled_prometheus() -> Iterator[str]:
    """A loopback server whose answer never arrives inside a short client timeout."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            time.sleep(2)
            payload = prometheus_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


# Lab-defect-fix Unit 2, W4. Mirrors `prometheus_body()`'s own default
# (`fake_incident.py`) by the same formula -- `fake_prometheus` always
# calls `prometheus_body()` with no argument, so this is exactly how many
# samples every test using that fixture actually receives.
FAKE_PROMETHEUS_SAMPLE_COUNT = MAX_METRIC_SAMPLES + 5


def metric_arguments(
    service: str = "gateway",
    template: MetricTemplate = MetricTemplate.GATEWAY_ERROR_RATE,
) -> QueryMetricArguments:
    return QueryMetricArguments(
        template=template,
        service=service,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )


def logs_arguments(
    row_limit: int = 20, log_filter: LogFilter = LogFilter.ERRORS_ONLY
) -> QueryLogsArguments:
    return QueryLogsArguments(
        log_filter=log_filter,
        service="orders",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        row_limit=row_limit,
    )


def test_a_metric_query_returns_bounded_samples(
    tmp_path: Path,
    fake_prometheus: RecordingPrometheus,
) -> None:
    """Lab-defect-fix Unit 2, W4. `fake_prometheus` returns `prometheus_body()`'s
    default `MAX_METRIC_SAMPLES + 5` samples with `value = index * 0.5`,
    strictly increasing with index -- so the newest-kept slice (indices
    `5..MAX_METRIC_SAMPLES+4`) peaks at its own last index, not at
    `MAX_METRIC_SAMPLES - 1` the way the pre-W4 oldest-kept slice did.
    Derived from `FAKE_PROMETHEUS_SAMPLE_COUNT`, not hand-typed, so a future
    change to `prometheus_body`'s own default cannot silently desync this
    literal again."""
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), fake_prometheus.url, 5
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.kind is EvidenceKind.METRIC
    assert outcome.payload["sample_count"] == MAX_METRIC_SAMPLES
    assert outcome.payload["truncated"] is True
    assert outcome.payload["max_value"] == (FAKE_PROMETHEUS_SAMPLE_COUNT - 1) * 0.5
    assert f"{MAX_METRIC_SAMPLES} samples" in outcome.summary


@pytest.fixture
def small_fake_prometheus() -> Iterator[str]:
    """A loopback stand-in returning fewer samples than `MAX_METRIC_SAMPLES`
    -- unlike `fake_prometheus`, this one is genuinely NOT truncated. The
    negative case for the truncated-summary tests below, without which
    nothing could tell "always renders (truncated)" apart from correctly
    rendering it only when it is actually true."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            payload = prometheus_body(sample_count=3)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_a_metric_summary_names_truncation_when_samples_were_cut(
    fake_prometheus: RecordingPrometheus,
) -> None:
    """Round 6 review, the P1. `prompts.py`'s `render_context` puts only
    `CheckOutcome.summary` in front of the model -- the full payload
    (which correctly carries `payload["truncated"]`) never reaches it.
    `run_logs_check`/`run_changes_check` already named truncation in
    their summary strings this round; `run_metric_check` did not, so a
    cut metric window read as complete with no signal the true peak
    might lie outside what survived."""
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), fake_prometheus.url, 5
    )

    assert outcome.payload["truncated"] is True
    # Lab-defect-fix Unit 2, W4: the note now names the kept direction, not
    # just that trimming happened.
    assert outcome.summary.endswith(
        f"(kept the newest {MAX_METRIC_SAMPLES} of {FAKE_PROMETHEUS_SAMPLE_COUNT})"
    )


def test_a_metric_summary_names_no_truncation_when_nothing_was_cut(
    small_fake_prometheus: str,
) -> None:
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), small_fake_prometheus, 5
    )

    assert outcome.payload["truncated"] is False
    assert not outcome.summary.endswith("(truncated)")


def test_a_metric_summary_reports_the_post_trim_count_not_the_pre_trim_one(
    monkeypatch: pytest.MonkeyPatch, fake_prometheus: RecordingPrometheus
) -> None:
    """Post-freeze review. `run_metric_check`'s summary string used to read
    `len(kept)` (the PRE-trim count) instead of `payload['sample_count']`
    (kept honest post-trim by `trim_to_bytes`, Unit 3b-4 addendum's C3)
    four lines below where the payload was already fixed -- reproduced
    live by correctness (payload said `sample_count: 371`, the summary
    still said "900 samples"). Unreachable through this function's own
    REAL data shape in this test suite: `MetricSample.at`/`.value` are
    both floats, always small, so `trim_to_bytes`'s byte-level popping
    never actually fires below the `MAX_METRIC_SAMPLES` count cap already
    applied earlier in the function. Monkeypatching `trim_to_bytes` to
    additionally drop one sample (simulating what a future, larger row
    shape would trigger) forces `payload['sample_count']` and the
    pre-trim `len(kept)` to genuinely differ, proving the summary string
    reads the field the payload itself reports, not a stale local
    variable, regardless of whether today's data can reach the
    difference."""
    import causalops.prometheus as prometheus_module

    real_trim_to_bytes = prometheus_module.trim_to_bytes

    def shrinking_trim_to_bytes(
        payload: dict[str, JsonValue],
        rows_key: str,
        rows: list[JsonValue],
        count_key: str,
    ) -> dict[str, JsonValue]:
        result = real_trim_to_bytes(payload, rows_key, rows, count_key)
        current_count = result[count_key]
        assert isinstance(current_count, int)
        result[count_key] = current_count - 1
        return result

    monkeypatch.setattr(prometheus_module, "trim_to_bytes", shrinking_trim_to_bytes)

    outcome = run_metric_check(
        metric_arguments(), incident_scope(), fake_prometheus.url, 5
    )

    assert outcome.payload["sample_count"] == MAX_METRIC_SAMPLES - 1
    assert f"{MAX_METRIC_SAMPLES - 1} samples" in outcome.summary
    assert f"{MAX_METRIC_SAMPLES} samples" not in outcome.summary


def test_a_metric_max_value_reflects_the_post_trim_samples_not_every_fetched_one(
    monkeypatch: pytest.MonkeyPatch, fake_prometheus: RecordingPrometheus
) -> None:
    """Round 4 review, F3. `max_value` (payload) used to be computed from
    `kept`, entirely before `trim_to_bytes` runs -- so a sample popped by
    byte trimming could still be reported as the peak even though it no
    longer appears in `payload["samples"]`. Not reachable through this
    function's own real data shape in this test suite (`MetricSample.at`/
    `.value` are both floats, always small, so byte-level popping never
    fires below the `MAX_METRIC_SAMPLES` count cap already applied
    earlier), fixed anyway per the owner's ruling to close the class, not
    just the reachable `event_codes` instance. Monkeypatching
    `trim_to_bytes` to additionally pop the LAST sample drops the
    highest-value sample -- `test_a_metric_query_returns_bounded_samples`
    above already establishes values increase with index, so the last
    sample in the kept (post-W4, newest-first-preserved) slice is always
    its peak -- forcing `payload["max_value"]` and the pre-trim peak to
    genuinely differ, proving the field reads the samples the payload
    itself reports, not a stale local variable.

    Lab-defect-fix Unit 2, W4 note: production's own `trim_to_bytes` now
    pops from the FRONT, not the end (see its docstring) -- this
    monkeypatch deliberately pops the opposite end anyway. That is still a
    valid test of the property under test here (`max_value` is rebuilt
    from whatever `payload["samples"]` actually holds after trimming, not
    a stale pre-trim computation) -- that property holds regardless of
    which end trimming removes from. It is no longer a simulation of a
    real, larger-row-shape trigger of production's own byte-trim path,
    which pops the front and would instead drop the LOWEST-value sample
    from this fixture's strictly-increasing data."""
    import causalops.prometheus as prometheus_module

    real_trim_to_bytes = prometheus_module.trim_to_bytes

    def popping_trim_to_bytes(
        payload: dict[str, JsonValue],
        rows_key: str,
        rows: list[JsonValue],
        count_key: str,
    ) -> dict[str, JsonValue]:
        result = real_trim_to_bytes(payload, rows_key, rows, count_key)
        kept_rows = result[rows_key]
        assert isinstance(kept_rows, list)
        kept_rows.pop()
        result[rows_key] = kept_rows
        result[count_key] = len(kept_rows)
        return result

    monkeypatch.setattr(prometheus_module, "trim_to_bytes", popping_trim_to_bytes)

    outcome = run_metric_check(
        metric_arguments(), incident_scope(), fake_prometheus.url, 5
    )

    assert outcome.payload["sample_count"] == MAX_METRIC_SAMPLES - 1
    assert outcome.payload["max_value"] == (FAKE_PROMETHEUS_SAMPLE_COUNT - 2) * 0.5


def test_the_aligned_window_keeps_window_end_as_an_evaluated_point() -> None:
    """Lab-defect-fix Unit 2, W2. Asserts on the EMITTED request parameters
    only -- `(end - start) % step == 0` and `start > window_start` --
    never inferred Prometheus rounding/millisecond semantics (round-3
    correction, `LAB_DEFECTS_FIX_PLAN.md` §11.5: a sound inference is not a
    documented guarantee).

    Post-freeze review (three independent reviewers, same finding): a
    `WINDOW_START`/`WINDOW_END` span is exactly `120 * METRIC_STEP_SECONDS`
    -- an exact multiple of the step -- so floor division (the correct
    fix) and ceiling division (the exact bug this design explicitly warns
    against: it pushes `start` *before* `window_start`, widening the
    authorized scope past what policy granted) produce IDENTICAL results
    on that fixture. Confirmed by mutation: reverting `aligned_metric_
    window` to `ceil`-based rounding, this test still passed -- it could
    not actually distinguish the fix from the bug it exists to catch.
    `non_round_start`, offset 7 seconds from an exact multiple of the
    step, makes floor and ceil diverge: floor division lands `start` a few
    seconds *after* `non_round_start` (this test's own `start >
    non_round_start`, now strict), while ceiling division would land it a
    few seconds *before* -- exactly the scope-widening bug. `end ==
    window_end` by construction is what makes `window_end` itself an
    evaluated grid point: `query_range` evaluates `start, start+step, ...,
    end`, and `(end - start) % step == 0` guarantees `end` lands exactly
    on that grid, never between two points."""
    non_round_start = WINDOW_START + timedelta(seconds=7)
    start, end, step = aligned_metric_window(non_round_start, WINDOW_END)

    assert step == timedelta(seconds=METRIC_STEP_SECONDS)
    assert (end - start) % step == timedelta(0)
    assert start > non_round_start
    assert end == WINDOW_END


@contextmanager
def _recording_prometheus() -> Iterator[tuple[str, list[dict[str, str]]]]:
    """A loopback stand-in that records every full query-string dict it
    received (`start`/`end`/`step`/`query`), not just `query` alone the
    way `conftest.py`'s `fake_prometheus` does -- needed here because W2's
    own claim is about the emitted `start`/`end`/`step` parameters
    themselves, not the PromQL string."""
    received: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            received.append({key: values[0] for key, values in parsed.items()})
            payload = prometheus_body(sample_count=1)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_metric_samples_emits_a_plain_integer_second_step() -> None:
    """Lab-defect-fix Unit 2, W2/(a). `step` is a `timedelta` parameter as
    of this unit (passed through from `aligned_metric_window`, not read a
    second time from `METRIC_STEP_SECONDS` inside `query_prometheus` --
    see its own docstring for the drift hazard this closes). `f"{step}s"`
    on a bare `timedelta` renders its own `str()` form (`"0:00:15s"` for 15
    seconds), which Prometheus would reject with a 400 -- and that failure
    would misleadingly present as `TOOL_UNAVAILABLE` through this module's
    existing `except (urllib.error.URLError, OSError)` handler, not the
    real formatting cause. Asserts the emitted `step` query parameter is
    the exact string `"15s"`, not just that the modulo-alignment property
    holds."""
    with _recording_prometheus() as (url, received):
        outcome = fetch_metric_samples(url, "up", metric_arguments(), 5, "query_metric")

    assert isinstance(outcome, ParsedSamples)
    (call,) = received
    assert call["step"] == f"{METRIC_STEP_SECONDS}s"
    start = datetime.fromisoformat(call["start"])
    end = datetime.fromisoformat(call["end"])
    assert end == WINDOW_END
    assert start <= end
    assert (end - start) % timedelta(seconds=METRIC_STEP_SECONDS) == timedelta(0)


def test_the_metric_payload_records_what_was_actually_queried(
    fake_prometheus: RecordingPrometheus,
) -> None:
    """Q7's audit fields. Deliberately uses a window whose span is NOT a
    whole multiple of METRIC_STEP_SECONDS: with the default 30-minute
    fixture window (1800s, an exact multiple of 15s) `aligned_metric_window`
    is a no-op and `query_window_start` would equal the raw `window_start`,
    so this test could not tell the resolved grid apart from the requested
    window. Offsetting the start by 7s makes them differ, which is the
    whole point of recording it."""
    arguments = QueryMetricArguments(
        template=MetricTemplate.GATEWAY_ERROR_RATE,
        service="gateway",
        window_start=WINDOW_START + timedelta(seconds=7),
        window_end=WINDOW_END,
    )

    outcome = run_metric_check(arguments, incident_scope(), fake_prometheus.url, 5)

    payload = outcome.payload
    assert payload["promql"] == (
        'sum(rate(causalops_requests_total{service="gateway",'
        f'incident="{INCIDENT_ID}",outcome="error"}}[30s]))'
    )
    assert payload["query_window_start"] == (
        (WINDOW_START + timedelta(seconds=15)).isoformat()
    )
    assert payload["query_window_end"] == WINDOW_END.isoformat()
    assert payload["query_step_seconds"] == METRIC_STEP_SECONDS
    assert payload["grid_points"] == 120


def test_a_metric_query_against_nothing_is_unavailable() -> None:
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), "http://127.0.0.1:1", 1
    )

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_a_metric_query_that_outlasts_its_timeout_times_out(
    stalled_prometheus: str,
) -> None:
    """A genuine `urllib.request.urlopen` timeout, not a stand-in for one."""
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), stalled_prometheus, 1
    )

    assert outcome.outcome is ToolOutcome.TIMEOUT
    assert outcome.reason_code is ReasonCode.TOOL_TIMEOUT


def test_a_service_name_that_could_reach_promql_is_refused() -> None:
    outcome = run_metric_check(
        metric_arguments(service='gateway"} or up{'), incident_scope(), "http://x", 1
    )

    assert outcome.outcome is ToolOutcome.ERROR
    assert outcome.reason_code is ReasonCode.TOOL_ERROR


def test_an_unreadable_prometheus_answer_is_an_error() -> None:
    assert parse_samples({"status": "error"}) is None
    assert parse_samples(
        {"status": "success", "data": {"result": []}}
    ) == ParsedSamples([], 0, 0)
    assert parse_samples("nonsense") is None


def test_read_sample_rejects_non_finite_readings() -> None:
    """Round 7 review. `float()` parses "NaN", "Infinity", and an overflow
    literal like "1e400" into non-finite floats without raising -- so
    those used to become an ordinary-looking `MetricSample`. `float()`
    also accepts a bare numeric NaN/inf for the timestamp field, so both
    positions are checked.

    Round 8 review, P2. `float()` on a STRING that overflows (`"1e400"`,
    above) rounds to `inf` without raising -- caught by the `isfinite`
    check below. `float()` on a Python `int` that is too large to
    represent raises `OverflowError` INSTEAD, which is not a subclass of
    `ValueError` -- `json.loads` produces a genuine Python `int` for a
    large integer literal with no decimal point or exponent, so this is
    reachable from a real Prometheus response, not just a synthetic case.
    Both the timestamp and value positions are checked, matching the two
    `float()` calls in the same `try` block."""
    assert read_sample([0, "NaN"]) is None
    assert read_sample([0, "Infinity"]) is None
    assert read_sample([0, "-Infinity"]) is None
    assert read_sample([0, "1e400"]) is None  # overflows to inf
    assert read_sample([float("nan"), "0.5"]) is None
    assert read_sample([float("inf"), "0.5"]) is None
    huge_int = 10**400
    assert read_sample([huge_int, "0.5"]) is None  # OverflowError on `at`
    assert read_sample([0, huge_int]) is None  # OverflowError on `value`
    assert read_sample([0, "0.5"]) == MetricSample(at=0.0, value=0.5)


def test_a_nan_sample_does_not_win_the_reported_peak_regardless_of_position() -> None:
    """`histogram_quantile` (GATEWAY_LATENCY_P95) is documented to return
    NaN over an all-zero-rate bucket -- a quiet minute, not an exotic
    input. Before this fix, `max()` over a sample list containing NaN
    was order-dependent: whether the reported peak survived depended on
    whether the NaN sample happened to sit before or after it in the
    fetched list. Both orderings are checked so a fix that only handles
    one of them cannot pass silently."""
    nan_after_the_peak = {
        "status": "success",
        "data": {"result": [{"values": [[0, "0.1"], [1, "0.9"], [2, "NaN"]]}]},
    }
    nan_before_the_peak = {
        "status": "success",
        "data": {"result": [{"values": [[0, "NaN"], [1, "0.1"], [2, "0.9"]]}]},
    }
    for answer in (nan_after_the_peak, nan_before_the_peak):
        parsed = parse_samples(answer)
        assert parsed is not None
        assert len(parsed.samples) == 2
        assert max(sample.value for sample in parsed.samples) == 0.9


@contextmanager
def _serve_fixed_prometheus_body(body: bytes) -> Iterator[str]:
    """Same loopback pattern as `fake_prometheus`/`stalled_prometheus`
    above, but for a caller-supplied raw body instead of `prometheus_body`'s
    fixed shape -- used once, for the end-to-end NaN test below, so it is a
    plain context manager rather than another named fixture."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_metric_checks_reported_peak_survives_a_nan_sample_end_to_end() -> None:
    """Same claim as the `parse_samples`-level test above, exercised through
    `run_metric_check`'s full pipeline (including the post-trim
    `payload["max_value"]` rebuild) rather than only the parsing layer.

    Round 8 review, P2. This is also the partially-NaN case for
    `readings_discarded`: 3 raw rows, 1 unreadable, 2 kept -- the summary
    must name the 1 discarded reading, distinct from `(truncated)`."""
    body = json.dumps(
        {
            "status": "success",
            "data": {"result": [{"values": [[0, "NaN"], [1, "0.1"], [2, "0.9"]]}]},
        }
    ).encode("utf-8")
    with _serve_fixed_prometheus_body(body) as url:
        outcome = run_metric_check(metric_arguments(), incident_scope(), url, 5)

    assert outcome.payload["sample_count"] == 2
    assert outcome.payload["max_value"] == 0.9
    assert outcome.payload["readings_discarded"] == 1
    assert outcome.payload["truncated"] is False
    assert "1 unreadable, discarded" in outcome.summary


def test_an_all_nan_metric_window_does_not_read_as_confirmed_zero() -> None:
    """Round 8 review, P2. `histogram_quantile` (GATEWAY_LATENCY_P95) is
    documented to return NaN over an all-zero-rate bucket -- a realistic
    "quiet minute," not an exotic input. Once round 7's fix correctly
    drops every non-finite sample, an all-NaN window produces
    `sample_count: 0, max_value: 0.0` -- which, without `readings_
    discarded`, is bit-for-bit identical to a genuinely empty, valid
    response (`test_an_unreadable_prometheus_answer_is_an_error`'s `{
    "result": []}` case). A model reading only the summary could not tell
    "confirmed zero, nothing measured was wrong" from "every reading this
    window returned was unreadable" -- exactly the ambiguity round 6's own
    `truncated` fix closed for the byte-trim case, reopened here in a new
    shape."""
    body = json.dumps(
        {
            "status": "success",
            "data": {"result": [{"values": [[0, "NaN"], [1, "NaN"], [2, "Infinity"]]}]},
        }
    ).encode("utf-8")
    with _serve_fixed_prometheus_body(body) as url:
        outcome = run_metric_check(metric_arguments(), incident_scope(), url, 5)

    assert outcome.payload["sample_count"] == 0
    assert outcome.payload["max_value"] == 0.0
    assert outcome.payload["readings_discarded"] == 3
    assert "3 unreadable, discarded" in outcome.summary
    # Lab-defect-fix Unit 2, post-freeze review (three independent
    # reviewers, same finding): a direct `payload["status"]` assertion,
    # not just the summary-text behavior above -- all 3 raw rows came back
    # but none survived `read_sample`, `ALL_READINGS_DISCARDED`.
    assert outcome.payload["status"] == "ALL_READINGS_DISCARDED"
    assert "every returned reading was unreadable" in outcome.summary

    empty_body = json.dumps(
        {"status": "success", "data": {"result": [{"values": []}]}}
    ).encode("utf-8")
    with _serve_fixed_prometheus_body(empty_body) as url:
        genuinely_empty = run_metric_check(metric_arguments(), incident_scope(), url, 5)

    assert genuinely_empty.payload["sample_count"] == 0
    assert genuinely_empty.payload["max_value"] == 0.0
    assert genuinely_empty.payload["readings_discarded"] == 0
    assert "unreadable, discarded" not in genuinely_empty.summary
    # A series came back but yielded no raw rows at all -- `NO_USABLE_
    # SAMPLES`, distinct in `payload["status"]` from `ALL_READINGS_
    # DISCARDED` above, not just in prose.
    assert genuinely_empty.payload["status"] == "NO_USABLE_SAMPLES"
    assert "a series returned but had no usable samples" in genuinely_empty.summary
    # The two summaries must differ -- this is the actual claim: a reader
    # (or the model) can no longer confuse "confirmed zero" with "every
    # reading was unreadable" from the summary text alone.
    assert outcome.summary != genuinely_empty.summary


def test_multiple_series_reports_status_without_suppressing_real_data() -> None:
    """Lab-defect-fix Unit 2, W7/W14, round-2 refinement (c). The positive
    case for `MULTIPLE_SERIES`: it is a status LABEL, never a gate that
    withholds `series[0]`'s real samples. Without a test proving this
    specifically, a future refactor that early-returns on `MULTIPLE_SERIES`
    (treating it as another failure-shaped status, the same trap `§8.11`
    already caught once for `readings_discarded`) would pass every other
    test in this file while silently regressing exactly the property this
    status exists to preserve."""
    body = json.dumps(
        {
            "status": "success",
            "data": {
                "result": [
                    {"values": [[0, "1.5"], [1, "9.5"], [2, "3.0"]]},
                    {"values": [[0, "100.0"]]},
                ]
            },
        }
    ).encode("utf-8")
    with _serve_fixed_prometheus_body(body) as url:
        outcome = run_metric_check(metric_arguments(), incident_scope(), url, 5)

    assert outcome.payload["status"] == "MULTIPLE_SERIES"
    assert outcome.payload["sample_count"] == 3
    samples = outcome.payload["samples"]
    assert isinstance(samples, list)
    assert samples
    # The real peak from series[0] (9.5), not series[1]'s 100.0 -- proving
    # the samples genuinely came from series[0], not some other placeholder.
    assert outcome.payload["max_value"] == 9.5
    assert "3 samples" in outcome.summary
    assert "peak 9.5" in outcome.summary
    assert "multiple series returned" in outcome.summary
    assert "series 1 of 2" in outcome.summary


def test_multiple_series_wins_even_when_series_zero_is_itself_unusable() -> None:
    """Lab-defect-fix Unit 2, W7/W14, post-freeze review P2 (three
    independent reviewers, same finding). `raw_count`/`all_samples` are
    both computed from `series[0]` alone, so a response can have
    `series_count > 1` AND an empty, unusable `series[0]` at the same
    time. If the empty-series checks (`NO_USABLE_SAMPLES`/
    `ALL_READINGS_DISCARDED`) ran before the `series_count` check, this
    exact response would silently render as `NO_USABLE_SAMPLES` --
    correct about `series[0]` alone, but hiding the multi-series anomaly
    entirely. `run_metric_check`'s own comment states why the order is
    load-bearing; this is the regression test for it."""
    body = json.dumps(
        {
            "status": "success",
            "data": {
                "result": [
                    {"values": []},
                    {"values": [[0, "5.0"]]},
                ],
            },
        }
    ).encode("utf-8")
    with _serve_fixed_prometheus_body(body) as url:
        outcome = run_metric_check(metric_arguments(), incident_scope(), url, 5)

    assert outcome.payload["status"] == "MULTIPLE_SERIES"
    assert outcome.payload["status"] != "NO_USABLE_SAMPLES"
    assert outcome.payload["sample_count"] == 0
    assert "multiple series returned" in outcome.summary
    assert "series 1 of 2" in outcome.summary


def test_no_returned_series_is_distinguishable_from_no_usable_samples() -> None:
    """Lab-defect-fix Unit 2, W7. `NO_RETURNED_SERIES` (empty `result`
    list) and `NO_USABLE_SAMPLES` (a series returned but yielded nothing)
    are different facts about what the query returned -- the neutral
    naming this unit adopted specifically avoids claiming either one means
    the series does not exist (round-1 correction, plan §9.1)."""
    empty_result = json.dumps({"status": "success", "data": {"result": []}}).encode(
        "utf-8"
    )
    with _serve_fixed_prometheus_body(empty_result) as url:
        outcome = run_metric_check(metric_arguments(), incident_scope(), url, 5)

    assert outcome.payload["status"] == "NO_RETURNED_SERIES"
    assert "no series returned for this query" in outcome.summary


def test_the_ordinary_case_reports_sampled(small_fake_prometheus: str) -> None:
    """Lab-defect-fix Unit 2, post-freeze review (three independent
    reviewers, same finding): `SAMPLED` -- the ordinary, successful case,
    exercised by nearly every other test in this file -- had no test
    asserting `payload["status"]` directly. `small_fake_prometheus`
    returns a genuinely un-truncated, real response (`prometheus_body
    (sample_count=3)`), so this is the plain happy path, not an edge
    case."""
    outcome = run_metric_check(
        metric_arguments(), incident_scope(), small_fake_prometheus, 5
    )

    assert outcome.payload["status"] == "SAMPLED"
    assert outcome.payload["sample_count"] == 3
    # None of the non-SAMPLED status notes leak into an ordinary summary.
    assert "status:" not in outcome.summary
    assert "multiple series" not in outcome.summary


def test_a_log_query_returns_only_matching_rows_in_the_window(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [
            log_row(1),
            log_row(2, severity="info", event="request_served"),
            log_row(99_999),
        ],
    )

    outcome = run_logs_check(logs_arguments(), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
    assert outcome.payload["event_codes"] == "config_rejected_request"


def test_a_log_query_clamps_the_row_limit_to_the_budget(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(paths, [log_row(minute) for minute in range(MAX_LOG_ROWS + 10)])

    outcome = run_logs_check(logs_arguments(row_limit=200), paths)

    assert outcome.payload["row_count"] == MAX_LOG_ROWS
    assert outcome.payload["truncated"] is True


def test_a_log_result_stays_inside_the_byte_bound(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(paths, [log_row(minute, detail="x" * 2000) for minute in range(30)])

    outcome = run_logs_check(logs_arguments(row_limit=30), paths)

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True
    # Post-freeze review, P3-1. 30 rows were matched, but byte-trimming
    # (not just the `row_limit` cap) drops some of them -- `outcome.summary`
    # must report the SAME, smaller, post-trim count `payload["row_count"]`
    # holds, not the pre-trim 30 `len(rows)` used to report regardless of
    # what trimming did.
    assert outcome.payload["row_count"] < 30
    assert f"{outcome.payload['row_count']} rows" in outcome.summary
    # Lab-defect-fix Unit 2, W4: the note now names the kept direction. All
    # 30 rows matched the filter/window (`row_limit=30` never triggers the
    # ring's own eviction here -- only byte-trimming reduces the count
    # below 30), so the "of" figure is 30, the matched total.
    assert outcome.summary.endswith(
        f"(kept the newest {outcome.payload['row_count']} of 30)"
    )


def test_log_event_codes_reflect_the_post_trim_rows_not_every_matched_row(
    tmp_path: Path,
) -> None:
    """Round 4 review, F3. `event_codes` used to be built from `events`,
    gathered during the loop that fills `rows` -- entirely BEFORE
    `trim_to_bytes` runs. So a row popped by byte trimming (as opposed to
    the `row_limit` cap the loop already respects) could still have its
    event code listed even though it no longer appears in
    `payload["rows"]`. `trim_to_bytes` pops from the FRONT of the row list
    (`kept.pop(0)`) as of lab-defect-fix Unit 2, W4, so the FIRST matched
    row here -- the only one carrying "config_loaded" -- is the first one
    trimmed away once the payload is oversized; every surviving row is
    "config_rejected_request". Reachable with real data (same
    oversized-detail shape as the byte-bound test above), not a
    monkeypatch."""
    paths = RunPaths(root=tmp_path)
    rows = [log_row(0, event="config_loaded", detail="x" * 2000)]
    rows.extend(
        log_row(minute, event="config_rejected_request", detail="x" * 2000)
        for minute in range(1, 30)
    )
    write_log(paths, rows)

    outcome = run_logs_check(
        logs_arguments(row_limit=30, log_filter=LogFilter.CONFIG_RELOAD), paths
    )

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True
    assert outcome.payload["row_count"] < 30
    kept_rows = outcome.payload["rows"]
    assert isinstance(kept_rows, list)
    assert not any(
        isinstance(row, dict) and row.get("event") == "config_loaded"
        for row in kept_rows
    )
    assert "config_loaded" not in str(outcome.payload["event_codes"])
    assert "config_rejected_request" in str(outcome.payload["event_codes"])


def test_a_log_query_for_a_run_without_that_log_is_unavailable(tmp_path: Path) -> None:
    outcome = run_logs_check(logs_arguments(), RunPaths(root=tmp_path))

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_within_window_rejects_a_naive_timestamp_instead_of_raising() -> None:
    """Unit 3b-4 addendum, C2. `datetime.fromisoformat` returns a naive
    `datetime` for a string carrying no UTC offset; comparing it against
    the aware `WINDOW_START`/`WINDOW_END` this function is always called
    with used to raise `TypeError`, uncaught (only `ValueError`, the parse
    failure, was caught) -- a naive timestamp is now excluded the same way
    an unparseable one already was, not raised."""
    naive = WINDOW_START.replace(tzinfo=None).isoformat()

    assert within_window(naive, WINDOW_START, WINDOW_END) is False


def test_a_log_row_with_a_naive_timestamp_is_excluded_not_crashed(
    tmp_path: Path,
) -> None:
    """The end-to-end sibling of `test_within_window_rejects_a_naive_
    timestamp_instead_of_raising`: before this fix, this call raised
    `TypeError` out of `within_window`, uncaught by `run_logs_check`,
    turning one malformed row into a crash for the whole check rather than
    excluding just that row."""
    paths = RunPaths(root=tmp_path)
    naive_row = log_row(5)
    naive_row["at"] = WINDOW_START.replace(tzinfo=None).isoformat()
    write_log(paths, [naive_row, log_row(10)])

    outcome = run_logs_check(logs_arguments(), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1


def test_a_change_row_with_a_naive_timestamp_is_excluded_not_crashed(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": WINDOW_START.replace(tzinfo=None).isoformat(),
                    "service": "orders",
                    "summary": "naive-timestamp change",
                },
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": "aware change",
                },
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["change_count"] == 1


def test_an_oversized_single_summary_still_forces_the_payload_under_budget(
    tmp_path: Path,
) -> None:
    """Unit 3b-4 addendum, C3. `run_changes_check` joins every matched
    change's `summary` into one scalar `summaries` string before calling
    `trim_to_bytes` -- a single change with an oversized summary makes that
    scalar bigger than `MAX_RESULT_BYTES` all by itself, which no amount of
    popping the `changes` list could ever fix (there is only one change to
    begin with). Before this fix, `trim_to_bytes` would try popping it,
    empty the list, and still return an over-budget payload silently."""
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": "x" * (MAX_RESULT_BYTES + 4000),
                }
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True
    # `changes` (which embeds the raw entry, oversized `summary` field and
    # all) is popped before the scalar fallback ever runs -- see
    # `trim_to_bytes`'s own docstring for why that ordering matters here:
    # the single change's OWN `summary` field is a second copy of the same
    # oversized text `summaries` holds, invisible to a fallback that only
    # inspects top-level scalars. `change_count` still honestly matches
    # what `changes` actually holds, whatever that ends up being.
    assert outcome.payload["change_count"] == len(outcome.payload["changes"])  # type: ignore[arg-type]
    # Post-freeze review, P3-1. Before that fix, `outcome.summary` was
    # built from `len(changes)` (the PRE-trim count, always 1 here) --
    # "1 recent changes on orders" while the payload actually held zero.
    # Reproduced by correctness directly against this exact scenario.
    assert outcome.payload["change_count"] == 0
    # Lab-defect-fix Unit 2, W4: the note now names the kept direction --
    # one change matched (`len(changes) == 1`), none survived.
    assert outcome.summary == "0 recent changes on orders (kept the newest 0 of 1)"
    assert isinstance(outcome.payload["summaries"], str)
    assert len(outcome.payload["summaries"]) < MAX_RESULT_BYTES
    # Round 6 review, item 2. `changes` was fully emptied here, so the
    # rebuild below (built from the post-trim `changes` list) replaces
    # whatever text the scalar-shrinking fallback left in `summaries` with
    # an empty string -- there is nothing left to summarize.
    assert outcome.payload["summaries"] == ""


def test_changes_summaries_only_name_changes_still_present_after_trimming(
    tmp_path: Path,
) -> None:
    """Round 6 review, item 2. `summaries` is joined from EVERY matched
    change before `trim_to_bytes` runs and is a scalar, not a row list --
    the same pre-trim-aggregate shape already fixed twice this round for
    `event_codes` and `max_value`. Unlike the single-oversized-summary
    case above (where `changes` is fully emptied), this scenario is the
    more common partial trim: enough changes to force byte trimming, but
    not so much oversized text that popping rows alone cannot bring the
    payload back under budget. Every entry here shares the identical `at`
    timestamp, so `run_changes_check`'s own chronological sort (lab-defect-
    fix Unit 2, W4/#4) is a no-op and declaration order survives unchanged
    into `trim_to_bytes`. `trim_to_bytes` pops rows from the FRONT of the
    list as of that same unit (`kept.pop(0)`), so `changes` 023 through 029
    survive and 000 through 022 do not -- reproduced directly against the
    real function (not asserted blindly): 30 changes here reduce to
    `change_count == 7`, with `summaries` still holding the text of every
    one of the 30 (only `changes` had been trimmed) before this fix's
    rebuild. `summaries` must name only the survivors, and the payload must
    still fit. The assertions below read whatever actually survived
    dynamically, so they hold regardless of which end trims -- only this
    docstring's specific illustration is direction-sensitive."""
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": f"change-{index:03d}: " + "s" * 300,
                }
                for index in range(30)
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True
    change_count = outcome.payload["change_count"]
    assert isinstance(change_count, int)
    assert 0 < change_count < 30
    surviving_indices = {
        change["summary"][:10]  # type: ignore[index]
        for change in outcome.payload["changes"]  # type: ignore[union-attr]
    }
    summaries = str(outcome.payload["summaries"])
    for index in range(30):
        marker = f"change-{index:03d}"
        if marker in surviving_indices:
            assert marker in summaries, f"{marker} survived trimming but is missing"
        else:
            assert marker not in summaries, (
                f"{marker} was trimmed away but still named in summaries"
            )


def test_a_pathologically_long_changes_summary_still_builds_valid_evidence(
    tmp_path: Path,
) -> None:
    """Lab-defect-fix F8. `run_changes_check`'s summary now embeds real change
    content, capped to a budget computed from the rest of the summary's own
    runtime length (Codex round 1 -- a fixed content cap alone cannot
    guarantee the overall bound, since `arguments.service` has no schema
    `max_length`) and marked with "..." -- this proves the cap actually holds
    against `Evidence.summary`'s hard `max_length=400` bound (`domain.py`),
    not just against this function's own final clamp. `build_evidence`
    (`tool_wrappers.py:556`) constructs `Evidence` with no try/except around
    it by design, so a summary this function ever let through over 400
    chars would not fail safely -- it would crash the whole investigation.
    Reuses the same 30-change, 300-char-summary fixture shape as
    `test_changes_summaries_only_name_changes_still_present_after_trimming`
    above, which already proves the payload itself stays inside
    `MAX_RESULT_BYTES`; this test's own claim is narrower and different --
    that the returned `.summary` string specifically survives the trip
    through `build_evidence` into a real `Evidence` record."""
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": f"change-{index:03d}: " + "s" * 300,
                }
                for index in range(30)
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert len(outcome.summary) <= 400
    evidence = build_evidence(
        incident_id=INCIDENT_ID,
        kind=outcome.kind,
        source=outcome.source,
        observed_at=WINDOW_END,
        summary=outcome.summary,
        payload=outcome.payload,
    )
    assert evidence.summary == outcome.summary


def test_changes_summary_content_keeps_the_newest_text_not_the_oldest(
    tmp_path: Path,
) -> None:
    """Codex round-1 finding on lab-defect-fix F8. `payload["summaries"]` (the
    real change content, joined) is already ordered oldest-to-newest by the
    time it reaches the summary's content-display cap -- `changes.sort(key=
    _change_moment)` above, then `trim_to_bytes`'s own front-pop keeps the
    newest survivors. A HEAD slice of that joined string (`content[:budget]`,
    the pre-fix code) would keep the OLDEST of what's shown and silently
    discard the newest -- the opposite of every other truncation direction in
    this function (`trim_to_bytes` itself keeps the newest ROWS; this is about
    what survives once rows are joined into one display string). Both changes
    below are individually short enough that `trim_to_bytes` never drops
    either ROW -- `outcome.payload["truncated"]` stays `False` and both
    survive into `payload["changes"]` -- so this exercises ONLY the summary's
    own content-display cap, never the row-trimming path `test_changes_
    summaries_only_name_changes_still_present_after_trimming` above already
    covers.

    The content budget is derived from the real runtime prefix ("N recent
    changes on <service>"), the same way `run_changes_check` itself derives
    it, rather than a hand-picked magic size -- so this test does not
    silently drift out of sync if that prefix's shape ever changes. The older
    change's marker sits at the very START of its own text (so any nonzero
    head-drop excludes it) and the newer change's marker sits at the very END
    of its own text (so it is the last thing in the joined content, and the
    filler ahead of it is sized to fall comfortably inside the kept tail)."""
    paths = RunPaths(root=tmp_path)
    service = "orders"
    change_count = 2
    prefix = f"{change_count} recent changes on {service}"
    content_budget = 400 - len(prefix) - len(": ")
    keep = content_budget - len("...")

    newer_marker = "NEWER_MARKER"
    older_marker = "OLDER_MARKER"
    newer_summary = "b" * 5 + newer_marker
    # Pad the older change well past `keep` so the joined tail-kept slice
    # lands deep inside the "a" filler -- far from `older_marker`, which sits
    # at index 0 of `older_summary` -- regardless of the exact `keep` value.
    older_summary = older_marker + "a" * (keep + 50)

    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": WINDOW_START.isoformat(),
                    "service": service,
                    "summary": older_summary,
                },
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": service,
                    "summary": newer_summary,
                },
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service=service, window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.payload["truncated"] is False
    assert outcome.payload["change_count"] == change_count
    assert len(outcome.summary) <= 400
    assert newer_marker in outcome.summary
    assert older_marker not in outcome.summary


def test_a_moderately_long_service_name_shrinks_the_budget_but_shows_content(
    tmp_path: Path,
) -> None:
    """Codex round-1 finding. `arguments.service` has no schema `max_length`
    (`ListRecentChangesArguments`, `tools.py`), so a longer-than-usual service
    name must correctly shrink the content budget around itself rather than
    assume a fixed content cap always leaves the summary under 400 chars.

    Post-freeze review, correctness's P3 + simplicity's P2 (same gap, two
    angles). The original version of this test used a change summary (~57
    chars) far shorter than the computed budget for an 80-char service name
    (298 chars) -- it always landed in the "content fits whole, no
    truncation" branch and never exercised the tail-slice logic the
    docstring claimed to prove. This raw content is deliberately longer than
    the computed budget, so the tail-slice branch genuinely runs. The
    computed budget is derived here with the SAME formula `run_changes_check`
    uses (`400 - len(prefix) - len(": ")`), not asserted only as `<= 400`,
    so a mutation that hardcodes `content_budget` to a fixed wrong constant
    (confirmed by the correctness reviewer's own mutation test: reverting to
    a hardcoded `220` leaves this test's old assertions all still green)
    would make the exact-length and marker-survival assertions below fail."""
    paths = RunPaths(root=tmp_path)
    service = "svc-" + "x" * 76
    marker = "..."
    start_marker = "STARTMARKER-"
    end_marker = "-ENDMARKER"
    # Long enough that, once the computed budget for this service name is
    # subtracted from it, more than `len(marker)` characters must be cut
    # from the front -- guaranteeing the tail-slice branch, not the
    # fits-whole branch.
    change_summary = start_marker + "z" * 328 + end_marker
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": WINDOW_START.isoformat(),
                    "service": service,
                    "summary": change_summary,
                }
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service=service, window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    # Mirrors run_changes_check's own arithmetic exactly: one matched,
    # untruncated change, so `prefix` carries no "(truncated)" note.
    prefix = f"1 recent changes on {service}"
    content_budget = 400 - len(prefix) - len(": ")
    assert content_budget > len(marker), (
        "this fixture is meant to test the shrink-but-still-show-content "
        "case -- if this fails, the service name or content need retuning"
    )
    assert len(change_summary) > content_budget, (
        "the raw content must be longer than the computed budget, or the "
        "fits-whole branch runs instead of the tail-slice branch"
    )
    keep = content_budget - len(marker)
    expected_content = marker + change_summary[-keep:]
    expected_summary = f"{prefix}: {expected_content}"
    assert outcome.summary == expected_summary
    assert len(outcome.summary) == 400
    assert start_marker not in outcome.summary
    assert end_marker in outcome.summary


def test_a_pathologically_long_service_name_still_produces_valid_evidence(
    tmp_path: Path,
) -> None:
    """Codex's own reproduction, made permanent. Before this fix, a service
    name long enough that the fixed prefix alone exceeded `Evidence.summary`'s
    hard `max_length=400` bound (`domain.py`) crashed the whole investigation
    via an uncaught `AssertionError` -- `build_evidence` (`tool_wrappers.py`)
    constructs `Evidence` with no try/except around it by design, so this is
    not a safe failure. This service name alone (450 chars) already puts the
    fixed prefix ("N recent changes on <service>") over 400 chars before any
    change content is even considered -- the content budget goes negative,
    content is dropped entirely, and the final unconditional clamp is what
    keeps the summary under the bound. Routes the result through
    `build_evidence`, reusing the pattern `test_a_pathologically_long_
    changes_summary_still_builds_valid_evidence` above already established,
    to prove it doesn't raise -- not just that the string itself is short.

    Post-freeze review note: this service name is long enough (450 chars)
    that the fixed prefix alone (470 chars) already exceeds the final
    unconditional clamp's own 397-char cut point, before any change content
    is considered -- so this case cannot distinguish a correctly-computed
    negative `content_budget` from a wrong, hardcoded one: the final clamp
    truncates within the prefix either way, producing byte-identical output
    regardless of what `content_budget` was. That is precisely why the
    correctness reviewer's `content_budget = 220` mutation still passed this
    test's own original assertions. This test stays -- it is still real
    proof that an oversized prefix cannot crash the check via `build_
    evidence` -- but the arithmetic itself is pinned by the adjacent test
    below, `test_a_negative_content_budget_produces_the_exact_expected_
    summary`, whose service length is chosen specifically so the final
    clamp does NOT mask a wrong budget."""
    paths = RunPaths(root=tmp_path)
    service = "s" * 450
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": WINDOW_START.isoformat(),
                    "service": service,
                    "summary": "a real change happened here",
                }
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service=service, window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert len(outcome.summary) <= 400
    prefix = f"1 recent changes on {service}"
    content_budget = 400 - len(prefix) - len(": ")
    assert content_budget <= 0
    expected_summary = prefix[:397] + "..."
    assert outcome.summary == expected_summary
    evidence = build_evidence(
        incident_id=INCIDENT_ID,
        kind=outcome.kind,
        source=outcome.source,
        observed_at=WINDOW_END,
        summary=outcome.summary,
        payload=outcome.payload,
    )
    assert evidence.summary == outcome.summary


def test_a_negative_content_budget_produces_the_exact_expected_summary(
    tmp_path: Path,
) -> None:
    """Post-freeze review, correctness's P3 + simplicity's P2. Neither
    existing service-name test above can distinguish a correctly-computed
    `content_budget` from a wrong, hardcoded one: the moderate case (~80
    chars) used to use content far shorter than any budget, and the
    pathological case (450 chars) has its result masked by the final
    unconditional clamp regardless of the budget's value (see that test's
    own updated docstring).

    This service length (379 chars) is chosen precisely so `content_budget`
    computed from it (`400 - len(prefix) - len(": ")`) is a small negative
    number (-1) while `prefix` itself (399 chars) still fits under the
    400-char bound on its own -- so the final unconditional clamp never
    fires, and the summary is exactly `prefix`, unpadded. A mutation that
    hardcodes `content_budget` to a fixed positive constant (e.g. the
    correctness reviewer's own `220`) would instead take the "content fits
    whole" branch, append `": a real change happened here"` after the
    prefix, push the result to 429 chars, and get clamped to a DIFFERENT
    397-char cut -- verified directly against real production arithmetic in
    this docstring's own review round, not asserted on faith."""
    paths = RunPaths(root=tmp_path)
    service = "s" * 379
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": WINDOW_START.isoformat(),
                    "service": service,
                    "summary": "a real change happened here",
                }
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service=service, window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    prefix = f"1 recent changes on {service}"
    content_budget = 400 - len(prefix) - len(": ")
    assert content_budget <= 0, (
        "this fixture is meant to test the negative-budget, unclamped-"
        "prefix case -- if this fails, the service length needs retuning"
    )
    assert len(prefix) <= 400, (
        "the prefix itself must stay under the bound, or the final "
        "unconditional clamp masks the budget arithmetic this test exists "
        "to pin -- see the pathological-service-name test's docstring"
    )
    assert outcome.summary == prefix
    assert len(outcome.summary) == 399


def test_a_content_budget_of_exactly_three_still_shows_no_content(
    tmp_path: Path,
) -> None:
    """External review, P3. `run_changes_check`'s own comment on the
    `keep > 0` guard states the failure range as `0 < content_budget <=
    len(marker)` (correct) but glosses it in English as "content_budget of
    1 or 2" (wrong -- `marker` is 3 chars, so 3 is in range too). The math
    was always right; only the prose omitted the boundary. This test pins
    the omitted case directly: a service length (375 chars) chosen so
    `content_budget` (`400 - len(prefix) - len(": ")`) computes to exactly
    3 -- `content_budget > 0`, so this is NOT the negative-budget branch
    the sibling test above covers, but `keep = content_budget - len(marker)`
    is `3 - 3 == 0`, and `keep > 0` is still `False`, so content is still
    correctly suppressed. Without the `keep > 0` guard, `raw_content[-0:]`
    would return the WHOLE unclamped summaries string per the comment's own
    documented Python slicing gotcha -- this fixture is the one case that
    would actually demonstrate that regression, since `content_budget <= 0`
    fixtures never reach the `keep` arithmetic at all."""
    paths = RunPaths(root=tmp_path)
    service = "s" * 375
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": WINDOW_START.isoformat(),
                    "service": service,
                    "summary": "a real change happened here",
                }
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service=service, window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    prefix = f"1 recent changes on {service}"
    content_budget = 400 - len(prefix) - len(": ")
    assert content_budget == 3, (
        "this fixture is meant to test the exactly-three-byte budget "
        "boundary -- if this fails, the service length needs retuning"
    )
    assert outcome.summary == prefix
    assert len(outcome.summary) == 395


def test_changes_trim_drops_the_chronologically_oldest_not_the_first_declared(
    tmp_path: Path,
) -> None:
    """Lab-defect-fix Unit 2, W4/#4. `changes.json` can declare entries out
    of chronological order -- confirmed directly against this repo's own
    `configuration_change.json`, whose `require_order_token` entry
    (`offset_seconds: -60`, chronologically newer) is declared BEFORE its
    `image_rebuild` entry (`offset_seconds: -240`, chronologically older).
    A naive front-pop over declaration order would drop whichever entry
    happens to be declared first, which is not necessarily the oldest --
    here it would wrongly drop the newer, real one. `run_changes_check`
    sorts matched changes by their parsed `at` (ascending) before
    `trim_to_bytes` pops from the front, so trimming always drops the
    truly oldest change regardless of declaration order.

    Post-freeze review (three independent reviewers, same finding): both
    timestamps used to come from `.isoformat()` on datetimes sharing the
    same `tzinfo=UTC`, so both strings rendered with an identical `+00:00`
    offset -- a raw lexicographic string sort and the correct parsed-
    `datetime` sort agree on that shape, so this test could not actually
    tell them apart. Confirmed by mutation: reverting `_change_moment` to
    return the raw string instead of `datetime.fromisoformat(moment)`, all
    6 changes-related tests still passed. `newer_at` below is the SAME
    absolute instant as `WINDOW_START + timedelta(minutes=5)`, re-rendered
    in a `-05:00` offset instead of `+00:00` -- its string now starts
    `"...T05:05..."`, which sorts lexicographically BEFORE `older_at`'s
    `"...T10:00..."`, the opposite of true chronological order. A raw
    string sort would therefore keep `older_at` (wrong); the real,
    datetime-based sort keeps `newer_at` (correct) -- exactly the
    divergence needed to catch a regression back to string comparison."""
    paths = RunPaths(root=tmp_path)
    newer_at = "2026-08-16T05:05:00-05:00"
    assert datetime.fromisoformat(newer_at) == WINDOW_START + timedelta(minutes=5)
    older_at = WINDOW_START.isoformat()
    assert newer_at < older_at, (
        "the fixture must string-sort backwards to prove anything"
    )
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": newer_at,
                    "service": "orders",
                    "summary": "newer: " + "n" * 4000,
                },
                {
                    "at": older_at,
                    "service": "orders",
                    "summary": "older: " + "o" * 4000,
                },
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True
    assert outcome.payload["change_count"] == 1
    assert "newer" in str(outcome.payload["summaries"])
    assert "older" not in str(outcome.payload["summaries"])


def test_recent_changes_are_filtered_by_service_and_window(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": "configuration update: require_order_token enabled",
                },
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "inventory",
                    "summary": "image rebuild",
                },
                {
                    "at": (WINDOW_START - timedelta(days=1)).isoformat(),
                    "service": "orders",
                    "summary": "older change",
                },
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.payload["change_count"] == 1
    assert "require_order_token" in str(outcome.payload["summaries"])
    # Lab-defect-fix F8. `render_context` (`prompts.py`) only ever shows the
    # model `Evidence.summary`, never `.payload` -- a model that correctly
    # calls `list_recent_changes` and gets the right evidence still could
    # not see what it said until the change content itself reached this
    # field too, not just the structured payload.
    assert "require_order_token" in outcome.summary


def test_topology_reads_the_run_manifest(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    paths.topology_file.write_text(
        json.dumps({"services": list(SERVICES), "edges": ["gateway>orders"]}),
        encoding="utf-8",
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["edge_count"] == 1
    assert "1 service edges" in outcome.summary
    assert outcome.payload["truncated"] is False
    assert not outcome.summary.endswith("(truncated)")


def test_topology_summary_names_truncation_when_the_payload_was_cut(
    tmp_path: Path,
) -> None:
    """Round 6 review, the P1. Same gap as the metric case: `payload
    ["truncated"]` already carried whether trimming cut this topology, but
    `run_topology_check`'s summary string never rendered it -- the only
    part of a `CheckOutcome` `prompts.py`'s `render_context` puts in front
    of the model."""
    paths = RunPaths(root=tmp_path)
    edges = [f"service-{i}>service-{i + 1}" for i in range(600)]
    paths.topology_file.write_text(
        json.dumps({"services": [], "edges": edges}), encoding="utf-8"
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert outcome.payload["truncated"] is True
    assert outcome.summary.endswith("(truncated)")


def test_topology_summary_reports_the_post_trim_edge_count(tmp_path: Path) -> None:
    """Post-freeze review. `run_topology_check`'s summary string used to
    read `len(edge_list)` (the PRE-trim count) instead of
    `payload['edge_count']` (kept honest post-trim by `trim_to_bytes`,
    Unit 3b-4 addendum's C3) -- the same bug already fixed this round in
    `run_logs_check`/`run_changes_check`/`run_metric_check`. Unlike the
    metric case, this one is genuinely reachable with real data: many
    long edge strings really do exceed the byte budget without needing a
    monkeypatch."""
    paths = RunPaths(root=tmp_path)
    edges = [f"service-{i}>service-{i + 1}" for i in range(600)]
    paths.topology_file.write_text(
        json.dumps({"services": [], "edges": edges}), encoding="utf-8"
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert outcome.payload["truncated"] is True
    kept_edge_count = outcome.payload["edge_count"]
    assert isinstance(kept_edge_count, int)
    assert kept_edge_count < len(edges)
    assert f"{kept_edge_count} service edges" in outcome.summary
    assert f"{len(edges)} service edges" not in outcome.summary


def test_an_oversized_services_list_still_fits_the_byte_bound(tmp_path: Path) -> None:
    """The services-list fix. `services` is a LIST, not the string-valued
    scalar `trim_to_bytes`'s own fallback (C3) can shrink, and it is not
    `edges`, the row list `run_topology_check`'s other `trim_to_bytes`
    call already bounds -- before this fix, an oversized `services` list
    would pass through both mechanisms untouched and blow the byte
    budget. Reachable with real data, same as the edges case above: many
    long service names really do exceed it. Not reachable through any of
    the four shipped lab topologies (all under 100 bytes total) -- this
    is a defensive bound, not a scenario this project's own lab can
    trigger, per the owner's P3 severity ruling.

    Round 6 review, item 3. This test used to pair the oversized `services`
    list with an EMPTY `edges` list (`"edges": []`) -- a degenerate shape
    that pinned nothing about what happens to a genuinely non-empty `edges`
    list under the same oversized-services condition, which is exactly the
    kind of gap that let the round-3 regression (F1: an all-`services`,
    no-`edges` case masking what a real, non-empty `edges` list does) ship
    unnoticed. `edges` here is a handful of real, short edges -- reproduced
    directly against the real function before writing this assertion (not
    assumed): under the shipped edges-first call order, `edges`-first
    pops the whole (small, cheap) `edges` list to empty without ever
    bringing the still-huge `services`-dominated payload under budget, so
    `edge_count` goes to zero; `services`, trimmed second against a payload
    that no longer carries any `edges` weight, still comes away non-empty
    and reduced. That is the documented tradeoff (`edges` absorbs the loss
    so the short, bounded `services` list does not) -- pinned explicitly
    here rather than left implicit via an empty list nobody could tell
    apart from "not tested"."""
    paths = RunPaths(root=tmp_path)
    services = [f"service-with-a-long-descriptive-name-{i}" for i in range(400)]
    edges = [f"edge-{i}" for i in range(8)]
    paths.topology_file.write_text(
        json.dumps({"services": services, "edges": edges}), encoding="utf-8"
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    assert outcome.payload["truncated"] is True
    kept_service_count = outcome.payload["service_count"]
    assert isinstance(kept_service_count, int)
    assert 0 < kept_service_count < len(services)
    assert kept_service_count == len(outcome.payload["services"])  # type: ignore[arg-type]
    # `edges` is sacrificed to zero under the shipped edges-first order --
    # small enough (8 real edges) to lose entirely while `services` (the
    # field actually responsible for the overage) still survives trimmed
    # rather than wiped. See this test's own docstring for why.
    assert outcome.payload["edge_count"] == 0


def test_a_small_services_list_survives_an_oversized_edges_list(tmp_path: Path) -> None:
    """Round 4 review. The services-list fix above was tested only against
    the case it was built for -- an oversized `services` list on its own --
    and shipped with `trim_to_bytes(payload, "services", ...)` called
    BEFORE the `"edges"` call. That order broke the realistic asymmetric
    case: a handful of real service names next to a genuinely oversized
    `edges` list. `trim_to_bytes` pops rows from its own list
    unconditionally until `fits(payload)`, so the `services` call never
    saw `fits()` turn true until it had popped `services` to EMPTY --
    `edges`, the field actually responsible for the overage, came away
    barely trimmed. An incident with 4 real services would have reported
    "0 services" to the investigator. The fix is call order: `edges`
    trims first, protecting the short, bounded `services` list at the
    expense of `edges`, the field this codebase's real data actually
    grows without bound. This test fails under the old services-first
    order (kept_service_count == 0) and passes under the fixed
    edges-first order."""
    paths = RunPaths(root=tmp_path)
    services = ["payments-api", "checkout-web", "orders-db", "gateway"]
    edges = [f"edge-{i}-" + "z" * 30 for i in range(400)]
    paths.topology_file.write_text(
        json.dumps({"services": services, "edges": edges}), encoding="utf-8"
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert len(json.dumps(outcome.payload).encode("utf-8")) <= MAX_RESULT_BYTES
    kept_service_count = outcome.payload["service_count"]
    assert isinstance(kept_service_count, int)
    assert kept_service_count == len(services)
    assert kept_service_count == len(outcome.payload["services"])  # type: ignore[arg-type]


def test_a_missing_manifest_is_unavailable_rather_than_a_crash(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)

    topology = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)
    changes = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert topology.outcome is ToolOutcome.UNAVAILABLE
    assert changes.outcome is ToolOutcome.UNAVAILABLE


def test_the_runner_sends_each_tool_to_its_own_backend(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(paths, [log_row(1)])
    paths.topology_file.write_text(json.dumps({"edges": []}), encoding="utf-8")
    run = _registered_check_runner(paths, "http://127.0.0.1:1", 1)

    logs = run(
        ToolProposal(
            arguments=logs_arguments(), evidence_gap="gap", expected_observation="rows"
        ),
        incident_scope(),
    )
    topology = run(
        ToolProposal(
            arguments=GetTopologyArguments(incident_id=INCIDENT_ID),
            evidence_gap="gap",
            expected_observation="edges",
        ),
        incident_scope(),
    )
    metric = run(
        ToolProposal(
            arguments=metric_arguments(),
            evidence_gap="gap",
            expected_observation="samples",
        ),
        incident_scope(),
    )

    assert logs.kind is EvidenceKind.LOG
    assert topology.kind is EvidenceKind.TOPOLOGY
    assert metric.kind is EvidenceKind.METRIC


def test_the_runner_raises_loudly_for_search_runbooks(tmp_path: Path) -> None:
    """This seam predates `search_runbooks` and is superseded by
    `tool_wrappers.dispatch_registry` -- nothing in `cli.py` calls it. Its
    own `RunCheck` return type is `CheckOutcome` only, which cannot express
    `RunbookCheckOutcome`, so a `search_runbooks` proposal must raise here
    rather than silently falling through to `run_topology_check` the way an
    unconditional last branch would have."""
    paths = RunPaths(root=tmp_path)
    run = _registered_check_runner(paths, "http://127.0.0.1:1", 1)

    with pytest.raises(ValueError, match="SearchRunbooksArguments"):
        run(
            ToolProposal(
                arguments=SearchRunbooksArguments(
                    topic=RunbookTopic.GATEWAY_ERRORS, limit=3
                ),
                evidence_gap="gap",
                expected_observation="a passage",
            ),
            incident_scope(),
        )


# Cross-incident isolation
#
# A run backend never receives an incident_id argument for logs or changes; it
# only ever sees the RunPaths of the run it was handed. These tests build two
# separate incident directories with distinguishable content and confirm a
# check pointed at one run's paths never surfaces the other run's content.

OTHER_INCIDENT_ID = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"


def other_incident_scope() -> IncidentScope:
    return incident_scope().model_copy(update={"incident_id": OTHER_INCIDENT_ID})


def test_a_log_check_never_surfaces_another_incidents_rows(tmp_path: Path) -> None:
    paths_a = RunPaths(root=tmp_path / "incident-a")
    paths_b = RunPaths(root=tmp_path / "incident-b")
    write_log(paths_a, [log_row(1, detail="incident-a-only-marker")])
    write_log(paths_b, [log_row(1, detail="incident-b-only-marker")])

    outcome = run_logs_check(logs_arguments(), paths_a)

    dumped = json.dumps(outcome.payload)
    assert "incident-a-only-marker" in dumped
    assert "incident-b-only-marker" not in dumped


def write_changes(paths: RunPaths, summary: str) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": summary,
                }
            ]
        ),
        encoding="utf-8",
    )


def test_a_changes_check_never_surfaces_another_incidents_entries(
    tmp_path: Path,
) -> None:
    paths_a = RunPaths(root=tmp_path / "incident-a")
    paths_b = RunPaths(root=tmp_path / "incident-b")
    write_changes(paths_a, "incident-a-only-change")
    write_changes(paths_b, "incident-b-only-change")

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths_a,
    )

    dumped = json.dumps(outcome.payload)
    assert "incident-a-only-change" in dumped
    assert "incident-b-only-change" not in dumped


def test_w5_narrows_only_the_two_rate_templates_the_finding_measured(
    fake_prometheus: RecordingPrometheus,
) -> None:
    """Lab-defect-fix Unit 2, W5. `GATEWAY_ERROR_RATE`/`DOWNSTREAM_TIMEOUT_RATE`
    render `[30s]`; `GATEWAY_LATENCY_P95` -- whose own `rate(...)` feeds
    `histogram_quantile`, a different computation this unit does not touch
    -- still renders `[1m]`, confirmed directly rather than assumed from
    the plan's own "two templates" phrasing."""
    run_metric_check(
        metric_arguments(template=MetricTemplate.GATEWAY_ERROR_RATE),
        incident_scope(),
        fake_prometheus.url,
        5,
    )
    run_metric_check(
        metric_arguments(template=MetricTemplate.DOWNSTREAM_TIMEOUT_RATE),
        incident_scope(),
        fake_prometheus.url,
        5,
    )
    run_metric_check(
        metric_arguments(template=MetricTemplate.GATEWAY_LATENCY_P95),
        incident_scope(),
        fake_prometheus.url,
        5,
    )

    error_query, timeout_query, latency_query = fake_prometheus.queries[-3:]
    assert "[30s]" in error_query
    assert "[1m]" not in error_query
    assert "[30s]" in timeout_query
    assert "[1m]" not in timeout_query
    assert "[1m]" in latency_query
    assert "[30s]" not in latency_query


def test_the_metric_query_label_is_derived_from_the_scope_not_an_argument(
    fake_prometheus: RecordingPrometheus,
) -> None:
    run_metric_check(metric_arguments(), incident_scope(), fake_prometheus.url, 5)
    run_metric_check(metric_arguments(), other_incident_scope(), fake_prometheus.url, 5)

    query_a, query_b = fake_prometheus.queries[-2], fake_prometheus.queries[-1]
    assert f'incident="{incident_scope().incident_id}"' in query_a
    assert f'incident="{other_incident_scope().incident_id}"' in query_b


def test_topology_decides_from_paths_not_the_argument_incident_id(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    paths.topology_file.write_text(
        json.dumps({"services": list(SERVICES), "edges": ["gateway>orders"]}),
        encoding="utf-8",
    )

    outcome = run_topology_check(
        GetTopologyArguments(incident_id="not-the-real-one"), paths
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["edge_count"] == 1


# Template and filter coverage
#
# The tests above only exercise one metric template and one log filter. These
# confirm the remaining registered templates and filters run end to end.


def test_the_downstream_timeout_rate_template_executes(
    fake_prometheus: RecordingPrometheus,
) -> None:
    arguments = metric_arguments(template=MetricTemplate.DOWNSTREAM_TIMEOUT_RATE)

    outcome = run_metric_check(arguments, incident_scope(), fake_prometheus.url, 5)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["template"] == MetricTemplate.DOWNSTREAM_TIMEOUT_RATE.value


def test_the_resource_pool_attempts_per_capacity_template_executes(
    fake_prometheus: RecordingPrometheus,
) -> None:
    arguments = metric_arguments(
        template=MetricTemplate.RESOURCE_POOL_ATTEMPTS_PER_CAPACITY
    )

    outcome = run_metric_check(arguments, incident_scope(), fake_prometheus.url, 5)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert (
        outcome.payload["template"]
        == MetricTemplate.RESOURCE_POOL_ATTEMPTS_PER_CAPACITY.value
    )


def test_the_gateway_latency_p95_template_executes(
    fake_prometheus: RecordingPrometheus,
) -> None:
    arguments = metric_arguments(template=MetricTemplate.GATEWAY_LATENCY_P95)

    outcome = run_metric_check(arguments, incident_scope(), fake_prometheus.url, 5)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["template"] == MetricTemplate.GATEWAY_LATENCY_P95.value


def test_the_timeouts_only_filter_matches_timeout_events(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [
            log_row(1, event="upstream_timeout"),
            log_row(2, event="config_rejected_request"),
        ],
    )

    outcome = run_logs_check(logs_arguments(log_filter=LogFilter.TIMEOUTS_ONLY), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
    assert outcome.payload["event_codes"] == "upstream_timeout"


def test_the_pool_exhaustion_filter_matches_pool_exhausted_events(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [
            log_row(1, event="pool_exhausted"),
            log_row(2, event="upstream_timeout"),
        ],
    )

    outcome = run_logs_check(
        logs_arguments(log_filter=LogFilter.POOL_EXHAUSTION), paths
    )

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
    assert outcome.payload["event_codes"] == "pool_exhausted"


def test_the_config_reload_filter_matches_config_loaded_events(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path)
    write_log(
        paths,
        [
            log_row(1, event="config_loaded"),
            log_row(2, event="upstream_timeout"),
        ],
    )

    outcome = run_logs_check(logs_arguments(log_filter=LogFilter.CONFIG_RELOAD), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
    assert outcome.payload["event_codes"] == "config_loaded"


def test_read_json_line_rejects_non_finite_tokens() -> None:
    """Round 8 review (codex). `json.loads` accepts the non-standard
    `NaN`/`Infinity`/`-Infinity` tokens by default -- an extension beyond
    RFC-8259 most other JSON readers reject. Without `parse_constant`
    refusing them, a NaN token anywhere in a log line would parse into an
    ordinary-looking Python `nan` float instead of the line being refused
    the same way any other malformed line already is."""
    assert read_json_line('{"value": NaN}') is None
    assert read_json_line('{"value": Infinity}') is None
    assert read_json_line('{"value": -Infinity}') is None
    assert read_json_line('{"value": 1}') == {"value": 1}


def test_read_json_file_rejects_non_finite_tokens(tmp_path: Path) -> None:
    """Same claim as the line-reader test above, for the manifest reader
    `run_changes_check`/`run_topology_check` both use."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"edges": [], "weight": NaN}', encoding="utf-8")
    assert read_json_file(manifest) is None

    manifest.write_text('{"edges": [], "weight": 1}', encoding="utf-8")
    assert read_json_file(manifest) == {"edges": [], "weight": 1}


def test_json_readers_reject_overflowing_numeric_literals(tmp_path: Path) -> None:
    """A raw JSON numeral takes `parse_float`, not `parse_constant`."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"weight": 1e400}', encoding="utf-8")

    assert read_json_line('{"weight": 1e400}') is None
    assert read_json_file(manifest) is None


def test_a_log_line_with_a_non_finite_token_is_skipped_not_crashed(
    tmp_path: Path,
) -> None:
    """Round 8 review, codex finding, independently reproduced. Before this
    fix, `read_json_line`'s bare `json.loads` accepted a NaN/Infinity token
    anywhere in a log line and returned an ordinary-looking record carrying
    a Python `nan`/`inf` float. Confirms the poisoned line is now skipped
    the same way any other malformed line already is (`read_json_line`
    returning `None`, `continue`d over), not that the whole check fails or
    that the bad value silently reaches `payload["rows"]`."""
    paths = RunPaths(root=tmp_path)
    poisoned_row: dict[str, object] = {
        "at": (WINDOW_START + timedelta(seconds=2)).isoformat(),
        "request_id": "r2",
        "service": "orders",
        "severity": "error",
        "event": "config_rejected_request",
        "fields": {"config_key": "require_order_token", "detail": float("nan")},
    }
    write_log(paths, [log_row(1), poisoned_row])

    outcome = run_logs_check(logs_arguments(), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1


def test_a_log_line_with_an_overflowing_numeric_literal_is_skipped(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    paths.logs.mkdir()
    overflowing_row = json.dumps(log_row(2)).rsplit("}", 1)[0] + ',"weight":1e400}'
    paths.logs.joinpath("orders.jsonl").write_text(
        json.dumps(log_row(1)) + "\n" + overflowing_row + "\n",
        encoding="utf-8",
    )

    outcome = run_logs_check(logs_arguments(), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1


def test_a_changes_manifest_with_a_non_finite_token_is_refused(tmp_path: Path) -> None:
    """Same codex finding as the log-line test above, for `run_changes_
    check`'s whole-manifest reader (`read_json_file`): the non-standard
    token can sit anywhere in the file, not just in a field this function
    reads, so the entire manifest becomes unreadable rather than one row
    being silently poisoned."""
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        json.dumps(
            [
                {
                    "at": (WINDOW_START + timedelta(minutes=1)).isoformat(),
                    "service": "orders",
                    "summary": "change",
                    "risk_score": float("nan"),
                }
            ]
        ),
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_a_changes_manifest_with_an_overflowing_numeric_literal_is_refused(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_text(
        '[{"at":"2026-01-01T00:01:00+00:00","service":"orders",'
        '"summary":"change","risk_score":1e400}]',
        encoding="utf-8",
    )

    outcome = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_a_topology_manifest_with_a_non_finite_token_is_refused(tmp_path: Path) -> None:
    """Same codex finding, for `run_topology_check`'s manifest reader."""
    paths = RunPaths(root=tmp_path)
    paths.topology_file.write_text(
        json.dumps(
            {
                "services": list(SERVICES),
                "edges": ["gateway>orders"],
                "weight": float("inf"),
            }
        ),
        encoding="utf-8",
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_a_topology_manifest_with_an_overflowing_numeric_literal_is_refused(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    paths.topology_file.write_text(
        '{"services":["gateway","orders"],"edges":["gateway>orders"],"weight":1e400}',
        encoding="utf-8",
    )

    outcome = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_malformed_utf8_manifest_is_unavailable(tmp_path: Path) -> None:
    """Manifest decoding is intentionally unavailable, not an uncaught failure."""
    paths = RunPaths(root=tmp_path)
    paths.changes_file.write_bytes(b"[\xff]")
    paths.topology_file.write_bytes(b"{\xff}")

    changes = run_changes_check(
        ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        paths,
    )
    topology = run_topology_check(GetTopologyArguments(incident_id=INCIDENT_ID), paths)

    assert changes.outcome is ToolOutcome.UNAVAILABLE
    assert changes.reason_code is ReasonCode.TOOL_UNAVAILABLE
    assert topology.outcome is ToolOutcome.UNAVAILABLE
    assert topology.reason_code is ReasonCode.TOOL_UNAVAILABLE


def test_malformed_utf8_log_line_is_skipped_while_valid_lines_execute(
    tmp_path: Path,
) -> None:
    paths = RunPaths(root=tmp_path)
    paths.logs.mkdir()
    paths.logs.joinpath("orders.jsonl").write_bytes(
        json.dumps(log_row(1)).encode("utf-8") + b"\n" + b"{\xff}\n"
    )

    outcome = run_logs_check(logs_arguments(), paths)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.payload["row_count"] == 1
