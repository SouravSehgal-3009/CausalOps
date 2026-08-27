from datetime import datetime

import pytest
from fake_incident import WINDOW_END, WINDOW_START, logs_proposal, metric_proposal
from pydantic import TypeAdapter, ValidationError

from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
    RunbookTopic,
    SearchRunbooksArguments,
    ToolArguments,
    ToolName,
    fingerprint,
)

arguments_adapter: TypeAdapter[ToolArguments] = TypeAdapter(ToolArguments)


def every_tool() -> dict[ToolName, ToolArguments]:
    return {
        ToolName.QUERY_METRIC: QueryMetricArguments(
            template=MetricTemplate.GATEWAY_ERROR_RATE,
            service="gateway",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        ),
        ToolName.QUERY_LOGS: QueryLogsArguments(
            log_filter=LogFilter.ERRORS_ONLY,
            service="orders",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            row_limit=10,
        ),
        ToolName.LIST_RECENT_CHANGES: ListRecentChangesArguments(
            service="orders", window_start=WINDOW_START, window_end=WINDOW_END
        ),
        ToolName.GET_TOPOLOGY: GetTopologyArguments(incident_id="inc-1"),
        ToolName.SEARCH_RUNBOOKS: SearchRunbooksArguments(
            topic=RunbookTopic.GATEWAY_ERRORS, limit=3
        ),
    }


def test_every_registered_tool_round_trips_through_the_union() -> None:
    registered = every_tool()

    assert set(registered) == set(ToolName)
    for name, arguments in registered.items():
        restored = arguments_adapter.validate_python(arguments.model_dump(mode="json"))
        assert restored.tool is name


def test_an_unregistered_tool_is_rejected() -> None:
    with pytest.raises(ValidationError):
        arguments_adapter.validate_python({"tool": "run_shell", "command": "whoami"})


def test_an_unregistered_template_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryMetricArguments(
            template="sum(rate(http_requests_total[5m]))",  # type: ignore[arg-type]
            service="gateway",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        )


def test_a_row_limit_outside_the_schema_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        logs_proposal(row_limit=0)
    with pytest.raises(ValidationError):
        logs_proposal(row_limit=500)


def test_the_same_request_always_has_the_same_fingerprint() -> None:
    first = metric_proposal()
    second = metric_proposal()

    assert fingerprint(first.arguments) == fingerprint(second.arguments)
    assert fingerprint(first.arguments) != fingerprint(logs_proposal().arguments)


def test_field_order_does_not_change_the_fingerprint() -> None:
    written_one_way = metric_proposal().arguments.model_dump(mode="json")
    written_another_way = dict(reversed(list(written_one_way.items())))

    assert fingerprint(
        arguments_adapter.validate_python(written_one_way)
    ) == fingerprint(arguments_adapter.validate_python(written_another_way))


def test_a_timestamp_without_a_timezone_is_rejected() -> None:
    """A naive value would later be compared against an aware incident window."""
    with pytest.raises(ValidationError):
        QueryMetricArguments(
            template=MetricTemplate.GATEWAY_ERROR_RATE,
            service="gateway",
            window_start=datetime(2026, 8, 16, 10, 0),
            window_end=datetime(2026, 8, 16, 10, 30),
        )


def test_two_spellings_of_one_instant_share_a_fingerprint() -> None:
    """Otherwise the same check could be re-run by re-spelling the offset."""
    in_utc = arguments_adapter.validate_python(
        {
            "tool": "query_metric",
            "template": "gateway_latency_p95",
            "service": "gateway",
            "window_start": "2026-08-16T10:00:00+00:00",
            "window_end": "2026-08-16T10:30:00+00:00",
        }
    )
    in_another_offset = arguments_adapter.validate_python(
        {
            "tool": "query_metric",
            "template": "gateway_latency_p95",
            "service": "gateway",
            "window_start": "2026-08-16T12:00:00+02:00",
            "window_end": "2026-08-16T12:30:00+02:00",
        }
    )

    assert fingerprint(in_utc) == fingerprint(in_another_offset)


def test_a_different_service_changes_the_fingerprint() -> None:
    assert fingerprint(metric_proposal("gateway").arguments) != fingerprint(
        metric_proposal("orders").arguments
    )


def test_an_omitted_window_defaults_to_none_and_round_trips() -> None:
    """`window_start`/`window_end` became
    optional so the model can ask for "the incident" without retyping its
    window verbatim -- `tool_wrappers.resolve_effective_window` is what
    turns an omitted bound into a real one before dispatch, not this
    schema. This only proves the schema itself: omitting both fields
    leaves them `None`, and that `None` survives a JSON round trip rather
    than being silently coerced into something else."""
    for arguments in (
        QueryMetricArguments(
            template=MetricTemplate.GATEWAY_ERROR_RATE, service="gateway"
        ),
        QueryLogsArguments(
            log_filter=LogFilter.ERRORS_ONLY, service="orders", row_limit=20
        ),
        ListRecentChangesArguments(service="orders"),
    ):
        assert arguments.window_start is None
        assert arguments.window_end is None
        restored = arguments_adapter.validate_json(arguments.model_dump_json())
        assert restored.window_start is None  # type: ignore[union-attr]
        assert restored.window_end is None  # type: ignore[union-attr]


def test_extra_forbid_still_rejects_an_unknown_field() -> None:
    """The new optional window fields must not have loosened `extra="forbid"`
    -- an unrecognized field is still a validation error, not silently
    dropped, so a model's argument that fails to land is reported as a
    named repair rather than a quiet partial acceptance (see
    `QueryMetricArguments`'s own `extra="forbid"` rationale comment)."""
    with pytest.raises(ValidationError):
        QueryMetricArguments(
            template=MetricTemplate.GATEWAY_ERROR_RATE,
            service="gateway",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            unknown_field="whatever",  # type: ignore[call-arg]
        )
