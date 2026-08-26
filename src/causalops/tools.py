"""The five registered read-only investigator tools and their typed arguments.

The model picks a tool and a registered template or filter. Application code turns
that choice into a real query, so raw PromQL, SQL, shell, paths, and URLs have no
representation here and cannot be proposed.
"""

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field

TOOL_REGISTRY_VERSION = "3"


def to_utc(moment: datetime) -> datetime:
    return moment.astimezone(UTC)


# Every timestamp field in the project shares this type. A naive value is rejected
# rather than compared against an aware one, and two spellings of the same instant
# normalize to one, so they cannot produce two different fingerprints. It lives in
# this module because it is the first one imported, not because it is about tools.
UtcDatetime = Annotated[AwareDatetime, AfterValidator(to_utc)]


class ToolName(StrEnum):
    QUERY_METRIC = "query_metric"
    QUERY_LOGS = "query_logs"
    LIST_RECENT_CHANGES = "list_recent_changes"
    GET_TOPOLOGY = "get_topology"
    SEARCH_RUNBOOKS = "search_runbooks"


class MetricTemplate(StrEnum):
    GATEWAY_ERROR_RATE = "gateway_error_rate"
    GATEWAY_LATENCY_P95 = "gateway_latency_p95"
    DOWNSTREAM_TIMEOUT_RATE = "downstream_timeout_rate"
    RESOURCE_POOL_IN_USE = "resource_pool_in_use"


class LogFilter(StrEnum):
    ERRORS_ONLY = "errors_only"
    TIMEOUTS_ONLY = "timeouts_only"
    POOL_EXHAUSTION = "pool_exhaustion"
    CONFIG_RELOAD = "config_reload"


class RunbookTopic(StrEnum):
    """The closed set of guidance topics `search_runbooks` may request.

    Milestone 3's owner decision, taken over the plan's original "sanitize a
    free-text query" framing: this module's own docstring already promises
    "raw ... queries ... have no representation here and cannot be
    proposed," and a free-text `query` field would have made `search_runbooks`
    the one tool that broke that promise instead of merely constraining it.
    A closed enum keeps the promise literally true -- there is no FTS5 MATCH
    syntax for the model to write, so there is nothing for an injected
    document or a malicious model turn to smuggle through it -- and the
    small, curated corpus this topic set searches has no long tail a fixed
    handful of topics can't cover.
    """

    GATEWAY_ERRORS = "gateway_errors"
    GATEWAY_LATENCY = "gateway_latency"
    DOWNSTREAM_TIMEOUTS = "downstream_timeouts"
    RESOURCE_POOL_PRESSURE = "resource_pool_pressure"
    RECENT_CONFIG_CHANGES = "recent_config_changes"


class QueryMetricArguments(BaseModel):
    # `extra="forbid"` rationale, stated once here and cross-referenced from
    # every other class that carries it: pydantic's default (`extra="ignore"`)
    # silently drops a field the model sends that this schema does not
    # declare -- the model gets no signal its argument was dropped, and the
    # application validates and acts on a call that is not what was actually
    # sent. `extra="forbid"` turns that into an explicit `ValidationError`
    # instead, which reaches the model as a named repair (`graph.py`'s
    # existing structured-output repair path) rather than a silent partial
    # acceptance.
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: Literal[ToolName.QUERY_METRIC] = ToolName.QUERY_METRIC
    template: MetricTemplate
    service: str
    # Lab-defect-fix Unit 3, W1. Optional, not required: the model that just
    # wants "the incident" -- every observed case -- no longer has to
    # retype the window verbatim. `tool_wrappers.resolve_effective_window`
    # resolves an omitted bound to the matching scope boundary and narrows
    # (never widens) a supplied one before dispatch, so a backend or
    # `authorize()` reached through the ordinary wrapper path never sees
    # `None` here -- see `policy.authorize`'s own docstring for the direct
    # (non-wrapper) call contract this optionality also has to cover.
    window_start: UtcDatetime | None = Field(
        default=None,
        description="Defaults to the start of the incident window when omitted.",
    )
    window_end: UtcDatetime | None = Field(
        default=None,
        description="Defaults to the end of the incident window when omitted. "
        "A window outside the incident is narrowed to fit it.",
    )


class QueryLogsArguments(BaseModel):
    # `extra="forbid"` rationale: see `QueryMetricArguments` above.
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: Literal[ToolName.QUERY_LOGS] = ToolName.QUERY_LOGS
    log_filter: LogFilter
    service: str
    # Lab-defect-fix Unit 3, W1. See `QueryMetricArguments.window_start`
    # above for the full rationale -- identical here.
    window_start: UtcDatetime | None = Field(
        default=None,
        description="Defaults to the start of the incident window when omitted.",
    )
    window_end: UtcDatetime | None = Field(
        default=None,
        description="Defaults to the end of the incident window when omitted. "
        "A window outside the incident is narrowed to fit it.",
    )
    # The schema allows a wider range than the budget so that an oversized request
    # is a policy decision with a reason code, not a silent schema rejection.
    row_limit: int = Field(ge=1, le=200)


class ListRecentChangesArguments(BaseModel):
    # `extra="forbid"` rationale: see `QueryMetricArguments` above.
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: Literal[ToolName.LIST_RECENT_CHANGES] = ToolName.LIST_RECENT_CHANGES
    service: str
    # Lab-defect-fix Unit 3, W1. See `QueryMetricArguments.window_start`
    # above for the full rationale -- identical here.
    window_start: UtcDatetime | None = Field(
        default=None,
        description="Defaults to the start of the incident window when omitted.",
    )
    window_end: UtcDatetime | None = Field(
        default=None,
        description="Defaults to the end of the incident window when omitted. "
        "A window outside the incident is narrowed to fit it.",
    )


class GetTopologyArguments(BaseModel):
    # `extra="forbid"` rationale: see `QueryMetricArguments` above.
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: Literal[ToolName.GET_TOPOLOGY] = ToolName.GET_TOPOLOGY
    incident_id: str


class SearchRunbooksArguments(BaseModel):
    """Not incident-scoped: a runbook topic has no service or time window,
    only a topic and how many passages are worth returning. `limit`
    follows `QueryLogsArguments.row_limit`'s own pattern -- the schema
    allows a wider range than the budget, so an oversized request is a
    policy decision with a reason code (`policy.py`'s new branch), not a
    silent schema rejection."""

    # `extra="forbid"` rationale: see `QueryMetricArguments` above.
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: Literal[ToolName.SEARCH_RUNBOOKS] = ToolName.SEARCH_RUNBOOKS
    topic: RunbookTopic
    # `le=20` is deliberately not tied to `Budgets.runbook_passages` (5): a
    # schema bound is a hard shape limit, a budget is the number policy
    # actually allows through, and the two must stay independently
    # editable, the same separation `QueryLogsArguments.row_limit`'s `le=200`
    # against `Budgets.log_rows`'s default of 40 already establishes. 20 is
    # a headroom multiple of the current default (4x, versus row_limit's
    # 5x), not a value derived from the corpus's own size (ten passages
    # today) -- the corpus can grow without this bound needing to change.
    limit: int = Field(ge=1, le=20)


# This union is the tool registry. A second lookup table would be a competing
# source of truth about which tools and arguments exist.
ToolArguments = Annotated[
    QueryMetricArguments
    | QueryLogsArguments
    | ListRecentChangesArguments
    | GetTopologyArguments
    | SearchRunbooksArguments,
    Field(discriminator="tool"),
]


def fingerprint(arguments: ToolArguments) -> str:
    """Stable hash of a tool and its arguments, used to catch a repeated proposal.

    Pydantic serializes fields in declaration order, so the same request always
    produces the same hash regardless of the order the model wrote them in.
    """
    return sha256(arguments.model_dump_json().encode("utf-8")).hexdigest()
