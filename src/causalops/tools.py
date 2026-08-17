"""The four registered read-only investigator tools and their typed arguments.

The model picks a tool and a registered template or filter. Application code turns
that choice into a real query, so raw PromQL, SQL, shell, paths, and URLs have no
representation here and cannot be proposed.
"""

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field

TOOL_REGISTRY_VERSION = "1"


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


class QueryMetricArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: Literal[ToolName.QUERY_METRIC] = ToolName.QUERY_METRIC
    template: MetricTemplate
    service: str
    window_start: UtcDatetime
    window_end: UtcDatetime


class QueryLogsArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: Literal[ToolName.QUERY_LOGS] = ToolName.QUERY_LOGS
    log_filter: LogFilter
    service: str
    window_start: UtcDatetime
    window_end: UtcDatetime
    # The schema allows a wider range than the budget so that an oversized request
    # is a policy decision with a reason code, not a silent schema rejection.
    row_limit: int = Field(ge=1, le=200)


class ListRecentChangesArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: Literal[ToolName.LIST_RECENT_CHANGES] = ToolName.LIST_RECENT_CHANGES
    service: str
    window_start: UtcDatetime
    window_end: UtcDatetime


class GetTopologyArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: Literal[ToolName.GET_TOPOLOGY] = ToolName.GET_TOPOLOGY
    incident_id: str


# This union is the tool registry. A second lookup table would be a competing
# source of truth about which tools and arguments exist.
ToolArguments = Annotated[
    QueryMetricArguments
    | QueryLogsArguments
    | ListRecentChangesArguments
    | GetTopologyArguments,
    Field(discriminator="tool"),
]


def fingerprint(arguments: ToolArguments) -> str:
    """Stable hash of a tool and its arguments, used to catch a repeated proposal.

    Pydantic serializes fields in declaration order, so the same request always
    produces the same hash regardless of the order the model wrote them in.
    """
    return sha256(arguments.model_dump_json().encode("utf-8")).hexdigest()
