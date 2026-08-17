"""Read-only backends for the registered tools, and the runner that dispatches them.

These read the active run's JSONL logs and manifests. The metric backend lives in
`prometheus.py` because it is the one that talks to the network. Results are bounded
at the source, because a tool result is where untrusted lab output enters the
investigation.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from causalops.domain import (
    CheckOutcome,
    EvidenceKind,
    IncidentScope,
    ReasonCode,
    RunCheck,
    ToolOutcome,
    ToolProposal,
)
from causalops.evidence import executed_check, failed_check, trim_to_bytes
from causalops.prometheus import run_metric_check
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    LogFilter,
    QueryLogsArguments,
    QueryMetricArguments,
)

MAX_LOG_ROWS = 40


class RunPaths(BaseModel):
    """The investigator-visible files of one run.

    There is deliberately no accessor for the evaluator directory that sits beside
    these: code that cannot name a path cannot read it by accident.
    """

    model_config = ConfigDict(frozen=True)

    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def changes_file(self) -> Path:
        return self.root / "changes.json"

    @property
    def topology_file(self) -> Path:
        return self.root / "topology.json"

    @property
    def incident_file(self) -> Path:
        return self.root / "incident.json"


def read_json_file(path: Path) -> JsonValue | None:
    try:
        loaded: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded


def read_json_line(line: str) -> dict[str, JsonValue] | None:
    try:
        record: JsonValue = json.loads(line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def matches_filter(log_filter: LogFilter, record: dict[str, JsonValue]) -> bool:
    if log_filter is LogFilter.ERRORS_ONLY:
        return record.get("severity") == "error"
    if log_filter is LogFilter.TIMEOUTS_ONLY:
        return record.get("event") == "upstream_timeout"
    if log_filter is LogFilter.POOL_EXHAUSTION:
        return record.get("event") == "pool_exhausted"
    if log_filter is LogFilter.CONFIG_RELOAD:
        return record.get("event") in ("config_loaded", "config_rejected_request")
    return False


def within_window(moment: str, start: datetime, end: datetime) -> bool:
    try:
        observed = datetime.fromisoformat(moment)
    except ValueError:
        return False
    return start <= observed <= end


def run_logs_check(arguments: QueryLogsArguments, paths: RunPaths) -> CheckOutcome:
    source = "query_logs"
    started = time.monotonic()
    log_file = paths.logs / f"{arguments.service}.jsonl"
    if not log_file.is_file():
        return failed_check(
            EvidenceKind.LOG,
            source,
            ToolOutcome.UNAVAILABLE,
            ReasonCode.TOOL_UNAVAILABLE,
            f"no log for {arguments.service} in this run",
        )
    limit = min(arguments.row_limit, MAX_LOG_ROWS)
    rows: list[JsonValue] = []
    events: set[str] = set()
    truncated = False
    with log_file.open(encoding="utf-8") as handle:
        for line in handle:
            record = read_json_line(line)
            if record is None or not matches_filter(arguments.log_filter, record):
                continue
            moment = record.get("at")
            if not isinstance(moment, str) or not within_window(
                moment, arguments.window_start, arguments.window_end
            ):
                continue
            if len(rows) >= limit:
                truncated = True
                break
            rows.append(record)
            event = record.get("event")
            if isinstance(event, str):
                events.add(event)
    payload: dict[str, JsonValue] = {
        "filter": arguments.log_filter.value,
        "service": arguments.service,
        "row_count": len(rows),
        "event_codes": ",".join(sorted(events)),
        "truncated": truncated,
    }
    payload = trim_to_bytes(payload, "rows", rows)
    return executed_check(
        EvidenceKind.LOG,
        source,
        f"{arguments.log_filter.value} on {arguments.service}: "
        f"{len(rows)} rows, events {payload['event_codes'] or 'none'}",
        payload,
        started,
    )


def run_changes_check(
    arguments: ListRecentChangesArguments, paths: RunPaths
) -> CheckOutcome:
    source = "list_recent_changes"
    started = time.monotonic()
    loaded = read_json_file(paths.changes_file)
    if not isinstance(loaded, list):
        return failed_check(
            EvidenceKind.CHANGE,
            source,
            ToolOutcome.UNAVAILABLE,
            ReasonCode.TOOL_UNAVAILABLE,
            "this run records no changes",
        )
    changes: list[JsonValue] = []
    summaries: list[str] = []
    for entry in loaded:
        if not isinstance(entry, dict) or entry.get("service") != arguments.service:
            continue
        moment = entry.get("at")
        if not isinstance(moment, str) or not within_window(
            moment, arguments.window_start, arguments.window_end
        ):
            continue
        changes.append(entry)
        summary = entry.get("summary")
        if isinstance(summary, str):
            summaries.append(summary)
    payload: dict[str, JsonValue] = {
        "service": arguments.service,
        "change_count": len(changes),
        "summaries": "; ".join(summaries),
        "truncated": False,
    }
    payload = trim_to_bytes(payload, "changes", changes)
    return executed_check(
        EvidenceKind.CHANGE,
        source,
        f"{len(changes)} recent changes on {arguments.service}",
        payload,
        started,
    )


def run_topology_check(
    arguments: GetTopologyArguments, paths: RunPaths
) -> CheckOutcome:
    source = "get_topology"
    started = time.monotonic()
    loaded = read_json_file(paths.topology_file)
    if not isinstance(loaded, dict):
        return failed_check(
            EvidenceKind.TOPOLOGY,
            source,
            ToolOutcome.UNAVAILABLE,
            ReasonCode.TOOL_UNAVAILABLE,
            "this run records no topology",
        )
    edges = loaded.get("edges")
    edge_list: list[JsonValue] = edges if isinstance(edges, list) else []
    payload: dict[str, JsonValue] = {
        "services": loaded.get("services", []),
        "edge_count": len(edge_list),
        "truncated": False,
    }
    payload = trim_to_bytes(payload, "edges", edge_list)
    return executed_check(
        EvidenceKind.TOPOLOGY,
        source,
        f"{len(edge_list)} service edges in this incident",
        payload,
        started,
    )


def registered_check_runner(
    paths: RunPaths, prometheus_url: str, timeout_seconds: int
) -> RunCheck:
    """The runner Step 2 left a seam for: it turns an approved proposal into a result.

    Everything a backend needs beyond the proposal is configuration, so it is
    captured here and the returned callable matches the seam exactly.
    """

    def run(proposal: ToolProposal, scope: IncidentScope) -> CheckOutcome:
        arguments = proposal.arguments
        if isinstance(arguments, QueryMetricArguments):
            return run_metric_check(arguments, scope, prometheus_url, timeout_seconds)
        if isinstance(arguments, QueryLogsArguments):
            return run_logs_check(arguments, paths)
        if isinstance(arguments, ListRecentChangesArguments):
            return run_changes_check(arguments, paths)
        return run_topology_check(arguments, paths)

    return run
