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
from causalops.evidence import executed_check, failed_check, fits, trim_to_bytes
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


def _reject_non_finite_json_token(token: str) -> JsonValue:
    """`json.loads` accepts the non-standard tokens `NaN`/`Infinity`/
    `-Infinity` by default -- an extension beyond RFC-8259 most other JSON
    readers reject. Passed as `parse_constant`, this makes `read_json_file`/
    `read_json_line` refuse them the same way they already refuse anything
    else unreadable, instead of silently threading a Python `nan`/`inf`
    float into a typed record. `prometheus.py`'s `read_sample` was hardened
    against this same token class in metric API responses; these two
    functions are the general JSON-parsing entry point for every file this
    module reads -- logs (`read_json_line`, `run_logs_check`), changes and
    topology (`read_json_file`, `run_changes_check`/`run_topology_check`)
    -- so without this, any of those could carry a NaN/Infinity token and
    have it silently propagate. (`incident.json`/`report.json` go through a
    separate loader, `cli.py`'s `_load_stored_artifact`, not through
    either function here.)"""
    raise ValueError(f"non-standard JSON token {token!r} is not accepted")


def read_json_file(path: Path) -> JsonValue | None:
    try:
        loaded: JsonValue = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_json_token,
        )
    except (OSError, ValueError):  # json.JSONDecodeError subclasses ValueError
        return None
    return loaded


def read_json_line(line: str) -> dict[str, JsonValue] | None:
    try:
        record: JsonValue = json.loads(
            line, parse_constant=_reject_non_finite_json_token
        )
    except ValueError:  # json.JSONDecodeError subclasses ValueError
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
    """A record with a malformed or naive timestamp is excluded, the same
    way any other malformed field in a record is -- never raised, which
    would turn one bad record into a crash for the whole check
    (`run_logs_check`/`run_changes_check`, then `graph.py`'s blanket
    handler, then `FAILED_SAFE` for the entire run instead of skipping one
    row).

    Unit 3b-4 addendum, C2: `datetime.fromisoformat` returns either an
    aware or a naive `datetime` depending on whether `moment` carries a UTC
    offset; comparing a naive one against the aware `start`/`end` this
    function is always called with raises `TypeError`, not `ValueError` --
    a second, distinct failure mode the original `except ValueError` alone
    never caught, inconsistent with this function's own evident intent
    (skip a bad record, don't crash the check). The contract is explicit:
    a naive timestamp is REJECTED, never silently coerced to UTC -- this
    project has no way to know what offset a naive value was actually
    meant to carry, and guessing UTC could misplace a record outside its
    real window in either direction. Consistent with every other timestamp
    in this codebase being required aware (`tools.UtcDatetime`)."""
    try:
        observed = datetime.fromisoformat(moment)
    except ValueError:
        return False
    if observed.tzinfo is None:
        return False
    return start <= observed <= end


# Kept together so the read, filter, window, and truncate steps for one log query
# stay in one readable pass rather than being split into single-use helpers.
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
    payload = trim_to_bytes(payload, "rows", rows, "row_count")
    # Post-freeze review, P3-1. `len(rows)` here is the PRE-trim count --
    # `trim_to_bytes` mutates a COPY (`kept = list(rows)`), never `rows`
    # itself, so this summary used to claim more rows than `payload["rows"]`
    # actually holds whenever byte-trimming (not just the `limit` cap
    # above) removed any. `payload["row_count"]` is what `trim_to_bytes`
    # keeps honest throughout (Unit 3b-4 addendum, C3); a trailing
    # "(truncated)" names the gap explicitly rather than leaving a reader
    # to notice the count looks low.
    #
    # Round 4 review, F3. `event_codes` (above, built from `events`) has
    # the SAME pre-trim-aggregate shape as `row_count` did -- it is
    # assembled during the loop that fills `rows`, entirely before
    # `trim_to_bytes` runs, so a row popped by BYTE trimming (as opposed
    # to the `limit` cutoff the loop already respects) still left its
    # event code listed even though it no longer appears in
    # `payload["rows"]`. Rebuilt from `payload["rows"]`, the POST-trim
    # list, so the codes shown always match the rows actually returned.
    # Only ever removes codes (every kept row was already in the
    # pre-trim set), so this cannot make the payload grow back over
    # budget.
    kept_rows = payload["rows"]
    assert isinstance(kept_rows, list)
    kept_events: set[str] = set()
    for kept_row in kept_rows:
        if isinstance(kept_row, dict):
            kept_event = kept_row.get("event")
            if isinstance(kept_event, str):
                kept_events.add(kept_event)
    payload["event_codes"] = ",".join(sorted(kept_events))
    truncated_note = " (truncated)" if payload["truncated"] else ""
    return executed_check(
        EvidenceKind.LOG,
        source,
        f"{arguments.log_filter.value} on {arguments.service}: "
        f"{payload['row_count']} rows, events "
        f"{payload['event_codes'] or 'none'}{truncated_note}",
        payload,
        started,
    )


# Kept together so the read, filter, window, and truncate steps for one changes
# query stay in one readable pass rather than being split into single-use helpers.
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
    payload = trim_to_bytes(payload, "changes", changes, "change_count")
    # Round 6 review, item 2. `summaries` (above) has the same pre-trim-
    # aggregate shape already fixed twice this round elsewhere
    # (`event_codes`, `max_value`): it was joined from EVERY matched
    # change before `trim_to_bytes` ran and never rebuilt, so it could
    # still name changes no longer present in `payload["changes"]` after
    # trimming -- worst case, `change_count: 0` with `summaries` still
    # listing changes for an empty list. Rebuilt below from
    # `payload["changes"]`, the POST-trim list.
    #
    # Unlike `event_codes`/`max_value`, this rebuild is not simply
    # guaranteed smaller by construction without checking: `trim_to_bytes`
    # pops rows from the END of the list, so `kept_summaries` is always a
    # PREFIX (in original order) of the full `summaries` this scalar was
    # first built from -- if `changes` still holds any rows, `summaries`
    # was never touched during the row-popping loop above, so the payload
    # already fit WITH the full string present, and a prefix of it can
    # only be smaller or equal. If `changes` was fully emptied, the
    # scalar-shrinking fallback already halved `summaries` down before
    # this rebuild replaces it with an even shorter (or empty) string.
    # Both branches can only shrink the payload -- but that reasoning is
    # exactly the shape of claim a previous fix in THIS SAME FUNCTION made
    # about seeding order and shipped wrong (F1, round 3), so it is
    # checked with `fits()` below rather than trusted silently.
    kept_changes = payload["changes"]
    assert isinstance(kept_changes, list)
    kept_summaries: list[str] = []
    for kept_change in kept_changes:
        if isinstance(kept_change, dict):
            kept_summary = kept_change.get("summary")
            if isinstance(kept_summary, str):
                kept_summaries.append(kept_summary)
    payload["summaries"] = "; ".join(kept_summaries)
    # Round 7 review confirmed this check is genuinely unreachable here
    # (this function always has a string-valued field -- `summaries` --
    # for `trim_to_bytes`'s own scalar fallback to shrink, so `fits()`
    # cannot come back `False` at this point). Kept anyway as harmless
    # defense-in-depth: it costs nothing at runtime and would catch a
    # future change to this rebuild that broke the reasoning above.
    #
    # Round 8 review, P3. This assert's sibling in `run_topology_check`
    # was, at one point, described as "the genuinely load-bearing one" by
    # contrast with this one -- wrong: a reviewer measured directly (120
    # randomized trials, 5 adversarial shapes, and a mutation test
    # disabling the topology assert entirely) that IT is unreachable too,
    # today, for the same reason any `run_topology_check` payload converges
    # once both lists are trimmable to empty. Both asserts are currently
    # unreachable defense-in-depth, not one provably-needed and one
    # decorative -- see the topology assert's own comment for why it is
    # still worth keeping despite that.
    assert fits(payload), (
        "rebuilding summaries from the post-trim changes list must not "
        "grow the payload back over the byte bound"
    )
    truncated_note = " (truncated)" if payload["truncated"] else ""
    return executed_check(
        EvidenceKind.CHANGE,
        source,
        f"{payload['change_count']} recent changes on "
        f"{arguments.service}{truncated_note}",
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
    services_field = loaded.get("services")
    service_list: list[JsonValue] = (
        services_field if isinstance(services_field, list) else []
    )
    # Post-freeze review. `services` is a LIST, not the string-valued
    # scalar `trim_to_bytes`'s own fallback (Unit 3b-4 addendum, C3) can
    # shrink, and it is not `edges`, the ROW list this function's other
    # `trim_to_bytes` call below already bounds -- an oversized `services`
    # list would pass through both mechanisms untouched. Correctness
    # measured this concretely (34,863 bytes against the 12,288-byte cap,
    # not reachable through any of the four shipped lab topologies, all
    # under 100 bytes total -- P3, not P2) and confirmed `services` is the
    # ONLY non-string, non-row field anywhere in this codebase's `trim_to_
    # bytes` callers that needs its own bounding pass: not evidence for a
    # general recursive mechanism over arbitrary payload shapes, which
    # would be solving a class of problem this codebase has exactly one
    # instance of. A second `trim_to_bytes` call, treating `services` as
    # its own row list with its own `service_count`, reuses the identical
    # byte-bounding logic `edges` already gets below rather than inventing
    # a second one.
    #
    # Both `"services"` and `"edges"` are seeded into `payload` BEFORE
    # either `trim_to_bytes` call, full and untrimmed -- caught by this
    # function's own mutation testing: seeding only `"services"` up front
    # and letting the `"edges"` call add its OWN key for the first time,
    # the way this function used to before `services` needed bounding too,
    # let a payload that had already converged to fit (services trimmed
    # down against a payload with no `"edges"` key yet) go back over
    # budget the moment `"edges": []` was added afterward -- with nothing
    # left to pop (`edge_list` was already empty) and no STRING field for
    # the scalar fallback to shrink (`services`, a list, is invisible to
    # it), that state could never re-converge. Seeding both up front means
    # each call's own `fits()` checks always see the TRUE combined size
    # from its first iteration, so WHETHER the combined payload ends up
    # fitting the byte cap does not depend on call order.
    #
    # WHICH list absorbs the trimming is a different question, and order
    # DOES decide it -- `trim_to_bytes` pops rows from its own list
    # unconditionally until `fits(payload)`, so whichever call runs first
    # keeps popping against the OTHER list's still-untrimmed full weight.
    # A round of review found this the hard way: with `services`-first
    # (the order this function used to run in) and a realistic incident
    # shape -- a handful of real service names next to a genuinely
    # oversized `edges` list -- the `services` call never sees `fits()`
    # turn true until it has popped `services` to EMPTY, because `edges`
    # is still full size on every iteration; `edges` itself, the field
    # actually responsible for the overage, comes away barely trimmed.
    # `edges`-first is the order below because it is the field this
    # codebase's real data grows without bound (topology connections);
    # `services` is a short, bounded list of service names that should
    # almost never need trimming at all. Trimming `edges` first protects
    # `services` at `edges`'s expense, which is the right tradeoff for
    # that shape -- it does not eliminate the underlying asymmetry, it
    # only points it at the field where losing rows is safe to read.
    payload: dict[str, JsonValue] = {
        "services": service_list,
        "service_count": len(service_list),
        "edges": edge_list,
        "edge_count": len(edge_list),
        "truncated": False,
    }
    payload = trim_to_bytes(payload, "edges", edge_list, "edge_count")
    payload = trim_to_bytes(payload, "services", service_list, "service_count")
    # Round 7 review. This payload has NO string-valued field at all
    # (`services`/`edges` are lists, `service_count`/`edge_count`/
    # `truncated` are int/bool) -- unlike every other `trim_to_bytes`
    # caller in this codebase, `trim_to_bytes`'s own scalar-shrinking
    # fallback (see its docstring) is a true no-op here, not a second
    # layer of defense. A reviewer measured directly that the fallback's
    # `widest_key is None` escape IS reached in normal operation (a
    # realistic small-services/large-edges shape hits it inside the
    # `edges` call, while `services` is still its full, untrimmed size) --
    # not just hypothetically.
    #
    # Round 8 review, P3. The paragraph above used to go on to call this
    # assert "the actual safety net" for that shape -- overclaiming what
    # was actually verified. A reviewer measured directly (120 randomized
    # trials, 5 adversarial shapes, and a mutation test that disabled this
    # assert entirely) that it is currently UNREACHABLE, same as its
    # sibling in `run_changes_check`: both lists can always be popped to
    # empty, and the remaining fixed structure (85 bytes) is far under the
    # 12,288-byte cap, so `fits(payload)` is always true by the time this
    # line runs. What the reachability finding above actually supports is
    # narrower: the fallback's escape hatch fires in real cases, so THIS
    # function -- alone among `trim_to_bytes`'s callers -- has no string
    # field left standing if the byte math or the shape of lab data ever
    # changed enough to make that stop being true. That is a real reason to
    # keep this assert as defense-in-depth; it is not evidence the assert
    # is load-bearing today.
    assert fits(payload), (
        "trimming both edges and services down to empty must still leave "
        "a payload under the byte bound -- this function has no string "
        "field for trim_to_bytes's own fallback to shrink instead"
    )
    # Round 6 review, the P1. Same gap as `run_metric_check`: `payload
    # ["truncated"]` already carries whether either list above was cut,
    # but this summary string never rendered it, and the summary is the
    # only part of a `CheckOutcome` `prompts.py` puts in front of the
    # model. A truncated topology read as complete with no signal that
    # edges or services were dropped.
    truncated_note = " (truncated)" if payload["truncated"] else ""
    return executed_check(
        EvidenceKind.TOPOLOGY,
        source,
        f"{payload['edge_count']} service edges in this incident{truncated_note}",
        payload,
        started,
    )


def _registered_check_runner(
    paths: RunPaths, prometheus_url: str, timeout_seconds: int
) -> RunCheck:
    """The runner Step 2 left a seam for: it turns an approved proposal into a result.

    Everything a backend needs beyond the proposal is configuration, so it is
    captured here and the returned callable matches the seam exactly.

    Unit 3b-4 addendum, C6: private, not a real dispatch path. Superseded by
    `tool_wrappers.dispatch_registry` before `search_runbooks` existed;
    nothing in `cli.py` calls this today, and its `RunCheck` return type
    (`CheckOutcome` only) cannot even express a `search_runbooks` result
    (`RunbookCheckOutcome`) if something ever did. Kept, not deleted,
    because `tests/unit/test_telemetry.py` still exercises it directly as
    a documented historical seam -- a leading underscore says "do not wire
    this up as a second dispatch path" without discarding that coverage.
    """

    def run(proposal: ToolProposal, scope: IncidentScope) -> CheckOutcome:
        arguments = proposal.arguments
        if isinstance(arguments, QueryMetricArguments):
            return run_metric_check(arguments, scope, prometheus_url, timeout_seconds)
        if isinstance(arguments, QueryLogsArguments):
            return run_logs_check(arguments, paths)
        if isinstance(arguments, ListRecentChangesArguments):
            return run_changes_check(arguments, paths)
        if isinstance(arguments, GetTopologyArguments):
            return run_topology_check(arguments, paths)
        # `RunCheck`'s own return type is `CheckOutcome` only -- it cannot
        # express `RunbookCheckOutcome`, so this orphaned seam (superseded
        # by `tool_wrappers.dispatch_registry`; nothing in `cli.py` calls
        # this function) has no correct way to route a `search_runbooks`
        # proposal at all. Raising loudly documents that gap instead of
        # silently mis-dispatching it to `run_topology_check`, which the
        # unconditional fallthrough this replaced would have done the
        # moment a fifth argument type existed.
        raise ValueError(
            f"_registered_check_runner cannot route {type(arguments).__name__} -- "
            "this seam predates search_runbooks and returns CheckOutcome only"
        )

    return run
