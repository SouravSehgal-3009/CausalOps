"""The tool-policy-bypass control `TECHNICAL_SPEC.md` §9 requires.

No tool backend may be reachable except through a policy wrapper. An import
scan alone cannot prove that: `workflow.py` imports no backend module today,
yet reaches every one of them by injection at `cli.py:154`
(`registered_check_runner(...)` is passed in as a bare `RunCheck` callable,
never imported by name where it is called). §9 requires three independent
controls, and this file is all three:

1. An AST import test -- necessary, not sufficient, for the reason above.
2. A wrapper-identity test -- every dispatch-registry entry was produced by a
   wrapper factory, checked structurally (`isinstance`) and by construction
   (a hand-built `ToolWrapper` is refused), not by import graph.
3. A spy-backend test -- a denied proposal invokes the backend zero times,
   for every registered tool.
"""

from pathlib import Path

import pytest
from fake_incident import (
    INCIDENT_ID,
    WINDOW_END,
    WINDOW_START,
    RecordingChangesBackend,
    RecordingLogsBackend,
    RecordingMetricBackend,
    RecordingTopologyBackend,
    StepClock,
    incident_scope,
)
from import_scan import PACKAGE, imported_modules

from causalops.domain import Budgets, PolicyResult, ReasonCode, ToolProposal
from causalops.tool_wrappers import ReservationLedger, ToolWrapper, dispatch_registry
from causalops.tools import (
    GetTopologyArguments,
    ListRecentChangesArguments,
    LogFilter,
    MetricTemplate,
    QueryLogsArguments,
    QueryMetricArguments,
    ToolName,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPOSITORY / "src" / PACKAGE
BACKEND_MODULES = {f"{PACKAGE}.telemetry", f"{PACKAGE}.prometheus"}
DISPATCH_SOURCE_FILES = ("tool_wrappers.py", "tool_calls.py", "graph.py")


def out_of_scope_metric_proposal() -> ToolProposal:
    return ToolProposal(
        arguments=QueryMetricArguments(
            template=MetricTemplate.GATEWAY_ERROR_RATE,
            service="billing",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        ),
        evidence_gap="whether billing's error rate moved",
        expected_observation="nothing, billing is out of scope",
    )


def out_of_scope_logs_proposal() -> ToolProposal:
    return ToolProposal(
        arguments=QueryLogsArguments(
            log_filter=LogFilter.ERRORS_ONLY,
            service="billing",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            row_limit=20,
        ),
        evidence_gap="whether billing logged anything",
        expected_observation="nothing, billing is out of scope",
    )


def out_of_scope_changes_proposal() -> ToolProposal:
    return ToolProposal(
        arguments=ListRecentChangesArguments(
            service="billing",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
        ),
        evidence_gap="whether billing changed recently",
        expected_observation="nothing, billing is out of scope",
    )


def out_of_scope_topology_proposal() -> ToolProposal:
    """`get_topology`'s only model field is `tool` plus `incident_id` -- there
    is no service or window to put out of scope. Its one refusable shape is a
    cross-incident id (`policy.authorize`'s `CROSS_INCIDENT_REQUEST` branch),
    so the id below is deliberately not `INCIDENT_ID`."""
    return ToolProposal(
        arguments=GetTopologyArguments(incident_id="a" * len(INCIDENT_ID)),
        evidence_gap="another incident's topology",
        expected_observation="nothing, that incident is out of scope",
    )


# One arm per registered tool -- the property this table backs (§9's spy
# control) outlives the tool count, so a fifth registered tool (Milestone 2's
# `search_runbooks`) must add its own arm here or the assertion below fails
# loudly rather than silently narrowing coverage.
OUT_OF_SCOPE_PROPOSAL_BY_TOOL: dict[ToolName, ToolProposal] = {
    ToolName.QUERY_METRIC: out_of_scope_metric_proposal(),
    ToolName.QUERY_LOGS: out_of_scope_logs_proposal(),
    ToolName.LIST_RECENT_CHANGES: out_of_scope_changes_proposal(),
    ToolName.GET_TOPOLOGY: out_of_scope_topology_proposal(),
}

# Pinned so a denial for the wrong reason cannot pass silently -- without
# this, a future field change that trips an earlier `policy.authorize`
# branch (budget, duplicate fingerprint) before reaching a tool's own check
# would still leave `policy_result is PolicyResult.DENIED` true, the same
# silent-narrowing risk `set(registry) == set(OUT_OF_SCOPE_PROPOSAL_BY_TOOL)`
# exists to catch for missing arms.
EXPECTED_REASON_BY_TOOL: dict[ToolName, ReasonCode] = {
    ToolName.QUERY_METRIC: ReasonCode.UNKNOWN_SERVICE,
    ToolName.QUERY_LOGS: ReasonCode.UNKNOWN_SERVICE,
    ToolName.LIST_RECENT_CHANGES: ReasonCode.UNKNOWN_SERVICE,
    ToolName.GET_TOPOLOGY: ReasonCode.CROSS_INCIDENT_REQUEST,
}


def test_the_dispatch_boundary_modules_import_no_backend() -> None:
    """Necessary, not sufficient -- see the module docstring above."""
    for name in DISPATCH_SOURCE_FILES:
        imported = imported_modules(SOURCE_DIR / name)
        assert imported & BACKEND_MODULES == set(), name


def test_every_dispatch_registry_entry_is_wrapper_produced() -> None:
    """A bare backend callable assigned directly into the registry would pass
    the import test above (it imports nothing) but fail this one: it would
    not be a `ToolWrapper` instance, so this check is structural, not import-
    graph-based, exactly the distinction §9 draws."""
    registry = dispatch_registry(
        run_metric=RecordingMetricBackend(),
        run_logs=RecordingLogsBackend(),
        run_changes=RecordingChangesBackend(),
        run_topology=RecordingTopologyBackend(),
    )

    assert registry, "the registry must not be empty for this check to mean anything"
    for tool, entry in registry.items():
        assert isinstance(entry, ToolWrapper)
        assert entry.tool is tool


def test_a_hand_built_tool_wrapper_is_rejected() -> None:
    """`isinstance` alone cannot prove a registry entry is wrapper-produced --
    a hand-built instance passes it too. This is that exact reproduction,
    now refused at construction time instead of silently joining a registry.
    """
    with pytest.raises(TypeError):
        ToolWrapper(tool=ToolName.QUERY_LOGS, dispatch=lambda *args: None)  # type: ignore[arg-type]


def test_every_registered_tool_denies_an_out_of_scope_proposal_untouched() -> None:
    """Four separate spies, not one: with a single shared spy, denying three
    tools and executing the fourth would still read `backend.calls == []` if
    the one spy happened to sit behind the untouched three, silently proving
    only a quarter of the registry. Each backend below is asserted
    independently, so a regression in any one wrapper is caught by its own
    tool's assertion, not hidden behind another tool's."""
    metric_backend = RecordingMetricBackend()
    logs_backend = RecordingLogsBackend()
    changes_backend = RecordingChangesBackend()
    topology_backend = RecordingTopologyBackend()
    registry = dispatch_registry(
        run_metric=metric_backend,
        run_logs=logs_backend,
        run_changes=changes_backend,
        run_topology=topology_backend,
    )

    assert set(registry) == set(OUT_OF_SCOPE_PROPOSAL_BY_TOOL), (
        "add an out-of-scope proposal above for every newly registered tool"
    )
    for tool, wrapper in registry.items():
        result = wrapper.dispatch(
            OUT_OF_SCOPE_PROPOSAL_BY_TOOL[tool],
            incident_scope(),
            set(),
            Budgets(),
            ReservationLedger(executed_tools_budget=2),
            StepClock(),
        )
        assert result.receipt.policy_result is PolicyResult.DENIED
        assert result.receipt.reason_code is EXPECTED_REASON_BY_TOOL[tool]

    assert metric_backend.calls == []
    assert logs_backend.calls == []
    assert changes_backend.calls == []
    assert topology_backend.calls == []
