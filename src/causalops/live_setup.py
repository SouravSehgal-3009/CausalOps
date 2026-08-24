"""Shared live/replay model and tool-registry construction.

Unit 3c extracted this out of `cli.py` (`_build_model_and_registry`,
`_live_evaluation_ceiling_usd`, `REPLAY_FIXTURE`/`REPLAY_FIXTURE_DIR`) so
`causalops.evaluate_cli` could reuse the exact same wiring `causalops.cli`
uses for a live `investigate`, instead of copy-pasting it. `CLAUDE.md`
requires `causalops evaluate` never be imported by `causalops.cli`; the
reverse also has to hold or the two would form a cycle. This module is the
neutral ground both import from -- it imports neither `causalops.cli` nor
`causalops.evaluate_cli`, and is not itself a command entry point.
"""

import math
import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from causalops.approvals import CheckpointStoreError, CheckpointStoreReasonCode
from causalops.cost_ledger import (
    RESERVATION_CEILING_BUFFER_USD,
    ensure_cost_ledger_table,
)
from causalops.doctor import API_KEY_VARIABLE
from causalops.domain import REPLAY_MODEL_NAME, Budgets, StoredIncident
from causalops.live_model import MODEL_NAME as LIVE_MODEL_NAME
from causalops.live_model import LiveClaudeModel
from causalops.models import (
    ReplayReasoningModel,
    ReplayToolCallingModel,
    ToolCallingModel,
)
from causalops.prometheus import DEFAULT_PROMETHEUS_URL, run_metric_check
from causalops.runbooks import RunbookIndex, run_runbook_search
from causalops.telemetry import (
    RunPaths,
    run_changes_check,
    run_logs_check,
    run_topology_check,
)
from causalops.tool_wrappers import ToolWrapper, dispatch_registry
from causalops.tools import ToolName

REPLAY_FIXTURE_DIR = Path(__file__).parent / "replay_fixtures"
# `dispatch_registry` wraps all four tools as of Unit 1c, so the graph
# orchestrator runs `lab_diagnosis.json` -- two executed checks across two
# tools -- exactly as the retired loop orchestrator did; parity between the
# two was established in Unit 1d-1 before the loop was retired.
REPLAY_FIXTURE = REPLAY_FIXTURE_DIR / "lab_diagnosis.json"

# Unit 3b-2. `.env.example`-documented, application-wide, covering standalone
# and paired-evaluation runs together (`TECHNICAL_SPEC.md` §10). Only an
# ABSENT or BLANK `LIVE_EVALUATION_MAX_USD` silently falls back to the
# default below -- an owner who never set the variable, or set it to
# whitespace, gets the documented default rather than a startup failure.
# Unit 3b-3: that default was raised from 2.00 to 5.00, re-derived from the
# smoke call's measured per-call reservation size after the ratio replan;
# the calibration record is in `TECHNICAL_OVERVIEW.md`. Unit 3c: extracted
# here from `cli.py`, still the one ceiling both `causalops` and
# `causalops-evaluate` read.
#
# Every malformed or unusable shape -- unparseable text, a non-finite value
# (`inf`/`-inf`/`nan`), a non-positive value, or a well-formed value too
# small to ever authorize a reservation -- raises `CheckpointStoreError`
# rather than silently falling back to the default. Silently defaulting on
# any of those would actually be MORE dangerous, not safer: `DEFAULT_
# LIVE_EVALUATION_MAX_USD` ($5.00) is far larger than a malformed `0`, a
# negative number, or a typo like `"0.05USD"` -- so silently defaulting
# would authorize far more spend than the owner's malformed input
# suggested they wanted. Only unset/blank still defaults, because there
# both "the value the owner typed" and "the value the owner is implicitly
# asking for" are the same thing -- nothing was typed, so there is no wrong
# signal to silently override. Full design rationale, including why this
# was worth a dedicated review pass, is in `TECHNICAL_OVERVIEW.md`'s
# "Live-evaluation cost ceiling validation" section.
#
# Values at or below the fixed reservation buffer cannot authorize any
# request. A larger structural value is not a universal usability guarantee:
# the exact reservation remains incident- and stage-dependent and is checked
# immediately before provider invocation.
DEFAULT_LIVE_EVALUATION_MAX_USD = 5.00
LIVE_EVALUATION_MAX_USD_VARIABLE = "LIVE_EVALUATION_MAX_USD"


def live_evaluation_ceiling_usd(environment: Mapping[str, str]) -> float:
    """Resolves `LIVE_EVALUATION_MAX_USD`. Only unset or blank falls back to
    `DEFAULT_LIVE_EVALUATION_MAX_USD` -- see this module's own comment above
    for why every other malformed or unusable shape now raises instead of
    silently defaulting to a MORE permissive value than what was configured.

    Three distinct failure shapes, in the order checked:

    1. Unparseable text (`float()` raises), or a parseable but non-finite
       value (`inf`, `-inf`, `nan`) -- not a well-formed number at all.
       Raised as `CEILING_MALFORMED`.
    2. A well-formed, finite number at or below the fixed reservation buffer.
       This also covers zero and negative values. Raised as
       `CEILING_BELOW_RESERVATION_BUFFER`.
    3. Anything else is structurally valid and is returned as-is. Its actual
       request-time usability remains dependent on the rendered request.

    Both raised cases use `CheckpointStoreError` -- the same reusable
    `FAIL <CODE> <message>` shape `cost_ledger.py` already reuses this type
    for (`RESERVATION_NOT_SETTLEABLE`) -- so both `causalops.cli`'s `main`
    and `causalops.evaluate_cli`'s `main` report this cleanly: both catch
    `CheckpointStoreError` explicitly in their typed refusal handlers.
    """
    raw = environment.get(LIVE_EVALUATION_MAX_USD_VARIABLE, "").strip()
    if not raw:
        return DEFAULT_LIVE_EVALUATION_MAX_USD
    try:
        value = float(raw)
    except ValueError as error:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.CEILING_MALFORMED,
            f"{LIVE_EVALUATION_MAX_USD_VARIABLE}={raw!r} is not a number "
            f"({error}). Set it to a plain positive figure such as 5.00, "
            "or leave it unset to use the default "
            f"(${DEFAULT_LIVE_EVALUATION_MAX_USD:.2f}, see .env.example).",
        ) from error
    if not math.isfinite(value):
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.CEILING_MALFORMED,
            f"{LIVE_EVALUATION_MAX_USD_VARIABLE}={raw!r} is not a finite "
            "number -- infinite and NaN ceilings are not a real spending "
            "limit. Set it to a plain positive figure such as 5.00 (see "
            ".env.example).",
        )
    if value <= RESERVATION_CEILING_BUFFER_USD:
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER,
            f"{LIVE_EVALUATION_MAX_USD_VARIABLE}={value:.4f} is at or below "
            f"${RESERVATION_CEILING_BUFFER_USD:.2f} reservation safety "
            "buffer every live request is checked against. A value above it "
            "can still be refused by the exact, incident- and stage-dependent "
            "reservation check.",
        )
    return value


def build_model_and_registry(
    incident: StoredIncident,
    paths: RunPaths,
    budgets: Budgets,
    model_choice: Literal["replay", "claude"],
    db_path: Path,
) -> tuple[
    ToolCallingModel, Mapping[ToolName, ToolWrapper], str, sqlite3.Connection | None
]:
    """The model, tool registry, display model name, and (for a live model
    only) the cost-ledger connection one incident needs -- built the same
    way for `causalops.cli`'s fresh `investigate` and Unit 2c's
    `approve`/`reject` resume, and (Unit 3c) for `causalops.evaluate_cli`'s
    paired live comparison, all three of which need the exact same wiring
    `build_graph` requires.

    The cost-ledger connection is this function's own concern, not the
    checkpointer's: it is a second, independent connection to the same
    `checkpoints.db` file, matching `cli.py`'s own `decisions_connection`
    pattern rather than reusing the checkpointer's connection --
    `SqliteSaver`'s own transaction handling around a node's execution is
    not part of any contract this module can rely on, and a second
    connection is exactly how this project already isolates one concern's
    SQLite transactions from another's on the same file. Returned to the
    caller (`None` for replay) so its lifetime is the caller's to close.
    """
    # A fresh in-memory index per call -- the corpus is small and read-only,
    # so rebuilding it costs nothing measurable, and it keeps this function's
    # "everything an incident needs, built fresh" contract intact rather
    # than reaching for a module-level singleton `search_runbooks` alone
    # would need.
    runbook_index = RunbookIndex()
    registry = dispatch_registry(
        run_metric=lambda arguments, scope: run_metric_check(
            arguments, scope, DEFAULT_PROMETHEUS_URL, budgets.tool_timeout_seconds
        ),
        run_logs=lambda arguments, scope: run_logs_check(arguments, paths),
        run_changes=lambda arguments, scope: run_changes_check(arguments, paths),
        run_topology=lambda arguments, scope: run_topology_check(arguments, paths),
        run_search=lambda arguments, scope: run_runbook_search(
            arguments, runbook_index
        ),
    )
    if model_choice == "replay":
        replay_model = ReplayToolCallingModel(
            ReplayReasoningModel(
                REPLAY_FIXTURE,
                substitutions={
                    "incident_id": incident.scope.incident_id,
                    "window_start": incident.scope.started_at.isoformat(),
                    "window_end": incident.scope.ended_at.isoformat(),
                    "symptom_evidence_id": incident.packet.symptom_evidence_id,
                },
            )
        )
        return replay_model, registry, REPLAY_MODEL_NAME, None
    ledger_conn = sqlite3.connect(str(db_path), check_same_thread=False)
    ensure_cost_ledger_table(ledger_conn)
    # Unit 3b-2, P3-3. Presence only, mirroring `doctor.check_api_key`'s own
    # check -- the value itself never leaves this line. `LiveClaudeModel`
    # never reads `ANTHROPIC_API_KEY` itself (`live_model.MissingCredential`'s
    # docstring; `tests/security/test_credential_isolation.py` proves the
    # module neither imports `os` nor names the variable in code), so this
    # `bool` is the only thing that crosses that boundary.
    credential_present = bool(os.environ.get(API_KEY_VARIABLE, "").strip())
    live_model = LiveClaudeModel(
        ledger_conn,
        ceiling_usd=live_evaluation_ceiling_usd(os.environ),
        credential_present=credential_present,
    )
    return live_model, registry, LIVE_MODEL_NAME, ledger_conn
