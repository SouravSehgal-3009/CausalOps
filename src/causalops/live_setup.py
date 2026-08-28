"""Shared live/replay model and tool-registry construction.

Extracted out of `cli.py` (`_build_model_and_registry`,
`_live_evaluation_ceiling_usd`, `REPLAY_FIXTURE`/`REPLAY_FIXTURE_DIR`) so
`causalops.evaluate_cli` could reuse the exact same wiring `causalops.cli`
uses for a live `investigate`, instead of copy-pasting it.
`tests/security/test_evaluate_cli_isolation.py` requires `causalops.cli`
never import `causalops.evaluate_cli`; the reverse also has to hold or the
two would form a cycle. This module is the neutral ground both import from
-- it imports neither `causalops.cli` nor `causalops.evaluate_cli`, and is
not itself a command entry point.
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
from causalops.live_model import LiveClaudeModel, minimum_possible_reservation_usd
from causalops.models import (
    ReplayReasoningModel,
    ReplayToolCallingModel,
    ToolCallingModel,
)
from causalops.pricing import CLAUDE_SONNET_5_PRICING
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
# `dispatch_registry` wraps all four tools, so the graph
# orchestrator runs `lab_diagnosis.json` -- two executed checks across two
# tools -- exactly as the retired loop orchestrator did; parity between the
# two was established and proven before the loop was retired.
REPLAY_FIXTURE = REPLAY_FIXTURE_DIR / "lab_diagnosis.json"

# `.env.example`-documented, application-wide, covering standalone
# and paired-evaluation runs together (`TECHNICAL_SPEC.md` §10). Only an
# ABSENT or BLANK `LIVE_EVALUATION_MAX_USD` silently falls back to the
# default below -- an owner who never set the variable, or set it to
# whitespace, gets the documented default rather than a startup failure.
# That default was raised from 2.00 to 5.00, re-derived from a real
# smoke call's measured per-call reservation size after the ratio replan;
# the calibration record is in `TECHNICAL_OVERVIEW.md`. Extracted
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
# signal to silently override. Full design rationale is in
# `TECHNICAL_OVERVIEW.md`'s
# "Live-evaluation cost ceiling validation" section.
#
# A well-formed, finite, positive value can still be unusable: `cost_ledger.
# record_reservation_before_request` always subtracts `cost_ledger.
# RESERVATION_CEILING_BUFFER_USD` ($0.10) from `ceiling_usd` before checking
# remaining budget, so any value at or below that buffer leaves `remaining`
# permanently negative and refuses every single reservation. But the buffer
# alone is not the true floor either: the cheapest real reservation this
# project's pricing could ever produce is not $0 -- no real call ever sends
# zero input tokens, because every call's own system prompt and tool schema
# are themselves billed input. `MINIMUM_POSSIBLE_RESERVATION_USD` below is
# that cheapest-real-reservation figure; `MINIMUM_USABLE_CEILING_USD` is the
# buffer plus that floor. A configured value at or below it can never
# authorize even one reservation and is refused the same way the too-small
# case always was.
DEFAULT_LIVE_EVALUATION_MAX_USD = 5.00
LIVE_EVALUATION_MAX_USD_VARIABLE = "LIVE_EVALUATION_MAX_USD"

# The genuine floor, not an approximation of one -- and, as of this fix, not
# a hand-reconstruction of `live_model._send`'s reservation formula either.
# Three earlier versions of this constant each hand-copied that formula in
# THIS module and missed one real component of it: the first counted only
# the tool schema (missing that every call also bills its own prose); the
# second counted the tool schema and an empty prose string (missing that
# `SYSTEM_TEXT` alone is 843 tokens on every real call, never zero). Each
# fix patched the one gap found that round without any guarantee a further
# gap did not remain -- which is exactly what happened, three times.
# `live_model.minimum_possible_reservation_usd` closes this bug class
# instead of extending it: it does not reconstruct `_send`'s formula at
# all, it calls `_send`'s own, real, unmodified reservation code (through
# `LiveClaudeModel.respond`) against a throwaway in-memory ledger and a
# fake transport that never touches the network, and reads back whatever
# `_send` actually reserved. Whatever `_send` bills in the future, this
# figure moves with it automatically -- there is no formula copy left here
# to fall out of date. Re-verify by reading `test_live_setup.py`'s own
# pinned assertion on this figure if `_send`'s reservation formula, the
# system prompt, or the final-assessment tool schema ever changes.
MINIMUM_POSSIBLE_RESERVATION_USD = minimum_possible_reservation_usd(
    CLAUDE_SONNET_5_PRICING
)
MINIMUM_USABLE_CEILING_USD = (
    RESERVATION_CEILING_BUFFER_USD + MINIMUM_POSSIBLE_RESERVATION_USD
)


def live_evaluation_ceiling_usd(environment: Mapping[str, str]) -> float:
    """Resolves `LIVE_EVALUATION_MAX_USD`. Only unset or blank falls back to
    `DEFAULT_LIVE_EVALUATION_MAX_USD` -- see this module's own comment above
    for why every other malformed or unusable shape now raises instead of
    silently defaulting to a MORE permissive value than what was configured.

    Three distinct failure shapes, in the order checked:

    1. Unparseable text (`float()` raises), or a parseable but non-finite
       value (`inf`, `-inf`, `nan`) -- not a well-formed number at all.
       Raised as `CEILING_MALFORMED`.
    2. A well-formed, finite, positive number at or below
       `MINIMUM_USABLE_CEILING_USD` -- no reservation, however small, could
       ever be authorized under it (`cost_ledger.
       record_reservation_before_request` always subtracts `cost_ledger.
       RESERVATION_CEILING_BUFFER_USD` before checking remaining budget, and
       the cheapest possible real reservation is `MINIMUM_POSSIBLE_
       RESERVATION_USD`). This also covers zero and negative values, which
       are trivially below a positive floor. Raised as
       `CEILING_BELOW_RESERVATION_BUFFER`.
    3. Anything else is a usable ceiling and is returned as-is -- including
       a well-formed positive number that is simply the wrong one (`500`
       typed for `5.00` parses cleanly and is honoured as written). Guarding
       against a fat-fingered magnitude that is still comfortably above the
       floor is the owner's job, not a parser's.

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
    if value <= MINIMUM_USABLE_CEILING_USD:
        # `.6f`, not `.4f`, on the two derived-floor figures below: the real
        # minimum reservation this module computes at import time
        # (`minimum_possible_reservation_usd`) is small enough that `.4f`
        # would round it away entirely and print a floor the owner could
        # not reproduce by reading this message.
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER,
            f"{LIVE_EVALUATION_MAX_USD_VARIABLE}={value:.4f} is at or below "
            f"${MINIMUM_USABLE_CEILING_USD:.6f} -- the "
            f"${RESERVATION_CEILING_BUFFER_USD:.2f} reservation safety "
            f"buffer every live request is checked against, plus "
            f"${MINIMUM_POSSIBLE_RESERVATION_USD:.6f}, the cheapest real "
            "reservation this project's pricing could ever produce -- so no "
            "reservation, however small, could ever be authorized at this "
            f"ceiling. Set {LIVE_EVALUATION_MAX_USD_VARIABLE} comfortably "
            f"above ${MINIMUM_USABLE_CEILING_USD:.6f} (see .env.example).",
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
    way for `causalops.cli`'s fresh `investigate` and its
    `approve`/`reject` resume, and for `causalops.evaluate_cli`'s
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
    # Presence only, mirroring `doctor.check_api_key`'s own
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
