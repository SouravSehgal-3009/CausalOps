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
# the calibration record is in `TECHNICAL_OVERVIEW.md`. `TECHNICAL_SPEC.md`
# §10 carries the same amendment. Unit 3c: extracted here from `cli.py`,
# still the one ceiling both `causalops` and `causalops-evaluate` read.
#
# Round 6 review (Codex, three findings) corrected an earlier posture that
# had this fallback cover a much wider set of shapes: unparseable text,
# non-finite values (`inf`/`-inf`/`nan`), and non-positive values (`<= 0`)
# used to default silently too, alongside unset/blank -- on the reasoning
# that each was "the smallest plausible ceiling," so defaulting on them
# could only make the gate stricter, never more permissive. That reasoning
# does not actually hold: `DEFAULT_LIVE_EVALUATION_MAX_USD` ($5.00) is far
# LARGER than $0, so silently defaulting a `0`, a negative number, or a
# typo like `"0.05USD"` actually authorizes far MORE spend than the owner's
# malformed input suggested they wanted -- exactly the more-permissive,
# more-dangerous surprise this module's own reasoning about the too-small
# case (immediately below) already correctly refuses to allow. This module
# now applies that same "fail loudly rather than silently authorize more"
# posture consistently: every one of those shapes raises `CheckpointStoreError`
# now, the same as the too-small case always did. Only unset/blank still
# defaults, because there both the "value the owner typed" and "value the
# owner is implicitly asking for" are the same thing -- nothing was typed,
# so there is no wrong signal to silently override.
#
# A well-formed, finite, positive value can still be unusable: `cost_ledger.
# record_reservation_before_request` always subtracts `cost_ledger.
# RESERVATION_CEILING_BUFFER_USD` ($0.10) from `ceiling_usd` before checking
# remaining budget, so any value at or below that buffer leaves `remaining`
# permanently negative and refuses every single reservation. But the buffer
# alone is not the true floor either: the cheapest real reservation this
# project's pricing could ever produce is not $0 -- it is the fixed output
# allowance alone, with zero input tokens, `pricing.CLAUDE_SONNET_5_PRICING.
# reservation_usd(input_tokens=0)`, currently $0.016 (`pricing.
# MAX_OUTPUT_TOKENS` tokens at the output rate; no real request could ever
# reserve less, since every request reserves at least its full output
# allowance regardless of how little input it sends). A ceiling of $0.11 --
# above the $0.10 buffer, so an earlier version of this check accepted it --
# still leaves only $0.01 of headroom once the buffer is subtracted, less
# than the $0.016 floor, so even the cheapest possible request would still
# be refused. `MINIMUM_USABLE_CEILING_USD` below is the buffer plus that
# floor; a configured value at or below it can never authorize even one
# reservation and is refused the same way the too-small case always was.
DEFAULT_LIVE_EVALUATION_MAX_USD = 5.00
LIVE_EVALUATION_MAX_USD_VARIABLE = "LIVE_EVALUATION_MAX_USD"

# The cheapest real reservation `pricing.py`'s math could ever produce:
# every request reserves its full output allowance up front regardless of
# how little input it sends, so `input_tokens=0` is the genuine theoretical
# floor, not an approximation of one.
MINIMUM_POSSIBLE_RESERVATION_USD = CLAUDE_SONNET_5_PRICING.reservation_usd(
    input_tokens=0
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
        raise CheckpointStoreError(
            CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER,
            f"{LIVE_EVALUATION_MAX_USD_VARIABLE}={value:.4f} is at or below "
            f"${MINIMUM_USABLE_CEILING_USD:.4f} -- the "
            f"${RESERVATION_CEILING_BUFFER_USD:.2f} reservation safety "
            f"buffer every live request is checked against, plus "
            f"${MINIMUM_POSSIBLE_RESERVATION_USD:.4f}, the cheapest real "
            "reservation this project's pricing could ever produce -- so no "
            "reservation, however small, could ever be authorized at this "
            f"ceiling. Set {LIVE_EVALUATION_MAX_USD_VARIABLE} comfortably "
            f"above ${MINIMUM_USABLE_CEILING_USD:.4f} (see .env.example).",
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
