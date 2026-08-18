"""Run events and atomic finalization of one investigation's artifacts.

Deciding what happened and writing bytes durably fail in different ways, so the
workflow builds the report and this module makes it permanent.
"""

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from causalops.domain import (
    SCHEMA_VERSION,
    Evidence,
    InvestigationReport,
    ReasonCode,
    ToolReceipt,
)
from causalops.tools import UtcDatetime


class RunRecordError(Exception):
    """Finalization refused, with a stable reason code."""

    def __init__(self, reason_code: ReasonCode, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class RunEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    sequence: int
    at: UtcDatetime
    state: str
    name: str
    fields: dict[str, JsonValue] = {}


class RunRecorder:
    """Collects the ordered events of one run until they are finalized together."""

    def __init__(self, clock: Callable[[], datetime]) -> None:
        self.clock = clock
        self.recorded: list[RunEvent] = []

    def event(self, state: str, name: str, **fields: JsonValue) -> None:
        self.recorded.append(
            RunEvent(
                sequence=len(self.recorded) + 1,
                at=self.clock(),
                state=state,
                name=name,
                fields=fields,
            )
        )

    @property
    def events(self) -> tuple[RunEvent, ...]:
        return tuple(self.recorded)


def write_jsonl(path: Path, records: Sequence[BaseModel]) -> None:
    lines = [record.model_dump_json() for record in records]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def finalize_investigation(
    results_root: Path,
    report: InvestigationReport,
    events: Sequence[RunEvent],
    evidence: Sequence[Evidence],
    receipts: Sequence[ToolReceipt],
    report_markdown: str,
) -> Path:
    """Write the artifacts beside each other, then move them into place in one step."""
    target = results_root / "investigations" / report.investigation_id
    if target.exists():
        raise RunRecordError(
            ReasonCode.RESULT_ALREADY_FINALIZED,
            f"{target} already holds a finalized investigation",
        )
    staging = target.parent / f".staging-{report.investigation_id}"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (staging / "report.md").write_text(report_markdown, encoding="utf-8")
    write_jsonl(staging / "events.jsonl", events)
    write_jsonl(staging / "evidence.jsonl", evidence)
    write_jsonl(staging / "receipts.jsonl", receipts)
    staging.replace(target)
    return target
