"""The three families added alongside configuration change, against the real lab.

Marked `docker` for the same reason as `test_configuration_change.py`. The replay
model always plays back the same fixed script (a CONFIG_CHANGE diagnosis), so these
cases check that a real run completes and produces a well-formed report rather than
that the diagnosis matches the family's actual root cause.
"""

import json
from pathlib import Path

import pytest

from causalops import cli
from causalops.domain import Evidence, InvestigationReport, ToolReceipt
from causalops.scenario_control import reset_scenario, runs_root

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILIES = [
    "downstream_timeout_retry_amplification",
    "resource_pool_saturation",
    "ambiguous_telemetry",
]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def latest_investigation(report_id: str) -> Path:
    return REPOSITORY / "results" / "investigations" / report_id


@pytest.mark.parametrize("family", FAMILIES)
def test_a_scenario_runs_from_start_to_a_well_formed_report(
    family: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPOSITORY)

    assert cli.main(["scenario", "start", family, "--seed", "development"]) == 0
    incident_id = capsys.readouterr().out.strip().splitlines()[-1]

    try:
        assert cli.main(["investigate", incident_id, "--model", "replay"]) == 0
        printed = capsys.readouterr().out

        investigation_id = printed.strip().splitlines()[-2]
        written = latest_investigation(investigation_id)
        report = InvestigationReport.model_validate_json(
            (written / "report.json").read_text(encoding="utf-8")
        )
        for row in read_jsonl(written / "evidence.jsonl"):
            Evidence.model_validate(row)
        for row in read_jsonl(written / "receipts.jsonl"):
            ToolReceipt.model_validate(row)

        assert report.tools_executed > 0
        assert (written / "report.md").read_text(encoding="utf-8").startswith("# ")
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()
