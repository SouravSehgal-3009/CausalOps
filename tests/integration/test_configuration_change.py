"""One real incident, end to end, against the running lab.

Marked `docker` because it needs the containers. CI deselects it; the owner runs it
deliberately. Scoring happens here, on the evaluator side: the investigator process
never reads an expected outcome or a predicate.
"""

import json
from pathlib import Path

import pytest

from causalops import cli
from causalops.domain import Disposition, Evidence, InvestigationReport, ToolReceipt
from causalops.evaluation import ExpectedOutcome, score_run
from causalops.scenario_control import reset_scenario, runs_root

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILY = "configuration_change"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def latest_investigation(report_id: str) -> Path:
    return REPOSITORY / "results" / "investigations" / report_id


def test_one_incident_runs_from_scenario_start_to_a_scored_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPOSITORY)

    assert cli.main(["scenario", "start", FAMILY, "--seed", "development"]) == 0
    incident_id = capsys.readouterr().out.strip().splitlines()[-1]

    try:
        assert cli.main(["investigate", incident_id, "--model", "replay"]) == 0
        printed = capsys.readouterr().out
        assert "DIAGNOSED CONFIG_CHANGE" in printed

        investigation_id = printed.strip().splitlines()[-2]
        written = latest_investigation(investigation_id)
        report = InvestigationReport.model_validate_json(
            (written / "report.json").read_text(encoding="utf-8")
        )
        evidence = [
            Evidence.model_validate(row)
            for row in read_jsonl(written / "evidence.jsonl")
        ]
        receipts = [
            ToolReceipt.model_validate(row)
            for row in read_jsonl(written / "receipts.jsonl")
        ]

        assert report.disposition is Disposition.DIAGNOSED
        assert report.tools_executed == 2
        assert (written / "report.md").read_text(encoding="utf-8").startswith("# ")

        expected_file = (
            runs_root(REPOSITORY) / incident_id / "evaluator" / "expected.json"
        )
        expected = ExpectedOutcome.model_validate(
            {
                key: value
                for key, value in json.loads(
                    expected_file.read_text(encoding="utf-8")
                ).items()
                if key in {"root_cause", "disposition", "predicates"}
            }
        )
        scores = score_run(report, evidence, receipts, expected)

        assert scores.diagnosis_correct
        assert scores.disposition_correct
        assert scores.citations_valid
        assert scores.citations_sufficient
        assert scores.control.denied == 0
    finally:
        reset_scenario(REPOSITORY, incident_id)
        assert not (runs_root(REPOSITORY) / incident_id).exists()
