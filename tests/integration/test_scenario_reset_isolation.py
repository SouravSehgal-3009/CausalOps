"""A reset scenario must leave nothing behind for the next one to inherit.

Marked `docker` for the same reason as the other integration tests: it needs the
running lab containers. Family A starts, is reset, and a different family B then
starts and investigates cleanly, proving the baseline-healthy gate that opens
`scenario start` is not contaminated by A's fault state.
"""

from pathlib import Path

import pytest

from causalops import cli
from causalops.scenario_control import reset_scenario, runs_root

pytestmark = pytest.mark.docker

REPOSITORY = Path(__file__).resolve().parents[2]
FAMILY_A = "resource_pool_saturation"
FAMILY_B = "configuration_change"


def test_resetting_one_family_leaves_a_clean_baseline_for_the_next(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPOSITORY)

    assert cli.main(["scenario", "start", FAMILY_A, "--seed", "development"]) == 0
    incident_a = capsys.readouterr().out.strip().splitlines()[-1]

    assert cli.main(["scenario", "reset", incident_a]) == 0
    capsys.readouterr()
    assert not (runs_root(REPOSITORY) / incident_a).exists()

    assert cli.main(["scenario", "start", FAMILY_B, "--seed", "development"]) == 0
    incident_b = capsys.readouterr().out.strip().splitlines()[-1]

    try:
        assert cli.main(["investigate", incident_b, "--model", "replay"]) == 0
    finally:
        reset_scenario(REPOSITORY, incident_b)
        assert not (runs_root(REPOSITORY) / incident_b).exists()
