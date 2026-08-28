"""`causalops-evaluate`'s own pure helpers and orchestration.

`run_evaluation`'s real path always builds a live `LiveClaudeModel`
(`causalops.live_setup.build_model_and_registry` with `model_choice="claude"`
hardcoded -- evaluate has no replay mode) and drives real scenario-controller
traffic against a running Docker lab. Neither is available to this fast,
network-free suite (`tests/conftest.py`'s network guard covers the whole
session), so the orchestration test below monkeypatches
`causalops.evaluate_cli.build_model_and_registry`,
`causalops.evaluate_cli.start_scenario`, and
`causalops.evaluate_cli.reset_scenario` -- the same seam-testing approach
`test_live_model.py` already uses for `LiveClaudeModel` itself (a fake
`client=`), applied here to this script's own three external dependencies.
The docker-marked `tests/integration/` suite is where a real lab is
required; this file never needs one.
"""

import hashlib
import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_incident import (
    RecordingLogsBackend,
    RecordingMetricBackend,
    alert_packet,
    incident_scope,
    packet_evidence,
    registry_with,
)
from pydantic import BaseModel

import causalops.evaluate_cli as evaluate_cli
from causalops.approvals import CheckpointStoreError, CheckpointStoreReasonCode
from causalops.cost_ledger import (
    RESERVATION_CEILING_BUFFER_USD,
    ensure_cost_ledger_table,
    record_reservation_before_request,
)
from causalops.domain import (
    Budgets,
    Disposition,
    EvidenceKind,
    IncidentScope,
    ReasonCode,
    RetrievalMode,
    RootCauseCode,
    StoredIncident,
    Versions,
)
from causalops.evaluate_cli import (
    EVALUATION_FAMILIES,
    EVALUATION_SEEDS,
    EVIDENCE_BUDGET_CURVE,
    MODE_NO_TOOL_BASELINE,
    MODE_TOOL_ENABLED,
    _check_preflight_cost,
    _fixture_sha256,
    _git_provenance,
    _load_expected_outcome,
    _new_evaluation_target,
    _preflight_worst_case_batch_usd,
    _run_id_from_events,
    _write_json_atomic,
    build_parser,
    main,
    render_evaluation_summary,
    render_paired_evaluation_summary,
    resolve_budgets,
    run_evaluation,
    summarize_paired_evaluation,
)
from causalops.evaluation import (
    ControlCounts,
    Efficiency,
    EvaluationRecord,
    EvaluationSummary,
    ExpectedOutcome,
    MechanicalScores,
    PredicateOperator,
    RequiredEvidencePredicate,
    summarize_evaluation,
)
from causalops.evidence import new_opaque_id
from causalops.live_setup import MAXIMUM_POSSIBLE_RESERVATION_USD
from causalops.models import ReplayReasoningModel, ReplayToolCallingModel
from causalops.run_records import RunEvent
from causalops.scenario_control import run_paths

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "causalops" / "replay_fixtures"
)


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _init_git_repo(root: Path) -> None:
    _run_git(root, "init", "--quiet")
    _run_git(root, "config", "user.email", "test@example.com")
    _run_git(root, "config", "user.name", "Test")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    _run_git(root, "add", "tracked.txt")
    _run_git(root, "commit", "--quiet", "-m", "initial")


def _fake_start_scenario(root: Path, family: str, seed: str) -> str:
    """A `causalops.evaluate_cli.start_scenario` stand-in shared by every
    `main()` test that does not need to observe or vary its own calls into
    it -- a fresh incident, its `incident.json`, and its evaluator/
    expected.json, written the same way a real `start_scenario` call would
    leave them for `_run_one` to read back. `test_run_id_from_events_finds_
    the_investigation_started_run_id` keeps its OWN local closure instead
    of this one: it needs to append to a test-local `started` list on
    every call, a real behavioural difference this shared stand-in
    deliberately does not carry."""
    incident_id = new_opaque_id()
    scope = IncidentScope.model_validate(
        {**incident_scope().model_dump(), "incident_id": incident_id}
    )
    packet = alert_packet().model_copy(update={"incident_id": incident_id})
    evidence = tuple(
        record.model_copy(update={"incident_id": incident_id})
        for record in packet_evidence()
    )
    incident = StoredIncident(scope=scope, packet=packet, evidence=evidence)
    paths = run_paths(root, incident_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
    (paths.root / "evaluator").mkdir(exist_ok=True)
    (paths.root / "evaluator" / "expected.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "family": family,
                "root_cause": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION",
                "disposition": "DIAGNOSED",
                "predicates": [],
            }
        ),
        encoding="utf-8",
    )
    return incident_id


def _disable_preflight_cost_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_evaluation`'s pre-flight cost check (`_check_preflight_cost`)
    compares a real batch's worst-case cost against
    `LIVE_EVALUATION_MAX_USD` before any scenario starts -- real machinery
    this file's own dedicated pre-flight tests exercise directly. Every
    OTHER test in this file that drives `run_evaluation`/`main()` through
    monkeypatched, zero-cost fakes is testing something else entirely and
    would otherwise trip this real, unrelated gate (the default
    `DEFAULT_LIVE_EVALUATION_MAX_USD` ceiling is sized for real spend, not
    for the fact that no test here sets `LIVE_EVALUATION_MAX_USD` at all) --
    disabled here the same way `start_scenario`/`build_model_and_registry`
    are already faked out for the same reason."""
    monkeypatch.setattr(evaluate_cli, "_check_preflight_cost", lambda *a, **k: None)


DEFAULT_TEST_BUDGETS = Budgets()


def test_build_parser_accepts_no_arguments() -> None:
    parsed = build_parser().parse_args([])

    assert parsed == build_parser().parse_args([])
    assert parsed.executed_tools == 2


def test_build_parser_accepts_each_curve_point() -> None:
    for point in EVIDENCE_BUDGET_CURVE:
        parsed = build_parser().parse_args(["--executed-tools", str(point)])
        assert parsed.executed_tools == point


def test_build_parser_rejects_a_value_off_the_curve() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--executed-tools", "5"])


def test_resolve_budgets_couples_model_calls_to_executed_tools_plus_two() -> None:
    for executed_tools in EVIDENCE_BUDGET_CURVE:
        budgets = resolve_budgets(executed_tools)
        assert budgets.executed_tools == executed_tools
        assert budgets.model_calls == executed_tools + 2


def test_resolve_budgets_at_the_default_matches_budgets_own_class_defaults() -> None:
    """`resolve_budgets(2)` -- the default curve point -- must produce
    exactly what `Budgets()`'s own class-level defaults already are, so the
    first point on the curve is behaviourally identical to every existing
    caller that still constructs a bare `Budgets()`."""
    assert resolve_budgets(2) == Budgets()


def test_git_provenance_reads_a_clean_commit(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    sha, dirty = _git_provenance(tmp_path)

    assert len(sha) == 40
    assert dirty is False


def test_git_provenance_detects_a_dirty_tree(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("two\n", encoding="utf-8")

    _, dirty = _git_provenance(tmp_path)

    assert dirty is True


def test_fixture_sha256_matches_the_scenario_files_own_bytes(tmp_path: Path) -> None:
    scenarios = tmp_path / "lab" / "scenarios"
    scenarios.mkdir(parents=True)
    content = b'{"family": "ambiguous_telemetry"}'
    (scenarios / "ambiguous_telemetry.json").write_bytes(content)

    assert (
        _fixture_sha256(tmp_path, "ambiguous_telemetry")
        == hashlib.sha256(content).hexdigest()
    )


def test_load_expected_outcome_reads_the_evaluator_directory(tmp_path: Path) -> None:
    evaluator_dir = tmp_path / "evaluator"
    evaluator_dir.mkdir()
    (evaluator_dir / "expected.json").write_text(
        json.dumps(
            {
                "seed": "evaluation",
                "family": "configuration_change",
                "root_cause": "CONFIG_CHANGE",
                "disposition": "DIAGNOSED",
                "predicates": [
                    {
                        "source": "list_recent_changes",
                        "kind": "CHANGE",
                        "field": "summaries",
                        "operator": "CONTAINS",
                        "value": "require_order_token",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    expected = _load_expected_outcome(tmp_path)

    assert expected.root_cause is RootCauseCode.CONFIG_CHANGE
    assert expected.disposition is Disposition.DIAGNOSED
    assert len(expected.predicates) == 1
    assert expected.predicates[0].field == "summaries"


def test_run_id_from_events_finds_the_investigation_started_run_id() -> None:
    events = [
        RunEvent(
            sequence=0,
            at=datetime(2026, 8, 24, tzinfo=UTC),
            state="CREATED",
            name="investigation_started",
            fields={"incident": "inc-1", "run_id": "run-abc"},
        )
    ]

    assert _run_id_from_events(events) == "run-abc"


def test_run_id_from_events_raises_when_absent() -> None:
    with pytest.raises(RuntimeError):
        _run_id_from_events([])


def test_run_evaluation_drives_every_family_as_a_baseline_then_tool_enabled_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the orchestration `run_evaluation` is responsible for: one
    `start_scenario(..., seed)` per `(family, seed)` pair in
    `EVALUATION_FAMILIES` x `EVALUATION_SEEDS`, two scored runs per incident
    in baseline-then-tool-enabled order, one `reset_scenario` per incident
    even though nothing raises, and a resulting `EvaluationRecord` per run
    carrying the evaluator-only expected outcome -- without ever handing
    that expected outcome to the investigation call itself (this script
    builds it and the model+registry from two entirely separate functions;
    `_run_one` never receives `expected` as investigator input, only as a
    scoring argument after the run has already finished)."""
    _disable_preflight_cost_check(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )

    started: list[tuple[str, str]] = []
    reset: list[str] = []

    def fake_start_scenario(root: Path, family: str, seed: str) -> str:
        started.append((family, seed))
        incident_id = new_opaque_id()
        scope = IncidentScope.model_validate(
            {**incident_scope().model_dump(), "incident_id": incident_id}
        )
        packet = alert_packet().model_copy(update={"incident_id": incident_id})
        evidence = tuple(
            record.model_copy(update={"incident_id": incident_id})
            for record in packet_evidence()
        )
        # `alert_packet()`'s own evidence-id fields still point at
        # `packet_evidence()`'s fixed ids -- those ids are unaffected by
        # this incident_id substitution, so the packet stays internally
        # consistent without also rewriting them.
        incident = StoredIncident(scope=scope, packet=packet, evidence=evidence)
        paths = run_paths(root, incident_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.incident_file.write_text(incident.model_dump_json(), encoding="utf-8")
        (paths.root / "evaluator").mkdir(exist_ok=True)
        (paths.root / "evaluator" / "expected.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "family": family,
                    "root_cause": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION",
                    "disposition": "DIAGNOSED",
                    "predicates": [],
                }
            ),
            encoding="utf-8",
        )
        return incident_id

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        reset.append(incident_id)

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        assert model_choice == "claude"
        model = ReplayToolCallingModel(
            ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
        )
        registry = registry_with(
            run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
        )
        ledger_conn = sqlite3.connect(":memory:")
        return model, registry, "fake-claude-model", ledger_conn

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )
    graph_calls: list[dict[str, object]] = []
    real_run_graph = evaluate_cli.run_graph_investigation

    def spy_run_graph(*args: object, **kwargs: object) -> object:
        graph_calls.append(kwargs)
        return real_run_graph(*args, **kwargs)

    monkeypatch.setattr("causalops.evaluate_cli.run_graph_investigation", spy_run_graph)

    target = _new_evaluation_target(tmp_path)
    records = run_evaluation(tmp_path, target, DEFAULT_TEST_BUDGETS)

    expected_started = [
        (family, seed_name)
        for family in EVALUATION_FAMILIES
        for seed_name in EVALUATION_SEEDS
    ]
    assert started == expected_started
    incidents_started = len(EVALUATION_FAMILIES) * len(EVALUATION_SEEDS)
    assert len(reset) == incidents_started
    assert len(records) == 2 * incidents_started
    modes = [record.run_key.rsplit("/", 1)[-1] for record in records]
    assert modes == ["no_tool_baseline", "tool_enabled"] * incidents_started
    seed_names = [record.seed_name for record in records]
    assert seed_names == [seed for _, seed in expected_started for _ in range(2)]
    assert len(graph_calls) == len(records)
    for index, call in enumerate(graph_calls):
        assert call["suppress_escalation"] is True
        assert call["no_tool_baseline"] is (index % 2 == 0)
    for record in records:
        assert record.model_name == "fake-claude-model"
        assert record.git_sha == "f" * 40
        assert record.git_dirty is False
        assert record.executed_tools == 2
        assert record.model_calls_budget == 4
        assert (
            record.expected.root_cause
            is RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION
        )


def test_records_already_scored_before_a_crash_survive_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this test exists for: `run_evaluation` must rewrite
    `records.jsonl` after every completed run, not batch everything to the
    end. Proves it by making the run that would build the SECOND
    `(family, seed)` pair's model construction raise -- after the first
    pair's baseline and tool-enabled runs have both already completed and
    been appended -- and then reading `records.jsonl` back from disk (not
    the in-memory exception, and not the in-memory `records` list this
    function never even gets to return) to confirm those two
    already-scored, already-paid-for records are durable despite the
    crash. Reading only the in-memory list would not prove anything: the
    bug this test guards against was exactly that records could be correct
    in memory yet never reach disk before a crash discarded them."""
    _disable_preflight_cost_check(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        pass

    call_count = 0

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        nonlocal call_count
        call_count += 1
        # Calls 1 and 2 are the first `(family, seed)` pair's baseline and
        # tool-enabled runs -- let both succeed and land on disk. Call 3 is
        # the SECOND `(family, seed)` pair's baseline run (still the first
        # family, its second seed, since the loop is family-major, seed-
        # minor) -- raise there, simulating a real mid-batch failure (a
        # crashed request, a killed process) after real, billed work has
        # already produced scoreable results.
        if call_count == 3:
            raise RuntimeError("simulated mid-batch failure on the third run")
        model = ReplayToolCallingModel(
            ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
        )
        registry = registry_with(
            run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
        )
        ledger_conn = sqlite3.connect(":memory:")
        return model, registry, "fake-claude-model", ledger_conn

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", _fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    target = _new_evaluation_target(tmp_path)
    records_path = target / "records.jsonl"

    with pytest.raises(RuntimeError, match="simulated mid-batch failure"):
        run_evaluation(tmp_path, target, DEFAULT_TEST_BUDGETS)

    assert call_count == 3

    on_disk = [
        EvaluationRecord.model_validate_json(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(on_disk) == 2
    modes = [record.run_key.rsplit("/", 1)[-1] for record in on_disk]
    assert modes == ["no_tool_baseline", "tool_enabled"]
    # Both surviving records share one incident_id -- the first
    # `(family, seed)` pair's -- confirming they are the pair from before
    # the crash, not some other mix.
    incident_ids = {record.run_key.split("/", 1)[0] for record in on_disk}
    assert len(incident_ids) == 1
    for record in on_disk:
        assert record.model_name == "fake-claude-model"


def _prepare_stubbed_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in ("first", "second"):
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate_cli, "EVALUATION_FAMILIES", ("first", "second"))
    monkeypatch.setattr(evaluate_cli, "start_scenario", _fake_start_scenario)
    monkeypatch.setattr(evaluate_cli, "reset_scenario", lambda root, incident_id: None)
    monkeypatch.setattr(evaluate_cli, "_git_provenance", lambda root: ("f" * 40, False))
    _disable_preflight_cost_check(monkeypatch)
    return _new_evaluation_target(tmp_path)


def test_infrastructure_failure_is_persisted_before_evaluation_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _prepare_stubbed_evaluation(tmp_path, monkeypatch)
    calls: list[bool] = []

    def fake_run_one(**kwargs: object) -> EvaluationRecord:
        calls.append(kwargs["no_tool_baseline"] is True)
        return _paired_record(
            run_key=f"incident-{len(calls)}/model/no_tool_baseline",
            diagnosis_correct=False,
        ).model_copy(update={"failure_reason": ReasonCode.COST_CEILING_EXCEEDED})

    monkeypatch.setattr(evaluate_cli, "_run_one", fake_run_one)

    with pytest.raises(evaluate_cli.EvaluationAborted) as aborted:
        run_evaluation(tmp_path, target, DEFAULT_TEST_BUDGETS)

    assert aborted.value.reason is ReasonCode.COST_CEILING_EXCEEDED
    assert calls == [True]
    records = [
        EvaluationRecord.model_validate_json(line)
        for line in (target / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0].failure_reason is ReasonCode.COST_CEILING_EXCEEDED


def test_model_quality_failure_does_not_abort_later_evaluation_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _prepare_stubbed_evaluation(tmp_path, monkeypatch)
    calls = 0

    def fake_run_one(**kwargs: object) -> EvaluationRecord:
        nonlocal calls
        calls += 1
        failure = ReasonCode.MODEL_OUTPUT_INVALID if calls == 1 else None
        return _paired_record(
            run_key=f"incident-{calls}/model/no_tool_baseline",
            diagnosis_correct=False,
        ).model_copy(update={"failure_reason": failure})

    monkeypatch.setattr(evaluate_cli, "_run_one", fake_run_one)

    records = run_evaluation(tmp_path, target, DEFAULT_TEST_BUDGETS)

    # 2 stubbed families x the real EVALUATION_SEEDS x 2 arms -- this test
    # does not monkeypatch EVALUATION_SEEDS, only EVALUATION_FAMILIES.
    expected_calls = 2 * len(EVALUATION_SEEDS) * 2
    assert calls == expected_calls
    assert len(records) == expected_calls
    assert records[0].failure_reason is ReasonCode.MODEL_OUTPUT_INVALID
    on_disk = (target / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(on_disk) == expected_calls


def test_the_original_run_failure_survives_cleanup_also_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bug this test exists for: `run_evaluation`'s `finally:
    reset_scenario(...)` used to be unconditional -- if the scored run
    inside `try` failed (a real, billed run failure) AND `reset_scenario` in
    `finally` ALSO raised, Python's ordinary exception chaining would
    replace the original exception with the cleanup failure, burying the
    actual reason a billed run failed behind an unrelated lab-reset
    problem. Simulates both failing on the very first family and confirms
    the ORIGINAL run failure is what actually propagates and gets reported
    -- the cleanup failure is only printed, never silently discarded."""
    _disable_preflight_cost_check(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        raise RuntimeError("simulated reset_scenario cleanup failure")

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        raise RuntimeError("simulated billed-run failure")

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", _fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    target = _new_evaluation_target(tmp_path)

    with pytest.raises(RuntimeError, match="simulated billed-run failure"):
        run_evaluation(tmp_path, target, DEFAULT_TEST_BUDGETS)

    out = capsys.readouterr().out
    assert "RESET_SCENARIO_FAILED_DURING_CLEANUP" in out
    assert "simulated reset_scenario cleanup failure" in out


def test_a_cleanup_failure_after_a_successful_run_still_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier implementation of the test above queried `sys.exc_info()`
    from INSIDE the nested `except Exception as reset_error:` handler --
    which always describes `reset_error` itself, never an outer exception,
    since `sys.exc_info()` reports whatever the *nearest enclosing* `except`
    is currently handling at the point it is called, and at that point the
    nearest enclosing handler IS this one. So that code's `if sys.
    exc_info()[0] is None: raise` was always `False` -- the `raise` was dead
    code, and the cleanup failure was printed and silently swallowed
    UNCONDITIONALLY, even when the scored run underneath succeeded cleanly.
    This is the opposite of the intended behaviour and is the real bug this
    test guards against: a `reset_scenario` failure after a successful run
    must raise, not be swallowed, since swallowing it lets `main()` return 0
    while an active-scenario marker is left stranded. Run against that
    earlier code, this test fails: `run_evaluation` returns its records
    normally instead of raising, because every family's reset failure gets
    caught, printed, and discarded."""
    _disable_preflight_cost_check(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        raise RuntimeError("simulated reset_scenario failure after a clean run")

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        model = ReplayToolCallingModel(
            ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
        )
        registry = registry_with(
            run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
        )
        ledger_conn = sqlite3.connect(":memory:")
        return model, registry, "fake-claude-model", ledger_conn

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", _fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    target = _new_evaluation_target(tmp_path)

    with pytest.raises(
        RuntimeError, match="simulated reset_scenario failure after a clean run"
    ):
        run_evaluation(tmp_path, target, DEFAULT_TEST_BUDGETS)


def test_render_evaluation_summary_shows_counts_and_ranges_not_a_percentile() -> None:
    """The gap this test exists for: `main()` used to print one line per
    record and never a batch-level aggregate at all. This project
    forbids p95 or any other broad performance claim from a small
    synthetic sample -- this asserts the rendered text carries the counts
    and ranges an `EvaluationSummary` holds, including the honest "how many
    runs' cost is actually known" annotation, citation counts, and
    policy/control aggregates, and never mentions a percentile."""
    summary = EvaluationSummary(
        total_records=2,
        correct_and_grounded_count=1,
        diagnosis_correct_count=1,
        disposition_correct_count=2,
        citations_valid_count=2,
        citations_sufficient_count=1,
        citations_sufficient_applicable_count=2,
        scorer_versions=("3",),
        latency_ms_min=100,
        latency_ms_max=900,
        model_calls_min=1,
        model_calls_max=4,
        tools_executed_min=0,
        tools_executed_max=2,
        denied_min=0,
        denied_max=1,
        duplicate_min=0,
        duplicate_max=0,
        out_of_scope_min=0,
        out_of_scope_max=1,
        invalid_responses_min=0,
        invalid_responses_max=0,
        unsettled_min=0,
        unsettled_max=0,
        input_tokens_min=500,
        input_tokens_max=4000,
        input_tokens_known_count=2,
        output_tokens_min=100,
        output_tokens_max=800,
        output_tokens_known_count=2,
        reserved_usd_min=0.01,
        reserved_usd_max=0.05,
        actual_usd_min=0.008,
        actual_usd_max=0.008,
        actual_usd_known_count=1,
    )

    rendered = render_evaluation_summary(summary)

    assert "evaluation summary: 2 record(s)" in rendered
    assert "correct_and_grounded:1/2 (2/2 applicable)" in rendered
    assert "diagnosis_correct:   1/2" in rendered
    assert "disposition_correct: 2/2" in rendered
    assert "citations_valid:     2/2" in rendered
    assert "citations_sufficient:1/2 (2/2 applicable)" in rendered
    assert "scorer_versions: 3" in rendered
    assert "latency_ms:      100-900" in rendered
    assert "model_calls:     1-4" in rendered
    assert "tools_executed:  0-2" in rendered
    assert "control.denied:            0-1" in rendered
    assert "control.duplicate:         0-0" in rendered
    assert "control.out_of_scope:      0-1" in rendered
    assert "control.invalid_responses: 0-0" in rendered
    assert "control.unsettled:         0-0" in rendered
    assert "input_tokens:    500-4000 (2/2 known)" in rendered
    assert "output_tokens:   100-800 (2/2 known)" in rendered
    assert "reserved_usd:    0.0100-0.0500" in rendered
    assert "actual_usd:      0.0080-0.0080 (1/2 known)" in rendered
    assert "p95" not in rendered.lower()
    assert "percentile" not in rendered.lower()


def test_render_evaluation_summary_shows_the_applicable_denominator_apart() -> None:
    """When a batch mixes records with and without a required-evidence
    predicate (e.g. `ambiguous_telemetry` alongside families that do
    declare one), the rendered line must show the applicable denominator
    (how many records had any predicate to score) distinctly from the
    batch total -- not silently equate "not applicable" with the total
    record count."""
    summary = EvaluationSummary(
        total_records=3,
        correct_and_grounded_count=1,
        diagnosis_correct_count=3,
        disposition_correct_count=3,
        citations_valid_count=3,
        citations_sufficient_count=1,
        citations_sufficient_applicable_count=2,
        scorer_versions=("3",),
        input_tokens_known_count=0,
        output_tokens_known_count=0,
        actual_usd_known_count=0,
    )

    rendered = render_evaluation_summary(summary)

    assert "citations_sufficient:1/2 (2/3 applicable)" in rendered


def test_render_evaluation_summary_shows_correct_and_grounded_first() -> None:
    """`correct_and_grounded` renders above `diagnosis_correct`, not merely
    present anywhere in the output -- a precise ordering assertion, since
    the two metrics answer overlapping questions and a reader scanning
    top-to-bottom should see the joint metric first."""
    summary = EvaluationSummary(
        total_records=2,
        correct_and_grounded_count=1,
        diagnosis_correct_count=1,
        disposition_correct_count=2,
        citations_valid_count=2,
        citations_sufficient_count=1,
        citations_sufficient_applicable_count=2,
        scorer_versions=("4",),
        input_tokens_known_count=0,
        output_tokens_known_count=0,
        actual_usd_known_count=0,
    )

    rendered = render_evaluation_summary(summary)
    lines = rendered.splitlines()
    correct_and_grounded_index = next(
        index for index, line in enumerate(lines) if "correct_and_grounded:" in line
    )
    diagnosis_correct_index = next(
        index for index, line in enumerate(lines) if "diagnosis_correct:" in line
    )

    assert correct_and_grounded_index < diagnosis_correct_index


def test_render_evaluation_summary_of_an_empty_batch_says_no_data() -> None:
    rendered = render_evaluation_summary(summarize_evaluation([]))

    assert "evaluation summary: 0 record(s)" in rendered
    assert "no data" in rendered


def test_main_writes_a_summary_alongside_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end through `main()`: the aggregate summary this fix adds must
    actually reach both stdout and a saved `summary.json`, not just exist as
    a pure function nothing calls."""
    _disable_preflight_cost_check(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        pass

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        model = ReplayToolCallingModel(
            ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
        )
        registry = registry_with(
            run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
        )
        ledger_conn = sqlite3.connect(":memory:")
        return model, registry, "fake-claude-model", ledger_conn

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", _fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    # The 12-incident corpus (EVALUATION_FAMILIES x EVALUATION_SEEDS) at the
    # default --executed-tools 2: `main()` prints one block per
    # (arm, retrieval_mode, executed_tools) group plus a plain trailing
    # record count -- every run here is the replay fixture's fake model,
    # which never calls `search_runbooks`, so both arms stay
    # `RetrievalMode.DISABLED` and both stay at executed_tools=2, so each
    # arm still produces exactly one group of 12 records; the trailing
    # count reports all 24, but carries no diagnosis/citation/control/
    # latency/cost figure of its own.
    incidents = len(EVALUATION_FAMILIES) * len(EVALUATION_SEEDS)
    total = incidents * 2
    assert (
        f"[{MODE_NO_TOOL_BASELINE}, retrieval_mode=disabled, executed_tools=2]" in out
    )
    assert f"[{MODE_TOOL_ENABLED}, retrieval_mode=disabled, executed_tools=2]" in out
    assert (
        "total_records (all arms, retrieval modes, and evidence budgets): "
        f"{total}" in out
    )
    # No blended benchmark figure anywhere in the output: every
    # "evaluation summary: N record(s)" block (produced only by
    # `render_evaluation_summary`, one call per group) reports exactly
    # `incidents`, the size of one (arm, retrieval_mode, executed_tools)
    # group -- never `total`, which would mean a figure spanning both arms.
    assert out.count("evaluation summary:") == 2
    assert f"evaluation summary: {incidents} record(s)" in out
    assert f"evaluation summary: {total} record(s)" not in out
    assert "summary:" in out
    summary_path = (
        next((tmp_path / "results" / "evaluations").iterdir()) / "summary.json"
    )
    assert str(summary_path) in out
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    # `summary.json` is now `PairedEvaluationSummary`'s `groups`/
    # `total_records` shape -- each group readable separately, and
    # `total_records` is a bare count with no `diagnosis_correct_count`,
    # `citations_valid_count`, latency, or cost figure anywhere on it, so no
    # single reported field in this file blends records across retrieval
    # modes or evidence budgets.
    assert len(saved["groups"]) == 2
    groups_by_arm = {group["arm"]: group for group in saved["groups"]}
    assert groups_by_arm[MODE_NO_TOOL_BASELINE]["retrieval_mode"] == "disabled"
    assert groups_by_arm[MODE_NO_TOOL_BASELINE]["executed_tools"] == 2
    assert groups_by_arm[MODE_NO_TOOL_BASELINE]["summary"]["total_records"] == incidents
    assert groups_by_arm[MODE_TOOL_ENABLED]["retrieval_mode"] == "disabled"
    assert groups_by_arm[MODE_TOOL_ENABLED]["executed_tools"] == 2
    assert groups_by_arm[MODE_TOOL_ENABLED]["summary"]["total_records"] == incidents
    assert saved["total_records"] == total
    assert "combined" not in saved
    assert set(saved.keys()) == {"groups", "total_records"}


def test_main_reports_a_clean_failure_when_the_summary_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`summary_path.write_text(...)` used to run outside every
    `try/except` in `main()` -- a write failure there (disk full, permission
    error) escaped as a raw traceback instead of the clean `FAIL ...`
    message every other refusal path in this script already gives, even
    though `records.jsonl` was already durably written by this point. Only
    the summary's own write is made to fail here (matched by filename
    prefix, not by patching `Path.write_text` unconditionally), so `write_
    jsonl`'s own earlier writes -- which is how `records.jsonl` gets to
    disk in the first place -- are unaffected. `_write_json_atomic` writes
    to a sibling `summary.json.tmp-<hex>` before renaming it onto
    `summary.json`, so the matched name is a prefix, not the exact final
    filename."""
    _disable_preflight_cost_check(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for family in EVALUATION_FAMILIES:
        scenarios = tmp_path / "lab" / "scenarios"
        scenarios.mkdir(parents=True, exist_ok=True)
        (scenarios / f"{family}.json").write_text(
            json.dumps({"family": family}), encoding="utf-8"
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_reset_scenario(root: Path, incident_id: str) -> None:
        pass

    def fake_build_model_and_registry(
        incident: StoredIncident,
        paths: object,
        budgets: Budgets,
        model_choice: str,
        db_path: Path,
    ) -> tuple[object, object, str, object]:
        model = ReplayToolCallingModel(
            ReplayReasoningModel(FIXTURE_DIR / "valid_diagnosis.json")
        )
        registry = registry_with(
            run_metric=RecordingMetricBackend(), run_logs=RecordingLogsBackend()
        )
        ledger_conn = sqlite3.connect(":memory:")
        return model, registry, "fake-claude-model", ledger_conn

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", _fake_start_scenario)
    monkeypatch.setattr("causalops.evaluate_cli.reset_scenario", fake_reset_scenario)
    monkeypatch.setattr(
        "causalops.evaluate_cli.build_model_and_registry", fake_build_model_and_registry
    )
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    original_write_text = Path.write_text

    def failing_write_text(self: Path, *args: object, **kwargs: object) -> int:
        if self.name.startswith("summary.json"):
            raise OSError("simulated summary write failure")
        return original_write_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    exit_code = main([])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAIL SUMMARY_WRITE_FAILED" in out
    assert "simulated summary write failure" in out
    records_path = (
        next((tmp_path / "results" / "evaluations").iterdir()) / "records.jsonl"
    )
    assert "records (unaffected):" in out
    assert str(records_path) in out
    # `records.jsonl` itself is untouched by the summary write failure --
    # every real, already-billed result is still readable back.
    on_disk = [
        EvaluationRecord.model_validate_json(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(on_disk) == len(EVALUATION_FAMILIES) * len(EVALUATION_SEEDS) * 2


class _TinyPayload(BaseModel):
    """A minimal stand-in for `PairedEvaluationSummary` -- these tests are
    about `_write_json_atomic`'s own file-write mechanics, not about
    evaluation scoring, so a one-field model keeps that focus."""

    value: int


def test_write_json_atomic_writes_the_complete_content(tmp_path: Path) -> None:
    """`_write_json_atomic`'s ordinary path: the full, valid
    JSON content lands at the target path, and nothing else is left behind
    in the directory -- the sibling temp file is gone once the rename
    completes."""
    target = tmp_path / "summary.json"

    _write_json_atomic(target, _TinyPayload(value=7))

    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 7}
    assert list(tmp_path.iterdir()) == [target]


def test_write_json_atomic_leaves_no_corrupted_file_on_an_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap this test exists for: a plain `path.write_text` truncates
    its target before writing a byte of the new content, so a hard process kill
    mid-write -- not a catchable `OSError`, so no `except` clause anywhere
    ever runs to warn about it -- could leave `summary.json` itself as a
    truncated, corrupted file with no exception ever having fired.
    Simulated here by monkeypatching `Path.write_text` to write only HALF
    its content and then raise, the same shape a hard kill mid-write leaves
    behind, without needing to actually kill this test process to prove the
    guarantee. `_write_json_atomic` writes to a sibling temp file first and
    only renames it onto the real target afterward, so an interruption here
    must leave no file at all at the target path -- never a truncated one
    -- and the temp file itself must be cleaned up, not left behind."""
    target = tmp_path / "summary.json"
    original_write_text = Path.write_text

    def interrupted_write_text(
        self: Path, content: str, *args: object, **kwargs: object
    ) -> int:
        original_write_text(self, content[: len(content) // 2], *args, **kwargs)
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(Path, "write_text", interrupted_write_text)

    with pytest.raises(OSError, match="simulated interrupted write"):
        _write_json_atomic(target, _TinyPayload(value=7))

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_json_atomic_preserves_prior_content_on_an_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This project's one real call site (`main`'s `summary.json` write)
    never has prior content to protect, but `_write_json_atomic` is written
    as a general atomic-replace helper -- the same contract `run_records.
    write_jsonl` already guarantees for `records.jsonl`, which IS rewritten
    repeatedly onto the same path across a batch. This proves that broader
    contract directly: an interrupted write must leave the ORIGINAL file
    exactly as it was, never a partial mix of old and new content."""
    target = tmp_path / "summary.json"
    target.write_text('{"value": 1}', encoding="utf-8")

    def interrupted_write_text(
        self: Path, content: str, *args: object, **kwargs: object
    ) -> int:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(Path, "write_text", interrupted_write_text)

    with pytest.raises(OSError, match="simulated interrupted write"):
        _write_json_atomic(target, _TinyPayload(value=2))

    assert target.read_text(encoding="utf-8") == '{"value": 1}'


def _paired_record(
    *,
    run_key: str,
    diagnosis_correct: bool,
    retrieval_mode: RetrievalMode = RetrievalMode.DISABLED,
    citations_sufficient: bool | None = True,
    has_predicate: bool = True,
    executed_tools: int = 2,
) -> EvaluationRecord:
    """A minimal `EvaluationRecord` for `summarize_paired_evaluation`/
    `render_paired_evaluation_summary` tests -- only `run_key` (which arm
    encodes the mode as its final `/`-segment), `diagnosis_correct`,
    `retrieval_mode`, `citations_sufficient`, `has_predicate`, and
    `executed_tools` vary across the records these tests build; everything
    else is fixed, plausible filler.

    `has_predicate` controls whether `expected.predicates` is non-empty.
    Defaults to `True` (a predicate-bearing outcome). A caller building a
    not-applicable `citations_sufficient=None` record must pass
    `has_predicate=False` explicitly: applicability is derived from
    `expected.predicates` itself, and real `score_run` output can never
    pair a non-empty `expected.predicates` with `citations_sufficient=None`.

    `executed_tools` is a real field on the record (not parsed out of
    `run_key`, which stays free-form for these tests) -- defaults to 2, the
    value every group test that does not care about the evidence-budget
    curve builds against."""
    return EvaluationRecord(
        run_key=run_key,
        investigation_id="inv-1",
        incident_id=run_key.split("/", 1)[0],
        expected=ExpectedOutcome(
            root_cause=RootCauseCode.DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION,
            disposition=Disposition.DIAGNOSED,
            predicates=(
                (
                    RequiredEvidencePredicate(
                        source="query_metric",
                        kind=EvidenceKind.METRIC,
                        template="downstream_timeout_rate",
                        field="timeouts_per_minute",
                        operator=PredicateOperator.AT_LEAST,
                        value=10,
                    ),
                )
                if has_predicate
                else ()
            ),
        ),
        scores=MechanicalScores(
            diagnosis_correct=diagnosis_correct,
            disposition_correct=True,
            citations_valid=True,
            citations_sufficient=citations_sufficient,
            control=ControlCounts(),
            efficiency=Efficiency(latency_ms=100, model_calls=1, tools_executed=0),
        ),
        git_sha="f" * 40,
        git_dirty=False,
        versions=Versions(
            prompt_version="1", policy_version="1", tool_registry_version="1"
        ),
        retrieval_mode=retrieval_mode,
        fixture_sha256="a" * 64,
        model_name="fake-claude-model",
        pricing_source="https://platform.claude.com/docs/en/about-claude/pricing",
        pricing_verified_on="2026-08-24",
        configured_ceiling_usd=5.0,
        reserved_usd=0.01,
        actual_usd=0.008,
        executed_tools=executed_tools,
    )


def _group(
    paired: object, arm: str, retrieval_mode: RetrievalMode, executed_tools: int = 2
) -> object:
    """Finds the one `EvaluationGroupSummary` for `(arm, retrieval_mode,
    executed_tools)` in `paired.groups` -- raises `StopIteration` (a clear
    test failure) if no such group exists, which is itself part of what
    these tests check: a group that should not exist must genuinely be
    absent, not merged into another one. `executed_tools` defaults to 2,
    the value every group test that does not care about the evidence-budget
    curve builds against."""
    return next(
        group
        for group in paired.groups  # type: ignore[attr-defined]
        if group.arm == arm
        and group.retrieval_mode == retrieval_mode
        and group.executed_tools == executed_tools
    )


def test_summarize_paired_evaluation_distinguishes_the_two_arms() -> None:
    """`evaluate_cli.py`'s `main()` used to call
    `summarize_evaluation(records)` once on the full flat 8-record list --
    a batch that scores `6/8 diagnosis_correct` gives no way to tell that
    apart from `2/4 baseline + 4/4 tool-enabled` versus the reverse split.
    Builds a batch where the baseline arm gets 1/2 correct and the
    tool-enabled arm gets 2/2 correct and proves both the structured summary
    and its rendered text keep the two arms visibly distinct, not blended
    into one number. Every record here stays `RetrievalMode.DISABLED`, so
    each arm produces exactly one group -- the retrieval-mode split itself
    is covered separately below."""
    records = [
        _paired_record(
            run_key=f"incident-a/model/{MODE_NO_TOOL_BASELINE}", diagnosis_correct=True
        ),
        _paired_record(
            run_key=f"incident-b/model/{MODE_NO_TOOL_BASELINE}",
            diagnosis_correct=False,
        ),
        _paired_record(
            run_key=f"incident-a/model/{MODE_TOOL_ENABLED}", diagnosis_correct=True
        ),
        _paired_record(
            run_key=f"incident-b/model/{MODE_TOOL_ENABLED}", diagnosis_correct=True
        ),
    ]

    summary = summarize_paired_evaluation(records)

    assert len(summary.groups) == 2
    baseline_group = _group(summary, MODE_NO_TOOL_BASELINE, RetrievalMode.DISABLED)
    tool_enabled_group = _group(summary, MODE_TOOL_ENABLED, RetrievalMode.DISABLED)
    assert baseline_group.summary.total_records == 2
    assert baseline_group.summary.diagnosis_correct_count == 1
    assert tool_enabled_group.summary.total_records == 2
    assert tool_enabled_group.summary.diagnosis_correct_count == 2
    # A plain batch-wide count is still reported too -- the per-arm split
    # does not remove the total record count a reader may also want -- but,
    # unlike the removed `combined` field, it carries no
    # `diagnosis_correct_count` or other benchmark figure blending both
    # arms into one number.
    assert summary.total_records == 4
    assert not hasattr(summary, "combined")

    rendered = render_paired_evaluation_summary(summary)

    assert (
        f"[{MODE_NO_TOOL_BASELINE}, retrieval_mode=disabled, executed_tools=2]"
        in rendered
    )
    assert (
        f"[{MODE_TOOL_ENABLED}, retrieval_mode=disabled, executed_tools=2]" in rendered
    )
    assert (
        "total_records (all arms, retrieval modes, and evidence budgets): 4" in rendered
    )
    # No merged "diagnosis_correct: 3/4" figure anywhere -- each arm's own
    # line stays visibly separate, and the trailing total carries no
    # diagnosis figure of its own to blend them.
    assert "diagnosis_correct:   1/2" in rendered
    assert "diagnosis_correct:   2/2" in rendered
    assert "diagnosis_correct:   3/4" not in rendered


def test_summarize_paired_evaluation_keeps_retrieval_modes_within_an_arm_apart() -> (
    None
):
    """The behaviour under test: the no-tool baseline is always
    `RetrievalMode.DISABLED` structurally, but within the TOOL-ENABLED arm,
    `retrieval_mode` depends on whether the model actually called
    `search_runbooks` on that particular run -- so two tool-enabled runs in
    the same batch can legitimately land in different retrieval modes. This
    builds exactly that: one tool-enabled run that never retrieved
    (`DISABLED`) and one that did (`FTS5_LEXICAL`), reproducing the scenario
    this project's "never mix modes in one benchmark aggregate"
    rule forbids blending. Before this fix, `summarize_paired_evaluation`
    partitioned by arm alone, so both records would have landed in one
    `tool_enabled` bucket with no way to tell the two retrieval modes
    apart.

    Also proves that no blended figure crosses retrieval modes: with three
    records spanning two retrieval modes (2 correct, 1 incorrect), a
    blended figure across all of them would read `diagnosis_correct: 2/3`
    -- this test confirms that exact string never appears anywhere in the
    structured summary or its rendered text, because no field left on
    `PairedEvaluationSummary` computes a figure across more than one
    `(arm, retrieval_mode)` group."""
    records = [
        _paired_record(
            run_key=f"incident-a/model/{MODE_NO_TOOL_BASELINE}",
            diagnosis_correct=True,
            retrieval_mode=RetrievalMode.DISABLED,
        ),
        _paired_record(
            run_key=f"incident-a/model/{MODE_TOOL_ENABLED}",
            diagnosis_correct=True,
            retrieval_mode=RetrievalMode.DISABLED,
        ),
        _paired_record(
            run_key=f"incident-b/model/{MODE_TOOL_ENABLED}",
            diagnosis_correct=False,
            retrieval_mode=RetrievalMode.FTS5_LEXICAL,
        ),
    ]

    summary = summarize_paired_evaluation(records)

    # Three distinct (arm, retrieval_mode) pairs are present -- one baseline
    # group and TWO tool-enabled groups, never one merged tool-enabled
    # bucket.
    assert len(summary.groups) == 3
    tool_enabled_no_retrieval = _group(
        summary, MODE_TOOL_ENABLED, RetrievalMode.DISABLED
    )
    tool_enabled_retrieved = _group(
        summary, MODE_TOOL_ENABLED, RetrievalMode.FTS5_LEXICAL
    )
    assert tool_enabled_no_retrieval.summary.total_records == 1
    assert tool_enabled_no_retrieval.summary.diagnosis_correct_count == 1
    assert tool_enabled_retrieved.summary.total_records == 1
    assert tool_enabled_retrieved.summary.diagnosis_correct_count == 0
    # The fix itself: `PairedEvaluationSummary` no longer has any field
    # computed across the whole batch except a bare count. There is no
    # `combined` attribute, and `total_records` (unlike the removed
    # `combined.diagnosis_correct_count`) carries no diagnosis figure at
    # all -- it cannot blend the two retrieval modes' correctness data
    # because it does not report correctness data.
    assert not hasattr(summary, "combined")
    assert summary.total_records == 3

    rendered = render_paired_evaluation_summary(summary)

    # Both tool-enabled groups render as visibly separate blocks -- neither
    # retrieval-mode figure below is blended into the other.
    no_retrieval_label = (
        f"[{MODE_TOOL_ENABLED}, retrieval_mode=disabled, executed_tools=2]"
    )
    retrieved_label = (
        f"[{MODE_TOOL_ENABLED}, retrieval_mode=fts5_lexical, executed_tools=2]"
    )
    assert no_retrieval_label in rendered
    assert retrieved_label in rendered
    no_retrieval_block, retrieved_block = (
        rendered.split(no_retrieval_label)[1].split(retrieved_label)[0],
        rendered.split(retrieved_label)[1],
    )
    assert "diagnosis_correct:   1/1" in no_retrieval_block
    assert "diagnosis_correct:   0/1" in retrieved_block
    # The exact blended figure a cross-mode aggregate would report (2 of
    # the batch's 3 records are diagnosis_correct, spanning both retrieval
    # modes and both arms) appears nowhere -- proving no reported field
    # blends diagnosis data across the different retrieval modes present in
    # this batch.
    assert "diagnosis_correct:   2/3" not in rendered
    assert (
        "total_records (all arms, retrieval modes, and evidence budgets): 3" in rendered
    )


def test_summarize_paired_evaluation_keeps_evidence_budget_settings_apart() -> None:
    """The single most important fix in the evidence-budget-curve unit:
    without partitioning by `executed_tools` too, every budget-setting
    variant of the tool-enabled arm would collapse into one blended
    summary, and the entire curve this design exists to produce would be
    lost even though the underlying records are genuinely distinct. Builds
    a batch with identical arm/retrieval_mode but three different
    `executed_tools` values and proves `summarize_paired_evaluation`
    produces three separate groups, not one."""
    records = [
        _paired_record(
            run_key=f"incident-a/model/et2/{MODE_TOOL_ENABLED}",
            diagnosis_correct=True,
            executed_tools=2,
        ),
        _paired_record(
            run_key=f"incident-b/model/et3/{MODE_TOOL_ENABLED}",
            diagnosis_correct=False,
            executed_tools=3,
        ),
        _paired_record(
            run_key=f"incident-c/model/et4/{MODE_TOOL_ENABLED}",
            diagnosis_correct=True,
            executed_tools=4,
        ),
    ]

    summary = summarize_paired_evaluation(records)

    assert len(summary.groups) == 3
    et2_group = _group(
        summary, MODE_TOOL_ENABLED, RetrievalMode.DISABLED, executed_tools=2
    )
    et3_group = _group(
        summary, MODE_TOOL_ENABLED, RetrievalMode.DISABLED, executed_tools=3
    )
    et4_group = _group(
        summary, MODE_TOOL_ENABLED, RetrievalMode.DISABLED, executed_tools=4
    )
    assert et2_group.summary.total_records == 1
    assert et2_group.summary.diagnosis_correct_count == 1
    assert et3_group.summary.total_records == 1
    assert et3_group.summary.diagnosis_correct_count == 0
    assert et4_group.summary.total_records == 1
    assert et4_group.summary.diagnosis_correct_count == 1
    assert summary.total_records == 3

    rendered = render_paired_evaluation_summary(summary)

    assert (
        f"[{MODE_TOOL_ENABLED}, retrieval_mode=disabled, executed_tools=2]" in rendered
    )
    assert (
        f"[{MODE_TOOL_ENABLED}, retrieval_mode=disabled, executed_tools=3]" in rendered
    )
    assert (
        f"[{MODE_TOOL_ENABLED}, retrieval_mode=disabled, executed_tools=4]" in rendered
    )
    # No blended figure across the three budget points anywhere -- the exact
    # figure a cross-budget aggregate would report (2 of the batch's 3
    # records are diagnosis_correct) never appears.
    assert "diagnosis_correct:   2/3" not in rendered


def test_summarize_paired_evaluation_reports_citations_sufficient_per_group() -> None:
    """Proves `EvaluationGroupSummary.summary.citations_sufficient_count`/
    `citations_sufficient_applicable_count` are correct per group, not just
    structurally present -- the claim under test is that
    `summarize_paired_evaluation` needs no change because it delegates
    entirely to `summarize_evaluation` per `(arm, retrieval_mode)` group.
    This test proves that claim directly: each group's counts are compared
    against calling `summarize_evaluation` on that exact subset of records
    directly, not merely asserted to be plausible numbers.

    A real mix of applicable-true, applicable-false, and not-applicable
    records is required to exercise this meaningfully -- `has_predicate` is
    wired consistently with each record's own `citations_sufficient` value
    so applicability, derived from `expected.predicates` rather than from
    `citations_sufficient is None` alone, cannot collapse every record in
    this batch to not-applicable and make the counts below vacuously
    `0 == 0`."""
    baseline_records = [
        _paired_record(
            run_key=f"incident-a/model/{MODE_NO_TOOL_BASELINE}",
            diagnosis_correct=True,
            citations_sufficient=True,
            has_predicate=True,
        ),
        _paired_record(
            run_key=f"incident-b/model/{MODE_NO_TOOL_BASELINE}",
            diagnosis_correct=False,
            citations_sufficient=None,
            has_predicate=False,
        ),
    ]
    tool_enabled_records = [
        _paired_record(
            run_key=f"incident-a/model/{MODE_TOOL_ENABLED}",
            diagnosis_correct=True,
            citations_sufficient=False,
            has_predicate=True,
        ),
        _paired_record(
            run_key=f"incident-b/model/{MODE_TOOL_ENABLED}",
            diagnosis_correct=True,
            citations_sufficient=True,
            has_predicate=True,
        ),
    ]
    records = baseline_records + tool_enabled_records

    paired = summarize_paired_evaluation(records)

    baseline_group = _group(paired, MODE_NO_TOOL_BASELINE, RetrievalMode.DISABLED)
    tool_enabled_group = _group(paired, MODE_TOOL_ENABLED, RetrievalMode.DISABLED)
    expected_baseline = summarize_evaluation(baseline_records)
    expected_tool_enabled = summarize_evaluation(tool_enabled_records)

    assert (
        baseline_group.summary.citations_sufficient_count
        == expected_baseline.citations_sufficient_count
        == 1
    )
    assert (
        baseline_group.summary.citations_sufficient_applicable_count
        == expected_baseline.citations_sufficient_applicable_count
        == 1
    )
    assert (
        tool_enabled_group.summary.citations_sufficient_count
        == expected_tool_enabled.citations_sufficient_count
        == 1
    )
    assert (
        tool_enabled_group.summary.citations_sufficient_applicable_count
        == expected_tool_enabled.citations_sufficient_applicable_count
        == 2
    )


def test_summarize_paired_evaluation_rejects_an_unrecognized_run_key_mode() -> None:
    """A `run_key` whose final segment is neither known mode word is a
    data-shape bug upstream, not a value `summarize_paired_evaluation`
    should silently drop from one arm's scored total."""
    record = _paired_record(
        run_key="incident-a/model/mystery_mode", diagnosis_correct=True
    )

    with pytest.raises(ValueError, match="cannot partition this batch by arm"):
        summarize_paired_evaluation([record])


def test_main_reports_the_ceiling_reason_code_not_internal_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`live_setup.live_evaluation_ceiling_usd` raises `CheckpointStoreError(
    CEILING_BELOW_RESERVATION_BUFFER, ...)` for a too-small
    `LIVE_EVALUATION_MAX_USD` -- `.env.example` documents that exact `FAIL
    CEILING_BELOW_RESERVATION_BUFFER` output. Before this fix, `main()`'s
    typed refusal handler only caught `(LabError, RunRecordError)`, so
    `CheckpointStoreError` fell through to the generic `except Exception`
    and reported the opaque `FAIL INTERNAL_ERROR` instead, contradicting
    that documented contract. `_git_provenance` is faked, the same seam
    every other `main()` test in this file uses, so this fails at the
    ceiling check itself rather than needing a real git repo -- the ceiling
    is read before `start_scenario` is ever called, so no scenario-controller
    fake is needed either."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_EVALUATION_MAX_USD", "0.05")
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )

    exit_code = main([])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAIL CEILING_BELOW_RESERVATION_BUFFER" in out
    assert "FAIL INTERNAL_ERROR" not in out
    assert "records so far:" in out


def test_main_refuses_a_live_evaluation_without_a_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert main([]) == 1

    assert "FAIL MISSING_API_KEY" in capsys.readouterr().out


def test_main_fails_cleanly_when_the_evaluation_target_cannot_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bug this test exists for: `_new_evaluation_target` used to run
    before `main`'s `try:` block, so a failure there (read-only filesystem,
    full disk, permission error) escaped as a raw, uncaught traceback
    instead of the clean `FAIL ...` every other refusal path in this script
    already gives. `records.jsonl` genuinely does not exist yet for this
    specific failure, so this only asserts the clean failure message, not a
    'records so far' line."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def failing_target(root: Path) -> Path:
        raise OSError("simulated read-only filesystem")

    monkeypatch.setattr("causalops.evaluate_cli._new_evaluation_target", failing_target)

    exit_code = main([])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAIL EVALUATION_TARGET_UNWRITABLE" in out
    assert "simulated read-only filesystem" in out


def test_preflight_worst_case_batch_usd_scales_with_model_calls() -> None:
    """The formula itself: runs-per-invocation (fixed, regardless of which
    curve point is selected) x `budgets.model_calls` x
    `MAXIMUM_POSSIBLE_RESERVATION_USD`."""
    runs = len(EVALUATION_FAMILIES) * len(EVALUATION_SEEDS) * 2
    for executed_tools in EVIDENCE_BUDGET_CURVE:
        budgets = resolve_budgets(executed_tools)
        expected = runs * budgets.model_calls * MAXIMUM_POSSIBLE_RESERVATION_USD
        assert _preflight_worst_case_batch_usd(budgets) == pytest.approx(expected)


def test_check_preflight_cost_passes_with_sufficient_headroom(tmp_path: Path) -> None:
    """An empty, freshly-created ledger and a generous ceiling: the
    pre-flight check must not raise."""
    conn_path = tmp_path / "checkpoints.db"
    budgets = resolve_budgets(2)
    worst_case = _preflight_worst_case_batch_usd(budgets)

    _check_preflight_cost(
        conn_path, budgets, worst_case + RESERVATION_CEILING_BUFFER_USD + 1.0
    )  # must not raise


def test_check_preflight_cost_refuses_when_the_batch_could_exceed_the_ceiling(
    tmp_path: Path,
) -> None:
    """A ceiling one cent below this batch's own worst case must be refused
    before any scenario starts -- `CheckpointStoreError` with the same
    `CEILING_BELOW_RESERVATION_BUFFER` code `live_evaluation_ceiling_usd`
    itself already uses for a too-small ceiling, so `main`'s existing typed
    refusal handler reports it identically."""
    conn_path = tmp_path / "checkpoints.db"
    budgets = resolve_budgets(2)
    worst_case = _preflight_worst_case_batch_usd(budgets)
    too_small = worst_case + RESERVATION_CEILING_BUFFER_USD - 0.01

    with pytest.raises(CheckpointStoreError) as excinfo:
        _check_preflight_cost(conn_path, budgets, too_small)

    assert (
        excinfo.value.reason_code
        is CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER
    )
    assert "worst-case batch cost" in str(excinfo.value)


def test_check_preflight_cost_accounts_for_already_spent_application_wide(
    tmp_path: Path,
) -> None:
    """A ceiling that would cover this batch's worst case ALONE is still
    refused once existing application-wide spend is added -- the pre-flight
    check reads the real ledger, not just this invocation's own cost."""
    conn_path = tmp_path / "checkpoints.db"
    conn_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(conn_path))
    try:
        ensure_cost_ledger_table(conn)
        record_reservation_before_request(
            conn,
            run_id="already-spent",
            graph_phase="INVESTIGATE",
            model_turn=0,
            context_digest="digest-1",
            reserved_usd=4.0,
            requested_at=datetime(2026, 8, 24, tzinfo=UTC),
            ceiling_usd=1_000_000.0,
        )
    finally:
        conn.close()
    budgets = resolve_budgets(2)
    worst_case = _preflight_worst_case_batch_usd(budgets)
    # Exactly enough for the batch alone, none of the 4.0 already reserved.
    ceiling = worst_case + RESERVATION_CEILING_BUFFER_USD

    with pytest.raises(CheckpointStoreError) as excinfo:
        _check_preflight_cost(conn_path, budgets, ceiling)

    assert (
        excinfo.value.reason_code
        is CheckpointStoreReasonCode.CEILING_BELOW_RESERVATION_BUFFER
    )


def test_main_refuses_before_any_scenario_starts_when_the_batch_is_too_costly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end through `main()`: a `LIVE_EVALUATION_MAX_USD` well above
    the per-request floor (so `live_evaluation_ceiling_usd` itself accepts
    it) but below this invocation's own worst-case batch cost must still be
    refused, before `start_scenario` is ever called -- proven directly by
    asserting `start_scenario` was never invoked, not just by the exit code
    and message."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_EVALUATION_MAX_USD", "1.00")
    monkeypatch.setattr(
        "causalops.evaluate_cli._git_provenance", lambda root: ("f" * 40, False)
    )
    started: list[str] = []

    def spy_start_scenario(root: Path, family: str, seed: str) -> str:
        started.append(family)
        raise AssertionError("start_scenario must not be called")

    monkeypatch.setattr("causalops.evaluate_cli.start_scenario", spy_start_scenario)

    exit_code = main([])

    assert exit_code == 1
    assert started == []
    out = capsys.readouterr().out
    assert "FAIL CEILING_BELOW_RESERVATION_BUFFER" in out
    assert "worst-case batch cost" in out
