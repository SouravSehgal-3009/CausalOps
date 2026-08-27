import json
import shutil
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from causalops import scenario_control
from causalops.domain import RootCauseCode, StoredIncident
from causalops.scenario_control import (
    LabError,
    LabReasonCode,
    active_incident_file,
    apply_seed_variant,
    compose,
    reset_scenario,
    run_paths,
    runs_root,
    start_scenario,
    validated_run_paths,
    write_json,
)
from causalops.telemetry import RunPaths

FAMILY = "configuration_change"
FAMILIES = [
    "configuration_change",
    "downstream_timeout_retry_amplification",
    "resource_pool_saturation",
    "ambiguous_telemetry",
]
FAMILY_SYMPTOM = {
    "configuration_change": "ELEVATED_ERRORS",
    "downstream_timeout_retry_amplification": "ELEVATED_ERRORS_AND_LATENCY",
    "resource_pool_saturation": "ELEVATED_ERRORS_AND_LATENCY",
    "ambiguous_telemetry": "ELEVATED_ERRORS_AND_LATENCY",
}
FAMILY_ROOT_CAUSE = {
    "configuration_change": "CONFIG_CHANGE",
    "downstream_timeout_retry_amplification": "DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION",
    "resource_pool_saturation": "RESOURCE_POOL_SATURATION",
    "ambiguous_telemetry": "UNDETERMINED",
}
FAMILY_DISPOSITION = {
    "configuration_change": "DIAGNOSED",
    "downstream_timeout_retry_amplification": "DIAGNOSED",
    "resource_pool_saturation": "DIAGNOSED",
    "ambiguous_telemetry": "INSUFFICIENT_EVIDENCE",
}
REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway project root holding the real checked-in scenario definitions."""
    scenarios = tmp_path / "lab" / "scenarios"
    scenarios.mkdir(parents=True)
    for family in FAMILIES:
        shutil.copy(
            REPOSITORY / "lab" / "scenarios" / f"{family}.json",
            scenarios / f"{family}.json",
        )
    monkeypatch.setattr(scenario_control, "REQUEST_PAUSE_SECONDS", 0.0)
    # Same reason as `REQUEST_PAUSE_SECONDS`
    # above: every `start_scenario(...)` call in this file goes through
    # this fixture and uses the default (real) `sleeper`, which reads this
    # module-level constant fresh on every call -- zeroing it here keeps
    # this whole file a real unit suite instead of adding 12 real seconds
    # to every one of its `start_scenario` calls.
    monkeypatch.setattr(scenario_control, "SCRAPE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(
        scenario_control, "call_gateway", gateway_reading_config(tmp_path)
    )
    return tmp_path


def use_family_gateway(
    monkeypatch: pytest.MonkeyPatch, root: Path, family: str
) -> None:
    monkeypatch.setattr(scenario_control, "call_gateway", GATEWAYS[family](root))


def read_active_config(root: Path) -> dict[str, object]:
    incident = active_incident_file(root).read_text(encoding="utf-8").strip()
    config: dict[str, object] = json.loads(
        (runs_root(root) / incident / "lab" / "config.json").read_text(encoding="utf-8")
    )
    return config


def test_validated_run_paths_refuses_a_symlink_loop(tmp_path: Path) -> None:
    incident_id = "loop"
    target = runs_root(tmp_path) / incident_id
    target.parent.mkdir(parents=True)
    target.symlink_to(target)

    with pytest.raises(LabError) as excinfo:
        validated_run_paths(tmp_path, incident_id)

    assert excinfo.value.reason_code is LabReasonCode.INCIDENT_NOT_FOUND


def gateway_reading_config(root: Path) -> Callable[[], int]:
    """A stand-in gateway that fails exactly when the faulted setting is in place.

    It reads the same file the real orders service reads, so the test proves the
    controller flips the configuration before it drives the fault traffic.
    """

    def call() -> int:
        config = read_active_config(root)
        return 500 if config.get("require_order_token") else 200

    return call


def gateway_reading_response_delay(root: Path) -> Callable[[], int]:
    """Mirrors orders: any injected inventory delay times the request out."""

    def call() -> int:
        config = read_active_config(root)
        return 504 if config.get("response_delay_seconds", 0) else 200

    return call


def gateway_reading_pool_capacity(root: Path) -> Callable[[], int]:
    """Mirrors LeakyPool: a per-incident count that never releases a slot."""
    counts: dict[str, int] = {}

    def call() -> int:
        incident = active_incident_file(root).read_text(encoding="utf-8").strip()
        config = read_active_config(root)
        counts[incident] = counts.get(incident, 0) + 1
        capacity = config.get("pool_capacity")
        if capacity is not None and counts[incident] > capacity:
            return 500
        return 200

    return call


def gateway_reading_pool_or_delay(root: Path) -> Callable[[], int]:
    """The ambiguous family: either active knob can make a request fail."""
    counts: dict[str, int] = {}

    def call() -> int:
        incident = active_incident_file(root).read_text(encoding="utf-8").strip()
        config = read_active_config(root)
        counts[incident] = counts.get(incident, 0) + 1
        capacity = config.get("pool_capacity")
        pool_exhausted = capacity is not None and counts[incident] > capacity
        delayed = bool(config.get("response_delay_seconds", 0))
        return 500 if pool_exhausted or delayed else 200

    return call


GATEWAYS: dict[str, Callable[[Path], Callable[[], int]]] = {
    "configuration_change": gateway_reading_config,
    "downstream_timeout_retry_amplification": gateway_reading_response_delay,
    "resource_pool_saturation": gateway_reading_pool_capacity,
    "ambiguous_telemetry": gateway_reading_pool_or_delay,
}


def stored_incident(project: Path, incident_id: str) -> StoredIncident:
    text = (runs_root(project) / incident_id / "incident.json").read_text(
        encoding="utf-8"
    )
    return StoredIncident.model_validate_json(text)


@pytest.mark.parametrize("family", FAMILIES)
def test_starting_a_scenario_leaves_a_complete_run_directory(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    use_family_gateway(monkeypatch, project, family)

    incident_id = start_scenario(project, family, "development")

    run = runs_root(project) / incident_id
    assert len(incident_id) == 32
    assert (run / "logs").is_dir()
    assert (run / "changes.json").is_file()
    assert (run / "topology.json").is_file()
    assert (run / "incident.json").is_file()
    assert (run / "evaluator" / "expected.json").is_file()
    assert active_incident_file(project).read_text(encoding="utf-8") == incident_id


@pytest.mark.parametrize("family", FAMILIES)
def test_the_fault_is_applied_after_the_healthy_baseline(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    use_family_gateway(monkeypatch, project, family)

    incident_id = start_scenario(project, family, "development")

    incident = stored_incident(project, incident_id)
    assert incident.packet.symptom.value == FAMILY_SYMPTOM[family]
    payload = dict(incident.evidence[0].payload)
    assert payload["failed_requests"] > 0


def test_recorded_changes_fall_inside_the_recorded_window(project: Path) -> None:
    incident_id = start_scenario(project, FAMILY, "development")

    incident = stored_incident(project, incident_id)
    changes = json.loads(
        (runs_root(project) / incident_id / "changes.json").read_text(encoding="utf-8")
    )
    for change in changes:
        assert incident.scope.started_at.isoformat() <= change["at"]
        assert change["at"] <= incident.scope.ended_at.isoformat()


def test_start_scenario_settles_before_stamping_window_end(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `project` fixture zeroes
    `SCRAPE_SETTLE_SECONDS` for every other test in this file (so the suite
    does not spend 12 real seconds per scenario start); this test restores
    the real constant and passes a recording `sleeper` explicitly instead,
    asserting *that* the wait happened, *for how long*, and *where* --
    strictly after the fault-traffic phase, strictly before `window_end` is
    read from `clock` -- a stronger assertion than a real sleep would give,
    and the reason `start_scenario` takes an injectable `sleeper` at all.

    `clock` is called exactly twice by `start_scenario`: once for
    `window_start` (before any traffic), once for `window_end` (after the
    fault phase). Recording both `clock` and `sleeper` calls into one
    shared, ordered list proves the settle wait lands strictly between
    them, not merely that it happened somewhere."""
    monkeypatch.setattr(scenario_control, "SCRAPE_SETTLE_SECONDS", 12)
    events: list[str] = []
    fixed_moment = datetime(2026, 1, 1, tzinfo=UTC)

    def recording_clock() -> datetime:
        events.append("clock")
        return fixed_moment

    def recording_sleeper(seconds: float) -> None:
        events.append(f"sleep:{seconds}")

    start_scenario(
        project, FAMILY, "development", recording_clock, sleeper=recording_sleeper
    )

    assert events == ["clock", "sleep:12", "clock"]


def test_w9_change_summary_reflects_the_resolved_config_value(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the evaluation seed's override
    (`pool_capacity` 9 -> 7 for `resource_pool_saturation`), the rendered
    change summary must state the value actually applied, not the
    scenario file's static prose -- which still says `9`, this family's
    own *development*-seed value, since `apply_seed_variant` never
    touched `entry["summary"]` before this fix. Proves `write_manifests`
    now reads the resolved `faulted_config`, not the static string."""
    family = "resource_pool_saturation"
    use_family_gateway(monkeypatch, project, family)

    incident_id = start_scenario(project, family, "evaluation")

    changes = json.loads(
        (runs_root(project) / incident_id / "changes.json").read_text(encoding="utf-8")
    )
    orders_change = next(c for c in changes if c["service"] == "orders")
    assert orders_change["summary"] == "configuration update: pool_capacity set to 7"


def test_w9_a_change_entry_with_no_config_key_keeps_its_static_summary(
    project: Path,
) -> None:
    """An entry naming no `config_key` (image
    rebuilds, dependency bumps) has no resolved config value to render
    from and must keep its own static prose, unaffected by the fix."""
    incident_id = start_scenario(project, FAMILY, "development")

    changes = json.loads(
        (runs_root(project) / incident_id / "changes.json").read_text(encoding="utf-8")
    )
    inventory_change = next(c for c in changes if c["service"] == "inventory")
    assert inventory_change["summary"] == "image rebuild: dependency bump to 2.4.1"


def test_a_change_entry_naming_an_unknown_config_key_raises() -> None:
    """`_resolved_change_summary`'s
    `config_key not in faulted_config` branch used to degrade silently to
    the entry's static `summary` text -- but that silent fallback can
    recreate the exact stale-summary defect this function exists to fix, just
    triggered by a scenario-authoring typo instead of a missing
    resolution step. A declared-but-unmatched `config_key` must now raise
    `LabError(UNKNOWN_CONFIG_KEY, ...)` instead, so a typo fails scenario
    startup loudly. No checked-in scenario file's `config_key` is ever
    actually missing from its own `faulted_config` (see
    `test_every_declared_config_key_resolves_in_its_own_faulted_config`
    below, which proves this for the whole corpus), so this branch is
    unreachable through `start_scenario`/`write_manifests` today and
    needs its own direct call to exercise."""
    entry = {
        "service": "orders",
        "config_key": "typo_d_key_name",
        "summary": "configuration update: static fallback prose",
    }
    faulted_config = {"pool_capacity": 7}

    with pytest.raises(LabError) as excinfo:
        scenario_control._resolved_change_summary(entry, faulted_config)

    assert excinfo.value.reason_code is LabReasonCode.UNKNOWN_CONFIG_KEY
    assert "typo_d_key_name" in str(excinfo.value)
    assert "orders" in str(excinfo.value)


def test_every_declared_config_key_resolves_in_its_own_faulted_config() -> None:
    """A corpus-wide self-check: for
    every checked-in scenario file, under its base (un-seeded) config and
    every declared `seed_variants` entry, every change entry's declared
    `config_key` (when present) must actually resolve in the resulting
    `faulted_config`. This is exactly the typo class
    `_resolved_change_summary` now raises on at scenario-start time --
    this test catches it in CI, before any scenario ever runs, instead of
    only when someone happens to start the specific faulted family/seed
    combination that exercises the typo."""
    scenarios_dir = REPOSITORY / "lab" / "scenarios"
    for path in sorted(scenarios_dir.glob("*.json")):
        definition = json.loads(path.read_text(encoding="utf-8"))
        resolutions = {"<base>": definition} | {
            seed: apply_seed_variant(definition, seed)
            for seed in definition.get("seed_variants", {})
        }
        for label, resolved in resolutions.items():
            faulted_config = resolved["faulted_config"]
            for entry in resolved["changes"]:
                config_key = entry.get("config_key")
                if config_key is not None:
                    assert config_key in faulted_config, (
                        f"{path.name} seed={label!r} service={entry['service']!r} "
                        f"declares config_key={config_key!r}, absent from "
                        "faulted_config"
                    )


def test_w10_alert_total_requests_covers_the_whole_window(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`window_start` opens `WINDOW_LEAD_IN`
    before the healthy baseline phase even runs, so the alert's own
    window covers baseline *and* fault -- `total_requests` must count
    both, not just the fault phase `build_incident` used to be passed.
    `configuration_change`'s baseline (6 requests, `require_order_token`
    false) is entirely healthy under `gateway_reading_config`, and its
    fault phase (10 requests, the setting flipped true) is entirely
    failed, so the pre-fix bug (`total_requests = served + failed`, fault
    phase only) reports `10 of 10` where the fix must report `10 of 16`."""
    use_family_gateway(monkeypatch, project, FAMILY)

    incident_id = start_scenario(project, FAMILY, "development")

    incident = stored_incident(project, incident_id)
    payload = dict(incident.evidence[0].payload)
    assert payload["failed_requests"] == 10
    assert payload["total_requests"] == 16


def test_nothing_the_investigator_can_read_names_the_family(project: Path) -> None:
    incident_id = start_scenario(project, FAMILY, "development")

    run = runs_root(project) / incident_id
    visible = [path for path in run.rglob("*") if "evaluator" not in path.parts]
    for path in visible:
        assert FAMILY not in path.name
        assert "config_change" not in path.name.lower()
    readable = "".join(
        path.read_text(encoding="utf-8") for path in visible if path.is_file()
    )
    assert FAMILY not in readable
    assert "development" not in readable
    for code in RootCauseCode:
        assert code.value not in readable


@pytest.mark.parametrize("family", FAMILIES)
def test_the_expected_outcome_stays_on_the_evaluator_side(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    use_family_gateway(monkeypatch, project, family)

    incident_id = start_scenario(project, family, "development")

    expected = json.loads(
        (runs_root(project) / incident_id / "evaluator" / "expected.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected["root_cause"] == FAMILY_ROOT_CAUSE[family]
    assert expected["disposition"] == FAMILY_DISPOSITION[family]
    assert expected["seed"] == "development"


@pytest.mark.parametrize("family", FAMILIES)
def test_a_lab_that_never_worked_stops_before_faulting(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    monkeypatch.setattr(scenario_control, "call_gateway", lambda: 503)

    with pytest.raises(LabError) as refused:
        start_scenario(project, family, "development")

    assert refused.value.reason_code is LabReasonCode.BASELINE_NOT_HEALTHY


@pytest.mark.parametrize("family", FAMILIES)
def test_a_fault_that_changes_nothing_is_reported(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    monkeypatch.setattr(scenario_control, "call_gateway", lambda: 200)

    with pytest.raises(LabError) as refused:
        start_scenario(project, family, "development")

    assert refused.value.reason_code is LabReasonCode.FAULT_NOT_OBSERVED


def test_only_one_scenario_may_be_active(project: Path) -> None:
    start_scenario(project, FAMILY, "development")

    with pytest.raises(LabError) as refused:
        start_scenario(project, FAMILY, "development")

    assert refused.value.reason_code is LabReasonCode.SCENARIO_ALREADY_ACTIVE


def test_claim_creation_failure_removes_its_candidate_directory(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = scenario_control.os.open

    def failing_open(*args: object, **kwargs: object) -> int:
        if args[0] == active_incident_file(project):
            raise OSError("simulated marker creation failure")
        return real_open(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(scenario_control.os, "open", failing_open)

    with pytest.raises(OSError, match="marker creation"):
        start_scenario(project, FAMILY, "development")

    assert not active_incident_file(project).exists()
    assert [path for path in runs_root(project).iterdir() if path.is_dir()] == []


def _attempt_claim(
    root: Path,
    incident_id: str,
    paths: RunPaths,
    barrier: threading.Barrier,
    results: dict[str, object],
    key: str,
) -> None:
    """One thread's side of `test_two_racing_claims_only_one_succeeds`'s
    race. Takes every dependency as an explicit argument, not a closure over
    the caller's loop variables -- a per-round closure would rebind on the
    next iteration before a still-running thread from this one reads it."""
    barrier.wait()
    try:
        with scenario_control._scenario_lock(root):
            paths.root.mkdir(parents=True, exist_ok=False)
            scenario_control._claim_scenario(root, incident_id, paths)
        results[key] = "claimed"
    except LabError as error:
        results[key] = error.reason_code


def test_two_racing_claims_only_one_succeeds(tmp_path: Path) -> None:
    """Proved by hand, in ad-hoc scripts never committed to the
    suite, that `_claim_scenario` inside `_scenario_lock` correctly
    serializes two contenders racing for the same active-scenario slot via
    SQLite's `BEGIN IMMEDIATE`. This puts that proof in the shipped suite,
    using real OS threads (not asyncio, not mocked timing) racing to claim
    two DIFFERENT incidents against the SAME marker -- exactly the shape a
    real double-launch of `causalops investigate` would take. A single race
    is not reliable proof of correct serialization, since a lucky thread
    interleaving can pass even with a real bug present, so this repeats the
    race across many rounds, each against a fresh root so a round's outcome
    can never leak into the next."""
    rounds = 25
    for round_index in range(rounds):
        root = tmp_path / f"round-{round_index}"
        incident_a = f"incident-a-{round_index}"
        incident_b = f"incident-b-{round_index}"
        paths_a = run_paths(root, incident_a)
        paths_b = run_paths(root, incident_b)
        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        thread_a = threading.Thread(
            target=_attempt_claim,
            args=(root, incident_a, paths_a, barrier, results, "a"),
        )
        thread_b = threading.Thread(
            target=_attempt_claim,
            args=(root, incident_b, paths_b, barrier, results, "b"),
        )
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()

        claimed = [key for key, outcome in results.items() if outcome == "claimed"]
        refused = [
            key
            for key, outcome in results.items()
            if outcome == LabReasonCode.SCENARIO_ALREADY_ACTIVE
        ]
        assert len(claimed) == 1, f"round {round_index}: {results}"
        assert len(refused) == 1, f"round {round_index}: {results}"


def test_a_stale_nonempty_marker_is_reclaimed_when_its_run_directory_is_absent(
    tmp_path: Path,
) -> None:
    """A marker naming an incident whose run directory no longer exists (the
    controller crashed, or cleanup already ran) is stale, not active -- the
    next claim silently recovers it rather than permanently blocking every
    future scenario start."""
    marker = active_incident_file(tmp_path)
    marker.parent.mkdir(parents=True)
    marker.write_text("an-incident-whose-directory-is-gone", encoding="utf-8")

    incident_id = "a-fresh-incident"
    paths = run_paths(tmp_path, incident_id)
    paths.root.mkdir(parents=True)

    with scenario_control._scenario_lock(tmp_path):
        scenario_control._claim_scenario(tmp_path, incident_id, paths)

    assert marker.read_text(encoding="utf-8") == incident_id


def test_an_empty_marker_is_never_reclaimed_even_without_a_directory(
    tmp_path: Path,
) -> None:
    """An empty marker means a claim is in progress (the exclusive create
    happens before the incident_id is written), so it must never be treated
    as stale just because there is, as yet, no run directory to check --
    reclaiming it would let a second claimant race into an already-owned
    slot."""
    marker = active_incident_file(tmp_path)
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")

    incident_id = "a-fresh-incident"
    paths = run_paths(tmp_path, incident_id)
    paths.root.mkdir(parents=True)

    with pytest.raises(LabError) as refused:
        with scenario_control._scenario_lock(tmp_path):
            scenario_control._claim_scenario(tmp_path, incident_id, paths)

    assert refused.value.reason_code is LabReasonCode.SCENARIO_ALREADY_ACTIVE
    assert marker.read_text(encoding="utf-8") == ""


def test_an_unknown_family_is_refused(project: Path) -> None:
    with pytest.raises(LabError) as refused:
        start_scenario(project, "not_a_family", "development")

    assert refused.value.reason_code is LabReasonCode.UNKNOWN_FAMILY


@pytest.mark.parametrize("family", FAMILIES)
def test_reset_deletes_the_run_and_never_the_results(
    project: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    use_family_gateway(monkeypatch, project, family)
    incident_id = start_scenario(project, family, "development")
    finalized = project / "results" / "investigations" / "inv-1"
    finalized.mkdir(parents=True)
    (finalized / "report.json").write_text("{}", encoding="utf-8")

    reset_scenario(project, incident_id)

    assert not (runs_root(project) / incident_id).exists()
    assert not active_incident_file(project).exists()
    assert (finalized / "report.json").read_text(encoding="utf-8") == "{}"


def test_reset_refuses_anything_that_is_not_an_incident(project: Path) -> None:
    for name in ("..", "../results", "unknown"):
        with pytest.raises(LabError) as refused:
            reset_scenario(project, name)
        assert refused.value.reason_code is LabReasonCode.INCIDENT_NOT_FOUND


def test_a_missing_docker_is_reported_rather_than_raised(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_docker(*arguments: object, **keywords: object) -> None:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(scenario_control.subprocess, "run", no_docker)

    with pytest.raises(LabError) as refused:
        compose(project, ["up", "-d"], 5)

    assert refused.value.reason_code is LabReasonCode.DOCKER_UNAVAILABLE


def test_apply_seed_variant_is_identity_without_seed_variants() -> None:
    definition = {
        "changes": [{"service": "orders", "summary": "x", "offset_seconds": -10}]
    }

    result = apply_seed_variant(definition, "development")

    assert result is definition


def test_apply_seed_variant_is_identity_for_an_unlisted_seed() -> None:
    definition = {
        "changes": [{"service": "orders", "summary": "x", "offset_seconds": -10}],
        "seed_variants": {"development": {"change_offsets": [-20]}},
    }

    result = apply_seed_variant(definition, "evaluation")

    assert result is definition


def test_apply_seed_variant_replaces_offsets_and_merges_config_overrides() -> None:
    definition = {
        "changes": [
            {"service": "orders", "summary": "a", "offset_seconds": -10},
            {"service": "inventory", "summary": "b", "offset_seconds": -20},
        ],
        "faulted_config": {"pool_capacity": 9},
        "seed_variants": {
            "evaluation": {
                "change_offsets": [-30, -40],
                "faulted_config_overrides": {"pool_capacity": 7},
            }
        },
    }

    result = apply_seed_variant(definition, "evaluation")

    assert [entry["offset_seconds"] for entry in result["changes"]] == [-30, -40]
    assert result["changes"][0]["service"] == "orders"
    assert result["faulted_config"] == {"pool_capacity": 7}


def test_apply_seed_variant_requires_matching_offset_and_change_counts() -> None:
    definition = {
        "changes": [{"service": "orders", "summary": "a", "offset_seconds": -10}],
        "seed_variants": {"development": {"change_offsets": [-10, -20]}},
    }

    with pytest.raises(ValueError):
        apply_seed_variant(definition, "development")


def test_write_json_never_exposes_a_torn_read_under_concurrent_reads(
    tmp_path: Path,
) -> None:
    """Real threads, no mocked timing: a
    concurrent reader must never observe a torn `write_json` -- neither an
    empty file, invalid JSON, nor a value that is not exactly one of the two
    payloads being written. Measured against the pre-fix `write_text`-based
    implementation, this fails reliably (thousands of errors); against the
    fixed atomic-replace implementation, it passes with zero.
    """
    target = tmp_path / "lab" / "config.json"
    # 5000 chars: large enough to keep `write_text` from completing in a
    # single syscall on a typical filesystem, so a reader has a real chance
    # to observe a torn write.
    payload_a = {"require_order_token": True, "padding": "a" * 5000}
    payload_b = {"require_order_token": False, "padding": "b" * 5000}
    stop = threading.Event()
    errors: list[Exception] = []

    def reader() -> None:
        while not stop.is_set():
            try:
                text = target.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                errors.append(error)
                continue
            if value not in (payload_a, payload_b):
                errors.append(RuntimeError(f"corrupted value: {value!r}"))

    write_json(target, payload_a)
    # 4 reader threads, 500 write-pairs: enough concurrent readers and
    # iterations to reliably hit the race, not tuned precisely -- if this
    # ever flakes, increase iterations first.
    readers = [threading.Thread(target=reader) for _ in range(4)]
    for thread in readers:
        thread.start()
    for _ in range(500):
        write_json(target, payload_a)
        write_json(target, payload_b)
    stop.set()
    for thread in readers:
        thread.join()

    assert errors == []
