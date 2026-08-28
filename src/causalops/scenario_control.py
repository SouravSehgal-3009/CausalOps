"""The controller-only trust domain: bring the lab up, fault it, verify it, reset it.

Nothing here is a registered tool and nothing here takes model input. It writes the
run directory the investigator later reads, and keeps the expected outcome in a
directory the investigator side has no accessor for.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import JsonValue

from causalops.domain import (
    EvidenceKind,
    GatewaySymptom,
    IncidentScope,
    InitialAlertPacket,
    StoredIncident,
)
from causalops.evidence import build_evidence, new_opaque_id
from causalops.prometheus import DEFAULT_PROMETHEUS_URL
from causalops.telemetry import RunPaths

GATEWAY_URL = "http://127.0.0.1:8080/api/orders"
HEALTH_URLS = {
    "gateway": "http://127.0.0.1:8080/healthz",
    "orders": "http://127.0.0.1:8081/healthz",
    "inventory": "http://127.0.0.1:8082/healthz",
    "prometheus": f"{DEFAULT_PROMETHEUS_URL}/-/ready",
}

# The recorded window opens before the first request so that a change made shortly
# beforehand falls inside it, which is what makes `list_recent_changes` useful.
WINDOW_LEAD_IN = timedelta(minutes=5)
REQUEST_TIMEOUT_SECONDS = 5
# Traffic is paced so Prometheus records more than one scrape of it.
REQUEST_PAUSE_SECONDS = 0.2
# `lab/prometheus.yml`'s `scrape_interval: 5s` --
# two full scrape intervals, plus margin, before `window_end` is stamped, so
# at least one scrape has observed the post-fault state and a `rate()` query
# has a real denominator (`rate()` needs two samples inside its lookback).
# Without this wait, `window_end = clock()` could land before Prometheus
# ever scraped the fault, making every metric query over the window
# structurally unable to see it -- not a query bug, a timing one.
SCRAPE_SETTLE_SECONDS = 12
ALERT_SOURCE_VERSION = "alert-1"


def _real_sleep(seconds: float) -> None:
    """`start_scenario`'s default `sleeper`, kept as a named module-level
    function rather than binding `time.sleep` directly as the default
    value. A default value is evaluated once, at import time -- binding
    `time.sleep` itself there would capture that one function object
    permanently, immune to a test later monkeypatching `time.sleep` (the
    same reason `clock`'s own default is a lambda, not a bound method).
    This function's own body looks up `time.sleep` fresh on every call, the
    same pattern `test_graph_frozen_reports.py` already relies on for
    `evidence_module.time.monotonic` -- so a test can monkeypatch the
    shared `time` module's `sleep` attribute directly, with no change
    needed to any of this module's existing `start_scenario(...)` call
    sites."""
    time.sleep(seconds)


class LabReasonCode(StrEnum):
    """Stable codes for lab and scenario commands, printed rather than recorded."""

    DOCKER_UNAVAILABLE = "DOCKER_UNAVAILABLE"
    LAB_NOT_HEALTHY = "LAB_NOT_HEALTHY"
    SCENARIO_ALREADY_ACTIVE = "SCENARIO_ALREADY_ACTIVE"
    UNKNOWN_FAMILY = "UNKNOWN_FAMILY"
    # A declared `config_key` absent from `faulted_config` -- see
    # `_resolved_change_summary`'s docstring for why this raises rather
    # than falls back.
    UNKNOWN_CONFIG_KEY = "UNKNOWN_CONFIG_KEY"
    INCIDENT_NOT_FOUND = "INCIDENT_NOT_FOUND"
    BASELINE_NOT_HEALTHY = "BASELINE_NOT_HEALTHY"
    FAULT_NOT_OBSERVED = "FAULT_NOT_OBSERVED"
    # A stored JSON artifact (`incident.json`,
    # `report.json`) exists but could not be turned back into the typed
    # record it is supposed to hold -- unreadable (permissions, a file
    # that vanished after an earlier existence check), not valid UTF-8, or
    # JSON that does not satisfy the model's own schema. Distinct from
    # `INCIDENT_NOT_FOUND`: that code means "there is no such artifact,"
    # this one means "there is one, and it is corrupt" -- an owner reading
    # the reason code needs to tell "wrong id" apart from "something on
    # disk is actually broken" (a crashed write, a hand-edited byte, disk
    # corruption), since only the second is worth investigating the
    # filesystem over.
    CORRUPT_ARTIFACT = "CORRUPT_ARTIFACT"


class LabError(Exception):
    """A lab or scenario command refused, with one stable reason code."""

    def __init__(self, reason_code: LabReasonCode, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def runs_root(root: Path) -> Path:
    return root / "runs"


def run_paths(root: Path, incident_id: str) -> RunPaths:
    return RunPaths(root=runs_root(root) / incident_id)


def validated_run_paths(root: Path, incident_id: str) -> RunPaths:
    """Builds `incident_id`'s run directory path
    only after confirming it cannot escape `runs_root(root)` -- the check
    `reset_scenario` below already had, extracted so a second caller
    (`cli.py`'s `run_investigate_command`) does not hand-copy the same two
    checks a second time. `run_paths` above stays unvalidated on purpose: it
    is called only with `new_opaque_id()`'s own output inside
    `start_scenario`, never with an external string, so it has nothing to
    validate against.

    `isalnum()` refuses anything containing `/`, `.`, `..`, or a null byte
    outright -- a real incident id (always `new_opaque_id()`'s output)
    never fails this. The parent-resolves-to-`runs_root` check is defense
    in depth against a platform-specific path oddity `isalnum()` alone
    might miss, not a second, independent proof technique -- both guard the
    same property.

    This project is a single-operator local CLI, not a networked service,
    so a path-traversal argument here has no separate "attacker" from
    "victim" the way `reset_scenario`'s destructive `rmtree` does -- it is
    recorded here as a defensive check rather than a security boundary,
    since this project has no separate party to defend against. The
    identity-mismatch check this function's callers add afterward
    (comparing the loaded `StoredIncident.scope.incident_id` back against
    `incident_id`) is worth having regardless of the security framing: it
    is what catches `runs/<id>/incident.json` ever diverging from its own
    directory name, a correctness bug this check can catch even though a
    traversal attempt from this project's own user cannot really "attack"
    anyone but themselves.
    """
    if not incident_id.isalnum():
        raise LabError(LabReasonCode.INCIDENT_NOT_FOUND, "that is not an incident ID")
    paths = run_paths(root, incident_id)
    try:
        root_path = runs_root(root).resolve()
        parent_path = paths.root.resolve().parent
    except RuntimeError as error:
        raise LabError(
            LabReasonCode.INCIDENT_NOT_FOUND, "that path is not inside the run tree"
        ) from error
    if root_path != parent_path:
        raise LabError(
            LabReasonCode.INCIDENT_NOT_FOUND, "that path is not inside the run tree"
        )
    return paths


def active_incident_file(root: Path) -> Path:
    return runs_root(root) / "active-incident.txt"


def compose_file(root: Path) -> Path:
    return root / "lab" / "docker-compose.yml"


def write_json(path: Path, value: object) -> None:
    """Writes `value` to `path` in one atomic replace, never a
    truncate-then-write in place.

    `path.write_text` truncates before writing a
    byte of the new content -- a concurrent reader (every lab service reads
    this same file on every request, each from its own thread) can observe a
    torn file: empty, or old bytes followed by new bytes, neither valid JSON.
    Reproduced directly by a real concurrency test. Mirrors
    `run_records.py`'s `write_jsonl`: build the complete content in a sibling
    temporary file first, then atomically rename it onto `path`
    (`Path.replace`, atomic on POSIX). A reader always sees either the
    complete old file or the complete new one, never a mixture.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def compose(root: Path, arguments: list[str], timeout: int) -> None:
    """Run one docker compose command, or say plainly that Docker is not there."""
    command = ["docker", "compose", "-f", str(compose_file(root)), *arguments]
    try:
        finished = subprocess.run(
            command, capture_output=True, timeout=timeout, text=True
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise LabError(
            LabReasonCode.DOCKER_UNAVAILABLE, f"docker compose did not run: {error}"
        ) from error
    if finished.returncode != 0:
        raise LabError(
            LabReasonCode.DOCKER_UNAVAILABLE,
            f"docker compose {' '.join(arguments)} failed: {finished.stderr.strip()}",
        )


def responds(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError):
        return False


def wait_for_health(deadline_seconds: int) -> None:
    """Poll every service and Prometheus until all answer or the wait runs out."""
    deadline = time.monotonic() + deadline_seconds
    waiting = dict(HEALTH_URLS)
    while waiting and time.monotonic() < deadline:
        for name, url in list(waiting.items()):
            if responds(url):
                waiting.pop(name)
        if waiting:
            time.sleep(2)
    if waiting:
        raise LabError(
            LabReasonCode.LAB_NOT_HEALTHY,
            f"these did not become healthy: {', '.join(sorted(waiting))}",
        )


def lab_up(root: Path, timeout_seconds: int = 600) -> None:
    runs_root(root).mkdir(parents=True, exist_ok=True)
    compose(root, ["up", "-d", "--build"], timeout_seconds)
    wait_for_health(120)


def lab_down(root: Path, timeout_seconds: int = 120) -> None:
    compose(root, ["down"], timeout_seconds)


def call_gateway() -> int:
    """The status the gateway returned, or 0 when it did not answer at all."""
    try:
        with urllib.request.urlopen(  # noqa: S310
            GATEWAY_URL, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)
    except (OSError, urllib.error.URLError):
        return 0


def drive_traffic(count: int) -> tuple[int, int]:
    """Send requests through the front door and count how they came back."""
    succeeded = 0
    failed = 0
    for _ in range(count):
        status = call_gateway()
        if status == 200:
            succeeded += 1
        else:
            failed += 1
        time.sleep(REQUEST_PAUSE_SECONDS)
    return succeeded, failed


def load_definition(root: Path, family: str) -> dict[str, Any]:
    path = root / "lab" / "scenarios" / f"{family}.json"
    if not path.is_file():
        raise LabError(LabReasonCode.UNKNOWN_FAMILY, f"no scenario named {family}")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def apply_seed_variant(definition: dict[str, Any], seed: str) -> dict[str, Any]:
    """Deterministically vary a definition's timing and fault magnitude by seed.

    A family without a matching `seed_variants` entry is returned unchanged, so
    this stays backward compatible with a definition that has none at all.
    """
    variants = definition.get("seed_variants")
    if not variants or seed not in variants:
        return definition
    variant = variants[seed]
    varied = dict(definition)
    if "change_offsets" in variant:
        varied["changes"] = [
            {**change, "offset_seconds": offset}
            for change, offset in zip(
                definition["changes"], variant["change_offsets"], strict=True
            )
        ]
    if "faulted_config_overrides" in variant:
        varied["faulted_config"] = {
            **definition["faulted_config"],
            **variant["faulted_config_overrides"],
        }
    return varied


@contextmanager
def _scenario_lock(root: Path) -> Iterator[None]:
    """Serialize ownership mutations across controller processes.

    The active marker is the durable state visible to operators; this sibling
    SQLite's cross-platform write transaction protects the
    check/recover/create and reset/cleanup transitions so a contender cannot
    act on a stale marker observation. It is intentionally a separate tiny
    lock database: this controller runs before an incident exists and must
    not depend on LangGraph's checkpoint schema.
    """
    lock_path = runs_root(root) / ".active-scenario-lock.db"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(lock_path, timeout=30)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _claim_scenario(root: Path, incident_id: str, paths: RunPaths) -> None:
    """Atomically claim the sole active-scenario marker for this run.

    The candidate directory exists before claiming so a stale nonempty marker
    can be recovered only when its referenced directory is absent. An empty
    marker is an in-progress exclusive claim and is never reclaimed.
    """
    marker = active_incident_file(root)
    if marker.is_file():
        active = marker.read_text(encoding="utf-8").strip()
        if not active or (runs_root(root) / active).is_dir():
            raise LabError(
                LabReasonCode.SCENARIO_ALREADY_ACTIVE,
                "reset "
                f"{active or 'the in-progress scenario'} before "
                "starting another scenario",
            )
        marker.unlink()
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        os.write(descriptor, incident_id.encode("utf-8"))
    except BaseException:
        marker.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)


def _resolved_change_summary(
    entry: dict[str, Any], faulted_config: dict[str, Any]
) -> str:
    """Render a config-carrying change entry's summary from the value the
    lab actually applied (`faulted_config`, after `apply_seed_variant`'s
    overrides).

    Two distinct cases diverge here, where an earlier version
    conflated them into one silent fallback:

    - An entry naming no `config_key` at all keeps its own static
      `summary` text unchanged -- true for image rebuilds and dependency
      bumps, which have no config value to resolve against. This is the
      legitimate, expected case.
    - An entry naming a `config_key` that is absent from `faulted_config`
      is almost certainly a scenario-authoring typo, not a legitimate
      no-config-value case -- falling back to static prose here would
      silently recreate the exact stale-summary defect this function
      exists to fix, just triggered by a misspelled key instead of a missing
      resolution step. This raises `LabError(UNKNOWN_CONFIG_KEY, ...)`
      instead, so a typo fails scenario startup loudly rather than
      shipping a quietly wrong summary.
    """
    config_key = entry.get("config_key")
    if config_key is None:
        return str(entry["summary"])
    if config_key not in faulted_config:
        raise LabError(
            LabReasonCode.UNKNOWN_CONFIG_KEY,
            f"{entry['service']}'s change entry declares "
            f"config_key={config_key!r}, absent from faulted_config -- "
            "fix the scenario definition",
        )
    return f"configuration update: {config_key} set to {faulted_config[config_key]}"


def write_manifests(
    paths: RunPaths, definition: dict[str, Any], window_end: datetime
) -> None:
    """Topology and recent changes, named and worded without the family in them."""
    write_json(paths.topology_file, definition["topology"])
    faulted_config = definition["faulted_config"]
    changes = [
        {
            "change_id": new_opaque_id()[:12],
            "at": (window_end + timedelta(seconds=entry["offset_seconds"])).isoformat(),
            "service": entry["service"],
            "summary": _resolved_change_summary(entry, faulted_config),
        }
        for entry in definition["changes"]
    ]
    write_json(paths.changes_file, changes)


# Kept together so the whole incident-packet assembly (scope, evidence, alert) is
# visible in one place rather than split across several tiny builders.
def build_incident(
    definition: dict[str, Any],
    incident_id: str,
    window_start: datetime,
    window_end: datetime,
    failed_requests: int,
    total_requests: int,
) -> StoredIncident:
    services = tuple(definition["services"])
    scope = IncidentScope(
        incident_id=incident_id,
        services=services,
        started_at=window_start,
        ended_at=window_end,
        endpoint=definition["endpoint"],
    )
    symptom_payload: dict[str, JsonValue] = {
        "endpoint": definition["endpoint"],
        "failed_requests": failed_requests,
        "total_requests": total_requests,
    }
    symptom = build_evidence(
        incident_id=incident_id,
        kind=EvidenceKind.SYMPTOM,
        source="alert",
        observed_at=window_end,
        summary=f"{failed_requests} of {total_requests} requests to "
        f"{definition['endpoint']} did not succeed",
        payload=symptom_payload,
    )
    topology = build_evidence(
        incident_id=incident_id,
        kind=EvidenceKind.TOPOLOGY,
        source="alert",
        observed_at=window_end,
        summary="gateway calls orders, orders calls inventory",
        payload=dict(definition["topology"]),
    )
    packet = InitialAlertPacket(
        incident_id=incident_id,
        window_start=window_start,
        window_end=window_end,
        endpoint=definition["endpoint"],
        symptom=GatewaySymptom(definition["symptom"]),
        services=services,
        alerted_at=window_end,
        alert_source_version=ALERT_SOURCE_VERSION,
        symptom_evidence_id=symptom.evidence_id,
        topology_evidence_id=topology.evidence_id,
    )
    return StoredIncident(scope=scope, packet=packet, evidence=(symptom, topology))


def clear_unstarted_scenario(root: Path, paths: RunPaths, incident_id: str) -> None:
    """Undo the early marker write and half-formed run directory.

    Called when a scenario fails before reaching a real fault, so it does not
    permanently block the next `scenario start` with SCENARIO_ALREADY_ACTIVE.
    """
    marker = active_incident_file(root)
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == incident_id:
        marker.unlink()
    shutil.rmtree(paths.root, ignore_errors=True)


# Kept together so the healthy-then-faulted sequence and its failure points stay
# in one readable order rather than being split across helper functions.
def start_scenario(
    root: Path,
    family: str,
    seed: str,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    *,
    sleeper: Callable[[float], None] = _real_sleep,
) -> str:
    """Create one incident: healthy baseline, then the fault, then the packet."""
    definition = apply_seed_variant(load_definition(root, family), seed)
    incident_id = new_opaque_id()
    paths = run_paths(root, incident_id)
    try:
        with _scenario_lock(root):
            paths.root.mkdir(parents=True, exist_ok=False)
            _claim_scenario(root, incident_id, paths)
        paths.logs.mkdir(parents=True, exist_ok=True)
        write_json(paths.root / "lab" / "config.json", definition["healthy_config"])
        window_start = clock() - WINDOW_LEAD_IN
        extra = 2 if seed == "evaluation" else 0
        healthy, unhealthy = drive_traffic(int(definition["baseline_requests"]) + extra)
        if healthy == 0 or unhealthy > 0:
            raise LabError(
                LabReasonCode.BASELINE_NOT_HEALTHY,
                f"{incident_id}: lab was not healthy before the fault: "
                f"{unhealthy} failed",
            )
        write_json(paths.root / "lab" / "config.json", definition["faulted_config"])
        served, failed = drive_traffic(int(definition["fault_requests"]) + extra)
        if failed == 0:
            raise LabError(
                LabReasonCode.FAULT_NOT_OBSERVED,
                f"{incident_id}: the fault produced no failing request",
            )
        # Let telemetry settle before stamping
        # `window_end` -- see `SCRAPE_SETTLE_SECONDS`'s own comment for why.
        sleeper(SCRAPE_SETTLE_SECONDS)
        window_end = clock()
        write_manifests(paths, definition, window_end)
        # `window_start` opens `WINDOW_LEAD_IN`
        # before the baseline phase even runs, so the alert's own window
        # covers baseline + fault, not fault alone -- `total_requests` must
        # count both. `healthy` is not folded into `failed_requests`: the
        # guard a few lines up (`if healthy == 0 or unhealthy > 0: raise`)
        # already proves `unhealthy` is 0 by this point, so every baseline
        # request counted here is a success, never a hidden failure.
        incident = build_incident(
            definition,
            incident_id,
            window_start,
            window_end,
            failed,
            healthy + served + failed,
        )
        paths.incident_file.write_text(
            incident.model_dump_json(indent=2), encoding="utf-8"
        )
        write_json(
            paths.root / "evaluator" / "expected.json",
            {"seed": seed, "family": family, **definition["expected"]},
        )
        return incident_id
    except BaseException:
        with _scenario_lock(root):
            clear_unstarted_scenario(root, paths, incident_id)
        raise


def reset_scenario(root: Path, incident_id: str) -> None:
    """Delete only this run's transient state. Finalized results are never touched."""
    with _scenario_lock(root):
        target = validated_run_paths(root, incident_id).root
        if not target.is_dir():
            raise LabError(
                LabReasonCode.INCIDENT_NOT_FOUND, f"no run directory for {incident_id}"
            )
        marker = active_incident_file(root)
        if (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == incident_id
        ):
            marker.unlink()
        shutil.rmtree(target)
