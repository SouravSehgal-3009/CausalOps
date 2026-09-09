# Phase A validation record — contract-correctness fixes

Dedicated VM (`causalops-test`), commits `2065881` ("Wire causalops-evaluate
onto MCP transport, drop search_runbooks from MCP, default MCP on the VM")
and `f33d312` ("Fix POST /investigations contract, /api/v1 prefix, and
lifecycle state names to match spec"). Run 2026-09-10.

This is the first of the lettered phases (A → B → C → D) closing the gap
between the actual repository and the original, uncompressed CausalOps
Agentic AI v1 Specification's §3-§5 architecture, found by a fork-agent audit
this session. Owner emails and investigation report contents are
deliberately excluded below.

## 1. Scope

Six items, all local code changes, no new GCP spend:

1. `search_runbooks` removed from the MCP-approved tool set (spec §5:
   advisory guidance, not an incident observation, stays a direct retrieval
   backend, never MCP-callable).
2. `POST /investigations` contract fixed: no evaluator-seed field accepted
   (server always uses `ReplaySeed.DEVELOPMENT`), a required
   `Idempotency-Key` header with real dedupe-and-replay semantics.
3. Route prefix `/v1` → `/api/v1`.
4. Lifecycle state renames: `PAUSED`→`PAUSED_APPROVAL`,
   `FINALIZED`→`COMPLETED`, `FAILED`→`FAILED_SAFE`.
5. MCP flipped to the default hosted-worker dispatch path on the VM
   (`CAUSALOPS_MCP_DISPATCH` is now an opt-OUT escape hatch back to direct
   dispatch, not an opt-in) — Phase 3's exit criterion, "replay runs through
   worker/MCP," read literally.
6. Ollama/Qwen: explicitly out of scope, no changes (owner: "ollama can be
   closed as it is failing" — Phase 4's negative finding stands closed).

## 2. Pre-live verification

- `uv run pytest -q` — 914 passed (up from 913 pre-Phase-A; one net new
  regression test, `test_repeated_idempotency_key_replays_the_same_
  investigation`).
- `uv run ruff format --check . && uv run ruff check . && uv run mypy src
  lab` — clean throughout.

## 3. Live acceptance pass

Real HTTP over `TestClient`, real `SqliteReplayControlPlane`, real
`McpBackedReplayRuntimeWiring` (the new default), real lab containers
(`gateway`/`orders`/`inventory`, all healthy), real spawned MCP child
processes — not mocked. Identity verification used a substituted
`IdentityVerifier` rather than a real Google sign-in: Phase A does not touch
`GoogleIdentityVerifier` at all, and Phase 2's own prior live run already
proved real Google sign-in on this exact deployment shape; owner-approved
scope decision for this phase's acceptance pass, not a gap.

Isolated control-plane/checkpoint databases; `artifacts_root` pointed at the
real `results/investigations/` directory to match where
`ReplayGraphJobRunner` actually writes (mirroring exactly what `app()`'s own
factory does — this alignment matters, see the debugging note below).

| Check | Result |
|---|---|
| Old `/v1` prefix | `GET /v1/investigations/...` → **404** (route genuinely gone, not just undocumented) |
| Missing `Idempotency-Key` on create | **400** |
| Owner A: create `configuration_change` (no seed field sent) | 200, `QUEUED` |
| Repeated `Idempotency-Key`, same owner | Same `investigation_id` returned, no duplicate row created |
| MCP child spawned for owner A's investigation | `[mcp child] MCP observability server ready` on stderr, confirmed via real subprocess, not asserted from logs alone — traced `start_scenario`/`reset_scenario` calls line up 1:1 with the investigation's own lifecycle |
| Owner A investigation reaches terminal state | `COMPLETED` (not `FAILED_SAFE`) once lab/artifact-path setup was correct — see debugging note |
| Event chain | `investigation_queued, investigation_started, scenario_reserved, report_finalized, report_delivered` — same shape Phase 2 recorded, new state vocabulary |
| Report retrieval | `GET .../report` → 200, non-empty |
| Cross-owner isolation under new prefix | Owner B `GET` on owner A's investigation → **404** |
| Owner B: create + complete own investigation | 200 → `COMPLETED`, independent MCP child spawned |

### Debugging note — two real setup bugs found and fixed, neither a Phase A code defect

Both were in this throwaway acceptance script, not in `src/causalops/`:

1. **Leftover active-scenario marker across script runs.** The lab enforces
   exactly one active scenario system-wide
   (`scenario_control._claim_scenario`, a single marker file keyed to the
   project root, not per-investigation). An earlier iteration of this
   acceptance script exited (via `TestClient`'s context-manager `__exit__`)
   right after creating owner B's investigation, without waiting for it to
   reach a terminal state — leaving that investigation's scenario claimed.
   The next run's owner A investigation then failed 3 times with
   `LabError: reset scenario<X> before starting another scenario`, X being
   the STALE investigation from the prior run, not the current one — traced
   definitively by wrapping `start_scenario`/`reset_scenario` with
   thread-tagged, timestamped logging and cross-referencing against
   `replay_jobs` rows directly. Fixed in the script two ways: clear any
   stale marker at start, and wait for owner B's own investigation to reach
   a terminal state before the script exits.
2. **`artifacts_root`/runner-root mismatch.** The script initially isolated
   `SqliteReplayControlPlane`'s `artifacts_root` under a dedicated
   acceptance directory while leaving `ReplayGraphJobRunner`'s own root
   (which fixes where `finalize_investigation` actually writes reports, and
   where scenario/lab files are looked up) at the real project root — an
   inconsistency `app()`'s own factory never has, since it always derives
   both from the same `root`. This surfaced as `finalize()` raising
   `FileNotFoundError`/`ValueError("report_artifact is not readable
   UTF-8")` on every attempt. Fixed by pointing `artifacts_root` at the real
   `results/investigations/` directory, matching the runner's own root,
   while keeping only the control-plane/checkpoint databases isolated.

Both bugs were caught and root-caused with real evidence (traced call logs,
direct `replay_jobs` row inspection, full tracebacks via targeted
instrumentation) before concluding the underlying MCP-wired replay path
itself was correct — the standalone single-shot reproduction that first
proved the MCP path works end-to-end predates both fixes and was not
retroactively rationalized.

## 4. Not covered by this pass

Full crash-recovery (`kill -9` mid-`RUNNING`, dead-letter after repeated lab
failure) was not re-exercised live here — that mechanism is unchanged by
Phase A (`SqliteReplayControlPlane`'s claim/retry/lease logic was not
touched; only routes, the create contract, state names, and the MCP-default
selection changed) and is already covered by Phase 2's own validation
record. Re-running it was judged disproportionate to what Phase A actually
changed; flagged here explicitly rather than silently skipped.

## 5. Outcome

All 6 Phase A items verified live on the VM against the real MCP transport,
real lab, and real (substituted-identity) HTTP surface. Ready for its own
PR into `feat/agentic-ai-v1` (not `master` — per the owner's standing rule,
`master` stays untouched until the whole project goal is achieved).
