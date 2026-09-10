# Phase 3 restricted-environment validation record

Dedicated GCE VM (same VM referenced throughout this project's other
validation records, project details omitted here).
Two entries: the initial local slice (2026-09-09, on top of Phase 2's
already-deployed control plane and lab), and the real MCP transport built
and live-validated later the same day (below).

`_APPROVED_MCP_DISPATCH` remains `None` in `mcp_policy_adapter.py` throughout
both entries (fail closed) — confirmed by reading the source and, in the
second entry, by a live run against the deployed server, not just a static
check.

## VM Deployment (steps 1–5)

1. `uv sync --locked` — 69 packages, no drift (no new Phase 3 dependency was
   introduced beyond the stdlib + existing pydantic).
2. `.env.vm` (untracked, gitignored) extended with `CAUSALOPS_EXECUTION_ENV=vm`.
   `CAUSALOPS_CANDIDATE_EVALUATION` deliberately left unset — no approved
   candidate run has happened. Existing `results/` storage from Phase 2
   reused (same `CAUSALOPS_PROJECT_ROOT`, same persistent VM disk).
3. `infra/phase3/compose.env` (untracked, gitignored) created with one
   reviewed digest:
   `CAUSALOPS_OLLAMA_IMAGE=ollama/ollama@sha256:32931b46719f673c05fdbaa81ccb26da18ea4a1c57590a754874ab28ba269eb2`
   — resolved from Docker Hub's registry API (read-only manifest-list lookup
   against the `ollama/ollama:latest` tag, no pull performed before review)
   and explicitly approved by the owner before use. Started via
   `infra/phase3/run_compose.sh --env-file infra/phase3/compose.env up -d`.
   Container `causalops-phase3-ollama-1`, image `32931b46719f`, bound to
   `127.0.0.1:11434` only (confirmed via `docker ps` — no other port
   mapping). Ollama server version `0.33.3` (`GET /api/version`).
4. Synthetic lab confirmed running as its own separate Compose project
   (`causalops-lab-{gateway,orders,inventory,prometheus}-1`, 10h+ uptime at
   check time) — not a service inside `infra/phase3/docker-compose.yml`,
   untouched by this step.
5. `docker exec causalops-phase3-ollama-1 ollama pull qwen3.5:4b` — success.
   Recorded manifest (`ollama show`/manifest file on the container):
   - architecture `qwen35`, 4.7B parameters, Q4_K_M quantization, context
     length 262144, requires ollama `0.17.1`.
   - model config digest: `sha256:de9fed2251b37295b763727a59ca35cf5cfe5c7379bc3e2104b2ce3c145aa887`
   - weights layer digest: `sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`
   - license layer digest: `sha256:7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2`

## MCP manifest identity

Computed directly from `causalops.mcp_manifest.pinned_observability_manifest()`
on the VM, matching `RESTRICTED_HANDOFF.md`'s "MCP Manifest Gate" pin:

- protocol revision: `2025-11-25`
- manifest sha256: `7886e3705cee64ca76e2a79cb72a44bfe0ac703a01465f8f483461c3b164074a`
- exactly 5 tools, all read-only observability: `query_metric`, `query_logs`,
  `list_recent_changes`, `get_topology`, `search_runbooks`.

## Real local-stdio MCP transport — built, tested, and live-validated

A prior version of this record said the transport "does not exist yet" and
deferred the stdio-server checks. That is superseded: `mcp_server_main.py`
(the child entry point), `mcp_child_process.py` (the client-side subprocess
supervisor — spawn, crash/timeout/respawn, all genuinely new territory in
this codebase), and `mcp_client_registry.py` (client wiring, reusing the
existing `dispatch_registry` factory so policy enforcement is inherited,
not reinvented) now exist, are unit-tested against **real spawned OS
subprocesses** (`tests/unit/test_mcp_child_process.py`,
`tests/unit/test_mcp_policy_equivalence.py`, 8 tests, all real processes,
no mocked transport), and were exercised live on this VM below.

**Deployment shape, decided deliberately:** no new Dockerfile or Compose
service. There has never been a Dockerfile for the `causalops` application
itself anywhere in this repo (only `lab/services/Dockerfile`, for the
synthetic lab) — every deployment this project has ever done, Phase 2
through this entry, launches the hosted control plane as a direct host
process (`uv run uvicorn causalops.api_runtime:app --factory ...`).
Containerizing the app for the first time is separate, real scope,
orthogonal to proving MCP dispatch works — deferred as its own future
initiative if wanted, not bundled into this one. MCP-enabled launch is the
same command already used throughout this project, plus one env var:

```
CAUSALOPS_EXECUTION_ENV=vm CAUSALOPS_MCP_DISPATCH=true \
  uv run uvicorn causalops.api_runtime:app --factory --workers 1 \
  --host 127.0.0.1 --port 8000
```

`CAUSALOPS_MCP_DISPATCH=true` selects `McpBackedReplayRuntimeWiring` in
`api_runtime.py`'s `app()` factory instead of `HostedReplayRuntimeWiring`
— belt-and-suspenders: `mcp_policy_adapter._APPROVED_MCP_DISPATCH` being
`None` still makes every dispatch attempt refuse safely regardless of this
flag, confirmed live below.

**Live VM run:** started the server as above, inserted one real
`configuration_change`/`development` investigation directly into the
running server's own control-plane DB (bypassing the OAuth dashboard flow
for this check only — the request/response HTTP path itself was already
proven separately in `infra/phase2/VALIDATION.md`), and let the server's
own background worker thread pick it up and run it for real:

- Two real tool proposals from the replay fixture (`query_logs`,
  `list_recent_changes`) each spawned/reused a real `mcp_server_main` child
  process, completed the full stdio handshake, and came back
  `policy_result=ALLOWED, outcome=ERROR` — exactly the honest "policy
  agreed, dispatch safely refused" result the unit tests predict, now
  reproduced against the actual deployed server, not a test double.
- The investigation did not crash or hang on either refusal — it continued
  normally and reached `DIAGNOSED` (per the scripted replay fixture),
  total latency 48ms.
- `ss -tlnp` before and after showed no new listening port at any point —
  only the hosted API's own `127.0.0.1:8000`. The transport is stdio pipes
  only, by construction, not by absence-of-testing.
- `ps aux | grep mcp_server_main` after completion: no output — no leaked
  child process.
- Investigation artifacts and the inserted test scenario were reset/deleted
  after the check; the server was stopped afterward (this was a one-off
  validation run, not a standing deployment).

**Still not done, and out of scope for an agent:** the actual reviewed
`_APPROVED_MCP_DISPATCH` approval record. Confirmed still `None` in
`mcp_policy_adapter.py` — a named human reviewer must independently verify
all 5 `POLICY_APPROVAL.md` conditions against this evidence (condition 2,
real EXECUTED-outcome equivalence against direct backends, specifically
still needs that approval to exist before it can be proven — see
`test_mcp_policy_equivalence.py`'s own last test, which pins today's
honest refusal) and author that record as its own separate, reviewed
commit.

## Sanitized command evidence

- `docker ps` (Ollama container only): `causalops-phase3-ollama-1`, image
  `32931b46719f`, `Up`, port `127.0.0.1:11434->11434/tcp`.
- `curl http://127.0.0.1:11434/api/version` → `{"version":"0.33.3"}`.
- `docker exec causalops-phase3-ollama-1 ollama list` →
  `qwen3.5:4b  2a654d98e6fb  3.4 GB`.
- No credentials, tokens, owner emails, prompts, report contents, or raw
  telemetry are recorded above.
