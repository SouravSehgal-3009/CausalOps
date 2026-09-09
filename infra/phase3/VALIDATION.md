# Phase 3 restricted-environment validation record — initial local slice

Dedicated VM (`causalops-test`, asia-south1-c, project
`project-7b68a103-d482-4418-a35`), same VM as `infra/phase2/VALIDATION.md`.
Run 2026-09-09, on top of Phase 2's already-deployed control plane and lab.

This covers exactly `RESTRICTED_HANDOFF.md`'s "initial local slice" — it does
**not** start the MCP stdio server, the application worker, or MCP-backed
dispatch. Those stay deferred per that doc's own "Deferred Worker and MCP
Composition" section: `mcp_stdio.py` is, by its own docstring, "no
subprocess, socket, backend, or model composition" — a pure, unwired
protocol state machine, not a launchable server, until a separately reviewed
transport exists and `POLICY_APPROVAL.md` records a real approval.
`_APPROVED_MCP_DISPATCH` remains `None` in `mcp_policy_adapter.py` (fail
closed, verified by reading the source, not a live probe).

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

## MCP manifest identity (no live server — see scope note above)

Computed directly from `causalops.mcp_manifest.pinned_observability_manifest()`
on the VM, matching `RESTRICTED_HANDOFF.md`'s "MCP Manifest Gate" pin:

- protocol revision: `2025-11-25`
- manifest sha256: `7886e3705cee64ca76e2a79cb72a44bfe0ac703a01465f8f483461c3b164074a`
- exactly 5 tools, all read-only observability: `query_metric`, `query_logs`,
  `list_recent_changes`, `get_topology`, `search_runbooks`.

The stdio-server checks in `RESTRICTED_HANDOFF.md`'s "Validation and
Evidence" section (stdout/stderr separation, cross-incident refusal,
malformed-argument refusal) require an actual server subprocess, which does
not exist in this codebase yet. The equivalent logic is exercised hermetically
today by `tests/unit/test_mcp_stdio.py`, `tests/unit/test_mcp_manifest.py`,
and `tests/unit/test_mcp_policy_adapter.py` (all passing — see PR #1 CI).
Live reproduction against a real stdio child is a follow-up once that
transport is built and reviewed, not part of this slice.

## Sanitized command evidence

- `docker ps` (Ollama container only): `causalops-phase3-ollama-1`, image
  `32931b46719f`, `Up`, port `127.0.0.1:11434->11434/tcp`.
- `curl http://127.0.0.1:11434/api/version` → `{"version":"0.33.3"}`.
- `docker exec causalops-phase3-ollama-1 ollama list` →
  `qwen3.5:4b  2a654d98e6fb  3.4 GB`.
- No credentials, tokens, owner emails, prompts, report contents, or raw
  telemetry are recorded above.
