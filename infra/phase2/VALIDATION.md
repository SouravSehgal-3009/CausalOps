# Phase 2 restricted-environment validation record

Dedicated VM (`causalops-test`, `asia-south1-c`, project
`project-7b68a103-d482-4418-a35`), reviewed commit `8b466698f3172a423abaab8c70d4dd55c9c57459`
("Add replay control plane durability"). Run 2026-09-08/09.

No `ANTHROPIC_API_KEY`, Ollama, Pinecone, or other provider credential was
ever set on this VM. Owner emails, bearer tokens, and investigation report
contents are deliberately excluded below — owners are labeled A/B, not by
address.

## 1. Environment

- `uv sync --locked` — 69 packages resolved, no drift from `uv.lock`.
- `ENABLE_CLAUDE=false`, `CAUSALOPS_PROJECT_ROOT=/home/acer/CausalOps`,
  `CAUSALOPS_ALLOWED_OWNERS` (3 allowlisted addresses),
  `CAUSALOPS_GOOGLE_CLIENT_ID` set via untracked `.env.vm`
  (gitignored, not this file).
- `results/control-plane.db`, `results/checkpoints.db`,
  `results/investigations/`, `results/deliveries/` live on the VM's boot
  disk (`/dev/root`, persistent, not tmpfs).

## 2. Phase 2 cloud resources (`infra/phase2`)

- `terraform init` — provider `hashicorp/google` v6.50.0, terraform 1.16.1.
- `terraform plan` — 5 to add, 0 change, 0 destroy.
- `terraform apply` — first attempt failed (`ACCESS_TOKEN_SCOPE_INSUFFICIENT`
  — the VM's attached compute service account only carries
  `devstorage.read_only` / `logging.write` / `monitoring.write` /
  `service.management.readonly` / `servicecontrol` / `trace.append`, no
  `cloud-platform`). Fixed by `gcloud auth application-default login` under
  a separate user account with project IAM permissions — the VM's own
  attached service account was never modified. Second `terraform apply`:
  **Apply complete! Resources: 5 added, 0 changed, 0 destroyed.**
  - `artifact_bucket_name = "causalops-replay-artifacts-7b68a103"`
  - `control_plane_service_account_email = "causalops-replay-control@project-7b68a103-d482-4418-a35.iam.gserviceaccount.com"`
- Drift check: `terraform plan` re-run post-apply — "No changes. Your
  infrastructure matches the configuration." (confirms bucket has
  `uniform_bucket_level_access=true`, `public_access_prevention=enforced`,
  `versioning.enabled=true`, and both bucket IAM bindings carry the
  prefix-scoped `resource.name.startsWith('projects/_/buckets/.../objects/investigations/')`
  condition — verified directly from `main.tf`, `terraform apply`, and
  `terraform plan` output; the VM's narrow-scope `gcloud` CLI account could
  not itself call `storage.buckets.get`/`getIamPolicy`, so read-back went
  through terraform/ADC instead of raw `gcloud`/`gsutil`).
  - No `google_service_account_key` resource exists in `main.tf` — workload
    identity only, no key ever created.
  - Project IAM: `google_project_iam_member.control_plane_log_writer` is the
    only project-level grant (`roles/logging.logWriter`).

## 3. Control plane + lab deployment

- Lab: `docker compose -f lab/docker-compose.yml up -d` — gateway, orders,
  inventory (all `healthy`), prometheus.
- `uv run uvicorn causalops.api_runtime:app --factory --workers 1 --host
  127.0.0.1 --port 8000` — single application worker. `/`, `/docs`,
  `/openapi.json` all returned 200 on first boot.
- `app()`'s factory always constructs `HostedReplayRuntimeWiring` — no
  Claude/Ollama/Pinecone code path is reachable from this deployment
  regardless of `ENABLE_CLAUDE`; confirmed by reading
  `live_setup.py`'s `HostedReplayRuntimeWiring.build` (docstring: "Build
  only hosted replay dependencies; no other provider is reachable").
- `BackgroundControlPlaneWorkers.start()` fires on FastAPI startup
  (`api.py` `create_app`, `worker_service.start()`); both the replay
  worker and the report-delivery worker ran inside the same daemon
  thread, confirmed by every acceptance run below completing without a
  second process.

## 4. Owner and recovery acceptance (all live HTTP, real Google sign-in)

Real ID tokens obtained via a throwaway `google-auth-oauthlib` manual
authorization-code exchange (no browser reachable from the VM directly;
loopback-port `ssh -L` also failed under the operator's Windows client, so
the manual copy-paste redirect flow was used instead). Owner A signed in
2026-09-08T20:14:13Z, owner B 2026-09-08T20:31:54Z.

| Check | Result |
|---|---|
| Owner A: create `configuration_change`/`development` | `217c461a4a5140e7b501dfbd534ac090` — QUEUED → RUNNING → FINALIZED; events `investigation_queued, investigation_started, scenario_reserved, report_finalized, report_delivered`; report retrieved 200 |
| Same 3 other seeds (`evaluation`, `evaluation_b`, `evaluation_c`) | `081b6e14…`, `092261bf…`, `bbbe2589…` — all FINALIZED, same clean event chain |
| Decision on an already-FINALIZED investigation | `POST .../decision` on `217c461a…` → **409 Conflict** |
| Owner B: cross-owner access to owner A's investigation | `GET status` / `GET events` / `GET report` on `217c461a…` with owner B's token → **404** on all three |
| Owner B: create own investigation | `0113c735…` — QUEUED, accepted normally |
| Crash-and-recover at RUNNING | `04d73a38585a4c06846f1a3666cd2739` — server hard-killed (`kill -9`) immediately after QUEUED; job left `RUNNING` with a live 300s claim lease. Restarted server; job correctly held (lease not yet expired) until lease timeout, then `investigation_lease_expired` (attempt 1) → re-ran from `investigation_started` → single `report_finalized`/`report_delivered`. Exactly one delivery artifact on disk, no duplicates. Prior FINALIZED investigations' event counts and delivery-file mtimes were unchanged by the restart. |
| Failed replay job → backoff → dead-letter | `cfc13588ba41418c9a48a24c1e82342a` — lab containers stopped, investigation created. `investigation_retry_scheduled` at attempt 1 (~5s) and attempt 2 (~11s), then `investigation_failed` at attempt 3 — owner-facing status **FAILED** (`status()`/`scenario_status()` both key off `failure_reason IS NOT NULL`, not the raw `QUEUED` row status). |
| Failure does not block later work | Lab containers restarted; `8db21331a5c543e0aec109d710108e86` created immediately after — FINALIZED normally with the standard 5-event chain |

### Accepted Phase 2 exception — pause/decision path not exercised live

`PAUSED` / owner accept-reject / idempotent resubmission / API-CLI conflict
handling on a live checkpoint, and restart recovery at the
pending-interrupt / decision-recorded boundaries specifically, were **not**
reproduced live against the deployed control plane. This is recorded here
as a formal, accepted Phase 2 exception, not an open action item.

**Root cause** (pre-existing, not introduced by this deployment): the
hosted `configuration_change` replay fixture used by
`HostedReplayRuntimeWiring`/`ReplayGraphJobRunner` is scripted to always
conclude a clean `DIAGNOSED`/`CONFIG_CHANGE` result for all four available
seeds — `graph.py`'s `_escalation_reason` never fires
(`TOOL_UNAVAILABLE`, `RETRIEVAL_COVERAGE_INSUFFICIENT`,
`CONFLICTING_EVIDENCE`, `INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING` are
all unreachable with today's fixture data). This matches the carried
limitation already on record: `cli.py`'s replay path "always uses one
hardcoded fixture scripted to conclude CONFIG_CHANGE regardless of
family."

**Two ways to close this were weighed:**

1. Add a new replay fixture, scripted to deterministically hit one of the
   four `EscalationReason` triggers, so the live pause/decision path
   becomes reachable through the hosted API today.
2. Record the gap as an accepted exception and rely on the existing
   hermetic coverage instead.

**Decision: (2), record as an accepted exception.** A deterministically
escalating fixture is real feature work — a new scripted incident,
threaded through `HostedReplayRuntimeWiring`, chosen and shaped to trip
exactly one trigger without disturbing the four existing seeds' scored
behavior — not a deployment-validation task, and it duplicates coverage
this codebase already carries at the hermetic layer. `decide()`'s
conflict/idempotency logic, and the paused-worker-outcome path, are
already covered by `tests/unit/test_api_runtime.py` (`FakeRunner`-driven,
e.g. `test_worker_uses_durable_checkpoint_and_finalizes_once`); what's
missing is only the *live, real-fixture* reproduction, not the underlying
logic. Building a real escalating fixture is better scoped alongside
Phase 3, when a real model exists and escalation triggers stop being
purely synthetic scripting exercises (see `causalops-carried-limitations`
memory: "becomes provable once Phase 3 Step 1 wires in the real Claude
model"). Revisit this exception at that point, not before.

## 5. Notes for reviewers

- Bucket name, service-account email, and terraform resource identifiers
  above are real and safe to keep (no secret material).
- `infra/phase2/terraform.tfvars` is untracked (gitignored) — a reviewer
  needs their own `project_id`/`region`/`artifact_bucket_name` to
  reproduce; this file's captured outputs are the record of what was
  actually applied.
- `.env.vm` (gitignored) holds the real owner emails and Google client ID
  used for this run; not reproduced here.
