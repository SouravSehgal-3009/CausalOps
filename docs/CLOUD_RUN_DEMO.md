# Using the hosted (Cloud Run) deployment

This is the walkthrough for the **hosted** surface — a real browser-facing
sign-in flow, backed by GCP Firestore and (optionally) Cloud Run — once
it's deployed. To deploy it yourself, follow
[`infra/DEPLOYMENT.md`](../infra/DEPLOYMENT.md) first; this page picks up
from "you have a URL and can sign in."

There's no live URL linked here on purpose: this project's own hosted
deployment is demo infrastructure, torn down between sessions to avoid
ongoing cost, so a link here would eventually 404. A recorded walkthrough
exists instead — see the root [`README.md`](../README.md) for the video.
A reader who deploys their own copy gets their own URL from `terraform
output cloud_run_api_url`.

## What's different about this surface

The hosted API is **replay-only** — no live model, incident, or tool
selection is ever exposed here, by design (a spec-driven safety boundary,
not a missing feature). Every investigation replays a real, fixed fixture
against the real synthetic lab; the point of this surface is to demonstrate
the owner-facing product experience (sign-in, create, poll, approve/reject,
retrieve a report) without exposing a billed live-model surface to the
public internet.

Sign-in is a real Google ID token verified server-side against an explicit
per-owner allowlist — not "anyone with the link." An account outside the
allowlist is refused outright; that refusal is itself worth demonstrating,
since it's easy to mistake for a bug rather than the intended behavior.

## Walkthrough

1. **Sign in.** The dashboard shows "Sign in with an approved Google
   account." Talking point: this replay-only hosted API never exposes live
   model/incident/tool selection to a browser, by design.

2. **Create an investigation.** All 4 scenario families are real, distinct
   faults against the real synthetic lab (gateway/orders/inventory
   containers on the worker machine) — pick `ambiguous_telemetry` to show
   the pause/decide flow, any other family for a clean `DIAGNOSED` run.
   Each family has its own real fixture matched to its actual root cause
   (`live_setup.FAMILY_REPLAY_FIXTURES`).

3. **Poll status.** The investigation moves `QUEUED` → `RUNNING`. Talking
   point: the VM-hosted worker picked this up from Firestore — the exact
   same database the Cloud Run API just wrote to; two independently
   deployed processes agreeing through one durable backend, not a
   monolith.

4. **If `ambiguous_telemetry` — pauses at `PAUSED_APPROVAL`.** Talking
   point: the graph reaches a decision point and stops for a human —
   evidence-grounded, not autonomous-and-unaccountable. Look at the events
   timeline before deciding. `ambiguous_telemetry`'s real fault always
   produces two conflicting error signals, so this pause is genuine, not a
   staged demo trick. The other 3 families resolve straight to `COMPLETED`
   — that's correct, not a bug.

5. **Approve or reject.** Either is a fine demo path — approving resumes
   the graph to a finalized report; rejecting exercises the safe-refusal
   branch. Talking point: the resumed graph continues from the exact
   LangGraph checkpoint (Firestore-backed), not a re-run from scratch.

6. **`COMPLETED` — retrieve the report.** Talking point: the report content
   was written into Firestore once, by the worker's own `finalize()` — the
   Cloud Run API never touches the worker machine's local disk to serve it.

## If something looks wrong mid-walkthrough

- Worker-side status: `systemctl status causalops-api` (auto-restarts on
  crash, `enabled` so it survives a machine reboot too).
- Lab containers: `docker ps` — `gateway`/`orders`/`inventory`/
  `prometheus` should all show `healthy`. Restart the stack fresh
  (`docker compose -f lab/docker-compose.yml down && docker compose -f
  lab/docker-compose.yml up -d`) before an important walkthrough — the lab
  isn't designed for indefinite uptime across dozens of runs, and stale
  in-memory state can make fault injection stop reliably reproducing a
  failing request.
- Cloud Run logs (if deployed): `gcloud run services logs read
  causalops-api --region <region> --limit 50`.
