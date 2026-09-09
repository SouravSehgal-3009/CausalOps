# Phase 4 restricted-environment validation record — candidate evaluation attempt

Dedicated VM (`causalops-test`, asia-south1-c, project
`project-7b68a103-d482-4418-a35`), plus one temporary compute-optimized VM
(`causalops-phase4-temp`, same zone, `c2-standard-8`, deleted after use) spun
up specifically because `causalops-test`'s `e2-standard-4` proved too slow.
Run 2026-09-09, git `33a02ff58cfed2ca781f653acb964322d5d746e3`.

**No candidate evaluation run reached a scored, complete result.** This
record captures why, so the next attempt doesn't repeat the same
diagnostics. Nothing here is evidence of Qwen's diagnostic quality — every
attempt failed before producing a usable diagnosis.

## What was confirmed working

- `_assert_candidate_environment` correctly refuses without
  `CAUSALOPS_EXECUTION_ENV=vm`, `CAUSALOPS_CANDIDATE_EVALUATION=true`, and
  two well-formed `sha256:` digests.
- `causalops-qwen-evaluate` correctly writes `records.jsonl` incrementally
  and correctly stops the batch immediately on a `FAILED_SAFE` result — this
  behavior fired exactly as designed, just earlier than intended (see
  below).
- Manifest/digest identity used throughout:
  - Ollama image: `sha256:32931b46719f673c05fdbaa81ccb26da18ea4a1c57590a754874ab28ba269eb2`
  - Qwen model manifest: `sha256:2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
    (matches `ollama list`'s displayed short ID `2a654d98e6fb`)

## Finding 1 — CPU-only inference is far too slow for the configured budgets

- `causalops-test` (`e2-standard-4`, 4 shared vCPU, Xeon 2.20GHz, no GPU):
  a trivial single-word prompt took 29–41s end-to-end even with the model
  already warm and loaded (`ollama ps` confirmed resident). A real
  investigation's very first model call did not return within 400+ seconds
  in one attempt.
- `causalops-phase4-temp` (`c2-standard-8`, 8 dedicated vCPU,
  compute-optimized, ~2x the token throughput measured directly: ~8.9
  tok/s vs. an estimated ~4.5 tok/s): still took 442 seconds
  (`wall_clock_ms: 442495`) for a single model call — well past the
  investigation's own `wall_clock_seconds: 360` budget.
- Root cause: `qwen3.5:4b` is a "thinking" model — Ollama's own response
  showed 179 hidden reasoning tokens for a prompt whose entire visible
  answer was "OK". A real investigation's system prompt plus tool-schema
  context multiplies this. GPU quota exists in this project
  (`NVIDIA_T4_GPUS`/`NVIDIA_L4_GPUS`, limit 1.0 each in `us-central1`) but
  free-trial billing blocks GPU instance creation regardless of quota —
  confirmed by the account owner, not just a quota read. No CPU-only VM
  size tried so far gets this model within budget.

## Finding 2 — a likely separate adapter issue, independent of speed

One attempt (`c2-standard-8`, generous 600s per-call timeout so it wasn't
just a timeout) let the first model call actually finish: 442s real
latency, then failed anyway with `OllamaModelError` inside
`_request_content` (`ollama_model.py`), disposition `FAILED_SAFE`,
`model_calls: 1`, `tools_executed: 0`.

Ollama's own `/api/generate`/`/api/chat` response format keeps "thinking"
in a separate field from the final `content` (verified directly against a
raw response), so this is **not** simply chain-of-thought text leaking into
the JSON the adapter tries to parse. More likely: the model's
schema-constrained JSON answer was truncated or malformed after spending
most of its token budget on internal reasoning (no `num_predict`/context
override is set in the request payload; default context is 4096). This
needs the exact rejected content to confirm precisely — that repro was not
completed (see below) — but it is a real, reproducible-looking gap in the
adapter's handling of a reasoning-capable model under structured output,
independent of the hardware-speed problem above.

## Not completed

- A targeted repro isolating `_request_content`'s exact rejected payload
  (to know precisely which of its four `OllamaModelError` branches fired)
  was started but not finished — constructing a correct standalone
  `ModelRequest` requires fields (`run_id`/`graph_phase`/`model_turn`/
  `context_digest`) this session didn't take the time to get right, and
  each attempt costs several minutes of real GPU-less inference time.
- No case in the fixed 12-case corpus scored a real diagnosis. Zero
  progress toward the 9/12 experiment-eligibility gate.

## Recommendation

Two independent blockers, either of which stops a full run today:

1. Compute: needs either an approved (non-free-tier) GPU allocation, or
   empirical confirmation that some larger CPU-only shape actually clears
   the 360s budget — not yet found.
2. Adapter: `ollama_model.py`'s `_request_content` likely needs explicit
   handling for this model's reasoning/token-budget behavior (e.g. an
   explicit `think`/`options.num_predict` setting, or tolerance for a
   truncated/malformed first attempt with a repair turn) — needs review by
   whoever owns this adapter, not a guess fixed in isolation here.

The temporary VM (`causalops-phase4-temp`) was deleted after this session;
`causalops-test`'s own state (lab, Ollama container, pulled model) is
unchanged from before this attempt.
