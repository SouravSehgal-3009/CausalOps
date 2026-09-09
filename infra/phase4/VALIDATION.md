# Phase 4 restricted-environment validation record — candidate evaluation attempt

Dedicated VM (`causalops-test`, asia-south1-c, project
`project-7b68a103-d482-4418-a35`), plus one temporary compute-optimized VM
(`causalops-phase4-temp`, same zone, `c2-standard-8`, deleted after use).
Two work sessions, 2026-09-09, git range `33a02ff` through `23cb23f`
(local, uncommitted-to-push at time of writing per this project's CI-push
frugality practice — see commit history for exact SHAs once pushed).

**Verdict: `qwen3.5:4b` does not currently clear the preregistered
eligibility bar.** The infra/adapter blockers that made this untestable are
now fixed (below); with them fixed, the model itself failed the same way
on two independent full-batch attempts. This is treated as a real,
recorded negative finding, not an open bug.

## Infra and adapter fixes applied (all confirmed working)

Four scoped changes, each verified against real live runs before being
kept:

1. **`"think": False`** added to the Ollama request payload
   (`ollama_model.py`). qwen3.5:4b defaults to extended hidden reasoning —
   Ollama's own response showed 179 reasoning tokens for a prompt whose
   entire visible answer was "OK". Measured effect: a real call went from
   400-900+ seconds (frequently not returning at all) to ~60-200 seconds.
2. **HTTP request timeout raised 30s → 200s** (same file). The original
   30s was tuned before `think: false` existed; even with thinking off, a
   real production prompt (full system text + tool schemas) measured just
   over 30s, and a repair-turn prompt (original context plus the
   rejection reason appended) measured over 120s.
3. **A Qwen-only system-prompt reminder** appended in `_request_content`
   telling the model to always include a literal `"tool"` field inside its
   tool-call `arguments`, matching the proposed tool's name. Root cause:
   each tool-arguments schema marks that field `const` with a `default`,
   but Pydantic's discriminated-union resolution needs the key physically
   present in the response to route to the correct sub-schema — it does
   not apply the schema default before discriminating. Scoped to this
   adapter only, not the shared prompt in `prompts.py` that Claude/replay
   also use.
4. **`wall_clock_seconds` raised 360s → 900s**, scoped to
   `qwen_evaluate_cli.py`'s own `Budgets()` construction only (the shared
   `domain.py` default, used by Claude/replay, is untouched). Measured
   need: a run that self-corrected correctly (repair succeeded, tool
   executed, reached `hypothesis_update` cleanly) still hit
   `WALL_CLOCK_EXPIRED` at 362s — one repair round-trip alone can consume
   the whole original budget on this hardware.

GPU quota exists in this project (`NVIDIA_T4_GPUS`/`NVIDIA_L4_GPUS`,
limit 1.0 each in `us-central1`) but free-trial billing blocks GPU
instance creation regardless of quota — confirmed by the account owner. A
temporary `c2-standard-8` CPU VM measured roughly 2x the token throughput
of `causalops-test`'s `e2-standard-4`, but was not ultimately needed once
the above four fixes landed — `causalops-test` alone now completes
individual cases cleanly within budget.

## The recurring failure, after all four fixes

Two independent full 12-case batch attempts (`git 8f29ee5` and
`git 3f428e8`+900s budget) both stopped at **case 2 of 12** with a
`FAILED_SAFE` disposition, for the same underlying reason each time:

- Attempt A: repair succeeded structurally (valid tool call, tool
  executed, `hypothesis_update` completed) but the sequence took long
  enough that `final_assessment` found the 360s budget already spent →
  `WALL_CLOCK_EXPIRED`. (This is the run that justified fix 4 above.)
- Attempt B (after fix 4): the discriminator-tag error
  (`"proposal.arguments: Unable to extract tag using discriminator
  'tool'"`) fired on the **first** attempt in `initial_plan`, exactly as
  fix 3 targets — but the **repair attempt hit the identical error a
  second time**, exhausting the one-repair budget → clean
  `MODEL_OUTPUT_INVALID` (a proper reason code, not a crash).

Case 1 of 12 (`configuration_change/evaluation`) completed cleanly in
every attempt — `INSUFFICIENT_EVIDENCE`, valid citations, no crash,
250-345s — but the model chose not to call any tool before concluding,
each time. Zero cases so far have both attempted a tool call and produced
a scored diagnosis.

**Conclusion:** the discriminator-field reminder (fix 3) measurably helps
— it was directly observed succeeding once on a repair attempt — but is
not reliable enough to clear the strict preregistered bar
(`PREREGISTRATION.md`: "Stop the batch on a policy escape or `FAILED_SAFE`
result. Neither may be averaged away"). This reads as a genuine
reliability limitation of `qwen3.5:4b` at Q4_K_M quantization for this
specific discriminated-union JSON-schema shape, not a bug left to fix.
Further prompt tuning was considered and deliberately not pursued
further, to avoid tuning the evaluation protocol itself toward a forced
pass.

## Manifest/digest identity used throughout

- Ollama image: `sha256:32931b46719f673c05fdbaa81ccb26da18ea4a1c57590a754874ab28ba269eb2`
- Qwen model manifest: `sha256:2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
  (matches `ollama list`'s displayed short ID `2a654d98e6fb`)

## Recommendation

Do not keep tuning `qwen3.5:4b` in isolation. Two real paths forward, both
out of scope for this session:

1. Reconsider model choice (a different local model, or a larger/different
   quantization of Qwen) — would need its own preregistration update and
   repeat this same infra validation.
2. Use the already-working Claude live-evaluation path
   (`causalops-evaluate`) as the working reference baseline instead, and
   treat local-model candidate evaluation as a separately scoped effort to
   revisit later.

The temporary VM (`causalops-phase4-temp`) was deleted after use;
`causalops-test`'s own state (lab, Ollama container, pulled model) is
otherwise unchanged.
