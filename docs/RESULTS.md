# Real-run results

Every number below is a real, billed run against the live model — never a
projection or simulation. Run IDs are given so any row can be independently
checked against `results/evaluations/<id>/`. See the root
[`README.md`](../README.md) for architecture and setup.

## Headline: evidence-budget curve (FTS5, Claude Sonnet 5)

12-incident corpus, `causalops-evaluate --executed-tools <N>`, one no-tool
baseline + one tool-enabled run per incident.

| et | Baseline diagnosis | Tool-enabled diagnosis | Correct & grounded | Citation valid | `FAILED_SAFE` |
|---|---:|---:|---:|---:|---:|
| 2 | 3/12 | 6/12 | 3/12 | — | 0 |
| 3 | 3/12 | **11/12** | 5/12 | 12/12 | 0 |
| 4 | 3/12 | 7/12 | 6/12 | 7/12 | 5 |

- **et=3 is the recommended operating point** — best diagnosis rate, zero
  `FAILED_SAFE`.
- et=4's 5 `FAILED_SAFE` runs all trace to one mechanism: a single
  investigation-wide repair credit gets consumed by an early structured-
  output violation, leaving none for a later one. Safe stops, not wrong
  diagnoses — still an open item, not yet fixed.
- Diagnosis correctness and grounding are distinct scores: et=3 hit 11/12
  diagnoses but only 5/12 were also fully grounded in required evidence.
- A real bug was found and fixed via these runs, not code review: `query_logs`'s
  `row_limit` schema didn't state the real 40-row policy limit, so the
  model's default guess (50) drew a policy denial in 21 of 36 tool-enabled
  runs, each one burning the run's one spare repair credit. Fix: state the
  real limit in the schema description. Result: 21/36 denials → 0/36.
- 12 incidents = 4 families × 3 seeds (near-replicates); effective sample
  size is closer to 4 than 12 — read per-point percentages accordingly.
- `ambiguous_telemetry` (correct answer: abstain) was answered correctly by
  the no-tool baseline every time (18/18 across all early batches) and
  incorrectly by the tool-enabled arm every time in those same batches —
  since fixed (see below).

## Retrieval backend comparison: FTS5 vs Pinecone

Matched arms, same 12 incidents, same budgets, real billed requests.

| et | Backend | `search_runbooks` used | Correct & grounded | Run ID |
|---|---|---:|---:|---|
| 3 | FTS5 | 12/12 | 5/12 | `4e2f8ec48a354e088c675c9b46ed8acb` |
| 3 | Pinecone | 12/12 | 4/12 | `f493cc19efc04f04a37c029f8d51c85d` |
| 4 | FTS5 | 12/12 | 3/12 | `20f432b02ee44cd89da31be43ff4d907` |
| 4 | Pinecone | 12/12 | 5/12 | `b6ecdf0fe605453780d8aa3625e81b45` |

- **Preregistered selection rule**: select Pinecone only with ≥3/12 real
  uses, no safety regression, and non-inferior grounding vs. FTS5 — at
  et=3, the recommended operating point.
- **Not selected at et=3** — grounding 4/12 vs FTS5's 5/12, one incident
  short of non-inferior. Zero policy denials, zero safety regressions in
  either arm.
- **Mixed at et=4** — Pinecone actually beat FTS5 there (5/12 vs 3/12), but
  et=4 isn't the operating point the rule was scoped to, so this doesn't
  change the selection decision. Reported honestly, not cherry-picked away.
- **Production stays FTS5.**
- Before the runbook-usage fix below, both backends showed 0/12 usage at
  et=3 — the comparison above only became testable after that fix; see
  run IDs `48225f19a9ea452abb8bb6a51c5fa3a2` (FTS5) /
  `fabad365a29c48b7ab77a22d7731f4ca` (Pinecone) for the original 0-vs-0
  result.

## Why the model never used `search_runbooks` — and the fix

0 real uses across 192 tool-enabled records, four independent conditions
tried and ruled out, before one fix worked:

| Lever tried | Result | Usage |
|---|---|---:|
| FTS5 only (default) | No change | 0/12 |
| Real Pinecone backend | No change | 0/12 |
| Free dedicated budget + informational prompt sentence | No change | 0/12 |
| Claude Haiku 4.5 (5x cheaper, presumably less confident) | No change (and diagnosis collapsed to 0/12 — Haiku struggled with structured tool calls generally) | 0/12 |
| **Imperative instruction** ("your first proposal must be search_runbooks") | **Fixed** | **12/12**, both et=3 and et=4 |

- Backend quality, budget pricing, and model capability were all ruled out
  as the blocker.
- Telling the model directly what to do — not just informing it a tool was
  free or available — is what worked. An informational prompt sentence and
  a directive one are not interchangeable here.
- No cost to correctness: diagnosis/grounding at et=3 with the instruction
  (8/12, 5/12) match or slightly beat the pre-instruction baseline
  (7/12, 5/12).
- `Budgets.runbook_searches` (a dedicated pool, separate from
  `executed_tools`) stays in place regardless — the more correct design.

## Cost

~$30 in real Anthropic spend across every batch on this page, application-
wide, tracked in `cost_ledger` and reserved/settled before every request
against `LIVE_EVALUATION_MAX_USD`.
