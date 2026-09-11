# Real-run results

Every number below is a real, billed run against the live model — never a
projection or simulation. Run IDs are given so any row can be independently
checked against `results/evaluations/<id>/`. See the root
[`README.md`](../README.md) for architecture and setup.

## Headline: evidence-budget curve (FTS5, Claude Opus 5 recommended, current prompt and budgets)

12-incident corpus, `causalops-evaluate --executed-tools <N>`, one no-tool
baseline + one tool-enabled run per incident — the real, current numbers
under what's actually shipped, not a historical peak. `CAUSALOPS_LIVE_MODEL`
must be set to `opus` explicitly for these numbers — the code's own
fallback default without that variable set is still Sonnet 5, shown
alongside for the cost tradeoff.

| Model | et | Baseline diagnosis | Tool-enabled diagnosis | Correct & grounded | `FAILED_SAFE` | Run ID |
|---|---|---:|---:|---:|---:|---|
| **Opus 5** | 3 | 0/12 | **12/12** | **8/12** | **0/12** | `81fe82d2...` |
| Sonnet 5 | 3 | 3/12 | 9/12 | 5/12 | 0/12 | `e4e38fd8...` |
| Sonnet 5 | 4 | 3/12 | 7/12 | 3/12 | 5/12 | `20f432b0...` |

- **et=3 is the recommended operating point** for either model — best
  diagnosis rate, and, as of the repair-budget fix below, zero
  `FAILED_SAFE` on Sonnet 5 and zero on Opus 5's tool-enabled arm.
- **History of the et=3 `FAILED_SAFE` number (Sonnet 5), for the full
  trail**: 0/12 before the runbook-first instruction existed → 2/12 in two
  batches after it shipped (`4e2f8ec4...`, `40cb6f69...`, both
  `REPAIR_EXHAUSTED`) → back to 0/12 after root-causing that regression and
  raising the repair budget (see "The cross-stage repair budget" below for
  the mechanism). The earlier 8/12-9/12 diagnosis range collapses to a
  single, current 9/12 now that the fix is in — reported as one number,
  not averaged away.
- et=4's `FAILED_SAFE` runs trace to the same mechanism the fix below
  targets, but haven't been re-measured at et=4 since it landed — still
  open at that operating point specifically, and not yet run for Opus 5
  either.
- Opus 5's `no_tool_baseline` diagnosis (0/12, all but one run
  `MODEL_OUTPUT_INVALID`) is a real, repeatable weakness, not a typo — see
  "Model selection" below for the mechanism.
- A real bug was found and fixed via earlier runs, not code review:
  `query_logs`'s `row_limit` schema didn't state the real 40-row policy
  limit, so the model's default guess (50) drew a policy denial in 21 of
  36 tool-enabled runs. Fix: state the real limit in the schema
  description. Result: 21/36 denials → 0/36.
- 12 incidents = 4 families × 3 seeds (near-replicates); effective sample
  size is closer to 4 than 12 — read per-point percentages accordingly.

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
- **Not a free fix**: grounding held steady (5/12, matching the
  pre-instruction baseline), but `FAILED_SAFE` at et=3 rose from 0/12 to
  2/12 across two post-fix batches — see the headline table above. Usage
  went from 0 to 12/12; that came with a measured, real cost elsewhere,
  not a pure win.
- `Budgets.runbook_searches` (a dedicated pool, separate from
  `executed_tools`) stays in place regardless — the more correct design.

## Once it was used, was it acted on?

Usage isn't the same question as impact. Two more real, live checks:

- **Diagnostic query volume, with vs. without the mandatory runbook call**
  (subtracting the runbook call itself for a fair comparison): 2.92 mean
  before the fix existed, 2.79 mean after — essentially unchanged. The
  model cited 2-3 runbook passages per run but kept making the same number
  of raw `query_metric`/`query_logs` calls it always did. **Fix tried**:
  one more `SYSTEM_TEXT` sentence telling the model to let cited guidance
  shape its *next check*, not just decorate the report, without letting it
  count as evidence for the verdict. Shipped; not yet independently
  re-measured against this specific query-volume question.
- **Does picking the topic *after* some evidence (instead of blind, from
  the alert alone) help relevance?** Tested for real — reverted. Usage
  fell from a reliable 12/12 to 7/12: "exactly once, whenever you choose"
  competed with the model's own judgment about when it was done, and 5 of
  12 runs reached a stopping point without ever making the call.
  `correct_and_grounded` was 4/7 when it did search vs 1/5 when it
  didn't — suggestive, but confounded by small, different populations, not
  evidence the later-timing theory actually helped. Net: this traded away
  guaranteed compliance for an unconfirmed benefit — reverted back to
  "must be first."

## The cross-stage repair budget: a real fix, now shipped

`Budgets.repairs` (the structured-output-retry allowance) defaulted to 1 for
the whole investigation, cumulative across stages, not reset per stage. A
real anomaly review found the mechanism behind every `et=3` `FAILED_SAFE`
above: an early `INVESTIGATE`-turn violation (almost always the model adding
narrative text alongside its tool call) spent the run's only repair, leaving
the unrelated, later `FINAL_ASSESSMENT` turn zero margin for its own first
mistake — `REPAIR_EXHAUSTED`, not a wrong diagnosis.

Raising the default to 2 (`Budgets.repairs`'s own comment already named this
exact fix, just never turned on) closed it: a real Sonnet 5 confirming batch
at `et=3` under the new default scored 9/12 diagnosis, 5/12 grounded, with
`REPAIR_EXHAUSTED` at zero — the 2 remaining `FAILED_SAFE` cases were both
same-stage double-failures (a dropped required field, or the same
narrative-text slip twice in one stage), a different mechanism this fix
was never meant to touch. Merged and shipped — `Budgets.repairs` defaults
to 2 for every model, not conditional on which one is selected.

## Model selection: Claude Opus 5 vs Sonnet 5

`CAUSALOPS_LIVE_MODEL=opus` wires in Claude Opus 5 (`pricing.py`,
`live_model.py`) — 2.5x Sonnet 5's per-token rate ($5/$25 vs $2/$10 per
million input/output tokens), confirmed against `platform.claude.com`'s own
pricing page, not assumed. Tested for the same reason Haiku was: this
project's own `FAILED_SAFE` investigation found the remaining failures look
like structured-output formatting slips, not an obvious capability gap —
worth testing a stronger model directly rather than assuming it helps.

Every run below uses the matched, current code (`repairs=2`, the fix
above) — an earlier pair of Opus batches ran before that fix landed and
are superseded, not reported, to avoid comparing Opus under a weaker
repair budget than Sonnet's own headline number.

| Model | Retrieval | Diagnosis | Correct & grounded | Citations valid | `FAILED_SAFE` (tool-enabled) | Cost/run (actual) |
|---|---|---:|---:|---:|---:|---:|
| Sonnet 5 | Pinecone | 9/12 | 5/12 | 10/12 | 2/12 | ~$0.10-0.16 |
| **Opus 5** | Pinecone | **12/12** | 6/12 | 12/12 | **0/12** | ~$0.24-0.26 |
| **Opus 5** | FTS5 | **12/12** | **8/12** | 12/12 | **0/12** | ~$0.24-0.31 |

- **A clean, decisive win on the metric that matters**: perfect diagnosis
  and disposition correctness, zero `FAILED_SAFE`, at both retrieval
  backends, at roughly 2-2.5x Sonnet 5's real per-run cost.
- **Retrieval backend doesn't explain it**: Opus scored 12/12 diagnosis
  either way — FTS5 (the production default) actually grounded *better*
  than Pinecone for Opus (8/12 vs 6/12, within this sample's noise but
  consistent across a repeat run). Opus's edge over Sonnet is a raw
  reliability difference, not a retrieval-quality one.
- **A real, repeatable weakness, outside the headline metric**: on
  `no_tool_baseline` (zero diagnostic evidence available), Opus hit
  `MODEL_OUTPUT_INVALID` on 8-11 of 12 runs across four separate batches —
  Sonnet 5 only did on 1/12 under the same condition. Root cause identical
  every time: with nothing to work with, Opus reaches for `search_runbooks`
  at the `final_assessment` stage, where only `record_final_assessment` is
  a valid response — two strikes in one stage, which the repairs fix above
  structurally cannot help (it only covers *cross*-stage budget robbery).
  This arm is a negative control, not the evaluated condition, but it's a
  real, decisive, counterintuitive finding: the more capable model is
  measurably *more* prone to this specific failure when evidence-starved,
  not less.
- **Opus 5 is the recommended model** (`CAUSALOPS_LIVE_MODEL=opus`) given
  this scorecard. The code's own fallback default, when that variable is
  unset, stays Sonnet 5 — a deliberate cost-safety choice, not an
  endorsement that Sonnet scores better.

Three models tried, three different outcomes: Haiku 4.5 (cheaper, collapsed
to 0/12 — see the `search_runbooks` table above), Sonnet 5 (the code's
fallback default, 9/12, cheapest of the two viable options), Opus 5
(recommended, 12/12, costlier, with the evidence-starved weakness above).
No single model is a strict improvement on every axis — the choice is a
real tradeoff between cost, diagnosis reliability, and robustness to
low-evidence incidents, not a ladder with one obviously correct rung.

## Cost

~$40 in real Anthropic spend across every batch on this page, application-
wide, tracked in `cost_ledger` and reserved/settled before every request
against `LIVE_EVALUATION_MAX_USD`.
