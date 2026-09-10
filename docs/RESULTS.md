# Real-run results and findings

Everything on this page is a real, billed run against the live model —
never a projection, a simulation, or a number carried over from an
earlier draft. See the root [`README.md`](../README.md) for the
architecture, setup, and command reference; this page is the evidence
behind the claims made there.

## Paired live evaluation

```bash
uv run causalops-evaluate                      # executed_tools=2 (default)
uv run causalops-evaluate --executed-tools 3
uv run causalops-evaluate --executed-tools 4
```

A genuinely separate console script, not a `causalops` subcommand — it runs
a fixed, evaluator-hidden 12-incident corpus (4 families x 3 seeds —
`evaluation`, `evaluation_b`, `evaluation_c`) against the live model: one
no-tool baseline and one tool-enabled run per incident, saving every record
and a per-group summary under `results/evaluations/<id>/`. Each invocation
runs exactly one point on an evidence-budget curve —
`Budgets(executed_tools=N, model_calls=N + 2)` for `N` in `{2, 3, 4}` —
never all three in one run, so real spend can be checked between phases
rather than committed at once; the owner runs the command up to three
times, once per `--executed-tools` value, to build the full curve. Before
any scenario starts, a pre-flight check refuses cleanly if the configured
ceiling could not possibly cover this invocation's own worst-case batch
cost, on top of what the application has already spent or committed. It
requires `ANTHROPIC_API_KEY`, persists each completed record as it
finishes (not only at the end), and stops issuing further paid requests
only after an infrastructure-level failure (a missing credential, a
provider error, or the cost ceiling itself) — an ordinary model mistake is
still scored as a result, not treated as a reason to abort the batch.

Reported scores are mechanical: diagnosis and disposition correctness
against evaluator-only labels, citation validity and sufficiency against
required-evidence predicates, and a joint correct-and-grounded figure
combining the two. Every record also carries the git SHA, clean/dirty
status, fixture and prompt versions, retrieval mode, seed name, evidence
budget, exact model, tokens, latency, and cost — reproducibility is part of
the record, not an afterthought. Results are partitioned by `(arm,
retrieval_mode, executed_tools)` and reported as counts and ranges, never
blended across a retrieval mode or evidence-budget setting and never as a
p95 or a broad performance claim.

**The first full run of this curve found two mechanical bugs, not a model
problem.** `executed_tools`=2/3/4 against the same 12 incidents produced a
non-monotonic tool-enabled diagnosis-correct count — 6/12, then 9/12, then
5/12 — that traced back to `query_logs`'s `row_limit` argument: the model's
near-universal guess was 50, above the real 40-row policy budget, and that
guess drew a policy denial in 21 of the 36 tool-enabled runs, spread across
all three budget levels. Each denial still cost a model call. `model_calls =
executed_tools + 2` reserves exactly one spare call for a structured-output
repair, and a denial silently spent that spare before any repair was ever
needed — so when a later validation failure needed it (several runs'
`uncertainty`/`stop_reason` fields exceeded the 300-character cap then in
force, and that field's length genuinely grows with the evidence gathered),
nothing was left, and the run ended `REPAIR_EXHAUSTED` or
`MODEL_CALL_BUDGET_EXHAUSTED` instead of a diagnosis.

Three fixes landed together: the real 40-row limit is now named
directly in `QueryLogsArguments.row_limit`'s own schema description, not
only the denial message — see the root README's "A real defect this
project found and fixed" — and likewise for `SearchRunbooksArguments.limit`;
structured-output repairs now draw from their own independent budget
(`Budgets.repairs`), so a denial earlier in a run can no longer starve a
repair a later turn needs; and the fields that hit the 300-character cap
in real runs were raised to 600.

Re-run against the fix, same 12 incidents, same model config:

| `executed_tools` | baseline diagnosis-correct | tool-enabled diagnosis-correct | correct-and-grounded | `FAILED_SAFE` |
|---|---:|---:|---:|---:|
| 2 | 3/12 | 6/12 | 3/12 | 0 |
| 3 | 3/12 | 9/12 | 8/12 | 0 |
| 4 | 3/12 | 8/12 | 8/12 | 1 |

The one `FAILED_SAFE` at `executed_tools`=4 is unrelated to the row_limit
and repair-budget bug this section is about: that run hit
`MODEL_OUTPUT_INVALID` — the model returned a structurally empty assessment
object, missing required fields, on both its original attempt and its one
guaranteed repair — a separate, still-open failure mode.

The cleanest result: 21 policy denials across the three pre-fix batches'
36 tool-enabled runs became 0 across the three post-fix batches' 36
tool-enabled runs, and every one of the 24 paired incidents at
`executed_tools`=2 and =3 scored identically before and after — nothing
shuffled except the denial/repair mechanics. Tool-enabled also beat the
no-tool baseline at every budget level tested (6, 9, 8 against a flat
3/12), consistent across all six batches.

At `executed_tools`=4, three incidents flipped from incorrect to correct
and none regressed:

- **Flip 1 (attributable):** a `row_limit=50` denial, then an `uncertainty`
  cap failure with zero repairs attempted — `REPAIR_EXHAUSTED`. Post-fix:
  no denial, a clean diagnosis.
- **Flip 2 (attributable):** the same denial, then a `stop_reason` cap
  failure whose repair itself succeeded — but the denial's wasted call left
  no budget for the final-assessment call that came after, so the run still
  ended `MODEL_CALL_BUDGET_EXHAUSTED` at 6 of 6 calls used. Post-fix: no
  denial, a clean diagnosis.
- **Flip 3 (not attributable):** zero denials both before and after.
  Pre-fix, the run had already reached a correct, safe
  `INSUFFICIENT_EVIDENCE`/`UNDETERMINED` abstention after one successful
  repair — not a failure. Post-fix it diagnosed correctly instead, because
  the model chose to call `list_recent_changes` this time, which it hadn't
  pre-fix — a difference traceable to the model's own first-turn hypothesis
  ranking, before either run ever touched a tool call, denial, or repair.
  Read as ordinary run-to-run variance, not the fix working a third time.

The 12 incidents are 4 fault families × 3 seeds each, near-replicates
rather than independent draws — every seed within a family scored
identically at `executed_tools`=2 and =3, and 11 of 12 did at =4, so the
effective sample size behind this curve is closer to 4 than 12. That isn't
enough to establish an optimal evidence budget or a real accuracy trend;
the denial-elimination result above is the defensible claim, and the
per-point accuracy numbers are reported honestly and explained by the
traced mechanism, not asserted as a statistically established curve.

One family, `ambiguous_telemetry`, is correctly answered only by
abstaining. The no-tool baseline abstained correctly in all 18 runs across
all six batches; the tool-enabled arm never abstained once, diagnosing
something — almost always `RESOURCE_POOL_SATURATION` — every time. Tools
didn't help, and by these numbers hurt, on this one family.

`search_runbooks` was never called in any of the 72 tool-enabled records
across all six batches, at any budget level, though it's always available
in the tool-enabled arm — see "The runbook-usage investigation" below for
how this was eventually root-caused and fixed.

The three post-fix batches cost $3.02 in real spend; all six batches in
this investigation, pre- and post-fix combined, cost $6.40.

All six batches above ran under `PROMPT_VERSION`/`TOOL_REGISTRY_VERSION`
`"7"`/`"7"`; a run made after these moved on is not directly comparable to
the numbers in this section.

## The v8 validation run

Commit `6b27e228d08ab82a0b5d3437a54e9bc10ea0c63c` ("Surface respond()
rejection reasons, require causal evidence") did two things: it gave
`LiveClaudeModel.respond()` a real error channel — 5 distinct rejection
reasons instead of one generic message reaching the repair prompt — and it
partially corrected a real `ambiguous_telemetry` abstention regression,
via two additive prompt-text changes that ask two different things:
`SYSTEM_TEXT` now says that collapsing two evidenced causes into one root
cause is itself a claim needing its own evidence, else the model must
answer UNDETERMINED and cite both; the separate `HYPOTHESIS_UPDATE` stage
instruction now asks the model to state evidence against its own
top-ranked hypothesis before ranking, recorded in that hypothesis's
`contrary_evidence_ids`. Alongside those two prompt changes, a
`downstream_timeout_rate` -> `downstream_timeout_share` metric rename and
reformulation also landed. Two live batches validated it against the same
12-incident corpus, at `executed_tools`=3 and `executed_tools`=4.

| Configuration | Diagnosis correct | Correct and grounded | Citation valid | `FAILED_SAFE` |
|---|---:|---:|---:|---:|
| No-tool baseline (et=3) | 3/12 | 0/12 | 12/12 | 0 |
| Tool-enabled, et=3 | 11/12 | 5/12 | 12/12 | 0 |
| Tool-enabled, et=4 | 7/12 | 6/12 | 7/12 | 5 |

The raw per-run records behind this table are checked in at
`evaluation-evidence/` — structured scores and metadata only, no model
prose — so this scorecard can be independently verified against real data.

The baseline's `12/12` citation-valid figure is real, not a placeholder:
every baseline run validly cites the free `SYMPTOM`/`TOPOLOGY` evidence every
investigation gets regardless of tool access — it just never has enough
evidence for a required predicate to satisfy `citations_sufficient`, which is
exactly why `correct_and_grounded` is `0/12` there.

- **`executed_tools`=3 is the recommended operating point**: near-perfect
  diagnosis (11/12), zero `FAILED_SAFE` runs, the best result of any budget
  tested under v8.
- **Ambiguous-case abstention improved from a historical 0/18 to 2/3 in this
  run** — the tool-enabled arm's prior 0/18 record on `ambiguous_telemetry`
  is the same one stated above under "Paired live evaluation" ("the
  tool-enabled arm never abstained once").
- **The strict grounding score requires predeclared log evidence.**
  `citations_sufficient`/`correct_and_grounded` require citing the specific
  required-evidence predicate for that incident — usually a `query_logs`
  result — not just any correct-looking evidence.
- **Of the 6 et=3 correct diagnoses that failed the grounding bar**: 5 never
  called `query_logs` at all — a correct diagnosis reached off metric/change
  evidence alone, missing the required log predicate entirely (4 non-
  `ambiguous_telemetry` runs plus 1 of the 2 `ambiguous_telemetry`
  cases) — and 1 (`ambiguous_telemetry`) called `query_logs` and retrieved
  one of the two required predicates (`pool_exhausted`) but not the other
  (`upstream_timeout`). These are genuinely different situations, not one
  blanket "didn't retrieve logs" failure.
- **`executed_tools`=4 showed that more evidence-gathering budget can reduce
  structured-output reliability, not just improve grounding.** All 5
  `REPAIR_EXHAUSTED` failures at et=4 trace to the same mechanism, not 5
  independent ones: `Budgets.repairs=1` is one repair credit for the whole
  investigation, not one per stage. In every one of the 5, an
  INVESTIGATE-stage turn hits `"tool-call response must not include visible
  text"` and spends that one credit. In 3 of the 5, that repair succeeds and
  it's a later `FINAL_ASSESSMENT` rejection that then finds zero credit left
  (an `uncertainty`-length-cap violation in 1, a missing-both-`uncertainty`-
  and-`next_step` violation in 1, a missing-`next_step`-only violation in
  1). In the other 2, a second visible-text violation follows within the
  same INVESTIGATE stage before any credit remains, and `FINAL_ASSESSMENT`
  still runs afterward and fails too, inheriting the same zero-credit state
  (an `uncertainty`-length-cap violation in 1, another visible-text
  violation in 1). The failures trace to an interaction between stochastic
  structured-output violations and a single investigation-wide repair
  credit; the extra turn at et=4 increased exposure to that interaction in
  this batch. None of the 5 reached a confident wrong diagnosis; every one
  is a contained, safe stop, the safety design working exactly as
  intended. The mechanism itself is still open in this codebase — a
  candidate for future work, not yet fixed.

These two v8 batches cost **$2.80** in real spend ($1.26 at et=3, $1.54 at
et=4).

## The runbook-usage investigation

`search_runbooks` sitting at 0 real uses across every batch above — 72
tool-enabled records at v6/v7, plus the two v8 batches, 120 total — raised
an obvious question worth its own investigation: is the model simply never
*incentivized* to reach for it?

### The Pinecone semantic-retrieval experiment

`infra/phase4/PREREGISTRATION.md`'s "Fixed retrieval comparison" set the bar
before any Pinecone code existed: select Pinecone over FTS5 only with
**at least 3/12 actual retrieval uses, no safety regression, and
non-inferior correct-and-grounded results** against a matched FTS5 arm over
the same corpus, budgets, and model identity. A real, stated risk going in:
if the 0/120 usage pattern held with Pinecone wired too, the experiment
would fail on usage alone, independent of retrieval quality.

The engineering was built for real: a Pinecone serverless index
(`multilingual-e5-large` hosted embedding inference, no second API key),
`pinecone_runbooks.py` mirroring `runbooks.py`'s exact interface, gated
behind `RAG_EXPERIMENT_ENABLED` in both tool-registry composition roots
(`live_setup._build_tool_registry` and
`mcp_client_registry.build_mcp_tool_registry` — the real MCP transport
`causalops-evaluate` runs on), proven to fail closed without
`PINECONE_API_KEY` before any network call, and live-verified against the
real provisioned index for every `RunbookTopic` before any evaluation run.

**The preregistered comparison then ran for real** — two `causalops-evaluate
--executed-tools 3` invocations over the same 12-incident corpus, one per
arm, real billed Claude requests:

| Arm | Run ID | `search_runbooks` used | `correct_and_grounded` |
| --- | --- | --- | --- |
| FTS5 (`RAG_EXPERIMENT_ENABLED=false`) | `48225f19a9ea452abb8bb6a51c5fa3a2` | 0/12 | 5/12 |
| Pinecone (`RAG_EXPERIMENT_ENABLED=true`) | `fabad365a29c48b7ab77a22d7731f4ca` | 0/12 | 4/12 |

Zero policy denials and zero `REPAIR_EXHAUSTED`-or-worse failures in the
Pinecone arm; `control.denied`/`control.out_of_scope` were 0/12 in both
arms. **Pinecone was not selected: `pinecone_used_count` was 0/12, short of
the preregistered 3/12 floor.** The model never called `search_runbooks`
in either arm — the same behavior FTS5 alone already showed, now confirmed
with a real, working semantic backend genuinely available to call. This is
an honest negative result against the exact bar set in advance, not a
build defect: the retrieval backend works (see the topic-by-topic live
verification proved during development); the model simply never reached
for it, in either retrieval mode, at this evidence-budget point.

One scoring-path note for a reader comparing this to the Qwen candidate
track: `candidate_evaluation.py`'s `RetrievalComparisonRecord` and
`causalops-candidate-assess --retrieval-comparison` implement this same
preregistered rule, but against `CandidateEvaluationRecord`'s Qwen-specific
shape (`model_name: Literal["qwen3.5:4b"]`, Ollama image/manifest digests)
— built for the private-VM Qwen track the preregistration document also
covers, not for `causalops-evaluate`'s Claude-shaped `EvaluationRecord`.
The comparison above applies the same selection rule directly to the two
real `EvaluationRecord` batches instead of forcing them through a schema
built for a different model identity.

### Three more angles on the same question — two negative, one that worked

Three follow-up experiments, all real, all live — two negative, one that
reversed the pattern completely:

**Does pricing a lookup the same as a scarce diagnostic check explain it?**
`Budgets.runbook_searches` gives `search_runbooks` its own dedicated pool
(default 1), separate from the scarce `executed_tools` pool every other
check draws from — a call to it no longer competes with `query_logs`/
`query_metric`/etc. for the same slot, and `SYSTEM_TEXT` said so
explicitly ("Checking it does not spend any of your diagnostic check
budget"). Run for real (`causalops-evaluate --executed-tools 3`, FTS5,
12 incidents): **0/12 uses**, identical to every batch before this change.
Making the lookup free and saying so explicitly did not move the number at
all.

**Does a weaker, less confident model reach for guidance more, the way a
junior engineer leans on a runbook while a senior one skips it?**
`CAUSALOPS_LIVE_MODEL=haiku` (Claude Haiku 4.5, ~5x cheaper than Sonnet 5)
against the same 12 incidents, same free runbook budget: **0/12 uses**
again — and diagnosis_correct fell to 0/12 (from Sonnet's 7-11/12 across
earlier batches), with 27 `invalid_responses` across the 12 tool-enabled
runs against Sonnet's near-zero rate. Haiku did not struggle *confidently*
in a way that made it reach for more guidance; it struggled to produce
valid structured tool calls at all, repeatedly burning repair budget
before ever reaching a state where consulting a runbook would even be a
live option. Confidence was the wrong axis to test this way — a model
this far below the task's basic schema-compliance bar never gets far
enough into the investigation loop for a "would extra guidance help"
choice to arise.

**Does a direct imperative instruction, not just an informational one,
work?** The dedicated-budget sentence above only *informed* the model the
lookup was free; it never *told it to use it*. `SYSTEM_TEXT` gained a
third, separate instruction: its first proposal in every investigation
must be one `search_runbooks` call, before any incident-scoped check,
with an explicit stop condition (`"runbook searches left"` already zero,
or a runbook line already in evidence) so it does not repeat this on
later turns. Run for real (`causalops-evaluate --executed-tools 3`, FTS5,
Claude Sonnet 5, the same 12 incidents): **12/12 uses — every single run**,
each one on the first proposal, none repeated. Zero policy denials.
`diagnosis_correct` 8/12 and `correct_and_grounded` 5/12, matching or
slightly exceeding the pre-instruction FTS5 baseline (7/12, 5/12) — the
added check did not cost correctness or grounding to get. Cost $3.76 for
the batch.

So the three experiments together answer the original question precisely:
the model was capable of using the tool all along (backend quality,
budget pricing, and model strength all ruled out as blockers), it simply
never chose to on its own, at this evidence-budget point, under an
*informational* prompt. Telling it to, directly and unambiguously, is
what moved it from 0 to 12/12 in one change. `runbook_searches` stays a
separate budget (the more correct design regardless of this result —
general guidance should not compete with incident-scoped evidence for the
same scarce slot); `CAUSALOPS_LIVE_MODEL` stays available for future
experiments, not as a production alternative to Sonnet 5.

**Confirmed at et=4 too, not just et=3**: same instruction, same model,
`--executed-tools 4` — **12/12 uses again**, `diagnosis_correct` 7/12,
`correct_and_grounded` 3/12 (consistent with the earlier-observed
et=3-outperforms-et=4 pattern above, unrelated to retrieval).

### The Pinecone rerun, with usage no longer the blocker

The original Pinecone comparison above (0/12 vs 0/12) left the question of
retrieval *quality* untested, since neither arm ever actually retrieved
anything to judge. With the model now reliably calling `search_runbooks`
on every run, the same matched comparison was repeated for real, same
corpus, same budgets:

| Arm | Run ID | `search_runbooks` used | `correct_and_grounded` |
| --- | --- | --- | --- |
| FTS5 (`RAG_EXPERIMENT_ENABLED=false`) | `4e2f8ec48a354e088c675c9b46ed8acb` | 12/12 | 5/12 |
| Pinecone (`RAG_EXPERIMENT_ENABLED=true`) | `f493cc19efc04f04a37c029f8d51c85d` | 12/12 | 4/12 |

**Still not selected — but for a different, more informative reason this
time.** The usage floor (≥3/12) is now trivially cleared on both arms;
zero policy denials or safety regressions in either. The blocker is now
squarely `correct_and_grounded`: Pinecone's 4/12 is not non-inferior to
FTS5's 5/12, missing by exactly one incident. This is the first time the
preregistered rule has actually been tested against real retrieval
quality rather than real retrieval absence — a genuine, if narrow,
quality gap, not another usage no-op. FTS5 remains the only retrieval
path in production use.

## Measured lessons

- **Tool schema descriptions materially affect agent behavior.** The
  `query_logs`/`row_limit` schema-description fix under "A real defect this
  project found and fixed" (root README), and the denial-elimination result
  it produced under "Paired live evaluation" above, are the clearest
  evidence: nothing in policy or the graph changed, only what the model was
  told about a tool it already had.
- **Policy denials fell from 21/36 to 0/36** across the pre-fix and post-fix
  tool-enabled batches described under "Paired live evaluation" above — the
  same schema/budget fix.
- **Three diagnostic checks (et=3) outperformed four (et=4)** in the v8
  validation run above — see "The v8 validation run" for the traced
  repair-budget mechanism behind that result.
- **Diagnosis correctness and evidentiary grounding are measurably distinct
  properties, not one score.** 11/12 correct diagnoses at et=3 and only
  5/12 correct-and-grounded — see "The v8 validation run" for what the
  other 6 were missing.
- **Retrieval (`search_runbooks`) sat at 0 real uses for 192 consecutive
  tool-enabled records across four independent conditions — FTS5-only, a
  real Pinecone backend, a free dedicated budget, and a 5x-cheaper model —
  until one direct imperative instruction moved it to 12/12, confirmed at
  two separate evidence-budget points.** See "The runbook-usage
  investigation" above for the full sequence. Backend quality, budget
  pricing, and model capability were each tried and each ruled out as the
  blocker; only telling the model directly what to do, rather than
  informing it a tool was free or advantageous, actually changed its
  behavior. The mechanical lesson: for this model, on this task, an
  *informational* system-prompt sentence and a *directive* one are not
  interchangeable, even when they describe the same underlying incentive —
  and the true cause of the original 0/192 pattern (why an informational
  nudge never worked, only a direct instruction did) is still open.
- **Once usage stopped being the blocker, Pinecone lost on quality, not
  absence.** The rerun above found Pinecone's `correct_and_grounded` one
  incident short of FTS5's — a real, narrow finding the original
  0-uses-both-arms comparison could never have produced.
