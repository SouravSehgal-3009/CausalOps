# CausalOps v2: Governed Evidence-Grounded Incident Investigator

## 1. Document authority and current status

This is the sole current product specification for CausalOps v2. It supersedes
conflicting product decisions in `TECHNICAL_OVERVIEW_OLD.md`. `_OLD` files are
gitignored local convenience copies of tracked history and are not themselves
authoritative; the tracked git history (e.g.
`git show b6f4d9c:TECHNICAL_OVERVIEW.md`) is authoritative for anything an
`_OLD` file claims to preserve. Files ending in `_OLD` are historical evidence
of the Phase-2 foundation and must not be silently changed or treated as
current requirements.

**Owner-reported foundation:** Phase 2 of CausalOps includes a synthetic Docker
lab, Prometheus and JSONL telemetry, scenario controller, four typed read-only
evidence tools, opaque incident scope, evaluator-only ground truth, replay
conformance, deterministic scoring, and multiple incident families.

Before this status appears in a résumé, README, or demo, the repository must
contain the corresponding source, tests, commit SHA, and saved test/evaluation
artifacts. This specification does not by itself prove implementation.

**v2 objective:** evolve that working foundation into a bounded agentic
workflow without weakening its safety or evaluation boundaries.

## 2. Portfolio positioning

**Public name:** `CausalOps v2 — Governed Evidence-Grounded Incident
Investigator`

“Causal” refers to a practical loop—hypothesis, diagnostic check, evidence
update—not formal causal inference. CausalOps is decision support for an
on-call engineer. It does not autonomously operate infrastructure.

The interview story is:

> I evolved a deterministic, evidence-scored incident investigator into a
> stateful agent with real tool calling and human-approved escalation, while
> retaining strict scope, budget, provenance, and ground-truth controls.

## 3. Product contract

CausalOps v2 investigates a synthetic incident in the existing local lab. It
receives an answer-neutral alert with an opaque incident ID, forms competing
hypotheses, chooses at most two safe diagnostic checks, and returns either a
cited diagnosis or an explicit abstention.

The central trust boundary remains:

> The model proposes and interprets. Deterministic code validates, authorizes,
> executes read-only checks, stops, scores, and records.

The model receives no evaluator manifest, semantic scenario key, expected root
cause, required-evidence predicate, secret, host path, or scenario-controller
capability.

## 4. v2 architecture

```text
Existing CLI and report interface
    |
    v
LangGraph StateGraph + SQLite checkpoints
    |
    +-- investigator model (Claude or replay)
    +-- typed dispatch node reaching only policy-wrapped read-only tools
    |      +-- query_metric        -> existing Prometheus adapter
    |      +-- query_logs          -> existing JSONL adapter
    |      +-- list_recent_changes -> existing manifest adapter
    |      +-- get_topology        -> existing manifest adapter
    |      +-- search_runbooks     -> optional retrieval adapter
    +-- deterministic evidence normalizer
    +-- existing scorer and JSONL run records
    +-- LangGraph escalation interrupt
    +-- append-only approval/audit record
    |
    v
Diagnosis, abstention, or safe failure report
```

LangGraph replaces the old orchestration loop only. Existing domain models,
tool-policy rules, telemetry backends, scenario controller, evidence records,
and scorer remain the authority and should be reused as pure functions.

There is one investigator/planner. v2 does not add a supervisor, specialist
agents, a web UI, or multi-agent conversation.

## 5. Investigation graph and budgets

```text
CREATED
  -> INVESTIGATE
  -> DISPATCH_TOOL (only when a validated read-only tool call is proposed)
  -> NORMALIZE_EVIDENCE
  -> INVESTIGATE | FINAL_ASSESSMENT
  -> ESCALATION_INTERRUPT (only when required)
  -> FINAL_REPORT
```

- `INVESTIGATE` binds only the registered tool wrappers, not the existing
  backend functions.
- A wrapper validates typed arguments through the existing incident-scope,
  template, budget, and duplicate-proposal policy before calling a backend.
- A typed dispatch node reaches a tool backend only through its policy
  wrapper — never a bare backend function. A direct backend binding is a P0
  trust-boundary violation because it bypasses existing authorization,
  regardless of which graph node performs the dispatch.

  *Amendment, Unit 0:* the original wording named LangChain's `ToolNode`
  class. Milestone 1 implements this boundary with a plain typed dispatch
  node instead. The decisive reason is that `ToolNode` dispatches every tool
  call present in a message regardless of count, so it contributes nothing
  toward the one-call-per-turn rule, which must be enforced at parse time
  regardless of which node performs dispatch. The boundary is now stated by
  intent rather than tied to one framework class. Import analysis alone is
  insufficient proof, because a backend can reach the dispatch node by
  injection rather than import — as `workflow.py`'s existing `RunCheck`
  callable already does, at `cli.py:154`. See §9 for the full control this
  requires. The P0 severity of a direct backend binding is unchanged.
- V2 replaces the legacy “provider tool use is a failure” behavior with an
  explicit native-tool-call protocol. The Claude adapter parses a provider tool
  call into a strict registered-tool request; the replay adapter emits the same
  `AIMessage.tool_calls` shape. Unknown, malformed, or mixed text/tool output
  produces a deterministic safe failure or consumes the one output-repair slot,
  as defined by the model contract.
- At most two diagnostic checks execute per incident, including runbook
  retrieval. Preserve the Phase-2 four-call model budget, including at most one
  structured-output repair.
- A model turn may propose **exactly one** tool call. Before the dispatch node
  calls the backend, the wrapper atomically creates a `PENDING` tool receipt
  and reserves one remaining check. A second call, a call after reservation
  failure, or a duplicated proposal fingerprint is denied without reaching a
  backend. This prevents any dispatch path from exceeding the two-check
  budget.
- Every transition has a timeout/error route to a deterministic safe report.
- Graph state is JSON-serializable and is a projection of existing domain
  records, not a second domain model. Persist `thread_id`, incident ID, tool
  receipts, evidence records, budget state, phase, interrupt state, and
  immutable `run_id`.

  *Amendment, Unit 1b:* the original wording said "evidence IDs." A node
  communicates with the rest of the graph only through state, and rendering
  the next model turn's context needs each record's summary, payload, kind,
  source, and timestamp, not just its ID. State therefore carries the full
  evidence record, from which the ID is a trivial projection.
- SQLite stores graph checkpoints and approval/audit records only. Existing
  JSONL evidence and results remain the canonical investigation artifacts.

### Durable-operation rules

Every externally observable operation has a deterministic idempotency key:

- Model request: `run_id + graph_phase + model_turn`. Persist a `PENDING`
  request record before sending it. A timeout, crash, or missing provider usage
  never reissues that key; it produces `FAILED_SAFE` with the reservation left
  visible for accounting.
- Tool call: normalized proposal fingerprint. Its `PENDING` receipt reserves
  budget before dispatch; the result updates that receipt exactly once.
- Approval: `thread_id + proposal_fingerprint + checkpoint_id`. Store one
  append-only owner decision before graph resume; an identical retry returns the
  existing decision and a conflicting retry is rejected.

An interrupt node must be side-effect-free before calling `interrupt()`. Any
write occurs in the idempotent approval-record path after resumption. Tests must
cover process termination before and after each pending/settled transition.

## 6. Existing evidence and tool contracts

Keep the existing four tool contracts and their policy constraints:

| Tool | Allowed typed input | Existing backend |
|---|---|---|
| `query_metric` | Template enum, service, bounded window | Prometheus |
| `query_logs` | Filter-template enum, service, bounded window, row limit | Active-run JSONL |
| `list_recent_changes` | Service, bounded window | Change manifest |
| `get_topology` | Active incident ID | Topology manifest |

The model must never submit raw PromQL, log predicates, shell, SQL, URLs,
paths, code, infrastructure manifests, unknown fields, or cross-incident IDs.
Policy continues to reject and record unknown templates, out-of-scope services,
cross-incident access, forged citations, duplicated checks, exhausted budgets,
and any mutation request.

Every tool result must carry, directly or through its existing receipt:

- immutable evidence ID and incident ID;
- source kind, observed timestamp, bounded content/summary, and content hash;
- typed arguments, policy decision, outcome, duration, and stable reason code;
- a success, timeout, unavailable, or bounded-error status.

Evidence is deterministically ordered, bounded, and marked when truncated.
Never persist chain-of-thought, secrets, provider thinking blocks, or evaluator
ground truth.

An `Evidence` record is an incident-scoped observation and may support a root
cause or satisfy an evaluator-only evidence predicate. A `RunbookPassage` is
guidance and is a distinct contract; it may support a suggested next check but
can never prove an incident cause or satisfy an incident-evidence predicate.

## 7. Optional RAG

`search_runbooks(query, limit)` is a fifth read-only tool over a small,
curated, application-visible synthetic runbook corpus. The corpus must exclude
expected answers, evaluator-only predicates, semantic scenario keys, secrets,
and controller-only instructions.

The default mode is SQLite FTS5 and must be called **lexical retrieval**.
Pinecone Starter may be enabled as an opt-in semantic-retrieval experiment only
after the local project is complete. The project must run, test, and demo
without it.

Both implementations conform to one retriever interface and return:

```text
passage_id, content, source_version, content_hash, score, retrieval_mode
```

`retrieval_mode` is `disabled`, `fts5_lexical`, or `pinecone_semantic`. The
CLI report, audit record, and evaluation record must surface this value. Never
silently fall back, mix modes in one benchmark aggregate, or represent FTS5 as
semantic retrieval. `disabled` means no runbook passage was retrieved.

Retrieved telemetry and runbook text is untrusted data. Prompts isolate it as
quoted evidence. It cannot alter system policy, register tools, extend scope,
or influence a deterministic authorization decision.

The final assessment stores incident-evidence citations separately from
runbook-guidance citations. Citation validity and citation sufficiency scoring
resolve only the incident-evidence citations.

## 8. Final assessment and human escalation

The structured final assessment contains only:

- `DIAGNOSED` with an allowed root-cause code and cited supporting/contrary
  evidence IDs;
- `INSUFFICIENT_EVIDENCE` with `UNDETERMINED`, missing evidence, and a concise
  owner next step.

`FAILED_SAFE` is application-generated only. It covers invalid output,
provider/tool failure, denied unsafe behavior without a safe alternative, or
exhausted budget. The model can never select it.

An escalation interrupt occurs only when deterministic code records one of
these reasons: `CONFLICTING_EVIDENCE`, `TOOL_UNAVAILABLE`,
`INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING`, or
`RETRIEVAL_COVERAGE_INSUFFICIENT`. V2 does not use an uncalibrated model
confidence threshold as an authorization input.

The interrupt payload contains the immutable `thread_id`, `checkpoint_id`,
`run_id`, escalation reason, current evidence IDs, remaining-check count, and
the one policy-approved next-check proposal fingerprint when such a check
exists. The owner can:

- accept the diagnosis or abstention;
- reject it and stop the investigation; or
- approve one additional already-authorized check when remaining budget allows.

CLI persistence and resume are sufficient for v2, for example:

```text
causalops approve <thread-id> <proposal-fingerprint>
causalops reject <thread-id> <reason>
```

`approve` re-resolves the checkpoint and proposal fingerprint, verifies the
append-only approval record and remaining reservation, then routes to
`DISPATCH_TOOL`. `reject` and acceptance route directly to `FINAL_REPORT`. A stale,
changed, expired, or already-settled fingerprint is rejected and never resumes
a tool. Approval/denial tests cover each route.

v2 has **no remediation executor**. It may record an owner-approved suggested
next step, but it must not claim to execute, verify recovery from, or remediate
an incident. A future audit-only simulated action requires a separate approved
specification amendment.

## 9. Safety and threat-model requirements

Required controls and tests include:

| Threat | Required v2 control |
|---|---|
| Tool-policy bypass | Test that no tool backend is reachable except through a policy wrapper: an AST import test proves the dispatch node imports no backend module directly, a wrapper-identity test proves every registered dispatch callable is wrapper-produced (not merely un-imported), and a spy-backend test proves a denied proposal never invokes it |
| Ground-truth leakage | Separate app-visible and evaluator-only fixture paths; assert model/retrieval context excludes labels and predicates |
| Prompt injection | Tag telemetry/runbooks untrusted; use adversarial fixtures that request both forbidden and seemingly permitted actions |
| Scope escape | Incident-scoped allowlists and templates; deny cross-run IDs, services, time windows, and evidence |
| Forged citations | Re-resolve every cited ID from the active incident store |
| Resource exhaustion | Preserve model/tool/time/row/byte/context caps and safe timeout outcomes |
| Provider/secret leakage | Environment-only key; bounded synthetic context; no secret or thinking persistence |
| Retrieval degradation | Record retrieval mode; require abstention/escalation on insufficient coverage |

## 10. Evaluation and publication rules

Keep the existing development/held-out split, opaque IDs, evaluator-only
expected outcomes, required-evidence predicates, replay conformance, and
deterministic scorer.

Evaluate three distinct purposes:

1. **Replay conformance:** validates graph transitions, policy wrappers, tool
   dispatch, checkpoints, interrupts, citations, and reports. It proves
   system behavior, not model quality.
2. **Paired live comparison:** same model and same answer-neutral initial alert
   compare a no-tool baseline against the tool-enabled LangGraph workflow.
   Preserve identical model, initial packet, budgets, taxonomy, and safe prompt
   constraints wherever applicable.
3. **Optional retrieval ablation:** compare FTS5 lexical, Pinecone semantic, or
   retrieval-disabled behavior in separately labelled runs only.

The live comparison is a predefined paired set of at most six held-out
incidents: one no-tool baseline and one tool-enabled run per incident. It must
not invoke the escalation path; HITL is demonstrated and tested separately.
`LIVE_EVALUATION_MAX_USD=2.00` is an application-wide hard ceiling. Before each
provider request, persist a conservative reservation using the pricing snapshot
and the request's bounded input/output allowance. Do not send a request that
would exceed the remaining ceiling. Provider-reported usage settles the
reservation; ambiguous requests retain it and are never repeated.

Mechanical scores remain:

- diagnosis and disposition correctness against evaluator-only labels;
- citation validity and citation sufficiency against required-evidence
  predicates;
- policy/control behavior, tool count, model-call count, latency, and cost.

Record Git SHA, clean/dirty status, fixture/prompt/policy/tool versions,
retrieval mode/corpus version, exact model, tokens, latency, cost, and raw
artifact references. Include the pricing source/date and configured ceiling.
Report counts and ranges for small samples; do not report p95 or broad
performance claims from a small synthetic benchmark.

## 11. Non-goals for v2

- Rebuilding the Docker lab, telemetry backends, scenario controller, scorer,
  or existing CLI.
- FastAPI/React UI, Kubernetes, cloud hosting, Terraform, Redis, PostgreSQL,
  OpenTelemetry platform integration, or multi-agent personas.
- LangSmith tracing and hosted evaluation are prohibited. The `langsmith`
  package may be present as an inert transitive dependency of
  `langchain-core`, provided tracing is force-disabled at the entry point
  and a test proves no tracing client is constructed and no tracing request
  is attempted.
- Real Spark/Delta/OCI/GCP data or APIs.
- Autonomous or mutating remediation, external tickets, shell execution, or
  writing to production-like systems.
- A required Pinecone account or external service for tests and local demos.

## 12. Three release milestones

Each milestone may contain small owner-approved work units, but full dual
review is mandatory at its trust-boundary completion snapshot and at final
release. This retains owner control without turning every local refactor into
a separate portfolio milestone.

1. **Bounded tool-graph parity.** Amend the product contracts; add seam tests;
   run one replay incident through `StateGraph`, native tool-call parsing, one
   policy wrapper, atomic budget reservation, and the existing report/scorer.
   Then wrap all four tools and retire duplicate orchestration only after
   conformance parity.
2. **Durable escalation and local retrieval.** Add checkpoint/operation IDs,
   CLI interrupt resume, approval routing, and crash/idempotency tests. Add
   curated FTS5 runbooks, retrieval provenance, and injection/no-ground-truth
   leakage tests. Pinecone remains a post-milestone optional experiment.
3. **Evidence-backed portfolio release.** Run the fixed paired evaluation under
   the USD 2 cap, save raw records and limitations, produce architecture and
   threat-model documents, verify the clean source commit, and record a short
   diagnosis plus abstention/escalation demo.

Every work unit must preserve a runnable, tested existing path. Do not begin a
later milestone while a P0/P1 finding, owner disposition, or regression remains
open.

## 13. v2 completion criteria

CausalOps v2 is complete when the owner can run an existing synthetic incident
through the CLI, observe actual policy-wrapped LangGraph tool calls, inspect
cited evidence and provenance, resume a real escalation interrupt, receive a
safe diagnosis/abstention/failure report, and reproduce conformance plus a
clearly labelled paired evaluation. The README and demo must state that all
systems and data are synthetic and that CausalOps is decision support.
