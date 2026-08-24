# CausalOps: Technical Overview

> **Subtitle:** Evidence-Grounded Incident Investigator
> **Status:** as-built record, written continuously — one section per unit,
> landing in the same commit as the code it describes
> **Audience:** project owner, contributors, reviewers, and coding agents

## Vocabulary: phase, milestone, unit, section

This project uses four different grouping words across its documents. They
mean different things and must not be swapped:

- **Phase / step** — the v1 delivery history recorded in Part I below. Three
  phases, each of three steps, each step landing as one commit or a short
  branch merged as one. This vocabulary describes the past; no new phase is
  planned.
- **Milestone** — one of the three v2 releases defined in `TECHNICAL_SPEC.md`
  §12. A milestone is large: it closes only after a full owner and
  dual-reviewer trust-boundary review. Part III tracks milestone progress.
- **Unit** — one bounded, independently reviewable chunk of work inside a
  milestone, following the owner-controlled review protocol in `CLAUDE.md`.
  This document itself landed as Unit 0.
- **Section** — one `##`/`###` heading in *this* document, referenced by name
  ("the CLI contract section") rather than by number, because headings move
  as the document grows. Numbered references (`§9`) belong only to
  `TECHNICAL_SPEC.md`, which is a fixed contract that does not get
  restructured the same way.

Phases and milestones happen to both number three; that overlap is
coincidental, not structural, and units and sections do not follow a fixed
count. Use "phase" only for v1 history, "milestone" only for a v2 release,
and "section" only for a heading in this document; never call a unit a step,
or a milestone a phase.

## Document authority

`TECHNICAL_SPEC.md` governs product decisions for CausalOps v2. This file
records what is built and how — phase by phase for v1, unit by unit for
v2 — and must never contradict it. `CLAUDE.md`, not `AGENTS.md` (which no
longer exists in this repository), is the live source of truth for
contributor and coding-agent behavior. `CLAUDE.md` is gitignored by project
design — contributor instructions stay local — so it will not exist in a
fresh clone; this mirrors how this document previously cited the now-absent
`AGENTS.md` for the same reason.

README files, ADRs, issues, comments, and generated reports may explain or
record implementation detail but cannot redefine either document. An
approved product decision updates `TECHNICAL_SPEC.md`; an approved delivery
updates this file, in the same commit as the code it describes.

## Plain glossary

Use ordinary words unless one of these contract terms needs its exact meaning:

- **Opaque:** random-looking and carrying no clue about the incident cause.
- **Bounded:** limited by fixed time, count, token, row, or byte caps.
- **Incident:** one limited period of faulty lab behavior with an opaque ID.
- **Evidence:** a stored observation tied to one incident that can be cited.
- **Check:** one approved read-only tool execution used to gather evidence.
- **Answer-neutral:** describing the initial symptom and scope without hinting
  at the cause.
- **Proposal fingerprint:** a stable hash of a tool name and its typed
  arguments, used to detect the same proposal twice.
- **Required-evidence predicate:** an evaluator-only rule that checks whether
  cited evidence contains the observable fact needed to support an answer.
- **Structured-output repair:** the one allowed retry that asks the same model
  stage to correct a response that did not match its required schema.
- **Root cause:** one of the allowed cause labels for an incident.
- **Disposition:** the final result: diagnosis, abstention, or safe failure.
- **Abstention:** a valid decision that the evidence cannot distinguish a
  cause.
- **Conformance:** repeatable checks that show workflow and safety rules work;
  it is not a measure of model accuracy.
- **Scenario family:** an evaluator-visible kind of synthetic incident.
- **Run key:** the unique family, system, and repetition within an evaluation.

# Part I — As built

Every claim below cites a commit SHA, the source file it landed in, and the
tests that prove it, per `CLAUDE.md`'s evidence rule: do not claim a
component is implemented until source, tests, and commit SHA demonstrate it.

The original v1 plan was a three-phase, nine-step MVP (see §14 "Three-phase
delivery sequence" in `git show b6f4d9c:TECHNICAL_OVERVIEW.md`, the tracked
commit before this rewrite — the local, gitignored `TECHNICAL_OVERVIEW_OLD.md`
is a convenience copy of that same commit, not itself authoritative). Two
phases shipped in full; the third — the live Claude adapter and its
benchmark — was never started. That is not a gap in this document; it is the
actual state of the repository.

## Phase 1 — foundation and first vertical slice

### Step 1 — package foundation and `doctor`

**Commit:** `8c090dc` — Add package foundation and causalops doctor

What landed: `pyproject.toml`/`uv.lock` packaging with Ruff, strict mypy, and
pytest configured; `causalops doctor` checking Windows 11 by build number,
total RAM (fails below 7.5 GiB), available RAM (warns below 2.5 GiB), free
disk (fails below 12 GB), Docker, and `ANTHROPIC_API_KEY` presence; CI
running one `windows-latest` job (`mypy src`, not `mypy src lab`, since
`lab/` did not exist yet). The `ubuntu-latest` matrix and Linux support
arrive in step 3 (`4112022`).

Source: `src/causalops/doctor.py`, `src/causalops/system_probe.py`,
`src/causalops/cli.py` (`doctor` subcommand), `.github/workflows/ci.yml`.

Tests: `tests/unit/test_doctor.py`, `tests/unit/fake_machine.py`,
`tests/unit/test_cli.py`.

Known limitation: `doctor` does not yet make the authenticated
`claude-sonnet-5` metadata request the original spec called for.
`src/causalops/cli.py:33-38` carries an explicit `MODEL_CHECK_NOTE` marking
this unbuilt: *"Not checked yet: the authenticated claude-sonnet-5 metadata
request arrives in a later step."* It arrives with the live Claude adapter,
not yet scheduled to a specific v2 unit.

### Step 2 — investigation core

**Commit:** `8a9d4fb` — Add investigation core: replay loop, policy, budgets,
records, scoring

What landed: the domain types (`IncidentScope`, `Evidence`, `Hypothesis`,
`ToolProposal`, `ToolReceipt`, `InitialPlan`, `HypothesisUpdate`,
`FinalAssessment`, `InvestigationReport`), `ReplayReasoningModel`, policy
authorization, budget enforcement, JSONL run records, the three-call
investigation loop, and deterministic scoring. This step has no lab
dependency — it landed and was tested before the synthetic lab existed.

Source: `src/causalops/domain.py`, `models.py`, `policy.py`, `prompts.py`,
`run_records.py`, `tools.py`, `workflow.py`, `evidence.py`, `evaluation.py`.

Tests: `tests/unit/test_domain.py`, `test_policy.py`, `test_workflow.py`,
`test_run_records.py`, `test_replay_model.py`, `test_evidence.py`,
`test_evaluation.py`, `test_tools.py`, `test_prompts.py`. Ground-truth
isolation tests also landed in this commit:
`tests/security/test_ground_truth_isolation.py` (7 tests at this commit,
grown to 9 by the end of Phase 2).

Known limitation: `ReplayReasoningModel` is the only reasoning-model adapter
in the codebase. There is no live model call anywhere in `src/`.

### Step 3 — synthetic lab, Prometheus, tool backends, CLI wiring

**Commit:** `ae0d226` — Add synthetic lab, Prometheus, tool backends and CLI
wiring, on a short-lived branch that also contains `4112022` (Windows/Linux
support in `doctor`), `f800fa8` (a documentation-only correction), and
`fa79385` (scenario cleanup, Linux test-parsing, and metric-readability
fixes). The branch merged to `master` as `eab3609` — Merge synthetic lab,
Prometheus wiring, and platform support.

What landed: the Docker Compose lab (`gateway → orders → inventory`), the
Prometheus container and adapter, all four read-only tool backends
(`query_metric`, `query_logs`, `list_recent_changes`, `get_topology`) wired
end to end for one incident family (`configuration_change`), and the
`causalops lab up/down`, `scenario start/reset`, and
`investigate --model replay` CLI commands. The bounded Python resource pool
in `orders` is not part of this commit — `git log -S POOL_CAPACITY` places
it in Phase 2 step 2 (`23842ab`), where it's already credited correctly.

Source: `lab/docker-compose.yml`, `lab/prometheus.yml`,
`lab/services/{gateway,orders,inventory,service}.py`,
`src/causalops/prometheus.py`, `src/causalops/telemetry.py`,
`src/causalops/scenario_control.py`, `src/causalops/cli.py`,
`src/causalops/report.py`.

Tests: `tests/unit/test_compose.py`, `test_lab_services.py`,
`test_telemetry.py`, `test_scenario_control.py`, `test_report.py`;
`tests/integration/test_configuration_change.py`.

Known limitation: only `configuration_change` worked end to end at this
point. The other three families landed in Phase 2 step 2.

## Phase 2 — complete lab and safety

### Step 1 — policy invariants and cross-incident isolation

**Commit:** `c5482b2` — Enforce policy decision invariant and prove
cross-incident isolation, merged as `e58ef55`.

What landed: an invariant that every `policy.authorize` path returns one
consistent typed decision (no code path can silently allow without an
explicit `PolicyDecision`), and cross-incident isolation proof for the
telemetry backends — a query scoped to incident A cannot return incident B's
rows, even when both runs exist on disk at once.

Source: `src/causalops/policy.py`.

Tests: `tests/unit/test_policy.py`; `tests/unit/test_telemetry.py` (+219 lines
in this commit to add the isolation cases).

### Step 2 — remaining incident families

**Commit:** `23842ab` — Add remaining incident families with seed-based fault
variation, merged as `4e11124`.

What landed: the three remaining incident families —
`downstream_timeout_retry_amplification`, `resource_pool_saturation`, and
`ambiguous_telemetry` — each with development and evaluation seeds and
seed-based deterministic variation in timestamps, request IDs, affected
endpoint, and noise logs.

Source: `lab/scenarios/{downstream_timeout_retry_amplification,
resource_pool_saturation,ambiguous_telemetry}.json`,
`lab/scenarios/configuration_change.json` (extended for parity),
`lab/services/{inventory,orders,service}.py`,
`src/causalops/scenario_control.py`.

Tests: `tests/integration/test_incident_families.py`,
`tests/unit/test_scenario_control.py` (grown to 358 lines total),
`tests/unit/test_lab_services.py`.

### Step 3 — security and conformance hardening

**Commit:** `d73f67b` — Close security-conformance gaps: injection, forgery,
timeout, reset, merged as `b6f4d9c` (current `HEAD`).

What landed: prompt-injection resistance tests (untrusted telemetry text
proven inert, including against a model instructed to obey it), replay
fixtures exercising forged citations and duplicate proposal fingerprints,
out-of-scope service denial, and scenario-reset isolation proof (a reset
cannot delete a finalized result).

Source: `src/causalops/replay_fixtures/{forged_citation,duplicate_proposal,
service_out_of_scope}.json`.

Tests: `tests/security/test_prompt_injection.py` (2 tests),
`tests/integration/test_scenario_reset_isolation.py`,
`tests/unit/test_policy.py`, `tests/unit/test_telemetry.py` (536 lines
total), `tests/unit/test_workflow.py`.

At `HEAD` (`b6f4d9c`), all four CI gates pass:
`uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src lab`,
and `uv run pytest -m "not docker"` — **237 passed, 5 docker-marked
deselected** (docker-marked tests require `causalops lab up` and are run by
hand, not in CI).

## Phase 3 — never started

**Superseded, and partly overtaken:** the bullets below record the state at
the time of this audit. Milestone 3 has since landed a live adapter,
`--model claude`, and a cost ledger (`ecbf174`, merged `9ca9c2b`) — see
Part III. Only the paired evaluation remains unbuilt.

Nothing under this heading has landed. No commit, source file, or test
implements any of it. Specifically absent at the time of this audit:

- A live model adapter (`ClaudeReasoningModel` or equivalent). The only
  reasoning-model *adapter* class in `models.py` is `ReplayReasoningModel`; a
  repository-wide search for `tool_use` or `PROVIDER_` finds no live-adapter
  code. (`stop_reason` also returns no live-adapter hits, but is a poor
  search term on its own — it is a legitimate existing field on
  `InitialPlan` and `HypothesisUpdate` (`domain.py:246,262`), appearing in
  seven of the eight replay fixtures and five files under `tests/` for
  reasons unrelated to a live provider.)
- The authenticated `claude-sonnet-5` doctor metadata check (`cli.py:33-38`
  documents this gap explicitly, see Phase 1 step 1 above).
- `causalops investigate --model claude` — the CLI's `--model` flag accepts
  only `replay` (`cli.py:71`, `choices=("replay",)`).
- `causalops conformance` and `causalops benchmark` — neither subcommand is
  registered in `cli.py`'s argument parser.
- Any cost ledger, token-counting call, or live evaluation record.
  `evaluation.py` contains the deterministic scorer (`score_run`,
  `MechanicalScores`, `EvaluationRecord`) but no benchmark orchestrator and
  no baseline-vs-workflow paired runner.
- Any portfolio artifact tied to a live run — a recorded demo or a
  threat-model document scoped to Phase 3 — since no live run has ever
  executed. (README.md was rewritten in this unit for documentation
  accuracy, not as a live-run portfolio artifact.)

Phase 3 was superseded by `TECHNICAL_SPEC.md` §12's three v2 milestones
before any of it was built, and it will not be completed in its original
form. See Part III, "Superseded v1 evaluation design," for exactly what
changed and why.

# Part II — Architecture and contracts

This part is the evergreen engineering reference: current domain types,
tools, policy, budgets, and the parts of the threat model already provable by
a Phase 1/2 test. It changes when the underlying contract changes, not on
every commit.

## Product thesis

CausalOps is an incident investigation assistant for a synthetic Python
microservice system that runs locally and uses one hosted reasoning model. It
keeps several possible causes, gathers a limited amount of evidence that can
support or rule out each one, and runs safe read-only checks. It returns
either a diagnosis with evidence the owner can check or a clear statement
that the evidence is not enough.

"Causal" refers to a practical loop—hypothesis, diagnostic check, evidence
update—not formal causal inference. CausalOps does not build causal graphs
or estimate what would happen under an intervention.

It is decision support for an on-call engineer. It is not an autonomous
operator or a production SRE platform. Its central trust boundary is:

> The model proposes and interprets. Deterministic code validates,
> authorizes, executes read-only checks, stops, scores, and records.

## Architecture and trust boundaries

```text
Evaluator-only manifest              Investigator-visible data
-----------------------              -------------------------
scenario family                       opaque incident ID
development/evaluation seed           immutable alert packet
expected RootCauseCode                scoped logs and metrics
required evidence predicates          topology and change events
        |                                       |
        v                                       v
Scenario controller -> synthetic lab -> registered evidence tools
        |                                       |
        |                              schema validation + policy
        |                                       |
        +---------------------> scorer <--------+
                                     |
                          JSONL records + Markdown report
```

### Synthetic lab

Docker Compose runs these project-authored Python 3.12 services:

```text
gateway -> orders -> inventory
              |
       bounded resource pool
```

The `orders` resource pool is implemented in Python and can be saturated by a
scenario. PostgreSQL is not part of the lab. A Prometheus container provides
historical metric queries. Claude is accessed through Anthropic's hosted API
once the live adapter exists (Part I, Phase 3). "Python-only" means all
authored services, orchestration, tools, policy, and evaluation code are
Python; Docker and Prometheus are local infrastructure dependencies.

### Supported development platforms

CausalOps supports two owner environments, each with at least 7.5 GiB detected
total RAM and 12 GB free disk:

- **Windows 11.** Python, `uv`, Git, and Claude Code run natively from
  PowerShell. Docker Desktop uses its WSL2 backend only for the gateway,
  orders, inventory, and Prometheus containers; WSL is not the project shell.
- **Linux x86-64.** The same tools run from the system shell, and Docker
  Engine runs the same four containers directly.

Docker is an owner-installed prerequisite on both; project automation does not
install or upgrade it.

Project-authored code uses `pathlib.Path`, UTF-8, injected UTC timestamps, and
cross-platform Python APIs. Project commands and documentation must not depend
on Bash, hard-coded POSIX paths, or shell behavior unavailable in PowerShell.
Run one scenario and one Claude request at a time. Prometheus retains one hour
of data. Warn when current available RAM is below 2.5 GiB, but keep that check
advisory because available memory changes while the system runs.

Use these approximate container memory ceilings, set in
`lab/docker-compose.yml`:

| Container | Memory ceiling |
|---|---:|
| Gateway | 128 MB |
| Orders | 192 MB |
| Inventory | 128 MB |
| Prometheus | 256 MB |

### Telemetry

- Every service exposes `/metrics`; Prometheus scrapes and retains time-series
  samples labelled with the opaque incident ID.
- Each service writes structured logs to
  `runs/<incident-id>/logs/<service>.jsonl` on a mounted volume.
- Logs include injected timestamps, request IDs, service, severity, event
  code, and bounded structured fields.
- The scenario controller writes topology and recent-change manifests under
  the same opaque run directory. Filenames and contents must not identify the
  semantic scenario or expected cause.
- Metric and log adapters can query only the active incident ID and time
  window. Data from prior runs is never returned.

The `runs/<incident-id>` tree is transient lab state. Every completed
standalone investigation gets a separate opaque investigation ID and
atomically finalizes its cited evidence, tool receipts, run record, and
report under `results/investigations/<investigation-id>`. Finalized result
bytes are immutable. A correction creates a new investigation ID rather than
rewriting an existing result. (Benchmark-scoped `results/<evaluation-id>`
artifacts are specified but not yet produced — no benchmark orchestrator
exists; see Part I, Phase 3.)

### Scenario controller

The scenario controller is a separate trust domain. It may start, fault,
verify, and reset the synthetic lab before or after an investigation. Its
mutation operations are never registered as investigator tools, included in
model tool descriptions, or callable through model output.

Starting a scenario:

1. Accepts an owner-visible family and seed.
2. Creates an opaque UUID incident ID and run directory.
3. Starts or reconfigures the lab using controller-only mechanisms.
4. Verifies the healthy baseline and then the expected fault signal.
5. Records the incident window and creates an answer-neutral alert packet.
6. Stores expected outcomes in an evaluator-only manifest.
7. Returns only the opaque incident ID to the investigation command.

Reset removes only active lab and transient state under the matching
`runs/<incident-id>` tree, restarts affected services if necessary, and must
verify healthy behavior before another scenario begins. It cannot delete or
modify finalized evidence, records, or reports under `results/` — proven by
`tests/integration/test_scenario_reset_isolation.py`.

### Logical ground-truth isolation

Ground truth is open-source test metadata, not a filesystem secret. Isolation
means the investigator process receives no evaluator manifest, semantic
scenario key, expected outcome, required-evidence predicate, or answer-bearing
path. `tests/security/test_ground_truth_isolation.py` proves investigator
packages do not import evaluator-only modules and that model contexts contain
none of those values.

### Repository shape

```text
src/causalops/
  cli.py
  domain.py
  graph.py
  prompts.py
  models.py
  tools.py
  tool_calls.py
  tool_wrappers.py
  policy.py
  evidence.py
  prometheus.py
  telemetry.py
  run_records.py
  scenario_control.py
  report.py
  evaluation.py
  doctor.py
  system_probe.py
  replay_fixtures/
lab/
  docker-compose.yml
  prometheus.yml
  scenarios/
  services/
tests/
  unit/
  integration/
  security/
results/
```

Milestone 1 added `graph.py`, `tool_calls.py`, and `tool_wrappers.py` to
`src/causalops/`, and Unit 1d-2 removed `workflow.py` once the graph
orchestrator it ran beside had proven conformance parity with it. Create new
directories only when an implemented vertical slice needs them.

## Incident identity and initial evidence

`IncidentScope` contains only:

- Opaque incident ID.
- Environment identifier fixed to `local-lab`.
- Allowlisted services.
- Start and end timestamps.
- Affected public endpoint.

It must not contain a semantic scenario name, seed label, or expected cause.
Replay fixture keys and model-visible evidence IDs are also opaque.

The scenario controller creates one immutable `InitialAlertPacket` shared byte
for byte by the baseline and CausalOps workflow. It contains:

- Opaque incident ID and bounded time window.
- Affected public endpoint.
- Coarse gateway symptom: elevated errors, elevated latency, or both.
- Coarse service topology without health conclusions.
- Stable alert timestamp and alert-source version.

The coarse symptom and topology are represented as immutable, incident-scoped
`Evidence` records with opaque IDs included in the packet. This lets both the
baseline and workflow cite their initial evidence mechanically.

Initial alert construction is deterministic and does not count as an
investigator tool call. It contains no recent changes, detailed downstream
metrics, diagnostic logs, or root-cause-specific language.

## Investigation workflow and budgets

This is the current, implemented LangGraph orchestrator in
`src/causalops/graph.py` (`GraphPhase`, tracked in Part III). Milestone 1
built it beside a loop orchestrator in `src/causalops/workflow.py`, ran both
against the same incidents, and retired the loop once a 144-pair differential
sweep demonstrated conformance parity between the two — `workflow.py` is
gone as of Unit 1d-2.

```text
CREATED
  -> INVESTIGATE
  -> DISPATCH_TOOL -> NORMALIZE_EVIDENCE -> INVESTIGATE   (repeats once more,
                                                            budget permitting)
  -> FINAL_ASSESSMENT
  -> FINAL_REPORT
  -> DIAGNOSED | INSUFFICIENT_EVIDENCE | FAILED_SAFE
```

`INVESTIGATE` routes to `DISPATCH_TOOL` when the model proposes a tool check,
straight to `FINAL_ASSESSMENT` when it stops early, or to `FINAL_REPORT` on
an unrecoverable failure. After `NORMALIZE_EVIDENCE`, `route_after_normalize`
checks `failure_reason` first — any failure set upstream (by `investigate` or
`dispatch_tool`) routes straight to `FINAL_REPORT`, skipping a second turn
entirely — and only then applies the loop-back condition: the graph returns
to `INVESTIGATE` for a second turn only while `model_turn < 2` and budget
remains, otherwise it moves on to `FINAL_ASSESSMENT`. This is `build_graph`'s
own edge set (`graph.py`), not a paraphrase of it.

The second check may be skipped when the update concludes that available
evidence is already sufficient or no useful safe check remains.

### Model-call contract

Normal execution requires three valid structured responses:

1. `InitialPlan`: two or three hypotheses plus the first tool proposal or a
   justified stop.
2. `HypothesisUpdate`: revised ranks plus a second tool proposal or stop.
3. `FinalAssessment`: diagnosis or abstention with supporting and contrary
   evidence references.

The fourth and final model-call slot is reserved for one repair of an invalid
structured response at any stage. The repair reissues the same stage with
validation errors. A repair consumes the ordinary model-call budget. If a
repair, denial, timeout, or early stop leaves insufficient budget for another
diagnostic check and final assessment, the workflow skips the remaining check
and produces the safest valid terminal result. If it cannot obtain a valid
final assessment, it returns `FAILED_SAFE`.

At most two diagnostic tools may execute. A denied or invalid proposal consumes
the reasoning iteration and model call that produced it, but it does not count
as an executed tool. Policy denial is not automatically terminal when the
remaining call budget permits a safe alternative or assessment.

### Default limits

| Limit | Default | Status |
|---|---:|---|
| Investigation wall clock | 360 seconds | built — `Budgets.wall_clock_seconds` |
| Model call | 90 seconds | built, Unit 3b-2 — `pricing.MAX_REQUEST_SECONDS`, passed as `live_model.LiveClaudeModel`'s `ChatAnthropic(..., timeout=...)` |
| Tool execution | 10 seconds | built — `Budgets.tool_timeout_seconds` |
| Model calls, including repair | 4 | built — `Budgets.model_calls` |
| Executed diagnostic tools | 2 | built — `Budgets.executed_tools` |
| Structured-output repairs | 1 | built — `Budgets.repairs` |
| Maximum counted input per model call | 9,600 tokens (Unit 3b-2/3b-3; prose only, not the tool schema) | built — `pricing.MAX_INPUT_TOKENS`/`estimate_input_tokens`/`InputTooLarge`. See "The smoke call's findings" below for the replan, the measured figures, and why the tool schema stays out of this cap |
| Claude `max_tokens` | 1,600 tokens | built, Unit 3b-2 — `pricing.MAX_OUTPUT_TOKENS`, `live_model.LiveClaudeModel`'s `ChatAnthropic` construction |
| Claude adaptive-thinking effort | `medium` | built, Unit 3b-2 — `live_model.LiveClaudeModel`'s `ChatAnthropic(..., thinking={"type": "adaptive"}, effort="medium")` |
| Log result | 40 rows and 12 KB | built — `Budgets.log_rows`, `evidence.MAX_RESULT_BYTES` |
| Metric result | 60 samples and 12 KB | built — `prometheus.MAX_METRIC_SAMPLES`, `evidence.MAX_RESULT_BYTES` |
| Automatic retries | 0 | built — no retry logic exists anywhere in `src/` |

Every row above is enforced today by the cited constant. Cost caps are
recorded in `TECHNICAL_SPEC.md` §10 (superseding the v1 figures — see
Part III) rather than here, since they bound application-wide dollar
spend rather than one request's shape.

All limits are application-owned and visible to the model as immutable status.
Context construction uses an injected clock, stable evidence ordering, fixed
per-kind quotas, explicit truncation markers, and a digest of the final input.

### Terminal semantics

- `DIAGNOSED` requires a non-`UNDETERMINED` root-cause code supported by valid
  incident-scoped evidence.
- `INSUFFICIENT_EVIDENCE` requires `UNDETERMINED`. It means the workflow
  operated correctly, but bounded valid evidence cannot distinguish a cause.
- `FAILED_SAFE` requires `UNDETERMINED`. Only deterministic application code
  may produce it after malformed output, dependency timeout, internal error,
  invariant violation, disallowed behavior without a safe alternative, or
  operational-budget exhaustion. It is never offered as a model-selectable
  disposition.

Schema validation rejects every disposition/root-cause combination that
violates these invariants. The structured-output repair rule applies to an
invalid model combination.

Normal diagnostic-iteration exhaustion with unresolved hypotheses produces
`INSUFFICIENT_EVIDENCE`. A hard technical failure before a valid assessment
produces `FAILED_SAFE`.

Replay results are scripted determinism, not agent behavior. They must never
be reported as diagnostic accuracy, agent competence, or improvement over a
baseline — no such measurement is possible until Phase 3's live adapter
exists and a paired evaluation actually runs (Part III, Milestone 3).

## Public contracts

All model, tool, artifact, and CLI boundaries use Pydantic v2 models. Models
that are persisted or exchanged with the reasoning model also carry a schema
version.

### Deterministic enums

```text
RootCauseCode
  CONFIG_CHANGE
  DOWNSTREAM_TIMEOUT_RETRY_AMPLIFICATION
  RESOURCE_POOL_SATURATION
  UNDETERMINED

Disposition
  DIAGNOSED
  INSUFFICIENT_EVIDENCE
  FAILED_SAFE
```

### Domain types

- `IncidentScope`: opaque identity and allowlisted temporal/service scope.
- `InitialAlertPacket`: immutable answer-neutral input shared by both systems,
  including opaque IDs for its initial symptom and topology evidence.
- `Evidence`: opaque ID, incident ID, kind, source, observation time, bounded
  summary, structured payload, tool receipt, and content hash.
- `Hypothesis`: a possible cause that states what evidence could support or
  rule it out, its ordinal rank, supporting and contrary evidence IDs, and
  missing evidence. Rank is not a calibrated probability.
- `ToolProposal`: registered tool, typed arguments, evidence gap, and expected
  observation.
- `ToolReceipt`: normalized proposal fingerprint, policy result, timing,
  bounded result digest, outcome, and stable reason code.
- `InitialPlan`: hypotheses and first tool proposal or stop.
- `HypothesisUpdate`: revised hypotheses and second proposal or stop.
- `FinalAssessment`: a model-selected `DIAGNOSED` or `INSUFFICIENT_EVIDENCE`
  outcome, matching root-cause code, supporting and contrary evidence IDs,
  uncertainty, and proposed human next step. Its model schema excludes
  `FAILED_SAFE`.
- `InvestigationReport`: opaque investigation ID, validated assessment,
  budgets, latency, usage, versions, limitations, and artifact references.
  Application code may create a `FAILED_SAFE` report without a model
  `FinalAssessment`.
- `EvaluationRecord`: paired system/run identity, expected outcome, mechanical
  scores, reproducibility manifest, and raw artifact reference. The type
  exists in `evaluation.py`; nothing populates it yet, since no live paired
  run has ever executed.

Do not request or persist private chain-of-thought. Model responses contain
only structured decisions, short summaries, and evidence references.

## Investigator tools and policy

v1 implemented the first four read-only tools. Milestone 3's Unit 3a adds
the fifth, `search_runbooks`, over a small curated corpus -- optional in the
sense that a run may never dispatch it, not in the sense that it is unbuilt.

| Tool | Typed input and backend |
|---|---|
| `query_metric` | Registered PromQL template ID, service, and bounded window; executed against Prometheus |
| `query_logs` | Registered filter ID, service, bounded window, and row limit; scans active-run JSONL |
| `list_recent_changes` | Service and bounded window; reads the active-run change manifest |
| `get_topology` | Active incident ID; reads the active-run topology manifest |
| `search_runbooks` | Registered `RunbookTopic` and passage limit; queries an in-memory SQLite FTS5 index built from `runbook_corpus.json` |

The model selects registered template IDs and typed parameters. Application
code constructs PromQL and log predicates; for `search_runbooks`, application
code constructs the FTS5 `MATCH` query from the topic -- `query` is a closed
enum, not free text, so nothing about this fifth tool weakens the same claim
for the other four. The model cannot submit raw shell, SQL, PromQL, URLs,
paths, code, or infrastructure manifests.

Policy denies and records:

- Unknown tools, templates, fields, services, versions, or incident IDs.
- Cross-incident evidence or file access.
- Forged or nonexistent evidence references.
- Write or mutation requests.
- Duplicate or out-of-order proposals.
- Requests outside the incident window or result limits.
- Requests after a hard budget is exhausted.

Each tool returns a typed success, timeout, unavailable, or bounded-error
result. Tools do not retry automatically. Untrusted telemetry may influence
model reasoning, but it cannot register tools, expand scope, alter policy or
budgets, or cause a disallowed operation.

## Incident families and variants

Implement exactly four root-cause families:

1. **Configuration change:** an orders configuration change causes failures.
2. **Downstream timeout with retry amplification:** inventory latency causes
   retries and elevated gateway latency.
3. **Resource-pool saturation:** a bounded Python resource pool in orders is
   exhausted and degrades requests.
4. **Ambiguous telemetry:** differentiating evidence is absent or contradictory;
   the expected result is `UNDETERMINED` with `INSUFFICIENT_EVIDENCE`.

At least the timeout and resource-pool families begin with the same coarse
gateway-latency symptom. At least one diagnosed family includes a plausible but
irrelevant recent change, and every family includes bounded noise logs. The
correct cause must require at least one follow-up tool call; the initial alert
packet alone must not be sufficient.

Each family has:

- A development seed used while implementing and debugging.
- A held-out evaluation seed used by the scored comparison.
- Deterministic variation in timestamps, request IDs, affected endpoint,
  signal magnitude, and irrelevant logs.
- Setup, healthy assertion, fault assertion, reset, and cleanup verification.

The evaluator-only expected outcome includes a root-cause code, disposition,
and required-evidence predicates. A predicate describes observable source,
kind, registered template/filter, and structured condition. The scorer
resolves cited evidence IDs and evaluates these predicates; predicate names
and expected values never enter model context.

## CLI contract

Implemented today:

```powershell
causalops doctor
causalops lab up
causalops lab down
causalops scenario start <family> --seed <development|evaluation>
causalops scenario reset <incident-id>
causalops investigate <incident-id> --model replay
```

- `doctor` checks the operating system, at least 7.5 GiB detected total RAM,
  current available RAM, at least 12 GB free disk, required writable
  directories, Docker, and the presence of `ANTHROPIC_API_KEY`. It warns but
  does not fail when available RAM is below 2.5 GiB. It does not yet make an
  authenticated model-metadata request (Part I, Phase 1 step 1).
- `lab up` verifies Docker, Prometheus, service health, and required writable
  run directories.
- `scenario start` is owner-facing and may use a semantic family name. It
  prints an opaque incident ID; semantic identity is not passed to
  `investigate`.
- `investigate` accepts only an opaque incident ID, creates an opaque
  investigation ID, and finalizes JSONL evidence, receipts, a run record, and
  a Markdown report under `results/investigations/<investigation-id>`.
- `scenario reset` verifies healthy state and cross-run isolation. It deletes
  only active lab/transient state for that incident and never finalized
  records, reports, receipts, or cited evidence under `results/`.

Specified, not yet built:

```powershell
causalops investigate <incident-id> --model claude --max-cost-usd 0.15
causalops benchmark --model claude --variant evaluation --repetitions 3 --max-cost-usd 1.75
causalops conformance
```

None of these three commands is registered in `cli.py`'s parser. The
`benchmark` signature above is the original v1 design; it is superseded by
`TECHNICAL_SPEC.md` §10's paired evaluation (at most six held-out incidents,
USD 5.00 cap, Unit 3b-3) — see Part III, "Superseded v1 evaluation design,"
for what changed and why the numbers differ.

Business outcomes `DIAGNOSED` and `INSUFFICIENT_EVIDENCE` are successful CLI
executions. `FAILED_SAFE`, invalid configuration, and unavailable dependencies
return nonzero with stable machine-readable reason codes.

## Threat model and tests

Protected assets are incident scope, tool registry, policy, budgets, evidence
integrity, evaluator ground truth, secrets, and the host environment.
Attacker-controlled inputs include model output, logs, metric labels, alert
text, and change descriptions.

Rows below marked **built** are proven today by a cited test — Phase 1/2
tests for most rows, Unit 3b-2's for the three that needed the live Claude
adapter to exist first. Every row in this table is built; none is deferred.

| Threat | Required control | Status |
|---|---|---|
| Ground-truth leakage | Opaque model inputs; assert prompt/context lacks semantic scenario keys and expected values | built — `tests/security/test_ground_truth_isolation.py` |
| Prompt injection in telemetry | Untrusted delimiters plus deterministic policy; verify no scope, tool, policy, or budget expansion | built — `tests/security/test_prompt_injection.py` |
| Arbitrary query execution | Template enums only; reject raw PromQL, shell, SQL, URL, and path input | built — `tests/unit/test_policy.py`, `test_tools.py` |
| Scope escape | Incident-labelled backends and allowlists; deny cross-run service, time, file, and evidence access | built — `tests/unit/test_telemetry.py`, `test_policy.py` |
| Forged citations | Resolve opaque evidence IDs from active store; reject missing and cross-incident IDs | built — `tests/unit/test_policy.py`, `src/causalops/replay_fixtures/forged_citation.json` |
| Resource exhaustion | Enforce call, time, row, sample, and byte limits, and per-kind context quotas (`evidence.CONTEXT_QUOTAS`) — distinct from the token-counted input cap, see "Default limits" above | built — `tests/unit/test_graph.py`, `test_telemetry.py`; the input cap itself is `tests/unit/test_live_model.py`'s `test_an_oversized_request_refuses_before_reserving_or_sending` |
| Scenario contamination | Reset volumes/state and assert health and empty run scope before the next scenario | built — `tests/integration/test_scenario_reset_isolation.py` |
| Model/tool failure | Timeout and malformed-output fixtures produce deterministic terminal states | built — `tests/unit/test_graph.py`, `test_tool_wrappers.py` |
| Credential leakage | Environment-only API key plus redaction; verify it never reaches CLI text, config, artifacts, logs, reports, receipts, or errors | built, Unit 3b-2 — `live_model.py`'s `LiveClaudeModel` never reads the key itself (`ChatAnthropic()` resolves it internally); `tests/security/test_credential_isolation.py` proves the module neither imports `os` nor names the variable in code. Not exercised with a real key — every test uses a fake transport |
| Provider data leakage | Send only bounded synthetic incident context; verify requests exclude secrets, evaluator ground truth, and host paths | built, inherited — the live adapter sends exactly `ModelRequest.system_text`/`context_text`, the same rendered context `tests/security/test_ground_truth_isolation.py`/`test_prompt_injection.py` already constrain; Unit 3b-2 adds no new context source |
| Unbounded provider spend | Durably flushed write-ahead reservations plus settled/outstanding accounting; verify both caps, crash behavior, and no request after denial | built, Unit 3b-2 — `cost_ledger.py`'s reserve-before-send gate, exactly-once settlement, and ceiling refusal; `tests/unit/test_cost_ledger.py`, `test_live_model.py` |

### Tests already proving Phase 1/2 behavior

- Domain invariants and valid/invalid transitions (`test_domain.py`).
- Every valid and invalid disposition/root-cause pairing, including proof that
  only application code can create `FAILED_SAFE` (`test_domain.py`,
  `test_graph.py`).
- One structured-output repair and repair exhaustion (`test_graph.py`,
  `test_replay_model.py`).
- Denied-proposal accounting and duplicate fingerprints (`test_policy.py`,
  `test_graph.py`, `replay_fixtures/duplicate_proposal.json`).
- Deterministic clock, evidence ordering, quotas, and truncation markers
  (`test_evidence.py`, `test_graph.py`).
- Development/evaluation seed separation (`test_scenario_control.py`).
- Healthy start, bounded fault activation, repeatable signals, and cleanup for
  every incident family (`test_incident_families.py`,
  `test_scenario_control.py`).
- Citation validity and required-evidence sufficiency scoring
  (`test_evaluation.py`).
- Windows drive letters, path separators, UTF-8 files, writable-directory
  checks (`test_doctor.py`, `fake_machine.py`), proven on both CI platforms.
- `causalops doctor` outcomes for missing Docker, missing API key, less than
  7.5 GiB detected total RAM, less than 12 GB free disk, the advisory warning
  below 2.5 GiB available RAM, and success (`test_doctor.py`).

The tool-timeout and dependency-unavailable behavior above are covered by
the threat table's *Model/tool failure* row (`test_graph.py`,
`test_tool_wrappers.py`) and *Resource exhaustion* row (`test_graph.py`,
`test_telemetry.py`); the finalized-result immutability proof is covered by
the *Scenario
contamination* row (`test_scenario_reset_isolation.py`) — same tests, same
citations, listed once.

Windows support above is proven by continuous integration on
`windows-latest`. A manual smoke test on the working platform, Linux x86-64,
with all required containers, is run by hand and produces no committed
artifact — it is a process step, not a test this repository can cite.

### Tests specified for the live Claude adapter

Unit 3b-2 built the adapter and several of these; the rest still describe
required behavior with no code behind it yet.

**Built, Unit 3b-2** (`tests/unit/test_live_model.py`,
`test_cost_ledger.py`, `tests/security/test_credential_isolation.py`,
unless noted):

- Exact Claude request shape: required `claude-sonnet-5`, adaptive thinking,
  `medium` effort, `max_tokens=1600`, no `temperature`/`top_p`/`top_k` —
  `live_model.MODEL_NAME`/`pricing.MAX_OUTPUT_TOKENS`, asserted by
  inspecting the fake client's captured `ChatAnthropic` construction.
- Synchronous SDK construction with `max_retries=0` and one concurrent
  request — no async client anywhere in `live_model.py`, no retry logic
  (project-wide: `src/` has none anywhere).
- Token counting over the rendered request text; refusal, not deterministic
  trimming, above the input cap (`pricing.estimate_input_tokens`,
  `InputTooLarge`) — a deliberate deviation from the original "trimming and
  recounting" framing: silently cutting context is a value lost where no
  assertion downstream could ever see it happened, so this refuses instead.
- Cost-cap gates, the reservation formula, and durable reservation/
  settlement ordering (`cost_ledger.py`). Unique logical request IDs are the
  amended §5 four-part key (`run_id + graph_phase + model_turn +
  context_digest`), not a separately minted ID. (The v1 figures — USD 0.15
  standalone, USD 1.75 for 24 runs — remain superseded; see Part III.)
- Crash/timeout and missing-usage fixtures retaining the full reservation
  (`test_a_failed_send_leaves_the_reservation_reserved`,
  `test_missing_usage_metadata_leaves_the_reservation_reserved`).
- API-key redaction: the adapter never reads, stores, or forwards the key
  itself (`ChatAnthropic()` resolves it from the environment internally).

**Still not built:**

- Authenticated `GET /v1/models/claude-sonnet-5` metadata check and exact
  required-model mismatch handling. Deliberately deferred, not forgotten
  (`cli.py`'s `MODEL_CHECK_NOTE`): a second routine network call from
  `doctor`, a command run far more casually than a live `investigate`, was
  judged not worth adding in the same unit that exists to make the one
  deliberate network call safe.
- Per-operation timeout behavior beyond the SDK's own default, and exactly
  one *inspected* HTTP attempt per logical operation (this unit proves zero
  retries; it does not inspect the wire-level HTTP attempt count).
- A per-`stop_reason` matrix (`end_turn`/`tool_use` handled; every other
  documented provider stop reason — `refusal`, `max_tokens`, `pause_turn` —
  currently falls through to ordinary invalid-output handling, not a
  distinct no-repair path each with its own fixture).
- A billed-refusal fixture proving usage settles before failure handling.
- Resume never repeating an outstanding request: not reachable today, since
  `graph.py`'s resume path never calls the model at all (see this
  document's Unit 3b-2 review-gap notes above).
- Proof that provider thinking blocks are never retained: true by
  construction today (nothing in `live_model.py` extracts or stores a
  `thinking` content block from the response), but not pinned by a test.
- Sequential and resumed benchmark runs, and clean/dirty commit provenance
  blocking publication of a non-reproducible score — Unit 3c's job, the
  paired evaluation, not this one.

Normal CI runs on `windows-latest` and `ubuntu-latest` using replay fixtures
and disposable local test data. Network access is allowed only while
installing locked dependencies; after that, formatting, linting, strict
typing, unit tests, security tests, and replay conformance make no external
calls and require no credentials — CI never invokes `causalops` at all, only
`pytest`/`ruff`/`mypy`. As of Unit 3b-2, `causalops investigate --model
claude` does send an authenticated request to Anthropic, but only that one
command, only when a person types it by hand: `--model` has no default,
`--model claude` is not scripted anywhere in this repository, and
`tests/conftest.py`'s loopback-only network guard covers the whole `pytest`
process regardless.

# Part III — v2 in progress

The old v1 delivery process (§14 "Three-phase delivery sequence" in
`git show b6f4d9c:TECHNICAL_OVERVIEW.md`) is superseded by `CLAUDE.md`'s
owner-controlled review protocol and by the milestone/unit vocabulary
defined at the top of this document.

## Milestone 1 — Bounded tool-graph parity

**Status:** complete, pending final dual review and commit (Unit 1d-2 is
frozen for review on branch `retire-investigation-loop`, not yet committed).
Unit 0 (this document and the matching
`TECHNICAL_SPEC.md` amendments) has landed, and so has the housekeeping unit
ahead of 1a (`9ffdc95`, merged `b3f5da0`): `tests/unit/import_scan.py` and the
`write_log`/`log_row` promotion into `fake_incident.py`, plus the last of the
stale numbered-section references.

Unit 1a (the trust boundary) has landed: `GraphPhase` (the seven phases in
§5's diagram, including `ESCALATION_INTERRUPT`, ahead of any code that reaches
it — the same precedent as `InvestigationState` naming every state a run can
reach); `ToolReceipt`'s `RESERVED`/`SETTLED` lifecycle and its coherence
validator; `tool_calls.py` (native-tool-call parsing, dependency-free);
`tool_wrappers.py` (`query_logs`'s policy wrapper, `ReservationLedger`, and
`DispatchResult`); `src/causalops/__init__.py` force-disabling both LangSmith
tracing variables unconditionally at import time; and the three-part
tool-policy-bypass proof required by `TECHNICAL_SPEC.md` §9 in
`tests/security/test_tool_boundary.py` — an AST import test showing the
wrapper module imports no backend module, a wrapper-identity test showing
every registered dispatch callable both is a `ToolWrapper` *and* was actually
built by a wrapper factory (a module-private factory token makes direct
construction raise `TypeError`, not merely an unenforced naming convention),
and a spy-backend test, parameterised over every registered tool, showing a
denied proposal never invokes one. `langgraph` and `langchain-core` landed in
this unit deliberately ahead of any code importing them, to isolate the
dependency's lock/CI risk from Unit 1b's orchestration code.

Unit 1b (the orchestration: `graph.py`, the composing replay tool-calling
adapter in `models.py`, the CLI `--orchestrator` flag, and the parity test)
has been implemented and frozen for review on branch `graph-orchestrator`.

Per `TECHNICAL_SPEC.md` §12: `graph.py` runs one replay incident through a
five-node LangGraph `StateGraph` (`investigate`, `dispatch_tool`,
`normalize_evidence`, `final_assessment`, `final_report`, wired with two
conditional edges) with native tool-call parsing, `query_logs`'s policy
wrapper, atomic budget reservation, and a report built the same shape as
`workflow.py`'s. `tests/unit/test_parity.py` proves the loop and the graph
agree on `graph_single_check.json` — the one fixture both can run, since
`dispatch_registry` wraps only `query_logs` until Unit 1c. Unit 1c wraps the
remaining three tools; 1d retires the loop once conformance parity extends to
them too.

Graph state (`GraphState`, a `TypedDict`) is JSON-only and holds the full
projection §5 requires: receipts, evidence records (not just IDs — see the
§5 amendment below), `investigation_id`, `incident_id`, budget counters, and
phase. Nothing survives off-state between graph turns: `dispatch_tool`
rebuilds a `ReservationLedger` from state's receipts on every call via the
new `ReservationLedger.from_receipts` constructor, and `_rebuild_store`
does the same for an `EvidenceStore`.

A hazard the pre-edit report caught before implementation began: a
`ReservationLedger` built inside `dispatch_tool` is a node-local object, so
if the wrapped backend raises, the node never returns and LangGraph never
learns about the fresh `RESERVED` receipt that reservation already wrote —
silently reopening the exact gap `tool_wrappers.py` exists to close (a
crashed check leaving no receipt at all). The fix is `dispatch_tool` catching
around its own call to the wrapper, writing `ledger.receipts()` into the
state update it returns, and routing to `final_report` via the ordinary
`normalize_evidence` conditional edge — a modeled transition, not an
exception escaping `invoke()`. `test_graph.py`'s
`test_a_raising_backend_leaves_a_visible_reserved_receipt_in_the_graph_report`
is the regression test.

`TECHNICAL_SPEC.md` §11's second half — "a test proves no tracing client is
constructed and no tracing request is attempted" — is now closed.
`graph.py` is the first module that actually imports `langchain-core`, which
made it testable for the first time; `tests/security/test_no_tracing.py`
asserts no `langsmith.client.Client` is ever constructed, no `httpx`/
`requests` send is attempted, and both tracing variables read `"false"`
after a full graph run — not the weaker (and false)
`"langsmith" not in sys.modules`, which the module docstring explains.

Independent dual review of the frozen snapshot found two P1s, both fixed:
`route_after_normalize` bounded its loop only by remaining tool/model-call
budget, not by turn count, so a denied second proposal (which spends no
slot) let the graph ask a third `INVESTIGATE` turn the loop never asks and
the model contract has no stage for; and `investigate`/`final_assessment`
had the same node-local-value-lost-on-crash shape the pre-edit report had
already caught and fixed for `dispatch_tool`'s ledger, applied to the
model-call budget instead — a model call already counted before a raise
was invisible in the final report once the node's frame was gone. Both are
now `_StageCounters`/`_ask_with_repair`/`try`/`except GraphBubbleUp: raise`
inside all three nodes that can crash mid-attempt, with regression tests at
both the graph level (`test_graph.py`) and the orchestrator-comparison
level (`test_parity.py`), each demonstrated failing against the pre-fix
code before landing. The review also restored `dispatch_tool`'s event
vocabulary (`proposal_denied`/`check_started`/`check_finished{outcome}`,
matching the loop's names and outcome semantics instead of one event
carrying `policy_result` for every case) and widened its `except` to cover
the full state-update tail, not just the wrapper call. `check_finished`
also carries `duration_ms`, taken from the receipt rather than measured
between the two events: `authorize()` runs inside `wrapper.dispatch`,
invisible to the node, so both events fire together only after dispatch
already returned, with the real backend call sitting entirely in the gap
*before* `check_started`. **Under the graph orchestrator, do not compute a
check's duration from `check_started`'s and `check_finished`'s
timestamps in `events.jsonl` — that gap is not a timing bracket, unlike
the loop's, where `check_started` fires before the backend call begins.**
`ToolReceipt.duration_ms` is the one authoritative figure for either
orchestrator.

Unit 1c (`wrap-remaining-tools`) wraps the other three tools —
`query_metric`, `list_recent_changes`, `get_topology` — and pays the two
debts Unit 1b's gap list named below. `tool_wrappers.py` gained one generic
factory, `_make_wrapper[ArgsT: BaseModel](tool, arguments_type, run_check)`,
built and proven against a standalone `mypy --strict` repro (including the
negative case: a backend typed for the wrong *argument type* is a `mypy`
error) before it replaced what would otherwise have been four near-identical
copies of the dispatch body. The type binding is only between
`arguments_type` and `run_check`; `tool` is a plain `ToolName` with no type
relationship to either, so a factory call where `tool` itself disagrees with
`arguments_type` still type-checks (caught only at dispatch time, by the
`isinstance` check, not by `mypy`) — see P3-1's fix to the runtime error
message below. `query_logs_wrapper` and three new named siblings
(`query_metric_wrapper`, `list_recent_changes_wrapper`, `get_topology_wrapper`)
are now thin aliases over it. The backend seam is uniform across all four —
`Callable[[ArgsT, IncidentScope], CheckOutcome]` — even though only
`query_metric`'s backend reads the scope it is given (the PromQL `incident`
label); the other three ignore it, which was judged a smaller cost than a
fourth, differently-shaped factory.

`dispatch_registry` is now keyword-only and requires all four backends —
`run_metric`, `run_logs`, `run_changes`, `run_topology` — and always builds
the full four-tool registry; there is no longer a way to build a partial one
through it. A caller that genuinely needs a partial registry (only
`test_an_unwrapped_tool_proposal_is_refused_before_a_backend_is_reached`
does) builds the `dict[ToolName, ToolWrapper]` by hand instead — exactly the
type `dispatch_registry` itself returns. The keyword-only signature was
chosen over relying on the four argument types' structural distinctness
under `mypy` alone: a stale positional call now fails at the Python call
itself, in plain `pytest` output, at every call site, not only under a
separate `mypy` run. `cli.py`'s graph path now wires all four real backends
(`run_metric_check` against `DEFAULT_PROMETHEUS_URL`, the other three
against the run's `RunPaths`) instead of `query_logs` alone.

`tests/security/test_tool_boundary.py`'s spy control is now four independent
spies (`fake_incident.py`'s `RecordingBackend[ArgsT: BaseModel]` and its four
named subclasses), each asserted `.calls == []` separately after every
tool's out-of-scope proposal is denied — a single shared spy would have kept
passing even if only one of four wrappers were actually reachable.
`get_topology`'s only refusable shape is a cross-incident id
(`policy.authorize`'s `CROSS_INCIDENT_REQUEST` branch, produced nowhere
else), not a service or window, since its arguments carry nothing else.

Parity (`test_parity.py`) changed in two ways. First, `DispatchResult`
gained a defaulted `message: str = ""` field, threaded from
`PolicyDecision.message` through `_denied_receipt`, so `dispatch_tool`'s
`proposal_denied` event now carries `message` the same way the loop's always
has — closing the first gap below. Second, and larger: every dispatch mints
receipt/evidence ids through `new_opaque_id()`, and the loop and the graph
call it independently, so their id sequences differed by construction even
on an identical script. `run_both` now monkeypatches `new_opaque_id` in
every module holding its own imported copy of the name
(`causalops.evidence`, `causalops.graph`, `causalops.tool_wrappers`,
`causalops.workflow`) to a plain call-counter, reset to zero immediately
before each orchestrator's own run. Because both orchestrators mint a
receipt id before an evidence id, once per dispatch, in the same order
regardless of which tool, the two id sequences come out byte-identical, not
merely equal in shape — which turns `final_context_digest`, `evidence_ids`,
and `receipt_ids` from excluded fields into hard equalities. Only wall-clock
fields remain excluded. `assert_dispatch_events_agree` additionally compares
the ordered `(name, fields)` pairs of the four dispatch-vocabulary events
(`proposal_received`, `proposal_denied`, `check_started`, `check_finished`),
excluding `at`/`sequence`/`state` because none of the three lives inside
`fields` to begin with, and treating the graph's `check_finished.duration_ms`
as a documented superset of the loop's.

`lab_diagnosis.json` — the loop's own default fixture, two executed checks
across two tools (`query_logs` then `list_recent_changes`) — is now also a
parity scenario (`test_the_loop_and_the_graph_agree_on_two_executed_checks`),
the first in this file where a tool besides `query_logs` does real work
under the graph. **It passed on the first run, with every comparison above
exact — the correctness reviewer's own measurement (byte-identical digests,
identical evidence/receipt ids, 10 clock reads each) held for the two-tool
case too, and no divergence needed classifying.** The pre-edit report had
predicted a divergence was likely here; tracing `EvidenceStore.ordered()`'s
sort key (`observed_at`, `kind`, `evidence_id`) before implementing showed
why it would not be one — each orchestrator's own `StepClock` only advances,
so check-order determines evidence order regardless of the two orchestrators'
absolute tick counts, and `lab_diagnosis.json`'s four evidence kinds never
tie on `observed_at` to fall through to the id tie-break.

**Known gaps carried into Milestone 2:**

- `graph.py` binds the concrete `ReplayToolCallingModel`, not a
  `ReasoningModel`-style protocol the way the now-retired `workflow.py` bound
  `ReasoningModel`. A `propose()`-shaped protocol would be speculative with
  only one implementation to validate its shape against — `CLAUDE.md`
  forbids indirection without a concrete demonstrated need, and one
  implementation is not that. Closes when the live Claude adapter unit adds
  a second implementation to design the protocol against.
- ~~`evaluation.py`'s `count_control` reads only `policy_result` and
  `reason_code`, never `outcome` or the new `state` field, so a run ending
  with a `RESERVED` receipt is invisible to the scorer's `ControlCounts`.~~
  **Closed in Unit 2d**, which added `ControlCounts.unsettled`. The closing
  condition recorded here at the time — "once Milestone 2 makes reservations
  durable across a checkpoint resume" — was wrong: reachability was never
  gated on durability. The in-process crash path has produced a `RESERVED`
  receipt in a finalized report since Unit 1d-1 (`test_graph.py:221-253`),
  before any SQLite checkpointer existed. What Unit 2d actually needed was
  simply teaching the scorer to count a state the domain model already
  supported.
- `ToolReceipt`'s lifecycle validator checks `state` against `outcome`/
  `result_digest`/`evidence_id` only. It does not (yet) reject a `RESERVED`
  receipt carrying a `reason_code` or a nonzero `duration_ms`, or a
  `policy_result` other than `ALLOWED`. Nothing constructs those combinations
  today and the docstring does not claim they are closed; tightening is
  deferred, not forgotten.
- **Closed in Unit 1d-1.** `dispatch_tool`'s crash handler (`graph.py`) used
  to carry only the *receipt* out of a mid-attempt crash, not evidence: its
  `except` block's return had no `"evidence"` key, so a `SETTLED` receipt
  minted in the narrow settle-then-crash window (a crash inside
  `wrapper.dispatch`, after `ledger.settle()` succeeded but before
  `DispatchResult` was constructed and returned) would cite an `Evidence`
  record that never entered state. `ledger.settle()` now durably stores that
  record the instant it runs, keyed by `receipt_id`, and the handler's
  `recovered_evidence = [record.model_dump(mode="json") for record in
  ledger.evidence()]` recovers it. One narrower gap remains, unrelated to
  this fix and not closed by it: if this handler's own `recorder.event` call
  raises, that exception propagates out of the node, losing this dispatch's
  receipt as well as its evidence — the code comment at the crash handler
  says so.
- `telemetry.registered_check_runner` and the `RunCheck` alias it returns
  (`domain.py:359`) are orphaned production code: `workflow.py`'s loop was
  their only caller, and `cli.py`'s graph path calls `dispatch_registry`
  instead, never `registered_check_runner`. The function is correct and
  still tested directly (`test_telemetry.py`), just unreachable from
  anywhere that runs — nothing in `src/` calls it any more. Deleting it is a
  separable decision from Unit 1d-2: it would also touch `test_telemetry.py`
  and `fake_incident.py`'s `check_runner`, which is why it was not folded
  into this unit. Owner-approved as a recorded gap, not a blocker, on the
  simplicity reviewer's suggestion.

**Deliberate, not a gap:** `domain.py`'s `SCHEMA_VERSION` stayed `"1"` even
though `ToolReceipt`'s persisted shape changed (the new `state` field,
`outcome` becoming optional). The change is backward-compatible — every
existing constructor call still produces a valid, equivalently-interpreted
receipt — no consumer keys behavior on the version string, and `results/`/
`runs/` are empty in this repository, so there is no persisted artifact to
migrate. Revisit this the moment a reader (a replay fixture, an external
consumer, a migration script) actually depends on the version number
distinguishing the two shapes.

Unit 1d-1 ported every loop-only behaviour `test_workflow.py` alone proved
into `test_graph.py`, run against the graph while `workflow.py` still ran
beside it, so a reviewer could recompute the frozen parity literals
independently against the live loop one last time before that comparison
became impossible. This is also where the settle-then-crash evidence-carry
fix above landed, plus `MODEL_OUTPUT_INVALID`, `WALL_CLOCK_EXPIRED`,
`MODEL_CALL_BUDGET_EXHAUSTED`, three `FORGED_EVIDENCE_REFERENCE` variants,
`INSUFFICIENT_EVIDENCE`, usage accumulation, digest determinism, and
`TIMEOUT`/`UNAVAILABLE`/`ERROR` tool outcomes (the last three landing in
`test_tool_wrappers.py`, not `test_graph.py`).

Unit 1d-2 ported the two remaining loop-only tests (a denied proposal that
still spends a model call but no check slot; the same proposal proposed
twice, denied as a duplicate the second time), then deleted `workflow.py`,
`test_workflow.py`, and `InvestigationState` outright. `cli.py`'s
`--orchestrator` flag is gone with it — `investigate` now always runs the
graph, against `lab_diagnosis.json` (`REPLAY_FIXTURE`), the same fixture the
retired loop used by default. `tests/unit/test_parity.py` was renamed to
`test_graph_frozen_reports.py`, its job from this point on a regression pin
on the graph's own behaviour rather than a two-orchestrator comparison — no
new fix landed in this unit; the evidence-carry fix above is 1d-1's, not
1d-2's.

## Milestone 2 — Durable escalation and owner approval

**Status:** Units 2a, 2b, 2c, and 2d are implemented below; Milestone 2 is
complete pending dual review of 2d's frozen snapshot. §12's "approval
routing" deliverable is two-thirds built, not fully: accept and reject both
route to `FINAL_REPORT` and are fully tested; the third owner option
(approving one additional already-authorized check, routed to
`DISPATCH_TOOL`) defers until a policy-approved next-check proposal exists to
approve (`TECHNICAL_SPEC.md` §8, *Amendment, Unit 2d*) — 2b's own gap list
assigned closing this to 2c, 2c did neither, and it went unrecorded until
2d's planning caught it. Curated FTS5 runbooks, retrieval provenance, and
injection/no-ground-truth-leakage tests defer to Milestone 3
(`TECHNICAL_SPEC.md` §12, *Amendment, Milestone 2*); Pinecone remains a
post-milestone optional experiment.

Unit 2a (durable checkpoints, no behaviour change) swapped the graph's
checkpointer from a process-local `InMemorySaver()`, rebuilt on every call,
to a `SqliteSaver` writing `results/checkpoints.db` — gitignored already by
the blanket `results/` rule, and outside `runs/`, so `reset_scenario`'s
`shutil.rmtree` never touches it. `run_graph_investigation` gained two
keyword-only parameters, both `None`-defaulted to today's exact behaviour so
no existing caller changed: `investigation_id` (minting moves behind a
"still `None`" branch — the same value already doubled as LangGraph's
`thread_id`, now invertible for a future resume to supply) and
`checkpointer`. `GraphState` gained `run_id` (`TECHNICAL_SPEC.md:140-142`;
minted via `uuid4().hex` directly rather than through the same counter
`evidence`/`receipt`/`investigation` IDs share, since it is never cited,
asserted, or rendered — routing it through that counter would perturb
externally visible ID sequences it has nothing to do with) and `events`: run
events, previously accumulated by a `RunRecorder` object shared and mutated
across a factory closure (and so lost at any process boundary), are now a
state list rebuilt into a local recorder per node and returned whole on
every one of that node's return paths, the same pattern `receipts`/`evidence`
already used. A second, independent clock parameter (`event_clock`) keeps
`RunEvent` timestamps isolated from the clock that times domain data (tool
duration, budget expiry, evidence timestamps) — in production both are
`utc_now` and the split is invisible, but it is what keeps recording an
event from ever perturbing a domain timestamp, which a `StepClock` test
double that advances on every read would otherwise expose immediately.

Everything above is a mechanical consequence of moving state that already
existed (receipts, evidence) or was already planned (`run_id`) into
`GraphState`. The unit's one piece of genuinely new production logic sits at
`run_graph_investigation`'s `except GraphBubbleUp:` branch. Before this
unit, the caller's `RunRecorder` was a single object every node mutated
directly through a factory closure, so it already held whatever had been
recorded up to the moment a `GraphBubbleUp` (interrupt, drain, parent
command) escaped `.invoke()` uncaught — the seed "investigation_started"
event, plus anything a node recorded before it raised. With events moved
into state, nothing writes to the caller's `recorder` during the run at
all; it is synced from `final_state["events"]` only at this function's
normal return, which a `raise` never reaches. Left alone, that would have
made the caller's `recorder` come back empty on every such escape — a
regression a reviewer caught before this snapshot froze. The fix reads the
last checkpoint LangGraph itself committed and syncs `recorder.recorded`
from it before re-raising, the same recovery `except Exception` already
performs to build a report. That read is I/O against the same database
`checkpointer` uses, so it is wrapped in its own `try/except Exception:
pass`: a checkpoint-read failure must not replace the control-flow signal
with an unhandled database error, even at the cost of the event-recovery
step itself.

This narrows one observable behaviour, honestly recorded here because the
unit's acceptance criterion was zero behaviour change even though this one
case is arguably an improvement, not a regression. At `17c65d4`, a
`GraphBubbleUp` escaping `investigate`'s first turn left the caller's
`recorder` holding `["investigation_started", "stage_started"]` — the
node's own in-flight event, recorded as a direct side effect before the
raise, survived because the shared object was mutated immediately. Today
the same escape leaves the caller's `recorder` holding only
`["investigation_started"]`: an uncommitted node's own events are never
merged into state, so recovering from the last *checkpoint* cannot recover
work a node was still in the middle of when it raised. The event genuinely
did not durably happen yet, so losing it is arguably more correct than the
old behaviour of reporting it anyway — but it is a real, measured
difference from `17c65d4`, verified with the model-double
`test_graph.py::test_a_graphbubbleup_escape_still_syncs_the_callers_recorder`
constructs, and it belongs in this record rather than living only inside
that one test's assertion.

`cli.py`'s `_sqlite_checkpointer` opens that connection with
`allowed_msgpack_modules=None` on the checkpoint serializer — the same
restriction `LANGGRAPH_STRICT_MSGPACK=true` applies, set as a constructor
argument so the one place this database connection is opened is also the
one place the policy is decided, rather than adding another `.env.example`
variable for a database path the owner decided should stay fixed, not
configurable. Without it, a compromised `checkpoints.db` could make its next
read import and instantiate an arbitrary class named in a checkpoint blob.
`test_checkpointing.py::test_the_hardened_serializer_refuses_an_unregistered_type`
proves the restriction itself actually blocks reconstruction (using a real
`causalops.domain` type, not a synthetic stand-in) rather than merely being
configured, and
`test_checkpointing.py::test_sqlite_checkpointer_yields_a_saver_with_the_hardened_serializer`
proves `_sqlite_checkpointer` actually wires that restriction into the
object it hands the caller — a first version of this test file proved only
the former and left the latter unverified, so a reviewer's mutation
(swapping the hardened construction for the permissive default inside
`cli.py`) passed the whole suite undetected.

`langgraph-checkpoint-sqlite` (3.1.1) is the one new direct dependency; it
ships `py.typed`, type-checks clean under this project's `mypy --strict`
with no override, and pulls in `aiosqlite` and `sqlite-vec` transitively —
an async saver and a vector-search extension, respectively, that this
codebase does not call anywhere. Recorded here so they are not first
discovered in a `uv.lock` diff: only the package's synchronous `SqliteSaver`
is used.

**Known gaps carried into Unit 2d, so the owner does not meet either one
first at a demo:**

- `cli.py`'s `_sqlite_checkpointer` opens `sqlite3.connect(...)` with no
  error translation. A real failure mode — disk full, permission denied, a
  locked or corrupted database file — raises a bare `sqlite3.OperationalError`
  or similar, which is neither `LabError` nor `RunRecordError`, so it
  escapes `main`'s `except (LabError, RunRecordError)` and surfaces as an
  unhandled traceback instead of this project's `FAIL <CODE> <message>`
  contract. The store error type belongs to Unit 2c (the approval database
  needs one too) and the crash/idempotency tests belong to Unit 2d, so the
  translation lands with the tests that exercise it rather than being added
  speculatively in Unit 2a.
- `doctor.py`'s `ProjectPaths` has no accessor for the checkpoint database
  path and no pre-flight write-probe for it, unlike the existing
  `RUN_DIRECTORY_NOT_WRITABLE` check. Unit 2a's own actual need — the
  directory existing before `_sqlite_checkpointer` opens a connection — is
  already covered inline by `mkdir(parents=True, exist_ok=True)`, verified
  against a fresh project with no `results/` directory
  (`test_checkpointing.py::test_investigate_leaves_a_checkpoint_database_in_a_fresh_project`).
  A doctor diagnostic for this database's writability is deferred alongside
  the SQLite error-translation gap above, for the same reason: it exists to
  pre-empt exactly the failure mode that gap still lets through, so the two
  should land together.

### Unit 2b — the escalation interrupt

Inserts a ninth node, `escalation_interrupt`, between `final_assessment` and
`final_report`, replacing the plain edge with a conditional one
(`route_after_final_assessment`) so a run pauses for the owner on one of
three deterministic triggers and survives its own process. `GraphPhase`
already named `ESCALATION_INTERRUPT` in Unit 1b's enum; this unit is its
first consumer.

**Triggers, checked in `_escalation_reason` — deliberately not the order
`TECHNICAL_SPEC.md` §8 lists them in.** The spec (and `EscalationReason`'s
own member order) lists `CONFLICTING_EVIDENCE`, `TOOL_UNAVAILABLE`,
`INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING`,
`RETRIEVAL_COVERAGE_INSUFFICIENT`. The function checks a receipt going
`UNAVAILABLE` first, ahead of the other two: an owner should see a
diagnosis reached with missing data before anything else, regardless of
what the model concluded from the checks that did run. `graph.py`'s own
docstring on `_escalation_reason` says so explicitly, after an earlier
version of that docstring and `test_domain.py`'s enum-pin test both claimed
the function followed the spec's listing order — both reviewers caught the
same contradiction independently (the docstring's very next paragraph
already said `TOOL_UNAVAILABLE` was checked first), and both comments were
corrected to say what the code actually does. `RETRIEVAL_COVERAGE_INSUFFICIENT`
is unreachable until Milestone 3 supplies `search_runbooks`; it is named in
the enum anyway, the same precedent `GraphPhase` itself set for phases no
code visited yet.

**The finding that shaped the unit, from the pre-edit report and confirmed
before implementation began:** nothing in the codebase can produce a
policy-approved next-check proposal at escalation time — the model proposes
at most one check per turn, it dispatches immediately, and a denied or
allowed proposal's fingerprint is marked seen either way, so re-proposing it
returns `DUPLICATE_PROPOSAL`. The owner gets accept or reject in this unit;
the approve-one-additional-check route (`TECHNICAL_SPEC.md` §8's third
option) is not built, so there is no dead route to `dispatch_tool` sitting
untested.

**The paused result is a sibling type, not a variant of `InvestigationResult`.**
`EscalatedInvestigation` (frozen, in `domain.py`) carries `thread_id`,
`run_id`, `checkpoint_id`, `reason: EscalationReason`, `evidence`, `receipts`
(mirroring `InvestigationResult`'s own shape, not just the spec's leaner
"evidence IDs" wording — `run_graph_investigation` already has the full
records in hand at pause time, at zero extra cost), `remaining_check_count`,
and `proposal_fingerprint` (always `None` this unit). `InvestigationReport`
cannot express "paused" — frozen, a required `report`, a strict
DIAGNOSED/INSUFFICIENT_EVIDENCE/FAILED_SAFE trichotomy validator — so
`run_graph_investigation` now returns `InvestigationResult | EscalatedInvestigation`,
detected by checking `"__interrupt__"` in `.invoke()`'s own return value
before the pre-existing `InvestigationReport.model_validate(final_state["report"])`
call, which raised a `ValidationError` on a real pause before this unit (the
landing hazard the pre-edit report located in advance).

**The owner's decision is recorded, not just acted on.** `InvestigationReport`
gained one additive optional field, `escalation: EscalationRecord | None`
(reason plus `"accept"`/`"reject"`), over the cheaper but weaker alternative
of folding it into the existing free-text `limitations` field — a decision
the pre-edit report flagged for the owner rather than resolving silently,
and the owner approved the new-field shape. Rejecting an escalation does
**not** overwrite `disposition`/`root_cause` to `FAILED_SAFE`: the model's
assessment stands, annotated as rejected, on the reasoning that "reject"
means the owner did not accept it as final, not that nothing was concluded.
The field does not interact with `check_terminal_invariants` and does not
trip `test_ground_truth_isolation.py`'s field-name scan.

**A bad resume value re-interrupts; it does not raise.** The first
implementation raised `ValueError` on an unrecognised decision, reasoning
that the node crosses no I/O boundary and so has nothing at risk from a
crash. Both reviewers found this wrong by actually reproducing it against a
real `SqliteSaver`: LangGraph persists a resume value against the
interrupt's own id and replays that same value on **every** later resume of
that interrupt, so raising on a typo left the thread permanently stuck
replaying the same bad value on every subsequent attempt — valid ones
included — because this unit never finalizes on pause and so the run never
gets a fresh interrupt id to retry against. A single typo destroyed the run
with no recoverable artifact. The fix asks again under the same interrupt
id: `while decision not in ("accept", "reject"): decision = interrupt({**payload, "retry": True})`.
The `try`/`except` question this raised dissolves with the fix in place, but
the real reason the node stays unwrapped is `normalize_evidence` and
`final_report` are unwrapped for the same reason: none of the three cross an
I/O boundary or hold a reservation a crash could strand.

**Purity before `interrupt()` is enforced by construction, and only
partially independently testable.** Everything before the `interrupt()`
call is a pure read of state — no `recorder.event`, no reservation, no
write — because only the interrupted node re-runs on resume (measured
against the installed LangGraph, not assumed) and anything written above
that call would run twice. A mutation moving a `recorder.event(...)` call
to *before* `interrupt()` survives every test in this project: on the
interrupted attempt `GraphInterrupt` raises before the node returns, so the
write never reaches a state channel regardless of where it sits in the
function, and on the resumed attempt the function runs once top-to-bottom
either way — placement relative to `interrupt()` is unobservable through
checkpointed state for a node that stays inside the rebuild-from-state
pattern every node already follows. This immunity covers *recorder* writes
only: a durable write that reaches outside the checkpoint (a real backend
call, a file write, a module-level mutable) or an asserted clock tick taken
above `interrupt()` would still double, and neither reviewer nor the
implementation found a way to make that specific class of mistake
observable without actually introducing an external effect the node does
not otherwise need. The operational form of the purity guarantee that *is*
mutation-tested and does fail cleanly: no upstream node with a real side
effect (`final_assessment`'s model call) re-executes on resume, proven by
asserting the replay model's own request count is unchanged after resuming
a pause, both in-process and reopened from a fresh `SqliteSaver` connection.

**The unit's own proof is a two-process resume, not the in-process
convenience every other test in this project's escalation coverage uses.**
Every other resume test drives `InMemorySaver()`, which would pass
identically whether or not this unit's actual target — resuming through the
hardened `SqliteSaver` `cli._sqlite_checkpointer` wires in production —
works at all. `test_checkpointing.py::test_a_two_process_pause_and_resume_settles_over_a_real_sqlite_file`
pauses under a real `cli._sqlite_checkpointer(path)`, closes that context
entirely, opens a second, independent one against the same file (standing
in for the second process `causalops approve`/`reject` will be in Unit 2c),
reads the pending interrupt and its payload off it, resumes, and asserts
zero additional model calls — a stronger claim than the in-process version,
since it holds across a reopened connection, not just within one process's
live object graph. This was the first version's most material gap: every
resume test it shipped used `InMemorySaver()`, so the milestone's actual
proof was simply missing.

**`run_graph_investigation` gained no `resume` parameter.** 2b's approved
boundary is graph-level resume driven directly from tests
(`build_graph(...)` plus `Command(resume=...)` against the same
`checkpointer`/`thread_id` the pause used), not a second production entry
point — Unit 2c owns the real resumable surface (`causalops approve`/
`reject`) and reviews it on its own. One consequence: a resumed run never
enters `run_graph_investigation` at all, so it gets none of that function's
own crash containment. A bad resume value is the caller's mistake to catch
before ever calling `Command(resume=...)`, not this graph's to paper over —
2b's own tests are the only caller today, and Unit 2c decides how a real
CLI command validates a decision before resuming for real.

**Two existing fixture scenarios started escalating, flagged in the
pre-edit report before implementation and dispositioned by the owner in
advance:** `test_graph_frozen_reports.py`'s `after_a_first_turn_denial` and
`after_a_repeated_proposal` throwaway scripts never call
`write_orders_error_row`, so their real `run_logs_check` backend genuinely
returns `UNAVAILABLE` — a real `TOOL_UNAVAILABLE` trigger, not a scripted
one. Rather than weaken those two tests' frozen-literal pins, `run_once`
now resumes a pause with `"accept"` and returns the settled report, proving
an accepted escalation is report-preserving rather than merely asserting
something weaker. Diffed against `aa834c5`: both tests' bodies, and every
shared assertion helper in that file, are byte-identical — only `run_once`'s
setup changed. A third, previously unflagged instance was caught during
implementation itself (`test_graph.py`'s
`test_citing_a_real_same_incident_id_as_contrary_reaches_its_terminal_disposition`,
whose forged-citation control case also happens to satisfy
`INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING`) and fixed the same way, since
a test-only assertion update is in-scope for the coding subagent mid-pass
under this project's protocol, while anything touching production behaviour
is not.

**Known gaps, deliberately not this unit's to close:**

- The `interrupt()` payload itself carries `reason`, `evidence_ids`, and
  `remaining_check_count`, but not `thread_id`/`run_id`/`checkpoint_id` —
  all three `TECHNICAL_SPEC.md:298-299` names as part of the payload. All
  three are recoverable by any second process that already knows the
  thread id (which it must, to call `compiled.get_state(config)` at all)
  and are attached by `run_graph_investigation` when assembling
  `EscalatedInvestigation` from outside the node — but a caller reading the
  raw LangGraph `Interrupt.value` directly, rather than going through
  `run_graph_investigation`, would not find them there. Closing this is
  Unit 2c's, once a real second-process caller exists to specify the
  contract against.
- `evaluation.py:195`'s `disposition_correct` check is blind to
  `escalation.decision` — an owner-rejected diagnosis still scores as
  correct if the underlying `disposition`/`root_cause` match the expected
  label, because rejecting an escalation deliberately does not change
  either field (see above). Whether a rejected run should score
  differently is an evaluation-design question outside this unit's scope,
  written down here so it is not silently assumed away. **Closed in
  Unit 2d.**
- `EscalationReason.TOOL_UNAVAILABLE` and `ReasonCode.TOOL_UNAVAILABLE`
  share the literal `"TOOL_UNAVAILABLE"`, mandated by `TECHNICAL_SPEC.md`
  §8 for the former. Both are `StrEnum`, so they compare equal (`==`)
  across the two vocabularies despite being different classes and
  different concepts — one names a receipt outcome, the other names why a
  run paused because of one. Commented at the enum definition as a trap
  for a future `==` comparison written against the wrong vocabulary.
- **Two third-party/mechanical traps for Unit 2c, verified against the
  installed LangGraph, not this project's bug to fix:**
  - `Command(resume=None)` raises `UnboundLocalError: cannot access local
    variable 'resume_is_map' where it is not associated with a value` from
    inside LangGraph itself (`pregel/_loop.py:910,927` — `resume_is_map`
    is bound only on one conditional branch but read unconditionally). A
    `None` resume value is a real possibility for a real CLI command
    parsing real argv, so `causalops approve`/`reject` must validate its
    decision argument before ever calling `Command(resume=...)`, not rely
    on the graph to reject a bad one gracefully.
  - After a re-pause (a second or later `interrupt()` call on the same
    interrupt id, following an invalid decision), `compiled.get_state(config).next`
    reads `()` — empty, as if the run had finished — while `.interrupts`
    still holds the one pending `Interrupt`. Verified directly: a bad
    resume leaves `next=()` and `interrupts=(Interrupt(...),)` together.
    Unit 2c must detect a pending approval by checking `.interrupts`,
    never `.next` — `.next` alone would misreport a re-paused thread as
    settled.

### Unit 2c — approval routing and the append-only decision record

Adds `causalops approve <thread-id>` and `causalops reject <thread-id>
<reason>`, resuming a paused investigation from a **second process** behind
one append-only decision, and the four-guard sequence that makes that
resume safe to retry. This is the first code in the project that takes an
authorization instruction from outside the process — `TECHNICAL_SPEC.md`
§12's mandatory dual-review trigger.

**A pre-edit-report finding overturned the plan's own design for the
rejection reason, before implementation began.** The plan proposed
splicing the owner's reason into a copy of the already-settled report via
`model_copy(update=...)`. Measured directly: `model_copy` bypasses both
field constraints and model validators — a 500-character note attached to
an **accepted** `EscalationRecord`, serialized clean, where the constructor
raised. Both reviewers, consulted independently on the alternative, each
found a real defect in their own first recommendation before converging:
the note now travels **inside the resume value itself** —
`Command(resume={"decision": ..., "rejection_note": ...})`, a mapping, not
the bare string every escalation test used through Unit 2b —
`escalation_interrupt` destructures it into two new flat `GraphState` keys
in the one return that already carries `escalation_reason`/
`escalation_decision`, and `_build_report` reads all three from the same
state together. A post-return splice would have left the *checkpointed*
report permanently missing the note while the *finalized artifact* had
it — two durable stores disagreeing about one field, and exactly what Unit
2d's deferred settled-but-not-finalized recovery would read back wrong.

**The resume contract changed for every caller, not just the CLI.**
`_parse_resume_decision` (new) requires a mapping with `decision` and
`rejection_note` keys; a bare `"accept"`/`"reject"` string — what every
escalation test sent through Unit 2b, and what a stray `Command(resume=...)`
outside this CLI would still send — is now exactly as invalid as a typo and
re-pauses the same way. Cost, measured rather than estimated up front: four
real `Command(resume=...)` call sites existed. Two (`test_graph.py`'s
deliberately-bad-string cases) needed no change, since a bad string stays
bad either way. `tests/unit/fake_incident.py::resume_graph_run` — the
helper seven other call sites go through — had its body changed to wrap the
plain string it still accepts into the compound shape before calling
`Command`; none of its seven callers' own signatures changed.
`test_the_escalation_interrupt_node_advances_the_phase_before_final_report`'s
bare `compiled.stream(Command(resume="accept"), ...)` is the one site that
had to change directly. A new test,
`test_a_bare_accept_string_re_pauses_under_the_unit_2c_resume_contract`,
pins the new rejection itself — nothing else in the suite would have caught
a mutation reverting `_parse_resume_decision` to accept the old shape.

**`EscalationRecord` gained `rejection_note: str | None`, bounded and
paired.** Named `rejection_note`, not `owner_reason` (`EscalationRecord`
already has `reason`, the enum trigger — a different concept) or
`decision_note` (would wrongly imply an accept can carry one, reviewer
consensus). `check_rejection_note_pairing` (new `model_validator`, the same
idiom `InvestigationReport.check_terminal_invariants` already uses) refuses
a reject with no note and an accept with one — the same pairing rule
`causalops.approvals.OwnerDecision` enforces at the CLI boundary and
`_parse_resume_decision` enforces on the resume value, checked a third time
here because this model's own constructor is directly reachable too (tests
already do it) — three points a caller could reach the same invariant
from, not three independent proof techniques the way the trust boundary's
AST scan / wrapper-identity / spy-backend controls are. `report.py`'s
`escalation_section` renders the note only when present, so an accept's
markdown carries no empty "owner's note" line.

*Correction, from this unit's review round:* a first version of this
paragraph claimed whitespace-stripping was `OwnerDecision`-only, something
"neither `EscalationRecord` nor the graph-side check need." Measured false
during review — `{"decision": "reject", "rejection_note": "   "}` fed
directly to `_parse_resume_decision` settled the run with a blank-looking
note, and `EscalationRecord(..., rejection_note="   ")` constructed clean.
All three now strip before the emptiness check (`domain.py`'s
`check_rejection_note_pairing`, `graph.py`'s `_parse_resume_decision`,
`approvals.py`'s `OwnerDecision`), plus a fourth point that did not exist
in the original design at all: `DecisionRow` (`approvals.py`) got the
identical check added, since a hand-corrupted database row can carry a
whitespace-only note too and this store must refuse it rather than pass it
through as if a real note. Mutation-tested in both directions at each of
the three checking points that gate a live resume (`DecisionRow`'s failure
mode is a corrupted row, covered separately below).

**`OwnerDecision` (new, `approvals.py`) is where a decision becomes
durable-safe before either store sees it.** Frozen, `model_validator`-paired
the same way `EscalationRecord` is. Two properties are genuinely its own,
not shared with the other checking points: whitespace is stripped *and the
stripped text becomes the value itself* — `_parse_resume_decision` and
`EscalationRecord` both strip only to test emptiness, so a directly
constructed `EscalationRecord(rejection_note="  padded  ")` keeps the
padding (Simplicity's own P3 finding on this round, confirmed unreachable
in production since `_build_report` only ever reads an already-stripped
note out of state) — and overflow is **refused, never truncated** — a
silent truncation loses whatever the owner wrote past the 300-character
bound (matching this project's other bounded free-text fields) with no
later assertion able to
see it. Normalizing once here, ahead of both the `owner_decisions` row and
the resume value, is what guarantees the two hold identical bytes.

**The four guards, and a plan correction found while implementing them.**
The plan specified pending-interrupt-check, decision-validation,
record-before-resume, and retry-semantics as one ordered list. Working
through what a *second* `approve` call actually does exposed two problems
with that flat ordering, both fixed before the coding subagent's frozen
report and confirmed independently by the primary agent:

1. After a *first* successful `approve`, the thread is settled —
   `.interrupts` is empty. A pending-interrupt guard checked first would
   refuse every retry, including the identical one the spec requires to
   succeed. The fix checks `owner_decisions` for an existing row **first**,
   by `thread_id` alone.
2. `checkpoint_id` advances across a resume — `escalation_interrupt` and
   `final_report` each commit a checkpoint after `Command(resume=...)`
   runs. A settled thread's *current* checkpoint id is therefore never the
   one its decision was recorded against, so the retry lookup cannot use
   the same composite `(thread_id, checkpoint_id)` key the write uses; it
   queries by thread alone (2c's reachable state has at most one decision
   per thread — there is no "approve one more check" route to
   `DISPATCH_TOOL` yet, so this is unambiguous today).

The resulting order `run_decision_command` actually implements: look up any
existing decision by thread id; a mismatch refuses immediately
(`CONFLICTING_DECISION`), before the checkpoint database or the incident
file is even opened; a match with the artifact already on disk is an
identical retry, reported straight from the finalized `report.json` with
the graph never touched; anything else (a first decision, or a matching
decision whose resume never finished — a crash between the record write and
`finalize_investigation`) resolves the incident, builds the graph, and
checks `.interrupts` only for a genuinely first decision — a decision that
already has a matching row skips the `.interrupts` check entirely and goes
straight to `resume_graph_investigation`, since `Command(resume=...)`
resumes a still-pending interrupt normally and is a genuinely inert no-op
on an already-*settled* thread either way, so both sub-cases of
"unfinished" converge on the same call.

*Correction to the record measured during this unit's review, since an
earlier note stated this more loosely:* "no pending interrupt" covers two
different thread states, and `Command(resume=...)` does not treat them the
same. On a **settled** thread it is a true no-op -- `checkpoint_id`,
`events`, and `model_calls_used` all measured unchanged across the call --
which is exactly what makes the retry branch above safe to call
unconditionally. On a thread that **never paused at all**, the same call
instead starts a fresh run from `START`, a materially different and much
more dangerous outcome this unit's guard #1 (`.interrupts` on a *first*
decision) exists specifically to keep `run_decision_command` from ever
reaching.

**`graph.py`'s crash-containment tail is now shared, not duplicated.**
`_settle_invocation` (new, private) holds the roughly eighty lines of
`GraphBubbleUp`/`Exception` handling, the caller's-`recorder` sync, and the
terminal-vs-paused split that `run_graph_investigation` used to own alone;
it takes a zero-argument `invoke` closure so it does not care whether the
caller is starting fresh (`compiled.invoke(initial_state, config)`) or
resuming (`compiled.invoke(Command(resume=...), config)`).
`resume_graph_investigation` (new, `graph.py`) is Unit 2c's real resumable
entry point — the one `run_graph_investigation`'s own docstring had named as
still owed since Unit 2b. It always returns `InvestigationResult`, never
`EscalatedInvestigation`: a decision that already passed `OwnerDecision`'s
validation exits `_parse_resume_decision`'s retry loop on the first pass, so
there is no path back to a second pause in this milestone's graph.

**`finalize_investigation`'s call site is shared, not duplicated, closing
the pre-edit report's second open question.** `_write_investigation_artifacts`
and `_report_exit` (both new, `cli.py`) are the one place a settled
`InvestigationResult` becomes a written artifact and an exit code, called by
both `run_investigate_command` (a fresh settle) and `run_decision_command`
(an approve/reject settle). Neither lives in `graph.py`: that module still
does no artifact I/O of its own, only checkpointer I/O.

**`main`'s dispatch is now an explicit branch, not a fall-through.** Before
this unit, `main`'s `try` block ended with an unconditional
`return run_investigate_command(...)` — correct only because `investigate`
was the sole command that could reach it. Adding `approve`/`reject` without
converting that into an explicit `if arguments.command == "investigate":`
branch would have made either command silently run an investigation instead
of resuming one; `argparse`'s `required=True` on subcommands makes every
other value structurally unreachable, so the final `else` is
`raise AssertionError`, not another guard.
`test_approve_and_reject_never_fall_through_to_investigate` monkeypatches
`run_investigate_command` to raise and proves neither command reaches it.

**The interrupt payload's missing identifiers, closed by explanation, not
by a code change.** Unit 2b's own recorded gap asked whether the node's
raw `interrupt()` payload should gain `thread_id`/`run_id`/`checkpoint_id`.
It does not: `EscalatedInvestigation` — the object `run_investigate_command`
actually prints from — already carries all three, attached by
`run_graph_investigation` once `.invoke()` returns, because the node itself
has no way to know its own checkpoint id before one exists. Nothing in 2c
reads the raw payload directly; `run_decision_command` recovers
`incident_id`/`run_id`/`checkpoint_id` from the checkpointer with no graph
built, the same pattern `test_a_second_connection_reads_back_the_finished_run`
established in Unit 2a.

**The store error type, closed for `owner_decisions` only.** `approvals.py`
defines `ApprovalReasonCode`/`ApprovalError` (`THREAD_NOT_FOUND`,
`NO_PENDING_INTERRUPT`, `CONFLICTING_DECISION`, `INVALID_REJECTION_NOTE`,
`STORE_UNAVAILABLE`), added to `main`'s `except (LabError, RunRecordError,
ApprovalError)` tuple. `cli._sqlite_checkpointer`'s own bare
`sqlite3.connect(...)` — the half of Unit 2a's gap this unit does *not*
close — still raises an unhandled `sqlite3.OperationalError` on a locked or
corrupt `checkpoints.db`; that translation, and the crash/idempotency
stress tests for both stores, remain Unit 2d's, exactly as recorded when
Unit 2a shipped.

*Renamed in Unit 2d:* `ApprovalReasonCode`/`ApprovalError` became
`CheckpointStoreReasonCode`/`CheckpointStoreError`, once `_sqlite_checkpointer`
— used unconditionally by a plain `causalops investigate`, with no approval
in sight — needed to raise the same type. The five members above kept their
names and values; only the two type names changed. This paragraph is left
under 2c's original names as an accurate record of what 2c actually shipped.

**Mutation-tested, against the real tree** (`PYTHONPATH=src`, `__pycache__`
purged before each run, each mutation reverted immediately after its
failure was observed): `_parse_resume_decision`'s rejection of the old bare-
string shape; guard #1 reading `.interrupts` rather than `.next` (confirmed
against a genuinely re-paused thread, not merely a never-paused one — the
distinction Unit 2b's own measured table exists to make); the
`CONFLICTING_DECISION` refusal (removing it did not merely fail to refuse —
it silently returned the *original* decision's report as if the conflicting
request had succeeded); the identical-retry short-circuit (removing it
surfaced `RESULT_ALREADY_FINALIZED` on the second call, confirming
`finalize_investigation`'s own guard is what the short-circuit exists to
avoid triggering); `record_decision_before_resume`'s
`IntegrityError`-to-`ApprovalError` translation; `DecisionRow.matches`'s
note comparison; `EscalationRecord.check_rejection_note_pairing`'s both
directions; `OwnerDecision`'s whitespace-normalization and overflow bound;
and `main`'s explicit dispatch branch. Every mutation failed a test before
being reverted; `grep -rn MUTATION src tests` is clean in the frozen
snapshot.

**A second, fix-round pass added 9 tests and 7 more mutations, closing
three P2s correctness found in the first frozen snapshot** (none P0/P1 —
"the shipped code path is correct and every failure path it constructed is
fail-safe" was correctness's own framing even at NO-GO): record-before-
resume had no test proving the ordering the function's own name promises
(`test_the_decision_is_recorded_before_the_graph_is_resumed`, behavioural —
`resume_graph_investigation` monkeypatched to crash, decision row confirmed
already committed); the three-layer pairing check was one working layer
plus two that a whitespace-only note walked straight through (mutated
independently at the graph node, at `EscalationRecord`, and at `DecisionRow`
— all three now strip and all three are mutation-tested in both directions,
`test_a_whitespace_only_rejection_note_re_pauses` pinning the exact
reproduction a reviewer measured); and a corrupted `owner_decisions` row or
a corrupted `report.json` on the identical-retry read both used to answer
with an uncaught `pydantic.ValidationError` instead of `FAIL <CODE>
<message>` -- five tests in `test_approvals.py` cover the store-corruption
and end-to-end `cli.main` cases together. Every one of the 7 new mutations
failed its test before being reverted, on the real tree with `__pycache__`
purged; `grep -rn MUTATION src tests` stayed clean throughout.

**Known gaps, deliberately not this unit's to close:**

- `cli._sqlite_checkpointer`'s bare `sqlite3.connect(...)` still has no
  error translation — Unit 2d's, as recorded when Unit 2a shipped.
  **Correction, recorded in Unit 2d:** this bullet originally went on to say
  this unit's own new connection (`cli.py`'s `owner_decisions` access) *does*
  translate that failure, "but only for that connection." That overstated
  what 2c actually built: 2c translated `sqlite3.Error` and
  `pydantic.ValidationError` (a corrupted row) *inside* `approvals.py`'s
  already-open-connection functions (`ensure_decisions_table`,
  `read_decision_for_thread`, `record_decision_before_resume`) into
  `ApprovalError(STORE_UNAVAILABLE)` — but the `sqlite3.connect(...)` call
  itself, in `cli.py`, was never wrapped by 2c at all. It was exactly as
  untranslated as `_sqlite_checkpointer`'s own, for the identical reason:
  neither connection open had a `try`/`except` around it yet. Unit 2d closes
  both.
- A crash between `record_decision_before_resume` committing and
  `Command(resume=...)` ever being called is handled (the retry path
  resumes again using the already-recorded decision), but a crash between a
  successful resume and `finalize_investigation` writing artifacts is only
  exercised by reasoning, not a test that actually interrupts the process
  mid-write — Unit 2d's crash/idempotency suite is where that belongs.
- `cli.py`'s identical-retry read of `report.json` catches `ValidationError`
  (a corrupted or truncated file) but not `OSError` (a hand-deleted
  `report.json` inside an `investigations_dir` that still exists) — a
  narrow gap, since `finalize_investigation`'s own staging-then-atomic-
  replace write is what makes that combination unlikely outside deliberate
  tampering. Correctness's own P3, recorded rather than fixed speculatively.
- The composite-primary-key race two identical concurrent `approve` calls
  can lose is fail-safe (the loser's `INSERT` hits `IntegrityError`, refused
  rather than double-resumed) but misdescribed: it surfaces as
  `CONFLICTING_DECISION` even though nothing about the two requests
  actually conflicts, only their timing did. Correctness's P3; the fix
  (re-reading the row after a caught `IntegrityError` and treating a match
  as success) is a real design question, not a one-line change, so it is
  recorded rather than built speculatively.
- `resume_graph_investigation` takes `decision`/`rejection_note` as two
  loose parameters rather than the validated `OwnerDecision` object `cli.py`
  already has in hand — correctness flagged this as *less* pressing after
  this round, since layer 2 (`_parse_resume_decision`) is now stripped and
  mutation-tested in both directions independently of what `cli.py` passes
  in, closing the specific gap a looser signature would otherwise leave
  open.
- `run_decision_command`'s `investigations_dir` is keyed by `thread_id`
  while `finalize_investigation` writes under `report.investigation_id` --
  equal today by construction (`investigation_id` doubles as `thread_id`
  throughout this milestone), but two names for one value with no assertion
  tying them together. Drift-safe today; worth a named invariant if a
  future unit ever lets them diverge.
- `evaluation.py:195`'s blindness to `escalation.decision`, recorded when
  Unit 2b shipped, is unchanged: an owner-rejected diagnosis still scores as
  correct. Still open, still outside this unit's scope. **Closed in Unit
  2d.**
- `doctor.py`'s `ProjectPaths` still has no accessor or write-probe for
  `checkpoints.db`, and now implicitly covers `owner_decisions` too, since
  both live in the same file. Deferred alongside the error-translation gap
  above for the same reason. **Closed in Unit 2d** (the `checkpoints_db`
  accessor and `check_checkpoint_database` probe below) -- narrower than a
  general write-probe: it reads an *existing* file, which
  `check_writable_directories` cannot prove anything about.

### Unit 2d — crash durability, the scorer, and closing the milestone

Closes Milestone 2. Four items, plus the two spec amendments recorded at
`TECHNICAL_SPEC.md` §5 and §8 above: the crash/idempotency test suite
(`TECHNICAL_SPEC.md:186-187`'s last outstanding named deliverable), the
`ControlCounts.unsettled` field and the `disposition_correct` rejection fix
in `evaluation.py`, SQLite connection-open translation at both remaining
untranslated sites plus a `checkpoints.db` doctor probe, and the
`ApprovalError`→`CheckpointStoreError` rename.

**Measured before implementing, not assumed:**

- The settled-thread no-op claim `run_decision_command`'s docstring made
  (`Command(resume=...)` against an already-settled thread returns the
  finished state unchanged) was run directly against production entry
  points — pause, resume once to settle, reopen a fresh `SqliteSaver`,
  resume again — before any recovery test was written around it. Confirmed
  true: identical report bytes, zero additional model calls. The recovery
  path was untested, not broken.
- `sqlite3.connect(...)`'s actual failure surface, measured directly: a
  read-only `results/` directory with no existing `checkpoints.db` makes
  `connect()` itself raise `OperationalError` immediately, because it has to
  create the file. A *corrupt but already-existing* file passes `connect()`
  (SQLite's own connect is lazy) and only raises on first `execute()` —
  for `_sqlite_checkpointer`, that happens deep inside
  `compiled.invoke(...)`/`.get_state(...)`, past any wrap at the connection
  open. The connection-open translation below covers the first case only;
  the doctor probe covers the second. The read-only-directory trigger above
  is POSIX-only — `os.chmod` does not restrict directory access on Windows
  — so the two connect-time tests below
  (`test_investigate_reports_a_locked_checkpoint_store_instead_of_a_traceback`/
  `test_approve_reports_a_locked_checkpoint_store_instead_of_a_traceback`)
  no longer use it; they occupy `checkpoints.db`'s own path with a
  directory instead, which produces the identical `OperationalError` at the
  identical `connect()` call on every platform, since SQLite's file-open
  logic refuses a directory the same way everywhere.

**`ControlCounts` gains `unsettled`.** `count_control` counts
`receipt.state is ReceiptState.RESERVED`, not the more cautious compound
`policy_result is ALLOWED and state is RESERVED` it might look like it
should be — checked against `tool_wrappers.py:142-153` (every `RESERVED`
receipt is constructed `policy_result=ALLOWED`) and `:205-222` (`record()`,
the only path that ever writes a `DENIED` receipt, refuses anything not
already `SETTLED`), so no `DENIED` receipt can be `RESERVED` by construction
and the simpler predicate is the same set, not an approximation.

**`disposition_correct` accounts for `escalation.decision`.**
`report.disposition` stays `DIAGNOSED` on a reject — rejection deliberately
preserves the assessment — so `disposition_correct` now also checks
`report.escalation is None or report.escalation.decision != "reject"`.
`diagnosis_correct` (root-cause match) is deliberately left alone: it
answers a factual question about what the model proposed, independent of
what the owner did with it. Conflating the two would make a
rejected-but-correct diagnosis indistinguishable from a rejected-and-wrong
one, losing exactly the signal a diagnostic-quality evaluation needs.

**SQLite translation, at both remaining connection opens, scoped
deliberately.** `_sqlite_checkpointer` (`cli.py`) and `run_decision_command`'s
`owner_decisions` connection (`cli.py`) both wrap their `mkdir`/`connect`
pair in `try`/`except (OSError, sqlite3.Error)`, raising
`CheckpointStoreError(STORE_UNAVAILABLE)`. This closes the can't-create/
can't-open class (measured above) and does **not** claim to close the
corrupt-but-openable-existing-file class — that residual gap is recorded
below and is exactly what the doctor probe exists for.

**The store error type, renamed.** `ApprovalError`/`ApprovalReasonCode` →
`CheckpointStoreError`/`CheckpointStoreReasonCode`, staying in
`approvals.py`. All five members kept their names and values. `_sqlite_
checkpointer` is used unconditionally by a plain `causalops investigate`,
with no approval anywhere in that path; keeping the old name once that
caller needed the same type too would have left it wrong at half its call
sites. Both connections open the identical physical file
(`results/checkpoints.db`), so this is one resource's exception type, named
for the first feature that happened to use it, not two domains needing two
vocabularies.

**The `checkpoints_db` doctor probe.** `ProjectPaths.checkpoints_db` (new
accessor, replacing two independent literal spellings in `cli.py`) names
`results/checkpoints.db`. `check_checkpoint_database` passes without
touching the filesystem when the file does not exist yet (a project that
has never run `investigate`); when it exists, opens it and runs
`SELECT name FROM sqlite_master LIMIT 1` — not `SELECT 1`, which needs no
page from the file at all and so does not reliably detect corruption.
Measured directly: on the `uv`-pinned SQLite 3.53.1 this project runs on,
`SELECT 1` happens to raise against the corrupt-file fixture too, but on
the system's SQLite 3.46.1 it returns no error on the exact same file —
the check's entire job, silently skipped. `sqlite_master` raises on both
builds. Failure reports `CHECKPOINT_DATABASE_UNREADABLE` on
`sqlite3.Error`/`OSError`; the `is_file()` existence check runs *inside*
the same `try` as the query, not as a guard before it, since `is_file()`
itself can raise `PermissionError` on a directory this process cannot
stat (a root-owned `results/` from the Docker lab) — `run_doctor_command`
runs outside `main`'s own `try`/`except`, so a raise there would have
crashed the command whose job is reporting a broken machine without
crashing. Inserted between `writable_directories` and `docker` in
`run_doctor`'s check order — eight checks now, not seven.

**Crash/idempotency tests, one per named transition:**

| Transition | Test | Note |
|---|---|---|
| Tool receipt `RESERVED`→`SETTLED`, durable half | `test_a_raising_backend_leaves_a_durable_reserved_receipt` (`test_checkpointing.py`) | The in-process half has existed since 1d-1; this closes the durable half `tool_wrappers.py:24-28` promises — a fresh `SqliteSaver` connection, opened only against the file path, reads the `RESERVED` receipt back. The authorize-to-reserve window is genuinely vacuous (one synchronous call, no I/O in between, `tool_wrappers.py:130-153`) — stated in the test module rather than written as a test that would prove nothing. |
| Approval row exists, thread never actually resumed | `test_a_retry_after_a_crash_before_resume_still_settles` (`test_approvals.py`) | Extends 2c's own `test_the_decision_is_recorded_before_the_graph_is_resumed` scaffold: after the identical crash, remove the injected fault and retry — the thread is still genuinely paused (`Command(resume=...)` was never called at all), and the retry must find the existing row, skip the redundant write, and actually resume. |
| Approval row exists, graph settled, artifacts never written | `test_a_retry_after_a_crash_before_finalize_still_writes_artifacts` (`test_approvals.py`) | The transition deferred from 2c. `_write_investigation_artifacts` is monkeypatched to crash after a genuine resume settles the graph; a retry must rebuild the graph, resume the now-settled thread again (a measured no-op, not a docstring claim — confirmed by `_checkpoint_snapshot`'s own assertions before the retry: no `.interrupts`, `.next == ()`, a real `report` already in `.values`, the identical read `run_decision_command` itself performs to decide pending-vs-settled), and finish the write. `finalize_investigation`'s own staging-then-atomic-rename write needed no change — a crash before it completes leaves no `investigations_dir`, so a second call proceeds normally. |

Both new recovery tests restore only the one `cli` attribute they patched
(`cli.resume_graph_investigation` / `cli._write_investigation_artifacts`),
never `monkeypatch.undo()` — that would also revert the test's own
`monkeypatch.chdir`, which cost one round of `FAIL THREAD_NOT_FOUND`
failures to notice.

**Mutation-tested, against the real tree** (`PYTHONPATH=src`, `__pycache__`
purged before each run, each mutation reverted immediately after its
failure was observed):

| # | Control | Mutation | Result |
|---|---|---|---|
| 1 | `count_control`'s `unsettled` predicate | forced to `False` | `test_control_counts_a_reserved_receipt_as_unsettled` failed (`0 == 1`) |
| 2 | `disposition_correct`'s rejection guard | dropped `and not rejected` | `test_an_owner_rejected_diagnosis_does_not_score_as_correct_disposition` failed |
| 3 | `check_checkpoint_database`'s `FAIL` branch | swallowed the exception, fell through to `PASS` | `test_a_corrupt_checkpoint_database_fails_with_a_stable_code` failed |
| 4 | `check_checkpoint_database`'s existence guard | `if not db_path.is_file()` forced to `if False` | Both `test_scratch_file_leaves_nothing_behind` (pre-existing) and the new `test_a_missing_checkpoint_database_passes_without_creating_one` failed |
| 5 | `_sqlite_checkpointer` / `run_decision_command` connection-open wraps | removed both `try`/`except` blocks | Both `test_investigate_reports_a_locked_checkpoint_store_instead_of_a_traceback` and `test_approve_reports_a_locked_checkpoint_store_instead_of_a_traceback` failed with an uncaught `sqlite3.OperationalError`, exactly the pre-2d behaviour |
| 6 | `run_decision_command`'s `investigations_dir.is_dir()` fall-through guard | forced to `True` (always take the identical-retry shortcut once a row exists) | Both new recovery tests failed with `FileNotFoundError` reading a `report.json` that was never written |
| 7 | `ProjectPaths.checkpoints_db` | typo'd the filename | **Not caught** by the doctor probe's own tests on the first pass — every one of them builds its fixture file *through* the same accessor it then checks, so the typo renamed both sides together and passed unnoticed. Caught only indirectly, by unrelated tests elsewhere in the suite that hardcode the literal path (`test_investigate_leaves_a_checkpoint_database_in_a_fresh_project` and most of `test_approvals.py`). Added `test_checkpoints_db_names_the_file_cli_py_actually_opens`, pinned against the literal `tmp_path / "results" / "checkpoints.db"` independently of the accessor — this is the control's own dedicated catch, not a borrowed one |
| 8 | `check_checkpoint_database`'s `is_file()` guard | moved back outside the `try` (correctness's P1-2) | `test_an_os_error_from_is_file_fails_cleanly_instead_of_raising` failed, reproducing the exact unhandled `PermissionError` traceback the review found — the command whose job is reporting a broken machine crashed instead. Renamed from `test_an_unreadable_results_directory_fails_cleanly_instead_of_raising` during the Windows portability fix below, which also switched the fault from `os.chmod` to a `Path.is_file` monkeypatch scoped to `checkpoints_db`'s own path, since `os.chmod` does not restrict directory access on Windows; the mutation still fails the same way after the rename |
| 9 | The two crash-recovery tests' `_checkpoint_snapshot` assertions | swapped between `test_a_retry_after_a_crash_before_resume_still_settles` and `test_a_retry_after_a_crash_before_finalize_still_writes_artifacts` (correctness's P2-2) | Both failed — the paused window does not satisfy the settled assertions and vice versa, confirming the two windows are genuinely distinguished, not copy-pasted boilerplate that would pass regardless of actual state |
| 10 | `count_control`'s `unsettled` predicate, scored against a real crashed run | forced to `False` | `test_a_real_crashed_receipt_scores_as_unsettled` (`test_graph.py`) failed (`0 == 1`) — the same predicate mutation 1 already covers against a hand-built fixture, now also covered against a production crash |
| 11 | `check_checkpoint_database`'s query | reverted from `SELECT name FROM sqlite_master LIMIT 1` to `SELECT 1` (simplicity's P1-1, reverted to check the fix itself) | **Not caught locally.** On the `uv`-pinned SQLite 3.53.1 this project runs on, `SELECT 1` happens to raise against the corrupt-file fixture too, so `test_a_corrupt_checkpoint_database_fails_with_a_stable_code` stays green under the mutation. Verified as a real gap instead, by reproducing independently against the system's SQLite 3.46.1 (`/usr/bin/python3`, distinct from `uv`'s venv): the identical corrupt file passes `SELECT 1` with no error there and would silently `PASS`, while `sqlite_master` raises on both builds. Correctness went on to try to falsify "not locally catchable" with five further corruption shapes against the pinned 3.53.1 build — a zeroed schema page, a junk schema page, clobbered `sqlite_master` row bytes, truncation to the 100-byte header, and plain garbage — and found all five raise on `SELECT 1` too, so there is no corruption shape that discriminates locally on this build at all. Recorded here rather than left as an unexplained query: without this row, "simplify this back to `SELECT 1`" looks free to a future editor holding only this repository. |

Mutation 7's first result is worth keeping on record: it is the same defect
class this project has now found repeatedly — a test that shares the exact
mechanism it is supposed to be checking cannot see that mechanism break.
Mutation 11 is the opposite lesson, worth keeping beside it: some fixes are
correct and necessary but cannot be proven locally at all, on any test, no
matter how the fixture is built — the only honest response is to disclose
the gap and record the out-of-repository evidence that closes it, not to
manufacture a local catch that does not exist.

**Spec citations re-grepped after the two amendments.** The §5 amendment (8
lines) shifted the Approval idempotency-key paragraph from `:162-164` to
`:170-172` (4 call sites updated: `approvals.py`, `cli.py`, `graph.py`,
`test_approvals.py`) and the §8 interrupt-payload paragraph from `:264-265`
to `:272-273` (1 call site, this file, above). `:140-142` (the `run_id`
requirement) sits entirely above both insertion points and is unchanged.
One citation, `graph.py:139`'s `TECHNICAL_SPEC.md:155-158` for the
model-request idempotency key, was already one line off the paragraph it
names (`:156-159`) before this unit touched the file — pre-existing, not
shifted by this unit's edits, left unfixed as out of scope for a citation
sweep triggered by this unit's own insertions.

**Correction, found by correctness review:** the sweep above covered every
*existing* citation of a moved range but missed two *new* ones this unit
wrote. The §5 amendment landed first in the document and pushed 2c's own
`*Amendment, Unit 2c:*` paragraph from `:166` to `:174` — so by the time
this unit's own §8 amendment was written, further down the same file, and
cited that paragraph as "the same structural gap `:166`'s Unit 2c amendment
recorded" (twice, `TECHNICAL_SPEC.md:324` and `:331`), the `:166` in that
new text was already wrong the moment it was typed, before this unit ever
ran a citation sweep. Fixed to `:174` at both sites. The lesson `CLAUDE.md`
already records from Unit 2c — "when an amendment moves spec content,
re-grep every citation of the moved range" — needs one clause added: re-grep
the text you are writing in the same edit, not only the text that already
existed before it, since a later paragraph in the same file can cite an
earlier one that your own earlier edit already moved.

**Known gaps, deliberately not this unit's to close:**

- The corrupt-but-openable-existing-`checkpoints.db` case: `sqlite3.connect`
  succeeds against it (lazy), and the failure only surfaces once something
  actually reads or writes it — for `_sqlite_checkpointer`, deep inside a
  LangGraph call this unit does not wrap. The doctor probe is the intended
  defense, proactive rather than reactive; widening the connection-open
  translation to cover it would mean wrapping every `compiled.invoke`/
  `.get_state` call site, materially more surface than "translate both
  connection opens" scoped.
- `evaluation.py:195`'s blindness to `escalation.decision`, recorded when
  Unit 2b shipped and still open through 2c, is closed by this unit's
  `disposition_correct` fix above.
- `SCORER_VERSION` stays `"1"`. **Reworded, found by correctness review:**
  the original justification here called this change "additive, existing
  counters keep identical meaning" — true of `ControlCounts.unsettled` (a
  new field, old counters unchanged), false of `disposition_correct` (an
  *existing* field): the same `report`/`expected` pair can score differently
  under this unit than it did before it, which is a behavioural change, not
  an additive one. The actual reason the version stays put is narrower and
  does not depend on that claim: no `evaluate` command exists yet, so no
  evaluation record has ever been produced under version `"1"` for a later
  one to disagree with — there is nothing to migrate, additive or not.
  Flagged for reviewers rather than settled unilaterally — the
  counter-argument is that `SCORER_VERSION` exists specifically to
  distinguish scorer outputs, and a scorer that scores the same input
  differently is arguably what it is for.
- §8's third owner option (approve one additional check, routed to
  `DISPATCH_TOOL`) remains deferred by this unit's own §8 amendment, not
  built — recorded above under Milestone 2's status, not repeated here.

### Post-Milestone-2 fix — `test_graph_frozen_reports.py`'s `duration_ms` was never a literal

`master` was green at `d6f06cd` by luck, not correctness. Reading the Windows
CI logs for the portability fix above turned up a second, unrelated defect
in the same file: the branch run for `e6eb574` **failed on Windows**
(`('check_finished', {'outcome': 'EXECUTED', 'duration_ms': 15}) !=
('check_finished', {'outcome': 'EXECUTED', 'duration_ms': 0})`) while the
merge run for `d6f06cd` — an **identical tree**, `git diff e6eb574 d6f06cd`
is empty — passed on Linux. `evidence.py:92`'s `executed_check` computes
`duration_ms=int((time.monotonic() - started) * 1000)`: a real measurement
of the backend call. A fast Linux JSONL read truncates to `0`; the same call
took 15 ms on Windows. Six of this file's frozen tuples pinned `duration_ms`
to an exact `0`. That was never a fact about the graph; it was an assertion
that the test machine is fast.

**Why this is a legitimate reason to touch a file byte-identical since
Milestone 1.** The project's rule — "if a literal ever moves, that is a
finding about the design, not a literal to update" — exists to stop an
inconvenient literal from being quietly edited away. This is the opposite
case, and the file's own text proves it: its module docstring already
stated, before this fix, *"Wall-clock fields (`latency_ms`, `started_at`,
`finished_at`) are excluded, as they always were."* `duration_ms` is a
wall-clock field by that same definition — timed with real `time.monotonic()`
around the real backend call, not the injected `StepClock` that
`latency_ms`/`started_at`/`finished_at` use — and six literals a few hundred
lines below that sentence pinned it anyway. The file documented a contract
and then broke it in its own literals. Fixing that is closing a gap between
the file's stated design and its actual content, not updating an
inconvenient number.

**The fix: exclude the value, assert the shape.** `dispatch_events` now
strips `duration_ms` from `check_finished`'s fields before comparison (a new
`_drop_duration` helper), the same way `latency_ms`/`started_at`/
`finished_at` were already excluded. A new
`assert_check_finished_durations_are_measured` asserts, separately, that
every `check_finished` event still carries a `duration_ms` that is present,
an `int`, and non-negative — the same invariant `ToolReceipt.duration_ms`
already enforces via `Field(ge=0)` (`domain.py:293`), asserted again at the
event-fields layer because `RunEvent.fields` is a plain, unvalidated
`dict[str, JsonValue]`, not a validated model. The six literals lost only
their `"duration_ms": 0` entry (e.g. `("check_finished", {"outcome":
"EXECUTED", "duration_ms": 0})` → `("check_finished", {"outcome":
"EXECUTED"})`); nothing else about them moved.

**A permanent regression test, not just a mutation check.**
`test_a_simulated_slow_machine_still_matches_the_frozen_report` monkeypatches
`time.monotonic` to advance 15 ms on every read — reproducing Windows'
exact measurement — and reruns the two-executed-check scenario. The frozen
comparison passes, and the test additionally asserts the two `check_finished`
events measured `[15, 15]`, not `0`, proving the fix does not depend on the
backend being fast rather than merely having not yet been unlucky. Nothing
in the suite forced a slow measurement before this test, which is why the
flake stayed latent through every unit since 1c.

**What is unaffected.** `git diff d6f06cd -- tests/unit/test_graph_frozen_reports.py`
touches only: the module docstring's "what each test pins" paragraph,
`dispatch_events`, the two new helpers, the six `duration_ms` removals, five
new one-line calls to `assert_check_finished_durations_are_measured`, and
the new test appended at the end. Every id, digest, disposition, receipt
shape and evidence kind in every one of the five original tests is
byte-identical to `d6f06cd`.

**Sibling search.** `executed_check` — the only function in the codebase
that measures `duration_ms` with real `time.monotonic()` — is called
exclusively from `telemetry.py` (×3) and `prometheus.py` (×1), the real
backends `run_once`'s registry wires; it is never called directly from a
test. `time.monotonic` appears nowhere in test code before this fix.
Every other `duration_ms`/`latency_ms` literal in the suite
(`test_report.py`, `test_domain.py`, `test_evaluation.py`,
`fake_incident.py`, `test_tool_wrappers.py`) is a value a test passes
*into* a `ToolReceipt`/`CheckOutcome`/`EfficiencyMetrics` constructor by
hand, never one measured by the code under test. `graph.py:356`'s
`latency_ms` and the report's `started_at`/`finished_at` are computed from
`domain_clock`, which in every test is the fake, deterministic `StepClock`
— not a Windows/Linux hazard, and already excluded from frozen comparison
for an unrelated design reason. These six `duration_ms` literals were the
whole exposure.

**Empirical confirmation, not just the grep above.** Correctness reviewed
`test_a_simulated_slow_machine_still_matches_the_frozen_report`'s frozen
`final_context_digest` (`44e5043842b3e3701b183c4b995d8d7e1935021daaba017c8321d0fff4fc802b`)
against `test_the_graph_reproduces_the_frozen_report_for_two_executed_checks`'s
— the same scenario under a real 15 ms measurement and a real 0 ms one.
They are identical. That is empirical proof `duration_ms` never fed the
context digest, not just an inference from reading `graph.py`'s digest
construction — the sibling search above established the claim by grep; this
test proves it by running two different clock speeds through the same
scenario and getting the same digest.

**What the helper's three assertions actually cover.** Of
`assert_check_finished_durations_are_measured`'s three checks — presence,
`isinstance(int)`, `>= 0` — only the last overlaps `ToolReceipt.duration_ms`'s
existing `Field(ge=0)` (`domain.py:293`). Presence and type are covered by
no validation anywhere else, because `RunEvent.fields` is an unvalidated
`dict[str, JsonValue]`: nothing stops a future edit to `graph.py`'s event
emission from dropping the key or changing its type, and nothing but this
helper would notice. The mutation table's row 3 below exercises exactly that
gap.

**The monkeypatch target is process-global, deliberately.** The new test
patches `evidence_module.time.monotonic`, not an attribute scoped to
`causalops.evidence` alone — `evidence_module.time is time`, the same
module object `telemetry.py:106/161/205` reads `started = time.monotonic()`
from before handing it to `executed_check`. A narrower patch would leave
`started` on the real clock while `executed_check`'s own read used the
patched one, producing a negative delta instead of 15 ms. The test file
carries this as an inline comment at the patch site so a future edit that
"tightens" the target does not silently break it.

**Mutation table** (each row: mutation applied to a clean tree at `d6f06cd`
plus this fix, one at a time, reverted before the next):

| # | Mutation | Result |
|---|---|---|
| 1 | Full reproduction, not a partial revert: `tests/unit/test_graph_frozen_reports.py` restored verbatim to `git show d6f06cd:...` (all six `"duration_ms": 0` literals present, no helpers), with only the 15 ms `time.monotonic` monkeypatch injected into the two-executed-check test | Failed with the CI log's exact line, character-for-character: `AssertionError: assert [('proposal_r...ion_ms': 15})] == [('proposal_r...tion_ms': 0})]` / `At index 2 diff: ('check_finished', {'outcome': 'EXECUTED', 'duration_ms': 15}) != ('check_finished', {'outcome': 'EXECUTED', 'duration_ms': 0})` — this is the original bug, reproduced on demand from the untouched pre-fix file rather than only observed once in a Windows CI log |
| 2 | `graph.py`'s `check_finished` event forced to emit `duration_ms=-7` instead of the real receipt value, `assert_check_finished_durations_are_measured` left intact | All six tests failed on `assert duration >= 0` — exercises only the one assertion that already overlaps `Field(ge=0)` upstream |
| 2 (continued) | Same `-7` injection, the helper's body replaced with a no-op | The five original tests **passed** — the corrupted value slipped through silently once the shape assertion was disabled |
| 3 | The realistic regression: the `duration_ms=` kwarg deleted entirely at `graph.py:737` (someone drops the field, rather than corrupting its value), helper intact | All six tests failed on `assert "duration_ms" in event.fields` — the presence assertion, which nothing else in the suite covers |
| 3 (continued) | Same deletion, helper neutralized | The five original tests **passed silently** — `dispatch_events` no longer carries the key at all, so nothing in the frozen comparison can observe it missing. Only the sixth test (the new regression test) failed, and only on its own separate literal indexing (`event.fields["duration_ms"]` → `KeyError`), not the shared helper — itself confirming the helper was the sole guard for the other five |
| 4 | `_drop_duration` widened to strip each surviving key in turn, one at a time: `outcome`, `tool`, `reason`, `message` | `outcome` → 6 tests failed. `tool` → 6 failed. `reason` → 2 failed (the two tests with a `proposal_denied` event). `message` → 2 failed (same two). Every remaining key in the dispatch-vocabulary field dicts is still pinned by at least one test — the fix does not over-strip |

## Milestone 3 — Local retrieval and evidence-backed portfolio release

**Status:** not started. Adds curated FTS5 runbooks, retrieval provenance,
and injection/no-ground-truth-leakage tests — deferred here from Milestone 2
by `TECHNICAL_SPEC.md` §12's *Amendment, Milestone 2*. Runs the fixed paired
evaluation under the USD 5 cap (Unit 3b-3, raised from USD 2), saves raw
records and limitations, produces architecture and threat-model documents,
verifies the clean source commit, and records a short diagnosis plus
abstention/escalation demo.

### Open gaps recorded during Unit 3b-1's review — for Unit 3b-2

Unit 3b-1 (the `ToolCallingModel` protocol, routing a malformed/ambiguous
tool call through the ordinary repair-then-fail-safe path instead of a
crash, `py.typed`, and a loopback-only network guard for the test suite) is
in review, not yet landed at the time this was written. These facts
surfaced during that review and matter specifically to whichever unit adds
the live model adapter (3b-2); recorded here so they are not lost to a
review thread once the unit lands and this document's as-built record for
it is written.

- **`mypy tests` (the whole directory, not a single file) aborts with a
  duplicate-module error** once `tests/conftest.py` exists alongside
  `tests/unit/conftest.py` — mypy assigns bare module names by filename when
  no `__init__.py` exists anywhere under `tests/`, and two files are both
  named `conftest.py`. The working invocation, re-measured against the
  frozen 3b-1 tree: `uv run mypy tests --exclude 'tests/unit/conftest\.py'`
  → 161 errors in 13 files, 37 source files checked. `--explicit-package-bases`
  does *not* fix this the way it looks like it should: it avoids the crash,
  but changes module-resolution semantics enough to introduce an error that
  is not otherwise there (`import-not-found` for `fake_incident`, 4 → 31
  occurrences) — it trades the crash for a differently-broken check, not a
  restored one. It reports fewer errors overall despite that (153 vs. 161):
  each newly-unresolvable import stops mypy from checking whatever in that
  file depended on it, so the 27 extra `import-not-found` errors suppress
  more downstream errors than they add. Only `mypy src lab` is gated, so
  this has no CI impact; it only affects an
  ad-hoc whole-`tests` sweep — the same kind of check that caught this
  unit's own `py.typed`/`import-untyped` gap (see Unit 3b-1's own review
  record once it lands) — worth knowing it still runs, just not as one
  invocation across the whole directory.
- **The network guard is in-process only.** `tests/conftest.py`'s
  loopback-only guard patches `socket.socket` in the current process; a
  subprocess this suite spawns
  (`tests/security/test_ground_truth_isolation.py:133` does, for an
  unrelated reason) starts with an unpatched `socket` module and is not
  covered. Imports only today, no exposure — an inherent limit of an
  in-process guard, not something the guard itself can close.
- **Windows: the `asyncio` proactor event loop can reach the network
  without going through `socket.socket.connect`/`connect_ex`**, the two
  methods the guard patches. Irrelevant today (nothing in this project is
  async); relevant the moment 3b-2's adapter uses an async client on
  Windows CI.
- **The guard blocks `connect()`, not DNS resolution.** A `getaddrinfo()`
  call that never reaches `.connect()` could still leak a hostname to a
  resolver. Harmless while nothing in the suite resolves a real hostname;
  becomes live the moment `api.anthropic.com` appears.
- **Current live proposal protocol — single native call.** Each
  INITIAL_PLAN/HYPOTHESIS_UPDATE turn binds the five registered check tools
  plus adapter-internal `record_stop`, with `parallel_tool_calls=False`.
  Exactly one native call is accepted: a check includes 2–3 hypotheses and
  its rationale; `record_stop` includes 2–3 hypotheses and a required,
  non-empty stop reason. The provider-facing check schemas omit the duplicate
  `tool` field; `tool_calls.parse_tool_call` validates the registered native
  name and restores the internal discriminator so policy and fingerprints are
  unchanged. Zero, multiple, unknown, malformed, or visible mixed output is
  repaired through the normal model-output path. This supersedes the historic
  two-call `record_plan` reconciliation described below.

### Unit 3b-2 — running the live smoke call

**This is the owner's runbook, not code.** Every code path it exercises is
already covered by `tests/unit/test_live_model.py` and `test_cost_ledger.py`
against a fake transport — no test in this repository ever contacts
Anthropic (`tests/conftest.py`'s network guard is process-wide, and every
test constructs `LiveClaudeModel` with `client=FakeChatAnthropic(...)`, its
own test seam). The smoke call is the one deliberate exception, and it is
never a `pytest` invocation: it is the real `causalops investigate` command,
run once by hand, in a process the guard was never installed in because it
never imports `tests/conftest.py`. See Unit 3b-2's pre-edit report for the
full argument for why this is safe; this section is only the "how," for
whoever actually runs it.

**Preconditions.** `ANTHROPIC_API_KEY` set in the environment (`causalops
doctor` warns, does not fail, if it is not — a replay run needs no key, so
`doctor` cannot use its absence to refuse a command that might not need it).
`LIVE_EVALUATION_MAX_USD` optionally set (`.env.example`'s documented
default, `5.00` as of Unit 3b-3, applies if not). The local Docker lab
running. **No scenario already active for the incident you are about to
start.** Unit 3b-3, reproduced live: running `scenario start` again while a
previous smoke call's incident is still active fails with `FAIL
SCENARIO_ALREADY_ACTIVE`. Reset it first —
`uv run causalops scenario reset <incident-id>` — using the incident id
printed in that earlier `FAIL` line (or in the earlier `scenario start`
output, if you still have it). This is the same "repaired for the failures
already observed, never re-run from the state the first run left" pattern
this section's other two runbook fixes belong to.

**The exact command sequence.** Unit 3b-3, found by the owner actually
running this: `causalops` is a console script installed at `.venv/bin/
causalops`, not on `PATH` — a bare `causalops lab up` fails immediately.
Every command below is prefixed `uv run`, the same way `README.md`'s own
example already is:

```bash
uv run causalops lab up
uv run causalops scenario start configuration_change --seed development
# prints an opaque incident id, e.g. a1b2c3d4e5f6...
uv run causalops investigate a1b2c3d4e5f6... --model claude
```

`--model claude` is the only thing that distinguishes this from an ordinary
replay run. Nothing else about the command changes.

**What a successful run's artifact contains** — all of it produced by the
same `finalize_investigation` path an ordinary replay run already uses, not
anything new this unit built:

- `results/investigations/<investigation-id>/report.json` — the
  `InvestigationReport`. `usage` is populated (unlike every replay run's
  `usage: null`), so `limitations` will *not* contain "this model reports no
  token usage" — the one frozen-literal difference this unit's pre-edit
  report flagged as expected to move, and here is where it moves.
- The rendered markdown report, labelled with the real model
  (`live_model.MODEL_NAME`, `"claude-sonnet-5"`), not `"replay"` —
  `_resolve_thread_incident_and_model`'s fix, provable end to end only by
  an actual live run, since no replay-backed test ever exercises the
  resume-path label bug this closes.
- `results/investigations/<investigation-id>/events.jsonl` — the ordinary
  `RunEvent` stream, unchanged in shape from a replay run.
- `results/checkpoints.db`'s `cost_ledger` table: **exactly one row per
  model call this run made**. Unit 3b-3, found by the owner actually
  running this: the `sqlite3` CLI binary is not a dependency of this
  project and may not be installed, even though the `sqlite3` *Python
  module* (stdlib, used throughout `src/`) always is. Read the table with
  `uv run python` instead of assuming a `sqlite3` binary on `PATH`. A
  multi-line heredoc does not survive a copy-paste from inside this list
  item -- Unit 3b-3's own review found and reproduced this: every line,
  including a `<<'PY'` heredoc's closing `PY`, inherits this list item's
  2-space indent, and a heredoc terminator must sit flush at column 0 or
  the shell never sees it end. A single-line invocation has no terminator
  to misalign, so that is what this runbook uses instead:

  ```bash
  uv run python -c "import sqlite3; conn = sqlite3.connect('results/checkpoints.db'); conn.row_factory = sqlite3.Row; [print(dict(r)) for r in conn.execute(\"SELECT * FROM cost_ledger ORDER BY reserved_at DESC LIMIT 5\")]"
  ```

  Edit the `LIMIT` (or add a `WHERE run_id = '<run-id>'`, `run_id` is
  internal, not printed by the CLI) to match how many model calls the run
  actually made. Each row's `state` is `SETTLED` (a row still `RESERVED`
  after the process exits means a crash or timeout interrupted that
  specific request — see below), `reserved_usd` is the pessimistic upper
  bound charged against the ceiling — priced off prose *and* the fixed
  tool-definition schema every call sends (`live_model.py`'s `_send`; the
  input-token *cap* in "Default limits" above stays prose-only, a
  deliberately different scope) — `actual_usd` is what the request really
  cost from the provider's own reported `input_tokens`/`output_tokens`,
  and `actual_usd` must be `<= reserved_usd` on every row. `test_pricing.
  py`'s `test_a_settled_request_never_costs_more_than_its_own_reservation`
  pins this as a property of the pricing math itself, not just something
  that happened to hold this run; `test_live_model.py`'s
  `test_propose_reserves_at_least_the_full_wire_payload` additionally pins
  that a real `propose()` call's reservation genuinely counts the tool
  payload, not just prose — the P1-1 bug this unit fixed shipped past an
  earlier version of the `test_pricing.py` assertion that priced both
  sides off the identical number and so could never observe the omission.

  This invariant holds whenever the pessimistic estimate
  (`pricing.estimate_input_tokens`) actually bounds the tokens the
  provider bills. **Unit 3b-3's smoke call checked it against a real
  billed request for the first time and it held on both settled rows —
  but only because output stayed under its allowance both times.** See
  "The smoke call's findings" below for the measured numbers and what
  changed as a result. If a row ever does show `actual_usd >
  reserved_usd`, suspect the two unmodelled contributions
  (provider-side tool-use and message-envelope overhead, which
  `estimate_input_tokens` deliberately does not receive) first,
  and treat it as **a signal to re-derive `PESSIMISTIC_CHARS_PER_TOKEN`,
  not an incident**: the violation is bounded to a fraction of a cent per
  row, and the application-wide ceiling still counts the *reserved*
  amount against `LIVE_EVALUATION_MAX_USD` regardless — a mispriced row
  does not let spend run away, it only means this one row under-priced
  itself.

**Record the estimate beside the settled row — every live call is a
calibration point `pricing.py`'s own docstring asks for.** The call is
happening anyway, its prompt is known, and its settled `cost_ledger` row
carries the provider's own `input_tokens`. Before running it, compute
`estimate_input_tokens(system_text + content)` plus
`estimate_input_tokens(json.dumps(tools))` for the turn you are about to
send (`system_text`/`content` are what `_send` actually estimates;
`tools` is whatever `propose`/`respond` bound for that stage). After the
run settles, compare that sum against the row's real `input_tokens`. If
the estimate is still `>=` the billed count, that is one more confirming
point for the current ratio. If the estimate comes in *below* the billed
count, that is real evidence `PESSIMISTIC_CHARS_PER_TOKEN` needs to move
again — with the measurement as the reason, not a guess.

**What an escalated run looks like — a normal outcome, not a failure.**
The graph can pause for owner approval on an ordinary `investigate` run
(Milestone 2's escalation interrupt), and a live model's plan is not
scripted the way a replay fixture is, so this is a real, reachable outcome
of the smoke call, not a hypothetical. Escalation is only reachable from
`final_assessment`, so two to four model calls — and their settled
`cost_ledger` rows — exist by the time it can fire. `run_investigate_command`
(`cli.py:465-476`) never calls `finalize_investigation` on this path — no
`report.json` and no `events.jsonl` exist yet. Instead the command prints
`ESCALATED <reason> <thread_id>` and `remaining checks: N`, and exits `3`
(`EXIT_ESCALATED`), distinct from the `0`/`1` of a settled run. **The
`cost_ledger` rows for the model calls already made are `SETTLED` at this
point — the money is spent** before the pause, since settlement happens
per model call, inside the graph, before the escalation interrupt runs.
Run `uv run causalops approve <thread_id>` to accept the paused diagnosis,
or `uv run causalops reject <thread_id> "<reason>"` to reject it; either
produces
the terminal `report.json` and `events.jsonl` this section describes
above. Resuming spends nothing further: `escalation_interrupt` routes
straight to `final_report`, so neither `approve` nor `reject` makes
another model call, and neither needs `ANTHROPIC_API_KEY` set. Seeing exit
code 3 after this call means the gate paused for a decision it is designed
to require, not that anything broke.

**What a refused run looks like — the gate working, not broken:**

- **Cost ceiling refused.** `causalops investigate` prints `FAILED_SAFE
  <root-cause>`, the investigation's `report.json` has
  `reason_code: COST_CEILING_EXCEEDED`, and **no new `cost_ledger` row
  exists for the refused request** — `record_reservation_before_request`
  raises before any insert (`cost_ledger.py`, `test_cost_ledger.py`'s
  `test_a_refused_reservation_writes_nothing`). If this run was meant to
  proceed, either raise `LIVE_EVALUATION_MAX_USD` (after checking why the
  running total is where it is, using the `uv run python` reader above
  with `"SELECT SUM(reserved_usd) FROM cost_ledger"`) or accept that the
  ceiling did its job.
- **Input too large.** `reason_code: INPUT_TOKEN_CAP_EXCEEDED` — the
  rendered context exceeded the 9,600-token pessimistic estimate (Unit
  3b-3, raised from 3,200 — see "The smoke call's findings" below).
  Nothing was sent, nothing was reserved. This should not happen on the
  checked-in scenarios at their default budgets; if it does, that is worth
  investigating on its own before re-running.
- **A crash or timeout mid-request.** The run reports `FAILED_SAFE`/
  `INTERNAL_ERROR`, and the `cost_ledger` row for that specific request is
  left `state = 'RESERVED'` — visible, not silently dropped, per
  `TECHNICAL_SPEC.md` section 5's "the reservation left visible for
  accounting" rule. This is expected, not a bug to chase, unless it
  recurs.
- **A malformed model turn.** Consumes the one repair
  (`Budgets.repairs`), same as replay; a second consecutive failure ends
  the run at `REPAIR_EXHAUSTED`/`MODEL_OUTPUT_INVALID`, same reason codes
  replay already produces.

### The smoke call's findings (2026-08-22) — Unit 3b-3

**The owner ran the first live Claude call in this project's history on
2026-08-22.** Total spend: $0.03771, two model calls (one repair), zero
tools executed, outcome `FAILED_SAFE`/`MODEL_OUTPUT_INVALID`. Recorded here
because it drove real numeric changes to this document and to `pricing.py`,
`cli.py`, and `.env.example` — a dated record, not a silent renumbering.

**What worked**, unchanged by this unit: the cost gate (both ledger rows
`SETTLED`, `actual_usd <= reserved_usd` on both), the context digest
(distinguished the original call from the repair at the same `model_turn`),
3b-1's safe-failure path (one repair consumed, then a clean `FAILED_SAFE`
instead of a crash), and `record_plan` reconciliation (it parsed both
times — the failure was in the domain tool call, downstream).

**The blocker.** `parse_tool_call` failed identically on both calls:
`"tool: Unable to extract tag using discriminator 'tool'"` — Claude never
included the `tool` field in its domain-tool arguments. The schema was not
malformed: `tool` was present in `properties` and forced into `required`,
but pydantic's own `model_json_schema()` also gave it a `"default"`
(confirmed, by set comprehension over all five domain schemas, to be the
*only* property carrying one) — a required field that also names its own
default reads as omittable. Fixed by dropping only the wire schema's
`"default"` key (`live_model.py`'s `_domain_tool_definitions`); `"const"`
and the forced `required` entry are untouched, so `parse_tool_call`'s
confused-deputy check (`call.name` against `arguments.tool.value`) still
compares two independently-validated values, not one injected from the
other. `tests/unit/test_live_model.py`'s `test_domain_tool_schemas_drop_
default_but_keep_const_and_required` and `tests/unit/test_tool_calls.py`'s
`test_a_call_missing_the_tool_discriminator_is_refused` pin this offline;
whether it actually changes what Claude sends is unconfirmed until the
next live call — deliberately the *only* change to the discriminator path
this unit made, so that call's evidence is unambiguous about what worked.

**Open gap, recorded not fixed:** the repair turn's entire correction
message was a bare pydantic-error fragment — `": Unable to extract tag
using discriminator 'tool'"`, an empty `loc` and no guidance a model could
act on — and it is a second, independent candidate cause of the smoke
call's failure alongside the schema defect above: even a perfectly fixed
schema gives a model nothing to correct from if the repair prompt itself
carries no actionable text. Left unfixed on the owner's explicit ruling —
if the schema fix works, the first call succeeds and the repair never
runs, so improving the repair message changes no observed outcome in that
case, and fixing both at once would leave two variables in flight for one
live call's evidence to separate instead of one.

**The calibration.** The failed INITIAL_PLAN turn still composed and
billed real tokens: 9,249 characters sent (1,511 prose + 7,738
pre-3b-3 tool-definition payload), 4,099 tokens billed by the provider — a
real ratio of about 2.26 characters per token. The OLD estimate
(`PESSIMISTIC_CHARS_PER_TOKEN = 3.0`) would have estimated 3,084 tokens for
that same request (`_send` estimates prose and tools as two separate
ceiling divisions, 504 + 2,580 — not one combined division over 9,249
characters, which would give 3,083) — *below* the 4,099 actually billed,
a 33% undercount.
Both settled rows still held `actual_usd <= reserved_usd` only because
output stayed under its allowance on both calls (878 and 1,248 of the
1,600-token allowance); at saturation, the same request would have
measurably violated the invariant ($0.024198 actual against $0.022168
reserved) — exactly the risk the correctness reviewer flagged algebraically
during Unit 3b-2, now measured rather than argued.

**The replan, owner-approved, 100% buffer over the one measured point:**

| Constant | Was | Now |
|---|---:|---:|
| `pricing.PESSIMISTIC_CHARS_PER_TOKEN` | 3.0 | 1.0 |
| `pricing.MAX_INPUT_TOKENS` | 3,200 | 9,600 |
| `cli.DEFAULT_LIVE_EVALUATION_MAX_USD` / `.env.example` | 2.00 | 5.00 |

`MAX_INPUT_TOKENS` moved specifically to hold the *character* budget the
cap actually exists to bound at 9,600 (3,200 × 3.0 = 9,600 = 9,600 × 1.0):
the cap's job was never really "N tokens," it was always "N characters of
prose," and a ratio change without a matching cap change would have
silently re-tightened it. `pricing.py` records the current conservative
estimate; `test_pricing.py`'s `test_the_input_cap_preserves_the_
intended_9600_character_prose_budget` pins the 9,600-character figure
directly rather than deriving it from the two constants it is meant to
guard.

**Why the tool schema stays out of `MAX_INPUT_TOKENS` — corrected.** An
earlier version of this argument (both in this document and, before it, in
Unit 3b-2's own unpinned "512 tokens" claim) used a FINAL_ASSESSMENT turn
with zero evidence as its example and concluded folding the tool schema
into the cap "would refuse ordinary runs." That conclusion did not follow
from its own numbers: at 1,280 tokens of prose against the (then) 9,600 −
7,595 = 2,005-token folded headroom, that specific turn was *admitted*
(1,280 < 2,005), not refused — the same defect (512 < 620) was present in
the original figure and survived the rewrite to the new one. The valid
example is a HYPOTHESIS_UPDATE after runbook retrieval: one check remains,
it can retain all five `Budgets.runbook_passages`, and its proposal-tool
binding is 12,011 characters/tokens. By
contrast, FINAL_ASSESSMENT binds its 2,292-token schema, and its
5,577-character full-runbook context fits the resulting 7,308-token folded
headroom. `test_live_model.py`'s
`test_a_post_retrieval_proposal_sends_when_only_its_schema_exceeds_the_cap`
renders that real proposal-stage shape, proves its prose fits the cap while
prose plus tools does not, and proves the fake transport is still sent. Unit
3b-4's item 6 (below) shrank the tool payload from 7,595 to 6,727
characters; the addendum round's A1/A2 (also below) then grew it back to
**7,020** at that historical point, since both were additive prose fixes in
the opposite direction from item 6's strip. The current strict-schema payload
is 12,011, which is already larger than the 9,600-token prose cap. Folding
proposal schemas into that cap would therefore refuse even an empty proposal
request; final-assessment schemas do not have that problem. The test derives
the proposal-stage folded headroom from emitted definitions instead of
carrying it by hand.

### The second live run and its root-cause investigation — Unit 3b-4

**The owner ran a second live call after Unit 3b-3's discriminator fix
landed.** Cost $0.05560. It confirmed the fix: Claude sent the `tool`
discriminator, a check executed, evidence was collected, and the run
reached `FINAL_ASSESSMENT` for the first time. It still ended
`FAILED_SAFE`/`REPAIR_EXHAUSTED`, on two new failures neither of which was
the discriminator regressing.

**The root-cause investigation that followed found three tiers this
project had been conflating as one contract:** in-schema-and-enforced
(`required`/`enum`/`const`, but only under `strict: true`, which this
codebase does not set), in-schema-but-never-enforced (`maxLength`, numeric
bounds), and not expressible in schema at all (conditional requirement,
cross-field rules — a `model_validator(mode="after")` cannot appear in
`model_json_schema()`'s output). Both new failures fell in the second and
third tiers. **The reviewing error that let this ship:** Unit 3b-3's review
tested the schema against itself ("is `default` consistent with
`required`?"), which cannot detect a mismatch between the schema and the
*application code that refuses the run* — and it read a field's absence
from `required` as evidence the field was optional, which is not true for
a conditionally-required one.

**Historical protocol note.** The following Unit 3b-4 details describe the
then-current two-call `record_plan` protocol. They preserve the live-run and
review evidence, but its implementation and test names are superseded by the
single-call `record_stop` protocol documented above.

**Six items landed, all prose/schema-shaping only except item 1:**

1. **`PlanRecord.stop_reason` (`live_model.py`) is now required-and-nullable
   -- the only behavioural change.** The second run's first new failure was
   `record_plan` omitting `stop_reason` on a turn that proposed no check,
   legal under the old `= None` default. Reproducing Unit 3b-3's successful
   `tool`-field fix meant reproducing its *shape* (`required` membership
   plus no `default`), not its edit (dropping `= None`) — `tool` was
   already in `required`; `stop_reason` was not, so dropping only its
   default would have left it exactly as omittable as before. The field's
   `= None` assignment is gone instead, which pydantic marks required; the
   emitted schema was asserted directly by that unit's then-current
   live-model test.
2. **`record_final_assessment`'s three terminal-disposition invariants**
   (`domain.check_terminal_invariants`: a diagnosis needs a root cause and
   supporting evidence; an abstention needs `UNDETERMINED`) are now stated
   in prose, on `FinalAssessment.disposition`/`root_cause`/
   `supporting_evidence_ids`'s own `Field(description=...)` in `domain.py`
   and in the tool's top-level description in `live_model.py` — the
   validator itself is unchanged.
3. **The 300-character bound is now stated in words** on
   `FinalAssessment.uncertainty`/`next_step`, `Hypothesis.missing_evidence`
   (`domain.py`), and `PlanRecord.stop_reason` (`live_model.py`) — the
   second run's second new failure exceeded `uncertainty`'s bound on an
   *uncorrected* first attempt (an earlier stage had already spent the
   run's one repair), so whether a correction would have fixed it is
   untested; Anthropic's structured outputs do not enforce `maxLength`
   server-side, so prose is the only mechanism that can actually hold a
   model under it. The bound itself was not raised or relaxed.
4. **`FinalAssessment.supporting_evidence_ids` AND `contrary_evidence_ids`
   now tell the model to copy evidence ids exactly.** `graph.py:1069`'s
   `cited = parsed.supporting_evidence_ids + parsed.contrary_evidence_ids`
   feeds both fields into `store.unknown_ids(cited)`, which runs after
   parsing, so a forged or mistyped id in either one is terminal with no
   repair (`ReasonCode.FORGED_EVIDENCE_REFERENCE`). `contrary_evidence_ids`
   was missed in this item's first pass and documented in a follow-up
   round, once item 5's own cross-check (below) surfaced it as a live,
   undocumented instance of the identical gap it was built to find —
   repeating this document's own naming instance-not-class error a third
   time, caught before it cost a run rather than after.
5. **A schema-vs-application cross-check landed as a real test**
   (`tests/unit/test_live_model.py`: `assert_documented_prose_only_contract`
   plus `test_each_known_final_assessment_contract_is_schema_accepted_and_
   app_refused`, its then-current stop-record gap regression, and
   `test_an_undocumented_prose_only_contract_fails_
   the_check`, which demonstrates the guard against a real, currently
   unlisted gap on `search_runbooks.limit` exceeding `Budgets.
   runbook_passages` — `contrary_evidence_ids` was this demonstration's
   original example until item 4's follow-up fixed it, at which point it
   correctly stopped being a gap the demonstration could show): for every
   payload the emitted schema accepts and the application refuses, the
   contract must be named in `KNOWN_PROSE_ONLY_CONTRACTS` with a pointer to
   the prose that carries it, or the assertion fails. This closes the
   *class* of gap the investigation found, not just the five instances it
   found first — and, mid-unit, found a sixth.
6. **Maintainer-only class docstrings no longer ship to Claude.**
   `model_json_schema()` promotes a class docstring to `description` at the
   schema root and at every `$defs` entry; `_strip_maintainer_prose`
   (`live_model.py`) strips both sites, with a whitelist
   (`Hypothesis.__doc__`, "Rank is not a probability," is genuine model
   guidance and stays) — Unit 3b-3's P2-5 fix stripped two leaks by naming
   the two classes it was looking at, and this investigation found three
   more the same defect had reached (`SearchRunbooksArguments.__doc__`,
   `RunbookTopic.__doc__`, `ModelDisposition.__doc__`) because the fix was
   scoped to instances, not the class of the problem. The tool payload
   shrank from 7,595 to 6,727 characters as a result of this item alone --
   the addendum round's A1/A2 (below) later grew it back to 7,020, so
   6,727 is this item's OWN historical effect, not the figure
   `test_the_tool_payload_size_matches_what_pricingpy_assumes` currently
   pins; see "The addendum round," below, for the current number.

**Not approved, explicitly ruled out:** `strict: true` (the installed
`langchain-anthropic==1.6.1`'s `convert_to_anthropic_tool` silently drops a
`strict=` argument for a dict tool definition already carrying
`name`/`description`/`input_schema` — confirmed against the installed
package — so whether the API would even accept the result cannot be proven
offline); raising or truncating the 300-character bound to make a run pass.

**Recorded, not in this unit:** schema bounds exceeding policy budgets
(`query_logs.row_limit` 1–200 vs `Budgets.log_rows` 40;
`search_runbooks.limit` 1–20 vs `runbook_passages` 5 — this one is now
exercised directly by `test_an_undocumented_prose_only_contract_fails_
the_check`, as a real, deliberately-still-open gap, not a fixed one) cost
a model call on denial and are invisible to the model; `Budgets.repairs =
1` is run-wide, not per stage, so the second run's first failure consumed
the only repair before the second failure was ever offered one;
`events.jsonl` does not distinguish "this stage burned its own repair"
from "no budget remained when the stage began."

### The addendum round — correctness's own P1 and a second reviewer's findings

**On top of the Unit 3b-4 freeze above, a second review pass landed one more fold-in
(Group A) plus six independently-verified findings from an independent static review
tool ("codex") run against `master`, split by what they touch: Group B (money-safety,
same trust domain as the live-model work above) and Group C (pre-existing hardening
gaps unrelated to the live-model prose investigation).** Every finding was verified
against the actual code before being approved — one codex claim (an append-only
`CLAUDE.md` convention) was a misreading and is not included below.

**Group A — folded into the 3b-4 prose fixes.**

- **A1.** `evidence_gap`/`expected_observation` (`live_model.py`'s
  `_RATIONALE_PROPERTIES`) carried `maxLength: 300` without ever stating the bound in
  words — the same gap item 3 closed on four other fields, missed here because these
  two are synthetic properties this module injects, outside that sweep's scope. More
  exposed than any field item 3 already fixed: both are REQUIRED on every domain-tool
  call, not once per run.
- **A2.** In the then-current `record_plan` protocol, the tool description said
  "every turn" but never "once per turn"; its duplicate-call refusal already
  said "call it exactly once," and the description was updated to match.
- **A3, record only.** `DUPLICATE_PROPOSAL` (`policy.py`) has no prose anywhere —
  not fixed this round; it has no live-run precedent yet, and the fact is not
  expressible in schema (repetition across turns, not a payload shape).
- **A4, record only.** `KNOWN_PROSE_ONLY_CONTRACTS`'s pointers (`test_live_model.py`)
  are unverified — confirmed real by both reviewers (an equal-length placeholder
  swap leaves all tests green), ruled future-drift risk rather than next-run risk.
  A uniform verification mechanism needs real design (two entries point at
  wire-visible prose, two at `domain.py` validator messages) and is not attempted
  this round.
- **A5, no code change.** `disposition`/`root_cause`'s per-field descriptions
  duplicate the tool-level description in `live_model.py`. Verified deliberate: it is
  unconfirmed whether Anthropic's parser honours a `description` sibling to a
  property's own `$ref`, so collapsing to one copy risks losing the guidance
  entirely if the sibling form is silently ignored. A comment in `domain.py` records
  this so it is not "simplified away" later.

The propose-turn tool payload moved to **7,020 characters/tokens** at that
historical point (up from 6,727, since A1/A2 are additive) — re-derived and re-pinned by
`test_the_tool_payload_size_matches_what_pricingpy_assumes`, with every citation in
the then-current implementation and this document updated to match.

**Group B — the double-spend fix, the most important item in this round.**
`cost_ledger.record_reservation_before_request` returned an existing reservation row
indistinguishably whether it was `RESERVED` (unsettled) or freshly inserted, and
`live_model.py`'s `_send` invoked the provider unconditionally either way. A crash
between reserving and settling, followed by a LangGraph resume that re-renders the
identical stage (same `context_digest`), read back the same ledger row (correct
bookkeeping, no double-counted dollar) but still sent a second real paid request under
it — the exact "reissue an ambiguous model request" `TECHNICAL_SPEC.md` §5 forbids, in
the one scenario the amended idempotency key exists to prevent. No existing test caught
this: `test_an_identical_retry_reads_back_the_same_row_not_a_second_one`
(`test_cost_ledger.py`) only ever asserted the ledger stayed correct, never that the
transport was invoked twice.

`record_reservation_before_request` now returns `(row, is_new)`; `_send` refuses to
invoke the provider when `is_new` is `False`, raising the new
`AmbiguousReservationNotResent` (`cost_ledger.py`, not `live_model.py` — `graph.py`
catches it alongside `CostCeilingExceeded` without ever importing the concrete live
adapter) and reporting the new `ReasonCode.AMBIGUOUS_MODEL_REQUEST`. Both possible
states of a pre-existing row (`RESERVED` or `SETTLED`) are refused the same way, not
two different judgment calls — see the exception's own docstring for why a `SETTLED`
row cannot simply be replayed back (`CostLedgerRow` never stored the model's actual
response, only its cost and token counts). `test_live_model.py`'s
`test_a_pending_reservation_refuses_to_resend_without_touching_the_transport` is the
test codex specifically asked for: it asserts on the fake client's own call count
(`fake.sent == []`), not just ledger state, and is mutation-verified to fail if the
new `is_new` check is removed — with the removed check, the mutation run showed the
fake transport genuinely receiving the second call, the double-spend made visible in
a test for the first time.

**Open gap, recorded not fixed (post-freeze review, P3-4):** a crash after
`settle_reservation` commits but before the LangGraph checkpoint saves leaves that
thread's next resume attempt permanently refusing. The ledger row is genuinely
`SETTLED` (the request really did complete and really was billed), but the
checkpoint never advanced past the node that made it, so every resume re-renders
the identical stage, finds `is_new=False`, and raises `AmbiguousReservationNotResent`
again -- forever, for that specific thread. There is no code path that detects this
narrow window and resumes some other way; the fix that exists is procedural, not
automatic: `AmbiguousReservationNotResent`'s own message now tells the owner directly
("start a fresh investigation instead of resuming this thread"), so the dead end is
visible and actionable rather than a silent, repeating refusal an owner might retry
against forever. Closing it properly would mean either storing enough of the
model's actual response to replay it, or persisting settlement and the checkpoint
in one atomic step -- both are real design changes, not this round's scope.

**Group C — six pre-existing hardening gaps, verified real and independent of the
two observed live-run failures.**

- **C1 (P2, downgraded from codex's P1).** `run_investigate_command`
  (`cli.py`) built `root / "runs" / incident_id` from an unvalidated positional CLI
  argument, with no check that the loaded `StoredIncident.scope.incident_id` matched
  the requested directory. `reset_scenario` (`scenario_control.py`) already had the
  right check; the new `validated_run_paths` extracts it so both callers share one
  implementation instead of a second hand-copy. A single-operator local CLI has no
  separate attacker from victim for the path-traversal framing, but the
  identity-mismatch check is worth having regardless — it is what catches
  `runs/<id>/incident.json` ever diverging from its own directory name, a correctness
  bug a security framing alone would not motivate fixing. Both checks are
  mutation-verified independently: a decoy artifact planted exactly where an
  unvalidated `../decoy` argument would resolve to proves the `isalnum()` check fires
  first, and reverting the identity check alone lets a real (if degenerate)
  investigation attempt through.
- **C2 (P2).** `telemetry.py`'s `within_window` could raise `TypeError` comparing a
  naive `datetime.fromisoformat` result against the aware window bounds it is always
  called with — only `ValueError` (the parse failure) was caught, turning one
  malformed log or change record into a `FAILED_SAFE` for the entire check instead of
  excluding just that record. A naive timestamp is now explicitly rejected (never
  silently coerced to UTC — this project cannot know what offset was intended), the
  same way an unparseable one already was.
- **C3 (P2).** `evidence.py`'s `trim_to_bytes` only shrinks the list it is handed;
  `run_changes_check` builds a SCALAR `summaries` field (joined from every matched
  change's summary text) before calling it, so once the byte-trimming loop emptied
  the list it had nothing left to drop, and the function returned an over-budget
  payload silently if the scalar alone still exceeded `MAX_RESULT_BYTES`. It now
  falls back to shrinking the largest remaining string field once the row list is
  exhausted, and every caller's `row_count`/`change_count`/`edge_count`/
  `sample_count` is now kept equal to `len(kept)` throughout rather than set once
  before trimming and left stale. One real implementation bug was caught and fixed
  during this item's OWN mutation testing: an initial "skip popping rows if it looks
  like it wouldn't help" optimization broke on `run_changes_check`'s actual shape,
  where a kept row's raw dict still carries its own full-size `summary` field — a
  second, undetected copy of the same oversized text the top-level scalar holds, only
  reachable by removing the row itself. `trim_to_bytes`'s own docstring tells this
  story so it is not rediscovered.
- **C4 (P2).** `live_model.py`'s `respond()` used `next(...)` to silently take the
  first of two or more `record_final_assessment` calls in one message, discarding a
  conflicting second one instead of refusing — analogous to the then-current
  proposal-side duplicate-call refusal. Two-or-more matches now refuse the same way zero matches already
  did (empty `content`, routed through the existing repair path).
- **C5 (P2).** `cli.py`'s `main()` catches only `(LabError, RunRecordError,
  CheckpointStoreError)`; three call sites read and validated a stored JSON artifact
  (`incident.json` twice, `report.json` once) with nothing wrapping
  `Path.read_text()`/`model_validate_json()` — a missing file, invalid UTF-8, or a
  malformed JSON body all escaped as raw tracebacks. The new `_load_stored_artifact`
  helper centralizes this, catching `OSError`, `UnicodeDecodeError`, and pydantic's
  `ValidationError`, and reports the new `LabReasonCode.CORRUPT_ARTIFACT` — distinct
  from `INCIDENT_NOT_FOUND` ("there is no such artifact") and from the `report.json`
  site's previous `CheckpointStoreReasonCode.STORE_UNAVAILABLE` (a corrupt artifact
  is not the same fact as an unavailable store; `test_approvals.py`'s existing
  regression test for that site was updated to the more accurate code).
- **C6 (P3, low priority, owner-approved to fix this round).**
  `telemetry.py`'s `registered_check_runner` — superseded by
  `tool_wrappers.dispatch_registry` before `search_runbooks` existed, unused by
  `cli.py` — is now `_registered_check_runner`, a private name rather than a
  public-looking incomplete dispatch seam. Kept, not deleted: `test_telemetry.py`
  still exercises it directly as documented history of why the seam cannot route a
  `search_runbooks` proposal.

**Not approved, unchanged from the base 3b-4 scope:** `strict: true`, the item 2
`anyOf` schema encoding, raising or truncating the 300-character bound, and the
GitHub Actions Node 20 deprecation.

### Second dual review on `a44bf57` — P1, a live_model.py batch, and cleanup

**Both reviewers re-verified the pushed WIP export (`a44bf57`) independently.** Correctness
confirmed all four of codex's own findings from that pass (one worse than reported, one
disputed to a better fix) plus found four more of its own; simplicity found the readability
root cause underneath two of them. The owner disposed of every finding; this section records
what landed.

**P1 — `StoredIncident` identity validation, the serious one.** Correctness traced this past
"a mismatched artifact produces a raw traceback" to something worse: it defeats the
safe-failure guarantee entirely. `graph.py`'s `_rebuild_store` raises `ValueError` on a
mismatched `evidence[i].incident_id`, and that function is called from BOTH the normal
`_build_report` path and the outer crash-containment path meant to catch exactly this kind of
failure — a mismatched artifact that got past loading raises the identical error a SECOND
time, from inside the handler built to catch the first one, and escapes `main()`'s
`(LabError, RunRecordError, CheckpointStoreError)` catch entirely. `StoredIncident.
check_identity_agrees` (`domain.py`, a `model_validator(mode="after")`) now confirms
`scope.incident_id`, `packet.incident_id`, and every `evidence[i].incident_id` agree at LOAD
time, before either graph path ever sees the artifact — composes for free with
`_load_stored_artifact`'s existing `ValidationError` → `LabError(CORRUPT_ARTIFACT)`
translation, no new error handling needed. `EvidenceStore.add`'s own `ValueError` stays
exactly as it is, an internal invariant guard for a run already in progress, not the thing
this fix targets — correctness was explicit that converting it would fix the symptom, not
the cause, since the new validator makes it unreachable from the load path anyway.
`StoredIncident`'s own class docstring now lists all three identity-bearing fields and where
each is checked, the same auditable-list discipline `_RATIONALE_PROPERTIES` and
`KNOWN_PROSE_ONLY_CONTRACTS` already use elsewhere; `cli.py`'s own identity-check comment
(which overclaimed a whole-artifact guarantee while checking only `scope.incident_id`
against the directory name) is now precise about what it checks directly versus what the new
validator already guarantees transitively. Tests: `test_domain.py` pins a packet-mismatch
refusal, an evidence-mismatch refusal, and the positive self-consistent case; `test_
approvals.py`'s existing directory-drift test was rebuilt to construct a SELF-consistent
drifted artifact (the old version could no longer even construct its fixture once the new
validator landed — a good sign, not a broken test).

**Open gap, recorded not fixed (post-freeze review):** `graph.py`'s `_rebuild_receipts` does
not check `ToolReceipt.incident_id` against `state["incident_id"]` the way `_rebuild_store`
now indirectly benefits from checking evidence identity. Checkpoint-sourced, not loaded from
a potentially-tampered file the way `StoredIncident` is, so lower stakes — not fixed this
round.

**The `live_model.py` batch — Finding 3, N1, N2, taken together.**

- **Finding 3.** `respond()` checked "exactly one call is named `record_final_assessment`"
  but never checked the TOTAL call count — a turn with exactly one matching call plus some
  OTHER, unbound tool name would have passed through, silently dropping the extra call the
  same way C4 (the previous round) was built to stop happening for a second MATCHING call.
  Correctness read the installed `langchain-anthropic==1.6.1` source directly
  (`output_parsers.py:80-92`) and confirmed the client copies tool names verbatim with zero
  validation against the bound list — not provably reachable offline, but nothing rules it
  out. Fixed with the exact shape codex proposed:
  `if len(message.tool_calls) != 1 or len(matching_calls) != 1 or message.invalid_tool_calls`.
- **N1.** `test_the_tool_payload_size_matches_what_pricingpy_assumes` pins only `propose()`'s
  payload (`_stop_tool_definition()` plus the five `_domain_tool_definitions()`) —
  `_final_assessment_tool_definition()` was never in that binding, so `respond()`'s own
  payload (priced by `reservation_usd` on every FINAL_ASSESSMENT turn) was completely
  unpinned. A second test, `test_the_respond_tool_payload_size_matches_what_pricingpy_
  assumes`, pinned the final schema at **2,261 characters/tokens** at that
  historical point. The current single-call proposal measurement is 12,011 for
  proposals and 2,292 for final assessments; `_send` names them separately.
- **N2 — proves the open #27 finding for real.** `KNOWN_PROSE_ONLY_CONTRACTS` has always
  mapped a label to a string DESCRIBING where its prose lives; nothing ever verified the
  prose actually EXISTS. Correctness proved the gap with a mutation: deleting
  `FinalAssessment.disposition`'s `Field(description=...)` in `domain.py` — the exact prose
  one registry entry names — left all 529 tests green. N1 is the mechanism that let this
  hide: the only test that could have noticed was measuring the wrong payload. A new
  `_WIRE_VISIBLE_PROSE_PROOF` mapping in `test_live_model.py` pairs each contract with the
  exact tool, the exact property path in its REAL emitted schema, and a literal substring of
  that property's CURRENT description; `test_the_registrys_pointed_at_descriptions_are_
  actually_present` checks it directly against the schema, not the free-text pointer.
  Mutation-verified against correctness's own exact demonstration (deleting `disposition`'s
  description now fails this test, naming the missing substring) and independently against a
  second field (`stop_reason`).
- **Readability fold-in.** `respond()`'s own comment made the same shape of overclaim as the
  `StoredIncident` one above — "Zero matches and two-or-more matches are refused the same
  way ... rather than the codebase silently picking a winner either time" read as exhaustive
  when it covered only the matching-name count. Narrowed to say plainly what C4's fix checks
  (matching-name count) versus what Finding 3 adds (total call count), each in its own
  paragraph.

**The two pre-trim summary counts.** `run_metric_check` (`prometheus.py`) and
`run_topology_check` (`telemetry.py`) both had the identical bug already fixed twice in the
base round for `run_logs_check`/`run_changes_check`: the summary string read a PRE-trim local
variable (`len(kept)`, `len(edge_list)`) instead of the payload's own post-trim count field.
`prometheus.py`'s case was a same-round regression — `count_key="sample_count"` was added in
the very diff that also left the summary string unfixed four lines below. Both now read
`payload['sample_count']`/`payload['edge_count']`. The topology case was reproducible with
real data (many long edge strings genuinely exceed the byte budget); the metric case is not
reachable through this suite's real data shape (`MetricSample.at`/`.value` are both floats,
always small) and is tested instead by monkeypatching `trim_to_bytes` to simulate what a
larger future row shape would trigger.

**The services-list 12KiB fix.** Correctness disputed codex's severity and fix shape: an
oversized `services` list in `run_topology_check` really does produce an over-budget payload
(34,863 bytes measured against the 12,288-byte cap) but is not reachable through any of the
four shipped lab topologies (all under 100 bytes total) — **P3, not P2** — and correctness
enumerated every `trim_to_bytes` caller and every non-row payload field in the codebase to
confirm `services` is the ONLY non-string, non-row field anywhere that needs its own bounding
pass. A general recursive "bound every list/dict field" mechanism was explicitly rejected as
solving a class of problem with exactly one member; the actual fix is a second `trim_to_bytes`
call treating `services` as its own row list with its own `service_count`. **A real
composition bug was found and fixed during this fix's OWN mutation testing**, not shipped:
seeding `services`/`service_count` into the payload while leaving `edges` to be added later
by its own `trim_to_bytes` call meant a payload that had already converged to fit (services
trimmed against a payload with no `"edges"` key yet) could go back over budget the instant
`"edges": []` was added afterward, with nothing left to pop and no string field for the
scalar fallback to reach (a list is invisible to it). Both `"services"` and `"edges"` are now
seeded into the payload BEFORE either `trim_to_bytes` call runs, so each call's own `fits()`
checks see the true combined size from their first iteration.

**Record only, this round:**

- **N3.** `cli.py`'s `run_decision_command` builds `investigations_dir = root / "results" /
  "investigations" / thread_id` from an unvalidated positional `thread_id` argument — the
  same class of gap C1 fixed for `incident_id`, on a third call site C1 never covered.
  Correctness tested `approve ../../decoy` and `approve ..` directly: both refuse cleanly
  with `THREAD_NOT_FOUND`, since reaching `investigations_dir` in any dangerous way requires
  a PRE-EXISTING `owner_decisions` row keyed by a real, internally-minted thread id — a
  traversal string cannot forge one. Not exploitable today; recorded, not fixed.
- **N4.** `schema_accepts`/item 5's cross-check (`test_live_model.py`) structurally cannot
  detect any rule spanning more than one tool schema or more than one call in a turn —
  the then-current `record_plan` plus check shape, multiple domain calls in one
  turn, and the forged-citation refusal (which needs a live evidence store, not any one
  schema) are all real prose-only contracts this mechanism can never express, let alone list
  in the registry. The module-level comment above `schema_accepts` now names this explicitly
  as a boundary on what the mechanism CAN see, not an incomplete audit of what it has looked
  at. Docstring-only; no code change.

### Round 4 and round 6 review — trim-order regression, aggregate rebuilds, registry closure, truncation signal

**Two more review rounds on top of `37fcfdc`, landed as `751387d` (round 4) and this round
(round 6).** Round 5 was a confirmation pass that found nothing new. This section documents
both landed rounds together since round 4 shipped without a `TECHNICAL_OVERVIEW.md` update of
its own.

**F1 — the services-list fix (previous round) was tested only for the case it was built for,
and shipped a regression.** `run_topology_check`'s comment used to claim "order between the two
[`trim_to_bytes`] calls does not matter for the final result" — true for WHETHER the combined
payload fits (both `"services"` and `"edges"` are seeded into the payload before either call
runs, so each call's own `fits()` check always sees the true combined size), false for WHICH
list absorbs the trimming. `trim_to_bytes` pops rows from its own list unconditionally until
`fits(payload)`, so whichever call runs first keeps popping against the OTHER list's still
full weight. Reproduced directly: a realistic incident shape (4 real service names, 400
oversized edges) under the previously-shipped services-first order reported `service_count: 0`
— every real service name silently wiped — while `edges`, the field actually responsible for
the overage, came away barely trimmed. **Fix: `edges` trims first**, because it is the field
this codebase's real data grows without bound (topology connections); `services` is a short,
bounded list that should almost never need trimming. Trimming `edges` first protects `services`
at `edges`'s expense — the right tradeoff for that shape, though it does not eliminate the
underlying asymmetry, only points it at the field where losing rows is safe to read.
`test_a_small_services_list_survives_an_oversized_edges_list` pins the asymmetric case; round 6
additionally rewrote the ORIGINAL services-fix test
(`test_an_oversized_services_list_still_fits_the_byte_bound`), which had paired an oversized
`services` list with a degenerate EMPTY `edges` list — exactly the shape of gap that let the
regression itself ship undetected — to use a small but non-empty `edges` list instead, and
pins what the shipped edges-first order actually does to it (`edge_count` goes to zero,
sacrificed so `services` survives trimmed rather than wiped).

**F2 — the registry-closure test only proved one direction.** `test_wire_visible_prose_proof_
only_names_registered_contracts` used to assert `set(_WIRE_VISIBLE_PROSE_PROOF) <=
set(KNOWN_PROSE_ONLY_CONTRACTS)` — every PROOF entry is registered, but not the reverse. A
fifth prose-only contract added to `KNOWN_PROSE_ONLY_CONTRACTS` without a matching proof entry
would pass silently, with no wire-proof requirement on it at all — the exact shape of gap N2
(previous round) closed for the first four. Tightened to an exact-set check across
`_WIRE_VISIBLE_PROSE_PROOF` plus a new, explicitly reasoned exemption registry,
`_PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF`, for the rare contract whose prose is a Python error
string rather than schema text (empty today). Round 6 changed that exemption registry's type
from a bare `frozenset[str]` to `dict[str, str]` (label -> reason), matching its two siblings —
a frozenset had no room to carry the "stated reason" the surrounding comment already demanded
of every entry.

**F3 — the pre-trim-aggregate bug, two more instances closed, a third found this round.**
`event_codes` (`run_logs_check`) and `max_value` (`run_metric_check`) were built from local
loop state entirely before `trim_to_bytes` ran, the same shape already fixed once for
`row_count`/`change_count`/`edge_count` (Unit 3b-4 addendum, C3) — a row popped by BYTE
trimming (not just a count cap) could still be reported in the aggregate even though it no
longer appears in the trimmed row list. Both rebuilt from the POST-trim row list. `event_codes`
is reachable with real lab-shaped data; `max_value` is not reachable with this suite's current
data shape (`MetricSample.at`/`.value` are always small floats) but was fixed anyway per the
owner's ruling to close the class, not just the reachable instance.

Round 6 found a third instance: `summaries` (`run_changes_check`) has the identical shape,
joined from every matched change before `trim_to_bytes` ran and never rebuilt — worst case,
`change_count: 0` with `summaries` still listing changes for an empty list. Rebuilt from the
post-trim `changes` list. Unlike the first two instances, this rebuild is not simply safe by
inspection without checking: `trim_to_bytes` pops from the END of the row list, so the rebuilt
string is always a PREFIX of the original (if `changes` still holds rows, `summaries` was never
touched by trimming, so the payload already fit with the full string present; if `changes` was
fully emptied, the scalar-shrinking fallback had already reduced `summaries`, and the rebuild
from an empty list can only be shorter still) — safe by construction, but that is exactly the
shape of claim F1's own comment made and shipped wrong, so an explicit `fits(payload)` assertion
checks it at runtime rather than resting on the argument alone.
`test_changes_summaries_only_name_changes_still_present_after_trimming` reproduces the ordinary
partial-trim case (30 changes reduce to `change_count == 7`) and confirms `summaries` only names
the survivors.

**The P1 — the model never saw the truncation signal for two of four check types.**
`prompts.py`'s `render_context` puts only `CheckOutcome.summary` in front of the model — the
full JSON payload, which correctly carries `payload["truncated"]`, never reaches it.
`run_logs_check`/`run_changes_check` already appended `" (truncated)"` to their summary strings
when cut; `run_metric_check` and `run_topology_check` did not, so a truncated metric window or
topology read as complete with no signal the true peak (or the full edge/service set) might lie
outside what survived. Both now append the same `" (truncated)"` note, reading
`payload["truncated"]` directly.

**Cleanup.** `run_metric_check`'s `max_value`-rebuild comment claimed rebuilding "can only
shrink or hold `max_value`... never grow the payload back over budget" — true of the NUMERIC
value, not of JSON byte length (`900.0` serializes to 5 bytes, `0.30000000000000004` to 19), so
the false general claim was dropped in favor of the comment's own already-stated unreachability
argument. `test_stored_incident_refuses_an_evidence_incident_id_mismatch`'s docstring, which the
sibling packet-mismatch test's docstring points readers to for the `_rebuild_store` double-fault
explanation, did not actually contain it — fixed by stating the explanation there directly
(see "Second dual review on `a44bf57`" above for the full trace).

### Round 7 and round 8 review — non-finite values, topology comment overclaim, JSON token rejection

**Round 7 landed as `7008534` with no `TECHNICAL_OVERVIEW.md` update of its own — the same gap
round 4 shipped with, backfilled here together with round 8's own fixes on top of it.** Round 8
was a whole-branch review (20 mutations against load-bearing code from every prior round, 19
caught, no P0/P1 anywhere) that found two real P2s, both in round 7's own newest code, plus four
smaller items.

**Round 7 — `read_sample` rejects non-finite metric readings.** `float()` parses `"NaN"`,
`"Infinity"`, and an overflow literal like `"1e400"` without raising, and `histogram_quantile`
(`GATEWAY_LATENCY_P95`) is documented to return NaN over an all-zero-rate bucket — a realistic
quiet-minute case, not an exotic one. A NaN sample made `max()` in `run_metric_check`
order-dependent (a genuine peak could be silently replaced depending only on where the NaN
sample sat in the fetched list), and both non-finite kinds serialize outside the JSON spec.
`read_sample` now checks `math.isfinite` on both the timestamp and the value after parsing and
returns `None` — the same "unreadable field, skip this row" contract it already applies to
unparsable rows — instead of threading a Python `nan`/`inf` into a typed `MetricSample`.

**Round 7 — `run_topology_check` gained an `assert fits(payload)` after both `trim_to_bytes`
calls.** Unlike every other `trim_to_bytes` caller, this payload has no string-valued field
(`services`/`edges` are lists; `service_count`/`edge_count`/`truncated` are int/bool), so
`trim_to_bytes`'s own scalar-shrinking fallback is a true no-op here. A reviewer measured that
the fallback's `widest_key is None` escape IS reached in normal operation (a realistic
small-services/large-edges shape hits it), so this assert is real defense-in-depth, not
decoration — see round 8's finding below for the more precise claim about whether it is
*currently reachable*.

**Round 7 — the wire-proof exemption registry's values were unchecked.**
`_PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF` became a `dict[str, str]` in round 6 so an exemption
could not be added without a stated reason, matching its two sibling registries — but the
existing test only ever read `set(_PROSE_ONLY_CONTRACTS_WITHOUT_WIRE_PROOF)`, the dict's KEYS,
never the values it exists to force. `test_every_wire_proof_exemption_carries_a_real_reason` now
asserts every value is a non-empty stated reason. The registry is empty today, so this closes the
enforcement gap for whenever an entry is first added, not a live gap today.

**Round 8, P2 — `read_sample` didn't catch `OverflowError`.** `json.loads` produces a genuine
Python `int` for a large integer literal in a JSON response (no decimal point or exponent, so
`float()`'s string-parsing path — which rounds an oversized literal to `inf` rather than raising
— is never involved). Converting a Python int that large to `float` raises `OverflowError`, which
is *not* a subclass of `ValueError` — round 7's `except ValueError` alone missed it, so a single
oversized integer sample escaped `read_sample`'s own "unreadable field, skip this row" contract
entirely, propagated through `graph.py`'s blanket exception handler, and turned one bad sample
into `FAILED_SAFE` for the whole investigation. Fixed by widening the except clause to
`(ValueError, OverflowError)`. `test_read_sample_rejects_non_finite_readings` extended with an
oversized-integer case on both the timestamp and the value position; mutation-verified (reverting
to `except ValueError` alone reproduces the exact `OverflowError` traceback the finding describes).

**Round 8, P2 — an all-NaN metric window read as confirmed zero, not as "nothing measured."**
Once round 7's fix correctly drops non-finite samples, an all-NaN window (the same documented
`histogram_quantile` quiet-minute case round 7's own fix cites) produces `sample_count: 0,
max_value: 0.0` — bit-for-bit identical to a genuinely empty, valid Prometheus response. The
model had no way to distinguish "measured zero" from "nothing measured," reopening the same
problem class round 6's own P1 fix addressed (the summary not reflecting a data reduction) in a
new shape. Fixed with a new `ParsedSamples` type (`prometheus.py`) carrying `raw_count` — how
many rows Prometheus actually sent — alongside the surviving samples; `run_metric_check` computes
`readings_discarded = raw_count - len(samples)` and appends a distinct `" (N unreadable,
discarded)"` note to the summary, deliberately separate from `" (truncated)"` (truncation means
"more data than fit the budget"; this means "some of what was sent could not be read at all" —
conflating the two would tell the model the wrong story). Two new tests: an all-NaN window
(`readings_discarded == 3`, summary names it, distinct from a genuinely empty response) and the
partially-NaN case folded into the existing end-to-end NaN test (`readings_discarded == 1`).
Mutation-verified (blanking the discard note makes both tests fail for the right reason).

**Round 8, P3 — the topology assert's own comment overclaimed which of two siblings was "load-
bearing."** `run_changes_check`'s assert comment called `run_topology_check`'s sibling "the
genuinely load-bearing one." A reviewer measured directly (120 randomized trials, 5 adversarial
shapes, and a mutation test disabling the topology assert entirely — all tests still passed) that
it is currently unreachable too, for the same reason as its sibling: both lists can always be
popped to empty, and the remaining fixed structure (85 bytes) is far under the 12,288-byte cap.
Both comments reworded to state both asserts are currently unreachable defense-in-depth — the
topology one is still worth keeping because it is the one function with no string field left
standing if the byte math or the lab data shape ever changed, which is a real reason to keep it,
not evidence it is load-bearing today.

**Round 8, P3 — `schema_accepts`'s coverage check tracked keywords but not `type` values.**
`test_schema_accepts_implements_every_keyword_the_real_schemas_use` (`test_live_model.py`)
collects every JSON Schema *keyword* the real emitted schemas use and asserts each is handled or
allowlisted — but never collected the *values* of the `type` keyword itself. Today's schemas only
use `["array", "integer", "null", "object", "string"]`, all handled, so there was no live gap —
but a future field emitting `"type": "number"` would fall through `schema_accepts`'s `kind ==`
branches to a default `return True, ""`, silently accepting any value, while the coverage test
would still pass (`"type"` the keyword is recognized regardless of which value it holds).
`_collect_schema_keywords` now returns `(keywords, type_values)` from the same traversal; the
coverage test asserts both. A new direct test,
`test_the_type_coverage_check_catches_an_unhandled_type_value`, proves the mechanism against a
synthetic `{"type": "number"}` schema. Mutation-verified (dropping `"integer"` from
`_SCHEMA_ACCEPTS_HANDLED_TYPES` fails the coverage test, naming the real schemas' actual `type`
value).

**Round 8, P2 (found by a second, independent static reviewer — codex — reproduced directly
before folding in) — `read_json_file`/`read_json_line` accepted the same non-standard JSON
tokens `read_sample` was hardened against.** `json.loads` accepts `NaN`/`Infinity`/`-Infinity` by
default — an extension beyond RFC-8259 most other JSON readers reject — and neither of
`telemetry.py`'s two general-purpose parsers (the entry point for every log line, the changes
manifest, and the topology manifest) passed a `parse_constant` callback to refuse them. A
poisoned token would parse into an ordinary-looking Python `nan`/`inf` float instead of the
record/manifest being refused the same way any other malformed input already is. Fixed with
`_reject_non_finite_json_token`, a `parse_constant` callback raising `ValueError`, passed to both
`json.loads` calls; both except clauses widened to `ValueError` (which `json.JSONDecodeError`
already subclasses, so this is a simplification, not an addition). For a log line, one poisoned
row is skipped like any other malformed line (`run_logs_check`'s existing `continue`); for a
changes/topology manifest, the token can sit anywhere in the file, so the whole manifest becomes
unreadable (`ToolOutcome.UNAVAILABLE`) rather than one field being silently poisoned — a stronger
consequence than the line-level case, and tested as such. Five new tests cover both parsers
directly and all three call sites end-to-end (`run_logs_check`, `run_changes_check`,
`run_topology_check`); each mutation-verified by reverting to the bare `json.loads` call and
confirming the corresponding new test fails for the right reason. `incident.json`/`report.json`
are loaded through a separate path (`cli.py`'s `_load_stored_artifact`), not through either
function here, and were out of this round's scope.

**Round 10 — complete telemetry JSON and decoding hardening.** The round 8
`parse_constant` guard covered only literal `NaN`/`Infinity` tokens. A syntactically valid JSON
number such as raw `1e400` instead reaches Python's default `parse_float`, silently becoming
`inf`; `_parse_finite_float` now rejects it in both telemetry readers. This is exercised directly
and through logs, changes, and topology paths. A malformed UTF-8 changes/topology manifest is
intentionally unavailable (matching `cli.py`'s stored-artifact boundary); a malformed UTF-8 JSONL
line is intentionally skipped individually so valid log rows still produce evidence. Finally,
`settle_reservation` now reports `RESERVATION_NOT_SETTLEABLE`, not `STORE_UNAVAILABLE`, when its
caller supplies no matching `RESERVED` ledger row; actual SQLite failures retain the latter code.

**Post-review single-call schema update.** Current proposal tool schemas are 12,011 serialized
characters/tokens, larger than the prose-only 9,600-token cap; the final-assessment
schema is 2,292. The gap between the two is mechanical, not five independently large tools:
Anthropic tool schemas are self-contained, with no cross-tool `$ref`, so each of the five
domain tools embeds its own full copy of the hypotheses schema (`live_model.py`'s
`_domain_tool_definitions()`) rather than sharing one — that five-fold duplication of one
shared schema is essentially the whole size increase. The cap intentionally excludes tool
schemas while reservations include them.
Strict schemas reject unknown tool, stop, hypothesis, and final-assessment fields.
Native Anthropic tool-use responses may include `tool_use`, `thinking`, and `redacted_thinking`
blocks, but visible text and unsupported blocks remain refused.

This landed after two real problems were caught and fixed during review, before the commit
above. First, the earliest version of `live_model.py`'s visible-content guard
(`_has_visible_content`) rejected every list-typed response content outright rather than
allow-listing specific block types — which would have refused a genuine Anthropic turn carrying
only `tool_use`/`thinking` blocks, the ordinary shape once extended thinking is on
(`_build_chat_anthropic` sets `thinking={"type": "adaptive"}` unconditionally), burning the run's
one run-wide repair slot on a wholly valid turn. Fixed by allow-listing `tool_use`/`thinking`/
`redacted_thinking` explicitly instead of rejecting every list; `test_propose_accepts_provider_
tool_and_thinking_blocks` and `test_respond_accepts_provider_tool_and_redacted_thinking_blocks`
pin the corrected behaviour, and `test_propose_refuses_a_text_block_alongside_tool_use`/
`test_respond_refuses_an_unsupported_block_alongside_tool_use` confirm a genuine text or
unsupported block is still refused alongside a real tool call. Second, an early pass applied
`extra="forbid"` to only some of the eight schema classes it now covers (`tools.py`'s five
argument classes, `domain.py`'s `Hypothesis`/`FinalAssessment`, `live_model.py`'s
then-current `PlanRecord`) —
the same "fix scoped to the instance touched, not every instance of the class" shape this
document has recorded before (see "Round 4 and round 6 review" above). A model call reaching one
of the missed classes could still smuggle an unrecognized field in and have it silently dropped
under pydantic's default `extra="ignore"`, instead of surfacing as a named repair. Closed by
applying `extra="forbid"` uniformly across all eight, each now carrying (or cross-referencing) a
comment stating why: a dropped field is invisible to the model and to this application, while a
refused one is a repair attempt the model can see and correct.

**Round 8, P3 — a docstring cited a sibling test by a name that never existed.**
`test_an_ambiguous_reservation_refusal_at_final_assessment_reports_its_reason` (`test_graph.py`)
called itself the sibling of
`test_an_ambiguous_reservation_refusal_reports_its_own_reason_not_internal_error` — a name that
does not exist; the real sibling, a few hundred lines above, is
`test_an_ambiguous_reservation_refusal_reports_its_own_reason`, with no `_not_internal_error`
suffix (that suffix is real on the analogous cost-ceiling pair, which this docstring appears to
have been copied from). Citation corrected.

### Unit 3c — the paired live comparison

The last piece of Milestone 3: wiring `evaluation.py`'s scoring machinery (`score_run`, built in
an earlier milestone with zero callers until this unit) into a real, cost-bounded run of
`TECHNICAL_SPEC.md` §10's paired live comparison, plus the reproducibility manifest fields §10
requires on every scored record.

**A scored-run mode, not a new orchestrator.** `build_graph`/`run_graph_investigation` gained two
independent, keyword-only, default-`False` flags rather than a second graph implementation.
`suppress_escalation=True` changes exactly one thing: `_make_route_after_final_assessment`'s
router returns `"final_report"` unconditionally, without ever calling `_escalation_reason` at all
-- not "compute the reason but ignore it," which would leave a way for a future edit to
accidentally wire the result back in. A confinement test (`test_a_scored_run_suppresses_
escalation_while_an_ordinary_run_still_escalates`, `test_graph.py`) runs the same
`service_out_of_scope.json` fixture twice, unmodified except for the flag: the ordinary run
still pauses with `INSUFFICIENT_EVIDENCE_WITH_CHECK_REMAINING` exactly as before this unit, while
the scored run reaches a terminal report with `escalation is None` -- proving the suppression is
scoped to the flag, not a change to escalation behaviour generally.

**The no-tool baseline is a smaller graph, not a starved one.** `no_tool_baseline=True` never adds
the `investigate`/`dispatch_tool`/`normalize_evidence` nodes at all -- `START` edges directly to
`final_assessment` -- rather than binding the same five domain tools and hoping a zero
`executed_tools` budget keeps the model from trying them. The distinction matters: a model that
can still see tool schemas but has its proposals denied for budget exhaustion is not "no tools,"
it is "tools that always fail," a different and noisier comparison. `_make_final_assessment`
already tolerated empty receipts/evidence/passages at `model_turn=0` on every pre-existing call
path, so no node factory needed to change, only which edges `build_graph` wires. A topology test
(`test_a_no_tool_baseline_never_offers_a_domain_tool`) proves
this directly by asserting `ReplayToolCallingModel.requests` contains exactly one
`Stage.FINAL_ASSESSMENT` request and nothing else, reusing `valid_diagnosis.json`'s
`initial_plan`/`hypothesis_update` entries unmodified specifically so a regression that started
calling `investigate` again would consume them and pass silently if this test only checked the
final report's shape instead.

**The frozen four-pair corpus needed no new lab or scenario-controller work.** Every
`lab/scenarios/*.json` family already carried a `seed_variants.evaluation` block distinct from
`seed_variants.development`, and `start_scenario(root, family, seed="evaluation")` was already a
real, tested path. `causalops-evaluate` drives exactly the four existing families
(`ambiguous_telemetry`, `configuration_change`, `downstream_timeout_retry_amplification`,
`resource_pool_saturation`) through that seed, one incident per family, investigated twice each
(no-tool baseline, then tool-enabled) rather than two separate scenario starts -- `scenario_
control.py`'s "one active scenario at a time" rule makes two concurrent incidents structurally
impossible anyway, and §10's own "one no-tool baseline and one tool-enabled run *per incident*"
wording already says the pairing is same-incident, not same-family-different-incident.
Held-out enforcement reuses the existing evaluator/investigator boundary unchanged: `start_
scenario` already writes `runs/<incident_id>/evaluator/expected.json` for every scenario, and
`telemetry.RunPaths` still has no accessor for that directory. `causalops-evaluate` reads that
file directly by constructing the path itself, staying on the evaluator side of the same line
`tests/security/test_ground_truth_isolation.py` already polices, rather than widening
`RunPaths`.

**`causalops-evaluate` is a genuinely separate binary.** Registered in `pyproject.toml`'s
`[project.scripts]` as `causalops-evaluate` (a `[project.scripts]` key becomes an executable
filename on `PATH`, which cannot contain the literal space `causalops evaluate`'s prose
implied). `causalops.cli` never imports `causalops.evaluate_cli`, and the reverse also holds
(unstated by `CLAUDE.md` but just as load-bearing, since either direction would form the same
coupling). Both scripts share their live-model/tool-registry construction and cost-ceiling
parsing through a new neutral module, `causalops.live_setup` -- `cli.py`'s former `_build_model_
and_registry`/`_live_evaluation_ceiling_usd` extracted verbatim, renamed public, and imported by
both, so isolation did not force the alternative of copy-pasting that wiring into
`evaluate_cli.py`. `tests/security/test_evaluate_cli_isolation.py` proves the isolation two ways:
`import_scan.imported_modules`'s full-AST walk (`ast.walk`, not just top-level statements) so a
lazy import buried in a function body or a conditional is still caught, plus a plain substring
check on `cli.py`'s own source text that also closes a dynamic `importlib.import_module(...)`
loophole the AST scan alone cannot see -- the two-tier approach this project's own history (see
`CLAUDE.md`'s uv-run-pytest note) argues is necessary whenever "never imported" needs to survive
someone's later attempt to satisfy the letter of that rule while defeating its purpose.

**`EvaluationRecord` gained the reproducibility manifest §10 asks for.** Beyond the previously
missing `retrieval_mode`: `git_sha`/`git_dirty` (captured via `git rev-parse HEAD`/`git status
--porcelain` at evaluate-run time, not stored anywhere durable before this unit), the run's
`Versions` (prompt/policy/tool-registry, already computed per-run but never surfaced past
`InvestigationReport.versions`), `runbook_corpus_version` (from `RunbookIndex.corpus_version`,
now actually read and stringified instead of deliberately discarded -- the comment at its own
`__init__` had named this unit as the first real consumer since Unit 3a), `fixture_sha256` (a
SHA-256 of the exact `lab/scenarios/<family>.json` bytes an incident's family was started from --
a content hash chosen over a hand-maintained version string specifically so it cannot drift
silently and needs no edit to the frozen scenario files), `model_name`, `pricing_source`/
`pricing_verified_on` (from `CLAUDE_SONNET_5_PRICING`, already existed, now carried onto the
record), `configured_ceiling_usd`, and `reserved_usd`/`actual_usd`. "Raw artifact references"
needed no new field: `investigation_id` alone already names the deterministic `results/
investigations/<investigation_id>/` directory `run_records.finalize_investigation` writes every
raw artifact to. `EvaluationRecord` also gained `extra="forbid"`, matching the project-wide
tightening the immediately preceding unit applied to every other wire-facing model.

**Cost is read from the existing ledger, not tracked twice.** `cost_ledger.run_cost_totals(conn,
run_id)` sums `reserved_usd`/`actual_usd` scoped to one `run_id`, a different question from
`_reserved_and_settled_total`'s application-wide ceiling sum beside it -- both read the same
table through the same connection every other live call already uses. One real gap surfaced
during implementation: `run_id` is internal bookkeeping, deliberately absent from
`InvestigationReport` (a caller has no legitimate reason to see it -- it is never cited, never
displayed). `causalops-evaluate` still needs it to query the ledger, so `run_graph_investigation`
now records it as an extra field on the `investigation_started` event it already emits first --
`events.jsonl` is not schema-frozen the way `InvestigationReport` is, and no test pinned that
event's exact field set, making this a smaller, more honest fix than adding a field to the report
schema that nothing else in this unit needs. `_run_one` (`evaluate_cli.py`) passes no
`checkpointer` to `run_graph_investigation`, so each scored run gets a fresh, process-local
`InMemorySaver()`: a suppressed run never pauses, so there is nothing to resume across a process
boundary, and this keeps scored-run graph checkpoints out of the shared `checkpoints.db`
entirely -- the cost ledger is a separate connection to that same file, unrelated to the graph
checkpointer.

**Verified before freezing**: 579 pre-existing tests plus this unit's additions all pass; `ruff
check`/`ruff format --check` and `mypy --strict` are clean on every touched file; the confinement
test and both isolation tests were mutation-tested by reverting each guard in turn and confirming
the corresponding test fails for the right reason, then restored. `causalops-evaluate` itself was
never run against a live model or the Docker lab during this unit's own development --
`test_evaluate_cli.py`'s orchestration test monkeypatches `build_model_and_registry`/
`start_scenario`/`reset_scenario`, the same seam-testing approach `test_live_model.py` already
uses for `LiveClaudeModel` itself, so the whole fast suite stays network-free.

## Superseded v1 evaluation design

The original v1 plan (formerly this document's §11 "Evaluation and scoring"
and §13 "Definition of done") specified a Claude benchmark that was never
built: three repetitions of four families against two systems — 24 scored
runs — under a USD 1.75 evaluation cost cap, plus a USD 0.15 standalone
investigation cap.

That design is superseded by `TECHNICAL_SPEC.md` §10: **at most six held-out
paired incidents** (one no-tool baseline and one tool-enabled run each) under
a single **USD 5.00 application-wide ceiling** (raised from USD 2.00 by
Unit 3b-3 -- see "The smoke call's findings" above), covering both
standalone and paired runs together rather than separate caps. The
escalation path is explicitly excluded from scored runs; HITL is
demonstrated and tested separately.

Kept here rather than deleted because the scope change — 24 runs to 6, two
caps to one — is something a reader who remembers the original number will
reasonably ask about. `TECHNICAL_SPEC.md` §10 is authoritative; this section
is history.

The rest of the original evaluation design — mechanical scores (diagnosis
correctness, disposition correctness, citation validity, citation
sufficiency, control behavior, efficiency), the reproducibility manifest
fields, and the "do not use an LLM judge, do not report p95 from a small
sample" publication rules — is now implemented, by Unit 3c ("Unit 3c — the
paired live comparison," above). Mechanical scoring (`score_run`) predates
this unit and is otherwise unchanged; Unit 3c is what actually calls it.

## Remaining deferred extensions

Two items in the original v1 deferred list are no longer deferred:
**agent frameworks** and **SQLite-backed investigation restart** are now
adopted v2 requirements (LangGraph orchestration, SQLite-backed checkpoints)
per `CLAUDE.md`. What remains genuinely deferred, considered only after v2's
completion criteria and an explicit specification update:

- MCP adapter around the typed tool registry.
- OpenTelemetry traces as another evidence source.
- Additional incident families and a larger benchmark.
- Optional second-provider comparison for a specific measured question.
- Static hosted report or recorded-results viewer.
