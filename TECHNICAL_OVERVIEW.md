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

Nothing under this heading has landed. No commit, source file, or test
implements any of it. Specifically absent from the repository today:

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
  workflow.py
  prompts.py
  models.py
  tools.py
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

Milestone 1 adds `graph.py`, `tool_calls.py`, and `tool_wrappers.py` to
`src/causalops/`; none of the three exists yet. Create new directories only
when an implemented vertical slice needs them.

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

This is the current, implemented loop in `src/causalops/workflow.py`.
Milestone 1 adds a parallel LangGraph orchestrator beside it (a new
`GraphPhase` enum, tracked in Part III), retired only after conformance
parity is demonstrated — this loop is not replaced yet.

```text
CREATED
  -> PLAN_FIRST_CHECK
  -> VALIDATE_FIRST_CHECK
  -> EXECUTE_FIRST_CHECK
  -> UPDATE_AND_PLAN_SECOND
  -> VALIDATE_SECOND_CHECK
  -> EXECUTE_SECOND_CHECK
  -> FINAL_ASSESSMENT
  -> DIAGNOSED | INSUFFICIENT_EVIDENCE | FAILED_SAFE
```

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
| Model call | 90 seconds | specified, not enforced — `Budgets` has no per-call timeout field |
| Tool execution | 10 seconds | built — `Budgets.tool_timeout_seconds` |
| Model calls, including repair | 4 | built — `Budgets.model_calls` |
| Executed diagnostic tools | 2 | built — `Budgets.executed_tools` |
| Structured-output repairs | 1 | built — `Budgets.repairs` |
| Maximum counted input per model call | 3,200 tokens | specified, not enforced — no token counting exists in `src/` |
| Claude `max_tokens` | 1,600 tokens | specified for the live adapter, not yet built |
| Claude adaptive-thinking effort | `medium` | specified for the live adapter, not yet built |
| Log result | 40 rows and 12 KB | built — `Budgets.log_rows`, `evidence.MAX_RESULT_BYTES` |
| Metric result | 60 samples and 12 KB | built — `prometheus.MAX_METRIC_SAMPLES`, `evidence.MAX_RESULT_BYTES` |
| Automatic retries | 0 | built — no retry logic exists anywhere in `src/` |

Every row marked `built` is enforced today by the cited constant. The two
rows marked `specified, not enforced` (model-call timeout, input token
counting) and the two Claude-specific rows are live-adapter design, not yet
implemented — see Part I, Phase 3, and Part III. Cost caps are recorded in
`TECHNICAL_SPEC.md` §10 (superseding the v1 figures — see Part III) rather
than here, since they apply only once a live adapter exists.

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

v1 implements exactly these four read-only tools. `TECHNICAL_SPEC.md` §7
adds a fifth, optional `search_runbooks` retrieval tool for Milestone 2; it
is not implemented yet.

| Tool | Typed input and backend |
|---|---|
| `query_metric` | Registered PromQL template ID, service, and bounded window; executed against Prometheus |
| `query_logs` | Registered filter ID, service, bounded window, and row limit; scans active-run JSONL |
| `list_recent_changes` | Service and bounded window; reads the active-run change manifest |
| `get_topology` | Active incident ID; reads the active-run topology manifest |

The model selects registered template IDs and typed parameters. Application
code constructs PromQL and log predicates. The model cannot submit raw shell,
SQL, PromQL, URLs, paths, code, or infrastructure manifests.

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
USD 2.00 cap) — see Part III, "Superseded v1 evaluation design," for what
changed and why the numbers differ.

Business outcomes `DIAGNOSED` and `INSUFFICIENT_EVIDENCE` are successful CLI
executions. `FAILED_SAFE`, invalid configuration, and unavailable dependencies
return nonzero with stable machine-readable reason codes.

## Threat model and tests

Protected assets are incident scope, tool registry, policy, budgets, evidence
integrity, evaluator ground truth, secrets, and the host environment.
Attacker-controlled inputs include model output, logs, metric labels, alert
text, and change descriptions.

Rows below marked **built** are proven today by a cited Phase 1/2 test. Rows
marked **Phase 3** describe a control that only applies once the live Claude
adapter exists and cannot be tested until it does.

| Threat | Required control | Status |
|---|---|---|
| Ground-truth leakage | Opaque model inputs; assert prompt/context lacks semantic scenario keys and expected values | built — `tests/security/test_ground_truth_isolation.py` |
| Prompt injection in telemetry | Untrusted delimiters plus deterministic policy; verify no scope, tool, policy, or budget expansion | built — `tests/security/test_prompt_injection.py` |
| Arbitrary query execution | Template enums only; reject raw PromQL, shell, SQL, URL, and path input | built — `tests/unit/test_policy.py`, `test_tools.py` |
| Scope escape | Incident-labelled backends and allowlists; deny cross-run service, time, file, and evidence access | built — `tests/unit/test_telemetry.py`, `test_policy.py` |
| Forged citations | Resolve opaque evidence IDs from active store; reject missing and cross-incident IDs | built — `tests/unit/test_policy.py`, `src/causalops/replay_fixtures/forged_citation.json` |
| Resource exhaustion | Enforce call, time, row, sample, and byte limits, and per-kind context quotas (`evidence.CONTEXT_QUOTAS`) — distinct from the unbuilt token-counted input cap, see "Default limits" above | built — `tests/unit/test_workflow.py`, `test_telemetry.py` |
| Scenario contamination | Reset volumes/state and assert health and empty run scope before the next scenario | built — `tests/integration/test_scenario_reset_isolation.py` |
| Model/tool failure | Timeout and malformed-output fixtures produce deterministic terminal states | built — `tests/unit/test_workflow.py` |
| Credential leakage | Environment-only API key plus redaction; verify it never reaches CLI text, config, artifacts, logs, reports, receipts, or errors | Phase 3 — no code path reads or handles an API key yet |
| Provider data leakage | Send only bounded synthetic incident context; verify requests exclude secrets, evaluator ground truth, and host paths | Phase 3 — no provider request exists yet |
| Unbounded provider spend | Durably flushed write-ahead reservations plus settled/outstanding accounting; verify both caps, crash behavior, and no request after denial | Phase 3 — no cost ledger exists yet; caps superseded, see Part III |

### Tests already proving Phase 1/2 behavior

- Domain invariants and valid/invalid transitions (`test_domain.py`).
- Every valid and invalid disposition/root-cause pairing, including proof that
  only application code can create `FAILED_SAFE` (`test_domain.py`,
  `test_workflow.py`).
- One structured-output repair and repair exhaustion (`test_workflow.py`,
  `test_replay_model.py`).
- Denied-proposal accounting and duplicate fingerprints (`test_policy.py`,
  `replay_fixtures/duplicate_proposal.json`).
- Deterministic clock, evidence ordering, quotas, and truncation markers
  (`test_evidence.py`, `test_workflow.py`).
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
the threat table's *Model/tool failure* row (`test_workflow.py`) and
*Resource exhaustion* row (`test_workflow.py`, `test_telemetry.py`); the
finalized-result immutability proof is covered by the *Scenario
contamination* row (`test_scenario_reset_isolation.py`) — same tests, same
citations, listed once.

Windows support above is proven by continuous integration on
`windows-latest`. A manual smoke test on the working platform, Linux x86-64,
with all required containers, is run by hand and produces no committed
artifact — it is a process step, not a test this repository can cite.

### Tests specified for the live Claude adapter — not yet built

These describe required behavior once Phase 3 (or a later v2 milestone) adds
a live model adapter. None of them exist today because none of the code they
would test exists today:

- Authenticated `GET /v1/models/claude-sonnet-5` metadata check and exact
  required-model mismatch handling.
- Exact Claude request shape: required `claude-sonnet-5`, adaptive thinking,
  `medium` effort, `max_tokens=1600`, no `temperature`/`top_p`/`top_k`, and one
  concurrent request.
- Synchronous SDK construction with `max_retries=0`; per-operation timeout
  behavior; exactly one inspected HTTP attempt per logical operation.
- Token counting over system text, messages, schema, and evidence;
  deterministic trimming and recounting above the input cap.
- Cost-cap gates, the reservation formula, unique logical request IDs, and
  durable reservation/usage/settlement ordering. (The v1 figures — USD 0.15
  standalone, USD 1.75 for 24 runs — are superseded; see Part III.)
- Crash, timeout, missing-usage, and ambiguous-response fixtures retaining
  the full reservation, and resume never repeating an outstanding request.
- `end_turn` content validation and repair, and the distinct no-repair
  `FAILED_SAFE` result for every other documented provider stop reason.
- A billed-refusal fixture proving usage settles before failure handling.
- Missing-credential, authentication, rate-limit, network, timeout, and
  cost-denial failures producing stable `FAILED_SAFE` records.
- API-key redaction and proof that provider thinking blocks are never
  retained.
- Sequential and resumed benchmark runs, including proof a completed run key
  is never executed or recorded twice.
- Clean/dirty commit provenance blocking publication of a non-reproducible
  score.

Normal CI runs on `windows-latest` and `ubuntu-latest` using replay fixtures
and disposable local test data. Network access is allowed only while
installing locked dependencies; after that, formatting, linting, strict
typing, unit tests, security tests, and replay conformance make no external
calls and require no credentials. Outside CI, no command in this repository
today sends an authenticated request to Anthropic — that capability does not
exist yet.

# Part III — v2 in progress

The old v1 delivery process (§14 "Three-phase delivery sequence" in
`git show b6f4d9c:TECHNICAL_OVERVIEW.md`) is superseded by `CLAUDE.md`'s
owner-controlled review protocol and by the milestone/unit vocabulary
defined at the top of this document.

## Milestone 1 — Bounded tool-graph parity

**Status:** in progress. Unit 0 (this document and the matching
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

Unit 1b (the orchestration: `graph.py`, the CLI `--orchestrator` flag, parity
tests, and wiring `tool_wrappers.py` into an actual dispatch node) has not
started.

Per `TECHNICAL_SPEC.md` §12: run one replay incident through a LangGraph
`StateGraph` with native tool-call parsing, one policy wrapper, atomic budget
reservation, and the existing report and scorer — then wrap the remaining
three tools and retire the duplicate orchestration only after conformance
parity is demonstrated.

**Known gaps carried into Milestone 2:**

- `evaluation.py`'s `count_control` (`evaluation.py:164-183`) reads only
  `policy_result` and `reason_code`, never `outcome` or the new `state`
  field, so a run ending with a `RESERVED` receipt is invisible to the
  scorer's `ControlCounts`. Scorer changes were out of scope for Unit 1a;
  this closes once Milestone 2 makes reservations durable across a
  checkpoint resume.
- `TECHNICAL_SPEC.md` §11 permits `langsmith` as an inert transitive
  dependency only if "tracing is force-disabled at the entry point **and** a
  test proves no tracing client is constructed and no tracing request is
  attempted." Unit 1a satisfies only the first half — `src/causalops/__init__.py`
  forces both tracing variables off, proven by
  `tests/unit/test_tracing_disabled.py`. The second half cannot be tested
  until something actually constructs a `langchain-core` client, which is
  Unit 1b's `graph.py`. Do not read Unit 1a as closing §11 fully.
- `ToolReceipt`'s lifecycle validator checks `state` against `outcome`/
  `result_digest`/`evidence_id` only. It does not (yet) reject a `RESERVED`
  receipt carrying a `reason_code` or a nonzero `duration_ms`, or a
  `policy_result` other than `ALLOWED`. Nothing constructs those combinations
  today and the docstring does not claim they are closed; tightening is
  deferred, not forgotten.

**Deliberate, not a gap:** `domain.py`'s `SCHEMA_VERSION` stayed `"1"` even
though `ToolReceipt`'s persisted shape changed (the new `state` field,
`outcome` becoming optional). The change is backward-compatible — every
existing constructor call still produces a valid, equivalently-interpreted
receipt — no consumer keys behavior on the version string, and `results/`/
`runs/` are empty in this repository, so there is no persisted artifact to
migrate. Revisit this the moment a reader (a replay fixture, an external
consumer, a migration script) actually depends on the version number
distinguishing the two shapes.

## Milestone 2 — Durable escalation and local retrieval

**Status:** not started. Adds checkpoint/operation IDs, CLI interrupt resume,
approval routing, and crash/idempotency tests; curated FTS5 runbooks,
retrieval provenance, and injection/no-ground-truth-leakage tests. Pinecone
remains a post-milestone optional experiment.

## Milestone 3 — Evidence-backed portfolio release

**Status:** not started. Runs the fixed paired evaluation under the USD 2 cap,
saves raw records and limitations, produces architecture and threat-model
documents, verifies the clean source commit, and records a short diagnosis
plus abstention/escalation demo.

## Superseded v1 evaluation design

The original v1 plan (formerly this document's §11 "Evaluation and scoring"
and §13 "Definition of done") specified a Claude benchmark that was never
built: three repetitions of four families against two systems — 24 scored
runs — under a USD 1.75 evaluation cost cap, plus a USD 0.15 standalone
investigation cap.

That design is superseded by `TECHNICAL_SPEC.md` §10: **at most six held-out
paired incidents** (one no-tool baseline and one tool-enabled run each) under
a single **USD 2.00 application-wide ceiling**, covering both standalone and
paired runs together rather than separate caps. The escalation path is
explicitly excluded from scored runs; HITL is demonstrated and tested
separately.

Kept here rather than deleted because the scope change — 24 runs to 6, two
caps to one — is something a reader who remembers the original number will
reasonably ask about. `TECHNICAL_SPEC.md` §10 is authoritative; this section
is history.

The rest of the original evaluation design — mechanical scores (diagnosis
correctness, disposition correctness, citation validity, citation
sufficiency, control behavior, efficiency), the reproducibility manifest
fields, and the "do not use an LLM judge, do not report p95 from a small
sample" publication rules — remains accurate design intent for whichever
milestone eventually builds the live adapter. It is not implemented; see
Part I, Phase 3.

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
