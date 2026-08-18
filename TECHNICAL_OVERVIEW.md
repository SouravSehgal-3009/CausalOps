# CausalOps: Technical Overview

> **Subtitle:** Evidence-Grounded Incident Investigator  
> **Status:** balanced three-week MVP; planned work, not implementation claims  
> **Effort:** 35–45 focused hours  
> **Audience:** project owner, contributors, reviewers, and coding assistants

## Document authority

This file is the sole source of truth for CausalOps product behavior,
architecture, scope, interfaces, evaluation, milestones, and completion
criteria. `AGENTS.md` is the sole source of truth for contributor and coding
agent behavior.

README files, ADRs, issues, comments, and generated reports may explain or
record implementation details but cannot redefine this specification. Any
approved product or technical decision must update this file in the same
change. Do not create a parallel project brief or technical specification.

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

## 1. Product thesis

CausalOps is an incident investigation assistant for a synthetic Python
microservice system that runs locally and uses one hosted reasoning model. It
keeps several possible causes, gathers a limited
amount of evidence that can support or rule out each one, and runs safe
read-only checks. It returns either a diagnosis with evidence the owner can
check or a clear statement that the evidence is not enough.

“Causal” means a disciplined loop of **possible cause → diagnostic check →
evidence update**. CausalOps does not claim formal causal inference, build
causal graphs, or estimate what would happen under an intervention.

It is decision support for an on-call engineer. It is not an autonomous
operator or a production SRE platform. Its central trust boundary is:

> The model may propose and interpret. Deterministic Python validates,
> authorizes, executes, stops, scores, and records.

## 2. MVP and non-goals

The MVP is a complete four-family CLI investigation and evaluation system. Its
portfolio value comes from trustworthy tool use and honest evaluation, not
from platform size.

Included:

- Three project-authored Python services running through Docker Compose.
- A separate Python scenario controller for fault activation and cleanup.
- Prometheus metric history and scenario-scoped structured JSONL logs.
- Four typed read-only investigator tools.
- Several possible causes, evidence for and against each one, up to two checks,
  and a clear abstention when the evidence is not enough.
- Replay conformance tests and a pinned Claude API comparison.
- Append-only run records, deterministic scoring, and Markdown reports.

Deferred:

- Remediation, write-capable tools, multi-agent roles, and agent frameworks.
- Web UI, public API, background worker, and investigator database.
- PostgreSQL, Kubernetes, cloud hosting, Terraform, and managed infrastructure.
- MCP, distributed tracing, durable recovery, and a second live provider.

## 3. Architecture and trust boundaries

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
scenario. PostgreSQL is not part of the MVP. A Prometheus container provides
historical metric queries. Claude is accessed through Anthropic's hosted API.
“Python-only” means all authored services, orchestration, tools, policy, and
evaluation code are Python; Docker and Prometheus are local infrastructure
dependencies.

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

Use these approximate container memory ceilings:

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
atomically finalizes its cited evidence, tool receipts, run record, and report
under `results/investigations/<investigation-id>`. During a benchmark, each
completed run finalizes the same artifact kinds under
`results/<evaluation-id>`. Finalized result bytes are immutable. A correction
creates a new investigation or evaluation ID rather than rewriting an existing
result.

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
modify finalized evidence, records, or reports under `results/`.

### Logical ground-truth isolation

Ground truth is open-source test metadata, not a filesystem secret. Isolation
means the investigator process receives no evaluator manifest, semantic
scenario key, expected outcome, required-evidence predicate, or answer-bearing
path. Tests must prove that investigator packages do not import evaluator-only
modules and that model contexts contain none of those values.

### Intended repository shape

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
lab/
  services/
  scenarios/
tests/
  unit/
  conformance/
  integration/
  security/
results/
```

Create directories only when an implemented vertical slice needs them. Create
`docs/adr/` only when the owner explicitly approves an ADR decision.

## 4. Incident identity and initial evidence

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

## 5. Investigation workflow and budgets

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

| Limit | Default |
|---|---:|
| Investigation wall clock | 360 seconds |
| Model call | 90 seconds |
| Tool execution | 10 seconds |
| Model calls, including repair | 4 |
| Executed diagnostic tools | 2 |
| Structured-output repairs | 1 |
| Maximum counted input per model call | 3,200 tokens |
| Claude `max_tokens` | 1,600 tokens |
| Claude adaptive-thinking effort | `medium` |
| Log result | 40 rows and 12 KB |
| Metric result | 60 samples and 12 KB |
| Automatic retries | 0 |
| Claude standalone investigation cost cap | USD 0.15 |
| Complete 24-run Claude evaluation cost cap | USD 1.75 |

All limits are application-owned and visible to the model as immutable status.
Context construction uses an injected clock, stable evidence ordering, fixed
per-kind quotas, explicit truncation markers, and a digest of the final input.
Before every Claude generation, the application asks Anthropic's token-count
endpoint to count the complete request, including system text, messages, tool
or output schema, and evidence. If the result exceeds 3,200 input tokens, the
application applies its deterministic evidence-trimming rules and counts again.
It does not send the generation request until the recounted input is within the
limit.

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

## 6. Public contracts

All model, tool, artifact, and CLI boundaries use Pydantic v2 models. Models that
are persisted or exchanged with the reasoning model also carry a schema version.

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
- `FinalAssessment`: a model-selected `DIAGNOSED` or
  `INSUFFICIENT_EVIDENCE` outcome, matching root-cause code, supporting and
  contrary evidence IDs, uncertainty, and proposed human next step. Its model
  schema excludes `FAILED_SAFE`.
- `InvestigationReport`: opaque investigation ID, validated assessment,
  budgets, latency, usage, versions, limitations, and artifact references.
  Application code may create a `FAILED_SAFE` report without a model
  `FinalAssessment`.
- `EvaluationRecord`: paired system/run identity, expected outcome, mechanical
  scores, reproducibility manifest, and raw artifact reference.

Do not request or persist private chain-of-thought. Model responses contain
only structured decisions, short summaries, and evidence references.

## 7. Investigator tools and policy

Implement exactly four read-only tools:

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

## 8. Incident families and variants

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
kind, registered template/filter, and structured condition. The scorer resolves
cited evidence IDs and evaluates these predicates; predicate names and expected
values never enter model context.

## 9. CLI contract

```powershell
causalops doctor
causalops lab up
causalops lab down
causalops scenario start <family> --seed <development|evaluation>
causalops scenario reset <incident-id>
causalops conformance
```

Use exactly these model-execution commands:

```powershell
causalops investigate <incident-id> --model replay
causalops investigate <incident-id> --model claude --max-cost-usd 0.15
causalops benchmark --model claude --variant evaluation --repetitions 3 --max-cost-usd 1.75
causalops benchmark --model claude --variant evaluation --repetitions 3 --max-cost-usd 1.75 --resume <evaluation-id>
```

- `doctor` checks the operating system, at least 7.5 GiB detected total RAM,
  current available RAM, at least 12 GB free disk, required writable
  directories, Docker, and the presence of `ANTHROPIC_API_KEY`. It warns but
  does not fail when available RAM is below 2.5 GiB.
- When local checks and the key-presence check pass, `doctor` uses the official
  Anthropic Python SDK to make an authenticated HTTPS model-metadata request
  equivalent to `GET /v1/models/claude-sonnet-5`. It verifies the exact required
  model is available, does not request generated output, and has no intended
  model-token charge. It does not count toward investigation model calls or
  cost caps. It cannot inspect or guarantee the owner's Console credit balance.
- `lab up` verifies Docker, Prometheus, service health, and required writable
  run directories.
- `scenario start` is owner-facing and may use a semantic family name. It prints
  an opaque incident ID; semantic identity is not passed to `investigate`.
- `investigate` accepts only an opaque incident ID, creates an opaque
  investigation ID, and finalizes JSONL evidence, receipts, a run record, and
  a Markdown report under `results/investigations/<investigation-id>`.
- `conformance` runs replay-backed workflow, policy, safety, and artifact tests.
- `benchmark` manages held-out scenario lifecycle and produces paired baseline
  and workflow records. `--resume` accepts an existing evaluation ID, skips
  completed run keys, and appends only missing records. Resume fails when its
  stored model, thinking effort, token limits, pricing values/source date, or
  cost cap differs from the requested run. An outstanding logical request is
  never repeated; resume marks its run `FAILED_SAFE` and continues only when
  the evaluation contract and remaining cost cap permit the next distinct run
  key.
- `scenario reset` verifies healthy state and cross-run isolation. It deletes
  only active lab/transient state for that incident and never finalized
  records, reports, receipts, or cited evidence under `results/`.

Business outcomes `DIAGNOSED` and `INSUFFICIENT_EVIDENCE` are successful CLI
executions. `FAILED_SAFE`, invalid configuration, and unavailable dependencies
return nonzero with stable machine-readable reason codes.

## 10. Model strategy

### ReplayReasoningModel

Checked-in opaque fixtures provide deterministic valid and invalid stage
responses. Replay is used for ordinary development, CI, conformance, policy,
state, and report tests. Replay results are scripted and must never be reported
as agent accuracy or diagnostic improvement.

### ClaudeReasoningModel

The only live reasoning provider uses the official synchronous Anthropic
Python SDK behind the same small reasoning-model protocol as replay. Its
contract is:

- Construct the synchronous SDK client with `max_retries=0`.
- Exact required model `claude-sonnet-5`; aliases and silent model substitution
  are not allowed.
- Adaptive thinking with effort `medium` and `max_tokens=1600`; the output
  limit covers thinking plus the returned structured content.
- Do not send `temperature`, `top_p`, or `top_k`.
- Use Anthropic token counting before every generation. Count system text,
  messages, the structured-output schema, and evidence. Deterministically trim
  and recount until counted input is at most 3,200 tokens.
- Supply the Pydantic JSON Schema through Claude's structured-output support
  and validate the returned structured result locally.
- Run at most one Claude request and one active scenario at a time.

Each logical token-count, model-metadata, or generation operation makes exactly
one HTTP attempt. A generation timeout is the smaller of 90 seconds and the
remaining investigation time. Token-count and model-metadata timeouts are the
smaller of 10 seconds and the remaining command time. Token-count and metadata
operations do not consume the four-call model counter, but all HTTP operations
consume the active command's wall-clock budget.

Read `ANTHROPIC_API_KEY` only from the process environment. Never accept it as
a CLI argument or config value, and never write it to artifacts, logs, reports,
receipts, exception text, or validation errors.

Thinking blocks are untrusted temporary response data. Discard them after
extracting the response fields needed for accounting and validation. Persist
only the validated structured result, stop reason, usage, and non-secret
request metadata. Never retain provider reasoning or thinking text.

### Cost and provider failures

Use USD 2 per million input tokens and USD 10 per million output tokens. Record
the pricing values, source identifier, and source date in every standalone run
and evaluation manifest.

Before every generation:

1. Create a unique logical request ID.
2. Calculate a write-ahead reservation using 110% of the counted input at the
   input price plus the full 1,600-token output allowance at the output price.
3. Add settled cost and every outstanding reservation. Do not continue when
   adding the new reservation would exceed USD 0.15 for a standalone
   investigation or USD 1.75 for the complete evaluation.
4. Append the logical request ID, counted input, prices, calculation, and
   reservation amount to the run's cost ledger and durably flush it. Only then
   may the SDK send the generation request.

`--max-cost-usd` is mandatory for every Claude generation command and may not
exceed the matching hard cap. The application uses the lower of the supplied
value and the hard cap. Settled cost plus all outstanding reservations count
against that effective cap.

Immediately after every response, append and durably flush the provider-
reported usage and actual cost with the logical request ID. Only after that
usage record is durable may a separate settlement event replace the outstanding
reservation with actual cost. Process neither content nor stop reason before
both writes finish.

After a timeout, process crash, missing usage, or any ambiguous result, retain
the full reservation as outstanding. Resume never repeats an outstanding
logical request; it marks that run application-generated `FAILED_SAFE`. If
usage or settlement persistence fails, stop the run and send no further
provider request.

Missing credentials, model unavailability, authentication failure, rate
limiting, network failure, timeout, token-count failure, and cost-cap denial
produce `FAILED_SAFE` with stable non-secret reason codes. Do not retry
automatically, fall back to another model or provider, enable automatic
recharge, or make a request after a cost gate fails.

### Stop-reason handling

Inspect response content only after usage and cost settlement is durable. Only
`end_turn` permits local schema and domain validation. If `end_turn` content
fails that validation, it may consume the one structured-output repair. No
other stop reason can trigger schema repair.

These stop reasons produce distinct stable application-generated failures:

| Stop reason | Disposition and reason code |
|---|---|
| `refusal` | `FAILED_SAFE`, `PROVIDER_REFUSAL` |
| `max_tokens` | `FAILED_SAFE`, `PROVIDER_MAX_TOKENS` |
| `model_context_window_exceeded` | `FAILED_SAFE`, `PROVIDER_CONTEXT_WINDOW_EXCEEDED` |
| `tool_use` | `FAILED_SAFE`, `PROVIDER_TOOL_USE` |
| `pause_turn` | `FAILED_SAFE`, `PROVIDER_PAUSE_TURN` |
| unexpected `stop_sequence` | `FAILED_SAFE`, `PROVIDER_STOP_SEQUENCE` |
| any other stop reason | `FAILED_SAFE`, `PROVIDER_UNEXPECTED_STOP_REASON` |

CausalOps configures no stop sequence, so every `stop_sequence` result is
unexpected. A refusal is billed provider behavior, not malformed structured
content. It is settled using reported usage and does not consume the repair.

Claude Pro may assist development through Claude Code but does not include the
application's Anthropic API usage. API credit and billing remain separate. An
explicit live CLI command authorizes only the requests needed for that command
and only within its required cost cap.

## 11. Evaluation and scoring

Compare:

1. **Single-pass baseline:** the pinned model receives the immutable initial
   alert packet and must emit a `FinalAssessment` without diagnostic tools.
2. **CausalOps:** the same model and packet may execute at most two validated
   evidence checks before emitting `FinalAssessment`.

Both systems use the same exact required model, adaptive-thinking effort,
`max_tokens`, input cap, pricing, prompt-level safety policy, taxonomy, and
initial packet. Neither system sends temperature or sampling controls. Their
task instructions differ only where tool-enabled workflow behavior requires
it.

Run the held-out evaluation variant three times for each of four families and
both systems: **24 scored runs**. Repetitions measure run-to-run behavior; they
are not additional incident classes. Run them sequentially and append each
completed record immediately. A run key combines evaluation ID, family,
system, and repetition so an interrupted benchmark can resume without
duplicating work.

### Mechanical scores

- **Diagnosis correctness:** selected `RootCauseCode` equals evaluator-only
  expected code.
- **Disposition correctness:** selected `Disposition` equals expected result.
- **Citation validity:** every cited evidence ID exists and belongs to the
  active incident.
- **Citation sufficiency:** cited evidence satisfies the evaluator-only required
  evidence predicates.
- **Control behavior:** invalid, denied, duplicate, and out-of-scope proposals.
- **Efficiency:** median and range of latency, tokens, executed tools, and model
  calls, plus complete per-run values.

Do not use an LLM judge. Do not report p95 from twelve runs per system. Publish
a paired baseline/workflow row for every run, the complete raw JSONL records,
aggregate counts, and at least one owner-written failure narrative. Any result
must state that it covers four synthetic incident families and 24 scored runs.

### Reproducibility manifest

Every `EvaluationRecord` includes:

- Git `HEAD` SHA and clean/dirty working-tree status.
- A source-patch SHA-256 covering the exact tracked diff from `HEAD` plus the
  bytes and repository-relative paths of untracked source files. A clean tree
  records the SHA-256 of the empty patch.
- Provider name, requested and response-reported exact required model name,
  Anthropic API version, and synchronous Anthropic Python SDK version.
- Adaptive-thinking type, effort, `max_tokens`, counted-input limit, and proof
  that `temperature`, `top_p`, and `top_k` were absent. Record
  `max_retries=0`, applied operation timeouts, and HTTP attempt counts.
- Prompt, schema, policy, tool-registry, scorer, and scenario versions.
- Development/evaluation variant and repetition number.
- Structured-output enforcement mode.
- Input/output prices, pricing source identifier/date, command cost cap,
  accumulated provider-reported token usage, and calculated cost.
- Logical request IDs, 110%-input reservation calculations, reservation and
  settlement event references, and settled/outstanding totals.
- Non-secret request IDs, stop reasons, provider latency, token-count results,
  deterministic-trimming count, and stable failure reason when present.
- Non-sensitive hardware summary.
- Initial-alert digest, final-context digest, timestamps, tokens, and latency.

Dirty-tree runs are useful for development but their scores are marked
non-publishable. A publishable diagnostic score requires an owner-approved
clean commit, matching `HEAD` SHA, clean status, and empty source-patch digest.

## 12. Threat model and tests

Protected assets are incident scope, tool registry, policy, budgets, evidence
integrity, evaluator ground truth, secrets, and the host environment. Attacker-
controlled inputs include model output, logs, metric labels, alert text, and
change descriptions.

Required threats, controls, and acceptance tests:

| Threat | Required control and test |
|---|---|
| Ground-truth leakage | Opaque model inputs; assert prompt/context lacks semantic scenario keys and expected values |
| Prompt injection in telemetry | Untrusted delimiters plus deterministic policy; verify no scope, tool, policy, or budget expansion |
| Arbitrary query execution | Template enums only; reject raw PromQL, shell, SQL, URL, and path input |
| Scope escape | Incident-labelled backends and allowlists; deny cross-run service, time, file, and evidence access |
| Forged citations | Resolve opaque evidence IDs from active store; reject missing and cross-incident IDs |
| Resource exhaustion | Enforce call, time, context, row, sample, and byte limits |
| Credential leakage | Environment-only API key plus redaction; verify it never reaches CLI text, config, artifacts, logs, reports, receipts, or errors |
| Provider data leakage | Send only bounded synthetic incident context; verify requests exclude secrets, evaluator ground truth, and host paths |
| Unbounded provider spend | Durably flushed write-ahead reservations plus settled/outstanding accounting; verify both caps, crash behavior, and no request after denial |
| Scenario contamination | Reset volumes/state and assert health and empty run scope before the next scenario |
| Model/tool failure | Timeout and malformed-output fixtures produce deterministic terminal states |

Additional required tests cover:

- Domain invariants and valid/invalid transitions.
- Every valid and invalid disposition/root-cause pairing, including proof that
  only application code can create `FAILED_SAFE`.
- One structured-output repair and repair exhaustion.
- Denied-proposal accounting and duplicate fingerprints.
- Deterministic clock, evidence ordering, quotas, and truncation markers.
- Identical serialized initial packets for baseline and workflow.
- Development/evaluation seed separation.
- Healthy start, bounded fault activation, repeatable signals, and cleanup for
  every incident family.
- Citation validity and required-evidence sufficiency scoring.
- Budget, model timeout, tool timeout, and dependency-unavailable behavior.
- Windows drive letters, path separators, UTF-8 files, writable-directory
  checks, and commands that are safe to copy into PowerShell.
- `causalops doctor` outcomes for missing Docker, missing API key, less than
  7.5 GiB detected total RAM, less than 12 GB free disk, the advisory warning
  below 2.5 GiB available RAM, authenticated model-metadata failure, exact
  required model mismatch, and success. Tests fake the SDK and assert an
  authenticated `GET /v1/models/claude-sonnet-5` call. They prove no generation
  is requested and make no external call.
- Exact Claude requests: required `claude-sonnet-5`, adaptive thinking,
  `medium` effort, `max_tokens=1600`, no `temperature`, `top_p`, or `top_k`, and
  one concurrent request.
- Synchronous SDK construction with `max_retries=0`; generation timeout equal
  to the smaller of 90 seconds and remaining investigation time; token-count
  and metadata timeout equal to the smaller of 10 seconds and remaining time;
  and exactly one inspected HTTP attempt per logical operation.
- Token-count and metadata calls do not change the model-call counter but do
  reduce the active wall-clock budget.
- Token counting over system text, messages, schema, and evidence; deterministic
  trimming, recounting, and blocking above 3,200 counted input tokens.
- USD 0.15 standalone and USD 1.75 evaluation gates; the 110%-input plus full-
  output reservation formula; unique logical request IDs; durable reservation,
  usage, and settlement ordering; and settled-plus-outstanding accounting.
- Crash, timeout, missing-usage, and ambiguous-response fixtures retain the
  full reservation. Resume never repeats the outstanding logical request and
  marks its run `FAILED_SAFE`.
- Resume rejection for changed model, effort, limits, pricing, or cap.
- `end_turn` content validation and repair, plus distinct no-repair
  `FAILED_SAFE` results for `refusal`, `max_tokens`,
  `model_context_window_exceeded`, `tool_use`, `pause_turn`, unexpected
  `stop_sequence`, and any unknown stop reason.
- A billed-refusal fixture proves usage and settlement become durable before
  failure handling and that the schema-repair budget is unchanged.
- Missing credentials, authentication, rate limit, network failure, timeout,
  count failure, and cost denial produce stable `FAILED_SAFE` records without
  retry or fallback.
- API-key redaction and proof that thinking blocks are never retained.
- Captured fake requests proving only bounded synthetic incident context is
  sent and no secret, evaluator ground truth, or host path leaves the process.
- Container memory ceilings, one-hour Prometheus retention, and one active
  scenario at a time.
- Sequential and resumed benchmarks, including proof that a completed run key
  is not executed or recorded twice.
- Clean and dirty provenance records, source-patch digest changes, and blocking
  publication for any run that is not tied to an owner-approved clean commit.
- Finalized result immutability and proof that scenario reset cannot delete
  records, reports, receipts, or cited evidence from standalone paths under
  `results/investigations/` or benchmark paths under `results/<evaluation-id>`.
- A manual smoke test on the working platform, Linux x86-64, with all required
  containers and one explicitly authorized, USD 0.15-capped Claude
  investigation. Record the advisory warning if available RAM is below 2.5 GiB.
  Windows support is proven by continuous integration on `windows-latest`, not
  by a second manual run.
- A manual readability review covering concrete names, limited nesting, useful
  comments and docstrings, plain documentation, and no decorative abstractions.

Normal CI runs on `windows-latest` and `ubuntu-latest` and uses replay fixtures,
fake Anthropic SDK clients, and disposable local test data. Network access is
allowed only while installing the locked dependencies. After installation,
formatting, linting, strict typing, unit tests, security tests, and replay
conformance make no external calls and require no credentials or paid usage.
Doctor's model lookup, token counting, generation, refusal, and provider
failures are tested through fakes or mocks. Local in-process and loopback test
traffic is allowed. Outside CI, only an explicitly invoked live command may send
authenticated HTTPS requests to Anthropic, subject to its cost gate.

## 13. Definition of done

The MVP is complete only when:

- A clean clone installs and passes formatting, linting, strict typing, and
  tests on both `windows-latest` and `ubuntu-latest`.
- `causalops doctor` verifies a supported environment, hard memory and disk
  thresholds, required API key, and exact required model metadata before a
  scored run. It warns below 2.5 GiB available RAM and cannot check Console
  credit balance.
- All four development and evaluation variants activate, assert, and reset
  reproducibly without cross-run leakage.
- One real scenario works end to end through opaque incident ID, Prometheus,
  JSONL logs, policy, tools, report, and scorer.
- Every executed investigator tool is typed, scoped, read-only, bounded, and
  recorded.
- Replay conformance covers valid diagnosis, correct abstention, repair,
  malformed output, denial, timeout, and budget exhaustion.
- The synchronous Claude adapter enforces the exact required model, adaptive
  thinking at medium effort, exact token limits, environment-only credential,
  deterministic count/trim/recount flow, `max_retries=0`, deadline-derived
  timeouts, one HTTP attempt per operation, and no fallback.
- Every generation has a durably flushed write-ahead reservation with a unique
  logical request ID. Settlement follows durable provider usage; uncertain
  requests retain their full reservation, are never repeated on resume, and
  cause their run to be marked `FAILED_SAFE`.
- Only a durably accounted `end_turn` response can reach content validation.
  Every other documented stop reason produces its distinct no-repair
  `FAILED_SAFE` result, while invalid `end_turn` content alone may use schema
  repair.
- The ambiguous replay and Claude cases can be scored as abstention rather
  than operational failure.
- The pinned Claude benchmark produces all 24 traceable paired records within
  the USD 1.75 evaluation cap.
- An interrupted benchmark resumes without repeating completed run keys.
- Finalized standalone and benchmark artifacts remain immutable under their
  investigation or evaluation ID and survive scenario reset.
- Reports contain mechanical scores, complete failures, limitations, and no
  replay-derived accuracy claims.
- Published diagnostic scores come only from an owner-approved clean commit
  whose recorded provenance matches the evaluated source.
- A concise threat model is documented. Any ADR records an owner-approved
  decision and rationale; no ADR is required when no such decision exists.
- A five-minute recording shows one diagnosis, one abstention, and the paired
  evaluation report.

If implementation stops before the Claude benchmark, it may be published as an
evaluation harness with replay conformance, but it is not complete and cannot
claim demonstrated diagnostic improvement.

## 14. Three-phase delivery sequence

Every step is one bounded vertical slice and must pass the coder, two-reviewer,
owner-approval, correction, re-review, and final-explanation gate defined in
`AGENTS.md`. Do not begin the next step while the current gate is open. A phase
starts only after all three steps in the preceding phase close.

### Phase 1: foundation and first vertical slice

1. **Windows preflight and package foundation:** add packaging, quality tools,
   local `causalops doctor` checks, configuration, and Windows-safe paths. The
   provider metadata check belongs to Phase 3 step 1.
2. **Investigation core:** add domain types, replay model, policy, budgets,
   JSONL run records, the investigation loop, unit scoring, and focused tests.
   Do not add the synthetic lab, tool backends, scenario wiring, or benchmark
   orchestration in this step.
3. **First real vertical slice:** add the synthetic lab, three services,
   Prometheus, scenario controller, all four working tool backends for one
   opaque-ID incident, end-to-end wiring, report, and scorer integration. Reuse
   the core and scoring rules from step 2 rather than redefining them.

### Phase 2: complete lab and safety

1. **Harden telemetry tools:** generalize the four working backends and their
   registered templates for every family; add deterministic context,
   cross-incident isolation, and result bounds. Do not add benchmark
   orchestration or aggregation.
2. **Incident families:** add the remaining families, development and
   evaluation seeds, misleading evidence, repeatable activation, and cleanup.
3. **Security and conformance:** add ground-truth isolation, injection, scope
   escape, forged citations, malformed output, timeout, budget, and replay
   conformance cases.

### Phase 3: Claude API and portfolio evidence

1. **Claude API integration:** add the official synchronous Anthropic SDK,
   authenticated doctor metadata lookup, pinned model and request settings,
   token counting, credential protection, cost gates, provider failure
   behavior, and single-pass baseline.
2. **Benchmark delivery:** add only sequential 24-run orchestration, aggregate
   calculations over the existing scorer, resume behavior, finalized raw
   records, and paired reports. Do not redefine tools or scoring rules.
3. **Portfolio handoff:** add the plain-language README and threat model,
   limitations, failure analysis, reports, and a five-minute demo. Create an
   ADR only for an explicit owner-approved decision; zero ADRs is valid.

Do not replace missing evaluation with a UI, cloud deployment, or additional
architecture.

## 15. Deferred extensions

Consider these only after the definition of done and an explicit update to this
specification:

- MCP adapter around the typed tool registry.
- OpenTelemetry traces as another evidence source.
- SQLite-backed investigation restart.
- Additional incident families and a larger benchmark.
- Optional second-provider comparison for a specific measured question.
- Static hosted report or recorded-results viewer.
