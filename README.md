# CausalOps

An evidence-grounded incident investigator for a local, synthetic
microservice lab. CausalOps forms competing hypotheses about the cause of a
synthetic incident, runs a small number of safe read-only diagnostic checks
against a local Docker Compose lab, and returns either a cited diagnosis or
an explicit abstention — never a guess dressed up as confidence.

"Causal" describes the loop CausalOps runs — hypothesis, diagnostic check,
evidence update — not formal causal inference. CausalOps is decision support
for an on-call engineer, not an autonomous operator: it never executes
remediation, mutates the lab, or acts on production-like systems. Every
system and every incident it investigates is synthetic.

The central trust boundary, unchanged everywhere in this project:

> The model proposes and interprets. Deterministic code validates,
> authorizes, executes read-only checks, stops, scores, and records.

## How it works

CausalOps runs an incident investigation as a LangGraph `StateGraph`:

```text
CREATED
  -> investigate            (the model proposes a hypothesis and, optionally, one tool call)
  -> dispatch_tool           (a policy-wrapped read-only check runs, or the proposal is denied)
  -> normalize_evidence      (the result becomes a typed, bounded Evidence record)
  -> investigate | final_assessment   (looped, budget-gated)
  -> final_assessment        (DIAGNOSED or INSUFFICIENT_EVIDENCE, cited)
  -> escalation_interrupt     (only when a defined trigger fires)
  -> final_report
```

The model never talks to a tool backend directly. Every read-only tool call
goes through a policy wrapper that validates the incident scope, the
registered template, and the remaining budget *before* anything runs, and a
denied proposal never reaches a backend at all — see "Tool-policy bypass"
under "Safety and threat model, briefly" below for exactly how that's
tested.

The model can never submit raw PromQL, shell, SQL, a URL, a filesystem path,
or code — only a registered template ID and strictly typed arguments.
Application code turns a template selection into the real query.

### The synthetic lab

Docker Compose runs three project-authored Python services plus Prometheus:

```text
gateway -> orders -> inventory
              |
       bounded resource pool
```

`orders` holds a bounded, saturable resource pool implemented in plain
Python. A scenario controller — a separate trust domain from the
investigator, never callable through model output — starts an incident,
injects one of four fault families, verifies the fault signal, and hands
the investigator only an opaque incident ID and an answer-neutral alert.
Every evaluator-only fact (the scenario family, the expected root cause, the
evidence predicates that will be used to score the run) is held out of the
model's context entirely and enforced by import-graph and content tests, not
just convention.

### Incident families

1. **Configuration change** — an orders configuration change causes
   failures.
2. **Downstream timeout with retry amplification** — inventory latency
   causes retries and elevated gateway latency.
3. **Resource-pool saturation** — the bounded orders resource pool is
   exhausted and degrades requests.
4. **Ambiguous telemetry** — differentiating evidence is absent or
   contradictory; the correct answer is abstention (`UNDETERMINED` /
   `INSUFFICIENT_EVIDENCE`), not a guess.

Every family requires at least one follow-up tool call — the initial alert
alone is never enough to diagnose correctly.

### Tools available to the model

| Tool | Typed input | Backend |
|---|---|---|
| `query_metric` | Registered PromQL template, service, bounded window | Prometheus |
| `query_logs` | Registered filter, service, bounded window, row limit | Active-run JSONL logs |
| `list_recent_changes` | Service, bounded window | Change manifest |
| `get_topology` | Active incident ID | Topology manifest |
| `search_runbooks` | Registered topic, passage limit | Local SQLite FTS5 index over a small curated runbook corpus |

`search_runbooks` is lexical retrieval (FTS5) by default; every report and
evaluation record labels the retrieval mode it actually used
(`disabled`, `fts5_lexical`, or `pinecone_semantic`) rather than leaving it
implicit. Retrieved runbook text is untrusted data — it is quoted and
delimited in the model's context and cannot alter policy, extend scope, or
authorize a tool call on its own.

### A real defect this project found and fixed

Paired evaluation runs are how this project catches problems code review
alone misses. One showed up in the tool arguments themselves, not in the
policy or the graph.

`QueryMetricArguments.service` and `QueryLogsArguments.service` started as
bare, undescribed `str` fields — nothing told the model which service
actually emits which metric or log category. Across two real paid
evaluation batches (8 tool-enabled runs total), the model guessed
`service="inventory"` for `resource_pool_attempts_per_capacity`, a metric
only `orders` ever records, in 3 of the 8 runs. Each wrong guess returned
zero samples and burned half of that run's 2-check evidence budget on a
query that could never have returned anything.

The fix was a `Field(description=...)` on both arguments, naming the exact
per-service restrictions in prose (`src/causalops/tools.py`) — no policy or
graph code changed, only what the model was told about a tool it already
had. Re-run after the fix: zero wrong-service guesses, and `query_logs`
executed successfully in a real batch for the first time in this project's
history — it had been proposed once before and denied for an unrelated
reason (a `row_limit` mismatch between the tool's schema, which allows up
to 200, and the policy-enforced budget of 40). That gap between schema and
budget is deliberate and permanent, not a bug: a schema bound is a hard
shape limit, a budget is what policy actually allows through, and the two
are meant to stay independently editable. What changed here was only the
denial message — the field's own schema description still said nothing
about the real number. The same guess kept recurring at scale; see "Paired
live evaluation" below for how that was found and fixed.

This is this project's clearest example of a defect invisible to code
review — visible only by running real evaluations and reading what the
model actually did, not by reading the tool's code.

### Budgets

| Limit | Default |
|---|---:|
| Diagnostic checks executed | 2 |
| Model calls, including one structured-output repair | 4 |
| Structured-output repairs | 1 |
| Live-model spend, application-wide, all runs combined | USD 5.00 |

A denied or invalid proposal still consumes a model-call slot but is never
counted as an executed check. If budget runs out before a valid diagnosis or
abstention is reached, the investigation ends `FAILED_SAFE` — a disposition
only application code can produce, never something the model selects.

A configured ceiling too small to cover even the cheapest possible request
is refused outright at startup, rather than silently accepted and then
refusing every real request one at a time — see `.env.example` for the
exact refusal conditions.

### Safety and threat model, briefly

CausalOps is reviewed against a fixed set of threats, each backed by a real,
currently-passing test rather than a design intention:

- **Tool-policy bypass** — proven unreachable by three independent tests,
  each closing a gap the other two leave open.

  The import scan (`test_the_dispatch_boundary_modules_import_no_backend`)
  checks that `tool_wrappers.py`, `tool_calls.py`, and `graph.py` import
  none of `causalops.telemetry`, `causalops.prometheus`, or
  `causalops.runbooks`. That's necessary, not sufficient: a scan checking
  only those imports would still pass even if a backend were wired in
  through the registry's `lambda` arguments in `live_setup.py`'s
  `build_model_and_registry` — an indirection no import statement ever
  names.

  The wrapper-identity check proves a registry entry was actually built by
  a wrapper factory, not just that it looks like one. `ToolWrapper` is a
  frozen dataclass with a private `_factory_token` field that defaults to
  `None`; only `_make_wrapper` (used by every real factory —
  `query_metric_wrapper`, `query_logs_wrapper`, and the rest) ever supplies
  the real sentinel, `_WRAPPER_FACTORY_TOKEN`. `ToolWrapper.__post_init__`
  checks identity against that sentinel and raises `TypeError` on any
  mismatch, so a hand-built `ToolWrapper(tool=..., dispatch=some_closure)`
  fails at construction, before it can join a registry —
  `test_a_hand_built_tool_wrapper_is_rejected` is exactly that
  reproduction. `isinstance(x, ToolWrapper)` alone would not have caught
  it, since a hand-built instance satisfies that check too.

  The spy-backend test,
  `test_every_registered_tool_denies_an_out_of_scope_proposal_untouched`,
  proves a denial actually stops execution, not just that it gets labeled
  `DENIED`. It wires five separate spy backends, one per tool, sends each
  an out-of-scope proposal, and asserts both the denial and that its own
  spy recorded zero calls, independently. A single shared spy watching one
  tool position, with the other four wired to unwatched stand-ins, could
  report green while three or four wrappers silently leaked straight
  through — nothing would ever record it. Five independent spies mean a
  regression in any one wrapper is caught by that tool's own assertion,
  never masked by the other four passing.

- **Ground-truth leakage** — the model and the retrieval corpus never see
  the evaluator's scenario key, expected root cause, or evidence predicates;
  enforced by import-graph and content assertions, not just file placement.
- **Prompt injection** — telemetry and retrieved runbook text are wrapped as
  untrusted, delimited evidence; adversarial fixtures prove an injected
  instruction cannot expand scope, register a tool, or authorize an action
  policy would otherwise deny.
- **Scope escape and forged citations** — every tool call is checked against
  the active incident's own scope before it runs, and every cited evidence
  ID is re-resolved from the active incident's own store before it can reach
  a report; a forged or cross-incident ID never survives to output.
- **Resource exhaustion** — call, time, row, sample, and byte caps are
  enforced at multiple layers, independent of what the model asks for.
- **Provider and secret leakage** — the live model adapter reads its API key
  only from the process environment and never names or logs it; it sends
  only the same bounded, synthetic context that ground-truth isolation and
  prompt-injection tests already constrain.
- **Unbounded provider spend** — every live request is reserved against the
  application-wide ceiling *before* it is sent, using a durably persisted,
  conservative estimate; the reservation is exactly-once settled from the
  provider's real reported usage, and a request that would exceed the
  remaining ceiling is refused before it is sent, never after.

These boundaries have been tested by real defects during development, and
they held. Most were unrelated to any boundary at all: a mismeasured lab
metric, the wrong-service-argument defect described above, the row-limit
guess and repair-starvation defect described under "Paired live evaluation"
below, a scoring bug that vacuously passed a citation check with nothing
cited. One was boundary-adjacent and more serious: an early cost-ledger
implementation settled a request's real cost without checking it against
the reservation that authorized it, so an overrun on one request could
become permanently invisible to the spend ceiling — reproduced concretely
(a $0.01 reservation settling at $0.03 under a $0.02 cap, after which a
further $0.01 request was still wrongly accepted, for $0.04 of real spend
against a $0.02 authorized limit) and fixed before merge.

Every one of these was caught before it became a trust-boundary violation —
most by review before a live run, the wrong-service-argument and
row-limit/repair-starvation defects only by running real paid evaluations
and reading what the model actually did, not by review beforehand. That is
the honest claim: not "no boundary-adjacent bug ever happened," but "none of
them ever crossed a boundary above, whether review or evaluation is what
caught it."

## Setup

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/), and Docker
Compose. CausalOps runs on any machine `causalops doctor` can read a
platform, RAM, and disk reading from — there is no allowlist of specific
operating systems, Windows builds, or CPU architectures; every capability
CausalOps actually needs (Docker responding, enough memory and disk, a
writable checkpoint database and run directories) has its own explicit
check instead.

```bash
uv sync --locked
uv run causalops doctor
```

`doctor` checks the operating system reading, total and available RAM, free
disk, required writable directories, the checkpoint database, Docker, and
whether `ANTHROPIC_API_KEY` is set. The operating system, RAM (total),
disk, directory, database, and Docker checks are hard failures; low
available RAM and a missing API key only warn, since `--model replay` (see
below) needs neither. `doctor` exits 0 unless a hard check fails, and
prints a stable reason code for each problem it finds.

A live model call needs `ANTHROPIC_API_KEY` in the process environment —
there is no `.env` loader, so export it directly:

```bash
export ANTHROPIC_API_KEY="<your key>"
```

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
```

See `.env.example` for every environment variable CausalOps reads, including
`LIVE_EVALUATION_MAX_USD` (the application-wide live-spend ceiling described
above; defaults to 5.00 if unset).

## Command reference

| Command | What it does |
|---|---|
| `causalops doctor` | Checks this machine can run CausalOps; see Setup above. |
| `causalops lab up` | Starts the Docker Compose lab and waits for it to be healthy. |
| `causalops lab down` | Stops the lab. |
| `causalops scenario start <family> --seed <development\|evaluation\|evaluation_b\|evaluation_c>` | Starts one incident, prints its opaque incident ID. |
| `causalops scenario reset <incident-id>` | Clears one incident's active lab state. Never touches `results/`. |
| `causalops investigate <incident-id> --model <replay\|claude>` | Runs a full investigation; `replay` is free and deterministic, `claude` is a real billed request. |
| `causalops approve <thread-id>` | Accepts a paused investigation's diagnosis or abstention. |
| `causalops reject <thread-id> "<reason>"` | Rejects a paused investigation and records why. |
| `causalops-evaluate [--executed-tools <2\|3\|4>]` | Runs the fixed paired live-evaluation corpus at one evidence-budget curve point (separate binary; defaults to 2). |

## Running an investigation

Start the lab, then start one synthetic incident:

```bash
uv run causalops lab up
uv run causalops scenario start resource_pool_saturation --seed development
#   -> prints an opaque incident id, e.g. a1b2c3d4e5f6...
```

`scenario start` takes an owner-facing family name (one of the four listed
above) but only ever returns an opaque incident ID to the caller — the
semantic family name is never passed on to `investigate`, which never
learns it.

Investigate the incident. `--model` is required, with no default, so a live
run is never accidental:

```bash
# Replay mode: no network call, zero cost, deterministic fixture playback.
uv run causalops investigate <incident-id> --model replay

# Live mode: a real, billed request to Anthropic (claude-sonnet-5),
# reserved and settled against LIVE_EVALUATION_MAX_USD.
uv run causalops investigate <incident-id> --model claude
```

A completed investigation writes its cited evidence, tool receipts, run
record, and a Markdown report to
`results/investigations/<investigation-id>/`. `DIAGNOSED` and
`INSUFFICIENT_EVIDENCE` are both successful, exit-0 outcomes —
`INSUFFICIENT_EVIDENCE` means the investigation correctly recognized the
evidence couldn't distinguish a cause, not that anything went wrong.
`FAILED_SAFE` and an unavailable dependency exit nonzero with a stable
reason code.

When you're done with an incident:

```bash
uv run causalops scenario reset <incident-id>
```

`reset` only removes that incident's active lab and transient state; it
never touches a finalized report or record under `results/`.

### Escalation: owner approval and rejection

Some investigations pause for owner review instead of finishing
automatically — specifically when the model's evidence conflicts, a tool
becomes unavailable mid-investigation, evidence is insufficient with a check
still available, or runbook retrieval coverage is judged insufficient.
CausalOps never uses an uncalibrated model confidence score to decide this;
only one of those four deterministic reasons triggers it. A paused
investigation exits `3` and prints a `thread_id` you resume with:

```bash
uv run causalops approve <thread-id>
uv run causalops reject <thread-id> "<reason>"
```

`approve` accepts the paused diagnosis or abstention and resumes the graph
to a finished report. `reject` records the owner's disposition and reason
without changing the underlying assessment, then also finishes the report.
Both routes are checkpointed through SQLite (`checkpoints.db`), so a resume
survives a process restart, and an identical retry returns the same
recorded decision rather than resuming twice.

**This mechanism has been verified end to end against the real Docker lab,
not only through directly-constructed test fixtures.** Two real runs, both
under `--model replay` (zero API cost): starting a `configuration_change`
incident and deleting its `orders` log file before `investigate` makes the
first scripted log check come back `TOOL_UNAVAILABLE`, which is enough to
trigger escalation on its own.

- **Approve path** — `causalops approve <thread-id>` resumed the paused
  investigation to a finished `DIAGNOSED CONFIG_CHANGE` report with
  `"decision": "accept"` recorded alongside it.
- **Reject path** — `causalops reject <thread-id> "<reason>"` resumed a
  second, separately paused investigation to the same underlying
  `DIAGNOSED CONFIG_CHANGE` assessment, this time with `"decision":
  "reject"` and the given reason recorded — rejecting records the owner's
  disposition of the assessment, it does not change or re-run it.

Both runs were confirmed clean before and after (no stray source, test, or
lab-state changes left behind).

### Paired live evaluation

```bash
uv run causalops-evaluate                      # executed_tools=2 (default)
uv run causalops-evaluate --executed-tools 3
uv run causalops-evaluate --executed-tools 4
```

A genuinely separate console script, not a `causalops` subcommand — it runs
a fixed, held-out 12-incident corpus (4 families x 3 seeds — `evaluation`,
`evaluation_b`, `evaluation_c`) against the live model: one no-tool baseline
and one tool-enabled run per incident, saving every record and a per-group
summary under `results/evaluations/<id>/`. Each invocation runs exactly one
point on an evidence-budget curve — `Budgets(executed_tools=N,
model_calls=N + 2)` for `N` in `{2, 3, 4}` — never all three in one run, so
real spend can be checked between phases rather than committed at once; the
owner runs the command up to three times, once per `--executed-tools`
value, to build the full curve. Before any scenario starts, a pre-flight
check refuses cleanly if the configured ceiling could not possibly cover
this invocation's own worst-case batch cost, on top of what the application
has already spent or committed. It requires `ANTHROPIC_API_KEY`, persists
each completed record as it finishes (not only at the end), and stops
issuing further paid requests only after an infrastructure-level failure (a
missing credential, a provider error, or the cost ceiling itself) — an
ordinary model mistake is still scored as a result, not treated as a reason
to abort the batch.

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
only the denial message — see "A real defect this project found and fixed"
above — and likewise for `SearchRunbooksArguments.limit`; structured-output
repairs now draw from their own independent budget (`Budgets.repairs`), so
a denial earlier in a run can no longer starve a repair a later turn needs;
and the fields that hit the 300-character cap in real runs were raised to
600.

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
in the tool-enabled arm — the model consistently preferred direct
telemetry tools when the evidence-check budget was scarce, so the
`SearchRunbooksArguments` fix above has not yet been exercised by a live
call.

The three post-fix batches cost $3.02 in real spend; all six batches in
this investigation, pre- and post-fix combined, cost $6.40.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src lab
uv run pytest -q -m "not docker"
```

Docker-marked tests (`-m docker`) run against the real lab and are excluded
from the default run above; bring the lab up first with `causalops lab up`
before running them.

Repository shape:

```text
src/causalops/     application code (graph, tools, policy, CLI, telemetry adapters)
lab/                the synthetic Docker Compose services and their fixtures
tests/unit/         unit tests
tests/integration/  tests against the real Docker lab
tests/security/     trust-boundary and isolation tests
results/            gitignored investigation and evaluation artifacts
```

## Non-goals

CausalOps does not build causal graphs, estimate counterfactual outcomes, or
run more than one investigator. It has no remediation executor: it may
record an owner-approved suggested next step, but it never executes,
verifies, or claims to fix anything. It does not add a web UI, a second
model provider, a second database, Kubernetes, or cloud hosting. All data —
services, telemetry, incidents — is synthetic; nothing here touches a real
production system.

## License

MIT — see [`LICENSE`](LICENSE).
