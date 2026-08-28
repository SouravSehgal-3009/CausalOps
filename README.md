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
denied proposal never reaches a backend at all. This is tested three ways:
an import scan proving the dispatch node imports no backend module directly,
a wrapper-identity check proving every registered tool callable was actually
built by a wrapper factory (a hand-built wrapper raises `TypeError`), and a
spy-backend test proving a denied proposal never invokes one.

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

- **Tool-policy bypass** — proven unreachable by the import-scan,
  wrapper-identity, and spy-backend tests described above.
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
metric, a model that occasionally guessed the wrong service argument, a
scoring bug that vacuously passed a citation check with nothing cited. One
was boundary-adjacent and more serious: an early cost-ledger implementation
settled a request's real cost without checking it against the reservation
that authorized it, so an overrun on one request could become permanently
invisible to the spend ceiling — reproduced concretely (a $0.01 reservation
settling at $0.03 under a $0.02 cap, after which a further $0.01 request was
still wrongly accepted, for $0.04 of real spend against a $0.02 authorized
limit) and fixed before merge.

Every one of these, including that one, was caught by review before it ever
reached a live-exploited state — never after. That is the honest claim:
not "no boundary-adjacent bug ever happened," but "review caught every one
before it mattered."

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
| `causalops scenario start <family> --seed <development\|evaluation>` | Starts one incident, prints its opaque incident ID. |
| `causalops scenario reset <incident-id>` | Clears one incident's active lab state. Never touches `results/`. |
| `causalops investigate <incident-id> --model <replay\|claude>` | Runs a full investigation; `replay` is free and deterministic, `claude` is a real billed request. |
| `causalops approve <thread-id>` | Accepts a paused investigation's diagnosis or abstention. |
| `causalops reject <thread-id> "<reason>"` | Rejects a paused investigation and records why. |
| `causalops-evaluate` | Runs the fixed paired live-evaluation corpus (separate binary, takes no arguments). |

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
uv run causalops-evaluate
```

A genuinely separate console script, not a `causalops` subcommand — it runs
a fixed, held-out four-family corpus against the live model, one no-tool
baseline and one tool-enabled run per family, saving every record and a
per-arm summary under `results/evaluations/<id>/`. It
requires `ANTHROPIC_API_KEY`, persists each completed record as it finishes
(not only at the end), and stops issuing further paid requests only after an
infrastructure-level failure (a missing credential, a provider error, or the
cost ceiling itself) — an ordinary model mistake is still scored as a
result, not treated as a reason to abort the batch.

Reported scores are mechanical: diagnosis and disposition correctness
against evaluator-only labels, citation validity and sufficiency against
required-evidence predicates, and a joint correct-and-grounded figure
combining the two. Every record also carries the git SHA, clean/dirty
status, fixture and prompt versions, retrieval mode, exact model, tokens,
latency, and cost — reproducibility is part of the record, not an
afterthought. Given the small sample size, results are reported as counts
and ranges, never as a p95 or a broad performance claim.

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
