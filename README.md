# CausalOps

An incident investigation assistant for a local synthetic microservice lab.

`TECHNICAL_SPEC.md` is the product specification for CausalOps v2;
`TECHNICAL_OVERVIEW.md` records what has actually been built. This README
only points at them; it does not define behavior.

## Status

Phases 1 and 2 are complete: the synthetic lab, Prometheus and JSONL
telemetry, all four incident families, the four read-only tool backends,
policy, budgets, run records, and scoring. Milestones 1 and 2 are complete:
`uv run causalops investigate <incident-id> --model replay` runs an incident end to
end through the LangGraph `StateGraph` orchestrator, including the
escalation interrupt and owner approval/rejection — see
`TECHNICAL_OVERVIEW.md` Part III for what it currently supports. Milestone 3
is in progress: FTS5 runbook retrieval, the live Claude adapter
(`--model claude`), and the paired evaluation (`causalops-evaluate`) under
the cost cap have all landed on the `paired-live-evaluation` branch, not
yet merged to `master` or run for real against a live model.

## Check this machine

```powershell
uv sync --locked
uv run causalops doctor
```

The same commands run from a POSIX shell on Linux x86-64.

The operating system, total RAM, free disk, writable directories,
checkpoint database, and Docker checks are hard failures. Low available RAM
and a missing `ANTHROPIC_API_KEY` warn without failing — `--model replay`
needs neither. `doctor` exits 0 unless a hard check fails, printing a
stable reason code for each problem.

The authenticated `claude-sonnet-5` metadata request described in
`TECHNICAL_OVERVIEW.md`'s CLI contract section is not implemented yet.

`doctor` reads `ANTHROPIC_API_KEY` from the process environment only. Set it in
PowerShell before a live command:

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
```

## Running a live investigation

`uv run causalops investigate <incident-id> --model claude` sends a real, billed
request to Anthropic. `--model replay` sends nothing to Anthropic — it
reads a checked-in fixture, and the only network traffic either mode makes
on its own account is to the local lab. Before running `--model claude`,
follow `TECHNICAL_OVERVIEW.md`'s "Unit 3b-2 — running the live smoke call"
section for the exact command sequence, the preconditions, and what a
successful, escalated, or refused run's artifacts look like.
