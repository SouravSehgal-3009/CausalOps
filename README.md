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
is complete and merged: FTS5 runbook retrieval, the live Claude adapter
(`--model claude`), and the paired evaluation (`causalops-evaluate`) have
all landed on `master`. `causalops-evaluate` requires `ANTHROPIC_API_KEY`,
persists each completed record, and stops further paid requests only after
an infrastructure failure, leaving model-quality failures as scored
results. It has been run for real against the live Claude API multiple
times, with each run's records and per-arm summary saved under
`results/evaluations/`.

A follow-up defect-remediation arc (`TECHNICAL_OVERVIEW.md`'s "Unit A"/"Unit
6 follow-up" sections and their addenda) landed on top of Milestone 3
afterward, fixing real defects those live runs surfaced — see
`TECHNICAL_OVERVIEW.md` for what changed and why; this README doesn't
narrate it.

CausalOps remains decision support over a synthetic lab and synthetic data
throughout — it never executes remediation, mutates the lab, or acts as an
autonomous operator.

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
