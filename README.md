# CausalOps

An incident investigation assistant for a local synthetic microservice lab.

`TECHNICAL_SPEC.md` is the product specification for CausalOps v2;
`TECHNICAL_OVERVIEW.md` records what has actually been built. This README
only points at them; it does not define behavior.

## Status

Phases 1 and 2 are complete: the synthetic lab, Prometheus and JSONL
telemetry, all four incident families, the four read-only tool backends,
the replay-driven investigation loop, policy, budgets, run records, and
scoring. `causalops investigate <incident-id> --model replay` runs an
incident end to end. There is no live Claude adapter yet — see
`TECHNICAL_OVERVIEW.md` Part I, "Phase 3 — never started."

## Check this machine

```powershell
uv sync --locked
uv run causalops doctor
```

The same commands run from a POSIX shell on Linux x86-64.

`doctor` checks the operating system, total and available RAM, free disk on the
project drive, that `runs/` and `results/` are writable, that Docker answers,
and that `ANTHROPIC_API_KEY` is set. It exits 0 when every hard check passes and
1 otherwise, printing a stable reason code for each problem.

The authenticated `claude-sonnet-5` metadata request described in
`TECHNICAL_OVERVIEW.md`'s CLI contract section is not implemented yet.

`doctor` reads `ANTHROPIC_API_KEY` from the process environment only. Set it in
PowerShell before a live command:

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
```
