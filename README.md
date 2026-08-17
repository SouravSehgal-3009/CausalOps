# CausalOps

An incident investigation assistant for a local synthetic microservice lab.

`TECHNICAL_OVERVIEW.md` is the specification for this project. This README only
points at it; it does not define behavior.

## Status

The environment check below and the investigation core library are implemented:
domain contracts, the replay reasoning model, the tool registry, policy, budgets,
the evidence store, the investigation loop, run records, and scoring. There is no
synthetic lab and no `investigate` command yet.

## Check this machine

```powershell
uv sync --locked
uv run causalops doctor
```

The same commands run from a POSIX shell on Linux x86-64.

`doctor` checks the Windows version, total and available RAM, free disk on the
project drive, that `runs/` and `results/` are writable, that Docker answers,
and that `ANTHROPIC_API_KEY` is set. It exits 0 when every hard check passes and
1 otherwise, printing a stable reason code for each problem.

The authenticated `claude-sonnet-5` metadata request described in
`TECHNICAL_OVERVIEW.md` section 9 is not implemented yet.

`doctor` reads `ANTHROPIC_API_KEY` from the process environment only. Set it in
PowerShell before a live command:

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
```
