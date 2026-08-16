# CausalOps

An incident investigation assistant for a local synthetic microservice lab.

`TECHNICAL_OVERVIEW.md` is the specification for this project. This README only
points at it; it does not define behavior.

## Status

Early. Only the environment check below is implemented.

## Check this machine

```powershell
uv sync --locked
uv run causalops doctor
```

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
