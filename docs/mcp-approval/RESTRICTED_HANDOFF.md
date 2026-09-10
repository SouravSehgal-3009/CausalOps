# Phase 3 Restricted-Environment Handoff

## Scope and Preconditions

Run this only on the approved private VM after the Phase 3 branch changes are
reviewed. Do not run Docker, Ollama, MCP connectivity, image pulls, or any
networked test from a work laptop. Keep `ENABLE_CLAUDE=false`; hosted API and
dashboard paths remain replay-only.

The checked-in MCP capability pin is implemented in
`causalops.mcp_manifest`. It fixes the protocol revision at `2025-11-25` and
permits only the five existing read-only observability tools. A schema-pin
failure or an unexpected `tools/list` result is a deployment refusal, not a
fallback to discovery.

This initial local slice intentionally does not launch an MCP server or replace
direct telemetry dispatch. Apply the VM steps after the separately reviewed
stdio server/client composition is available and deterministic policy approval
authorizes its use.

## VM Deployment

1. Check out the reviewed branch and run `uv sync --locked`.
2. Configure persistent `results/` storage and the untracked VM environment.
   Export `CAUSALOPS_EXECUTION_ENV=vm` in the shell that invokes the wrapper
   (it is deliberately not read from `compose.env`); set
   `CAUSALOPS_CANDIDATE_EVALUATION=true` only for an approved candidate run.
3. Create ignored `infra/phase3/compose.env` from `compose.env.example`, set a
   reviewed immutable Ollama image digest, then start only the guarded Phase 3
   Ollama service through the digest-validating wrapper:
   ```sh
   infra/phase3/run_compose.sh --env-file infra/phase3/compose.env up -d
   ```
   `compose.env` must contain only comments/blank lines and one unquoted,
   literal `CAUSALOPS_OLLAMA_IMAGE=<digest>` assignment. The wrapper rejects
   mutable tags, malformed digests, alternate dotenv syntax, additional
   assignments, inherited Compose source variables, and caller-provided
   Compose file/environment overrides before it invokes Compose. Normal
   operational arguments such as `up -d`, `logs`, `ps`, and `config` remain
   available.
   Ollama is bound to loopback; do not expose it through an ingress or public
   firewall rule.
4. Start the existing synthetic lab separately, only under the approved VM
   Docker procedure, with `lab/docker-compose.yml`. It is not a service in
   `infra/phase3/docker-compose.yml` and must not be started from a work
   laptop.
5. Pull only the approved `qwen3.5:4b` image on the VM. Record its digest and
   the Compose image digests in sanitized validation evidence.

## Deferred Worker and MCP Composition

The checked-in Phase 3 Compose template intentionally starts no application
worker and no MCP stdio child. Do not interpret a successful
`run_compose.sh ... up -d` as an investigation, lab, or MCP deployment.

After `POLICY_APPROVAL.md` is satisfied on the private VM, a separately
reviewed composition change must launch the replay-only worker and local stdio
MCP child. It must use `policy_approved_mcp_server()` with a non-null reviewed
approval record, retain the existing synthetic lab as a separate Compose
deployment, bind no MCP network listener, and validate crash/reconnect and
policy-equivalence behavior on the VM before enabling dispatch.

## MCP Manifest Gate

1. Launch the observability MCP server as a local stdio child process.
2. Complete MCP initialization using the pinned protocol revision, then issue
   `tools/list`.
3. Parse its result as `McpToolDiscovery` values and call
   `verify_discovered_tools`. Refuse startup on missing, extra, repeated, or
   schema-different tools. Do not trust server descriptions or annotations.
4. Keep direct telemetry backends in use until deterministic policy approval
   explicitly authorizes MCP-backed dispatch. Discovery must never alter the
   graph registry, policy, or model-visible tool set.

## Validation and Evidence

Run only VM-approved Docker/integration checks. Verify the stdio server emits
JSON-RPC only on stdout, uses stderr for logs, cannot access a second incident,
and refuses unknown or malformed arguments. Record the manifest hash, protocol
revision, server version, image digests, and sanitized test results. Exclude
credentials, tokens, owner emails, prompts, report contents, and raw telemetry.
