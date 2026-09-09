# Phase 4 Private-VM Handoff

## Preconditions

Run this only on the approved private VM after this branch has been reviewed
and merged/deployed. Do not run it from a work laptop. Phase 3 worker/MCP
composition remains deferred and fail-closed; this candidate runner uses the
existing policy-wrapped direct backends and does not enable MCP dispatch.

Keep `ENABLE_CLAUDE=false` and leave `RAG_EXPERIMENT_ENABLED=false`. Hosted
API/dashboard paths remain replay-only.

## Candidate evaluation

1. Confirm the VM's reviewed Ollama image and Qwen manifest digests. Export:
   ```sh
   export CAUSALOPS_EXECUTION_ENV=vm
   export CAUSALOPS_CANDIDATE_EVALUATION=true
   export CAUSALOPS_OLLAMA_IMAGE_DIGEST=sha256:<reviewed-image-digest>
   export CAUSALOPS_QWEN_MANIFEST_DIGEST=sha256:<reviewed-model-digest>
   ```
2. Ensure the approved synthetic lab is already running under its separate VM
   procedure. Do not start it from `infra/phase3/docker-compose.yml`.
3. Run `causalops-qwen-evaluate`. It creates a separate candidate artifact
   directory and stops after a policy escape or `FAILED_SAFE` result.
4. Inspect the sanitized `records.jsonl` without contacting a provider:
   ```sh
   causalops-phase4-assess results/candidate-evaluations/<id>/records.jsonl
   ```

Only the exact frozen 12-case corpus, `qwen3.5:4b`, valid citations, and zero
policy escapes/`FAILED_SAFE` can qualify. Local compute cost remains
`null` until a reviewed VM/Ollama allocation method exists; never substitute
zero.

## Retrieval experiments

Do not enable Pinecone yet. A reviewed VM-only adapter, credentials procedure,
and matched FTS5/Pinecone artifact producer are prerequisites. Once approved,
record matching provenance/budgets/versions and actual executed retrieval use;
assess with `causalops-phase4-assess --retrieval-comparison <records.jsonl>`.
Pinecone is selectable only under the preregistered criteria in
`PREREGISTRATION.md`.

## Evidence handling

Keep candidate artifacts separate from README claims and omit credentials,
owner identifiers, prompts, raw telemetry, and report contents from any
validation record. Record only sanitized identifiers, manifest hashes, versions,
counts, latency, and measured cost where available.
