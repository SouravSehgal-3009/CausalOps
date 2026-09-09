# Phase 4 Preregistration — Qwen and Retrieval Experiments

## Status and scope

Phase 4 has started as an experiment-design and fail-closed configuration
slice. No candidate evaluation has been run and this document is not evidence
of model or retrieval performance. Phase 3's initial VM deployment is the
only prerequisite currently satisfied; worker and MCP transport remain
deferred and fail-closed as recorded in `../phase3/RESTRICTED_HANDOFF.md`.

Run experiments only on the approved private VM. A work laptop must not start
Ollama, use Pinecone, run Docker or MCP tests, or make network calls.

## Fixed Qwen protocol

Evaluate only `qwen3.5:4b` through the existing VM-only candidate adapter,
with `CAUSALOPS_EXECUTION_ENV=vm` and
`CAUSALOPS_CANDIDATE_EVALUATION=true`. Use the frozen 12-incident corpus:
the four `EVALUATION_FAMILIES` and each of `evaluation`, `evaluation_b`, and
`evaluation_c`. Keep replay as the hosted default and do not add a caller
model parameter to API or dashboard paths.

For every run, retain a separate, sanitized candidate record containing the
existing correctness, grounding, citation, sequence, latency, and cost
fields, plus the model/image/manifest identities and policy result. Stop the
batch on a policy escape or `FAILED_SAFE`; neither may be averaged away.
Qwen is eligible for experimentation only at at least 9/12 diagnosis-correct,
12/12 citation-valid, and zero policy escapes or `FAILED_SAFE`. Promotion is
not considered unless it reaches the roadmap's stricter reference thresholds.

## Fixed retrieval comparison

FTS5 lexical retrieval remains the only wired backend. `RAG_EXPERIMENT_ENABLED`
defaults to `false`; an affirmative value is a pre-I/O refusal until a
reviewed, VM-only semantic adapter exists. No fallback may select Pinecone.

Before any Pinecone run, preregister matched FTS5 and Pinecone arms across
the same corpus, budgets, prompt/policy/tool versions, and model identity.
Record all six outcome dimensions above, retrieval mode, corpus/index
identity, query count, and cost. Select Pinecone only with at least 3/12
actual uses, no safety regression, and non-inferior correct-and-grounded
results. Keep all candidate artifacts out of README headline claims.
