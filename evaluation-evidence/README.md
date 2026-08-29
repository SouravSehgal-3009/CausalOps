# Evaluation evidence

These two files are exact copies of the per-run scored records
(`records.jsonl`) behind the "The v8 validation run" scorecard in the main
`README.md`. Every field is structured JSON — investigation and incident
IDs, the expected root cause/disposition/required-evidence predicates, the
diagnosis/citation/efficiency scores, git SHA and version pins, and the
real reserved/actual cost per run. There is no model-generated prose here;
the raw investigation reports (which do contain the model's synthetic
reasoning text) stay out of version control, per this project's own
`results/` gitignore rule.

- `v8-et3-records.jsonl` — the `executed_tools`=3 batch, originally saved
  at the opaque artifact path
  `results/evaluations/3fb667d0993742f787035a11142d94e0/`.
- `v8-et4-records.jsonl` — the `executed_tools`=4 batch, originally saved
  at the opaque artifact path
  `results/evaluations/59ec53e1a98542ad9753da0ceb9dec49/`.

Each line is one `EvaluationRecord`. A reader can independently recompute
the main README's scorecard numbers (diagnosis-correct, correct-and-
grounded, citations-valid, `FAILED_SAFE` counts) and cross-check the
`git_sha`/`versions`/cost fields against the claims made there.
