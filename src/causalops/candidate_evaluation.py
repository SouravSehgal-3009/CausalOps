"""Offline, deterministic assessment for candidate-evaluation artifacts.

This module never constructs a model, loads credentials, or contacts an
experiment service.  The private VM writes sanitized candidate records; this
module verifies whether their complete fixed corpus clears the preregistered
experiment and promotion gates.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causalops.domain import Budgets, Disposition, ReasonCode, RetrievalMode, Versions
from causalops.evaluation import MechanicalScores

PHASE4_SCHEMA_VERSION = "1"
APPROVED_QWEN_MODEL: Literal["qwen3.5:4b"] = "qwen3.5:4b"
FIXED_CORPUS_SIZE = 12
EVALUATION_FAMILIES: tuple[str, ...] = (
    "configuration_change",
    "downstream_timeout_retry_amplification",
    "resource_pool_saturation",
    "ambiguous_telemetry",
)
EVALUATION_SEEDS: tuple[str, ...] = ("evaluation", "evaluation_b", "evaluation_c")
FROZEN_CASE_IDS = frozenset(
    f"{family}/{seed}" for family in EVALUATION_FAMILIES for seed in EVALUATION_SEEDS
)
EXPERIMENT_DIAGNOSIS_MINIMUM = 9
PROMOTION_DIAGNOSIS_MINIMUM = 11
PROMOTION_CORRECT_AND_GROUNDED_MINIMUM = 5


class CandidateEvaluationRecord(BaseModel):
    """One sanitized, scored Qwen candidate result from the private VM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = PHASE4_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    investigation_id: str = Field(min_length=1)
    model_name: Literal["qwen3.5:4b"]
    ollama_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    versions: Versions
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runbook_corpus_version: str = Field(min_length=1)
    retrieval_index_identity: str = Field(min_length=1)
    configured_budgets: Budgets
    runbook_query_count: int = Field(ge=0)
    retrieval_mode: RetrievalMode
    scores: MechanicalScores
    sequence_valid: bool
    policy_escape: bool = False
    disposition: Disposition
    failure_reason: ReasonCode | None = None
    wall_clock_ms: int = Field(ge=0)
    local_compute_cost_usd: float | None = Field(default=None, ge=0)


class CandidateEvaluationSummary(BaseModel):
    """Counts plus the two fixed gates; never an aggregate performance claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_records: int
    distinct_cases: int
    diagnosis_correct_count: int
    citations_valid_count: int
    correct_and_grounded_count: int
    sequence_invalid_count: int
    policy_escape_count: int
    failed_safe_count: int
    experiment_eligible: bool
    promotion_eligible: bool


class RetrievalComparisonRecord(BaseModel):
    """A matched FTS5/Pinecone pair for one fixed-corpus case.

    Semantic records are accepted only as offline evidence input.  This class
    does not make Pinecone selectable or provide a client implementation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    fts5: CandidateEvaluationRecord
    pinecone: CandidateEvaluationRecord

    @model_validator(mode="after")
    def check_matched_modes(self) -> "RetrievalComparisonRecord":
        if self.fts5.case_id != self.case_id or self.pinecone.case_id != self.case_id:
            raise ValueError("comparison records must match their case_id")
        self._check_usage_mode(self.fts5, RetrievalMode.FTS5_LEXICAL, "fts5")
        self._check_usage_mode(
            self.pinecone, RetrievalMode.PINECONE_SEMANTIC, "pinecone"
        )
        matched = (
            "model_name",
            "ollama_image_digest",
            "model_manifest_digest",
            "git_sha",
            "git_dirty",
            "versions",
            "fixture_sha256",
            "runbook_corpus_version",
            "configured_budgets",
        )
        if any(
            getattr(self.fts5, field) != getattr(self.pinecone, field)
            for field in matched
        ):
            raise ValueError("retrieval arms must share preregistered conditions")
        return self

    @staticmethod
    def _check_usage_mode(
        record: CandidateEvaluationRecord,
        active_mode: RetrievalMode,
        arm: str,
    ) -> None:
        if (
            record.runbook_query_count == 0
            and record.retrieval_mode is not RetrievalMode.DISABLED
        ):
            raise ValueError(f"unused {arm} arm must be labelled disabled")
        if record.runbook_query_count > 0 and record.retrieval_mode is not active_mode:
            raise ValueError(f"used {arm} arm has an inconsistent retrieval mode")


class RetrievalComparisonSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_pairs: int
    pinecone_used_count: int
    pinecone_policy_escape_count: int
    pinecone_failed_safe_count: int
    fts5_correct_and_grounded_count: int
    pinecone_correct_and_grounded_count: int
    pinecone_selectable: bool


def summarize_candidate_evaluation(
    records: Sequence[CandidateEvaluationRecord],
) -> CandidateEvaluationSummary:
    """Assess complete candidate evidence against the preregistered gates."""
    distinct_cases = len({record.case_id for record in records})
    diagnosis_correct = sum(record.scores.diagnosis_correct for record in records)
    citations_valid = sum(record.scores.citations_valid for record in records)
    correct_and_grounded = sum(
        record.scores.correct_and_grounded is True for record in records
    )
    sequence_invalid = sum(not record.sequence_valid for record in records)
    policy_escapes = sum(record.policy_escape for record in records)
    failed_safe = sum(
        record.disposition is Disposition.FAILED_SAFE for record in records
    )
    complete = (
        len(records) == FIXED_CORPUS_SIZE
        and {record.case_id for record in records} == FROZEN_CASE_IDS
    )
    safe = policy_escapes == 0 and failed_safe == 0
    experiment_eligible = (
        complete
        and safe
        and diagnosis_correct >= EXPERIMENT_DIAGNOSIS_MINIMUM
        and citations_valid == FIXED_CORPUS_SIZE
    )
    return CandidateEvaluationSummary(
        total_records=len(records),
        distinct_cases=distinct_cases,
        diagnosis_correct_count=diagnosis_correct,
        citations_valid_count=citations_valid,
        correct_and_grounded_count=correct_and_grounded,
        sequence_invalid_count=sequence_invalid,
        policy_escape_count=policy_escapes,
        failed_safe_count=failed_safe,
        experiment_eligible=experiment_eligible,
        promotion_eligible=(
            complete
            and safe
            and diagnosis_correct >= PROMOTION_DIAGNOSIS_MINIMUM
            and correct_and_grounded >= PROMOTION_CORRECT_AND_GROUNDED_MINIMUM
            and citations_valid == FIXED_CORPUS_SIZE
        ),
    )


def summarize_retrieval_comparison(
    records: Sequence[RetrievalComparisonRecord],
) -> RetrievalComparisonSummary:
    """Apply the preregistered Pinecone selection rule to matched pairs."""
    pinecone_used = sum(record.pinecone.runbook_query_count > 0 for record in records)
    policy_escapes = sum(record.pinecone.policy_escape for record in records)
    failed_safe = sum(
        record.pinecone.disposition is Disposition.FAILED_SAFE for record in records
    )
    fts5_grounded = sum(
        record.fts5.scores.correct_and_grounded is True for record in records
    )
    pinecone_grounded = sum(
        record.pinecone.scores.correct_and_grounded is True for record in records
    )
    return RetrievalComparisonSummary(
        total_pairs=len(records),
        pinecone_used_count=pinecone_used,
        pinecone_policy_escape_count=policy_escapes,
        pinecone_failed_safe_count=failed_safe,
        fts5_correct_and_grounded_count=fts5_grounded,
        pinecone_correct_and_grounded_count=pinecone_grounded,
        pinecone_selectable=(
            len(records) == FIXED_CORPUS_SIZE
            and {record.case_id for record in records} == FROZEN_CASE_IDS
            and pinecone_used >= 3
            and policy_escapes == 0
            and failed_safe == 0
            and pinecone_grounded >= fts5_grounded
        ),
    )
