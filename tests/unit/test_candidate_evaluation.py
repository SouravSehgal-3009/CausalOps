"""Pure, no-provider checks for candidate-evaluation assessment."""

from pathlib import Path

import pytest

from causalops.candidate_evaluation import (
    FROZEN_CASE_IDS,
    CandidateEvaluationRecord,
    RetrievalComparisonRecord,
    summarize_candidate_evaluation,
    summarize_retrieval_comparison,
)
from causalops.candidate_evaluation_cli import main
from causalops.domain import Budgets, Disposition, RetrievalMode, Versions
from causalops.evaluation import ControlCounts, Efficiency, MechanicalScores

_DIGEST = "sha256:" + "a" * 64
_SHA = "a" * 40
_FIXTURE_SHA = "b" * 64
_VERSIONS = Versions(prompt_version="8", policy_version="4", tool_registry_version="8")


def _record(
    case_id: str,
    *,
    mode: RetrievalMode = RetrievalMode.FTS5_LEXICAL,
    correct: bool = True,
    grounded: bool = True,
    citations_valid: bool = True,
    policy_escape: bool = False,
    disposition: Disposition = Disposition.DIAGNOSED,
) -> CandidateEvaluationRecord:
    return CandidateEvaluationRecord(
        case_id=case_id,
        investigation_id=f"investigation-{case_id}",
        model_name="qwen3.5:4b",
        ollama_image_digest=_DIGEST,
        model_manifest_digest=_DIGEST,
        git_sha=_SHA,
        git_dirty=False,
        versions=_VERSIONS,
        fixture_sha256=_FIXTURE_SHA,
        runbook_corpus_version="1",
        retrieval_index_identity="fts5:1",
        configured_budgets=Budgets(),
        runbook_query_count=1,
        retrieval_mode=mode,
        scores=MechanicalScores(
            diagnosis_correct=correct,
            disposition_correct=correct,
            citations_valid=citations_valid,
            citations_sufficient=grounded,
            correct_and_grounded=grounded,
            control=ControlCounts(),
            efficiency=Efficiency(latency_ms=1, model_calls=1, tools_executed=1),
        ),
        sequence_valid=True,
        policy_escape=policy_escape,
        disposition=disposition,
        wall_clock_ms=1,
        local_compute_cost_usd=0,
    )


def test_candidate_experiment_gate_requires_the_complete_safe_fixed_corpus() -> None:
    records = [_record(case_id) for case_id in sorted(FROZEN_CASE_IDS)]

    summary = summarize_candidate_evaluation(records)

    assert summary.experiment_eligible is True
    assert summary.promotion_eligible is True

    unsafe = summarize_candidate_evaluation(
        [*records[:-1], _record(records[-1].case_id, policy_escape=True)]
    )
    assert unsafe.experiment_eligible is False
    assert unsafe.promotion_eligible is False


def test_candidate_promotion_has_stricter_correct_and_grounded_requirement() -> None:
    records = [
        _record(case_id, grounded=number < 4)
        for number, case_id in enumerate(sorted(FROZEN_CASE_IDS))
    ]

    summary = summarize_candidate_evaluation(records)

    assert summary.experiment_eligible is True
    assert summary.promotion_eligible is False


def test_candidate_qualification_rejects_twelve_substituted_cases() -> None:
    summary = summarize_candidate_evaluation(
        [_record(f"substituted-{number}") for number in range(12)]
    )

    assert summary.experiment_eligible is False
    assert summary.promotion_eligible is False


def test_retrieval_comparison_requires_matched_modes_and_noninferiority() -> None:
    pairs = [
        RetrievalComparisonRecord(
            case_id=case_id,
            fts5=_record(case_id, grounded=number < 5),
            pinecone=_record(
                case_id,
                mode=RetrievalMode.PINECONE_SEMANTIC,
                grounded=number < 5,
            ),
        )
        for number, case_id in enumerate(sorted(FROZEN_CASE_IDS))
    ]

    summary = summarize_retrieval_comparison(pairs)

    assert summary.pinecone_used_count == 12
    assert summary.pinecone_selectable is True

    worse_pairs = [
        pair.model_copy(
            update={
                "pinecone": pair.pinecone.model_copy(
                    update={
                        "scores": pair.pinecone.scores.model_copy(
                            update={"correct_and_grounded": False}
                        )
                    }
                )
            }
        )
        for pair in pairs
    ]
    assert summarize_retrieval_comparison(worse_pairs).pinecone_selectable is False


def test_retrieval_comparison_rejects_different_model_identity() -> None:
    case_id = next(iter(FROZEN_CASE_IDS))
    with pytest.raises(ValueError, match="preregistered conditions"):
        RetrievalComparisonRecord(
            case_id=case_id,
            fts5=_record(case_id),
            pinecone=_record(
                case_id,
                mode=RetrievalMode.PINECONE_SEMANTIC,
            ).model_copy(update={"model_manifest_digest": "sha256:" + "c" * 64}),
        )


def test_candidate_record_rejects_any_model_other_than_approved_qwen() -> None:
    case_id = next(iter(FROZEN_CASE_IDS))
    with pytest.raises(ValueError, match="qwen3.5:4b"):
        CandidateEvaluationRecord.model_validate(
            {**_record(case_id).model_dump(), "model_name": "other"}
        )


def test_candidate_record_preserves_unmeasured_local_compute_cost() -> None:
    record = _record(next(iter(FROZEN_CASE_IDS))).model_copy(
        update={"local_compute_cost_usd": None}
    )

    assert record.local_compute_cost_usd is None


def test_retrieval_comparison_represents_a_semantic_arm_the_model_did_not_use() -> None:
    case_id = next(iter(FROZEN_CASE_IDS))
    pair = RetrievalComparisonRecord(
        case_id=case_id,
        fts5=_record(case_id),
        pinecone=_record(case_id).model_copy(
            update={
                "retrieval_mode": RetrievalMode.DISABLED,
                "runbook_query_count": 0,
            }
        ),
    )

    assert summarize_retrieval_comparison([pair]).pinecone_used_count == 0


def test_retrieval_comparison_rejects_a_no_query_semantic_claim() -> None:
    case_id = next(iter(FROZEN_CASE_IDS))
    with pytest.raises(ValueError, match="unused pinecone arm"):
        RetrievalComparisonRecord(
            case_id=case_id,
            fts5=_record(case_id),
            pinecone=_record(case_id, mode=RetrievalMode.PINECONE_SEMANTIC).model_copy(
                update={"runbook_query_count": 0}
            ),
        )


def test_offline_assessor_reads_sanitized_candidate_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "records.jsonl"
    artifact.write_text(
        "\n".join(_record(case_id).model_dump_json() for case_id in FROZEN_CASE_IDS)
        + "\n",
        encoding="utf-8",
    )

    assert main([str(artifact)]) == 0

    assert '"experiment_eligible": true' in capsys.readouterr().out
