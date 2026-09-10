"""Inspect sanitized candidate-evaluation artifacts without provider access."""

import argparse
from pathlib import Path

from pydantic import ValidationError

from causalops.candidate_evaluation import (
    CandidateEvaluationRecord,
    RetrievalComparisonRecord,
    summarize_candidate_evaluation,
    summarize_retrieval_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalops-candidate-assess",
        description="Assess sanitized candidate-evaluation JSONL artifacts.",
    )
    parser.add_argument("records", type=Path, help="A sanitized JSONL artifact.")
    parser.add_argument(
        "--retrieval-comparison",
        action="store_true",
        help="Parse matched FTS5/Pinecone comparison records instead of Qwen records.",
    )
    return parser


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        lines = _lines(arguments.records)
        if arguments.retrieval_comparison:
            comparison_records = [
                RetrievalComparisonRecord.model_validate_json(line) for line in lines
            ]
            rendered = summarize_retrieval_comparison(
                comparison_records
            ).model_dump_json(indent=2)
        else:
            candidate_records = [
                CandidateEvaluationRecord.model_validate_json(line) for line in lines
            ]
            rendered = summarize_candidate_evaluation(
                candidate_records
            ).model_dump_json(indent=2)
    except (OSError, ValidationError) as error:
        print(f"FAIL INVALID_CANDIDATE_ARTIFACT {error}")
        return 1
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - package entry point calls main
    raise SystemExit(main())
