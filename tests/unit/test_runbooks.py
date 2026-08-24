"""`search_runbooks`'s backend: the FTS5 index and the checked-in corpus.

`TECHNICAL_SPEC.md` §7 requires the corpus to exclude expected answers,
evaluator-only predicates, semantic scenario keys, secrets, and
controller-only instructions. The isolation tests below check that directly
against `runbook_corpus.json`'s actual content, not by convention alone --
`tests/security/test_ground_truth_isolation.py` makes the equivalent claim
about rendered context, once a passage is folded in.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from causalops.domain import RetrievalMode, RootCauseCode, ToolOutcome
from causalops.runbooks import CORPUS_PATH, RunbookIndex, run_runbook_search
from causalops.tools import RunbookTopic, SearchRunbooksArguments

FORBIDDEN_WORDS = ("seed", "scenario", "family", "expected", "predicate")


def loaded_corpus() -> dict[str, object]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def test_the_corpus_names_no_root_cause_code() -> None:
    # Checked over the whole passage blob, not just `content` -- a leaked
    # `RootCauseCode` in `passage_id` or `source_version` would be just as
    # real a leak as one in `content`, and checking only the field this
    # unit happened to write content into would miss it in any other field,
    # present or future. `json.dumps` keeps the exact case `RootCauseCode`
    # values use (`"CONFIG_CHANGE"`, all caps).
    corpus = loaded_corpus()
    passages = corpus["passages"]
    assert isinstance(passages, list)
    for passage in passages:
        assert isinstance(passage, dict)
        blob = json.dumps(passage)
        for code in RootCauseCode:
            assert code.value not in blob, (passage["passage_id"], code.value)


def test_the_corpus_names_no_evaluator_or_scenario_vocabulary() -> None:
    # Same whole-blob scope as the test above, case-insensitive to match
    # how the rendered-context leakage test already checks evidence.
    corpus = loaded_corpus()
    passages = corpus["passages"]
    assert isinstance(passages, list)
    for passage in passages:
        assert isinstance(passage, dict)
        blob = json.dumps(passage).lower()
        for word in FORBIDDEN_WORDS:
            assert word not in blob, (passage["passage_id"], word)


def test_every_passage_fits_the_content_length_bound() -> None:
    # `RunbookPassage.content` is `Field(max_length=800)` -- a corpus entry
    # over that bound would fail the very first `RunbookIndex()` search that
    # returned it, not at load time, since the index stores raw text and only
    # constructs `RunbookPassage` in `search()`. Checked here directly so a
    # future corpus edit fails this test instead of a random search call.
    corpus = loaded_corpus()
    passages = corpus["passages"]
    assert isinstance(passages, list)
    for passage in passages:
        assert isinstance(passage, dict)
        content = passage["content"]
        assert isinstance(content, str)
        assert len(content) <= 800, passage["passage_id"]


def test_every_passage_id_is_unique() -> None:
    corpus = loaded_corpus()
    passages = corpus["passages"]
    assert isinstance(passages, list)
    ids = [passage["passage_id"] for passage in passages]
    assert len(ids) == len(set(ids))


def test_search_returns_the_on_topic_passage_first() -> None:
    index = RunbookIndex()
    results = index.search(RunbookTopic.DOWNSTREAM_TIMEOUTS, limit=5)

    assert results
    assert "downstream-timeouts" in results[0].passage_id
    assert results[0].retrieval_mode is RetrievalMode.FTS5_LEXICAL
    assert results[0].content_hash


def test_search_respects_the_limit() -> None:
    index = RunbookIndex()
    results = index.search(RunbookTopic.GATEWAY_ERRORS, limit=1)

    assert len(results) == 1


def test_search_orders_by_score_descending() -> None:
    index = RunbookIndex()
    results = index.search(RunbookTopic.RESOURCE_POOL_PRESSURE, limit=5)

    scores = [passage.score for passage in results]
    assert scores == sorted(scores, reverse=True)


def test_a_missing_corpus_file_fails_loudly_at_construction() -> None:
    # Corpus loading happens at `RunbookIndex.__init__`, not per search --
    # see the class's own docstring for why a broken checked-in corpus is a
    # packaging defect, not a per-call condition to degrade from.
    with pytest.raises(FileNotFoundError):
        RunbookIndex(corpus_path=Path("/nonexistent/runbook_corpus.json"))


def test_the_checked_in_corpus_version_is_read_and_stringified() -> None:
    """Unit 3c. `runbook_corpus.json`'s own `corpus_version` key is `1`, an
    int -- stringified to match every other "_version" field in this
    codebase (`SCHEMA_VERSION`, `prompt_version`, ...), all of which are
    strings regardless of how small the underlying number is."""
    index = RunbookIndex()

    assert index.corpus_version == "1"


def test_a_corpus_file_with_no_version_key_reports_none(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "passages": [
                    {
                        "passage_id": "unrelated-1",
                        "source_version": "test",
                        "content": "xyzzy plugh unrelated filler text",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    index = RunbookIndex(corpus_path=corpus_path)

    assert index.corpus_version is None


def test_run_runbook_search_executes_and_stamps_the_retrieval_mode() -> None:
    index = RunbookIndex()
    arguments = SearchRunbooksArguments(topic=RunbookTopic.GATEWAY_LATENCY, limit=3)

    outcome = run_runbook_search(arguments, index)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.retrieval_mode is RetrievalMode.FTS5_LEXICAL
    assert outcome.passages


def test_run_runbook_search_stamps_the_mode_even_when_nothing_is_found(
    tmp_path: Path,
) -> None:
    # A corpus whose only passage cannot match any topic's FTS5 query --
    # `search()` still ran in `fts5_lexical` mode, it just found nothing.
    # `RetrievalMode`'s own docstring is the reason this must not come back
    # `disabled`: `graph.py`'s `dispatch_tool` reads exactly this field to
    # decide `retrieval_mode`, never the emptiness of `passages`.
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "corpus_version": "test",
                "passages": [
                    {
                        "passage_id": "unrelated-1",
                        "source_version": "test",
                        "content": "xyzzy plugh unrelated filler text",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    index = RunbookIndex(corpus_path=corpus_path)
    arguments = SearchRunbooksArguments(topic=RunbookTopic.GATEWAY_ERRORS, limit=3)

    outcome = run_runbook_search(arguments, index)

    assert outcome.outcome is ToolOutcome.EXECUTED
    assert outcome.retrieval_mode is RetrievalMode.FTS5_LEXICAL
    assert outcome.passages == ()


def test_run_runbook_search_turns_a_live_query_error_into_unavailable() -> None:
    # The narrower, genuinely per-call failure mode `run_runbook_search`
    # itself guards (see its own docstring): the index built successfully,
    # but the specific query call raises. Closing the connection reproduces
    # exactly that -- construction already succeeded, only `search()` fails.
    index = RunbookIndex()
    index._connection.close()  # noqa: SLF001 -- deliberately breaking the live query
    arguments = SearchRunbooksArguments(topic=RunbookTopic.GATEWAY_ERRORS, limit=3)

    outcome = run_runbook_search(arguments, index)

    assert outcome.outcome is ToolOutcome.UNAVAILABLE
    assert outcome.passages == ()
    assert outcome.retrieval_mode is RetrievalMode.FTS5_LEXICAL


def test_search_raising_a_non_sqlite_error_is_not_swallowed() -> None:
    # `run_runbook_search` only catches `sqlite3.Error` -- anything else
    # (a real bug) must still propagate to `tool_wrappers.py`'s deliberately
    # uncaught backend call, not be silently absorbed into `UNAVAILABLE`.
    index = RunbookIndex()

    def broken_search(topic: RunbookTopic, limit: int) -> tuple[()]:
        raise ValueError("not a sqlite3.Error")

    index.search = broken_search  # type: ignore[method-assign]
    arguments = SearchRunbooksArguments(topic=RunbookTopic.GATEWAY_ERRORS, limit=3)

    with pytest.raises(ValueError, match="not a sqlite3.Error"):
        run_runbook_search(arguments, index)


def test_a_corrupt_corpus_file_fails_loudly_at_construction(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text("not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        RunbookIndex(corpus_path=corpus_path)


def test_sqlite3_error_is_the_real_exception_class_search_can_raise() -> None:
    # Confirms the exception `run_runbook_search` catches is the one a real
    # FTS5 query failure actually raises, not an assumption -- mirrors the
    # project's standing rule to verify a caught exception class against the
    # real library rather than trust a docstring's claim about it.
    connection = sqlite3.connect(":memory:")
    with pytest.raises(sqlite3.Error):
        connection.execute("SELECT * FROM nonexistent_table")
