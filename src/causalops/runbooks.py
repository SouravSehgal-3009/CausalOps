"""Local retrieval over a small, curated runbook corpus.

`TECHNICAL_SPEC.md` §7's fifth read-only tool, `search_runbooks`. The default
mode is SQLite FTS5, labelled `fts5_lexical` -- it ships in this Python's
stdlib `sqlite3` with a working `bm25()`, so this module needs no new
dependency. Pinecone Starter is a later, opt-in, out-of-scope experiment;
nothing here wires it, and nothing here silently falls back to it.

The corpus lives in `runbook_corpus.json`, beside this module, not as Python
literals in it -- the same convention `replay_fixtures/*.json` and
`lab/scenarios/*.json` already use for curated, checked-in content: a module
holding both retriever logic and corpus text would make one file do two
jobs, code to debug and content to audit. §7 requires the corpus to exclude
expected answers, evaluator-only predicates, semantic scenario keys,
secrets, and controller-only instructions; `tests/unit/test_runbooks.py` and
`tests/security/test_ground_truth_isolation.py` check that directly against
this file's actual content, not by convention alone.

`query` is deliberately not a free-text field anywhere in this module or in
`tools.py`'s `SearchRunbooksArguments`. `RunbookTopic` is a closed enum, and
`_TOPIC_QUERIES` below is the only place a topic becomes an FTS5 `MATCH`
string -- written by this module, never by a model.

The corpus itself carries no per-passage topic tag, and `RunbookIndex`
reads only `passage_id`/`source_version`/`content` from each entry -- an
earlier draft's `topic` key was dropped as dead, misleading data (§7 asks
for a corpus an owner can audit; a field that reads as authoritative
partitioning but is silently never consulted is the opposite of that).
Retrieval genuinely ranks across the whole corpus by lexical relevance, not
by a fixed per-passage bucket: a `gateway_errors` search can and does
surface a `gateway_latency`-authored passage when it scores well, which is
the correct behaviour for lexical retrieval over a small collection, not a
bug a topic tag would have been right to suppress.
"""

import json
import sqlite3
import time
from pathlib import Path

from causalops.domain import (
    ReasonCode,
    RetrievalMode,
    RunbookCheckOutcome,
    RunbookPassage,
    ToolOutcome,
)
from causalops.evidence import digest_text
from causalops.tools import RunbookTopic, SearchRunbooksArguments

CORPUS_PATH = Path(__file__).parent / "runbook_corpus.json"

# Each topic's FTS5 `MATCH` query, assembled with `OR` so a small corpus
# still surfaces a relevant passage even when it does not repeat every
# keyword -- ranking, not filtering, is what narrows the result, the same
# job `bm25()` does for any lexical search over a small collection.
_TOPIC_QUERIES: dict[RunbookTopic, str] = {
    RunbookTopic.GATEWAY_ERRORS: "gateway OR error OR errors",
    RunbookTopic.GATEWAY_LATENCY: "latency OR p95 OR slow OR wait",
    RunbookTopic.DOWNSTREAM_TIMEOUTS: "timeout OR timeouts OR retry OR downstream",
    RunbookTopic.RESOURCE_POOL_PRESSURE: "pool OR queue OR pressure OR slot",
    RunbookTopic.RECENT_CONFIG_CHANGES: "configuration OR rollout OR change OR changed",
}


class RunbookIndex:
    """An in-memory FTS5 index built once from the curated corpus.

    `:memory:` -- the corpus is small, read-only, and shipped with the
    package, so there is nothing to persist across process runs; rebuilding
    it is cheap and means the corpus never needs a database migration path
    of its own, only a JSON file a reviewer can diff.

    Corpus loading happens here, at construction, not inside `search()` or
    `run_runbook_search` below: a malformed or missing checked-in corpus
    file is a packaging defect the project controls completely, not a
    per-call runtime condition to degrade gracefully from -- the same
    posture a corrupt Python module would get. `run_runbook_search` still
    guards the query itself (see its own docstring) for the narrower,
    genuinely per-call failure mode: a live `sqlite3.Error` during a
    specific search.
    """

    def __init__(self, corpus_path: Path = CORPUS_PATH) -> None:
        loaded = json.loads(corpus_path.read_text(encoding="utf-8"))
        # Unit 3c. `corpus_version` is the JSON's own top-level key -- a
        # string in `runbook_corpus.json` today (`"1"`), passed through
        # `str()` here (a no-op on an already-string value) to match every
        # other "_version" field in this codebase
        # (`SCHEMA_VERSION`, `SCORER_VERSION`, `prompt_version`, ...), all
        # of which are strings regardless of how small the underlying
        # number is. `TECHNICAL_SPEC.md` §10's reproducibility manifest is
        # the first real reader, via `EvaluationRecord.runbook_corpus_
        # version`. `None` for a corpus file that predates this key rather
        # than a hard `KeyError`, so an older checked-in corpus still loads.
        raw_corpus_version = loaded.get("corpus_version")
        self.corpus_version: str | None = (
            str(raw_corpus_version) if raw_corpus_version is not None else None
        )
        self._connection = sqlite3.connect(":memory:")
        self._connection.execute(
            "CREATE VIRTUAL TABLE runbook USING fts5("
            "passage_id UNINDEXED, source_version UNINDEXED, content)"
        )
        self._connection.executemany(
            "INSERT INTO runbook (passage_id, source_version, content) "
            "VALUES (:passage_id, :source_version, :content)",
            loaded["passages"],
        )
        self._connection.commit()

    def search(self, topic: RunbookTopic, limit: int) -> tuple[RunbookPassage, ...]:
        """Ranked passages for `topic`, most relevant first. `bm25()` scores
        a better match more negative, so `score` below negates it -- a
        caller (and a rendered context line) should read a higher `score`
        as a better match, not a lower one."""
        rows = self._connection.execute(
            "SELECT passage_id, content, source_version, bm25(runbook) AS rank "
            "FROM runbook WHERE runbook MATCH :query "
            "ORDER BY rank LIMIT :limit",
            {"query": _TOPIC_QUERIES[topic], "limit": limit},
        ).fetchall()
        return tuple(
            RunbookPassage(
                passage_id=passage_id,
                content=content,
                source_version=source_version,
                content_hash=digest_text(content),
                score=-rank,
                retrieval_mode=RetrievalMode.FTS5_LEXICAL,
            )
            for passage_id, content, source_version, rank in rows
        )


def run_runbook_search(
    arguments: SearchRunbooksArguments, index: RunbookIndex
) -> RunbookCheckOutcome:
    """The backend seam `search_runbooks_wrapper` calls, matching
    `run_logs_check`'s own shape (typed arguments in, a domain outcome out,
    nothing about policy or dispatch visible here).

    `index.search(...)` is the one call in this function that can fail at
    request time -- a live `sqlite3.Error`, not the corpus-loading failure
    `RunbookIndex.__init__` already ruled out at construction. Caught here
    and turned into `UNAVAILABLE`, the same `TOOL_UNAVAILABLE` precedent
    `run_logs_check`/`run_changes_check`/`run_topology_check` already use
    for a missing file, rather than left to reach `tool_wrappers.py`'s
    deliberately uncaught backend call.
    """
    started = time.monotonic()
    try:
        passages = index.search(arguments.topic, arguments.limit)
    except sqlite3.Error:
        return RunbookCheckOutcome(
            outcome=ToolOutcome.UNAVAILABLE,
            retrieval_mode=RetrievalMode.FTS5_LEXICAL,
            reason_code=ReasonCode.TOOL_UNAVAILABLE,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    return RunbookCheckOutcome(
        outcome=ToolOutcome.EXECUTED,
        passages=passages,
        retrieval_mode=RetrievalMode.FTS5_LEXICAL,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
