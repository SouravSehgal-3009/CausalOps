"""Administrative: upsert `runbook_corpus.json`'s passages into the
provisioned Pinecone index.

Not part of any hot path -- `search_runbooks`'s Pinecone backend
(`pinecone_runbooks.py`) only ever searches, never writes. Run this again
whenever `runbook_corpus.json` changes and the Pinecone arm needs to see
the update. Each record's `_id` is the passage's own `passage_id`, so a
rerun overwrites existing records in place rather than duplicating them --
safe to run repeatedly.

The index was provisioned once, out of band, with an embedding model
already bound to its `content` field (`multilingual-e5-large` hosted
inference, `pc.create_index_for_model(..., embed={"field_map":
{"text": "content"}})`) -- this script only ever calls `upsert_records`,
never `create_index`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from pinecone import Pinecone

from causalops.pinecone_runbooks import (
    PINECONE_API_KEY_VARIABLE,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)
from causalops.runbooks import CORPUS_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalops-pinecone-reindex",
        description=("Upsert the checked-in runbook corpus into the Pinecone index."),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help="Corpus JSON path (defaults to the checked-in runbook_corpus.json).",
    )
    return parser


def _records(corpus_path: Path) -> list[dict[str, str]]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [
        {
            "_id": passage["passage_id"],
            "content": passage["content"],
            "source_version": passage["source_version"],
        }
        for passage in corpus["passages"]
    ]


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    api_key = os.environ.get(PINECONE_API_KEY_VARIABLE, "").strip()
    if not api_key:
        print(f"{PINECONE_API_KEY_VARIABLE} is not set", file=sys.stderr)
        return 1

    records = _records(arguments.corpus)

    pc = Pinecone(api_key=api_key)
    host = pc.describe_index(PINECONE_INDEX_NAME)["host"]
    index = pc.Index(host=host)
    index.upsert_records(namespace=PINECONE_NAMESPACE, records=records)

    destination = f"{PINECONE_INDEX_NAME}/{PINECONE_NAMESPACE}"
    print(f"upserted {len(records)} records into {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
