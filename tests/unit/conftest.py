"""The one pytest fixture shared across `tests/unit/`.

`fake_incident.py` carries every other shared test double as a plain
importable name -- `pythonpath` (see `pyproject.toml`) already makes that
work without a `conftest.py`, which is why this project has gone without one
until now. A `pytest.fixture`, unlike a plain function, is discovered by
name matching a test's own parameter, and pytest only looks for that name in
a test's own module, its plugins, and its `conftest.py` chain -- not in an
arbitrary module a test file happens to import. Importing a same-named
fixture by hand worked, but only by also fighting `ruff`: `pyflakes` cannot
tell "imported so pytest can find it" from "imported and never used," so it
flagged both the import (F401) and every test parameter shadowing it (F811),
and stripping those `noqa`s left `ruff check --fix` free to delete the
import and silently break every test using it. A `conftest.py` fixture needs
no import at all, so the false-unused problem does not arise.

`RecordingPrometheus` and `prometheus_body` stay in `fake_incident.py`: they
are ordinary functions/types, not fixtures, so they were never part of this
problem.
"""

import threading
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fake_incident import RecordingPrometheus, prometheus_body


@pytest.fixture
def fake_prometheus() -> Iterator[RecordingPrometheus]:
    """A loopback stand-in for Prometheus; a local server keeps the test
    hermetic."""
    queries: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            received = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            queries.append(received.get("query", [""])[0])
            payload = prometheus_body()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield RecordingPrometheus(f"http://127.0.0.1:{server.server_port}", queries)
    server.shutdown()
    server.server_close()
