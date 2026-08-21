"""Root conftest: activates the loopback-only network guard for the whole
test suite, including collection.

Nothing in this project has ever needed a real network guard before, because
nothing in `src/` could reach one -- `prometheus.py` and `scenario_control.py`
both talk `urllib.request` to `DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"`
(`prometheus.py:29`), and `tests/unit/conftest.py`'s `fake_prometheus` binds a
real `ThreadingHTTPServer` on `127.0.0.1`. That structural guarantee ends the
moment a live model adapter exists (Unit 3b-2's `api.anthropic.com`), so this
guard ships now, before any code in this project can dial out, rather than
being bolted on after the first thing that can.

The guard mechanics (`NetworkAccessRefused`, `_is_loopback`, `_guard`) live in
`network_guard.py`, not here -- a bare `conftest.py` module name collides
with `tests/unit/conftest.py` (see that module's docstring for the exact
failure this caused).

Activated with `pytest_configure`/`pytest_unconfigure` hooks, not an autouse
fixture: a fixture only runs once the first test starts, which is after
every collected module has already been imported. A module-level connect
attempt -- ordinary for a client SDK, and exactly the shape a future live
adapter's module import might take -- would run unguarded during collection.
Confirmed empirically: a module-level `socket.connect()` to a non-loopback
address, placed in a throwaway test file and run standalone, produced a real
`TimeoutError` under the fixture instead of `NetworkAccessRefused`, because
the fixture had not yet activated when that module was imported. The hooks
run before collection begins, so the guard is active for every import this
process ever does, not just for every test function.

The guard patches `socket.socket.connect`/`connect_ex` rather than `httpx`'s or
`requests`' send methods, the narrower per-test mechanism
`tests/security/test_no_tracing.py` uses: the low-level socket method is the
one point every HTTP client in this project's dependency tree eventually calls
through, so it catches `urllib.request` today and any future `httpx`-based
adapter alike, with one guard instead of one per client library. Only
`AF_INET`/`AF_INET6` sockets are intercepted -- Unix-domain and other families
pass through untouched, since nothing in this project uses them and a
same-machine IPC socket is not the network egress this guard is closing.

Named limitations, not fixed here:
- The guard blocks the TCP `connect()`, not a bare hostname resolution -- a
  `getaddrinfo()` call that never reaches `.connect()` could still leak a
  hostname to a DNS resolver. Nothing in the current suite does this; it
  becomes relevant once a live adapter introduces a real hostname to resolve.
- This guard patches `socket.socket` in this process only. A subprocess this
  suite spawns (`tests/security/test_ground_truth_isolation.py:133` does,
  for a different reason) starts with an unpatched `socket` module and is not
  covered -- an inherent limit of an in-process guard, not something this
  patch closes.
"""

import socket

import pytest

from network_guard import _guard

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex


def pytest_configure(config: pytest.Config) -> None:
    socket.socket.connect = _guard(_ORIGINAL_CONNECT)
    socket.socket.connect_ex = _guard(_ORIGINAL_CONNECT_EX)


def pytest_unconfigure(config: pytest.Config) -> None:
    socket.socket.connect = _ORIGINAL_CONNECT
    socket.socket.connect_ex = _ORIGINAL_CONNECT_EX
