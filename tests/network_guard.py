"""The loopback-only network guard's mechanics, shared by `tests/conftest.py`
(which activates it for the whole suite) and `tests/test_network_guard.py`
(which proves it). Split into its own module, not defined directly in either
importer, because a bare `conftest.py` module name is not unique in this
tree: `tests/unit/conftest.py` already claims it, and `pyproject.toml`'s
`pythonpath = [..., "tests/unit"]` makes that second `conftest` importable
from anywhere. Two files racing for one `sys.modules["conftest"]` slot broke
`pytest tests/unit tests/test_network_guard.py` (and any other invocation
that collects both directories) with an `ImportError` that aborted the whole
run, not just one file. `network_guard` has no such collision.
"""

import ipaddress
import socket
from collections.abc import Callable
from typing import Any


class NetworkAccessRefused(BaseException):
    """A test (or, from Unit 3b-2 on, a live model adapter) tried to reach a
    destination outside loopback.

    Subclasses `BaseException`, not `Exception` -- deliberately, matching
    pytest's own `pytest.fail()`/`_pytest.outcomes.OutcomeException`, which
    does the same thing for the same reason: so a broad `except Exception:`
    in code under test cannot silently swallow a test-harness control
    signal. Every node this project runs a model or tool call from
    (`graph.py`'s `investigate`/`dispatch_tool`/`final_assessment`) wraps its
    body in exactly that kind of blanket `except Exception:`, the safety net
    that turns a live model's crash into a safe `FAILED_SAFE` report instead
    of an unhandled exception. If this were an ordinary `Exception`, a guard
    violation raised from inside `model.propose()`/`.respond()` would be
    caught by that same net and reported as an indistinguishable
    `INTERNAL_ERROR` -- the guard would still have blocked the connection,
    but nothing would say so, in exactly the run where it matters most.
    **Do not change this back to `Exception`; that silently defeats the
    guard in the one case it exists for.**

    Verified, not assumed, that this propagates out of a graph run: the
    installed `langgraph.pregel.main`'s `.invoke()`/`.stream()` path wraps
    its loop in `except BaseException as e: run_manager.on_chain_error(e);
    raise` -- it notifies and re-raises, it does not swallow. This is a
    caveat, not a guarantee, for one reason: `langgraph.pregel._executor.py`
    separately has `except BaseException: pass` inside `Submit.done()`, the
    callback for the framework's concurrent-task-execution path. CausalOps's
    graph never runs two nodes in the same superstep today, so that path is
    unreached -- but a future fan-out node would need this re-verified
    before relying on it.

    The same reasoning extends to `network_guard.py`'s `connect_ex` patch:
    `connect_ex`'s normal contract is to return an errno rather than raise,
    so ordinary socket-error handling expects an `int`, not an exception, and
    would absorb a plain `OSError` without comment. Raising `NetworkAccessRefused`
    here instead -- and its not being an `OSError` -- is the same "wrong base
    class swallows the signal" problem as the `Exception` question above, just
    reached through a different mechanism (return-value handling instead of an
    exception hierarchy) rather than the same finding twice.
    """


def _is_loopback(address: object) -> bool:
    if not isinstance(address, tuple) or not address or not isinstance(address[0], str):
        return False
    host = address[0]
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guard(original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(self: socket.socket, address: object, *a: object, **kw: object) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(
            address
        ):
            raise NetworkAccessRefused(f"refused a connection to {address!r}")
        return original(self, address, *a, **kw)

    return wrapped
