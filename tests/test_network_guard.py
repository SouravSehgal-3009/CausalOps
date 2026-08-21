"""Proves the root `conftest.py` guard actually refuses and actually allows,
on both the methods it patches, and that it is active early enough to matter.

A guard nobody has tried to trip is not evidence of anything -- these tests
are the whole reason the guard is trustworthy rather than merely present.
"""

import socket

import pytest

from network_guard import NetworkAccessRefused

# P2-3's regression test starts here, not inside a test function: it proves
# the guard is active during COLLECTION (module import), not only once the
# first test function starts running. Under the pre-fix autouse-fixture
# form, this connect attempt -- which runs the moment this file is
# imported, before any fixture could have activated -- would reach the real
# network and time out instead of being refused. The result is captured
# here, at import time, rather than asserted here, so a broken guard is an
# ordinary failing test below rather than a collection error that aborts
# the whole run.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _collection_time_probe:
    _collection_time_probe.settimeout(1)
    try:
        _collection_time_probe.connect(("192.0.2.1", 80))
    except BaseException as _collection_time_error:
        _COLLECTION_TIME_RESULT: BaseException | None = _collection_time_error
    else:
        _COLLECTION_TIME_RESULT = None


def test_the_guard_is_active_during_collection_not_only_after_the_first_test() -> None:
    """P2-3's regression test, continued: asserts on what the module-level
    probe above captured at import time. Must be `NetworkAccessRefused`
    specifically, not merely "some exception" -- a real `TimeoutError` (the
    pre-fix result) would also make `_COLLECTION_TIME_RESULT` non-`None`,
    so a weaker `is not None` assertion would not actually distinguish a
    working guard from a broken one."""
    assert isinstance(_COLLECTION_TIME_RESULT, NetworkAccessRefused)


def test_a_non_loopback_connect_is_refused() -> None:
    # RFC 5737 TEST-NET-1: guaranteed non-routable, so a broken guard fails
    # this test immediately with a refused connection rather than hanging on
    # a real network timeout.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        with pytest.raises(NetworkAccessRefused):
            sock.connect(("192.0.2.1", 80))


def test_a_loopback_connect_still_succeeds() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(1)
            client.connect(("127.0.0.1", port))  # must not raise


def test_a_non_loopback_connect_ex_is_refused_too() -> None:
    """`connect_ex`'s normal contract is to return an errno rather than
    raise -- ordinary socket-error handling expects an `int`, not an
    exception, and would silently absorb a plain `OSError` here. The guard
    raises `NetworkAccessRefused` from `connect_ex` anyway, the same
    "wrong base class swallows the signal" concern as `NetworkAccessRefused`
    subclassing `BaseException` rather than `Exception` (see its docstring
    in `network_guard.py`), reached through a different mechanism -- so this
    is a second, independent finding, not a restatement of that one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        with pytest.raises(NetworkAccessRefused):
            sock.connect_ex(("192.0.2.1", 80))
