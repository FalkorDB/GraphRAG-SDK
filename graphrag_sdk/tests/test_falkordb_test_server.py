"""The integration tests' server settings refuse port 6379 unless asked.

``RUN_INTEGRATION=1 pytest`` creates and drops graphs, and 6379 is where a
local FalkorDB with real data usually listens. No database is needed here.
"""

from __future__ import annotations

import pytest

from .conftest import falkordb_test_server


@pytest.fixture
def integration_env(monkeypatch):
    """A run with ``RUN_INTEGRATION=1`` and no other server settings."""
    for name in (
        "FALKOR_HOST",
        "FALKOR_PORT",
        "FALKOR_USERNAME",
        "FALKOR_PASSWORD",
        "FALKOR_TEST_ALLOW_6379",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RUN_INTEGRATION", "1")
    return monkeypatch


@pytest.mark.parametrize("port", [None, "6379"], ids=["FALKOR_PORT unset", "FALKOR_PORT=6379"])
def test_port_6379_is_refused(integration_env, port):
    if port is not None:
        integration_env.setenv("FALKOR_PORT", port)

    with pytest.raises(pytest.exit.Exception) as refused:
        falkordb_test_server()

    message = str(refused.value)
    assert "FALKOR_TEST_ALLOW_6379=1" in message
    assert "FALKOR_PORT" in message


@pytest.mark.parametrize("value", ["0", "true", ""])
def test_only_1_opts_in(integration_env, value):
    integration_env.setenv("FALKOR_TEST_ALLOW_6379", value)

    with pytest.raises(pytest.exit.Exception):
        falkordb_test_server()


def test_port_6379_is_allowed_with_the_opt_in(integration_env):
    integration_env.setenv("FALKOR_TEST_ALLOW_6379", "1")

    assert falkordb_test_server()["port"] == 6379


def test_another_port_passes(integration_env):
    integration_env.setenv("FALKOR_HOST", "db.example")
    integration_env.setenv("FALKOR_PORT", "6395")
    integration_env.setenv("FALKOR_USERNAME", "tester")
    integration_env.setenv("FALKOR_PASSWORD", "pw")

    assert falkordb_test_server() == {
        "host": "db.example",
        "port": 6395,
        "username": "tester",
        "password": "pw",
    }


def test_a_port_that_is_not_a_number_is_refused(integration_env):
    integration_env.setenv("FALKOR_PORT", "63g5")

    with pytest.raises(pytest.exit.Exception, match="not a port number"):
        falkordb_test_server()


def test_without_run_integration_nothing_is_refused(integration_env):
    # Nothing connects without RUN_INTEGRATION, so there is nothing to refuse.
    integration_env.delenv("RUN_INTEGRATION")

    assert falkordb_test_server()["port"] == 6379
