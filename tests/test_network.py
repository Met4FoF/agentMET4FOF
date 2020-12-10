import pytest

from tests.conftest import test_timeout


@pytest.mark.timeout(test_timeout)
@pytest.mark.parametrize('agent_network', ["mesa", "osbrain"], indirect=['agent_network'])
def test_shutdown(agent_network):
    # Check if the agent network gets properly setup and stopped again. The setup and
    # tear down are actually done inside the fixture agent_network() in conftest.py.
    assert agent_network
