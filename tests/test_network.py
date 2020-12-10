import pytest

from agentMET4FOF.agents import AgentNetwork
from tests.conftest import test_timeout


@pytest.mark.timeout(test_timeout)
@pytest.mark.last
@pytest.mark.parametrize("backend", ["osbrain", "mesa"])
def test_shutdown(backend):
    # Check if the agent network gets properly setup and stopped again. The setup and
    # tear down are actually done inside the fixture agent_network() in conftest.py.
    network = AgentNetwork(backend=backend)
    network.shutdown()
