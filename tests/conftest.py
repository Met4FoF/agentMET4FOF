# These lines are shared for all tests. They contain the basic fixtures needed for
# several of our test.
import pytest

from agentMET4FOF.agents import AgentNetwork


@pytest.fixture
def agent_network():
    # Create an agent network and shut it down after usage.
    a_network = AgentNetwork(dashboard_modules=False)
    yield a_network
    a_network.shutdown()
