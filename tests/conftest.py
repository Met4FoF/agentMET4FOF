# These lines are shared for all tests. They contain the basic fixtures needed for
# several of our test.
import numpy as np
import pytest

from agentMET4FOF.agents import AgentMET4FOF
from agentMET4FOF.agents.network import AgentNetwork

# Set time to wait for before agents should have done their jobs in networks.
test_timeout = 20

# Set random seed to achieve rep(roducibility
np.random.seed(123)


@pytest.fixture(scope="function")
def agent_network():
    # Create an agent network and shut it down after usage.
    a_network = AgentNetwork(dashboard_modules=False)
    yield a_network
    a_network.shutdown()
