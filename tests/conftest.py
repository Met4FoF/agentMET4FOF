# These lines are shared for all tests. They contain the basic fixtures needed for
# several of our test.
import numpy as np
import pytest

from agentMET4FOF.agents import AgentNetwork

# Set time to wait for before agents should have done their jobs in networks.
test_timeout = 10

# Set random seed to achieve reproducibility
np.random.seed(123)


@pytest.fixture(scope="function")
def agent_network():
    # Create an agent network and shut it down after usage.
    a_network = AgentNetwork(dashboard_modules=False)
    yield a_network
    a_network.shutdown()
