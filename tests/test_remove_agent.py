import time

import pytest

from agentMET4FOF.agents import AgentMET4FOF
from tests.conftest import test_timeout


@pytest.mark.timeout(test_timeout)
def test_remove_agent(agent_network):
    # init agents by adding into the agent network
    dummy_agent1 = agent_network.add_agent(agentType=AgentMET4FOF)
    dummy_agent2 = agent_network.add_agent(agentType=AgentMET4FOF)
    dummy_agent3 = agent_network.add_agent(agentType=AgentMET4FOF)
    dummy_agent4 = agent_network.add_agent(agentType=AgentMET4FOF)

    assert len(agent_network.agents()) != 0

    agent_network.bind_agents(dummy_agent1, dummy_agent2)
    agent_network.bind_agents(dummy_agent1, dummy_agent3)
    agent_network.bind_agents(dummy_agent1, dummy_agent4)

    agent_network.bind_agents(dummy_agent4, dummy_agent2)
    agent_network.bind_agents(dummy_agent3, dummy_agent4)
    agent_network.bind_agents(dummy_agent2, dummy_agent4)

    agent_network.remove_agent(dummy_agent1)
    agent_network.remove_agent(dummy_agent2)
    agent_network.remove_agent(dummy_agent3)
    agent_network.remove_agent(dummy_agent4)

    # Check if agent network really contains no agent after a short latency
    # period. This reduces test runtime in case of passed tests but
    # results in quite cryptic error messages in case it fails due to the
    # timeout causing the actual fail. So, if this line fails, regardless of
    # the error message, it means, the agents have not been properly removed.
    is_not_zero = True
    while is_not_zero:
        try:
            assert len(agent_network.agents()) == 0
            is_not_zero = False
        except AssertionError:
            pass
