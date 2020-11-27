import time

import pytest

from agentMET4FOF.metrological_agents import MetrologicalAgent, MetrologicalMonitorAgent

# Create a list of permutations with which we want to test creation of agents and
# binding.
from tests.conftest import test_timeout

agent_type_permutations = [
    [MetrologicalAgent],
    [MetrologicalMonitorAgent],
    [MetrologicalAgent, MetrologicalAgent],
    [MetrologicalAgent, MetrologicalMonitorAgent],
    [MetrologicalMonitorAgent, MetrologicalAgent],
    [MetrologicalMonitorAgent, MetrologicalMonitorAgent],
    [
        MetrologicalAgent,
        MetrologicalAgent,
        MetrologicalMonitorAgent,
        MetrologicalMonitorAgent,
    ],
]


@pytest.mark.timeout(test_timeout)
@pytest.mark.parametrize("agent_types", agent_type_permutations)
def test_addremove_metrological_agents(agent_network, agent_types):
    # Check for all combinations in agent_type_permutations, if agents can be added
    # to an agent network, which as a result contains those agents.
    n_agents = len(agent_types)
    # Create the agents as desired.
    for agent_type in agent_types:
        agent_network.add_agent(agentType=agent_type)
    # Check if for each element in agent_types there is an agent in the network.
    assert len(agent_network.agents()) == n_agents

    for agent in agent_network.agents():
        # Check if removal does not raise errors.
        assert agent_network.remove_agent(agent) is None
        n_agents -= 1
        # Check if agent network really contains one agent less after a short latency
        # period. This reduces test runtime in case of passed tests but
        # results in quite cryptic error messages in case it fails due to the
        # timeout causing the actual fail. So, if this line fails, regardless of
        # the error message, it means, the agents have not been properly removed.
        is_not_equal = True
        while is_not_equal:
            try:
                assert len(agent_network.agents()) == n_agents
                is_not_equal = False
            except AssertionError:
                pass


@pytest.mark.parametrize("agent_types", agent_type_permutations)
def test_bind_agents(agent_network, agent_types):
    # Check for all combinations in agent_type_permutations, if agents can be bound.
    # Create the agents according to agent_types.
    agents = [
        agent_network.add_agent(agentType=agent_type) for agent_type in agent_types
    ]

    # Interconnect all agents unidirectional.
    for index, agent in enumerate(agents):
        subsequent_agents = agents[index:]
        for subsequent_agent in subsequent_agents:
            assert agent_network.bind_agents(agent, subsequent_agent) == 0


@pytest.mark.parametrize("agent_types", agent_type_permutations)
def test_bind_agents(agent_network, agent_types):
    # Check for all combinations in agent_type_permutations, if agents can be removed
    # again.
    # Create the agents according to agent_types.
    agents = [
        agent_network.add_agent(agentType=agent_type) for agent_type in agent_types
    ]

    # Interconnect all agents unidirectional.
    for index, agent in enumerate(agents):
        subsequent_agents = agents[index:]
        for subsequent_agent in subsequent_agents:
            assert agent_network.bind_agents(agent, subsequent_agent) == 0
