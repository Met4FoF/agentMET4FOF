from agentMET4FOF.agents import AgentNetwork
from agentMET4FOF.metrological_agents import MetrologicalAgent, MetrologicalMonitorAgent
import time

def test_addremove_metrological_agents():
    agentNetwork = AgentNetwork(dashboard_modules=False)

    #init agents by adding into the agent network
    dummy_agent1 = agentNetwork.add_agent(agentType=MetrologicalAgent)
    dummy_agent2 = agentNetwork.add_agent(agentType=MetrologicalAgent)
    dummy_agent3 = agentNetwork.add_agent(agentType=MetrologicalMonitorAgent)
    dummy_agent4 = agentNetwork.add_agent(agentType=MetrologicalMonitorAgent)

    agentNetwork.bind_agents(dummy_agent1, dummy_agent2)
    agentNetwork.bind_agents(dummy_agent1, dummy_agent3)
    agentNetwork.bind_agents(dummy_agent1, dummy_agent4)

    agentNetwork.bind_agents(dummy_agent4, dummy_agent2)
    agentNetwork.bind_agents(dummy_agent3, dummy_agent4)
    agentNetwork.bind_agents(dummy_agent2, dummy_agent4)

    agentNetwork.remove_agent(dummy_agent1)
    agentNetwork.remove_agent(dummy_agent2)
    agentNetwork.remove_agent(dummy_agent3)
    agentNetwork.remove_agent(dummy_agent4)
    time.sleep(2)
    assert len(agentNetwork.agents()) == 0
    agentNetwork.shutdown()
