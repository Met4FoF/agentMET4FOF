from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork
import time

def test_remove_agent():
    #start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=False)

    #init agents by adding into the agent network
    dummy_agent1 = agentNetwork.add_agent(agentType= AgentMET4FOF)
    dummy_agent2 = agentNetwork.add_agent(agentType= AgentMET4FOF)
    dummy_agent3 = agentNetwork.add_agent(agentType= AgentMET4FOF)
    dummy_agent4 = agentNetwork.add_agent(agentType= AgentMET4FOF)

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

