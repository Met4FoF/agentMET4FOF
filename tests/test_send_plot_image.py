import time

from agentMET4FOF.agents import AgentNetwork
from agentMET4FOF.agents.base_agents import MonitorAgent
from tests.conftest import GeneratorAgent


def test_send_plot():
    # start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=False)

    # init agents
    gen_agent = agentNetwork.add_agent(agentType=GeneratorAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    agentNetwork.bind_agents(gen_agent, monitor_agent, channel="plot")

    gen_agent.dummy_send_graph(mode="image")
    time.sleep(3)

    assert monitor_agent.get_attr("plots")["GeneratorAgent_1"]

    agentNetwork.shutdown()
