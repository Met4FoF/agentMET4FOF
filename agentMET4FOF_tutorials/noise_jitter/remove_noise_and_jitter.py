from agentMET4FOF.agents.base_agents import MonitorAgent
from agentMET4FOF.agents.noise_jitter_removal_agents import NoiseJitterRemovalAgent
from agentMET4FOF.agents.signal_agents import StaticSineWithJitterGeneratorAgent, NoiseAgent
from agentMET4FOF.network import AgentNetwork


def demonstrate_noise_jitter_removal_agent():
    # start agent network server
    agentNetwork = AgentNetwork(backend="mesa")
    # init agents

    sine_with_jitter_agent = agentNetwork.add_agent(agentType=StaticSineWithJitterGeneratorAgent)

    noise_agent = agentNetwork.add_agent(agentType=NoiseAgent)

    njremove_agent = agentNetwork.add_agent(agentType=NoiseJitterRemovalAgent)

    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent, name="Sine with Noise and Jitter")
    monitor_agent2 = agentNetwork.add_agent(agentType=MonitorAgent, name="Output of Noise-Jitter Removal Agent")

    # connect agents : jitter generator -> noise -> njremoval agent
    agentNetwork.bind_agents(sine_with_jitter_agent, noise_agent)
    agentNetwork.bind_agents(noise_agent, njremove_agent)

    # connect monitor agents
    agentNetwork.bind_agents(noise_agent, monitor_agent)
    agentNetwork.bind_agents(njremove_agent, monitor_agent2)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    demonstrate_noise_jitter_removal_agent()
