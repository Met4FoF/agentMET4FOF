from agentMET4FOF.agents.base_agents import MonitorAgent
from agentMET4FOF.agents.noise_jitter_removal_agents import NoiseJitterRemovalAgent
from agentMET4FOF.agents.signal_agents import (
    NoiseAgent,
    SineWithJitterGeneratorAgent,
)
from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.utils import Backend


def demonstrate_noise_jitter_removal_agent():
    # start agent network server
    agentNetwork = AgentNetwork(backend=Backend.MESA)
    # init agents

    sine_with_jitter_agent = agentNetwork.add_agent(
        agentType=SineWithJitterGeneratorAgent
    )

    noise_agent = agentNetwork.add_agent(agentType=NoiseAgent)

    noise_jitter_removal_agent = agentNetwork.add_agent(
        agentType=NoiseJitterRemovalAgent
    )

    monitor_agent = agentNetwork.add_agent(
        agentType=MonitorAgent, name="Sine with Noise and Jitter"
    )
    monitor_agent2 = agentNetwork.add_agent(
        agentType=MonitorAgent, name="Output of Noise-Jitter Removal Agent"
    )

    # connect agents : jitter generator -> noise -> njremoval agent
    agentNetwork.bind_agents(sine_with_jitter_agent, noise_agent)
    agentNetwork.bind_agents(noise_agent, noise_jitter_removal_agent)

    # connect monitor agents
    agentNetwork.bind_agents(noise_agent, monitor_agent)
    agentNetwork.bind_agents(noise_jitter_removal_agent, monitor_agent2)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    demonstrate_noise_jitter_removal_agent()
