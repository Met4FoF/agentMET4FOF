from agentMET4FOF.agents.base_agents import (
    MonitorAgent,
)
from agentMET4FOF.agents.signal_agents import NoiseAgent, SineGeneratorAgent
from agentMET4FOF.network import AgentNetwork


def demonstrate_noise_agent_use():
    agent_network = AgentNetwork(backend="mesa")

    sine_agent = agent_network.add_agent(
        name="Clean sine signal", agentType=SineGeneratorAgent
    )
    noise_agent = agent_network.add_agent(
        name="Sine signal with noise", agentType=NoiseAgent
    )
    monitor_agent = agent_network.add_agent(
        name="Compare clean and noisy signal", agentType=MonitorAgent
    )

    agent_network.bind_agents(sine_agent, monitor_agent)
    agent_network.bind_agents(sine_agent, noise_agent)
    agent_network.bind_agents(noise_agent, monitor_agent)

    agent_network.set_running_state()

    return agent_network


if __name__ == "__main__":
    demonstrate_noise_agent_use()
