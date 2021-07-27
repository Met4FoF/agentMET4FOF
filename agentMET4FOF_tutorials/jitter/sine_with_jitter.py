import numpy as np

from agentMET4FOF.agents import (
    AgentNetwork,
    MonitorAgent,
    SineGeneratorAgent,
)
from agentMET4FOF.agents.signal_agents import StaticSineGeneratorWithJitterAgent


def demonstrate_generator_agent_use():
    agent_network = AgentNetwork(backend="mesa")

    sine_agent = agent_network.add_agent(
        name="Clean sine signal", agentType=SineGeneratorAgent
    )
    sine_agent.init_parameters(sfreq=395, sine_freq=2 * np.pi)
    jitter_agent = agent_network.add_agent(
        name="Sine signal with jitter", agentType=StaticSineGeneratorWithJitterAgent
    )
    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    agent_network.bind_agents(sine_agent, monitor_agent)
    agent_network.bind_agents(jitter_agent, monitor_agent)

    agent_network.set_running_state()

    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()
