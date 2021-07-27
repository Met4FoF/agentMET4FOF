import numpy as np

from agentMET4FOF.agents import MonitorAgent
from agentMET4FOF.agents.signal_agents import (
    SineGeneratorAgent,
    StaticSineGeneratorWithJitterAgent,
)
from agentMET4FOF.network import AgentNetwork


def demonstrate_generator_agent_use():
    agent_network = AgentNetwork(backend="mesa")

    sine_agent = agent_network.add_agent(
        name="Clean sine signal", agentType=SineGeneratorAgent
    )
    sine_agent.init_parameters(sfreq=395, sine_freq=2 * np.pi)
    jitter_agent = agent_network.add_agent(
        name="Sine signal with jitter", agentType=StaticSineGeneratorWithJitterAgent
    )
    jitter_agent.init_parameters(jitter_std=0.05)
    monitor_agent = agent_network.add_agent(
        name="Compare clean signal and signal with jitter", agentType=MonitorAgent
    )

    agent_network.bind_agents(sine_agent, monitor_agent)
    agent_network.bind_agents(jitter_agent, monitor_agent)

    agent_network.set_running_state()

    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()
