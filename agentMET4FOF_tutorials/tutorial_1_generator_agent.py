from time import sleep

import numpy as np

from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.agents import MonitorAgent, SineGeneratorAgent


def demonstrate_generator_agent_use() -> AgentNetwork:
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    default_sine_agent = agent_network.add_agent(
        name="Default Sine generator", agentType=SineGeneratorAgent
    )

    # Custom parameters of the specified agentType can either be handed over as **kwargs
    # in the network instance's add_agent() method, or...
    custom_sine_agent = agent_network.add_agent(
        name="Custom Sine generator",
        agentType=SineGeneratorAgent,
        sfreq=75,
        sine_freq=np.pi,
    )
    # ... in a separate call of the agent instance's init_parameters() method.
    custom_sine_agent.init_parameters(
        amplitude=0.75,
    )

    monitor_agent = agent_network.add_agent(
        name="Showcase a default and a customized sine signal", agentType=MonitorAgent
    )

    # Interconnect agents by either way:
    # 1) by agent network.bind_agents(source, target).
    agent_network.bind_agents(default_sine_agent, monitor_agent)

    # 2) by the agent.bind_output().
    custom_sine_agent.bind_output(monitor_agent)

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    signal_demo_network = demonstrate_generator_agent_use()
    sleep(60)
    signal_demo_network.shutdown()
