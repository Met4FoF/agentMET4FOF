from agentMET4FOF.agents import AgentNetwork
from agentMET4FOF.metrological_agents import (
    MetrologicalMonitorAgent,
    MetrologicalGeneratorAgent,
)
from agentMET4FOF.metrological_streams import (
    MetrologicalSineGenerator,
    MetrologicalMultiWaveGenerator,
)


def demonstrate_metrological_stream():
    """Demonstrate an agent network with two metrologically enabled agents

    The agents are defined as objects of the :class:`MetrologicalGeneratorAgent`
    class whose outputs are bound to a single monitor agent.

    The metrological agents generate signals from a sine wave and a multiwave generator
    source.

    Returns
    -------
    :class:`AgentNetwork`
        The initialized and running agent network object
    """
    # start agent network server
    agent_network = AgentNetwork(dashboard_modules=True)

    # Initialize metrologically enabled agent with a multiwave (sum of cosines)
    # generator as signal source taking name from signal source metadata.
    signal_multiwave = MetrologicalMultiWaveGenerator(
        quantity_names="Voltage", quantity_units="V"
    )
    source_name_multiwave = signal_multiwave.metadata.metadata["device_id"]
    source_agent_multiwave = agent_network.add_agent(
        name=source_name_multiwave, agentType=MetrologicalGeneratorAgent
    )
    source_agent_multiwave.init_parameters(signal=signal_multiwave)

    # Initialize second metrologically enabled agent with a sine generator as signal
    # source taking name from signal source metadata.
    signal_sine = MetrologicalSineGenerator()
    source_name_sine = signal_sine.metadata.metadata["device_id"]
    source_agent_sine = agent_network.add_agent(
        name=source_name_sine, agentType=MetrologicalGeneratorAgent
    )
    source_agent_sine.init_parameters(signal=signal_sine)

    # Initialize metrologically enabled plotting agent.
    monitor_agent = agent_network.add_agent(
        "MonitorAgent",
        agentType=MetrologicalMonitorAgent,
        buffer_size=50,
    )

    # Bind agents.
    source_agent_multiwave.bind_output(monitor_agent)
    source_agent_sine.bind_output(monitor_agent)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution.
    return agent_network


if __name__ == "__main__":
    demonstrate_metrological_stream()
