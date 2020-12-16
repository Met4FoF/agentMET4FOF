from agentMET4FOF.agents import AgentNetwork, MonitorAgent, SineGeneratorAgent


def demonstrate_metrological_stream():

    # start agent network server
    _agent_network = AgentNetwork(dashboard_modules=True, backend="mesa")

    # Initialize metrologically enabled agent taking name from signal source metadata.
    sine_agent = _agent_network.add_agent(agentType=SineGeneratorAgent)

    # Initialize metrologically enabled plotting agent.
    monitor_agent = _agent_network.add_agent(agentType=MonitorAgent, buffer_size=200)

    # Bind agents.
    sine_agent.bind_output(monitor_agent)

    # Set all agents states to "Running".
    _agent_network.set_running_state()

    return _agent_network


if __name__ == "__main__":
    demonstrate_metrological_stream()
