from agentMET4FOF.agents import AgentNetwork, MonitorAgent, SineGeneratorAgent


def demonstrate_mesa_backend():

    # Start agent network and specify backend via the corresponding keyword parameter.
    _agent_network = AgentNetwork(backend="mesa")

    # Initialize agents by adding them to the agent network.
    sine_agent = _agent_network.add_agent(agentType=SineGeneratorAgent)
    monitor_agent = _agent_network.add_agent(agentType=MonitorAgent, buffer_size=200)
    sine_agent.bind_output(monitor_agent)

    # Set all agents states to "Running".
    _agent_network.set_running_state()

    return _agent_network


if __name__ == "__main__":
    demonstrate_mesa_backend()
