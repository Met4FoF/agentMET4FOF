# By default, "osbrain" backend offers real connectivity between agents (each agent has its own port & IP address) in distributed systems (e,g connecting agents from raspberry pis to PCs, etc),
# which explains why it is harder to debug.
# In the "mesa" backend, there's only one real timer which is started in the AgentNetwork, and every timer tick will advance the agent actions by calling `step()` which includes `agent_loop` and `on_received_message`.
# Moreover, in the "mesa" backend, agents do not have their own port and IP addresses, they are simulated objects to emulate the behaviour of distributed agents.
# Hence, "osbrain" is closer to deployment phase, whereas mesa is suited for the simulation/designing phase.
# To switch between the backends, simply pass the backend parameter to either "mesa" or "osbrain" in the AgentNetwork instantiation.

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
