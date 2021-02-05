from agentMET4FOF.agents import AgentNetwork, MonitorAgent, SineGeneratorAgent


def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    gen_agent_1 = agent_network.add_agent(agentType=SineGeneratorAgent)
    gen_agent_2 = agent_network.add_agent(agentType=SineGeneratorAgent)
    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    # bind generator agents outputs to monitor
    agent_network.bind_agents(gen_agent_1, monitor_agent)
    agent_network.bind_agents(gen_agent_2, monitor_agent)

    # setup health coalition group
    agent_network.add_coalition(
        "REDUNDANT_SENSORS", [gen_agent_1, gen_agent_2, monitor_agent]
    )

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()
