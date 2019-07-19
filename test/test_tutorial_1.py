from AgentMET4FOF import AgentNetwork, MonitorAgent


def test_tutorial_1():
    #start agent network server
    agentNetwork = AgentNetwork()
    agentNetwork.start_server()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent()
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    #connect agents by either way:
    # 1) by agent network.bind_agents(source,target)
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output()
    gen_agent.bind_output(monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # Shutdown network
    agentNetwork.shutdown()
