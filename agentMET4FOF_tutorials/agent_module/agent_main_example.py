from agentMET4FOF.agents import AgentNetwork, MonitorAgent
from agentMET4FOF_tutorials.tutorial_1_generator_agent import SineGeneratorAgent
from develop_UI import agent_module_example

def demonstrate_generator_agent_use():
    # Start agent network server.
    # agent_network = AgentNetwork(dashboard_modules=[agent_module_example])
    agent_network = AgentNetwork(dashboard_modules=[agent_module_example], backend="mesa")
    sine_agent = agent_network.add_agent(agentType=SineGeneratorAgent)

    coalition = agent_network.add_coalition(agents=[])

    agent_network.add_coalition_agent(name=coalition.name,agents=[sine_agent])

    return agent_network

if __name__ == "__main__":
    demonstrate_generator_agent_use()
