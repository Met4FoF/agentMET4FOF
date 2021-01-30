from agentMET4FOF.agents import AgentNetwork
from develop_UI import agent_module_example

def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork(dashboard_modules=[agent_module_example])

    return agent_network

if __name__ == "__main__":
    demonstrate_generator_agent_use()
