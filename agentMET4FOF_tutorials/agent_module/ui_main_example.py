# This main example shows how to create and import an agent defined in ui_module_example.py
# Firstly, it is done via python import
# Then, in the instantiation of the AgentNetwork , we pass the `agent_module_example`
# All agents imported will be then available via the dashboard UI
# Note also, the exposed and tunable parameters in the `ParameterisedSineGeneratorAgent` are defined in the class

from agentMET4FOF.agents import AgentNetwork
from agentMET4FOF_tutorials.tutorial_1_generator_agent import SineGeneratorAgent
from agentMET4FOF_tutorials.agent_module import ui_module_example


def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork(dashboard_modules=[ui_module_example], backend="mesa")
    sine_agent = agent_network.add_agent(agentType=SineGeneratorAgent)

    coalition = agent_network.add_coalition(agents=[])

    agent_network.add_coalition_agent(name=coalition.name,agents=[sine_agent])

    return agent_network

if __name__ == "__main__":
    demonstrate_generator_agent_use()
