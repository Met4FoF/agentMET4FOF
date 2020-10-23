from agentMET4FOF.agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.agentMET4FOF.streams import SineGenerator

#We demonstrate the use of Coalition of agents to group agents together
#Rationale of grouping depends on the users and application
#For example, we can group sensors which are measuring the same measurand
#To this end, the coalition consists of a list of agent names and
#provides aesthetic differences in the dashboard
#All coalitions visible by the `agent_network` can be accessed via `agent_network.coalitions`

class SineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    # The datatype of the stream will be SineGenerator.
    _sine_stream: SineGenerator

    def init_parameters(self):
        """Initialize the input data

        Initialize the input data stream as an instance of the
        :py:mod:`SineGenerator` class
        """
        self._sine_stream = SineGenerator()

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["x"])


def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    gen_agent_1 = agent_network.add_agent(agentType=SineGeneratorAgent)
    gen_agent_2 = agent_network.add_agent(agentType=SineGeneratorAgent)
    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    #bind generator agents outputs to monitor
    agent_network.bind_agents(gen_agent_1, monitor_agent)
    agent_network.bind_agents(gen_agent_2, monitor_agent)

    #setup health coalition group
    agent_network.add_coalition("REDUNDANT_SENSORS", [gen_agent_1, gen_agent_2, monitor_agent])


    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()
