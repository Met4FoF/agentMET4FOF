from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import DataStreamMET4FOF
import numpy as np


class SineGenerator(DataStreamMET4FOF):
    """Copy of the built-in class of a streaming sine wave generator

    `sfreq` is sampling frequency which determines the time step when next_sample is called
    `F` is frequency of wave function
    `sine_wave_function` is a custom defined function which has a required keyword `time` as argument and any number of optional additional arguments (e.g `F`).
    to be supplied to the `set_generator_function`

    """

    def __init__(self, sfreq=500, F=5):
        super().__init__()
        self.set_metadata(
            "SineGenerator",
            "time",
            "s",
            ("Voltage"),
            ("V"),
            "Simple sine wave generator",
        )
        self.set_generator_function(
            generator_function=self.sine_wave_function, sfreq=sfreq, F=F
        )

    def sine_wave_function(self, time, F=50):
        value = np.sin(2 * np.pi * F * time)
        return value


class SineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    # # The datatype of the stream will be SineGenerator.
    _sine_stream: SineGenerator

    def init_parameters(self):
        """Initialize the input data

        Initialize the input data stream as an instance of the
        :py:mod:`SineGenerator` class
        """
        self.sine_stream = SineGenerator()

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self.sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["quantities"])


def demonstrate_generator_agent_use():
    # Start agent network server.
    agent_network = AgentNetwork()

    # Initialize agents by adding them to the agent network.
    gen_agent = agent_network.add_agent(agentType=SineGeneratorAgent)
    gen_agent.init_parameters()
    monitor_agent = agent_network.add_agent(agentType=MonitorAgent)

    # Interconnect agents by either way:
    # 1) by agent network.bind_agents(source, target).
    agent_network.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output().
    gen_agent.bind_output(monitor_agent)

    # Set all agents' states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution
    return agent_network


if __name__ == "__main__":
    demonstrate_generator_agent_use()
