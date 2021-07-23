from agentMET4FOF.agents import AgentNetwork
from agentMET4FOF.metrological_agents import (
    MetrologicalAgent,
    MetrologicalAgentBuffer,
    MetrologicalMonitorAgent,
)
from agentMET4FOF.metrological_streams import (
    MetrologicalDataStreamMET4FOF,
    MetrologicalSineGenerator,
)


class MetrologicalSineGeneratorAgent(MetrologicalAgent):
    """An agent streaming a sine signal

    Takes samples from an instance of :py:class:`MetrologicalSineGenerator` and pushes
    them sample by sample to connected agents via its output channel.
    """

    # The datatype of the stream will be MetrologicalSineGenerator.
    _stream: MetrologicalDataStreamMET4FOF

    def init_parameters(
        self,
        signal: MetrologicalDataStreamMET4FOF = MetrologicalSineGenerator(),
        **kwargs
    ):
        """Initialize the input data stream

        Parameters
        ----------
        signal : MetrologicalDataStreamMET4FOF
            the underlying signal for the generator
        """
        self._stream = signal
        super().init_parameters()
        self.set_output_data(channel="default", metadata=self._stream.metadata)

    def init_buffer(self, buffer_size):
        """
        A method to initialise the buffer. By overriding this method, user can provide
        a custom buffer, instead of the regular AgentBuffer. This can be used,
        for example, to provide a MetrologicalAgentBuffer in the metrological agents.
        """
        buffer = MetrologicalAgentBuffer(buffer_size)
        return buffer

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input
        datastream's content and push it into its output buffer.
        """
        if self.current_state == "Running":
            metrological_sine_data = self._stream.next_sample()

            # Equivalent to self.buffer_store but without logging.
            self.buffer.store(agent_from=self.name, data=metrological_sine_data)

            # The actual dictionary is stored in self.buffer.buffer
            self.log_info(str((self.buffer.buffer)))

            # Check if buffer is filled up, then send out computed mean on the buffer
            if self.buffer.buffer_filled(self.name):

                # Access buffer content by accessing its key, it works like a
                # dictionary. This is a TimeSeriesBuffer object.
                time_series_buffer = self.buffer[self.name]

                # np.ndarray of shape (self.buffer_size, 4)
                buffer_content = time_series_buffer.pop(self.buffer_size)

                # send out metrological data
                self.set_output_data(channel="default", data=buffer_content)
                super().agent_loop()

                # clear buffer
                self.buffer.clear(self.name)


def demonstrate_metrological_stream():

    # start agent network server
    agent_network = AgentNetwork(dashboard_modules=True, backend="mesa")

    # Initialize signal generating class outside of agent framework.
    signal = MetrologicalSineGenerator()

    # Initialize metrologically enabled agent taking name from signal source metadata.
    source_name = signal.metadata.metadata["device_id"]
    source_agent = agent_network.add_agent(
        name=source_name, agentType=MetrologicalSineGeneratorAgent, buffer_size=5
    )
    source_agent.init_parameters(signal)

    # Initialize metrologically enabled plotting agent.
    monitor_agent = agent_network.add_agent(
        "MonitorAgent", agentType=MetrologicalMonitorAgent, buffer_size=50,
    )

    # Bind agents.
    source_agent.bind_output(monitor_agent)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution.
    return agent_network


if __name__ == "__main__":
    demonstrate_metrological_stream()
