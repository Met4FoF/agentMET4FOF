import time
from typing import Dict

import numpy as np
from time_series_metadata.scheme import MetaData

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork
from agentMET4FOF.metrological_agents import MetrologicalAgent, MetrologicalMonitorAgent


class Signal:
    """
    Simple class to request time-series datapoints of a signal
    """

    def __init__(self):
        self._description = MetaData(
            device_id="my_virtual_sensor",
            time_name="time",
            time_unit="s",
            quantity_names="pressure",
            quantity_units="Pa",
        )

    @staticmethod
    def _time():
        return time.time()

    @staticmethod
    def _time_unc():
        return time.get_clock_info("time").resolution

    @staticmethod
    def _value(timestamp):
        return 1013.25 + 10 * np.sin(timestamp)

    @staticmethod
    def _value_unc():
        return 0.5

    @property
    def current_datapoint(self):
        t = self._time()
        ut = self._time_unc()
        v = self._value(t)
        uv = self._value_unc()

        return np.array((t, ut, v, uv))

    @property
    def metadata(self) -> MetaData:
        return self._description


class MetrologicalSineGeneratorAgent(MetrologicalAgent):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    # The datatype of the stream will be SineGenerator.
    _sine_stream: Signal

    def init_parameters(self, signal: Signal = Signal(), **kwargs):
        """Initialize the input data

        Initialize the input data stream as an instance of the
        :py:mod:`SineGenerator` class

        Parameters
        ----------
        signal : Signal
            the underlying signal for the generator
        """
        self._sine_stream = signal
        super().init_parameters()
        self.set_output_data(channel="default", metadata=self._sine_stream.metadata)

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking
        :py:method:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            self.set_output_data(
                channel="default", data=[self._sine_stream.current_datapoint]
            )
            super().agent_loop()

    @property
    def metadata(self) -> Dict:
        return self._sine_stream.metadata.metadata


def main():

    # start agent network server
    agent_network = AgentNetwork(dashboard_modules=True)

    # Initialize signal generating class outside of agent framework.
    signal = Signal()

    # Initialize metrologically enabled agent taking name from signal source metadata.
    source_name = signal.metadata.metadata["device_id"]
    source_agent = agent_network.add_agent(
        name=source_name, agentType=MetrologicalSineGeneratorAgent
    )
    source_agent.init_parameters(signal)

    # Initialize metrologically enabled plotting agent.
    monitor_agent = agent_network.add_agent(
        "MonitorAgent", agentType=MetrologicalMonitorAgent
    )

    # Bind agents.
    source_agent.bind_output(monitor_agent)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution.
    return agent_network


if __name__ == "__main__":
    main()
