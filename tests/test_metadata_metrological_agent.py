import time
from typing import Dict

import numpy as np
import pytest
from time_series_metadata.scheme import MetaData

from agentMET4FOF.metrological_agents import MetrologicalAgent, MetrologicalMonitorAgent
from tests.conftest import test_timeout


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
        return 1013.25

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


@pytest.mark.timeout(test_timeout)
def test_simple_metrological_agent(agent_network):
    # Create an agent with data source and metadata, attach it to a monitor agent and
    # check, if the metadata is present at the right place in the monitor agent after
    # a short while.

    # Create a data source.
    signal = Signal()
    source_name = signal.metadata.metadata["device_id"]

    # Init agents by adding into the agent network.
    simple_agent = agent_network.add_agent(
        name=source_name, agentType=MetrologicalSineGeneratorAgent
    )
    simple_agent.init_parameters(signal)
    monitor_agent_1 = agent_network.add_agent(agentType=MetrologicalMonitorAgent)

    # Connect agents.
    agent_network.bind_agents(simple_agent, monitor_agent_1)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Run check of expected and actual result until test times out.
    is_present = True
    while not is_present:
        try:
            # Run actual check. This reduces test runtime in case of passed tests but
            # results in quite cryptic error messages in case it fails due to the
            # timeout causing the actual fail. So, if this line fails, regardless of
            # the error message, it means, the addressed attribute's content does not
            # match the expected expression.
            # Check if key 'metadata' is present in the received data
            buffer_dict = list(monitor_agent_1.get_attr('buffer').values())[0]
            is_present = "metadata" in buffer_dict.keys()
        except IndexError:
            pass


