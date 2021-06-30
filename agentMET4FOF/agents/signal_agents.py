from .base_agents import AgentMET4FOF, MetrologicalAgent
from ..metrological_streams import (
    MetrologicalDataStreamMET4FOF,
    MetrologicalSineGenerator,
)
from ..streams.signal_streams import SineGenerator

__all__ = ["MetrologicalGeneratorAgent", "SineGeneratorAgent"]


class SineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    def init_parameters(self, sfreq=500, sine_freq=5):
        """Initialize the input data

        Initialize the input data stream as an instance of the :class:`SineGenerator`
        class.

        Parameters
        ----------
        sfreq : int
            sampling frequency for the underlying signal
        sine_freq : float
            frequency of the generated sine wave
        """
        self._sine_stream = SineGenerator(sfreq=sfreq, sine_freq=sine_freq)

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :meth:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["quantities"])


class MetrologicalGeneratorAgent(MetrologicalAgent):
    """An agent streaming a specified signal

    Takes samples from an instance of :py:class:`MetrologicalDataStreamMET4FOF` with sampling frequency `sfreq` and
    signal frequency `sine_freq` and pushes them sample by sample to connected agents via its output channel.
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
        signal : MetrologicalDataStreamMET4FOF (defaults to :py:class:`MetrologicalSineGenerator`)
            the underlying signal for the generator
        """
        self._stream = signal
        super().init_parameters()
        self.set_output_data(channel="default", metadata=self._stream.metadata)

    @property
    def device_id(self):
        return self._stream.metadata.metadata["device_id"]

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input
        datastream's content and push it into its output buffer.
        """
        if self.current_state == "Running":
            self.set_output_data(channel="default", data=self._stream.next_sample())
            super().agent_loop()