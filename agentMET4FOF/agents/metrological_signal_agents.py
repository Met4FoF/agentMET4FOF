from .metrological_base_agents import MetrologicalAgent
from ..streams.metrological_base_streams import MetrologicalDataStreamMET4FOF
from ..streams.metrological_signal_streams import MetrologicalSineGenerator

__all__ = ["MetrologicalGeneratorAgent"]


class MetrologicalGeneratorAgent(MetrologicalAgent):
    """An agent streaming a specified signal

    Takes samples from an instance of :py:class:`MetrologicalDataStreamMET4FOF` with
    sampling frequency ``sfreq`` and signal frequency ``sine_freq`` and pushes them
    sample by sample to connected agents via its output channel.
    """

    # The datatype of the stream will be MetrologicalSineGenerator.
    _stream: MetrologicalDataStreamMET4FOF

    def init_parameters(
        self,
        signal: MetrologicalDataStreamMET4FOF = MetrologicalSineGenerator(),
        **kwargs,
    ):
        """Initialize the input data stream

        Parameters
        ----------
        signal : MetrologicalDataStreamMET4FOF
            the underlying signal for the generator (defaults to
            :py:class:`MetrologicalSineGenerator`)
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
