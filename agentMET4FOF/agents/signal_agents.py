from .base_agents import AgentMET4FOF
from ..streams.signal_streams import SineGenerator, StaticSineGeneratorWithJitter

__all__ = ["SineGeneratorAgent", "StaticSineGeneratorWithJitterAgent"]


class SineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    _sine_stream: SineGenerator

    def init_parameters(self, sfreq=500, sine_freq=5, amplitude=1, initial_phase=0):
        """Initialize the input data

        Initialize the input data stream as an instance of the :class:`SineGenerator`
        class.

        Parameters
        ----------
        sfreq : int
            sampling frequency for the underlying signal
        sine_freq : float
            frequency of the generated sine wave
        amplitude : float
            amplitude of the generated sine wave
        initial_phase : float
            initial phase (at t=0) of the generated sine wave
        """
        self._sine_stream = SineGenerator(
            sfreq=sfreq,
            sine_freq=sine_freq,
            amplitude=amplitude,
            initial_phase=initial_phase,
        )

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :meth:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["quantities"])


class StaticSineGeneratorWithJitterAgent(AgentMET4FOF):
    """An agent streaming a pre generated sine signal of fixed length with jitter

    Takes samples from the :py:mod:`StaticSineGeneratorWithJitter` and pushes them
    sample by sample to connected agents via its output channel.
    """

    _sine_stream: StaticSineGeneratorWithJitter

    def init_parameters(self):
        """Initialize the input data

        Initialize the static input data as an instance of the
        :class:`StaticSineGeneratorWithJitter` class.
        """
        self._sine_stream = StaticSineGeneratorWithJitter()

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :meth:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["quantities"])


class NoiseAgent(AgentMET4FOF):
    def on_received_message(self, message):
        raise NotImplementedError
