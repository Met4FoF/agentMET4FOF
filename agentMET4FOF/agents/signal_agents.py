from typing import Any, Dict

import numpy as np

from .base_agents import AgentMET4FOF
from ..streams.signal_streams import SineGenerator, StaticSineWithJitterGenerator

__all__ = ["SineGeneratorAgent", "StaticSineWithJitterGeneratorAgent", "NoiseAgent"]


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


class StaticSineWithJitterGeneratorAgent(AgentMET4FOF):
    """An agent streaming a pre generated sine signal of fixed length with jitter

    Takes samples from the :py:mod:`StaticSineGeneratorWithJitter` and pushes them
    sample by sample to connected agents via its output channel.
    """

    _sine_stream: StaticSineWithJitterGenerator

    def init_parameters(self, jitter_std=0.02):
        """Initialize the input data

        Initialize the static input data as an instance of the
        :class:`StaticSineWithJitterGenerator` class with the provided parameters.

        Parameters
        ----------
        jitter_std : float, optional
            the standard deviation of the distribution to randomly draw jitter from,
            defaults to 0.02
        """
        self._sine_stream = StaticSineGeneratorWithJitter(jitter_std=jitter_std)

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :meth:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data["quantities"])


class NoiseAgent(AgentMET4FOF):
    r"""An agent adding white noise to the incoming signal

    Parameters
    ----------
    noise_std : float, optional
        the standard deviation of the distribution to randomly draw noise from,
        defaults to 0.05
    """
    _noise_std: float

    @property
    def noise_std(self):
        return self._noise_std

    def init_parameters(self, noise_std=0.05):
        """Initialize the noise's standard deviation

        Parameters
        ----------
        noise_std : float, optional
            the standard deviation of the distribution to randomly draw noise from,
            defaults to 0.05
        """
        self._noise_std = noise_std

    def on_received_message(self, message: Dict[str, Any]):
        """Add noise to the received message's data

        Parameters
        ----------
        message : Dictionary
            The message received is in form::

            dict like {
                "from": "<valid agent name>"
                "data": <time series data as a list, np.ndarray or pd.Dataframe>,
                "senderType": <any subclass of AgentMet4FoF>,
                "channel": "<channel name>"
                }
        """
        if self.current_state == "Running":
            noisy_data = np.random.normal(
                loc=message["data"],
                scale=self._noise_std,
            )
            self.send_output(noisy_data)
