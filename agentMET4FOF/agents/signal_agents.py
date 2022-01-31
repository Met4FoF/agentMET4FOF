from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base_agents import AgentMET4FOF
from ..streams.signal_streams import (
    SineGenerator,
    SineWithJitterGenerator,
    StaticSineWithJitterGenerator,
)

__all__ = [
    "SineGeneratorAgent",
    "SineWithJitterGeneratorAgent",
    "StaticSineWithJitterGeneratorAgent",
    "NoiseAgent",
]


class SineGeneratorAgent(AgentMET4FOF):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineGenerator` and pushes them sample by sample
    to connected agents via its output channel.
    """

    _sine_stream: SineGenerator

    def init_parameters(
        self, sfreq=100, sine_freq=2 * np.pi, amplitude=1.0, initial_phase=0.0
    ):
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
            self.send_output(sine_data)


class StaticSineWithJitterGeneratorAgent(AgentMET4FOF):
    """An agent streaming a pre generated sine signal of fixed length with jitter

    Takes samples from the :py:mod:`StaticSineGeneratorWithJitter` and pushes them
    sample by sample to connected agents via its output channel.
    """

    _sine_stream: StaticSineWithJitterGenerator

    def init_parameters(
        self, num_cycles: Optional[int] = 1000, jitter_std: Optional[float] = 0.02
    ):
        r"""Initialize the pre generated sine signal of fixed length with jitter

        Initialize the static input data as an instance of the
        :class:`StaticSineWithJitterGenerator` class with the provided parameters.

        Parameters
        ----------
        num_cycles : int, optional
            numbers of cycles, determines the signal length by :math:`\pi \cdot
            num\_cycles`, defaults to 1000
        jitter_std : float, optional
            the standard deviation of the distribution to randomly draw jitter from,
            defaults to 0.02
        """
        self._sine_stream = StaticSineWithJitterGenerator(
            num_cycles=num_cycles, jitter_std=jitter_std
        )

    def agent_loop(self):
        """Extract sample by sample the input data stream's content and push it"""
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()  # dictionary
            self.send_output(sine_data)


class NoiseAgent(AgentMET4FOF):
    """An agent adding white noise to the incoming signal"""

    _noise_std: float

    @property
    def noise_std(self):
        """Standard deviation of the distribution to randomly draw noise from"""
        return self._noise_std

    def init_parameters(self, noise_std: Optional[float] = 0.05):
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
            the received message in the expected form::

            dict like {
                "from": "<valid agent name>"
                "data": <time series data as a list, np.ndarray or pd.Dataframe> or
                    dict like {
                        "quantities": <time series data as a list, np.ndarray or
                            pd.Dataframe>,
                        "target": <target labels as a list, np.ndarray or pd.Dataframe>,
                        "time": <time stamps as a list, np.ndarray or pd.Dataframe of
                            float or np.datetime64>
                    }
                "senderType": <any subclass of AgentMet4FoF>,
                "channel": "<channel name>"
                }
        """

        def _compute_noisy_signal_from_clean_signal(
            clean_signal: Union[List, np.ndarray, pd.DataFrame]
        ):
            return np.random.normal(
                loc=clean_signal,
                scale=self._noise_std,
            )

        if self.current_state == "Running":
            data_in_message = message["data"].copy()
            if isinstance(data_in_message, (list, np.ndarray, pd.DataFrame)):
                self.send_output(
                    _compute_noisy_signal_from_clean_signal(data_in_message)
                )
            if isinstance(data_in_message, dict):
                fully_assembled_resulting_data = message["data"].copy()
                fully_assembled_resulting_data[
                    "quantities"
                ] = _compute_noisy_signal_from_clean_signal(
                    data_in_message["quantities"]
                )
                self.send_output(fully_assembled_resulting_data)


class SineWithJitterGeneratorAgent(SineGeneratorAgent):
    """An agent streaming a sine signal

    Takes samples from the :py:mod:`SineWithJitterGenerator` and pushes them sample by
    sample to connected agents via its output channel.
    """

    def init_parameters(
        self,
        sfreq: Optional[int] = 10,
        sine_freq: Optional[float] = np.reciprocal(2 * np.pi),
        amplitude: Optional[float] = 1.0,
        initial_phase: Optional[float] = 0.0,
        jitter_std: Optional[float] = 0.02,
    ):
        r"""Initialize the input data

        Initialize the input data stream as an instance of the
        :class:`SineWithJitterGenerator` class.

        Parameters
        ----------
        sfreq : int, optional
            sampling frequency which determines the time step when :meth:`.next_sample`
            is called, defaults to 10
        sine_freq : float, optional
            frequency of the generated sine wave, defaults to :math:`\frac{1}{2 \pi}`
        amplitude : float, optional
            amplitude of the generated sine wave, defaults to 1.0
        initial_phase : float, optional
            initial phase (at t=0) of the generated sine wave, defaults to 0.0
        jitter_std : float, optional
            the standard deviation of the distribution to randomly draw jitter from,
            defaults to 0.02
        """
        self._sine_stream = SineWithJitterGenerator(
            sfreq=sfreq,
            sine_freq=sine_freq,
            amplitude=amplitude,
            initial_phase=initial_phase,
            jitter_std=jitter_std,
        )

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :meth:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            sine_data = self._sine_stream.next_sample()
            self.send_output(sine_data)
