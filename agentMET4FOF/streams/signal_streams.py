from typing import Dict, Optional

import numpy as np

from .base_streams import DataStreamMET4FOF

__all__ = [
    "SineGenerator",
    "CosineGenerator",
    "SineWithJitterGenerator",
    "StaticSineWithJitterGenerator",
]


class SineGenerator(DataStreamMET4FOF):
    """Streaming sine wave generator

    Parameters
    ----------
    sfreq : int, optional
        sampling frequency which determines the time step when :meth:`.next_sample`
        is called, defaults to 500
    sine_freq : float, optional
        frequency of wave function, defaults to 50.0
    amplitude : float, optional
        amplitude of the wave function, defaults to 1.0
    initial_phase : float, optional
        initial phase of the wave function, defaults to 0.0
    """

    def __init__(
        self,
        sfreq: Optional[int] = 500,
        sine_freq: Optional[float] = 50.0,
        amplitude: Optional[float] = 1.0,
        initial_phase: Optional[float] = 0.0,
    ):
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
            generator_function=self.sine_wave_function,
            sfreq=sfreq,
            sine_freq=sine_freq,
            amplitude=amplitude,
            initial_phase=initial_phase,
        )

    def sine_wave_function(self, time, sine_freq, amplitude, initial_phase):
        """A simple sine wave generator

        Parameters
        ----------
        time : Union[List, DataFrame, np.ndarray]
            the time stamps at which to evaluate the function
        sine_freq : float
            frequency of wave function
        amplitude : float
            amplitude of the wave function
        initial_phase : float
            initial phase of the wave function

        Returns
        -------
        np.ndarray
            the sine values of the specified curve at ``time``
        """
        return amplitude * np.sin(2 * np.pi * sine_freq * time + initial_phase)


class CosineGenerator(DataStreamMET4FOF):
    """Streaming cosine wave generator

    Parameters
    ----------
    sfreq : int, optional
        sampling frequency which determines the time step when :meth:`.next_sample`
        is called, defaults to 500
    cosine_freq : float, optional
        frequency of wave function, defaults to 50.0
    amplitude : float, optional
            amplitude of the wave function, defaults to 1.0
    initial_phase : float, optional
            initial phase of the wave function, defaults to 0.0
    """

    def __init__(
        self,
        sfreq: Optional[int] = 500,
        cosine_freq: Optional[float] = 50.0,
        amplitude: Optional[float] = 1.0,
        initial_phase: Optional[float] = 0.0,
    ):
        super().__init__()
        self.set_metadata(
            "CosineGenerator",
            "time",
            "s",
            ("Voltage"),
            ("V"),
            "Simple cosine wave generator",
        )
        self.set_generator_function(
            generator_function=self.cosine_wave_function,
            sfreq=sfreq,
            cosine_freq=cosine_freq,
            amplitude=amplitude,
            initial_phase=initial_phase,
        )

    def cosine_wave_function(self, time, cosine_freq, amplitude, initial_phase):
        """A simple cosine wave generator

        Parameters
        ----------
        time : Union[List, DataFrame, np.ndarray]
            the time stamps at which to evaluate the function
        cosine_freq : float
            frequency of wave function
        amplitude : float
                amplitude of the wave function
        initial_phase : float
                initial phase of the wave function

        Returns
        -------
        np.ndarray
            the cosine values of the specified curve at ``time``
        """
        value = amplitude * np.cos(2 * np.pi * cosine_freq * time + initial_phase)
        return value


class StaticSineWithJitterGenerator(DataStreamMET4FOF):
    r"""Represents a fixed length sine signal with jitter

    Parameters
    ----------
    num_cycles : int, optional
        numbers of cycles, determines the signal length by :math:`\pi \cdot
        num_cycles`, defaults to 1000
    jitter_std : float, optional
        the standard deviation of the distribution to randomly draw jitter from,
        defaults to 0.02
    """

    def __init__(self, num_cycles=1000, jitter_std=0.02):
        super().__init__()
        timestamps = np.arange(0, np.pi * num_cycles, 0.1)
        timestamps_with_jitter = np.random.normal(loc=timestamps, scale=jitter_std)
        signal_values_at_timestamps = np.sin(timestamps_with_jitter)
        self.set_data_source(quantities=signal_values_at_timestamps, time=timestamps)


class SineWithJitterGenerator(SineGenerator):
    r"""Represents a streamed sine signal with jitter

    Parameters
    ----------
    sfreq : int, optional
        sampling frequency which determines the time step when :meth:`.next_sample`
        is called, defaults to 10
    sine_freq : float, optional
        frequency of wave function, defaults to :math:`\frac{1}{2 \pi}`
    amplitude : float, optional
        amplitude of the wave function, defaults to 1.0
    initial_phase : float, optional
        initial phase of the wave function, defaults to 0.0
    jitter_std : float, optional
        the standard deviation of the distribution to randomly draw jitter from,
        defaults to 0.02
    """

    _jitter_std: float

    @property
    def jitter_std(self):
        """The standard deviation of the distribution to randomly draw jitter from"""
        return self._jitter_std

    def __init__(
        self,
        sfreq: Optional[int] = 10,
        sine_freq: Optional[float] = np.reciprocal(2 * np.pi),
        amplitude: Optional[float] = 1.0,
        initial_phase: Optional[float] = 0.0,
        jitter_std: Optional[float] = 0.02,
    ):
        self._jitter_std = jitter_std
        super().__init__(
            sfreq=sfreq,
            sine_freq=sine_freq,
            amplitude=amplitude,
            initial_phase=initial_phase,
        )

    def _next_sample_generator(
        self, batch_size: Optional[int] = 1
    ) -> Dict[str, np.ndarray]:
        """Generate the next batch of samples from the sine function with jitter

        Parameters
        ----------
        batch_size : int, optional
            number of batches to get from data stream, defaults to 1

        Returns
        -------
        Dict[str, np.ndarray]
            latest samples of the sine signal with jitter in the form::

            dict like {
                "quantities": <time series data as np.ndarray>,
                "time": <time stamps as np.ndarray>
            }
        """
        sine_signal_with_time_stamps = super()._next_sample_generator()

        sine_signal_with_time_stamps["time"] = np.random.normal(
            loc=sine_signal_with_time_stamps["time"], scale=self.jitter_std
        )

        return sine_signal_with_time_stamps
