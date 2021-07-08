import numpy as np

from .base_streams import DataStreamMET4FOF

__all__ = [
    "SineGenerator",
    "CosineGenerator",
]


class SineGenerator(DataStreamMET4FOF):
    """
    Built-in class of sine wave generator which inherits all
    methods and attributes from :class:`DataStreamMET4FOF`.
    :func:`sine_wave_function` is a custom defined function which has a required
    keyword ``time`` as argument and any number of optional additional arguments
    (e.g ``F``) to be supplied to the :meth:`.DataStreamMET4FOF.set_generator_function`.

    Parameters
    ----------
    sfreq : int
        sampling frequency which determines the time step when :meth:`.next_sample`
        is called
    sine_freq : float
        frequency of wave function
    amplitude : float, optional
        Amplitude of the wave function. Defaults to 1.
    initial_phase : float, optional
        Initial phase of the wave function. Defaults to 0.0.
    """

    def __init__(
        self,
        sfreq=500,
        sine_freq=50,
        amplitude: float = 1.0,
        initial_phase: float = 0.0,
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
        """A simple sine wave generator"""
        value = amplitude * np.sin(2 * np.pi * sine_freq * time + initial_phase)
        return value


class CosineGenerator(DataStreamMET4FOF):
    """
    Built-in class of cosine wave generator which inherits all
    methods and attributes from :class:`DataStreamMET4FOF`.
    :func:`cosine_wave_function` is a custom defined function which has a required
    keyword ``time`` as argument and any number of
    optional additional arguments (e.g ``cosine_freq``) to be supplied to the
    :meth:`.DataStreamMET4FOF.set_generator_function`.

    Parameters
    ----------
    sfreq : int
        sampling frequency which determines the time step when :meth:`.next_sample`
        is called
    cosine_freq : int
        frequency of wave function
    amplitude : float, optional
        Amplitude of the wave function. Defaults to 1.0.
    initial_phase : float, optional
        Initial phase of the wave function. Defaults to 0.0.
    """

    def __init__(
        self, sfreq=500, cosine_freq=50, amplitude: float = 1, initial_phase: float = 0
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
        """A simple cosine wave generator"""
        value = amplitude * np.cos(2 * np.pi * cosine_freq * time + initial_phase)
        return value
