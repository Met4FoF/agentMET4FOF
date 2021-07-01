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
    """

    def __init__(self, sfreq=500, sine_freq=50):
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
            generator_function=self.sine_wave_function, sfreq=sfreq, sine_freq=sine_freq
        )

    def sine_wave_function(self, time, sine_freq):
        """A simple sine wave generator"""
        value = np.sin(2 * np.pi * sine_freq * time)
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
    F : int
        frequency of wave function
    """

    def __init__(self, sfreq=500, cosine_freq=5):
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
        )

    def cosine_wave_function(self, time, cosine_freq=50):
        """A simple cosine wave generator"""
        value = np.cos(2 * np.pi * cosine_freq * time)
        return value
