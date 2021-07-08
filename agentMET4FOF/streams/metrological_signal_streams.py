from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm

from .metrological_base_streams import MetrologicalDataStreamMET4FOF

__all__ = [
    "MetrologicalMultiWaveGenerator",
    "MetrologicalSineGenerator",
]


class MetrologicalSineGenerator(MetrologicalDataStreamMET4FOF):
    """Built-in class of sine wave generator

    Parameters
    ----------
    sfreq : int, optional
        Sampling frequency which determines the time step when :meth:`.next_sample` is
        called. Defaults to 500.
    sine_freq : float, optional
        Frequency of the wave function. Defaults to 50.0.
    amplitude : float, optional
        Amplitude of the wave function. Defaults to 1.0.
    initial_phase : float, optional
        Initial phase of the wave function. Defaults to 0.0.
    device_id : str, optional
        Name of the represented generator. Defaults to 'SineGenerator'.
    time_name : str, optional
        Name for the time dimension. Defaults to 'time'.
    time_unit : str, optional
        Unit for the time. Defaults to 's'.
    quantity_names : iterable of str or str, optional
        An iterable of names of the represented quantities' values.
        Defaults to ('Voltage')
    quantity_units : iterable of str or str, optional
        An iterable of units for the quantities' values. Defaults to ('V')
    misc : Any, optional
        This parameter can take any additional metadata which will be handed over to
        the corresponding attribute of the created :class:`Metadata` object. Defaults to
        'Simple sine wave generator'.
    value_unc : iterable of floats or float, optional
        standard uncertainty(ies) of the quantity values. Defaults to 0.1.
    time_unc : iterable of floats or float, optional
        standard uncertainty of the time stamps. Defaults to 0.0.
    """

    def __init__(
        self,
        sfreq: int = 500,
        sine_freq: float = 50,
        amplitude: float = 1.0,
        initial_phase: float = 0.0,
        device_id: str = "SineGenerator",
        time_name: str = "time",
        time_unit: str = "s",
        quantity_names: Union[str, Tuple[str, ...]] = "Voltage",
        quantity_units: Union[str, Tuple[str, ...]] = "V",
        misc: Optional[Any] = "Simple sine wave generator",
        value_unc: float = 0.1,
        time_unc: float = 0,
    ):
        super(MetrologicalSineGenerator, self).__init__(
            value_unc=value_unc, time_unc=time_unc
        )
        self.set_metadata(
            device_id=device_id,
            time_name=time_name,
            time_unit=time_unit,
            quantity_names=quantity_names,
            quantity_units=quantity_units,
            misc=misc,
        )
        self.value_unc = value_unc
        self.time_unc = time_unc
        self.set_generator_function(
            generator_function=self._sine_wave_function,
            uncertainty_generator=self._default_uncertainty_generator,
            sfreq=sfreq,
            sine_freq=sine_freq,
            amplitude=amplitude,
            initial_phase=initial_phase,
        )

    def _sine_wave_function(self, time, sine_freq, amplitude, initial_phase):
        """A simple sine wave generator"""
        value = amplitude * np.sin(
            np.multiply(2 * np.pi * sine_freq, time) + initial_phase
        )
        value += np.random.normal(0, self.value_unc, value.shape)
        return value


class MetrologicalMultiWaveGenerator(MetrologicalDataStreamMET4FOF):
    """Class to generate data as a sum of cosine wave and additional Gaussian noise.

    Values with associated uncertainty are returned.

    Parameters
    ----------
    sfreq : float
        sampling frequency which determines the time step when next_sample is called.
    intercept : float
        constant intercept of the signal
    freq_arr : np.ndarray of float
        array with frequencies of components included in the signal
    amplitude_arr : np.ndarray of float
        array with amplitudes of components included in the signal
    initial_phase_arr : np.ndarray of float
        array with initial phases of components included in the signal
    noisy : bool
        boolean to determine whether the generated signal should be noisy or "clean"
        defaults to True
    """

    def __init__(
        self,
        sfreq: int = 500,
        freq_arr: np.array = np.array([50]),
        amplitude_arr: np.array = np.array([1.0]),
        initial_phase_arr: np.array = np.array([0.0]),
        intercept: float = 0,
        device_id: str = "MultiWaveDataGenerator",
        time_name: str = "time",
        time_unit: str = "s",
        quantity_names: Union[str, Tuple[str, ...]] = ("Length", "Mass"),
        quantity_units: Union[str, Tuple[str, ...]] = ("m", "kg"),
        misc: Optional[Any] = " Generator for a linear sum of cosines",
        value_unc: Union[float, Iterable[float]] = 0.1,
        time_unc: Union[float, Iterable[float]] = 0,
        noisy: bool = True,
    ):
        super(MetrologicalMultiWaveGenerator, self).__init__(
            value_unc=value_unc, time_unc=time_unc
        )
        self.set_metadata(
            device_id=device_id,
            time_name=time_name,
            time_unit=time_unit,
            quantity_names=quantity_names,
            quantity_units=quantity_units,
            misc=misc,
        )
        self.value_unc = value_unc
        self.time_unc = time_unc
        self.set_generator_function(
            generator_function=self._multi_wave_function,
            sfreq=sfreq,
            intercept=intercept,
            freq_arr=freq_arr,
            amplitude_arr=amplitude_arr,
            initial_phase_arr=initial_phase_arr,
            noisy=noisy,
        )

    def _multi_wave_function(
        self, time, intercept, freq_arr, amplitude_arr, initial_phase_arr, noisy
    ):

        value_arr = intercept
        if noisy:
            value_arr += self.value_unc / 2 * norm.rvs(size=time.shape)

        for freq, amplitude, initial_phase in zip(
            freq_arr, amplitude_arr, initial_phase_arr
        ):
            value_arr += amplitude * np.cos(2 * np.pi * freq * time + initial_phase)

        return value_arr
