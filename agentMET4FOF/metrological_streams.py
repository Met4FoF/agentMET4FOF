import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from agentMET4FOF.streams import DataStreamMET4FOF


class MetrologicalDataStreamMET4FOF(DataStreamMET4FOF):
    """
    Abstract  class for creating datastreams with metrological information. Inherits
    from the :class:`.DataStreamMET4FOF` class

    To create a new :class:`MetrologicalDataStreamMET4FOF` class, inherit this class and
    call :meth:`.set_metadata` in the constructor. Choose one of two types of
    datastreams to be created:

    - from dataset file (:meth:`.set_data_source`), or
    - a waveform generator function (:meth:`.set_generator_function`).

    Alternatively, override the :meth:`.next_sample` function if neither option suits
    the application. For generator functions, :attr:`.sfreq` is a required variable to
    be set on `init` which sets the sampling frequency and the time-step which occurs
    when :meth:`.next_sample()` is called.

    For an example implementation of using generator function, see the built-in
    :class:`MetrologicalSineGenerator` class. See tutorials for more implementations.

    Attributes
    ----------
    _generator_function_unc : Callable
        A generator function for the time and quantity uncertainties which takes in at
        least one argument ``time`` which will be used in :meth:`.next_sample`. The
        return value must be a 2-tuple of time and value uncertainties each of one of
        the three types:

        - np.ndarray
        - pandas DataFrame
        - list

    _uncertainty_parameters : Dict
        Any additional keyword arguments to be supplied to the generator function.
        Both the calls of the value generator function and of
        the uncertainty generator function will be supplied with the
        :attr:`**_uncertainty_parameters`.
    """

    def __init__(
        self,
        value_unc: Optional[float] = 0.0,
        time_unc: Optional[float] = 0.0,
        exp_unc: Optional[float] = None,
        cov_factor: Optional[float] = 1.0,
    ):
        """Initialize a MetrologicalDataStreamMET4FOF object

        Parameters
        ----------
        value_unc : float, optional (defaults to 0)
            standard uncertainties associated with values
        time_unc : float, optional (defaults to 0)
            standard uncertainties associated with timestamps
        exp_unc : float, optional (defaults to None)
            expanded uncertainties associated with values
        cov_factor : float, optional (defaults to 1)
            coverage factor associated with the expanded uncertainty

        If exp_unc and cov_factor are given explicit values, they override value_unc
        according to value_unc = exp_unc / cov_factor
        """
        super().__init__()
        self._uncertainty_parameters: Dict
        self._generator_function_unc: Callable
        self._time_unc: float = time_unc
        self.exp_unc: float = exp_unc
        self.cov_factor: float = cov_factor
        if self.exp_unc is not None:
            self.value_unc: float = self.exp_unc / self.cov_factor
        else:
            self._value_unc: float = value_unc

        self._generator_function_unc = None
        self._uncertainty_parameters = None

    def set_generator_function(
        self,
        generator_function: Callable = None,
        uncertainty_generator: Callable = None,
        sfreq: int = None,
        **kwargs: Optional[Any]
    ) -> Callable:
        """
        Set value and uncertainty generators based on user-defined functions. By
        default, this function resorts to a sine wave generator function and a
        constant (zero) uncertainty. Initialisation of the generator's parameters
        should be done here such as setting the sampling frequency and wave
        frequency. For setting it with a dataset instead,
        see :meth:`.set_data_source`. Overwrites the default
        :meth:`.DataStreamMET4FOF.set_generator_function` method.

        Parameters
        ----------
        generator_function : callable
            A generator function which takes in at least one argument ``time`` which
            will be used in :meth:`.next_sample`.
        uncertainty_generator : callable
            An uncertainty generator function which takes in at least one argument
            ``time`` which will be used in :meth:`.next_sample`.
        sfreq : int
            Sampling frequency.
        **kwargs : Optional[Dict[str, Any]]
            Any additional keyword arguments to be supplied to the generator function.
            The ``**kwargs`` will be saved as :attr:`_uncertainty_parameters`.
            Both the calls of the value generator function and of
            the uncertainty generator function will be supplied with the
            ``**uncertainty_parameters``.

        Returns
        -------
        Callable
            The uncertainty generator function
        """
        # Call the set_generator_function from the parent class to set the generator
        # function.
        super().set_generator_function(
            generator_function=generator_function, sfreq=sfreq, **kwargs
        )

        self._uncertainty_parameters = kwargs

        # resort to default wave generator if one is not supplied
        if uncertainty_generator is None:
            warnings.warn(
                "No uncertainty generator function specified. Setting to default ("
                "constant)."
            )
            self._generator_function_unc = self._default_uncertainty_generator
        else:
            self._generator_function_unc = uncertainty_generator
        return self._generator_function_unc

    def _default_uncertainty_generator(
        self,
        time: Union[List, pd.DataFrame, np.ndarray],
        values: Union[List, pd.DataFrame, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Default (standard) uncertainty generator function

        Parameters
        ----------
        time : Union[List, DataFrame, np.ndarray]
            timestamps
        values : Union[List, DataFrame, np.ndarray]
            values corresponding to timestamps

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            constant time and value uncertainties each of the same shape
            as ``time``
        """
        _time_unc = np.full_like(time, fill_value=self.time_unc)
        _value_unc = np.full_like(values, fill_value=self.value_unc)

        return _time_unc, _value_unc

    def _next_sample_generator(self, batch_size: int = 1) -> np.ndarray:
        """
        Internal method for generating a batch of samples from the generator function.
        Overrides :meth:`.DataStreamMET4FOF._next_sample_generator`. Adds
        time uncertainty ``ut`` and measurement uncertainty ``uv`` to sample
        """
        _time: np.ndarray = (
            np.arange(self._sample_idx, self._sample_idx + batch_size, 1.0).reshape(
                -1, 1
            )
            / self.sfreq
        )
        self._sample_idx += batch_size

        _amplitude: np.ndarray = self._generator_function(
            _time, **self._generator_parameters
        )
        _time_unc, _value_unc = self._generator_function_unc(_time, _amplitude)

        return np.concatenate((_time, _time_unc, _amplitude, _value_unc), axis=1)

    @property
    def value_unc(self) -> Union[float, Iterable[float]]:
        """Union[float, Iterable[float]]: uncertainties associated with the values"""
        return self._value_unc

    @value_unc.setter
    def value_unc(self, value: Union[float, Iterable[float]]):
        self._value_unc = value

    @property
    def time_unc(self) -> Union[float, Iterable[float]]:
        """Union[float, Iterable[float]]: uncertainties associated with timestamps"""
        return self._time_unc

    @time_unc.setter
    def time_unc(self, value: Union[float, Iterable[float]]):
        self._time_unc = value


class MetrologicalSineGenerator(MetrologicalDataStreamMET4FOF):
    """Built-in class of sine wave generator

    Parameters
    ----------
    sfreq : int, optional
        Sampling frequency which determines the time step when :meth:`.next_sample` is
        called. Defaults to 500.
    sine_freq : float, optional
        Frequency of the wave function. Defaults to 50.
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
        standard uncertainty of the time stamps. Defaults to 0.
    """

    def __init__(
        self,
        sfreq: int = 500,
        sine_freq: float = 50,
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
        )

    def _sine_wave_function(self, time, sine_freq):
        """A simple sine wave generator"""
        amplitude = np.sin(np.multiply(2 * np.pi * sine_freq, time))
        amplitude += np.random.normal(0, self.value_unc, amplitude.shape)
        return amplitude


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
    ampl_arr : np.ndarray of float
        array with amplitudes of components included in the signal
    phase_ini_arr : np.ndarray of float
        array with initial phases of components included in the signal
    noisy : bool
        boolean to determine whether the generated signal should be noisy or "clean"
        defaults to True
    """

    def __init__(
                 self,
                 sfreq: int = 500,
                 freq_arr: np.array = np.array([50]),
                 ampl_arr: np.array = np.array([1]),
                 phase_ini_arr: np.array = np.array([0]),
                 intercept: float = 0,
                 device_id: str = "MultiWaveDataGenerator",
                 time_name: str = "time",
                 time_unit: str = "s",
                 quantity_names: Union[str, Tuple[str, ...]] = ("Length", "Mass"),
                 quantity_units: Union[str, Tuple[str, ...]] = ("m", "kg"),
                 misc: Optional[Any] = " Generator for a linear sum of cosines",
                 value_unc: Union[float, Iterable[float]] = 0.1,
                 time_unc: Union[float, Iterable[float]] = 0,
                 noisy: bool = True
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
            misc=misc
        )
        self.value_unc = value_unc
        self.time_unc = time_unc
        self.set_generator_function(
            generator_function=self._multi_wave_function,
            sfreq=sfreq,
            intercept=intercept,
            freq_arr=freq_arr,
            ampl_arr=ampl_arr,
            phase_ini_arr=phase_ini_arr,
            noisy=noisy
        )

    def _multi_wave_function(self, time, intercept, freq_arr, ampl_arr,
                             phase_ini_arr, noisy):

        value_arr = intercept
        if noisy:
            value_arr += self.value_unc / 2 * norm.rvs(size=time.shape)

        for freq, ampl, phase_ini in zip(freq_arr, ampl_arr, phase_ini_arr):
            value_arr += ampl * np.cos(2 * np.pi * freq * time + phase_ini)

        return value_arr
