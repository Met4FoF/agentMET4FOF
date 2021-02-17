import warnings
from random import gauss
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np

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

    def __init__(self):
        """Initialize a MetrologicalDataStreamMET4FOF object"""
        super().__init__()
        self._uncertainty_parameters: Dict = None
        self._generator_function_unc: Callable = lambda x: (0, 0)

    def set_generator_function(
        self,
        generator_function: Callable = None,
        uncertainty_generator: Callable = None,
        sfreq: int = None,
        **kwargs: Optional[Any]
    ):
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
                "zero)."
            )
            self._generator_function_unc = self.default_uncertainty_generator
        else:
            self._generator_function_unc = uncertainty_generator
        return self._generator_function_unc

    def default_uncertainty_generator(self, _):
        """Default uncertainty generator function

        Parameters
        ----------
        _ : Any
            unused parameters in place of the normally required time parameter
        
        Returns
        -------
        Tuple[float, float]
            constant (zero) time and amplitude uncertainties
        """
        value_unc = 0
        time_unc = 0
        return time_unc, value_unc

    def _next_sample_generator(
            self, batch_size: int = 1
    ) -> np.ndarray:
        """
        Internal method for generating a batch of samples from the generator function.
        Overrides :meth:`.DataStreamMET4FOF._next_sample_generator`. Adds
        time uncertainty ``ut`` and measurement uncertainty ``uv`` to sample
        """
        timeelement: np.ndarray = (
                np.arange(self._sample_idx, self._sample_idx + batch_size, 1) / self.sfreq
        )
        _time: float = timeelement.item()
        self._sample_idx += batch_size

        _time_unc, _value_unc = self._generator_function_unc(_time)
        amplitude: float = self._generator_function(_time, **self._generator_parameters)

        return np.array((_time, _time_unc, amplitude.item(), _value_unc))


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
        standard uncertainty(ies) of the quantity values. Defaults to 0.5.
    time_unc : iterable of floats or float, optional
        standard uncertainty of the time stamps. Defaults to 0.
    """

    def __init__(
        self,
        sfreq: int=500,
        sine_freq: float=50,
        device_id: str = "SineGenerator",
        time_name: str = "time",
        time_unit: str = "s",
        quantity_names: Union[str, Tuple[str, ...]] = "Voltage",
        quantity_units: Union[str, Tuple[str, ...]] = "V",
        misc: Optional[Any] = "Simple sine wave generator",
        value_unc: Union[float, Iterable[float]] = 0.5,
        time_unc: Union[float, Iterable[float]] = 0,
    ):
        super().__init__()
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
            uncertainty_generator=self._uncertainty_generator,
            sfreq=sfreq,
            sine_freq=sine_freq,
        )

    def _sine_wave_function(self, time, sine_freq):
        """A simple sine wave generator"""
        amplitude = np.sin(2 * np.pi * sine_freq * time) + gauss(0, self.value_unc ** 2)
        return amplitude

    def _uncertainty_generator(self, _):
        """A simple uncertainty generator"""
        return self.time_unc ** 2, self.value_unc ** 2
