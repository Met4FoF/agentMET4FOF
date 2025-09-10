import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base_streams import DataStreamMET4FOF

__all__ = ["MetrologicalDataStreamMET4FOF"]


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
        value_unc : float, optional (defaults to 0.0)
            standard uncertainties associated with values
        time_unc : float, optional (defaults to 0.0)
            standard uncertainties associated with timestamps
        exp_unc : float, optional
            expanded uncertainties associated with values. If ``exp_unc`` is given
            explicitly, it overrides ``value_unc`` according to ``value_unc = exp_unc
            / cov_factor``.
        cov_factor : float, optional (defaults to 1.0)
            coverage factor associated with the expanded uncertainty, only used,
            if ``exp_unc`` is specified
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
        **kwargs: Optional[Any],
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
                "value_unc and time_unc each constant)."
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

        _value: np.ndarray = self._generator_function(
            _time, **self._generator_parameters
        )
        _time_unc, _value_unc = self._generator_function_unc(_time, _value)

        return np.concatenate((_time, _time_unc, _value, _value_unc), axis=1)

    def _next_sample_data_source(
        self, batch_size: Optional[int] = 1
    ) -> np.ndarray:
        """Internal method for fetching latest samples from a dataset.
        Overrides :meth:`.DataStreamMET4FOF._next_sample_data_source`. Adds
        time uncertainty ``ut`` and measurement uncertainty ``uv`` to sample

        """
        if batch_size < 0:
            batch_size = self._quantities.shape[0]

        self._sample_idx += batch_size

        try:
            self._current_sample_quantities = self._quantities[
                self._sample_idx - batch_size : self._sample_idx
            ]

            # if target is available
            if self._target is not None:
                self._current_sample_target = self._target[
                    self._sample_idx - batch_size : self._sample_idx
                ]
            else:
                self._current_sample_target = None

            # if time is available
            if self._time is not None:
                self._current_sample_time = self._time[
                    self._sample_idx - batch_size : self._sample_idx
                ]
            else:
                self._current_sample_time = None
        except IndexError:
            self._current_sample_quantities = None
            self._current_sample_target = None
            self._current_sample_time = None

        _time_unc, _value_unc = (np.full_like(self._current_sample_time, fill_value=self.time_unc),
                                            np.full_like(self._current_sample_quantities, fill_value=self.value_unc))

        return np.concatenate((self._current_sample_time, _time_unc, self._current_sample_quantities, _value_unc), axis=1)

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
