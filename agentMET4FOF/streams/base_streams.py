import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from time_series_metadata.scheme import MetaData

__all__ = ["DataStreamMET4FOF", "MetrologicalDataStreamMET4FOF"]


class DataStreamMET4FOF:
    """
    Abstract class for creating datastreams.

    Data can be fetched sequentially using :func:`next_sample` or all at once
    :func:`all_samples`. This increments the internal sample index :attr:`_sample_idx`.

    For sensors data, we assume:

    - The format shape for 2D data stream (timesteps, n_sensors)
    - The format shape for 3D data stream (num_cycles, timesteps , n_sensors)

    To create a new DataStreamMET4FOF class, inherit this class and call
    :func:`set_metadata` in the constructor. Choose one of two types of datastreams
    to be created:

    - from dataset file (:func:`set_data_source`), or
    - a waveform generator function (:func:`set_generator_function`).

    Alternatively, override the :func:`next_sample` function if neither option suits
    the application. For generator functions, :attr:`sfreq` is a required variable to
    be set on `init` which sets the sampling frequency and the time-step which occurs
    when :func:`next_sample` is called.

    For an example implementation of using generator function, see the built-in
    :class:`SineGenerator` class. See tutorials for more implementations.

    Attributes
    ----------
    _quantities : Union[List, DataFrame, np.ndarray]
        Measured quantities such as sensors readings
    _target : Union[List, DataFrame, np.ndarray]
        Target label in the context of machine learning. This can be
        Remaining Useful Life in predictive maintenance application. Note this
        can be an unobservable variable in real-time and applies only for
        validation during offline analysis.
    _time : Union[List, DataFrame, np.ndarray]
        ``dtype`` can be either ``float`` or ``datetime64`` to indicate the time
        when the :attr:`_quantities` were measured.
    _current_sample_quantities : Union[List, DataFrame, np.ndarray]
        Last returned measured quantities from a call to :func:`next_sample`
    _current_sample_target : Union[List, DataFrame, np.ndarray]
        Last returned target labels from a call to :func:`next_sample`
    _current_sample_time : Union[List, DataFrame, np.ndarray]
        ``dtype`` can be either ``float`` or ``datetime64`` to indicate the time
        when the :attr:`_current_sample_quantities` were measured.
    _sample_idx : int
        Current sample index
    _n_samples : int
        Total number of samples
    _data_source_type : str
        Explicitly account for the data source type: either "function" or "dataset"
    _generator_function : Callable
        A generator function which takes in at least one argument ``time`` which will
        be used in :func:`next_sample`
    _generator_parameters : Dict
        Any additional keyword arguments to be supplied to the generator function.
        The generator function call for every sample will be supplied with the
        ``**generator_parameters``.
    sfreq : int
        Sampling frequency
    _metadata : MetaData
        The quantities metadata as :class:`time_series_metadata.scheme.MetaData`
    """

    def __init__(self):
        """Initialize a DataStreamMet4FoF object"""
        super().__init__()
        self._quantities: Union[List, DataFrame, np.ndarray]
        self._target: Union[List, DataFrame, np.ndarray]
        self._time: Union[List, DataFrame, np.ndarray]
        self._current_sample_quantities: Union[List, DataFrame, np.ndarray]
        self._current_sample_target: Union[List, DataFrame, np.ndarray]
        self._current_sample_time: Union[List, DataFrame, np.ndarray]
        self._sample_idx: int = 0  # current sample index
        self._n_samples: int = 0  # total number of samples
        self._data_source_type: str = "function"
        self._generator_function: Callable
        self._generator_parameters: Dict = {}
        self.sfreq: int = 1
        self._metadata: MetaData

    def _set_data_source_type(self, dt_type: str = "function"):
        """
        To explicitly account for the type of data source: either from dataset,
        or a generator function.

        Parameters
        ----------
        dt_type : str
            Either "function" or "dataset"
        """
        self._data_source_type = dt_type

    def randomize_data(self):
        """Randomizes the provided quantities, useful in machine learning contexts"""
        random_index = np.arange(self._quantities.shape[0])
        np.random.shuffle(random_index)
        self._quantities = self._quantities[random_index]

        if (
            type(self._target).__name__ == "ndarray"
            or type(self._target).__name__ == "list"
        ):
            self._target = self._target[random_index]
        elif type(self._target).__name__ == "DataFrame":
            self._target = self._target.iloc[random_index]

    @property
    def metadata(self):
        return self._metadata

    @property
    def sample_idx(self):
        return self._sample_idx

    def set_metadata(
        self,
        device_id: str,
        time_name: str,
        time_unit: str,
        quantity_names: Union[str, Tuple[str, ...]],
        quantity_units: Union[str, Tuple[str, ...]],
        misc: Optional[Any] = None,
    ):
        """Set the quantities metadata as a ``MetaData`` object

        Details you find in the :class:`time_series_metadata.scheme.MetaData`
        documentation.

        Parameters
        ----------
        device_id : str
            Name of the represented generator
        time_name : str
            Name for the time dimension
        time_unit : str
            Unit for the time
        quantity_names : iterable of str or str
            A string or an iterable of names of the represented quantities' values
        quantity_units : iterable of str or str
            An iterable of units for the quantities' values
        misc : Any, optional
            This parameter can take any additional metadata which will be handed over to
            the corresponding attribute of the created :class:`Metadata` object
        """
        self._metadata = MetaData(
            device_id=device_id,
            time_name=time_name,
            time_unit=time_unit,
            quantity_names=quantity_names,
            quantity_units=quantity_units,
            misc=misc,
        )

    def _default_generator_function(self, time):
        """This is the default generator function used, if non was specified

        Parameters
        ----------
        time : Union[List, DataFrame, np.ndarray]
        """
        value = np.sin(2 * np.pi * self.F * time)
        return value

    def set_generator_function(
        self, generator_function: Callable = None, sfreq: int = None, **kwargs: Any
    ):
        """
        Sets the data source to a generator function. By default, this function resorts
        to a sine wave generator function. Initialisation of the generator's
        parameters should be done here such as setting the sampling frequency and
        wave frequency. For setting it with a dataset instead,
        see :func:`set_data_source`.

        Parameters
        ----------
        generator_function : Callable
            A generator function which takes in at least one argument ``time`` which
            will be used in :func:`next_sample`. Parameters of the function can be
            fixed by providing additional arguments such as the wave frequency.
        sfreq : int
            Sampling frequency.
        **kwargs : Any
            Any additional keyword arguments to be supplied to the generator function.
            The ``**kwargs`` will be saved as :attr:`_generator_parameters`.
            The generator function call for every sample will be supplied with the
            ``**generator_parameters``.
        """
        # save the kwargs into generator_parameters
        self._generator_parameters = kwargs

        if sfreq is not None:
            self.sfreq = sfreq
        self._set_data_source_type("function")

        # resort to default wave generator if one is not supplied
        if generator_function is None:
            warnings.warn(
                "No uncertainty generator function specified. Setting to default ("
                "sine wave)."
            )
            self.F = 50
            self._generator_function = self._default_generator_function
        else:
            self._generator_function = generator_function
        return self._generator_function

    def _next_sample_generator(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        """
        Internal method for generating a batch of samples from the generator function.
        """
        time: np.ndarray = (
            np.arange(self._sample_idx, self._sample_idx + batch_size, 1) / self.sfreq
        )
        self._sample_idx += batch_size

        value: np.ndarray = self._generator_function(time, **self._generator_parameters)

        return {"quantities": value, "time": time}

    def set_data_source(
        self,
        quantities: Union[List, DataFrame, np.ndarray] = None,
        target: Optional[Union[List, DataFrame, np.ndarray]] = None,
        time: Optional[Union[List, DataFrame, np.ndarray]] = None,
    ):
        """
        This sets the data source by providing up to three iterables: ``quantities`` ,
        ``time`` and ``target`` which are assumed to be aligned.

        For sensors data, we assume:
        The format shape for 2D data stream (timesteps, n_sensors)
        The format shape for 3D data stream (num_cycles, timesteps , n_sensors)

        Parameters
        ----------
        quantities : Union[List, DataFrame, np.ndarray]
            Measured quantities such as sensors readings.

        target : Optional[Union[List, DataFrame, np.ndarray]]
            Target label in the context of machine learning. This can be
            Remaining Useful Life in predictive maintenance application. Note this
            can be an unobservable variable in real-time and applies only for
            validation during offline analysis.

        time : Optional[Union[List, DataFrame, np.ndarray]]
            ``dtype`` can be either ``float`` or ``datetime64`` to indicate the time
            when the ``quantities`` were measured.

        """
        self._sample_idx = 0
        self._current_sample_quantities = None
        self._current_sample_target = None
        self._current_sample_time = None

        if quantities is None and target is None:
            self._quantities = list(np.arange(10))
            self._target = list(np.arange(10))
            self._time = list(np.arange(10))
            self._target.reverse()
        else:
            self._quantities = quantities
            self._target = target
            self._time = time

        # infer number of samples
        if type(self._quantities).__name__ == "list":
            self._n_samples = len(self._quantities)
        elif type(self._quantities).__name__ == "DataFrame":  # dataframe or numpy
            self._quantities = self._quantities.to_numpy()
            self._n_samples = self._quantities.shape[0]
        elif type(self._quantities).__name__ == "ndarray":
            self._n_samples = self._quantities.shape[0]
        self._set_data_source_type("dataset")

    def prepare_for_use(self):
        self.reset()

    def all_samples(self) -> Dict[str, Union[List, DataFrame, np.ndarray]]:
        """
        Returns all the samples in the data stream

        Returns
        -------
        samples : Dict
            ``{'x': current_sample_x, 'y': current_sample_y}``

        """
        return self.next_sample(-1)

    def next_sample(self, batch_size: int = 1):
        """
        Fetches the latest ``batch_size`` samples from the iterables: ``quantities``,
        ``time`` and ``target``. This advances the internal pointer ``_sample_idx`` by
        ``batch_size``.

        Parameters
        ----------
        batch_size : int
            number of batches to get from data stream

        Returns
        -------
        samples : Dict
            ``{'time':current_sample_time, 'quantities':current_sample_quantities,
            'target':current_sample_target}``
        """

        if self._data_source_type == "function":
            return self._next_sample_generator(batch_size)
        elif self._data_source_type == "dataset":
            return self._next_sample_data_source(batch_size)

    def _next_sample_data_source(
        self, batch_size: int = 1
    ) -> Dict[str, Union[List, DataFrame, np.ndarray]]:
        """
        Internal method for fetching latest samples from a dataset.

        Parameters
        ----------
        batch_size : int
            number of batches to get from data stream

        Returns
        -------
        samples : Dict
            ``{'quantities':current_sample_quantities, 'target':current_sample_target}``

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

        return {
            "time": self._current_sample_time,
            "quantities": self._current_sample_quantities,
            "target": self._current_sample_target,
        }

    def reset(self):
        self._sample_idx = 0

    def has_more_samples(self):
        return self._sample_idx < self._n_samples


# Built-in classes with DataStreamMET4FOF


def extract_x_y(message):
    """
    Extracts features & target from ``message['data']`` with expected structure such as:

    1. tuple - (x,y)
    2. dict - {'x':x_data,'y':y_data}

    Handle data structures of dictionary to extract features & target
    """
    if type(message["data"]) == tuple:
        x = message["data"][0]
        y = message["data"][1]
    elif type(message["data"]) == dict:
        x = message["data"]["x"]
        y = message["data"]["y"]
    else:
        return 1
    return x, y


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

        _value: np.ndarray = self._generator_function(
            _time, **self._generator_parameters
        )
        _time_unc, _value_unc = self._generator_function_unc(_time, _value)

        return np.concatenate((_time, _time_unc, _value, _value_unc), axis=1)

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
