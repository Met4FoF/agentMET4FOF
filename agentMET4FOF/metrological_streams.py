import warnings
from typing import Any, Callable, Tuple

import numpy as np

from agentMET4FOF.streams import DataStreamMET4FOF


class MetrologicalDataStreamMET4FOF(DataStreamMET4FOF):
    """
    Abstract  class for creating datastreams with metrological information. Inherits
    from the DataStreamMET4FOF class

    To create a new MetrologicalDataStreamMET4FOF class, inherit this class and call
    `set_metadata` in the constructor. Choose one of two types of datastreams to be
    created: from dataset file (`set_data_source`), or a waveform generator function
    (`set_generator_function`). Alternatively, override the `next_sample` function if
    neither option suits the application. For generator functions, `sfreq` is a
    required variable to be set on `init` which sets the sampling frequency and the
    time-step which occurs when `next_sample()` is called.
    """

    def __init__(self):
        super().__init__()
        self.uncertainty_parameters: Any = None
        self.generator_function_unc: Callable = lambda x: (0, 0)

    def set_generator_function(
        self, generator_function=None, uncertainty_generator=None, sfreq=None, **kwargs
    ):
        """
        Set value and uncertainty generators based on user-defined functions. By
        default, this function resorts to a sine wave generator function and a
        constant (zero) uncertainty. Initialisation of the generator's parameters
        should be done here such as setting the sampling frequency and wave
        frequency. For setting it with a dataset instead,
        see :func:`set_data_source`. Overwrites the default
        :func:`DataStreamMET4FOF.set_generator_function` method in
        :class:`DataStreamMET4FOF`.

        Parameters
        ----------
        generator_function : callable
            A generator function which takes in at least one argument `time` which
            will be used in :func:`next_sample`.
        uncertainty_generator : callable
            An uncertainty generator function which takes in at least one argument
            `time` which will be used in :func:`next_sample`.
        sfreq : int
            Sampling frequency.
        **kwargs : any
            Any additional keyword arguments to be supplied to the generator function.
            The ``**kwargs`` will be saved as `uncertainty_parameters`.
            The generator function call for every sample will be supplied with the
            ``**uncertainty_parameters``.

        """
        # Call the set_generator_function from the parent class to set the generator
        # function.
        super().set_generator_function(
            generator_function=generator_function, sfreq=sfreq, **kwargs
        )

        self.uncertainty_parameters = kwargs

        # resort to default wave generator if one is not supplied
        if uncertainty_generator is None:
            warnings.warn(
                "No uncertainty generator function specified. Setting to default ("
                "zero)."
            )
            self.generator_function_unc = self.default_uncertainty_generator
        else:
            self.generator_function_unc = uncertainty_generator
        return self.generator_function_unc

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

    def _next_sample_generator(self, batch_size=1):
        """
        Internal method for generating a batch of samples from the generator function. Overrides
        _next_sample_generator() from DataStreamMET4FOF. Adds time uncertainty ut and measurement uncertainty
        uv to sample
        """
        timeelement = (
            np.arange(self.sample_idx, self.sample_idx + batch_size, 1) / self.sfreq
        )
        _time = timeelement.item()
        self.sample_idx += batch_size

        _time_unc, _value_unc = self.generator_function_unc(_time)
        amplitude = self.generator_function(_time, **self.generator_parameters)

        return np.array((_time, _time_unc, amplitude.item(), _value_unc))


class MetrologicalSineGenerator(MetrologicalDataStreamMET4FOF):
    """Built-in class of sine wave generator

    Parameters
    ----------
    sfreq : int, optional
        Sampling frequency which determines the time step when `next_sample` is
        called. Defaults to 500.
    F : int, optional
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
    """

    def __init__(
        self,
        sfreq=500,
        F=50,
        device_id="SineGenerator",
        time_name="time",
        time_unit="s",
        quantity_names=("Voltage"),
        quantity_units=("V"),
        misc="Simple sine wave generator",
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
        self.set_generator_function(
            generator_function=self.sine_wave_function, sfreq=sfreq, F=F
        )

    def sine_wave_function(self, time, F=50):
        """A simple sine wave generator"""
        amplitude = np.sin(2 * np.pi * F * time)
        return amplitude
