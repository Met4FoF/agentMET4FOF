import numpy as np
from agentMET4FOF.streams import DataStreamMET4FOF


class MetrologicalDataStreamMET4FOF(DataStreamMET4FOF):
    """
    Abstract  class for creating datastreams with metrological information
    """

    def __init__(self):
        super().__init__()

    def set_uncertainty_generator(self, generator_function=None, **kwargs):
        """
        Sets the uncertainty based on a user-defined function. By default,
        this function resorts to a constant (zero) uncertainty. The function returns
        a tuple corresponding to the amplitude and time uncertainty at a given time

        Parameters
        ----------
        generator_function : method
            A generator function which takes in at least one argument `time` which
            will be used in `next_sample`.

        **kwargs
            Any additional keyword arguments to be supplied to the generator function.
            The ``**kwargs`` will be saved as `uncertainty_parameters`.
            The generator function call for every sample will be supplied with the
            ``**uncertainty_parameters``.

        """
        #save the kwargs into uncertainty_parameters
        self.uncertainty_parameters = kwargs

        #resort to default wave generator if one is not supplied
        if generator_function is None:
            self.generator_function_unc = self.default_uncertainty_generator
        else:
            self.generator_function_unc = generator_function
        return self.generator_function_unc

    def default_uncertainty_generator(self, time):
        """
        Default uncertainty generator function. Returns a tuple of constant (zero) time and amplitude uncertainties
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
        timeelement = np.arange(self.sample_idx, self.sample_idx + batch_size, 1) / self.sfreq
        self._time = timeelement.item()
        self.sample_idx += batch_size

        self._time_unc, self._value_unc = self.default_uncertainty_generator(self._time)
        amplitude = self.generator_function(self._time, **self.generator_parameters)

        return np.array((self._time, self._time_unc, amplitude.item(), self._value_unc))


class MetrologicalSineGenerator(MetrologicalDataStreamMET4FOF):
    """
    Built-in class of sine wave generator.
    `sfreq` is sampling frequency which determines the time step when next_sample is called
    `F` is frequency of wave function
    `sine_wave_function` is a custom defined function which has a required keyword `time` as argument and any number of optional additional arguments (e.g `F`).
    to be supplied to the `set_generator_function`

    """
    def __init__(self,sfreq=500, F=50, value_unc=0.0):
        super().__init__()
        self.set_metadata(device_id="SineGenerator", time_name="time", time_unit="s", quantity_names=("Voltage"), quantity_units=("V"), misc="Simple sine wave generator")
        self.set_generator_function(generator_function=self.sine_wave_function, sfreq=sfreq, F=F)

    def sine_wave_function(self, time, F=50):
        amplitude = np.sin(2*np.pi*F*time)
        return amplitude
    
