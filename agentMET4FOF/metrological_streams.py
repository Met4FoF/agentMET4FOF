import time
import numpy as np
from agentMET4FOF.streams import DataStreamMET4FOF
import Dict

class MetrologicalDataStreamMET4FOF(DataStreamMET4FOF):
    """
    Abstract  class for creating datastreams with metrological information
    """

    def __init__(self, value_unc=0, absolute_timestamps=False):
        if value_unc < 0:
            raise ValueError("Uncertainty must be non-negative")
        else:
            self._value_unc = value_unc
        self.absolute_timestamps = absolute_timestamps
        super().__init__()


    def _next_sample_generator(self, batch_size=1):
        """
        Internal method for generating a batch of samples from the generator function. Overrides
        _next_sample_generator() from DataStreamMET4FOF. Adds time uncertainty ut and measurement uncertainty
        uv to sample
        """
        if self.absolute_timestamps:
            self._time = time.time()
            self._time_unc = time.get_clock_info("time").resolution
            self.sample_idx += batch_size
        else:
            timeelement = np.arange(self.sample_idx, self.sample_idx + batch_size, 1) / self.sfreq
            self._time = timeelement.item()
            self._time_unc = 0.0
            self.sample_idx += batch_size

        amplitude = self.generator_function(self._time, **self.generator_parameters)

        #return {'quantities': amplitude, 'time': time}
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
        super().__init__(value_unc=value_unc, absolute_timestamps=False)
        self.set_metadata(device_id="SineGenerator", time_name="time", time_unit="s", quantity_names=("Voltage"), quantity_units=("V"), misc="Simple sine wave generator")
        self.set_generator_function(generator_function=self.sine_wave_function, sfreq=sfreq, F=F)

    def sine_wave_function(self, time, F=50):
        amplitude = np.sin(2*np.pi*F*time)
        return amplitude
    
