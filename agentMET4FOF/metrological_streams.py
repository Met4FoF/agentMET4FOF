import numpy as np
from .streams import DataStreamMET4FOF
import time

class MetrologicalDataStreamMET4FOF(DataStreamMET4FOF):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _time():
        return time.time()

    @staticmethod
    def _time_unc():
        return time.get_clock_info("time").resolution

    @staticmethod
    def _value(timestamp):
        return 1013.25 + 10 * np.sin(timestamp)

    @staticmethod
    def _value_unc():
        return 0.5

    @property
    def current_datapoint(self):
        t = self._time()
        ut = self._time_unc()
        v = self._value(t)
        uv = self._value_unc()

        return np.array((t, ut, v, uv))

    def _next_sample_generator(self, batch_size=1):
        """
        Internal method for generating a batch of samples from the generator function.
        """
        time = np.arange(self.sample_idx, self.sample_idx+batch_size, 1)/self.sfreq
        self.sample_idx += batch_size

        amplitude = self.generator_function(time, **self.generator_parameters)

        return np.array((self._time(), self._time_unc(), amplitude, self._value_unc()))

    # @property
    # def metadata(self) -> MetaData:
    #     return self._description

class MetrologicalSineGenerator(MetrologicalDataStreamMET4FOF):
    """
    Built-in class of sine wave generator.
    `sfreq` is sampling frequency which determines the time step when next_sample is called
    `F` is frequency of wave function
    `sine_wave_function` is a custom defined function which has a required keyword `time` as argument and any number of optional additional arguments (e.g `F`).
    to be supplied to the `set_generator_function`

    """
    def __init__(self,sfreq = 500, F=5):
        super().__init__()
        self.set_metadata("SineGenerator","time","s",("Voltage"),("V"),"Simple sine wave generator")
        self.set_generator_function(generator_function=self.sine_wave_function, sfreq=sfreq, F=F)

    def sine_wave_function(self, time, F=50):
        amplitude = np.sin(2*np.pi*F*time)
        return amplitude