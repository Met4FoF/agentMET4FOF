import numpy as np
from time_series_metadata.scheme import MetaData
from scipy import signal

class DataStreamMET4FOF():
    """
    Abstract class for creating datastream.

    Data can be fetched sequentially using `next_sample()` or all at once
    `all_samples()`. This increments the internal sample index `sample_idx`.

    For sensors data, we assume:
    The format shape for 2D data stream (timesteps, n_sensors)
    The format shape for 3D data stream (num_cycles, timesteps , n_sensors)

    To create a new DataStreamMET4FOF class, inherit this class and call
    `set_metadata` in the constructor. Choose one of two types of datastreams to be
    created: from dataset file (`set_data_source`), or a waveform generator function
    (`set_generator_function`). Alternatively, override the `next_sample` function if
    neither option suits the application. For generator functions, `sfreq` is a
    required variable to be set on `init` which sets the sampling frequency and the
    time-step which occurs when `next_sample()` is called.

    For an example implementation of using generator function, see the built-in
    `SineGenerator` class. See tutorials for more implementations.
    """

    def __init__(self):
        super().__init__()
        self.quantities = None
        self.target = None
        self.time = None
        self.current_sample_quantities = None
        self.current_sample_target = None
        self.current_sample_time = None
        self.sample_idx = 0 #current sample index
        self.n_samples = 0 #total number of samples
        self.data_source_type = "function"
        self.generator_parameters = {}
        self.sfreq = 1

    def set_data_source_type(self, dt_type="function"):
        """
        To explicitly account for the type of data source: either from dataset,
        or a generator function.

        Parameters
        ----------
        dt_type : str
            Either "function" or "dataset"
        """
        self.data_source_type  = dt_type

    def randomize_data(self):
        random_index = np.arange(self.quantities.shape[0])
        np.random.shuffle(random_index)
        self.quantities = self.quantities[random_index]

        if type(self.target).__name__ == "ndarray" or type(self.target).__name__ == "list":
            self.target = self.target[random_index]
        elif type(self.target).__name__ == "DataFrame":
            self.target = self.target.iloc[random_index]

    def set_metadata(self, device_id, time_name, time_unit, quantity_names, quantity_units, misc):
        self.metadata = MetaData(
            device_id=device_id,
            time_name=time_name,
            time_unit=time_unit,
            quantity_names=quantity_names,
            quantity_units=quantity_units,
            misc=misc
        )

    def default_generator_function(self, time):
        amplitude = np.sin(2*np.pi*self.F*time)
        return amplitude

    def set_generator_function(self, generator_function=None, sfreq=None, **kwargs):
        """
        Sets the data source to a generator function. By default, this function resorts
        to a sine wave generator function. Initialisation of the generator's
        parameters should be done here such as setting the sampling frequency and
        wave frequency. For setting it with a dataset instead, see `set_data_source`.

        Parameters
        ----------
        generator_function : method
            A generator function which takes in at least one argument `time` which
            will be used in `next_sample`. Parameters of the function can be fixed by
            providing additional arguments such as the wave frequency.

        sfreq : int
            Sampling frequency.

        **kwargs
            Any additional keyword arguments to be supplied to the generator function.
            The ``**kwargs`` will be saved as `generator_parameters`.
            The generator function call for every sample will be supplied with the
            ``**generator_parameters``.

        """
        #save the kwargs into generator_parameters
        self.generator_parameters = kwargs

        if sfreq is not None:
            self.sfreq = sfreq
        self.set_data_source_type("function")

        #resort to default wave generator if one is not supplied
        if generator_function is None:
            self.F = 50
            self.generator_function = self.default_generator_function
        else:
            self.generator_function = generator_function
        return self.generator_function

    def _next_sample_generator(self, batch_size=1):
        """
        Internal method for generating a batch of samples from the generator function.
        """
        time = np.arange(self.sample_idx, self.sample_idx+batch_size, 1)/self.sfreq
        self.sample_idx += batch_size

        amplitude = self.generator_function(time, **self.generator_parameters)

        return {'quantities':amplitude, 'time':time}

    def set_data_source(self, quantities=None, target=None, time=None):
        """
        This sets the data source by providing three iterables: `quantities` ,
        `time` and `target` which are assumed to be aligned.

        For sensors data, we assume:
        The format shape for 2D data stream (timesteps, n_sensors)
        The format shape for 3D data stream (num_cycles, timesteps , n_sensors)

        Parameters
        ----------
        quantities : iterable
            Measured quantities such as sensors readings.

        target : iterable
            (Optional) Target label in the context of machine learning. This can be
            Remaining Useful Life in predictive maintenance application. Note this
            can be an unobservable variable in real-time and applies only for
            validation during offline analysis.

        time : iterable
            (Optional) dtype can be either float or datetime64 to indicate the time
            when the `quantities` were measured.

        """
        self.sample_idx = 0
        self.current_sample_quantities = None
        self.current_sample_target = None
        self.current_sample_time = None

        if quantities is None and target is None:
            self.quantities = list(np.arange(10))
            self.target = list(np.arange(10))
            self.time = list(np.arange(10))
            self.target.reverse()
        else:
            self.quantities = quantities
            self.target = target
            self.time = time

        #infer number of samples
        if type(self.quantities).__name__ == "list":
            self.n_samples = len(self.quantities)
        elif type(self.quantities).__name__ == "DataFrame": #dataframe or numpy
            self.quantities = self.quantities.to_numpy()
            self.n_samples = self.quantities.shape[0]
        elif type(self.quantities).__name__ == "ndarray":
            self.n_samples = self.quantities.shape[0]
        self.set_data_source_type("dataset")

    def prepare_for_use(self):
        self.reset()

    def all_samples(self):
        """
        Returns all the samples in the data stream

        Returns
        -------
        samples : dict of the form `{'x': current_sample_x, 'y': current_sample_y}`

        """
        return self.next_sample(-1)

    def next_sample(self, batch_size=1):
        """
        Fetches the latest `batch_size` samples from the iterables: quantities,
        time and target. This advances the internal pointer `current_idx` by
        `batch_size`.

        Parameters
        ----------
        batch_size : int
            number of batches to get from data stream

        Returns
        -------
        samples : dict of the form `{'time':current_sample_time,'quantities':
        current_sample_quantities, 'target': current_sample_target}`
        """

        if self.data_source_type == 'function':
            return self._next_sample_generator(batch_size)
        elif self.data_source_type == 'dataset':
            return self._next_sample_data_source(batch_size)

    def _next_sample_data_source(self, batch_size=1):
        """
        Internal method for fetching latest samples from a dataset.

        Parameters
        ----------
        batch_size : int
            number of batches to get from data stream

        Returns
        -------
        samples : dict of the form `{'quantities': current_sample_quantities,
        'target': current_sample_target}`

        """
        if batch_size < 0:
            batch_size = self.quantities.shape[0]

        self.sample_idx += batch_size

        try:
            self.current_sample_quantities = self.quantities[self.sample_idx - batch_size:self.sample_idx]

            #if target is available
            if self.target is not None:
                self.current_sample_target = self.target[self.sample_idx - batch_size:self.sample_idx]
            else:
                self.current_sample_target = None

            #if time is available
            if self.time is not None:
                self.current_sample_time = self.time[self.sample_idx - batch_size:self.sample_idx]
            else:
                self.current_sample_time = None
        except IndexError:
            self.current_sample_quantities = None
            self.current_sample_target = None
            self.current_sample_time = None

        return {'time':self.current_sample_time,'quantities': self.current_sample_quantities, 'target': self.current_sample_target}

    def reset(self):
        self.sample_idx = 0

    def has_more_samples(self):
        return self.sample_idx < self.n_samples

#Built-in classes with DataStreamMET4FOF
class SineGenerator(DataStreamMET4FOF):
    """
    Built-in class of sine wave generator.
    `sfreq` is sampling frequency which determines the time step when next_sample is called
    `F` is frequency of wave function
    `sine_wave_function` is a custom defined function which has a required keyword
    `time` as argument and any number of optional additional arguments (e.g `F`).
    to be supplied to the `set_generator_function`

    """
    def __init__(self,sfreq = 500, F=5):
        super().__init__()
        self.set_metadata("SineGenerator","time","s",("Voltage"),("V"),"Simple sine wave generator")
        self.set_generator_function(generator_function=self.sine_wave_function, sfreq=sfreq, F=F)

    def sine_wave_function(self, time, F=50):
        amplitude = np.sin(2*np.pi*F*time)
        return amplitude

class CosineGenerator(DataStreamMET4FOF):
    """
    Built-in class of cosine wave generator.
    `sfreq` is sampling frequency which determines the time step when next_sample is
    called `F` is frequency of wave function `cosine_wave_function` is a custom
    defined function which has a required keyword `time` as argument and any number
    of optional additional arguments (e.g `F`).to be supplied to the
    `set_generator_function`

    """
    def __init__(self,sfreq = 500, F=5):
        super().__init__()
        self.set_metadata("CosineGenerator","time","s",("Voltage"),("V"),"Simple cosine wave generator")
        self.set_generator_function(generator_function=self.cosine_wave_function, sfreq=sfreq, F=F)

    def cosine_wave_function(self, time, F=50):
        amplitude = np.cos(2*np.pi*F*time)
        return amplitude

def extract_x_y(message):
    """
    Extracts features & target from `message['data']` with expected structure such as :
        1. tuple - (x,y)
        2. dict - {'x':x_data,'y':y_data}

    Handle data structures of dictionary to extract features & target
    """
    if type(message['data']) == tuple:
        x = message['data'][0]
        y = message['data'][1]
    elif type(message['data']) == dict:
        x = message['data']['x']
        y = message['data']['y']
    else:
        return 1
    return x, y

