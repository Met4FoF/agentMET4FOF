import numpy as np

class DataStreamMET4FOF():
    """
    Class for creating finite datastream for ML with `x` as inputs and `y` as target
    Data can be fetched sequentially using `next_sample()` or all at once `all_samples()`

    For sensors data:
    The format shape for 2D data stream (num_samples, n_sensors)
    The format shape for 3D data stream (num_samples, sample_length , n_sensors)
    """

    def __init__(self):
        super().__init__()

    def randomize_data(self):
        random_index = np.arange(self.x.shape[0])
        np.random.shuffle(random_index)
        self.x = self.x[random_index]

        if type(self.y).__name__ == "ndarray" or type(self.y).__name__ == "list":
            self.y = self.y[random_index]
        elif type(self.y).__name__ == "DataFrame":
            self.y = self.y.iloc[random_index]

    def set_data_source(self, x,y):
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

        if x is None and y is None:
            self.x = list(np.arange(10))
            self.y = list(np.arange(10))
            self.y.reverse()
        else:
            self.x = x
            self.y = y

        if type(self.x).__name__ == "list":
            self.n_samples = len(self.x)
        elif type(self.x).__name__ == "DataFrame": #dataframe or numpy
            self.x = self.x.to_numpy()
            self.n_samples = self.x.shape[0]
        elif type(self.x).__name__ == "ndarray":
            self.n_samples = self.x.shape[0]

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
        Fetches the samples from the data stream and advances the internal pointer `current_idx`

        Parameters
        ----------
        batch_size : int
            number of batches to get from data stream

        Returns
        -------
        samples : dict of the form `{'x': current_sample_x, 'y': current_sample_y}`

        """
        if batch_size < 0:
            batch_size = self.x.shape[0]

        self.sample_idx += batch_size

        try:
            self.current_sample_x = self.x[self.sample_idx - batch_size:self.sample_idx]

            if self.y is not None:
                self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx]
            else:
                self.current_sample_y = None

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None

        return {'x': self.current_sample_x, 'y': self.current_sample_y}

    def reset(self):
        self.sample_idx = 0

    def has_more_samples(self):
        return self.sample_idx < self.n_samples

#Built-in classes with DataStreamMET4FOF
class SineGenerator(DataStreamMET4FOF):
    def __init__(self,num_cycles = 1000):
        x = np.sin(np.arange(0,3.142*num_cycles,0.5))
        self.set_data_source(x,y=None)

class CosineGenerator(DataStreamMET4FOF):
    def __init__(self,num_cycles = 1000):
        x = np.cos(np.arange(0,3.142*num_cycles,0.5))
        self.set_data_source(x,y=None)

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
        return -1
    return x, y

