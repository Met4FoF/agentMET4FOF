from skmultiflow.data.base_stream import Stream
import numpy as np

class DataStreamMET4FOF(Stream):
    def __init__(self, x=None, y=None):
        super().__init__()
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
        return self.next_sample(-1)

    def next_sample(self, batch_size=1):
        if batch_size < 0:
            batch_size = self.x.shape[0]

        self.sample_idx += batch_size

        if self.sample_idx > self.x.shape[0]:
            self.sample_idx = self.x.shape[0]

        try:
            self.current_sample_x = self.x[self.sample_idx - batch_size:self.sample_idx]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx]

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None

        return {'x': self.current_sample_x, 'y': self.current_sample_y}

    def reset(self):
        self.sample_idx = 0

    def has_more_samples(self):
        return self.sample_idx < self.n_samples
