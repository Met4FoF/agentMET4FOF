"""
This module defines a Redundancy Agent that can be used in the agentMET4FOF framework. It has two main data processing
types:
- lcs: best estimate calculation using Largest Consistent Subset method
- lcss: best estimate calculation using Largest Consistent Subset of Sensor values method

"""
from typing import Dict

import numpy as np
from agentMET4FOF.metrological_agents import MetrologicalAgent
from time_series_metadata.scheme import MetaData

from agentMET4FOF.metrological_streams import (
    MetrologicalDataStreamMET4FOF,
    MetrologicalMultiWaveGenerator,
)

from .redundancy1 import calc_lcs, calc_lcss


class MetrologicalMultiWaveGeneratorAgent(MetrologicalAgent):
    """An agent streaming a signal composed of various sine and cosine components.
    Takes samples from the :py:mod:`MultiWaveGenerator` and pushes them sample by sample (or in batches)
    to connected agents via its output channel.
    """

    # The datatype of the stream will be MultiWaveGenerator.
    _data_stream: MetrologicalDataStreamMET4FOF
    batch_size = 100

    def init_parameters(
        self,
        signal: MetrologicalDataStreamMET4FOF = MetrologicalMultiWaveGenerator(),
        **kwargs
    ):
        """
        Initialize the input data stream as an instance of the :py:mod:`MultiWaveGenerator` class

        Parameters
        ----------
        signal : MetrologicalDataStreamMET4FOF
            the underlying signal for the generator
        """
        self._data_stream = signal
        super().init_parameters()
        self.set_output_data(channel="default", metadata=self._data_stream.metadata)

    def agent_loop(self):
        """Model the agent's behaviour
        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:func:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            self.set_output_data(channel="default", data=self._data_stream._next_sample_generator(batch_size=10))
            super().agent_loop()

    @property
    def metadata(self) -> Dict:
        return self._data_stream.metadata.metadata


class RedundancyAgent(MetrologicalAgent):
    """
    This is the main Redundancy Agent class. Main calculation types are :py:func:`lcs` and :py:func:`lcss`, as defined
    in the module :mod:`redundancy1`.
    """
    metadata: MetaData
    n_pr: int
    problim: float
    calc_type: str
    sensor_key_list: list
    a_arr: np.ndarray
    a_arr2d: np.ndarray

    def init_parameters(self, input_data_maxlen=25, output_data_maxlen=25):
        """
        Initialize the redundancy agent as an instance of the :py:mod:`MetrologicalAgent` class.

        Parameters
        ----------
        input_data_maxlen: int

        output_data_maxlen: int
        """
        self.metadata = MetaData(
            device_id="RedAgent01",
            time_name="time",
            time_unit="s",
            quantity_names="m",
            quantity_units="kg",
            misc="nothing")

        super().init_parameters(input_data_maxlen=25, output_data_maxlen=25)
        self.set_output_data(channel="default", metadata=self.metadata)

    def init_parameters1(self, calc_type, sensor_key_list, n_pr, problim):
        """
        Parameters used for both methods :func:`lcs` and :func:`lcss`.

        Parameters
        ----------
        calc_type: str
                   calculation type: 'lcs' or 'lcss'
        sensor_key_list: list of strings
                        list containing the names of the sensors that should feed data to the Redundancy Agent
        n_pr:   integer
                size of the batch of data that is handled at once by the Redundancy Agent
        problim: float
                 limit probability used for conistency evaluation
        """
        self.calc_type = calc_type
        self.sensor_key_list = sensor_key_list
        self.n_pr = n_pr
        self.problim = problim

    def init_parameters2(self, fsam, f1, f2, ampl_ratio, phi1, phi2):
        """
        Additional parameters used for this particular example in combination with the :py:func:`lcss` method.
        It provides the prior knowledge needed to make the information contained in the data redundant.
        This method sets up the vector **a** and matrix *A* for the system **y** = **a** + *A* * **x**.

        Parameters
        ----------
        fsam:   float
                sampling frequency
        f1:     float
                first frequency of interest in signal
        f2 :    float
                second frequency of interest in signal
        ampl_ratio: float
                    ratio of the amplitudes of the two frequency components
        phi1:   float
                initial phase of first frequency component
        phi2:   float
                initial phase of second frequency component
        """
        # set-up vector a_arr and matrix a_arr2d for redundancy method
        a = np.identity(self.n_pr)
        n_pr2 = int(self.n_pr / 2)
        afft = np.fft.fft(a)
        bfft = np.real(afft[:n_pr2, :]) / n_pr2
        cfft = -np.imag(afft[:n_pr2, :]) / n_pr2
        c1 = 1 / np.cos(phi1)
        c2 = 1 / np.sin(-phi1)
        c3 = ampl_ratio / np.cos(phi2)
        c4 = ampl_ratio / np.sin(-phi2)

        t_max = self.n_pr / fsam
        df = 1 / t_max

        ind_freq1 = int(f1/df)
        ind_freq2 = int(f2/df)

        a_row1 = c1 * bfft[ind_freq1, :]
        a_row2 = c2 * cfft[ind_freq1, :]
        a_row3 = c3 * bfft[ind_freq2, :]
        a_row4 = c4 * cfft[ind_freq2, :]

        a_row1 = a_row1.reshape((1, len(a_row1)))
        a_row2 = a_row2.reshape((1, len(a_row2)))
        a_row3 = a_row3.reshape((1, len(a_row3)))
        a_row4 = a_row4.reshape((1, len(a_row4)))
        self.a_arr2d = np.concatenate((a_row1, a_row2, a_row3, a_row4), axis=0)
        self.a_arr = np.zeros(shape=(4, 1))

    def agent_loop(self):
        """
        Model the agent's behaviour
        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:func:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            # sometimes the buffer does not contain values for all sensors
            # sensor_key_list = ["Sensor1", "Sensor2"]
            key_list = [key for key in self.sensor_key_list if key in self.buffer.keys()]
            n_sensors = len(key_list)
            if n_sensors != len(self.sensor_key_list):  # expected number of sensors
                print('Not all sensors were present in the buffer. Not evaluating the data.')
                return 0

            for key in key_list:
                if self.buffer[key].shape[0] < self.n_pr:
                    print('Buffer size is ', self.buffer[key].shape[0], ', which is less than ', self.n_pr, '.')
                    print('Not enough data for redundancy agent evaluation.')
                    return 0

            buff = self.buffer.popleft(self.n_pr)  # take n_pr entries out from the buffer

            t_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            ut_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            x_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            ux_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            # print('buff = ', buff)
            i_sensor = 0
            # for key in buff.keys(): # arbitrary order

            for key in key_list:
                data_arr = buff[key]
                t_data_arr2d[:, i_sensor] = data_arr[:, 0]
                ut_data_arr2d[:, i_sensor] = data_arr[:, 1]
                x_data_arr2d[:, i_sensor] = data_arr[:, 2]
                ux_data_arr2d[:, i_sensor] = data_arr[:, 3]
                i_sensor = i_sensor + 1

            #print('calc_type: ', self.calc_type)
            if self.calc_type == "lcs":
                #print('case lcs')
                data = np.full(shape=(self.n_pr, 4), fill_value=np.nan)
                for i_pnt in range(self.n_pr):
                    y_arr = np.array(x_data_arr2d[i_pnt, :])
                    y_arr = y_arr.reshape((n_sensors, 1))
                    vy_arr2d = np.zeros(shape=(n_sensors, n_sensors))
                    for i_sensor in range(n_sensors):
                        vy_arr2d[i_sensor, i_sensor] = np.square(ux_data_arr2d[i_pnt, i_sensor])
                    #data = np.array([1, 2, 3, 4])
                    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, self.problim)
                    if n_sols == 1:  # time stamp is value of first sensor
                        if isinstance(ybest, np.ndarray):
                            ybest = ybest[0]
                        data[i_pnt, :] = np.array([t_data_arr2d[i_pnt, 0], ut_data_arr2d[i_pnt, 0], ybest, uybest])
                    else:  # only return the first solution
                        data[i_pnt, :] = np.array([t_data_arr2d[i_pnt, 0], ut_data_arr2d[i_pnt, 0], ybest[0], uybest[0]])

            elif self.calc_type == "lcss":
                # lcss applied to one data vector (required input)
                # Sum the signals to get one signal
                x_data_arr = np.sum(x_data_arr2d, axis=1)
                x_data_arr = x_data_arr.reshape((len(x_data_arr), 1))
                ux2_data_arr = np.sum(np.square(ux_data_arr2d), axis=1)
                vx_arr2d = np.zeros((self.n_pr, self.n_pr))
                for i_val in range(self.n_pr):
                    vx_arr2d[i_val, i_val] = ux2_data_arr[i_val]

                n_sols, ybest, uybest, chi2obs, indkeep = \
                    calc_lcss(self.a_arr, self.a_arr2d, x_data_arr, vx_arr2d, self.problim)
                print('calc lcss finished')
                print('n_sols: ', n_sols)
                print('ybest: ', ybest)
                print('uybest: ', uybest)
                if n_sols == 1:  # time stamp is latest value
                    if isinstance(ybest, np.ndarray):
                        ybest = ybest[0]
                    data = np.array([t_data_arr2d[-1, 0], ut_data_arr2d[-1, 0], ybest, uybest])
                else:  # only return the first solution
                    data = np.array([t_data_arr2d[-1, 0], ut_data_arr2d[-1, 0], ybest[0], uybest[0]])

            # Send the data
            if len(data.shape) == 1:
                data = data.reshape((1, len(data)))

            # print('data = ', data)
            self.set_output_data(channel="default", data=data)
            super().agent_loop()

    def on_received_message(self, message):
        """
        Handles incoming data from 'default' channels.
        Stores 'default' data into an internal buffer

        Parameters
        ----------
        message : dict
             Only acceptable channel value is 'default'.
        """
        if message['channel'] == 'default':
            self.buffer_store(agent_from=message["from"], data=message["data"])
        return 0
