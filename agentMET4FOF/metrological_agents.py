from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from time_series_buffer import TimeSeriesBuffer
from time_series_metadata.scheme import MetaData
from itertools import combinations
from scipy.special import comb
from scipy.stats import chi2

from .agents import AgentBuffer, AgentMET4FOF
from .metrological_streams import (
    MetrologicalDataStreamMET4FOF,
    MetrologicalSineGenerator,
)
from .exceptions import (
    ColumnNotZeroError,
    SystemMatrixNotReducibleError,
    SensorsNotLinearlyIndependentError,
)


class MetrologicalAgent(AgentMET4FOF):
    _input_data: Dict[str, Dict[str, Union[TimeSeriesBuffer, Dict]]]
    """Input dictionary of all incoming data including metadata::

        dict like {
            <from>: {
                "buffer": TimeSeriesBuffer(maxlen=buffer_size),
                "metadata": MetaData(**kwargs).metadata,
            }
    """
    _input_data_maxlen: int

    _output_data: Dict[str, Dict[str, Union[TimeSeriesBuffer, MetaData]]]
    """Output dictionary of all outgoing data including metadata::

        dict like {
            <from>: {
                "buffer": TimeSeriesBuffer(maxlen=buffer_size),
                "metadata": MetaData(**kwargs).metadata,
            }
    """
    _output_data_maxlen: int

    def init_parameters(self, input_data_maxlen=25, output_data_maxlen=25):
        super(MetrologicalAgent, self).init_parameters()
        self._input_data = {}
        self._input_data_maxlen = input_data_maxlen
        self._output_data = {}
        self._output_data_maxlen = output_data_maxlen

    def on_received_message(self, message):
        channel = message["channel"]
        sender = message["from"]

        if channel == "default":
            data = message["data"]
            metadata = None
            if "metadata" in message.keys():
                metadata = message["metadata"]

            self._set_input_data(sender, data, metadata)

    def _set_input_data(self, sender, data=None, metadata=None):
        # create storage for new senders
        if sender not in self._input_data.keys():
            self._input_data[sender] = {
                "metadata": metadata,
                "buffer": TimeSeriesBuffer(maxlen=self._input_data_maxlen),
            }

        if metadata is not None:
            # update received metadata
            self._input_data[sender]["metadata"] = metadata

        if data is not None:
            # append received data
            self._input_data[sender]["buffer"].add(data=data)

    def set_output_data(self, channel, data=None, metadata=None):
        # create storage for new output channels
        if channel not in self._output_data.keys():
            self._output_data[channel] = {
                "metadata": metadata,
                "buffer": TimeSeriesBuffer(maxlen=self._output_data_maxlen),
            }

        if metadata is not None:
            # update received metadata
            self._output_data[channel]["metadata"] = metadata

        if data is not None:
            # append received data
            self._output_data[channel]["buffer"].add(data=data)

    def agent_loop(self):
        if self.current_state == "Running":

            for channel, channel_dict in self._output_data.items():
                # short names
                metadata = channel_dict["metadata"]
                buffer = channel_dict["buffer"]

                # if there is something in the buffer, send it all
                buffer_len = len(buffer)
                if buffer_len > 0:
                    data = buffer.pop(n_samples=buffer_len)

                    # send data+metadata
                    self.send_output([data, metadata], channel=channel)

    def pack_data(self, data, channel="default"):

        # include metadata in the packed data
        packed_data = {
            "from": self.name,
            "data": data[0],
            "metadata": data[1],
            "senderType": type(self).__name__,
            "channel": channel,
        }

        return packed_data


class MetrologicalMonitorAgent(MetrologicalAgent):
    def init_parameters(self, *args, **kwargs):
        super(MetrologicalMonitorAgent, self).init_parameters(*args, **kwargs)
        # create alias/dummies to match dashboard expectations
        self.memory = self._input_data
        self.plot_filter = []
        self.plots = {}
        self.custom_plot_parameters = {}

    def on_received_message(self, message):
        """
        Handles incoming data from 'default' and 'plot' channels.

        Stores 'default' data into `self.memory` and 'plot' data into `self.plots`

        Parameters
        ----------
        message : dict
            Acceptable channel values are 'default' or 'plot'
        """
        if message["channel"] == "default":
            if self.plot_filter != []:
                message["data"] = {
                    key: message["data"][key] for key in self.plot_filter
                }
                message["metadata"] = {
                    key: message["metadata"][key] for key in self.plot_filter
                }
            self.buffer_store(
                agent_from=message["from"],
                data={"data": message["data"], "metadata": message["metadata"]},
            )
        elif message["channel"] == "plot":
            self.update_plot_memory(message)
        return 0

    def update_plot_memory(self, message):
        """
        Updates plot figures stored in `self.plots` with the received message

        Parameters
        ----------
        message : dict
            Standard message format specified by AgentMET4FOF class
            Message['data'] needs to be base64 image string and can be nested in dictionary for multiple plots
            Only the latest plot will be shown kept and does not keep a history of the plots.
        """

        if type(message["data"]) != dict or message["from"] not in self.plots.keys():
            self.plots[message["from"]] = message["data"]
        elif type(message["data"]) == dict:
            for key in message["data"].keys():
                self.plots[message["from"]].update({key: message["data"][key]})
        self.log_info("PLOTS: " + str(self.plots))

    def reset(self):
        super(MetrologicalMonitorAgent, self).reset()
        del self.plots
        self.plots = {}

    def custom_plot_function(self, data, sender_agent, **kwargs):
        # TODO: cannot set the label of the xaxis within this method
        # data display
        if "data" in data.keys():
            if len(data["data"]):
                # values = data["buffer"].show(n_samples=-1)  # -1 --> all
                values = data["data"]
                t = values[:, 0]
                ut = values[:, 1]
                v = values[:, 2]
                uv = values[:, 3]

                # use description
                desc = data["metadata"][0]
                t_name, t_unit = desc.time.values()
                v_name, v_unit = desc.get_quantity().values()

                x_label = f"{t_name} [{t_unit}]"
                y_label = f"{v_name} [{v_unit}]"

                trace = go.Scatter(
                    x=t,
                    y=v,
                    error_x=dict(type="data", array=ut, visible=True),
                    error_y=dict(type="data", array=uv, visible=True),
                    mode="lines",
                    name=f"{y_label} ({sender_agent})",
                )
            else:
                trace = go.Scatter()
        else:
            trace = go.Scatter()
        return trace


class MetrologicalAgentBuffer(AgentBuffer):
    """Buffer class which is instantiated in every metrological agent to store data

    This buffer is necessary to handle multiple inputs coming from agents.

    We can access the buffer like a dict with exposed functions such as .values(),
    .keys() and .items(). The actual dict object is stored in the attribute
    :attr:`buffer <agentMET4FOF.agents.AgentBuffer.buffer>`. The list in
    :attr:`supported_datatypes <agentMET4FOF.agents.AgentBuffer.supported_datatypes>`
    contains one more element
    for metrological agents, namely :class:`TimeSeriesBuffer
    <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>`.
    """

    def __init__(self, buffer_size: int = 1000):
        """Initialise a new agent buffer object

        Parameters
        ----------
        buffer_size: int
            Length of buffer allowed.
        """
        super(MetrologicalAgentBuffer, self).__init__(buffer_size)
        self.supported_datatypes.append(TimeSeriesBuffer)

    def convert_single_to_tsbuffer(self, single_data: Union[List, Tuple, np.ndarray]):
        """Convert common data in agentMET4FOF to :class:`TimeSeriesBuffer
        <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>`

        Parameters
        ----------
        single_data : iterable of iterables (list, tuple, np.ndarrray) with shape (N, M)

            * M==2 (pairs): assumed to be like (time, value)
            * M==3 (triple): assumed to be like (time, value, value_unc)
            * M==4 (4-tuple): assumed to be like (time, time_unc, value, value_unc)

        Returns
        -------
        TimeSeriesBuffer
            the new :class:`TimeSeriesBuffer
            <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>` object

        """
        ts = TimeSeriesBuffer(maxlen=self.buffer_size)
        ts.add(single_data)
        return ts

    def update(
        self,
        agent_from: str,
        data: Union[Dict, List, Tuple, np.ndarray],
    ) -> TimeSeriesBuffer:
        """Overrides data in the buffer dict keyed by `agent_from` with value `data`

        Parameters
        ----------
        agent_from : str
            Name of agent sender
        data : dict or iterable of iterables (list, tuple, np.ndarray) with shape (N, M
            the data to be stored in the metrological buffer

        Returns
        -------
        TimeSeriesBuffer
            the updated :class:`TimeSeriesBuffer
            <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>` object
        """
        # handle if data type nested in dict
        if isinstance(data, dict):
            # check for each value datatype
            for key, value in data.items():
                data[key] = self.convert_single_to_tsbuffer(value)
        else:
            data = self.convert_single_to_tsbuffer(data)
            self.buffer.update({agent_from: data})
        return self.buffer

    def _concatenate(
        self,
        iterable: TimeSeriesBuffer,
        data: Union[np.ndarray, list, pd.DataFrame],
        concat_axis: int = 0,
    ) -> TimeSeriesBuffer:
        """Concatenate the given ``TimeSeriesBuffer`` with ``data``

        Add ``data`` to the :class:`TimeSeriesBuffer
        <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>` object.

        Parameters
        ----------
        iterable : TimeSeriesBuffer
            The current buffer to be concatenated with.
        data : np.ndarray, DataFrame, list
            New incoming data

        Returns
        -------
        TimeSeriesBuffer
            the original buffer with the data appended
        """
        iterable.add(data)
        return iterable


class MetrologicalGeneratorAgent(MetrologicalAgent):
    """An agent streaming a specified signal

    Takes samples from an instance of :py:class:`MetrologicalDataStreamMET4FOF` with sampling frequency `sfreq` and
    signal frequency `sine_freq` and pushes them sample by sample to connected agents via its output channel.
    """

    # The datatype of the stream will be MetrologicalSineGenerator.
    _stream: MetrologicalDataStreamMET4FOF

    def init_parameters(
        self,
        signal: MetrologicalDataStreamMET4FOF = MetrologicalSineGenerator(),
        **kwargs,
    ):
        """Initialize the input data stream

        Parameters
        ----------
        signal : MetrologicalDataStreamMET4FOF (defaults to :py:class:`MetrologicalSineGenerator`)
            the underlying signal for the generator
        """
        self._stream = signal
        super().init_parameters()
        self.set_output_data(channel="default", metadata=self._stream.metadata)

    @property
    def device_id(self):
        return self._stream.metadata.metadata["device_id"]

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input
        datastream's content and push it into its output buffer.
        """
        if self.current_state == "Running":
            self.set_output_data(channel="default", data=self._stream.next_sample())
            super().agent_loop()


class RedundancyAgent(MetrologicalAgent):
    """
    This is the main Redundancy Agent class. Main calculation types are :py:func:`lcs` and :py:func:`lcss`, as defined
    in the module :mod:`redundancy1`.
    """

    def init_parameters(
        self,
        input_data_maxlen: int = 25,
        output_data_maxlen: int = 25,
        sensor_key_list: list = None,
        n_pr: int = 1,
        problim: float = 0.9,
        calc_type: str = "lcs",
    ):
        """
        Initialize the redundancy agent as an instance of the :py:mod:`MetrologicalAgent` class.


        Parameters
        ----------
        n_pr : int, optional
            size of the batch of data that is handled at a time by the Redundancy Agent. Defaults to 1
        problim : float, optional
            limit probability used for consistency evaluation. Defaults to .9
        calc_type : str, optional
            calculation type: 'lcs' or 'lcss'. Defaults to 'lcs'
        sensor_key_list : list of str
            list containing the names of the sensors that should feed data to the Redundancy Agent. Defaults to None

        Parent class parameters
        ----------
        input_data_maxlen: int

        output_data_maxlen: int

        """

        if sensor_key_list is None:
            sensor_key_list = []
        super().init_parameters(input_data_maxlen=25, output_data_maxlen=25)
        self.metadata = MetaData(
            device_id="RedAgent01",
            time_name="time",
            time_unit="s",
            quantity_names="m",
            quantity_units="kg",
            misc="nothing",
        )

        self.calc_type = calc_type
        self.sensor_key_list = sensor_key_list
        self.n_pr = n_pr
        self.problim = problim

        self.set_output_data(channel="default", metadata=self.metadata)

    def init_lcss_parameters(self, fsam, f1, f2, ampl_ratio, phi1, phi2):
        """
        Additional parameters used for this particular example in combination with the :py:func:`lcss` method.
        It provides the prior knowledge needed to make the information contained in the data redundant.
        This method sets up the vector **a** and matrix *A* for the system **y** = **a** + *A* * **x**.

        Parameters
        ----------
        fsam :   float
                sampling frequency
        f1 :     float
                first frequency of interest in signal
        f2 :    float
                second frequency of interest in signal
        ampl_ratio : float
                    ratio of the amplitudes of the two frequency components
        phi1 :   float
                initial phase of first frequency component
        phi2 :   float
                initial phase of second frequency component
        """
        # set-up vector a_arr and matrix a_arr2d for redundancy method
        id_mat = np.identity(self.n_pr)
        id_fft = np.fft.fft(id_mat)

        n_pr_floor = self.n_pr // 2
        bfft = id_fft[:n_pr_floor, :].real / n_pr_floor
        cfft = -id_fft[:n_pr_floor, :].imag / n_pr_floor

        c1 = 1 / np.cos(phi1)
        c2 = 1 / np.sin(-phi1)
        c3 = ampl_ratio / np.cos(phi2)
        c4 = ampl_ratio / np.sin(-phi2)

        ind_freq1 = f1 * self.n_pr // fsam
        ind_freq2 = f2 * self.n_pr // fsam

        a_row1 = c1 * bfft[ind_freq1, :].reshape((1, -1))
        a_row2 = c2 * cfft[ind_freq1, :].reshape((1, -1))
        a_row3 = c3 * bfft[ind_freq2, :].reshape((1, -1))
        a_row4 = c4 * cfft[ind_freq2, :].reshape((1, -1))

        self.a_arr2d = np.concatenate((a_row1, a_row2, a_row3, a_row4), axis=0)
        """a_arr : np.ndarray of float"""
        self.a_arr = np.zeros(shape=(4, 1))
        """a_arr2d : np.ndarray of float"""

    def agent_loop(self):
        """
        Model the agent's behaviour
        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking :py:func:`AgentMET4FOF.send_output`.
        """
        if self.current_state == "Running":
            # sometimes the buffer does not contain values for all sensors
            # sensor_key_list = ["Sensor1", "Sensor2"]
            key_list = [
                key for key in self.sensor_key_list if key in self.buffer.keys()
            ]
            n_sensors = len(key_list)
            if n_sensors != len(self.sensor_key_list):  # expected number of sensors
                print(
                    "Not all sensors were present in the buffer. Not evaluating the data."
                )
                return 0

            for key in key_list:
                if self.buffer[key].shape[0] < self.n_pr:
                    print(
                        "Buffer size is ",
                        self.buffer[key].shape[0],
                        ", which is less than ",
                        self.n_pr,
                        ".",
                    )
                    print("Not enough data for redundancy agent evaluation.")
                    return 0

            buff = self.buffer.popleft(
                self.n_pr
            )  # take n_pr entries out from the buffer

            t_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            ut_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            x_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            ux_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            i_sensor = 0
            # for key in buff.keys(): # arbitrary order

            for key in key_list:
                data_arr = buff[key]
                t_data_arr2d[:, i_sensor] = data_arr[:, 0]
                ut_data_arr2d[:, i_sensor] = data_arr[:, 1]
                x_data_arr2d[:, i_sensor] = data_arr[:, 2]
                ux_data_arr2d[:, i_sensor] = data_arr[:, 3]
                i_sensor = i_sensor + 1

            if self.calc_type == "lcs":
                data = np.full(shape=(self.n_pr, 4), fill_value=np.nan)
                for i_pnt in range(self.n_pr):
                    y_arr = np.array(x_data_arr2d[i_pnt, :])
                    y_arr = y_arr.reshape((n_sensors, 1))
                    vy_arr2d = np.zeros(shape=(n_sensors, n_sensors))
                    for i_sensor in range(n_sensors):
                        vy_arr2d[i_sensor, i_sensor] = np.square(
                            ux_data_arr2d[i_pnt, i_sensor]
                        )

                    n_sols, ybest, uybest, chi2obs, indkeep = self.calc_lcs(
                        y_arr, vy_arr2d, self.problim
                    )
                    if n_sols == 1:  # time stamp is value of first sensor
                        if isinstance(ybest, np.ndarray):
                            ybest = ybest[0]
                        data[i_pnt, :] = np.array(
                            [
                                t_data_arr2d[i_pnt, 0],
                                ut_data_arr2d[i_pnt, 0],
                                ybest,
                                uybest,
                            ]
                        )
                    else:  # only return the first solution
                        data[i_pnt, :] = np.array(
                            [
                                t_data_arr2d[i_pnt, 0],
                                ut_data_arr2d[i_pnt, 0],
                                ybest[0],
                                uybest[0],
                            ]
                        )
            elif self.calc_type == "lcss":
                # lcss applied to one data vector (required input)
                # Sum the signals to get one signal
                x_data_arr = np.sum(x_data_arr2d, axis=1)
                x_data_arr = x_data_arr.reshape((len(x_data_arr), 1))
                ux2_data_arr = np.sum(np.square(ux_data_arr2d), axis=1)
                vx_arr2d = np.zeros((self.n_pr, self.n_pr))
                for i_val in range(self.n_pr):
                    vx_arr2d[i_val, i_val] = ux2_data_arr[i_val]

                n_sols, ybest, uybest, chi2obs, indkeep = self.calc_lcss(
                    self.a_arr, self.a_arr2d, x_data_arr, vx_arr2d, self.problim
                )
                print("calc lcss finished")
                print("n_sols: ", n_sols)
                print("ybest: ", ybest)
                print("uybest: ", uybest)
                if n_sols == 1:  # time stamp is latest value
                    if isinstance(ybest, np.ndarray):
                        ybest = ybest[0]
                    data = np.array(
                        [t_data_arr2d[-1, 0], ut_data_arr2d[-1, 0], ybest, uybest]
                    )
                else:  # only return the first solution
                    data = np.array(
                        [t_data_arr2d[-1, 0], ut_data_arr2d[-1, 0], ybest[0], uybest[0]]
                    )

            # Send the data
            if len(data.shape) == 1:
                data = data.reshape((1, len(data)))

            self.set_output_data(channel="default", data=data)
            super().agent_loop()

    def calc_consistent_estimates_no_corr(y_arr2d, uy_arr2d, prob_lim):
        """
        Calculation of consistent estimate for n_sets of estimates y_ij (contained in
        y_arr2d) of a quantity Y, where each set contains n_estims estimates.
        The uncertainties are assumed to be independent and given in uy_arr2d.
        The consistency test is using limit probability limit prob_lim.
        For each set of estimates, the best estimate, uncertainty,
        observed chi-2 value and a flag if the
        provided estimates were consistent given the model are given as output.

        Parameters
        ----------
        y_arr2d:    np.ndarray of size (n_rows, n_estimates)
                    each row contains m=n_estimates independent estimates of a measurand
        uy_arr2d:   np.ndarray of size (n_rows, n_estimates)
                    each row contains the standard uncertainty u(y_ij) of y_ij = y_arr2d[i,j]
        prob_lim:   limit probability used in consistency test. Typically 0.95.

        Returns
        -------
        isconsist_arr:  bool array of shape (n_rows)
                        indicates for each row if the n_estimates are consistent or not
        ybest_arr:      np.ndarray of shape (n_rows)
                        contains the best estimate for each row of individual estimates
        uybest_arr:     np.ndarray of shape (n_rows)
                        contains the uncertainty associated with each best estimate for each row of *y_arr2d*
        chi2obs_arr:    observed chi-squared value for each row

        """

        if len(y_arr2d.shape) > 1:
            n_sets = y_arr2d.shape[0]
        else:
            n_sets = 1

        n_estims = y_arr2d.shape[-1]  # last dimension is number of estimates
        chi2_lim = chi2.ppf(1 - prob_lim, n_estims - 1)
        uy2inv_arr2d = 1 / np.power(uy_arr2d, 2)
        uy2best_arr = 1 / np.sum(uy2inv_arr2d, -1)
        uybest_arr = np.sqrt(uy2best_arr)
        ybest_arr = np.sum(y_arr2d * uy2inv_arr2d, -1) * uy2best_arr

        if n_sets > 1:
            ybest_arr = ybest_arr.reshape(
                n_sets, 1
            )  # make a column vector of ybest_arr

        chi2obs_arr = np.sum(
            np.power(
                (y_arr2d - np.broadcast_to(ybest_arr, (n_sets, n_estims))) / uy_arr2d, 2
            ),
            -1,
        )
        isconsist_arr = chi2obs_arr <= chi2_lim

        return isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr

    def print_output_single(self, isconsist, ybest, uybest, chi2obs):
        """
        Function to print the output of a single row of the calculate_best_estimate function.

        Parameters
        ----------
        isconsist:  bool
                    Indicates if provided estimates were consistent
        ybest:      float
                    best estimate
        uybest:     float
                    uncertainty of best estimate
        chi2obs:    float
                    observed value of chi-squared
        """
        print("\tThe observed chi-2 value is %3.3f." % chi2obs)

        if isconsist:
            print("\tThe provided estimates (input) were consistent.")
        else:
            print("\tThe provided estimates (input) were not consistent.")

        print(f"\tThe best estimate is {ybest:3.3f} with uncertainty {uybest:3.3f}.\n")

    def print_output_cbe(self, isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr):
        """
        Function to print the full output of calc_best_estimate.

        Parameters
        ----------
        isconsist_arr:  bool array of shape (n_rows)
                        indicates for each row if the n_estimates are consistent or not
        ybest_arr:      np.ndarray of shape (n_rows)
                        contains the best estimate for each row of individual estimates
        uybest_arr:     np.ndarray of shape (n_rows)
                        contains the uncertainty associated with each best estimate for each row of *y_arr2d*
        chi2obs_arr:    observed chi-squared value for each row

        Returns
        -------

        """
        if len(ybest_arr.shape) == 0:
            self.print_output_single(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr)
        else:
            n_sets = ybest_arr.shape[0]
            print(f"There are {n_sets:.0f} sets with estimates of the measurand.")
            for i_set in range(n_sets):
                print(f"The result of set {i_set:.0f} is:")
                self.print_output_single(
                    isconsist_arr.item(i_set),
                    ybest_arr.item(i_set),
                    uybest_arr.item(i_set),
                    chi2obs_arr.item(i_set),
                )

    def calc_best_estimate(self, y_arr, vy_arr2d, problim):
        """Calculate the best estimate for a set of estimates with associated uncertainty matrix,
        and determine if the set of estimates are consistent using a provided limit probability.

        Parameters
        ----------
        y_arr:      np.ndarray of shape (n)
                    vector of estimates of a measurand Y
        vy_arr2d:   np.ndarray of shape (n, n)
                    uncertainty matrix associated with y_arr
        problim:    float
                    probability limit used for assessing the consistency of the estimates. Typically, problim equals 0.95.

        Returns
        -------
        isconsist:  bool
                    indicator whether provided estimates are consistent in view of *problim*
        ybest:      float
                    best estimate of measurand
        uybest:     float
                    uncertainty associated with *ybest*
        chi2obs:    float
                    observed value of chi-squared, used for consistency evaluation
        """

        print(f"cbe y_arr = {y_arr}")
        n_estims = len(y_arr)

        if n_estims == 1:
            isconsist = True
            ybest = y_arr[0]
            uybest = np.sqrt(vy_arr2d[0, 0])
            chi2obs = 0.0
        else:
            e_arr = np.ones(n_estims)
            vyinve_arr = np.linalg.solve(vy_arr2d, e_arr)
            uy2 = 1 / np.dot(e_arr, vyinve_arr)
            uybest = np.sqrt(uy2)
            ybest = np.dot(vyinve_arr, y_arr) * uy2
            yred_arr = y_arr - ybest
            chi2obs = np.dot(
                yred_arr.transpose(), np.linalg.solve(vy_arr2d, yred_arr)
            )  # check need for transpose
            chi2lim = chi2.ppf(problim, n_estims - 1)
            isconsist = chi2obs <= chi2lim

        return isconsist, ybest, uybest, chi2obs

    def calc_best_estimate(self, y_arr, vy_arr2d, problim):
        """Calculate the best estimate for a set of estimates with associated uncertainty matrix,
        and determine if the set of estimates are consistent using a provided limit probability.

        Parameters
        ----------
        y_arr:      np.ndarray of shape (n)
                    vector of estimates of a measurand Y
        vy_arr2d:   np.ndarray of shape (n, n)
                    uncertainty matrix associated with y_arr
        problim:    float
                    probability limit used for assessing the consistency of the estimates. Typically, problim equals 0.95.

        Returns
        -------
        isconsist:  bool
                    indicator whether provided estimates are consistent in view of *problim*
        ybest:      float
                    best estimate of measurand
        uybest:     float
                    uncertainty associated with *ybest*
        chi2obs:    float
                    observed value of chi-squared, used for consistency evaluation
        """

        print(f"cbe y_arr = {y_arr}")
        n_estims = len(y_arr)

        if n_estims == 1:
            isconsist = True
            ybest = y_arr[0]
            uybest = np.sqrt(vy_arr2d[0, 0])
            chi2obs = 0.0
        else:
            e_arr = np.ones(n_estims)
            vyinve_arr = np.linalg.solve(vy_arr2d, e_arr)
            uy2 = 1 / np.dot(e_arr, vyinve_arr)
            uybest = np.sqrt(uy2)
            ybest = np.dot(vyinve_arr, y_arr) * uy2
            yred_arr = y_arr - ybest
            chi2obs = np.dot(
                yred_arr.transpose(), np.linalg.solve(vy_arr2d, yred_arr)
            )  # check need for transpose
            chi2lim = chi2.ppf(problim, n_estims - 1)
            isconsist = chi2obs <= chi2lim

        return isconsist, ybest, uybest, chi2obs

    def get_combination(self, val_arr, n_keep, indcomb):
        subsets = combinations(val_arr, n_keep)
        i_subset = -1

        for subset in subsets:
            i_subset += 1
            if i_subset == indcomb:
                return np.array(list(subset))

    def calc_lcs(self, y_arr, vy_arr2d, problim):
        """
        Function to calculate the best estimate of a measurand based on individual estimates of the
        measurand with associated uncertainty matrix.

        Parameters
        ----------
        y_arr:      np.ndarray of shape (n)
                    vector with estimates of the measurand
        vy_arr2d:   np.ndarray of shape (n, n)
                    uncertainty matrix of the vector y_arr
        problim:    float
                    limit probability used in the consistency evaluation. Typically 0.95.
        """
        isconsist, ybest, uybest, chi2obs = self.calc_best_estimate(
            y_arr, vy_arr2d, problim
        )
        n_estims = len(y_arr)
        estim_arr = np.arange(n_estims)
        n_remove = 0

        if isconsist:  # set the other return variables
            n_sols = 1
            indkeep = estim_arr

        while not isconsist:
            n_remove += 1
            subsets = combinations(estim_arr, n_estims - n_remove)
            n_subsets = comb(n_estims, n_remove, exact=True)
            isconsist_arr = np.full(n_subsets, np.nan)
            ybest_arr = np.full(n_subsets, np.nan)
            uybest_arr = np.full(n_subsets, np.nan)
            chi2obs_arr = np.full(n_subsets, np.nan)
            i_subset = -1

            for subset in subsets:
                i_subset += 1
                sublist = list(subset)
                yred_arr = y_arr[sublist]
                vyred_arr2d = vy_arr2d[np.ix_(sublist, sublist)]
                (
                    isconsist_arr[i_subset],
                    ybest_arr[i_subset],
                    uybest_arr[i_subset],
                    chi2obs_arr[i_subset],
                ) = self.calc_best_estimate(yred_arr, vyred_arr2d, problim)

            # Find smallest chi2obs value amongst all subsets. If multiple possibilities exist, return them all
            indmin = np.argmin(chi2obs_arr)

            if isconsist_arr[indmin]:
                # consistent solution found (otherwise isconsist remains false and the while loop continues)
                isconsist = True
                chi2obs = chi2obs_arr[indmin]  # minimum chi2obs value
                indmin = np.where(chi2obs_arr == chi2obs)[
                    0
                ]  # list with all indices with minimum chi2obs value
                n_sols = len(indmin)

                if n_sols == 1:
                    ybest = ybest_arr[indmin[0]]
                    uybest = uybest_arr[indmin[0]]
                    indkeep = self.get_combination(
                        estim_arr, n_estims - n_remove, indmin
                    )  # indices of kept estimates
                else:  # multiple solutions exist, the return types become arrays
                    ybest = np.full(n_sols, np.nan)
                    uybest = np.full(n_sols, np.nan)
                    indkeep = np.full((n_sols, n_estims - n_remove), np.nan)

                    for i_sol in range(n_sols):
                        ybest[i_sol] = ybest_arr[indmin[i_sol]]
                        uybest[i_sol] = uybest_arr[indmin[i_sol]]
                        indkeep[i_sol] = self.get_combination(
                            estim_arr, n_estims - n_remove, indmin[i_sol]
                        )

        return n_sols, ybest, uybest, chi2obs, indkeep

    def on_received_message(self, message):
        """
        Handles incoming data from 'default' channels.
        Stores 'default' data into an internal buffer

        Parameters
        ----------
        message : dict
             Only acceptable channel value is 'default'.
        """
        if message["channel"] == "default":
            self.buffer_store(agent_from=message["from"], data=message["data"])
        return 0

    def print_output_lcs(self, n_sols, ybest, uybest, chi2obs, indkeep, y_arr):
        """
        Method to print the output of the method :func:`calc_lcs`.

        Parameters
        ----------
        n_sols:     int
                    number of best solutions
        ybest:      float or np.ndarray of shape (n_sols)
                    best estimate or vector of best estimates
        uybest:     float or np.ndarray of shape (n_sols)
                    standard uncertainty of best estimate or vector with standard uncertainty of best estimates
        chi2obs:    float
                    observed chi-squared value of all best solutions
        indkeep:    np.ndarary of shape (n) or (n_sols, n)
                    indices of retained estimates of y_arr for the calculation of the best estimate ybest
        y_arr:      np.ndarray of shape (n)
                    individual estimates of measurand
        """
        n_estims = len(y_arr)
        n_keep = indkeep.shape[
            -1
        ]  # number of retained estimates in the best solution(s)

        if n_sols == 1:
            print(
                f"calc_lcs found a unique solution with chi2obs = {chi2obs:4.4f} using {n_keep:.0f} of the provided {n_estims:.0f} estimates."
            )
            print(f"\ty = {ybest:4.4f}, u(y) = {uybest:4.4f}")
            print(f"\tIndices and values of retained provided estimates:", end=" ")

            for ind in indkeep[:-1]:
                indint = int(ind)
                print(f"y[{indint:.0f}]= {y_arr[indint]:2.2f}", end=", ")

            indint = int(indkeep[-1])
            print(f"y[{indint:.0f}]= {y_arr[indint]:2.2f}.\n")
        else:
            print(
                f"calc_lcs found {n_sols:.0f} equally good solutions with chi2obs = {chi2obs:4.4f} using {n_keep:.0f} of the provided {n_estims:.0f} estimates."
            )

            for i_sol in range(n_sols):
                print(f"\tSolution {i_sol:.0f} is:")
                print(f"\ty = {ybest[i_sol]:4.4f}, u(y) = {uybest[i_sol]:4.4f}")
                print("\tIndices and values of retained provided estimates:", end=" ")

                for ind in indkeep[i_sol][:-1]:
                    indint = int(ind)
                    print("y[%d]= %2.2f" % (indint, y_arr[indint]), end=", ")

                indint = int(indkeep[i_sol][-1])
                print("y[%d]= %2.2f.\n" % (indint, y_arr[indint]))

        return

    # Function that returns the index of a row of A that can be written as a linear combination of the others.
    # This row does not contribute any new information to the system.
    def ind_reduce_a(self, a_arr2d, epszero):
        if a_arr2d.shape[0] <= np.linalg.matrix_rank(a_arr2d):
            raise SystemMatrixNotReducibleError("A cannot be reduced!")
        # Remove one row from A that is a linear combination of the other rows.
        # Find a solution of A' * b = 0.
        u, s, vh = np.linalg.svd(np.transpose(a_arr2d))
        # singVals = diag(S)%;
        b = vh[-1, :]
        indrem = np.where(abs(b) > epszero)[
            0
        ]  # remove a row corresponding to a non-zero entry in b.

        if len(indrem) == 0:
            raise ValueError("b is a zero vector!")

        indrem = indrem[-1]  # return the last row that can be taken out
        # print('ReduceA: Identified row %d to be removed from a and A.\n', indRem);
        return indrem

    # Reduced the system if matrix Vx is not of full rank.
    # This might be ambiguous, as constant sensor values or offsets have to be estimated and are not known.
    def reduce_vx(self, x_arr, vx_arr2d, a_arr, a_arr2d, epszero):
        if vx_arr2d.shape[0] <= np.linalg.matrix_rank(vx_arr2d, epszero):
            print("Vx cannot be reduced any further!")
            return
        # Remove one sensor from Vx, A and x that is a linear combination of the other sensors.
        # Find a solution of Vx * b = 0. This
        u, s, vh = np.linalg.svd(vx_arr2d)
        b = vh[-1, :]  # bottom row of vh is orthogonal to Vx

        if abs(np.dot(b, x_arr)) > epszero:
            raise SensorsNotLinearlyIndependentError(
                "Sensors in x should be linearly independent with b^T * x = 0, but this is not the case!"
            )

        indrem = np.where(abs(b) > epszero)[
            0
        ]  # remove a sensor corresponding to a non-zero entry in b.
        if len(indrem) == 0:
            raise ValueError("b is the zero vector!")

        indrem = indrem[-1]  # take out the last sensor
        indsenskeep = np.concatenate(
            np.arange(indrem), np.arange(indrem, vx_arr2d.shape[0])
        )
        vxred_arr2d = vx_arr2d[indsenskeep, indsenskeep]
        xred_arr = x_arr(indsenskeep)
        # Update A by removing the sensor and updating the system of equations

        ared_arr2d = a_arr2d - a_arr2d[:, indrem] / b[indrem] * np.transpose(b)
        if max(abs(ared_arr2d[:, indrem])) > epszero:
            print(ared_arr2d)
            raise ColumnNotZeroError(f"Column {indrem} should be zero by now!")

        ared_arr2d = a_arr2d[:, indsenskeep]  # remove the zero column from A
        ared_arr = a_arr + np.dot(
            a_arr2d - ared_arr2d, x_arr
        )  # adapt vector a_arr such that the vector of estimates y = a + A*x remains the same

        return xred_arr, vxred_arr2d, ared_arr, ared_arr2d

    def calc_best_est_lin_sys(self, a_arr, a_arr2d, x_arr, vx_arr2d, problim):
        """
        Function to calculate the best estimate of a linear system **y** = **a** + A * **x**
        and determines if the inputs are consistent in view of *problim*.

        Parameters
        ----------
        a_arr:      np.ndarray of shape (n_estimates)
                    vector **a** of linear system **y** = **a** + A * **x**
        a_arr2d:    np.ndarray of shape (n_estimates, n_sensors)
                    matrix A of linear system **y** = **a** + A * **x**
        x_arr:      np.ndarray of shape (n_sensors)
                    vector with sensor values
                    vector **x** of linear system **y** = **a** + A * **x**
        vx_arr2d:   np.ndarray of shape (n_sensors, n_sensors)
                    uncertainty matrix associated with vector x_arr
        problim:    float
                    probability limit used for consistency evaluation. Typically 0.95.

        Returns
        -------
        isconsist:  bool
                    indicator whether provided estimates are consistent in view of *problim*
        ybest:      float
                    best estimate
        uybest:     float
                    standard uncertainty of best estimate
        chi2obs:    float
                    observed chi-squared value
        """
        print("start calc_best_est_lin_sys")
        epszero = 1e-10  # some small constant used for some checks

        # The main procedure only works when vy_arr2d has full rank. Therefore first a_arr, a_arr2d and vx_arr2d need to be
        # reduced such that vy_arr2d will have full rank.
        xred_arr = x_arr
        vxred_arr2d = vx_arr2d
        ared_arr = a_arr
        ared_arr2d = a_arr2d

        #  print('cbels: x_arr = ', x_arr)
        #  print('cbels: a_arr2d = ', a_arr2d)

        # Reduce the system if the covariance matrix vx_arr2d is rank deficient.
        while np.linalg.matrix_rank(vxred_arr2d) < vxred_arr2d.shape[0]:
            print(
                "Reducing Vx. No of rows = ",
                vxred_arr2d.shape[0],
                ", rank = ",
                np.linalg.matrix_rank(vxred_arr2d),
            )
            [xred_arr, vxred_arr2d, ared_arr, ared_arr2d] = self.reduce_vx(
                xred_arr, vxred_arr2d, ared_arr, ared_arr2d, epszero
            )

        # Reduce the system if a_arr2d has more rows than its rank.
        while ared_arr2d.shape[0] > np.linalg.matrix_rank(ared_arr2d):
            print(
                "Reducing A. No of rows = ",
                ared_arr2d.shape[0],
                ", rank = ",
                np.linalg.matrix_rank(ared_arr2d),
            )
            print(f"ared_arr2d: {ared_arr2d}")
            ind_rem = self.ind_reduce_a(ared_arr2d, epszero)
            n_rows = ared_arr2d.shape[0]
            indrowskeep = np.concatenate(
                (np.arange(0, ind_rem), np.arange(ind_rem + 1, n_rows))
            )
            ared_arr = ared_arr[indrowskeep]
            ared_arr2d = ared_arr2d[
                indrowskeep,
            ]

        # calculate y vector and Vy matrix
        print("ared_arr2d: ", ared_arr2d)
        print("ared_arr: ", ared_arr.shape)
        print("ared_arr2d: ", ared_arr2d.shape)
        print("xred_arr: ", xred_arr.shape)
        y_arr = ared_arr + np.dot(ared_arr2d, xred_arr)
        vy_arr2d = np.matmul(
            np.matmul(ared_arr2d, vxred_arr2d), np.transpose(ared_arr2d)
        )

        # try to calculate a consistent solution with these y and Vy.
        isconsist, ybest, uybest, chi2obs = self.calc_best_estimate(
            y_arr, vy_arr2d, problim
        )
        # print('y_arr = ', y_arr)
        # print('chi2obs = ', chi2obs)
        return isconsist, ybest, uybest, chi2obs

    # function to calculate lcss
    def calc_lcss(self, a_arr, a_arr2d, x_arr, vx_arr2d, problim):
        """
        Calculation of the largest consistent subset of sensor values and the implied best estimate.

        Parameters
        ----------
        x_arr
        vx_arr2d
        a_arr
        a_arr2d
        problim
        a_arr:      np.ndarray of shape (n_estimates)
                    vector **a** of linear system **y** = **a** + A * **x**
        a_arr2d:    np.ndarray of shape (n_estimates, n_sensors)
                    matrix A of linear system **y** = **a** + A * **x**
        x_arr:      np.ndarray of shape (n_sensors)
                    vector with sensor values
                    vector **x** of linear system **y** = **a** + A * **x**
        vx_arr2d:   np.ndarray of shape (n_sensors, n_sensors)
                    uncertainty matrix associated with vector x_arr
        problim:    float
                    probability limit used for consistency evaluation. Typically 0.95.

        Returns
        -------
        isconsist:  bool
                    indicator whether provided estimates are consistent in view of *problim*
        ybest:      float
                    best estimate
        uybest:     float
                    standard uncertainty of best estimate
        chi2obs:    float
                    observed chi-squared value
        Returns
        -------

        """
        print("start calc_lcss")
        epszero = 1e-7  # epsilon for rank check
        eps_chi2 = 1e-7  # epsilon for chi2 equivalence check

        isconsist, ybest, uybest, chi2obs = self.calc_best_est_lin_sys(
            a_arr, a_arr2d, x_arr, vx_arr2d, problim
        )
        n_sens = len(x_arr)
        sens_arr = np.arange(n_sens)
        n_remove = 0
        if isconsist:  # set the other return variables
            n_sols = 1
            indkeep = sens_arr

        # no consistent solution, remove sensors 1 by 1, 2 by 2, etc.
        while not isconsist:
            n_remove += 1
            subsets = combinations(sens_arr, n_sens - n_remove)
            n_subsets = comb(n_sens, n_remove, exact=True)
            isconsist_arr = np.full(n_subsets, np.nan)
            ybest_arr = np.full(n_subsets, np.nan)
            uybest_arr = np.full(n_subsets, np.nan)
            chi2obs_arr = np.full(n_subsets, np.nan)
            i_subset = -1
            for subset in subsets:
                i_subset += 1
                sublistsenskeep = list(subset)
                xred_arr = x_arr[sublistsenskeep]
                vxred_arr2d = vx_arr2d[:, sublistsenskeep]
                vxred_arr2d = vxred_arr2d[sublistsenskeep, :]
                boolremove_arr = np.full(n_sens, True)
                boolremove_arr[sublistsenskeep] = False
                if (
                    np.linalg.matrix_rank(
                        np.concatenate(
                            (
                                a_arr2d[:, boolremove_arr],
                                np.ones((a_arr2d.shape[0], 1)),
                            ),
                            axis=1,
                        ),
                        epszero,
                    )
                    == np.linalg.matrix_rank(a_arr2d[:, boolremove_arr], epszero)
                ):
                    # there is no vector c such that c' * ones = 1 and c' * ai = 0 at the same time.
                    # Thus this combination of sensors cannot be removed as a group from the matrix A.
                    isconsist_arr[i_subset] = False
                    continue  # continue with next subset

                ared_arr2d = np.concatenate(
                    (a_arr2d[:, boolremove_arr], a_arr2d[:, sublistsenskeep]), axis=1
                )  # move the columns corresponding to sensors to be taken out to the front
                q, r = np.linalg.qr(ared_arr2d)
                q1 = q[
                    :, n_remove:
                ]  # these (n_sens-n_remove) columns are orthogonal to the first n_remove columns of ared_arr2d
                s = np.sum(
                    q1, axis=0
                )  # column sums might be zero which is a problem for normalization to unit sum
                indzero = np.where(np.abs(s) < epszero)[0]
                if len(indzero) > 0:
                    indnonzero = np.full(n_sens - n_remove, True)  # all True array
                    indnonzero[indzero] = False  # positions that are zero are false
                    indnonzero = np.where(
                        indnonzero
                    )  # conversion to indices instead of boolean array
                    if len(indnonzero) == 0:
                        print("ERROR: All columns have zero sum!")
                    b = q1[
                        :, indnonzero[0]
                    ]  # b is column vector with no zero column sum
                    for i_zero in range(len(indzero)):
                        q1[:, indzero[i_zero]] = (
                            q1[:, indzero[i_zero]] + b
                        )  # add b to prevent zero column sum
                q1 = q1 / np.sum(
                    q1, axis=0
                )  # unit column sums, in order not to introduce a bias in the estimate of the measurand

                ared_arr2d = np.matmul(np.transpose(q1), ared_arr2d)
                ared_arr = np.matmul(np.transpose(q1), a_arr)
                # The columns of matrix ared_arr2d are still in the wrong order compared to the order of the sensors
                ared2_arr2d = np.full_like(ared_arr2d, np.nan)
                ared2_arr2d[:, boolremove_arr] = ared_arr2d[
                    :, :n_remove
                ]  # columns 0, 1, ..., (n_remove-1)
                ared2_arr2d[:, np.invert(boolremove_arr)] = ared_arr2d[
                    :, n_remove:
                ]  # columns n_remove, ..., (n_sens-1)
                ared_arr2d = ared2_arr2d

                if np.linalg.norm(ared_arr2d[:, boolremove_arr]) > epszero:
                    raise ColumnNotZeroError(
                        f"These columns of A should be zero by now!"
                    )

                ared_arr2d = ared_arr2d[
                    :, sublistsenskeep
                ]  # np.invert(boolremove_arr)] # reduce the matrix A by removing the appropriate columns of A, which are zero anyway.

                (
                    isconsist_arr[i_subset],
                    ybest_arr[i_subset],
                    uybest_arr[i_subset],
                    chi2obs_arr[i_subset],
                ) = self.calc_best_est_lin_sys(
                    ared_arr, ared_arr2d, xred_arr, vxred_arr2d, problim
                )

            # After analyzing all subset, find the smallest chi2obs value amongst all subsets.
            # If multiple possibilities exist, return them all
            indmin = np.argmin(chi2obs_arr)
            if isconsist_arr[indmin]:
                # consistent solution found (otherwise isconsist remains false and the while loop continues)
                isconsist = True
                chi2obs = chi2obs_arr[indmin]  # minimum chi2obs value
                indmin = np.where(np.abs(chi2obs_arr - chi2obs) < eps_chi2)[
                    0
                ]  # list with all indices with minimum chi2obs value
                n_sols = len(indmin)
                if n_sols == 1:
                    ybest = ybest_arr[indmin[0]]
                    uybest = uybest_arr[indmin[0]]
                    indkeep = self.get_combination(
                        sens_arr, n_sens - n_remove, indmin
                    )  # indices of kept estimates
                else:  # multiple solutions exist, the return types become arrays
                    ybest = np.full(n_sols, np.nan)
                    uybest = np.full(n_sols, np.nan)
                    indkeep = np.full((n_sols, n_sens - n_remove), np.nan)
                    for i_sol in range(n_sols):
                        ybest[i_sol] = ybest_arr[indmin[i_sol]]
                        uybest[i_sol] = uybest_arr[indmin[i_sol]]
                        indkeep[i_sol] = self.get_combination(
                            sens_arr, n_sens - n_remove, indmin[i_sol]
                        )
        return n_sols, ybest, uybest, chi2obs, indkeep

    def print_input_lcss(self, x_arr, vx_arr2d, a_arr, a_arr2d, problim):
        print(
            f"""INPUT of lcss function call:
        Vector a of linear system: a_arr = {a_arr}
        Matrix A of linear system: a_arr2d = {a_arr2d}
        Vector x with sensor values: x_arr = {x_arr}
        Covariance matrix Vx of sensor values: vx_arr2d = {vx_arr2d}
        Limit probability for chi-squared test: p = {problim}"""
        )

    def print_output_lcss(
        self, n_sols, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d
    ):
        n_sensors = len(x_arr)
        n_eq = a_arr2d.shape[0]
        n_keep = indkeep.shape[
            -1
        ]  # number of retained estimates in the best solution(s)
        print(
            f"Provided number of sensors (or sensor values) was {n_sensors} and number of equations was {n_eq}."
        )
        if n_sols == 1:
            print(
                "calc_lcss found a unique solution with chi2obs = %4.4f using %d of the provided %d sensor values."
                % (chi2obs, n_keep, n_sensors)
            )
            print("\ty = %4.4f, u(y) = %4.4f" % (ybest, uybest))
            print("\tIndices and values of retained provided sensor values:", end=" ")
            for ind in indkeep[:-1]:
                indint = int(ind)
                print("x[%d]= %2.2f" % (indint, x_arr[indint]), end=", ")
            indint = int(indkeep[-1])
            print("x[%d]= %2.2f.\n" % (indint, x_arr[indint]))
        else:
            print(
                "calc_lcss found %d equally good solutions with chi2obs = %4.4f using %d of the provided %d sensor values."
                % (n_sols, chi2obs, n_keep, n_eq)
            )
            for i_sol in range(n_sols):
                print("\tSolution %d is:" % i_sol)
                print("\ty = %4.4f, u(y) = %4.4f" % (ybest[i_sol], uybest[i_sol]))
                print(
                    "\tIndices and values of retained provided sensor values:", end=" "
                )
                for ind in indkeep[i_sol][:-1]:
                    indint = int(ind)
                    print("x[%d]= %2.2f" % (indint, x_arr[indint]), end=", ")
                indint = int(indkeep[i_sol][-1])
                print("x[%d]= %2.2f.\n" % (indint, x_arr[indint]))
        return
