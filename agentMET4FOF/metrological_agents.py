from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from time_series_buffer import TimeSeriesBuffer
from time_series_metadata.scheme import MetaData

from agentMET4FOF.agents import AgentBuffer, AgentMET4FOF
from .metrological_streams import (
    MetrologicalDataStreamMET4FOF,
    MetrologicalSineGenerator,
)

class MetrologicalAgent(AgentMET4FOF):
    # dict like {
    #     <from>: {
    #         "buffer": TimeSeriesBuffer(maxlen=buffer_size),
    #         "metadata": MetaData(**kwargs).metadata,
    #     }
    _input_data: Dict[str, Dict[str, Union[TimeSeriesBuffer, Dict]]]
    """Input dictionary of all incoming data including metadata::

        dict like {
            <from>: {
                "buffer": TimeSeriesBuffer(maxlen=buffer_size),
                "metadata": MetaData(**kwargs).metadata,
            }
    """
    _input_data_maxlen: int

    # dict like {
    #     <channel> : {
    #         "buffer" : TimeSeriesBuffer(maxlen=buffer_size),
    #         "metadata" : MetaData(**kwargs)
    #     }
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
        if message['channel'] == 'default':
            if self.plot_filter != []:
                message['data'] = {key: message['data'][key] for key in self.plot_filter}
                message['metadata'] = {key: message['metadata'][key] for key in self.plot_filter}
            self.buffer_store(agent_from=message["from"], data={"data": message["data"], "metadata": message["metadata"]})
        elif message['channel'] == 'plot':
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

        if type(message['data']) != dict or message['from'] not in self.plots.keys():
            self.plots[message['from']] = message['data']
        elif type(message['data']) == dict:
            for key in message['data'].keys():
                self.plots[message['from']].update({key: message['data'][key]})
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
        concat_axis: int = 0
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
        **kwargs
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