from typing import Dict, Union

import plotly.graph_objs as go
from time_series_buffer import TimeSeriesBuffer
from time_series_metadata.scheme import MetaData

from agentMET4FOF.agents import AgentMET4FOF


class MetrologicalAgent(AgentMET4FOF):
    # dict like {
    #     <from>: {
    #         "buffer": TimeSeriesBuffer(maxlen=buffer_size),
    #         "metadata": MetaData(**kwargs).metadata,
    #     }
    _input_data: Dict[str, Dict[str, Union[TimeSeriesBuffer, Dict]]]
    _input_data_maxlen: int

    # dict like {
    #     <channel> : {
    #         "buffer" : TimeSeriesBuffer(maxlen=buffer_size),
    #         "metadata" : MetaData(**kwargs)
    #     }
    _output_data: Dict[str, Dict[str, Union[TimeSeriesBuffer, MetaData]]]
    _output_data_maxlen: int

    def init_parameters(self, input_data_maxlen=25, output_data_maxlen=25):
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
        super().init_parameters(*args, **kwargs)

        # create alias/dummies to match dashboard expectations
        self.memory = self._input_data
        self.plot_filter = []
        self.plots = {}
        self.custom_plot_parameters = {}

    def custom_plot_function(self, data, sender_agent, **kwargs):
        # TODO: cannot set the label of the xaxis within this method

        # data display
        if "buffer" in data.keys():
            if len(data["buffer"]):
                values = data["buffer"].show(n_samples=-1)  # -1 --> all
                t = values[:, 0]
                ut = values[:, 1]
                v = values[:, 2]
                uv = values[:, 3]

                # use description
                desc = data["metadata"]
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
